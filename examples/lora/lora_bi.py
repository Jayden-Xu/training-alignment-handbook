"""LoRA and multi-adapter batches: where the bits move, and a batch-invariant LoRA linear.

Forms of y = x W + s * (x A^T) B, bf16 weights and activations:
  trainer      PEFT's unmerged path: F.linear(x, W) + F.linear(F.linear(x, A), B) * s,
               one adapter over the whole batch (cuBLAS, bf16 at every step)
  merged       F.linear(x, W + s * B A), the merge done in bf16 as PEFT's merge does
  grouped      the engine sorts tokens by adapter and runs the trainer's two GEMMs once
               per adapter group (cuBLAS): M per GEMM = tokens sharing the adapter
  gathered     per-row A_i, B_i gathered, then torch.bmm (or einsum) over rows
  seg / row    models of vLLM 0.6.3.post1's segmented (prefill) and per-row (decode)
               Triton kernels: split-K with fp32 atomics, scale folded into B in bf16
  ours         examples/lora/lora_kernels.py: one kernel for trainer and engine

Sections: merge accuracy; rounding points; L1; L2 (request alone vs co-batched with
1..63 other adapters, positions, fillers, mixed ranks, group sizes around the cuBLAS
thresholds); prefill vs decode; trainer vs engine (L4); backward; accuracy; cost.

Run (CUDA, Triton):  python lora_bi.py [--quick] [--json results.json]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "tools"))
sys.path.insert(0, here)
from bi_check import bit_equal, first_mismatch  # noqa: E402
from bi_kernels import mm_one_config  # noqa: E402
from bi_report import Results  # noqa: E402
import lora_kernels as lk  # noqa: E402

BF = torch.bfloat16
dev = "cuda"
RANKS = (8, 16, 64)
ALPHAS = (16.0, 24.0, 40.0)           # alpha of adapter l: ALPHAS[(l // 3) % 3]
PROBE_ADAPTERS = (3, 4, 5)            # ranks 8, 16, 64; scales 8.49, 1.5, 3 (not powers of two)
GROUP_EXTRA = (0, 1, 2, 4, 5, 12, 13, 14, 28, 29, 30, 31, 60, 61, 62, 124, 125, 126,
               252, 253, 254, 508, 509, 1021, 2045, 4093)   # request group = 3 + extra


# ----------------------------------------------------------------------------- weights
class Bank:
    """A base weight and L adapters with mixed ranks, padded to the largest rank R.
    Even adapters use s = alpha / r, odd ones s = alpha / sqrt(r) (rsLoRA), with
    alpha = ALPHAS[(l // 3) % 3]. A power-of-two scale is exact anywhere it is
    applied, so the probes use adapters whose scale is not one."""

    def __init__(self, H, L=64, ranks=RANKS, rho=0.05, seed=0, R=None):
        g = torch.Generator(device=dev).manual_seed(seed)
        self.H = self.N = H
        self.L = L
        self.R = R or max(ranks)
        self.W = (torch.randn(H, H, generator=g, device=dev) * H ** -0.5).to(BF)     # y = x @ W
        self.Wl = self.W.t().contiguous()                                            # F.linear weight
        self.A = torch.zeros(L, self.R, H, device=dev, dtype=BF)
        self.B = torch.zeros(L, self.R, H, device=dev, dtype=BF)
        self.scale = torch.empty(L, device=dev)
        self.r, self.s = [], []
        for l in range(L):
            r = ranks[l % len(ranks)]
            alpha = ALPHAS[(l // 3) % 3]
            s = alpha / r if l % 2 == 0 else alpha / math.sqrt(r)
            sb = rho / (s * math.sqrt(r / 3))          # update rms ~ rho * base rms
            self.A[l, :r] = ((torch.rand(r, H, generator=g, device=dev) * 2 - 1) * H ** -0.5).to(BF)
            self.B[l, :r] = (torch.randn(r, H, generator=g, device=dev) * sb).to(BF)
            self.scale[l] = s
            self.r.append(r)
            self.s.append(float(torch.tensor(s, dtype=torch.float32)))
        self.Aw = [self.A[l, :self.r[l]].contiguous() for l in range(L)]               # lora_A.weight [r, H]
        self.Bw = [self.B[l, :self.r[l]].t().contiguous() for l in range(L)]           # lora_B.weight [N, r]
        self.Bs = (self.B.float() * self.scale[:, None, None]).to(BF)                  # scale folded in bf16

    # --- stock forms
    def trainer(self, x, a, dropout=None):
        xa = x if dropout is None else dropout(x)
        return F.linear(x, self.Wl) + F.linear(F.linear(xa, self.Aw[a]), self.Bw[a]) * self.s[a]

    def merged_weight(self, a, fp32=False):
        if fp32:
            return (self.Wl.float() + (self.Bw[a].float() @ self.Aw[a].float()) * self.s[a]).to(BF)
        return self.Wl + (self.Bw[a] @ self.Aw[a]) * self.s[a]       # PEFT merge, bf16

    def grouped(self, x, idx, bi_base=False):
        base = mm_one_config(x, self.W) if bi_base else F.linear(x, self.Wl)
        out = base.clone()
        for a in torch.unique(idx).tolist():
            if a < 0:
                continue
            rows = (idx == a).nonzero().squeeze(1)
            out[rows] = base[rows] + F.linear(F.linear(x[rows], self.Aw[a]), self.Bw[a]) * self.s[a]
        return out

    def gathered(self, x, idx, einsum=False):
        base = F.linear(x, self.Wl)
        i = idx.long().clamp(min=0)
        Ag, Bg = self.A[i], self.B[i]
        if einsum:
            d = torch.einsum("tj,tjn->tn", torch.einsum("tk,tjk->tj", x, Ag), Bg)
        else:
            d = torch.bmm(torch.bmm(x[:, None, :], Ag.transpose(1, 2)), Bg)[:, 0]
        d = (d.float() * self.scale[i][:, None]).to(BF)
        return torch.where((idx >= 0)[:, None], base + d, base)

    def seg(self, x, idx):
        return lk.seg_lora_add(x @ self.W, x, self.A, self.Bs, idx)

    def row(self, x, idx):
        return lk.row_lora_add(x @ self.W, x, self.A, self.Bs, idx)

    # --- ours
    def ours(self, x, idx):
        return lk.lora_linear(x, self.W, self.A, self.B, idx, self.scale)

    def ours_train(self, x, idx):
        return lk.lora_linear_train(x, self.W, self.A.float(), self.B.float(), idx, self.scale, self.Wl)

    def ref64(self, x, idx):
        xd = x.double()
        i = idx.long().clamp(min=0)
        has = (idx >= 0).double()[:, None]
        u = torch.einsum("tk,tjk->tj", xd, self.A.double()[i])
        upd = torch.einsum("tj,tjn->tn", u, self.B.double()[i])
        base = xd @ self.W.double()
        return base, base + has * self.scale.double()[i][:, None] * upd


def rand(gen, n, H, kind="random"):
    if kind == "zeros":
        return torch.zeros(n, H, device=dev, dtype=BF)
    x = torch.randn(n, H, generator=gen, device=dev)
    return (x * 1e3 if kind == "large" else x).to(BF)


def compose(segs, probe_at):
    """segs: list of (x, adapter). Returns x, idx, row offset of segment probe_at."""
    off = sum(s[0].shape[0] for s in segs[:probe_at])
    x = torch.cat([s[0] for s in segs])
    idx = torch.cat([torch.full((s[0].shape[0],), s[1], device=dev, dtype=torch.int32) for s in segs])
    return x, idx, off


def rel(a, r):
    return float((a.double() - r).norm() / r.norm())


def mm_count(a, b):
    m = first_mismatch(a, b)
    return 0 if m is None else m.get("mismatched_elements", -1)


# ----------------------------------------------------------------------------- L2 sweep
def l2_cases(quick, L, with_group):
    ks = (1, 7, 63) if quick else (1, 2, 3, 7, 15, 31, 63)
    ms = (1, 17) if quick else (1, 3, 17)
    pos = ("first", "last") if quick else ("first", "middle", "last")
    fill = ("random", "large") if quick else ("random", "zeros", "large")
    out = [dict(k=k, m=m, pos=p, fill=f, g=0) for k in ks if k < L for m in ms for p in pos for f in fill]
    if with_group:
        gs = (0, 13, 29, 61, 125, 509) if quick else GROUP_EXTRA
        out += [dict(k=1, m=5, pos=p, fill="random", g=g) for g in gs for p in ("first", "last")]
    return out


def l2_sweep(bank, fns, quick, seed=11, n_req=3, with_group=True, max_tokens=None):
    """A request of n_req tokens with adapter a, alone vs co-batched. fns: name -> fn(x, idx).
    Returns name -> {checked, failures, failing_group_sizes}."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = {n: {"checked": 0, "failures": [], "fail_count": 0, "failing_group_sizes": set()} for n in fns}
    for a in PROBE_ADAPTERS:
        probe = rand(gen, n_req, bank.H)
        pidx = torch.full((n_req,), a, device=dev, dtype=torch.int32)
        refs = {n: fn(probe, pidx) for n, fn in fns.items()}
        for c in l2_cases(quick, bank.L, with_group):
            others = [(rand(gen, c["m"], bank.H, c["fill"]), (a + 1 + i) % bank.L) for i in range(c["k"])]
            if c["k"] >= 3 and c["fill"] == "random":
                others.insert(1, (rand(gen, 2, bank.H), -1))          # rows with no adapter
            req = probe if c["g"] == 0 else torch.cat([probe, rand(gen, c["g"], bank.H)])
            p = {"first": 0, "middle": len(others) // 2, "last": len(others)}[c["pos"]]
            segs = others[:p] + [(req, a)] + others[p:]
            x, idx, off = compose(segs, p)
            for n, fn in fns.items():
                if max_tokens and x.shape[0] > max_tokens.get(n, 1 << 30):
                    continue
                got = fn(x, idx)[off:off + n_req]
                out[n]["checked"] += 1
                mm = first_mismatch(refs[n], got)
                if mm:
                    out[n]["fail_count"] += 1
                    out[n]["failing_group_sizes"].add(n_req + c["g"])
                    if len(out[n]["failures"]) < 12:
                        out[n]["failures"].append({"adapter": a, "rank": bank.r[a], **c, "tokens": x.shape[0],
                                                   "mismatched_elements": mm.get("mismatched_elements")})
    for n in out:
        out[n]["failing_group_sizes"] = sorted(out[n]["failing_group_sizes"])
    return out


# ----------------------------------------------------------------------------- sections
def merge_section(res, Hs):
    """Merged vs unmerged: different bits by construction; error against fp64, and what
    a bf16 merge loses when the update is small against W."""
    out = {}
    for H in Hs:
        for rho in (1e-1, 1e-2, 1e-3):
            bank = Bank(H, L=6, rho=rho, seed=1)
            gen = torch.Generator(device=dev).manual_seed(2)
            x = rand(gen, 256, H)
            for a in range(3, 6):
                idx = torch.full((256,), a, device=dev, dtype=torch.int32)
                base64, y64 = bank.ref64(x, idx)
                upd64 = y64 - base64
                delta64 = (bank.Bw[a].double() @ bank.Aw[a].double()) * bank.s[a]
                row = {}
                for name, Wm in (("merged_bf16", bank.merged_weight(a)), ("merged_fp32_then_bf16", bank.merged_weight(a, True))):
                    eff = Wm.double() - bank.Wl.double()
                    y = F.linear(x, Wm)
                    row[name] = {"y_rel": rel(y, y64), "update_rel": rel(y.double() - base64, upd64),
                                 "delta_rel": rel(eff, delta64),
                                 "delta_lost_frac": float(((eff == 0) & (delta64 != 0)).double().mean())}
                for name, y in (("trainer_unmerged", bank.trainer(x, a)), ("ours", bank.ours(x, idx))):
                    row[name] = {"y_rel": rel(y, y64), "update_rel": rel(y.double() - base64, upd64)}
                row["floor_bf16_of_fp64"] = {"y_rel": rel(y64.to(BF), y64)}
                row["merged_vs_unmerged_mismatch"] = mm_count(F.linear(x, bank.merged_weight(a)), bank.trainer(x, a))
                out[f"H={H} rho={rho:g} r={bank.r[a]}"] = row
                if rho == 1e-2 and a == 4:
                    res.check(f"merged (bf16 merge) vs unmerged, H={H} r=16 s=1.5, same batch", row["merged_vs_unmerged_mismatch"] == 0,
                              level="L4", impl="merged_vs_unmerged", expect="fail",
                              detail={"mismatched_elements": row["merged_vs_unmerged_mismatch"], "of": 256 * H})
    res.extra["merge_accuracy"] = out
    for k, v in out.items():
        print(f"  merge {k}: " + ", ".join(f"{n} y {d['y_rel']:.2e}" + (f" upd {d['update_rel']:.2e}" if "update_rel" in d else "")
                                           + (f" lost {d['delta_lost_frac']:.2f}" if "delta_lost_frac" in d else "")
                                           for n, d in v.items() if isinstance(d, dict)), flush=True)


def rounding_section(res, Hs):
    out = {}
    for H in Hs:
        bank = Bank(H, seed=3)
        gen = torch.Generator(device=dev).manual_seed(4)
        x = rand(gen, 256, H)
        for a in (0,) + PROBE_ADAPTERS:          # adapter 0: r=8, s=2 (a power of two)
            idx = torch.full((256,), a, device=dev, dtype=torch.int32)
            base64, y64 = bank.ref64(x, idx)
            Aw, Bw, s = bank.Aw[a], bank.Bw[a], bank.s[a]
            base = F.linear(x, bank.Wl)
            forms = {
                "s on B output (PEFT)": base + F.linear(F.linear(x, Aw), Bw) * s,
                "s folded into B in bf16 (vLLM load)": base + F.linear(F.linear(x, Aw), (Bw.float() * s).to(BF)),
                "s on A output": base + F.linear(F.linear(x, Aw) * s, Bw),
                "fp32 update, base+update in fp32, one cast": (base.float() + F.linear(F.linear(x, Aw).float(), Bw.float()) * s).to(BF),
            }
            u = lk.lora_shrink(x, bank.A, idx)
            ours = lk.lora_expand_add(x, bank.W, u, bank.B, idx, bank.scale, 0)
            forms["ours (one fp32 fma, one cast)"] = ours
            forms["ours, two bf16 values added (rejected)"] = lk.lora_expand_add(x, bank.W, u, bank.B, idx, bank.scale, 1)
            forms["ours, bf16 update scaled in bf16 (rejected)"] = lk.lora_expand_add(x, bank.W, u, bank.B, idx, bank.scale, 2)
            ref = forms["s on B output (PEFT)"]
            out[f"H={H} r={bank.r[a]} s={s:.4g}"] = {n: {"mismatch_vs_peft": mm_count(ref, y), "mismatch_vs_ours": mm_count(ours, y),
                                               "y_rel": rel(y, y64)} for n, y in forms.items()}
            # rank padding: the same adapter stored at rank r and zero-padded to 64
            if bank.r[a] < bank.R:
                Ap = torch.zeros(bank.R, H, device=dev, dtype=BF); Ap[:bank.r[a]] = Aw
                Bp = torch.zeros(H, bank.R, device=dev, dtype=BF); Bp[:, :bank.r[a]] = Bw
                stock_pad = base + F.linear(F.linear(x, Ap), Bp) * s
                n_st = mm_count(ref, stock_pad)
                A1 = bank.A[a:a + 1, :bank.r[a]].contiguous(); B1 = bank.B[a:a + 1, :bank.r[a]].contiguous()
                i0 = torch.zeros(256, device=dev, dtype=torch.int32)
                n_ours = mm_count(lk.lora_linear(x, bank.W, A1, B1, i0, bank.scale[a:a + 1]), bank.ours(x, idx))
                out[f"H={H} r={bank.r[a]} s={s:.4g}"]["rank_padding"] = {"cublas_mismatch": n_st, "ours_mismatch": n_ours}
                res.check(f"rank padding r={bank.r[a]} -> 64 with zeros, H={H}", n_ours == 0, level="L2", impl="ours",
                          detail={"mismatched_elements": n_ours})
                res.check(f"rank padding r={bank.r[a]} -> 64 with zeros, H={H}", n_st == 0, level="L2",
                          impl="trainer_cublas", expect="info", detail={"mismatched_elements": n_st})
        # dropout: only the trainer has it, on the LoRA branch input
        a = 4
        idx = torch.full((256,), a, device=dev, dtype=torch.int32)
        torch.manual_seed(0)
        drop = torch.nn.Dropout(0.05).train()
        n_drop = mm_count(bank.trainer(x, a), bank.trainer(x, a, dropout=drop))
        n_eval = mm_count(bank.trainer(x, a), bank.trainer(x, a, dropout=torch.nn.Dropout(0.05).eval()))
        out[f"H={H} dropout"] = {"train_mode_p0.05_mismatch": n_drop, "eval_mode_mismatch": n_eval, "of": 256 * H}
        res.check(f"trainer dropout p=0.05 (train mode) vs no dropout, H={H}", n_drop == 0, level="L4",
                  impl="trainer_cublas", expect="fail", detail=out[f"H={H} dropout"])
    res.extra["rounding_points"] = out
    for k, v in out.items():
        print(f"  rounding {k}: {v}", flush=True)


def l1_section(res, Hs, quick):
    for H in Hs:
        bank = Bank(H, seed=5)
        gen = torch.Generator(device=dev).manual_seed(6)
        T = 300
        x = rand(gen, T, H)
        idx = torch.randint(-1, bank.L, (T,), generator=gen, device=dev).to(torch.int32)
        sidx = torch.sort(idx.clamp(min=0)).values.to(torch.int32)
        n = 5 if quick else 20
        for impl, fn, ii, expect in (("ours", bank.ours, idx, "pass"), ("grouped_cublas", bank.grouped, idx, "info"),
                                     ("vllm_seg_model", bank.seg, sidx, "info"), ("vllm_row_model", bank.row, sidx, "info")):
            ref = fn(x, ii)
            bad = [mm_count(ref, fn(x, ii)) for _ in range(n - 1)]
            nd = sum(1 for b in bad if b)
            res.check(f"repeat x{n}, T={T}, mixed adapters, H={H}", nd == 0, level="L1", impl=impl, expect=expect,
                      detail={"runs_differing": nd, "worst_mismatched": max(bad), "of": ref.numel()})
            print(f"  L1 {impl} H={H}: {nd}/{n-1} runs differ, worst {max(bad)}", flush=True)
        # the engine models' shrink buffer (fp32, split-K atomics) on its own
        for impl, kern in (("vllm_seg_model shrink buffer", "seg"), ("vllm_row_model shrink buffer", "row")):
            def buf():
                uu = torch.zeros(T, bank.R, device=dev)
                if kern == "seg":
                    st, ln, ids = lk.segments(sidx)
                    lk._seg_shrink_kernel[(lk.triton.cdiv(int(ln.max()), 32) * lk.triton.cdiv(bank.R, 16), 8, st.shape[0])](
                        x, bank.A, uu, st, ln, ids, bank.R, H, 1.0, x.stride(0), x.stride(1), bank.A.stride(0),
                        bank.A.stride(1), bank.A.stride(2), uu.stride(0), uu.stride(1), BM=32, BN=16, BK=32, SPLIT_K=8)
                else:
                    lk._row_shrink_kernel[(64, T)](x, bank.A, uu, sidx, bank.R, H, 1.0, x.stride(0), x.stride(1),
                                                   bank.A.stride(0), bank.A.stride(1), bank.A.stride(2), uu.stride(0),
                                                   uu.stride(1), BN=64, BK=256, SPLIT_K=64, num_warps=8)
                return uu
            ref = buf()
            bad = [mm_count(ref, buf()) for _ in range(n - 1)]
            nd = sum(1 for b in bad if b)
            res.check(f"repeat x{n}, fp32 shrink buffer, H={H}", nd == 0, level="L1", impl=impl, expect="fail",
                      detail={"runs_differing": nd, "worst_mismatched": max(bad), "of": ref.numel()})
            print(f"  L1 {impl} H={H}: {nd}/{n-1} runs differ, worst {max(bad)}", flush=True)


def l2_section(res, Hs, quick):
    for H in Hs:
        bank = Bank(H, seed=7)
        fns = {"ours": bank.ours, "grouped_cublas": bank.grouped,
               "grouped_cublas_lora+bi_base": lambda x, i: bank.grouped(x, i, bi_base=True),
               "gathered_bmm": bank.gathered}
        lim = {"gathered_bmm": 1200}
        t0 = time.time()
        out = l2_sweep(bank, fns, quick, max_tokens=lim)
        for impl, r in out.items():
            ok = r["fail_count"] == 0
            expect = "pass" if impl == "ours" else "fail"
            res.check(f"request (3 tokens) alone vs co-batched with 1..63 adapters, positions, fillers, group sizes, H={H}",
                      ok, level="L2", impl=impl, expect=expect,
                      detail={"checked": r["checked"], "failing": r["fail_count"],
                              "failing_group_sizes": r["failing_group_sizes"], "failures": r["failures"]})
            print(f"  L2 {impl} H={H}: {r['fail_count']}/{r['checked']} fail; group sizes failing {r['failing_group_sizes']}", flush=True)
        print(f"  ({time.time() - t0:.1f}s)", flush=True)


def modes_section(res, Hs, quick):
    """Prefill M vs decode M, and trainer vs engine (L4)."""
    for H in Hs:
        bank = Bank(H, seed=9)
        gen = torch.Generator(device=dev).manual_seed(10)
        Tq = 7
        for a in PROBE_ADAPTERS:
            seq = rand(gen, Tq, H)
            others = [(rand(gen, 5, H), (a + 1 + i) % bank.L) for i in range(6)] + [(rand(gen, 3, H), -1)]
            x_pf, idx_pf, off = compose([others[0], (seq, a)] + others[1:], 1)
            dec_others = [rand(gen, 1, H) for _ in range(5)]

            def decode(fn):
                rows = []
                for t in range(Tq):                     # each step: the token among 5 other requests' tokens
                    segs = [(dec_others[i], (a + 1 + i) % bank.L) for i in range(2)] + [(seq[t:t + 1], a)] + \
                           [(dec_others[i], (a + 1 + i) % bank.L) for i in range(2, 5)]
                    x, idx, o = compose(segs, 2)
                    rows.append(fn(x, idx)[o:o + 1])
                return torch.cat(rows)

            for impl, fn, expect in (("ours", bank.ours, "pass"), ("grouped_cublas", bank.grouped, "fail")):
                ok = bit_equal(fn(x_pf, idx_pf)[off:off + Tq], decode(fn))
                res.check(f"prefill (M={x_pf.shape[0]}) vs decode (M=6 per step), adapter r={bank.r[a]}, H={H}", ok,
                          level="L3", impl=impl, expect=expect,
                          detail={"mismatched_elements": mm_count(fn(x_pf, idx_pf)[off:off + Tq], decode(fn))})
            sidx = idx_pf.clone()
            pf_seg = bank.seg(x_pf, sidx)[off:off + Tq]
            n_sr = mm_count(pf_seg, decode(lambda x, i: bank.row(x, i)))
            res.check(f"seg model prefill vs row model decode, r={bank.r[a]}, H={H}", n_sr == 0, level="L3",
                      impl="vllm_seg_vs_row_model", expect="fail", detail={"mismatched_elements": n_sr})
            # trainer vs engine: the trainer runs this sequence with a sibling of the same adapter
            sib = rand(gen, Tq, H)
            x_tr = torch.cat([seq, sib])
            i_tr = torch.full((2 * Tq,), a, device=dev, dtype=torch.int32)
            tr_ours = bank.ours_train(x_tr, i_tr).detach()[:Tq]
            tr_stock = bank.trainer(x_tr, a)[:Tq]
            for impl, eng, tr, expect in (("ours (train fn vs engine fn)", bank.ours, tr_ours, "pass"),
                                          ("trainer_cublas vs grouped_cublas", bank.grouped, tr_stock, "fail"),
                                          ("trainer_cublas vs gathered_bmm", bank.gathered, tr_stock, "fail"),
                                          ("trainer_cublas vs vllm_seg_model", bank.seg, tr_stock, "info")):
                n_pf = mm_count(tr, eng(x_pf, idx_pf)[off:off + Tq])
                res.check(f"trainer (one adapter, 2 seqs) vs engine prefill (7 adapters + no-adapter rows), r={bank.r[a]}, H={H}",
                          n_pf == 0, level="L4", impl=impl, expect=expect, detail={"mismatched_elements": n_pf})
            n_dec = mm_count(tr_ours, decode(bank.ours))
            res.check(f"trainer vs engine decode, r={bank.r[a]}, H={H}", n_dec == 0, level="L4", impl="ours", detail={"mismatched_elements": n_dec})
            n_dec = mm_count(tr_stock, decode(lambda x, i: bank.row(x, i)))
            res.check(f"trainer vs engine decode, r={bank.r[a]}, H={H}", n_dec == 0, level="L4",
                      impl="trainer_cublas vs vllm_row_model", expect="fail", detail={"mismatched_elements": n_dec})
        # gathered vs grouped vs trainer on one mixed batch: kernels differ
        x, idx, _ = compose([(rand(gen, 9, H), l) for l in range(12)], 0)
        g1, g2, g3 = bank.grouped(x, idx), bank.gathered(x, idx), bank.gathered(x, idx, einsum=True)
        res.extra.setdefault("form_mismatch", {})[f"H={H}"] = {
            "grouped_vs_gathered_bmm": mm_count(g1, g2), "gathered_bmm_vs_einsum": mm_count(g2, g3),
            "grouped_vs_ours": mm_count(g1, bank.ours(x, idx)), "of": g1.numel()}
        print(f"  forms H={H}: {res.extra['form_mismatch'][f'H={H}']}", flush=True)


def backward_section(res, Hs, quick):
    for H in Hs:
        bank = Bank(H, seed=13)
        gen = torch.Generator(device=dev).manual_seed(14)
        a = 4

        def stock_grads(x, g):
            xr = x.clone().requires_grad_(True)
            Ar = bank.Aw[a].clone().requires_grad_(True)
            Br = bank.Bw[a].clone().requires_grad_(True)
            y = F.linear(xr, bank.Wl) + F.linear(F.linear(xr, Ar), Br) * bank.s[a]
            y.backward(g)
            return xr.grad, Ar.grad, Br.grad.t().contiguous()

        def ours_grads(x, g, idx=None):
            idx = torch.full((x.shape[0],), a, device=dev, dtype=torch.int32) if idx is None else idx
            xr = x.clone().requires_grad_(True)
            Ar = bank.A.float().requires_grad_(True)
            Br = bank.B.float().requires_grad_(True)
            lk.lora_linear_train(xr, bank.W, Ar, Br, idx, bank.scale, bank.Wl).backward(g)
            r = bank.r[a]
            return xr.grad, Ar.grad[a, :r], Br.grad[a, :r]

        T = 1024
        x, g = rand(gen, T, H), rand(gen, T, H)
        for impl, fn, expect in (("ours", ours_grads, "pass"), ("trainer_cublas", stock_grads, "info")):
            ref = fn(x, g)
            bad = [[mm_count(r0, r1) for r0, r1 in zip(ref, fn(x, g))] for _ in range(4)]
            nd = sum(1 for b in bad if any(b))
            res.check(f"backward repeat x5, T={T}, H={H}: dX, dA, dB", nd == 0, level="L1", impl=impl, expect=expect,
                      detail={"runs_differing": nd, "mismatched": bad})
        # mixed-adapter training batch: ours, rows of other adapters interleaved
        xm, idxm, _ = compose([(rand(gen, 5, H), l) for l in range(16)], 0)
        gm = rand(gen, xm.shape[0], H)
        ref = ours_grads(xm, gm, idxm)
        perm = torch.randperm(xm.shape[0], generator=torch.Generator().manual_seed(0)).to(dev)
        own = (idxm == a)
        order = torch.cat([perm[~own[perm]][:20], torch.nonzero(own).squeeze(1), perm[~own[perm]][20:]])
        got = ours_grads(xm[order], gm[order], idxm[order])
        ok = bit_equal(ref[1], got[1]) and bit_equal(ref[2], got[2])
        res.check(f"dA, dB of one adapter: other adapters' rows reordered around it, H={H}", ok, level="L2", impl="ours",
                  detail={"dA": mm_count(ref[1], got[1]), "dB": mm_count(ref[2], got[2])})
        # a sequence alone vs right-padded / batched: dX per row, dA and dB with zero-gradient padding
        L_seq = 37
        xs, gs_ = rand(gen, L_seq, H), rand(gen, L_seq, H)
        pads = (64, 128) if quick else (38, 64, 65, 128, 129, 256, 512, 1024)
        for impl, fn, expect in (("ours", ours_grads, "pass"), ("trainer_cublas", stock_grads, "fail")):
            r0 = fn(xs, gs_)
            fails = []
            for P in pads:
                xp = torch.cat([xs, rand(gen, P - L_seq, H)])
                gp = torch.cat([gs_, torch.zeros(P - L_seq, H, device=dev, dtype=BF)])
                r1 = fn(xp, gp)
                c = {"pad_to": P, "dX": mm_count(r0[0], r1[0][:L_seq]), "dA": mm_count(r0[1], r1[1]), "dB": mm_count(r0[2], r1[2])}
                if c["dX"] or c["dA"] or c["dB"]:
                    fails.append(c)
            res.check(f"sequence of {L_seq} alone vs right-padded to {list(pads)} (zero output grad): dX, dA, dB, H={H}",
                      not fails, level="L2", impl=impl, expect=expect, detail={"failures": fails})
            print(f"  bwd padding {impl} H={H}: {len(fails)}/{len(pads)} fail {fails[:3]}", flush=True)
        # dX per row against batch size (other sequences, any adapter for ours)
        for impl, expect in (("ours", "pass"), ("trainer_cublas", "fail")):
            fails = []
            for n_other in ((1, 15) if quick else (1, 2, 7, 15, 31)):
                xo, go = rand(gen, 37 * n_other, H), rand(gen, 37 * n_other, H)
                if impl == "ours":
                    io = torch.randint(0, bank.L, (37 * n_other,), generator=gen, device=dev).to(torch.int32)
                    idx_all = torch.cat([io[:20], torch.full((L_seq,), a, device=dev, dtype=torch.int32), io[20:]])
                    r1 = ours_grads(torch.cat([xo[:20], xs, xo[20:]]), torch.cat([go[:20], gs_, go[20:]]), idx_all)
                    dx0 = ours_grads(xs, gs_)[0]
                else:
                    r1 = stock_grads(torch.cat([xo[:20], xs, xo[20:]]), torch.cat([go[:20], gs_, go[20:]]))
                    dx0 = stock_grads(xs, gs_)[0]
                n = mm_count(dx0, r1[0][20:20 + L_seq])
                if n:
                    fails.append({"other_seqs": n_other, "dX": n})
            res.check(f"dX of a sequence alone vs among other sequences, H={H}", not fails, level="L2", impl=impl,
                      expect=expect, detail={"failures": fails})
        # accuracy against fp64
        idx = torch.full((T,), a, device=dev, dtype=torch.int32)
        xd = x.double().requires_grad_(True)
        Ad = bank.Aw[a].double().requires_grad_(True)
        Bd = bank.Bw[a].double().requires_grad_(True)
        (xd @ bank.W.double() + (xd @ Ad.t()) @ Bd.t() * bank.s[a]).backward(g.double())
        acc = {}
        for impl, fn in (("ours", ours_grads), ("trainer_cublas", stock_grads)):
            dx, dA, dB = fn(x, g)
            acc[impl] = {"dX": rel(dx, xd.grad), "dA": rel(dA, Ad.grad), "dB": rel(dB, Bd.grad.t())}
        res.extra.setdefault("backward_accuracy", {})[f"H={H} r=16 T={T}"] = acc
        print(f"  bwd accuracy H={H}: {acc}", flush=True)


def accuracy_section(res, Hs):
    out = {}
    for H in Hs:
        bank = Bank(H, seed=15)
        gen = torch.Generator(device=dev).manual_seed(16)
        x, idx, _ = compose([(rand(gen, 25, H), l) for l in range(12)] + [(rand(gen, 4, H), -1)], 0)
        sidx = torch.where(idx < 0, torch.zeros_like(idx), idx)
        _, y64 = bank.ref64(x, idx)
        _, y64s = bank.ref64(x, sidx)
        out[f"H={H}"] = {"ours": rel(bank.ours(x, idx), y64), "grouped_cublas (= trainer rounding)": rel(bank.grouped(x, idx), y64),
                         "gathered_bmm": rel(bank.gathered(x, idx), y64), "vllm_seg_model": rel(bank.seg(x, sidx), y64s),
                         "vllm_row_model": rel(bank.row(x, sidx), y64s), "floor bf16(fp64)": rel(y64.to(BF), y64)}
        print(f"  accuracy H={H}: {out[f'H={H}']}", flush=True)
    res.extra["forward_accuracy_rel_fro"] = out


def timed(fn, n=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def cost_section(res, Hs, quick):
    Ts = (64, 1024) if quick else (64, 256, 1024, 4096)
    Rs = (16,) if quick else (16, 64)
    for H in Hs:
        for R in Rs:
            bank = Bank(H, L=64, ranks=(R,), seed=17)
            gen = torch.Generator(device=dev).manual_seed(18)
            Wm = bank.merged_weight(0)
            for T in Ts:
                x = rand(gen, T, H)
                for La in (1, 8, 64):
                    idx = (torch.arange(T, device=dev) * La // T).to(torch.int32)
                    forms = {"merged single GEMM (1 adapter)": lambda: F.linear(x, Wm),
                             "cuBLAS base only": lambda: F.linear(x, bank.Wl),
                             "grouped_cublas": lambda: bank.grouped(x, idx),
                             "vllm_seg_model": lambda: bank.seg(x, idx),
                             "vllm_row_model": lambda: bank.row(x, idx),
                             "ours": lambda: bank.ours(x, idx)}
                    if T * R * H * 2 <= 2 ** 30:
                        forms["gathered_bmm"] = lambda: bank.gathered(x, idx)
                    for name, fn in forms.items():
                        if La > 1 and name.startswith(("merged", "cuBLAS")):
                            continue
                        t = timed(fn)
                        res.measure(f"forward H={H} r={R} T={T} adapters={La}", round(T / t), "tokens/s", impl=name,
                                    ms=round(t * 1e3, 4), H=H, r=R, T=T, adapters=La)
                    print(f"  cost H={H} r={R} T={T}: " + ", ".join(
                        f"{m['impl']}@{m['adapters']} {m['ms']:.3f}" for m in res.measurements
                        if m.get("H") == H and m.get("r") == R and m.get("T") == T and "fwd" not in m), flush=True)
            # training step of one layer: forward + backward, one adapter
            for T in ((1024,) if quick else (1024, 4096)):
                x, g = rand(gen, T, H), rand(gen, T, H)
                idx = torch.zeros(T, device=dev, dtype=torch.int32)
                Ar = bank.Aw[0].clone().requires_grad_(True)
                Br = bank.Bw[0].clone().requires_grad_(True)
                xr = x.clone().requires_grad_(True)
                Am = bank.A.float().requires_grad_(True)
                Bm = bank.B.float().requires_grad_(True)

                def stock():
                    (F.linear(xr, bank.Wl) + F.linear(F.linear(xr, Ar), Br) * bank.s[0]).backward(g)

                def ours():
                    lk.lora_linear_train(xr, bank.W, Am, Bm, idx, bank.scale, bank.Wl).backward(g)
                for name, fn in (("trainer_cublas", stock), ("ours", ours)):
                    t = timed(fn, n=9)
                    res.measure(f"fwd+bwd H={H} r={R} T={T} adapters=1", round(T / t), "tokens/s", impl=name,
                                ms=round(t * 1e3, 4), H=H, r=R, T=T, adapters=1, fwd_bwd=True)
                    print(f"  fwd+bwd {name} H={H} r={R} T={T}: {t*1e3:.3f} ms", flush=True)


VLLM_READING = {
    "source": "read from source, vLLM 0.6.3.post1 (vllm/lora/punica.py, layers.py, lora.py, ops/*.py); not run; newer versions untested",
    "form": "unmerged; gathered by adapter index inside Triton kernels (Punica SGMV for prefill, BGMV for decode); no per-adapter cuBLAS GEMM, no merge",
    "base": "base_layer.quant_method.apply(x) first (the model's normal GEMM over all tokens), then add_lora adds the update in place into that bf16 output",
    "scale": "LoRALayerWeights.optimize folds scaling into lora_b in the LoRA dtype at load; kernels get scale 1.0",
    "shrink": "x @ A into a zeroed fp32 buffer; sgmv_shrink: tiles 32x16x32 per segment of consecutive same-adapter tokens, SPLIT_K=8, tl.atomic_add; bgmv_shrink: SPLIT_K=64, BLOCK_K=256, products and sum in bf16 (tl.sum of a*b), tl.atomic_add",
    "expand": "buffer cast to bf16 (CAST_TYPE), @ B; sgmv_expand: fp32 accumulate, cast to bf16, then + the bf16 output; bgmv_expand: bf16 products and sum + output",
    "prefill_vs_decode": "PunicaWrapper picks sgmv in prefill (mapping.is_prefill) and bgmv in decode: two kernels, two rounding orders",
    "rank": "lora_a/b_stacked sized max_lora_rank; smaller ranks zero-padded (set_lora copies into [:r])",
    "grouping": "compute_meta merges consecutive tokens with the same adapter into one segment, so a segment's M depends on the neighbours; the grid is cdiv(max segment length, 32) x tiles x segments",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--sections", default="merge,rounding,l1,l2,modes,backward,accuracy,cost")
    args = ap.parse_args()
    res = Results("lora", quick=args.quick)
    res.extra["vllm_0.6.3.post1_read"] = VLLM_READING
    res.extra["setup"] = {"dtype": "bf16 weights and activations", "ranks": RANKS, "alphas": ALPHAS, "adapters": 64,
                          "max_rank_padding": 64, "scales": "alpha/r on even adapters, alpha/sqrt(r) on odd",
                          "probe_adapters": "3, 4, 5: r=8 s=8.485, r=16 s=1.5, r=64 s=3"}
    Hs = (1024, 4096)
    secs = args.sections.split(",")
    t_start = time.time()
    torch.manual_seed(0)
    for name, fn in (("merge", lambda: merge_section(res, Hs[:1] if args.quick else Hs)),
                     ("rounding", lambda: rounding_section(res, Hs)),
                     ("l1", lambda: l1_section(res, Hs, args.quick)),
                     ("l2", lambda: l2_section(res, Hs, args.quick)),
                     ("modes", lambda: modes_section(res, Hs, args.quick)),
                     ("backward", lambda: backward_section(res, Hs, args.quick)),
                     ("accuracy", lambda: accuracy_section(res, Hs)),
                     ("cost", lambda: cost_section(res, Hs, args.quick))):
        if name in secs and not (args.quick and name in ("accuracy", "merge")):
            t0 = time.time()
            print(f"[{name}]", flush=True)
            fn()
            res.extra.setdefault("section_seconds", {})[name] = round(time.time() - t0, 1)
    res.extra["total_seconds"] = round(time.time() - t_start, 1)
    res.write(args.json)
    sys.exit(res.exit_code())


if __name__ == "__main__":
    main()
