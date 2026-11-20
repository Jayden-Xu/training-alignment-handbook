"""MLA (absorbed vs explicit) and DSA-like sparse attention, checked bitwise on one GPU.

Cases (all bf16 unless noted):

components  which baseline pieces are batch-variant: cuBLAS GEMMs (torch.mm, bmm),
            torch-op RMSNorm, vs the one-config / grouped / row-per-program kernels.
mla         a DeepSeek-V2-Lite-sized MLA layer (16 heads, kv_lora_rank 512, qk_nope 128,
            qk_rope 64, v_head 128, hidden 2048, latent KV cache) in two forms:
              explicit  decompress per-head K/V from the latent (W_UK, W_UV), attend (dims 192/128)
              absorbed  fold W_UK into the query and W_UV into the output, attend over the
                        shared latent plus rope key (dims 576/512, one KV head)
            1. absorbed vs explicit on the same input, against an fp64 reference
            2. L3: prefill vs per-token decode vs chunked prefill (7/16/64) within a form
            3. L2: one sequence alone vs among 2/3/8/16 others (prefill) and 2..64 (decode)
            4. mixed engine (explicit prefill, absorbed decode) vs one form everywhere
            for five implementations, from cuBLAS+SDPA to all batch-invariant kernels.
topk_ties   torch.topk on constructed ties: does the selected set change with the batch?
sparse      DSA-like sparse attention: indexer (4 heads x 64, ReLU, weighted head sum),
            top-64 of up to 2048 causal keys, attention over the selected keys only.
            L2 (prefill, decode) and L3 (prefill/decode/chunks), counting flipped
            selections and output mismatches.
timing      throughput of baseline vs invariant pipelines, and of each MLA form per mode
            (eager and CUDA-graph replay).
tune        times a few fixed configurations of the absorbed core (to choose ABS_CFG) and
            checks their bits against the pinned one.
sdpa_probe  which SDPA backends accept the baseline's calls.

Run (CUDA, Triton):  python mla_sparse.py --cases all --json results-a40.json
Regression suite:    python mla_sparse.py --quick --bi-json out.json
                     (--quick: every case except tune and sdpa_probe, at reduced sizes; --bi-json: the
                     checks in the bi-results-v1 schema, through suite/adapters/mla_sparse.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, HERE)
from bi_check import bits, check_batch, check_modes, check_repeat, first_mismatch  # noqa: E402
from bi_kernels import BF, mm_one_config  # noqa: E402
import mla_sparse_kernels as K  # noqa: E402

dev = "cuda"
sdpa = F.scaled_dot_product_attention
NEG = float("-inf")
QUICK = False


def timed(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / iters * 1e3, 4)


def timed_graph(fn, iters=50):
    """Replay fn captured in a CUDA graph (no Python or launch overhead), as engines run decode.
    Returns None if capture fails."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
    except Exception:                    # noqa: BLE001
        torch.cuda.synchronize()
        return None
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / iters * 1e3, 4)


def tm(fn, iters=20):
    return {"eager": timed(fn, iters), "graph": timed_graph(fn, max(iters, 20))}


def nmis(a, b):
    return int((bits(a) != bits(b)).sum())


def errs(x, ref):
    d = x.double() - ref
    return {"max_abs": float(d.abs().max()), "rel_fro": float(d.norm() / ref.norm())}


# ----------------------------------------------------------------------------- implementations
class Impl:
    """gemm: cublas | bi; norm: torch | triton; core: sdpa | bi (MLA);
    indexer: torch | bi; select: topk | stable; sattn: torch | bi (sparse)."""

    def __init__(self, name, gemm="cublas", norm="torch", core="sdpa", indexer="torch", select="topk",
                 sattn="torch"):
        self.name, self.gemm, self.norm_kind, self.core = name, gemm, norm, core
        self.indexer_kind, self.select_kind, self.sattn_kind = indexer, select, sattn

    def mm(self, a, b):
        return a @ b if self.gemm == "cublas" else mm_one_config(a, b)

    def heads_mm(self, a, w):                         # a [M, H, K], w [H, K, N] -> [M, H, N]
        if self.gemm == "cublas":
            return torch.bmm(a.transpose(0, 1), w).transpose(0, 1)
        return K.mm_grouped(a, w)

    def rms(self, x, g):
        if self.norm_kind == "torch":
            xf = x.float()
            return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(BF) * g
        return K.rmsnorm(x, g)


# ----------------------------------------------------------------------------- RoPE
def _tables(Tmax, dr, dtype):
    inv = 1.0 / (10000 ** (torch.arange(0, dr, 2, dtype=torch.float64, device=dev) / dr))
    f = torch.outer(torch.arange(Tmax, dtype=torch.float64, device=dev), inv)
    return f.cos().to(dtype), f.sin().to(dtype)


_ROPE = {}


def rope(x, start):
    """Rotate-half RoPE on x [B, n, ..., dr] at absolute positions start..start+n-1, in fp32
    (fp64 for an fp64 input), cast back. Elementwise, so the same per position in any mode."""
    dt = torch.float64 if x.dtype == torch.float64 else torch.float32
    dr, n = x.shape[-1], x.shape[1]
    if (dr, dt) not in _ROPE:
        _ROPE[(dr, dt)] = _tables(8192, dr, dt)
    cos, sin = _ROPE[(dr, dt)]
    shape = (1, n) + (1,) * (x.dim() - 3) + (dr // 2,)
    c, s = cos[start:start + n].view(shape), sin[start:start + n].view(shape)
    xf = x.to(dt)
    x1, x2 = xf[..., :dr // 2], xf[..., dr // 2:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)


# ----------------------------------------------------------------------------- MLA
d_m, H_m, C_m, DN, DR, DV = 2048, 16, 512, 128, 64, 128
ABS_CFG = dict(BM=32, BN=32, num_warps=8, num_stages=1)   # absorbed core: 576/512 dims (pinned, see tune)
EXP_CFG = dict(BM=64, BN=64, num_warps=4, num_stages=2)   # explicit core: 192/128 dims


class MLAParams:
    def __init__(self, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)

        def r(*s):
            return torch.randn(*s, generator=g, device=dev)
        self.W_q = (r(d_m, H_m * (DN + DR)) * d_m ** -0.5).to(BF)
        self.W_dkv = (r(d_m, C_m + DR) * d_m ** -0.5).to(BF)
        self.g_kv = (1 + 0.1 * r(C_m)).to(BF)
        w_kvb = (r(C_m, H_m, DN + DV) * C_m ** -0.5).to(BF)
        self.W_kvb = w_kvb.reshape(C_m, H_m * (DN + DV)).contiguous()          # explicit: one GEMM
        self.W_UKT = w_kvb[:, :, :DN].permute(1, 2, 0).contiguous()            # [H, dn, C]
        self.W_UV = w_kvb[:, :, DN:].permute(1, 0, 2).contiguous()             # [H, C, dv]
        self.W_o = (r(H_m * DV, d_m) * (H_m * DV) ** -0.5).to(BF)
        self.scale = (DN + DR) ** -0.5                                          # both forms


def sdpa_causal(q, k, v, start, scale):
    n, Tk = q.shape[2], k.shape[2]
    if start == 0:
        return sdpa(q, k, v, is_causal=True, scale=scale)
    if n == 1:
        return sdpa(q, k, v, scale=scale)               # decode: no mask, as engines call it
    mask = torch.ones(n, Tk, device=q.device, dtype=torch.bool).tril(diagonal=start)
    return sdpa(q, k, v, attn_mask=mask, scale=scale)


def core_explicit(I, qn, qpe, kn, kp, v, start, scale):
    if I.core == "bi":
        return K.mla_attention(qn, qpe, kn, kp, v, start, scale, **EXP_CFG)
    B, n, H, _ = qn.shape
    Tk = kn.shape[1]
    q = torch.cat([qn, qpe], -1).transpose(1, 2)
    k = torch.cat([kn, kp[:, :, None].expand(B, Tk, H, DR)], -1).transpose(1, 2)
    vv = F.pad(v, (0, DN + DR - DV)).transpose(1, 2)    # V padded to the QK head dim, as engines do
    return sdpa_causal(q, k, vv, start, scale)[..., :DV].transpose(1, 2)


def core_absorbed(I, ql, qpe, ckv, kp, start, scale):
    if I.core == "bi":
        return K.mla_attention(ql, qpe, ckv[:, :, None, :], kp, None, start, scale, **ABS_CFG)
    B, n, H, C = ql.shape
    Tk = ckv.shape[1]
    q = torch.cat([ql, qpe], -1).transpose(1, 2)        # [B, H, n, 576]
    kv = torch.cat([ckv, kp], -1)
    k = kv[:, None].expand(B, H, Tk, C + DR)
    v = ckv[:, None].expand(B, H, Tk, C)
    return sdpa_causal(q, k, v, start, scale).transpose(1, 2)


def new_mla_cache(B, Tmax, kvcache=False):
    c = {"ckv": torch.zeros(B, Tmax, C_m, device=dev, dtype=BF),
         "kpe": torch.zeros(B, Tmax, DR, device=dev, dtype=BF)}
    if kvcache:
        c["kvx"] = torch.zeros(B, Tmax, H_m, DN + DV, device=dev, dtype=BF)
    return c


def mla_step(P, I, form, x, cache, start):
    """One engine step: tokens x [B, n, d] at absolute positions start..start+n-1, latent
    cache (c_kv, k_pe) in `cache`. form: explicit (decompress the whole cached latent),
    explicit_kv (decompress only new tokens into a per-head K/V cache), absorbed.
    Returns the layer output [B, n, d] and the attention output [B, n, H, dv]."""
    B, n, _ = x.shape
    xf = x.reshape(B * n, d_m)
    q = I.mm(xf, P.W_q).view(B, n, H_m, DN + DR)
    kv = I.mm(xf, P.W_dkv)
    c = I.rms(kv[:, :C_m], P.g_kv).view(B, n, C_m)
    kpe = rope(kv[:, C_m:].reshape(B, n, DR), start)
    qpe = rope(q[..., DN:], start).contiguous()
    qn = q[..., :DN]
    Tk = start + n
    cache["ckv"][:, start:Tk] = c
    cache["kpe"][:, start:Tk] = kpe
    ckv, kp = cache["ckv"][:, :Tk], cache["kpe"][:, :Tk]
    if form in ("explicit", "explicit_kv"):
        if form == "explicit":
            kvx = I.mm(ckv.reshape(B * Tk, C_m), P.W_kvb).view(B, Tk, H_m, DN + DV)
        else:
            cache["kvx"][:, start:Tk] = I.mm(c.reshape(B * n, C_m), P.W_kvb).view(B, n, H_m, DN + DV)
            kvx = cache["kvx"][:, :Tk]
        o = core_explicit(I, qn, qpe, kvx[..., :DN], kp, kvx[..., DN:], start, P.scale)
    else:
        ql = I.heads_mm(qn.reshape(B * n, H_m, DN), P.W_UKT).reshape(B, n, H_m, C_m)
        ol = core_absorbed(I, ql, qpe, ckv, kp, start, P.scale)
        o = I.heads_mm(ol.reshape(B * n, H_m, C_m), P.W_UV).reshape(B, n, H_m, DV)
    out = I.mm(o.reshape(B * n, H_m * DV), P.W_o).view(B, n, d_m)
    return out, o


def schedule(T, chunk, form):
    c = T if chunk is None else chunk
    return [(s, min(s + c, T), form) for s in range(0, T, c)]


def mla_run(P, I, x, sched, cache=None):
    B, T, _ = x.shape
    if cache is None:
        cache = new_mla_cache(B, T, kvcache=any(f == "explicit_kv" for _, _, f in sched))
    outs, os_ = [], []
    for s, e, f in sched:
        out, o = mla_step(P, I, f, x[:, s:e], cache, s)
        outs.append(out)
        os_.append(o)
    return torch.cat(outs, 1), torch.cat(os_, 1)


def mla_stack_run(Ps, I, x, sched):
    """Layers h <- h + mla(h), one latent cache per layer, run on the schedule."""
    B, T, _ = x.shape
    caches = [new_mla_cache(B, T) for _ in Ps]
    outs = []
    for s, e, f in sched:
        h = x[:, s:e]
        for P, cache in zip(Ps, caches):
            h = h + mla_step(P, I, f, h, cache, s)[0]
        outs.append(h)
    return torch.cat(outs, 1)


def mla_ref64(P, x):
    """The same layer in fp64 (explicit math, no intermediate rounding). x [1, T, d]."""
    T = x.shape[1]
    xf = x.double().reshape(T, d_m)
    q = (xf @ P.W_q.double()).view(1, T, H_m, DN + DR)
    kv = xf @ P.W_dkv.double()
    c = kv[:, :C_m]
    c = c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + 1e-6) * P.g_kv.double()
    kpe = rope(kv[:, C_m:].reshape(1, T, DR), 0)[0]
    qpe = rope(q[..., DN:], 0)[0]
    kvx = (c @ P.W_kvb.double()).view(T, H_m, DN + DV)
    qq = torch.cat([q[0, :, :, :DN], qpe], -1).transpose(0, 1)
    kk = torch.cat([kvx[..., :DN], kpe[:, None].expand(T, H_m, DR)], -1).transpose(0, 1)
    vv = kvx[..., DN:].transpose(0, 1)
    s = (qq @ kk.transpose(-1, -2)) * P.scale
    s = s.masked_fill(torch.ones(T, T, device=dev, dtype=torch.bool).triu(1), NEG)
    o = (torch.softmax(s, -1) @ vv).transpose(0, 1)                         # [T, H, dv]
    return (o.reshape(T, H_m * DV) @ P.W_o.double())[None], o[None]


MLA_IMPLS = [
    Impl("cublas+torchnorm+sdpa (baseline)", "cublas", "torch", "sdpa"),
    Impl("cublas+torchnorm+bi_core", "cublas", "torch", "bi"),
    Impl("bi_gemm+bi_norm+sdpa", "bi", "triton", "sdpa"),
    Impl("bi_gemm+torchnorm+bi_core", "bi", "torch", "bi"),
    Impl("all_bi", "bi", "triton", "bi"),
]


def components_case(res):
    """Per-piece L2: a row alone vs inside batches (bi_check.check_batch)."""
    torch.manual_seed(0)
    P = MLAParams()
    out = {}
    sizes = (1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 1024, 2048)
    if QUICK:
        sizes = (1, 2, 3, 8, 16, 31, 63, 64, 65, 127, 256)
    cub, bi = Impl("cublas", "cublas", "torch"), Impl("bi", "bi", "triton")
    rows_d = torch.randn(2, d_m, device=dev).to(BF)
    rows_c = torch.randn(2, C_m, device=dev).to(BF)
    rows_hd = torch.randn(2, H_m, DN, device=dev).to(BF)
    rows_hc = torch.randn(2, H_m, C_m, device=dev).to(BF)
    rows_o = torch.randn(2, H_m * DV, device=dev).to(BF)
    checks = [
        ("W_q [2048->3072]", rows_d, lambda I: (lambda a: I.mm(a, P.W_q))),
        ("W_dkv [2048->576]", rows_d, lambda I: (lambda a: I.mm(a, P.W_dkv))),
        ("W_kvb decompress [512->4096]", rows_c, lambda I: (lambda a: I.mm(a, P.W_kvb))),
        ("absorb q: per-head [128->512]", rows_hd, lambda I: (lambda a: I.heads_mm(a, P.W_UKT))),
        ("absorb o: per-head [512->128]", rows_hc, lambda I: (lambda a: I.heads_mm(a, P.W_UV))),
        ("W_o [2048->2048]", rows_o, lambda I: (lambda a: I.mm(a, P.W_o))),
        ("rmsnorm [512]", rows_c, lambda I: (lambda a: I.rms(a, P.g_kv))),
    ]
    for name, rows, mk in checks:
        for I in (cub, bi):
            label = {"cublas": "torch.mm/bmm, torch-op norm", "bi": "one-config/grouped GEMM, Triton norm"}[I.name]
            rep = check_batch(mk(I), rows, sizes, name=f"{name} {label}")
            fs = rep.failures
            out[f"{name} | {I.name}"] = {"passed": rep.passed, "checked": rep.checked, "failed": len(fs),
                                         "failing_sizes": sorted({f["size"] for f in fs}),
                                         "max_mismatched_elements": max([f["mismatched_elements"] for f in fs
                                                                         if "mismatched_elements" in f] or [0]),
                                         "of": fs[0]["of"] if fs else None}
            print(rep.summary(), out[f"{name} | {I.name}"]["failing_sizes"][:12], flush=True)
    # DSA pieces: the fused projection, and the indexer logits q_I . k_I in fp32 for M query rows
    S = DSAParams()
    kI = torch.randn(2048, DI, device=dev).to(BF)
    rows_x = torch.randn(2, d_s, device=dev).to(BF)
    rows_qi = torch.randn(2, HI, DI, device=dev).to(BF)
    for name, rows, fns in (
            ("DSA fused projection [1024->1664]", rows_x,
             {"cublas": lambda a: a @ S.W_all, "bi": lambda a: mm_one_config(a, S.W_all)}),
            ("DSA indexer logits fp32 [M x 4 heads x 64] . [2048 keys]", rows_qi,
             {"cublas": lambda a: torch.matmul(a.float(), kI.float().t())})):
        # (the Triton indexer's causal mask ties a row to its position, so it is checked in the
        # sparse pipeline's L2/L3 tests instead)
        for which, fn in fns.items():
            rep_ = check_batch(fn, rows, sizes, name=f"{name} {which}")
            fs = rep_.failures
            out[f"{name} | {which}"] = {"passed": rep_.passed, "checked": rep_.checked, "failed": len(fs),
                                        "failing_sizes": sorted({f["size"] for f in fs}),
                                        "max_mismatched_elements": max([f["mismatched_elements"] for f in fs
                                                                        if "mismatched_elements" in f] or [0]),
                                        "of": fs[0]["of"] if fs else None}
            print(rep_.summary(), out[f"{name} | {which}"]["failing_sizes"][:12], flush=True)
    # the grouped kernel is the one-config kernel per group: bitwise equal to a loop
    a = torch.randn(300, H_m, DN, device=dev).to(BF)
    loop = torch.stack([mm_one_config(a[:, h], P.W_UKT[h]) for h in range(H_m)], 1)
    out["grouped_gemm_equals_one_config_loop"] = bool(torch.equal(bits(loop), bits(K.mm_grouped(a, P.W_UKT))))
    print("grouped == loop of mm_one_config:", out["grouped_gemm_equals_one_config_loop"])
    res["components"] = out


def mla_case(res):
    torch.manual_seed(0)
    P = MLAParams()
    T = 128 if QUICK else 512
    x = torch.randn(1, T, d_m, device=dev).to(BF)
    out = {"shapes": {"hidden": d_m, "heads": H_m, "kv_lora_rank": C_m, "qk_nope": DN, "qk_rope": DR,
                      "v_head": DV, "T": T, "absorbed_core_cfg": ABS_CFG, "explicit_core_cfg": EXP_CFG}}

    # 1. absorbed vs explicit, same input, against fp64
    ref_out, ref_o = mla_ref64(P, x)
    forms = {}
    for I in MLA_IMPLS:
        if I.name not in ("cublas+torchnorm+sdpa (baseline)", "all_bi"):
            continue
        oe, ae = mla_run(P, I, x, schedule(T, None, "explicit"))
        oa, aa = mla_run(P, I, x, schedule(T, None, "absorbed"))
        forms[I.name] = {
            "layer_out_mismatch": nmis(oe, oa), "attn_out_mismatch": nmis(ae, aa),
            "residual_stream_mismatch (x + layer_out)": nmis(x + oe, x + oa),
            "of_layer": oe.numel(), "of_attn": ae.numel(),
            "explicit_vs_fp64": {"layer": errs(oe, ref_out), "attn": errs(ae, ref_o)},
            "absorbed_vs_fp64": {"layer": errs(oa, ref_out), "attn": errs(aa, ref_o)}}
        print("forms", I.name, forms[I.name], flush=True)
    out["absorbed_vs_explicit"] = forms

    # 2. L3 within a form
    l3 = {}
    chunks = (7, 16, 64)
    for I in MLA_IMPLS:
        for form in ("explicit", "absorbed"):
            modes = {"prefill": lambda f=form: mla_run(P, I, x, schedule(T, None, f))[0],
                     "decode": lambda f=form: mla_run(P, I, x, schedule(T, 1, f))[0]}
            if form == "explicit":
                modes["decode (per-head K/V cache)"] = lambda: mla_run(P, I, x, schedule(T, 1, "explicit_kv"))[0]
            for c in chunks:
                modes[f"chunk-{c}"] = lambda c=c, f=form: mla_run(P, I, x, schedule(T, c, f))[0]
            rep = check_modes(modes, name=f"MLA {form} L3 [{I.name}]")
            l3[f"{form} | {I.name}"] = {"passed": rep.passed, "checked": rep.checked,
                                        "failures": {f["mode"]: f.get("mismatched_elements") for f in rep.failures},
                                        "of": T * d_m}
            print(rep.summary(), l3[f"{form} | {I.name}"]["failures"], flush=True)
    out["L3_modes"] = l3

    # 3. L2: prefill (whole sequences) and decode (one step, the probe's cache built alone)
    l2 = {}
    gen = torch.Generator(device=dev).manual_seed(1)
    batches_p = (2, 3) if QUICK else (2, 3, 8, 16)
    batches_d = (2, 3, 8) if QUICK else (2, 3, 8, 16, 32, 64)
    for I in MLA_IMPLS:
        for form in ("explicit", "absorbed"):
            ref = mla_run(P, I, x, schedule(T, None, form))[0][0]
            fails, checked = [], 0
            for B in batches_p:
                for pos in (0, B - 1):
                    xs = torch.randn(B, T, d_m, generator=gen, device=dev).to(BF)
                    xs[pos] = x[0]
                    checked += 1
                    mm = first_mismatch(ref, mla_run(P, I, xs, schedule(T, None, form))[0][pos])
                    if mm:
                        fails.append({"batch": B, "position": pos, "mismatched": mm["mismatched_elements"]})
            # decode: the probe's cache for positions < T-1 from its own prefill
            cache1 = new_mla_cache(1, T)
            mla_run(P, I, x[:, :T - 1], schedule(T - 1, None, form), cache=cache1)
            ref_d = mla_step(P, I, form, x[:, T - 1:], {k: v.clone() for k, v in cache1.items()}, T - 1)[0][0]
            dfails, dchecked = [], 0
            for B in batches_d:
                for pos in (0, B - 1):
                    cache = new_mla_cache(B, T)
                    cache["ckv"].copy_(torch.randn(B, T, C_m, generator=gen, device=dev).to(BF))
                    cache["kpe"].copy_(torch.randn(B, T, DR, generator=gen, device=dev).to(BF))
                    cache["ckv"][pos], cache["kpe"][pos] = cache1["ckv"][0], cache1["kpe"][0]
                    xs = torch.randn(B, 1, d_m, generator=gen, device=dev).to(BF)
                    xs[pos] = x[0, T - 1:]
                    dchecked += 1
                    mm = first_mismatch(ref_d, mla_step(P, I, form, xs, cache, T - 1)[0][pos])
                    if mm:
                        dfails.append({"batch": B, "position": pos, "mismatched": mm["mismatched_elements"]})
            l2[f"{form} | {I.name}"] = {"prefill": {"passed": not fails, "checked": checked, "failures": fails,
                                                    "of": T * d_m},
                                        "decode": {"passed": not dfails, "checked": dchecked, "failures": dfails,
                                                   "of": d_m}}
            print(f"MLA {form} L2 [{I.name}] prefill {'PASS' if not fails else fails} "
                  f"decode {'PASS' if not dfails else dfails}", flush=True)
    out["L2_batch"] = l2

    # 4. a mixed engine (explicit prefill of the prompt, absorbed decode) vs one form everywhere.
    #    The "trainer" recomputes the whole sequence in one prefill. With one layer the latent
    #    cache does not depend on the form; with two, layer 2's cache of the prompt does.
    P0 = T // 2
    mixed = {}
    Ps = [P, MLAParams(seed=1)]
    for I in MLA_IMPLS:
        if I.name not in ("cublas+torchnorm+sdpa (baseline)", "all_bi"):
            continue
        for nl in (1, 2):
            r = {}
            for eng_pre, eng_dec in (("explicit", "absorbed"), ("explicit", "explicit"), ("absorbed", "absorbed")):
                sched = [(0, P0, eng_pre)] + [(t, t + 1, eng_dec) for t in range(P0, T)]
                eng = mla_stack_run(Ps[:nl], I, x, sched)[:, P0:]
                for trainer in ("explicit", "absorbed"):
                    tr = mla_stack_run(Ps[:nl], I, x, schedule(T, None, trainer))[:, P0:]
                    r[f"engine {eng_pre} prefill + {eng_dec} decode vs trainer {trainer} prefill"] = nmis(eng, tr)
            mixed[f"{I.name} | {nl} layer(s)"] = {"mismatched_elements": r, "of": (T - P0) * d_m, "prompt_len": P0}
            print("mixed", I.name, nl, "layers", r, flush=True)
    out["mixed_forms"] = mixed

    # 5. L1: the same batch five times
    l1 = {}
    xb = torch.randn(4, T, d_m, device=dev).to(BF)
    for I in (MLA_IMPLS[0], MLA_IMPLS[-1]):
        for form in ("explicit", "absorbed"):
            rep = check_repeat(lambda _x, f=form: mla_run(P, I, _x, schedule(T, None, f))[0], xb,
                               name=f"MLA {form} L1 [{I.name}]")
            l1[f"{form} | {I.name}"] = {"passed": rep.passed, "checked": rep.checked}
            print(rep.summary(), flush=True)
    out["L1_repeat"] = l1
    res["mla"] = out


# ----------------------------------------------------------------------------- DSA-like sparse attention
d_s, HI, DI, H_s, D_s, KSEL = 1024, 4, 64, 8, 128, 64
NCOL = 1664                                            # q 1024 | k 128 | v 128 | qI 256 | kI 64 | w 4 | pad
BIG = 1 << 40


class DSAParams:
    def __init__(self, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        w = torch.randn(d_s, NCOL, generator=g, device=dev) * d_s ** -0.5
        w[:, 1604:] = 0
        self.W_all = w.to(BF)
        self.wscale = DI ** -0.5 * HI ** -0.5
        self.scale = D_s ** -0.5


def indexer(I, qi, ki, w, start):
    if I.indexer_kind == "bi":
        return K.indexer_scores(qi, ki, w, start)
    B, n = qi.shape[:2]
    Tk = ki.shape[1]
    lg = torch.matmul(qi.float().transpose(1, 2), ki.float().transpose(1, 2)[:, None])   # [B, HI, n, Tk]
    s = (lg.relu() * w.transpose(1, 2)[..., None]).sum(1)
    mask = torch.arange(Tk, device=dev)[None, :] > (start + torch.arange(n, device=dev))[:, None]
    return s.masked_fill(mask, NEG)


def _pad_sel(idx, k):
    if idx.shape[-1] < k:
        idx = F.pad(idx, (0, k - idx.shape[-1]), value=-1)
    return idx


def select_topk(scores, k):
    kk = min(k, scores.shape[-1])
    vals, idx = torch.topk(scores, kk, dim=-1)
    return _pad_sel(torch.where(vals == NEG, -1, idx), k)


def stable_key(scores):
    """int64 key ordering by (score desc, index asc): an order-preserving int from the fp32
    bits in the high word, the bit-inverted index in the low word. All keys are distinct."""
    b = scores.contiguous().view(torch.int32).to(torch.int64)
    b = torch.where(b < 0, b ^ 0x7FFFFFFF, b)
    pos = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int64)
    return (b << 32) | (0xFFFFFFFF - pos)


def select_stable(scores, k):
    """Top-k by (score desc, index asc) via a composite int64 key and torch.topk, returned in
    ascending index order, -1 padding last. Cross-check for select_stable_sort (-0.0 ranks
    below +0.0 here; the sort treats them as a tie)."""
    kk = min(k, scores.shape[-1])
    idx = torch.topk(stable_key(scores), kk, dim=-1).indices
    idx = torch.where(scores.gather(-1, idx) == NEG, BIG, idx).sort(-1).values
    return _pad_sel(torch.where(idx == BIG, -1, idx), k)


def select_stable_sort(scores, k):
    """The pipeline's stable top-k: a stable descending sort, so ties (including -0.0 vs +0.0)
    go to the lower index; the selection is returned in ascending index order, -1 last."""
    kk = min(k, scores.shape[-1])
    vals, idx = torch.sort(scores, dim=-1, descending=True, stable=True)
    idx = torch.where(vals[..., :kk] == NEG, BIG, idx[..., :kk]).sort(-1).values
    return _pad_sel(torch.where(idx == BIG, -1, idx), k)


def sattn(I, q, kc, vc, idx, scale):
    if I.sattn_kind == "bi":
        return K.sparse_attention(q, kc, vc, idx, scale)
    B, n = q.shape[:2]
    safe = idx.clamp_min(0)
    bidx = torch.arange(B, device=dev)[:, None, None]
    ks, vs = kc[bidx, safe], vc[bidx, safe]                                 # [B, n, K, D]
    s = torch.matmul(q, ks.transpose(-1, -2)).float() * scale              # [B, n, H, K]
    s = s.masked_fill((idx < 0)[:, :, None, :], NEG)
    return torch.matmul(torch.softmax(s, -1).to(BF), vs)


def new_dsa_cache(B, Tmax):
    return {n: torch.zeros(B, Tmax, dd, device=dev, dtype=BF) for n, dd in (("k", D_s), ("v", D_s), ("ki", DI))}


def dsa_step(S, I, x, cache, start):
    B, n, _ = x.shape
    y = I.mm(x.reshape(B * n, d_s), S.W_all).view(B, n, NCOL)
    q = y[..., :1024].unflatten(-1, (H_s, D_s))
    qi = y[..., 1280:1536].unflatten(-1, (HI, DI))
    w = y[..., 1600:1604].float() * S.wscale
    Tk = start + n
    cache["k"][:, start:Tk] = y[..., 1024:1152]
    cache["v"][:, start:Tk] = y[..., 1152:1280]
    cache["ki"][:, start:Tk] = y[..., 1536:1600]
    scores = indexer(I, qi, cache["ki"][:, :Tk], w, start)
    idx = (select_stable_sort if I.select_kind == "stable" else select_topk)(scores, KSEL)
    o = sattn(I, q, cache["k"][:, :Tk], cache["v"][:, :Tk], idx, S.scale)
    return o, idx, scores


def dsa_run(S, I, x, chunk=None, cache=None, keep_scores=False):
    B, T, _ = x.shape
    cache = new_dsa_cache(B, T) if cache is None else cache
    c = T if chunk is None else chunk
    os_, ids, scs = [], [], []
    for s in range(0, T, c):
        e = min(s + c, T)
        o, idx, sc = dsa_step(S, I, x[:, s:e], cache, s)
        os_.append(o)
        ids.append(idx)
        if keep_scores:
            scs.append(sc)
    return torch.cat(os_, 1), torch.cat(ids, 1), scs


def canon(idx):
    return torch.where(idx < 0, BIG, idx).sort(-1).values


def compare_sel(ref_o, ref_idx, o, idx):
    """Selected sets per query position, then the output: total mismatches and those in
    positions whose selected set did not change."""
    a, b = canon(ref_idx).reshape(-1, KSEL), canon(idx).reshape(-1, KSEL)
    differ = (a != b).any(-1)
    inb = (a[:, :, None] == b[:, None, :]).any(-1)
    flipped = int(((~inb) & (a != BIG)).sum())
    mis = (bits(ref_o) != bits(o)).reshape(a.shape[0], -1)
    return {"positions_with_changed_set": int(differ.sum()), "flipped_selections": flipped,
            "out_mismatch": int(mis.sum()), "out_mismatch_in_unchanged_sets": int(mis[~differ].sum()),
            "positions_with_out_mismatch": int(mis.any(-1).sum()), "positions": a.shape[0], "of": ref_o.numel()}


DSA_IMPLS = [
    Impl("cublas+torch_indexer+topk+torch_attn (baseline)", "cublas", indexer="torch", select="topk", sattn="torch"),
    Impl("cublas+torch_indexer+stable+bi_attn", "cublas", indexer="torch", select="stable", sattn="bi"),
    Impl("cublas+bi_indexer+stable+bi_attn", "cublas", indexer="bi", select="stable", sattn="bi"),
    Impl("bi_gemm+torch_indexer+topk+torch_attn", "bi", indexer="torch", select="topk", sattn="torch"),
    Impl("bi_indexer+topk+torch_attn", "bi", indexer="bi", select="topk", sattn="torch"),
    Impl("bi_indexer+stable+torch_attn", "bi", indexer="bi", select="stable", sattn="torch"),
    Impl("all_bi (bi_indexer+stable+bi_attn)", "bi", indexer="bi", select="stable", sattn="bi"),
]


def _cuda_kernels(fn):
    """Names of the CUDA kernels fn launches (to see which torch.topk path ran)."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.key[:90] for e in prof.key_averages() if e.device_type.name == "CUDA"})


def topk_ties_case(res):
    """torch.topk on rows with many exact ties at the k-th value: is the selected set, and the
    returned order, the same whatever else is in the batch and however long the row is?"""
    out = {}
    g = torch.Generator(device=dev).manual_seed(3)
    L, k = 2048, KSEL

    def tie_rows(n, kind):
        if kind == "32 levels":                        # ~64 copies of each of 32 values
            return torch.randint(0, 32, (n, L), generator=g, device=dev).float()
        r = torch.full((n, L), 1.0, device=dev)        # 40 distinct large values, the rest tied at 1.0
        top = torch.rand(n, L, generator=g, device=dev).argsort(-1)[:, :40]
        r.scatter_(1, top, 2.0 + torch.rand(n, 40, generator=g, device=dev))
        return r
    trials, rows_info = {}, {}
    for kind in ("32 levels", "40 above + plateau"):
        row = tie_rows(1, kind)[0]
        kth = row.sort(descending=True).values[k - 1]
        rows_info[kind] = {"ties_at_kth": int((row == kth).sum()), "needed_from_tie_group": int(k - (row > kth).sum())}
        stable = canon(select_stable(row[None], k))[0]
        rows_info[kind]["stable_key_equals_stable_sort"] = bool(torch.equal(stable, canon(select_stable_sort(row[None], k))[0]))
        alone_idx = torch.topk(row[None], k).indices[0]
        alone = canon(alone_idx[None])[0]
        rows_info[kind]["topk_alone_is_lowest_index"] = bool(torch.equal(alone, stable))

        def rec(name, idx_sorted, idx_unsorted=None):
            trials[f"{kind} | {name}"] = {"same_set_as_alone": bool(torch.equal(canon(idx_sorted[None])[0], alone)),
                                          "same_order_as_alone": bool(torch.equal(idx_sorted, alone_idx)),
                                          "lowest_index_set": bool(torch.equal(canon(idx_sorted[None])[0], stable))}
            if idx_unsorted is not None:
                trials[f"{kind} | {name}"]["sorted=False same_set"] = bool(torch.equal(canon(idx_unsorted[None])[0], alone))
        for N in (2, 8, 64, 512, 1024, 4096):
            for pos in (0, N - 1):
                for fill in ("ties", "random"):
                    others = tie_rows(N - 1, kind) if fill == "ties" else torch.randn(N - 1, L, generator=g, device=dev)
                    batch = torch.cat([others[:pos], row[None], others[pos:]])
                    rec(f"N={N} pos={pos} fill={fill}", torch.topk(batch, k).indices[pos],
                        torch.topk(batch, k, sorted=False).indices[pos])
        for padL in (2049, 4096, 8192, 32768):        # a decode row is shorter than a masked prefill row
            padded = torch.cat([row, torch.full((padL - L,), NEG, device=dev)])
            rec(f"row padded with -inf to {padL}", torch.topk(padded[None], k).indices[0])
        rb = row.to(BF)                                 # the same ties in bf16
        alone_b = torch.topk(rb[None], k).indices[0]
        for N in (64, 4096):
            batch = torch.cat([tie_rows(N - 1, kind).to(BF), rb[None]])
            got = torch.topk(batch, k).indices[-1]
            trials[f"{kind} | bf16 N={N} last"] = {"same_set_as_alone": bool(torch.equal(canon(got[None]), canon(alone_b[None]))),
                                                   "same_order_as_alone": bool(torch.equal(got, alone_b)),
                                                   "lowest_index_set": bool(torch.equal(canon(got[None])[0], stable))}
    out["rows"] = rows_info
    out["trials"] = trials
    out["n_trials"] = len(trials)
    for key in ("same_set_as_alone", "same_order_as_alone", "lowest_index_set", "sorted=False same_set"):
        out[f"failing_{key}"] = [k_ for k_, v in trials.items() if key in v and not v[key]]
    x1 = torch.randn(1, L, device=dev)
    xN = torch.randn(4096, L, device=dev)
    xlong = torch.randn(1, 32768, device=dev)
    out["kernels"] = {"1 row x 2048": _cuda_kernels(lambda: torch.topk(x1, k)),
                      "4096 rows x 2048": _cuda_kernels(lambda: torch.topk(xN, k)),
                      "1 row x 32768": _cuda_kernels(lambda: torch.topk(xlong, k))}
    print("topk ties:", rows_info, {k_: len(v) for k_, v in out.items() if k_.startswith("failing_")},
          "of", len(trials), "\nkernels:", json.dumps(out["kernels"], indent=0), flush=True)
    res["topk_ties"] = out


def sparse_case(res):
    torch.manual_seed(0)
    S = DSAParams()
    T = 256 if QUICK else 2048
    x = torch.randn(1, T, d_s, device=dev).to(BF)
    out = {"shapes": {"hidden": d_s, "indexer_heads": HI, "indexer_dim": DI, "heads": H_s, "head_dim": D_s,
                      "kv_heads": 1, "top_k": KSEL, "T": T}}
    gen = torch.Generator(device=dev).manual_seed(1)
    batches_p = (2, 3) if QUICK else (2, 3, 8, 16)
    batches_d = (2, 3, 8) if QUICK else (2, 3, 8, 16, 32, 64)
    chunks = (7, 16, 64)

    # natural ties and the two stable selections, on the invariant scores of the probe
    Ibi = DSA_IMPLS[-1]
    _, _, scs = dsa_run(S, Ibi, x, keep_scores=True)
    sc = scs[0][0]
    top = sc.sort(-1, descending=True).values[:, :KSEL + 1]
    tie = (top[:, KSEL - 1] == top[:, KSEL]) & (top[:, KSEL] > NEG)
    out["natural_boundary_ties"] = {"positions": int(tie.sum()), "of": T,
                                    "zero_scores_fraction_of_valid": float(((sc == 0).sum() / (sc > NEG).sum()))}
    out["stable_key_vs_stable_sort_rows_differ"] = int((canon(select_stable(sc, KSEL)) !=
                                                       canon(select_stable_sort(sc, KSEL))).any(-1).sum())
    out["torch_topk_vs_stable_rows_differ"] = int((canon(select_topk(sc, KSEL)) !=
                                                  canon(select_stable(sc, KSEL))).any(-1).sum())
    print("natural ties", out["natural_boundary_ties"], "stable key vs sort rows differ",
          out["stable_key_vs_stable_sort_rows_differ"], "topk vs stable rows differ",
          out["torch_topk_vs_stable_rows_differ"], flush=True)

    for I in DSA_IMPLS:
        r = {}
        ref_o, ref_idx, _ = dsa_run(S, I, x)
        # L3: modes vs prefill
        modes = {"decode": 1}
        modes.update({f"chunk-{c}": c for c in chunks})
        r["L3"] = {m: compare_sel(ref_o, ref_idx, *dsa_run(S, I, x, chunk=c)[:2]) for m, c in modes.items()}
        # L2 prefill
        l2p = {}
        for B in batches_p:
            for pos in (0, B - 1):
                xs = torch.randn(B, T, d_s, generator=gen, device=dev).to(BF)
                xs[pos] = x[0]
                o, idx, _ = dsa_run(S, I, xs)
                l2p[f"B={B} pos={pos}"] = compare_sel(ref_o, ref_idx, o[pos:pos + 1], idx[pos:pos + 1])
        r["L2_prefill"] = l2p
        # L2 decode: the probe's last token, its caches built alone; other rows random
        cache1 = new_dsa_cache(1, T)
        dsa_run(S, I, x[:, :T - 1], cache=cache1)
        c1 = {k: v.clone() for k, v in cache1.items()}
        ref_do, ref_di, _ = dsa_step(S, I, x[:, T - 1:], c1, T - 1)
        l2d = {}
        for B in batches_d:
            for pos in (0, B - 1):
                cache = {k: torch.randn(B, T, v.shape[-1], generator=gen, device=dev).to(BF) for k, v in cache1.items()}
                for k_ in cache:
                    cache[k_][pos] = cache1[k_][0]
                xs = torch.randn(B, 1, d_s, generator=gen, device=dev).to(BF)
                xs[pos] = x[0, T - 1:]
                o, idx, _ = dsa_step(S, I, xs, cache, T - 1)
                l2d[f"B={B} pos={pos}"] = compare_sel(ref_do, ref_di, o[pos:pos + 1], idx[pos:pos + 1])
        r["L2_decode"] = l2d
        # L2 over a whole decode: the probe generated token by token among B other sequences
        # (every step is a batch of B), vs decoded alone
        ref_seq_o, ref_seq_idx, _ = dsa_run(S, I, x, chunk=1)
        l2s = {}
        for B, pos in ((2, 1), (8, 0), (32, 31)) if not QUICK else ((2, 1),):
            xs = torch.randn(B, T, d_s, generator=gen, device=dev).to(BF)
            xs[pos] = x[0]
            o, idx, _ = dsa_run(S, I, xs, chunk=1)
            l2s[f"B={B} pos={pos}"] = compare_sel(ref_seq_o, ref_seq_idx, o[pos:pos + 1], idx[pos:pos + 1])
        r["L2_decode_sequence"] = l2s

        def agg(dct):
            return {"tests": len(dct),
                    "tests_failing": sum(1 for v in dct.values() if v["out_mismatch"] or v["positions_with_changed_set"]),
                    "max_positions_with_changed_set": max(v["positions_with_changed_set"] for v in dct.values()),
                    "max_flipped_selections": max(v["flipped_selections"] for v in dct.values()),
                    "max_out_mismatch": max(v["out_mismatch"] for v in dct.values()),
                    "max_out_mismatch_in_unchanged_sets": max(v["out_mismatch_in_unchanged_sets"] for v in dct.values())}
        r["summary"] = {k_: agg(r[k_]) for k_ in ("L3", "L2_prefill", "L2_decode", "L2_decode_sequence")}
        out[I.name] = r
        print("DSA", I.name, json.dumps(r["summary"]), flush=True)

    # L1
    xb = torch.randn(2, T, d_s, device=dev).to(BF)
    l1 = {}
    for I in (DSA_IMPLS[0], DSA_IMPLS[-1]):
        rep = check_repeat(lambda _x: dsa_run(S, I, _x)[0], xb, name=f"DSA L1 [{I.name}]")
        l1[I.name] = {"passed": rep.passed, "checked": rep.checked,
                      "failures": [f.get("mismatched_elements") for f in rep.failures]}
        print(rep.summary(), flush=True)
    out["L1_repeat"] = l1
    res["sparse"] = out


# ----------------------------------------------------------------------------- timing
def gpu_util():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as ex:          # noqa: BLE001
        return f"n/a ({ex})"


def timing_case(res):
    torch.manual_seed(0)
    out = {"nvidia_smi_before": gpu_util()}
    P = MLAParams()
    base, allbi = MLA_IMPLS[0], MLA_IMPLS[-1]
    # MLA prefill: B=4, T=1024, fresh cache
    Bp, Tp = (1, 128) if QUICK else (4, 1024)
    xp = torch.randn(Bp, Tp, d_m, device=dev).to(BF)
    mp = {}
    for I in (base, allbi):
        for form in ("explicit", "absorbed"):
            cache = new_mla_cache(Bp, Tp)
            mp[f"{form} | {I.name}"] = tm(lambda: mla_step(P, I, form, xp, cache, 0))
    out["mla_prefill_ms"] = {"B": Bp, "T": Tp, **mp}
    # MLA decode: one step at context L for B sequences, caches pre-filled
    md = {}
    for Bd in (1, 16, 64):
        L = 128 if QUICK else 1024
        xd = torch.randn(Bd, 1, d_m, device=dev).to(BF)
        cache = new_mla_cache(Bd, L, kvcache=True)
        for k_ in cache:
            cache[k_].normal_()
        for I in (base, allbi):
            for form in ("explicit", "explicit_kv", "absorbed"):
                md[f"B={Bd} | {form} | {I.name}"] = tm(lambda: mla_step(P, I, form, xd, cache, L - 1))
    out["mla_decode_ms"] = {"context": 128 if QUICK else 1024, **md}
    out["mla_cache_elements_per_token"] = {"latent (absorbed, or explicit re-decompressing)": C_m + DR,
                                           "per-head K/V (explicit_kv: k_nope, v per head + shared k_pe)":
                                           H_m * (DN + DV) + DR}
    print("MLA timing", json.dumps(out, indent=0), flush=True)

    # sparse: prefill B=4 T=2048, decode B=16 at context 2048
    S = DSAParams()
    dbase, dbi = DSA_IMPLS[0], DSA_IMPLS[-1]
    Bp, Tp = (1, 256) if QUICK else (4, 2048)
    xp = torch.randn(Bp, Tp, d_s, device=dev).to(BF)
    sp = {}
    for I in (dbase, dbi):
        cache = new_dsa_cache(Bp, Tp)
        sp[I.name] = tm(lambda: dsa_step(S, I, xp, cache, 0), iters=10)
    # pieces at prefill
    cache = new_dsa_cache(Bp, Tp)
    _, _, sc = dsa_step(S, dbi, xp, cache, 0)
    y = mm_one_config(xp.reshape(-1, d_s), S.W_all).view(Bp, Tp, NCOL)
    qi, w = y[..., 1280:1536].unflatten(-1, (HI, DI)), y[..., 1600:1604].float() * S.wscale
    q = y[..., :1024].unflatten(-1, (H_s, D_s))
    idx = select_stable_sort(sc, KSEL)
    pieces = {"indexer torch fp32 matmul": timed(lambda: indexer(dbase, qi, cache["ki"], w, 0), iters=10),
              "indexer bi kernel": timed(lambda: indexer(dbi, qi, cache["ki"], w, 0), iters=10),
              "select torch.topk": timed(lambda: select_topk(sc, KSEL), iters=10),
              "select stable key + topk": timed(lambda: select_stable(sc, KSEL), iters=10),
              "select stable sort": timed(lambda: select_stable_sort(sc, KSEL), iters=10),
              "sparse attn torch gather+matmul": timed(lambda: sattn(dbase, q, cache["k"], cache["v"], idx, S.scale), iters=10),
              "sparse attn bi kernel": timed(lambda: sattn(dbi, q, cache["k"], cache["v"], idx, S.scale), iters=10)}
    out["sparse_prefill_ms"] = {"B": Bp, "T": Tp, "step": sp, "pieces": pieces}
    Bd, L = (2, 256) if QUICK else (16, 2048)
    xd = torch.randn(Bd, 1, d_s, device=dev).to(BF)
    cache = new_dsa_cache(Bd, L)
    for k_ in cache:
        cache[k_].normal_()
    out["sparse_decode_ms"] = {"B": Bd, "context": L,
                               **{I.name: tm(lambda: dsa_step(S, I, xd, cache, L - 1)) for I in (dbase, dbi)}}
    out["nvidia_smi_after"] = gpu_util()
    print("sparse timing", json.dumps({k_: out[k_] for k_ in ("sparse_prefill_ms", "sparse_decode_ms")}, indent=0))
    res["timing"] = out


def tune_case(res):
    """Time a few fixed configurations of the absorbed core once, to pick the one pinned in
    ABS_CFG (any one of them is used for every mode; this only chooses which)."""
    out = {"nvidia_smi_before": gpu_util()}
    sc = (DN + DR) ** -0.5
    for label, B, Tq, Tk in (("prefill B=4 T=1024", 4, 1024, 1024), ("decode B=16 ctx=1024", 16, 1, 1024)):
        ql = torch.randn(B, Tq, H_m, C_m, device=dev).to(BF)
        qp = torch.randn(B, Tq, H_m, DR, device=dev).to(BF)
        ckv = torch.randn(B, Tk, C_m, device=dev).to(BF)
        kp = torch.randn(B, Tk, DR, device=dev).to(BF)
        r, same = {}, {}
        pinned = K.mla_attention(ql, qp, ckv[:, :, None], kp, None, Tk - Tq, sc, **ABS_CFG)
        for BM, BN, nw, ns in ((16, 32, 4, 1), (32, 32, 4, 1), (32, 32, 8, 1), (32, 64, 8, 1), (64, 32, 8, 1),
                               (16, 64, 4, 1)):
            cfg = dict(BM=BM, BN=BN, num_warps=nw, num_stages=ns)
            try:
                r[str(cfg)] = timed(lambda: K.mla_attention(ql, qp, ckv[:, :, None], kp, None, Tk - Tq, sc, **cfg))
                same[str(cfg)] = nmis(pinned, K.mla_attention(ql, qp, ckv[:, :, None], kp, None, Tk - Tq, sc, **cfg))
            except Exception as ex:      # noqa: BLE001
                r[str(cfg)] = f"{type(ex).__name__}: {str(ex)[:120]}"
        r["mismatched elements vs the pinned ABS_CFG"] = same
        qn = torch.randn(B, Tq, H_m, DN, device=dev).to(BF)
        kv = torch.randn(B, Tk, H_m, DN + DV, device=dev).to(BF)
        r["explicit core EXP_CFG (reference)"] = timed(
            lambda: K.mla_attention(qn, qp, kv[..., :DN], kp, kv[..., DN:], Tk - Tq, sc, **EXP_CFG))
        out[label] = r
        print(label, json.dumps(r, indent=0), flush=True)
    res["absorbed_core_configs_ms"] = out


def sdpa_probe(res):
    """Which SDPA backends accept the baseline's calls (default order: flash, efficient, math)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    B, H, T = 1, H_m, 64

    def r(*s):
        return torch.randn(*s, device=dev, dtype=BF)
    mask = torch.ones(16, T, dtype=torch.bool, device=dev).tril(T - 16)
    cases = {
        "explicit prefill causal (qk 192, v padded 192)": (r(B, H, T, 192), r(B, H, T, 192), r(B, H, T, 192),
                                                          dict(is_causal=True)),
        "explicit chunk with bool mask": (r(B, H, 16, 192), r(B, H, T, 192), r(B, H, T, 192), dict(attn_mask=mask)),
        "explicit decode, no mask": (r(B, H, 1, 192), r(B, H, T, 192), r(B, H, T, 192), {}),
        "absorbed prefill causal (qk 576, v 512, expanded MQA)": (
            r(B, H, T, 576), r(B, 1, T, 576).expand(B, H, T, 576), r(B, 1, T, 512).expand(B, H, T, 512),
            dict(is_causal=True)),
        "absorbed decode, no mask": (r(B, H, 1, 576), r(B, 1, T, 576).expand(B, H, T, 576),
                                     r(B, 1, T, 512).expand(B, H, T, 512), {}),
    }
    out = {}
    for name, (q, k, v, kw) in cases.items():
        ok = []
        for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
            try:
                with sdpa_kernel([be]):
                    sdpa(q, k, v, **kw)
                ok.append(be.name)
            except RuntimeError:
                pass
        out[name] = ok
    print("sdpa backends:", out)
    res["sdpa_backends_accepting"] = out


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["all"])
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true", help="reduced sizes, for the regression suite")
    ap.add_argument("--bi-json", help="also write the checks as bi-results-v1 (tools/bi_report.py)")
    args = ap.parse_args()
    QUICK = args.quick
    import triton
    res = {}
    if args.json and os.path.exists(args.json):
        with open(args.json) as fh:
            res = json.load(fh)
    res.update({"device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__})
    allc = {"sdpa_probe": sdpa_probe, "tune": tune_case, "components": components_case, "mla": mla_case, "topk_ties": topk_ties_case,
            "sparse": sparse_case, "timing": timing_case}
    names = list(allc) if args.cases == ["all"] else args.cases
    if QUICK and args.cases == ["all"]:
        names = [n for n in names if n not in ("sdpa_probe", "tune")]     # diagnostics, no claim
    for n in names:
        t0 = time.time()
        try:
            allc[n](res)
        except Exception as ex:          # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[n + "_error"] = f"{type(ex).__name__}: {ex}"
        print(f"== {n} done in {time.time() - t0:.1f}s", flush=True)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(res, fh, indent=1)
    if args.bi_json:
        sys.path.insert(0, os.path.join(HERE, "..", "..", "suite"))
        from adapters import emit
        sys.exit(emit("mla_sparse", res, args.bi_json, quick=QUICK))


if __name__ == "__main__":
    main()
