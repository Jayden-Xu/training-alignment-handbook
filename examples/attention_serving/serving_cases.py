"""Serving layouts and deployment wrappers around batch-invariant kernels, checked bitwise.

1. paged      paged KV (page size P = 16/32/64/128, randomly permuted pages, NaN in
              unwritten slots) with 64-token compute blocks aligned to absolute
              positions. L3: paged prefill, decode, chunked prefill (7/16/100) vs the
              contiguous fixed-block prefill. L2: mixed batches (prefill, decode and
              chunk rows of sequences of different lengths in one launch). Contrast:
              compute block = page size. Timing: the cost of the gather.
2. varlen     packed sequences (cu_seqlens). L2: a sequence alone vs packed among
              neighbours at different offsets; with chunked queries (pos0 > 0) too.
              Contrast: blocks aligned to the pack instead of the sequence start.
              Backward: atomic-free, gradients alone vs packed; L1 repeat.
3. eager      a small block (RMSNorm, QKV, attention, O-proj, SwiGLU MLP): L2 over
              batch sizes 1..256, positions, fillers; BI kernels vs cuBLAS / SDPA.
4. compile    torch.compile (dynamic None/True/False; BI kernels traced directly,
              registered as opaque custom ops, or everything numeric opaque):
              L2 under compile, compiled vs eager, distinct graphs.
   compile_diag  which ops differ compiled vs eager; Dynamo's recompile reasons.
5. graphs     CUDA graphs captured at buckets 1..64, real batches padded to the bucket.
6. timing     eager vs compiled vs graphed (paged_timing and varlen_timing: the kernels).

The A40 is shared: timing sections wait (up to 90 s) for the GPU to be idle and record
nvidia-smi before and after.

Run (CUDA, Triton):  python serving_cases.py --json results-a40.json [--only paged,varlen,...]
Regression suite:    python serving_cases.py --quick --bi-json out.json
                     (--quick: every section except the wrapper timing; block batch sizes 1, 2, 3, 8,
                     31, 64; 3 mixed batches per page size; 5 of the 11 compile configurations.
                     --bi-json: the checks in the bi-results-v1 schema, through
                     suite/adapters/attention_serving.py)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, HERE)
from bi_check import DEFAULT_SIZES, bits, check_batch, check_modes, first_mismatch  # noqa: E402
from bi_kernels import BF, attention_fixed_blocks, attention_fixed_blocks_train, mm_one_config  # noqa: E402
from serving_kernels import (attention_paged, attention_varlen, attention_varlen_train,  # noqa: E402
                             bi_attn, bi_mm, rmsnorm_op, rmsnorm_torch, silu_mul_op, silu_mul_torch)

dev = "cuda"
HQ, HK, D = 8, 2, 128
PAGES = (16, 32, 64, 128)
sdpa = torch.nn.functional.scaled_dot_product_attention
QUICK = False                                   # set by --quick
QUICK_SECTIONS = "paged,paged_timing,varlen,varlen_timing,eager,compile,compile_diag,graphs"
QUICK_BLOCK_SIZES = (1, 2, 3, 8, 31, 64)
# one configuration per kind of row of the case page's compile table
QUICK_COMPILE_PLAN = [("bi", None, None), ("bi_opaque_kernels", None, None), ("bi_opaque_all", None, None),
                      ("bi_opaque_all", False, None), ("bi_opaque_all", None, "reduce-overhead")]


def i32(x):
    return torch.tensor(x, device=dev, dtype=torch.int32)


def rand(gen, *shape, scale=1.0):
    return (torch.randn(*shape, generator=gen, device=dev) * scale).to(BF)


def rand_seq(gen, L):
    """q [Hq, L, D], k, v [Hk, L, D], do [L, Hq, D]."""
    return {"q": rand(gen, HQ, L, D), "k": rand(gen, HK, L, D), "v": rand(gen, HK, L, D),
            "do": rand(gen, L, HQ, D), "L": L}


def ref_attention(s):
    """Contiguous fixed-block prefill (tools/bi_kernels.py), as packed rows [L, Hq, D]."""
    return attention_fixed_blocks(s["q"][None], s["k"][None], s["v"][None], 0)[0].transpose(0, 1)


def mm_count(a, b):
    m = first_mismatch(a, b)
    return 0 if m is None else m.get("mismatched_elements", -1)


def gpu_state():
    try:
        util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.split()
        return {"util_pct_mem_mib": util, "compute_procs": len(apps)}
    except Exception as ex:  # noqa: BLE001
        return {"error": str(ex)}


def wait_idle(max_s=90):
    """Wait (briefly) for the shared GPU to be idle apart from this process."""
    if QUICK:
        max_s = 15
    t0 = time.time()
    while True:
        st = gpu_state()
        try:
            util = int(st["util_pct_mem_mib"].split(",")[0])
        except Exception:  # noqa: BLE001
            return st
        if util <= 5 or time.time() - t0 > max_s:
            st["waited_s"] = round(time.time() - t0, 1)
            return st
        time.sleep(5)


def time_ms(fn, iters=20, warmup=3, trials=3):
    """Median over trials of the mean time per call (CUDA events)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    res = []
    for _ in range(trials):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        res.append(st.elapsed_time(en) / iters)
    return round(sorted(res)[len(res) // 2], 4)


# ============================================================================ 1. paged KV
class PagedCache:
    """A shared pool of pages in random order; unwritten slots hold NaN."""

    def __init__(self, lens, P, gen_cpu, spare=5):
        n_pages = [-(-L // P) for L in lens]
        total = sum(n_pages) + spare
        perm = torch.randperm(total, generator=gen_cpu)
        self.k = torch.full((total, P, HK, D), float("nan"), device=dev, dtype=BF)
        self.v = torch.full((total, P, HK, D), float("nan"), device=dev, dtype=BF)
        bt = torch.zeros(len(lens), max(n_pages), dtype=torch.int32)
        i = 0
        for b, n in enumerate(n_pages):
            bt[b, :n] = perm[i:i + n]
            i += n
        self.bt, self.P = bt.to(dev), P

    def write(self, b, k_tok, v_tok, start):
        """k_tok, v_tok [Hk, n, D] for absolute positions start..start+n-1 of sequence b."""
        pos = torch.arange(start, start + k_tok.shape[1], device=dev)
        pages, slots = self.bt[b, pos // self.P].long(), pos % self.P
        self.k[pages, slots] = k_tok.transpose(0, 1)
        self.v[pages, slots] = v_tok.transpose(0, 1)


def paged_run_chunks(s, P, c, gen_cpu, BN=64, stages=2):
    """One sequence, chunks of c queries: write the chunk's K/V into the cache, then attend."""
    T = s["L"]
    cache = PagedCache([T], P, gen_cpu)
    outs = []
    for st in range(0, T, c):
        e = min(st + c, T)
        cache.write(0, s["k"][:, st:e], s["v"][:, st:e], st)
        outs.append(attention_paged(s["q"][:, st:e].transpose(0, 1), cache.k, cache.v, cache.bt,
                                    i32([0, e - st]), i32([st]), e - st, BN=BN, num_stages=stages))
    return torch.cat(outs)


def paged_case(res):
    gen = torch.Generator(device=dev).manual_seed(10)
    gen_cpu = torch.Generator().manual_seed(10)
    out = {}
    T = 300
    s = rand_seq(gen, T)
    ref = ref_attention(s)

    # L3 at every page size
    l3 = {}
    for P in PAGES:
        modes = {"contiguous prefill": lambda: ref}
        for st in (2, 1):                             # num_stages changes scheduling only
            modes[f"paged prefill stages={st}"] = lambda P=P, st=st: paged_run_chunks(s, P, T, gen_cpu, stages=st)
            modes[f"paged decode stages={st}"] = lambda P=P, st=st: paged_run_chunks(s, P, 1, gen_cpu, stages=st)
            for c in (7, 16, 100):
                modes[f"paged chunk-{c} stages={st}"] = lambda P=P, c=c, st=st: paged_run_chunks(
                    s, P, c, gen_cpu, stages=st)
        rep = check_modes(modes, name=f"paged P={P} L3 (T={T})")
        l3[f"P={P}"] = rep.to_dict() | {"summary": rep.summary()}
        print(rep.summary())
    out["L3"] = l3

    # L2: mixed batches, sequences of different lengths and page placements
    lens = [1, 2, 37, 64, 65, 129, 300, 511, 1000, 77, 200]
    seqs = [rand_seq(gen, L) for L in lens]
    refs = [ref_attention(x) for x in seqs]
    kinds = ("all decode", "all prefill", "mixed") + (() if QUICK else ("mixed", "mixed"))
    l2 = {}
    for P in PAGES:
        fails, checked = [], 0
        for trial, kind in enumerate(kinds):
            order = torch.randperm(len(seqs), generator=gen_cpu).tolist()
            pos0 = []
            for i in order:
                L = lens[i]
                pick = kind if kind != "mixed" else ("decode", "prefill", "chunk")[
                    int(torch.randint(0, 3, (1,), generator=gen_cpu))]
                if pick in ("all decode", "decode"):
                    pos0.append(L - 1)
                elif pick in ("all prefill", "prefill") or L == 1:
                    pos0.append(0)
                else:
                    pos0.append(int(torch.randint(1, L, (1,), generator=gen_cpu)))
            cache = PagedCache([lens[i] for i in order], P, gen_cpu)
            for j, i in enumerate(order):
                cache.write(j, seqs[i]["k"], seqs[i]["v"], 0)
            q_rows = [seqs[i]["q"][:, p0:].transpose(0, 1) for i, p0 in zip(order, pos0)]
            cu = [0]
            for r in q_rows:
                cu.append(cu[-1] + r.shape[0])
            for st in (2, 1):
                o = attention_paged(torch.cat(q_rows), cache.k, cache.v, cache.bt, i32(cu), i32(pos0),
                                    max(r.shape[0] for r in q_rows), num_stages=st)
                for j, (i, p0) in enumerate(zip(order, pos0)):
                    checked += 1
                    mm = first_mismatch(refs[i][p0:], o[cu[j]:cu[j + 1]])
                    if mm:
                        fails.append({"trial": trial, "kind": kind, "seq_len": lens[i], "pos0": p0,
                                      "slot": j, "num_stages": st, **mm})
        l2[f"P={P}"] = {"checked": checked, "passed": not fails, "failures": fails[:10]}
        print(f"paged P={P} L2 mixed batches: {'PASS' if not fails else 'FAIL'} ({len(fails)}/{checked})")
    out["L2_mixed_batches"] = {"seq_lens": lens, "batches_per_P": len(kinds), "num_stages": [2, 1], **l2}

    # contrast: the compute block follows the page size (BN = P)
    contrast, outs = {}, {}
    for P in PAGES:
        try:
            outs[P] = paged_run_chunks(s, P, T, gen_cpu, BN=P)
            dec = paged_run_chunks(s, P, 1, gen_cpu, BN=P)
            contrast[f"BN=P={P}"] = {"mismatch_vs_contiguous_BN64": mm_count(ref, outs[P]),
                                     "of": ref.numel(),
                                     "decode_vs_prefill_same_BN": mm_count(outs[P], dec)}
        except Exception as ex:  # noqa: BLE001
            contrast[f"BN=P={P}"] = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
    pairs = {}
    ks = sorted(outs)
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            pairs[f"{ks[a]} vs {ks[b]}"] = mm_count(outs[ks[a]], outs[ks[b]])
    contrast["pairwise_mismatches"] = pairs
    contrast["max_abs_diff_vs_contiguous"] = {f"BN={P}": float((outs[P].float() - ref.float()).abs().max())
                                              for P in ks}
    out["contrast_BN_equals_P"] = contrast
    print("contrast BN=P:", contrast)
    res["paged"] = out


def paged_timing(res):
    gen = torch.Generator(device=dev).manual_seed(11)
    gen_cpu = torch.Generator().manual_seed(11)
    st = wait_idle()
    out = {"gpu_before": st}
    # prefill: 4 sequences of 2048
    B, T = 4, 2048
    q, k, v = rand(gen, B, HQ, T, D), rand(gen, B, HK, T, D), rand(gen, B, HK, T, D)
    ref = attention_fixed_blocks(q, k, v, 0)                      # [B, Hq, T, D]
    qp = q.transpose(1, 2).reshape(B * T, HQ, D).contiguous()
    kp = k.transpose(1, 2).reshape(B * T, HK, D).contiguous()
    vp = v.transpose(1, 2).reshape(B * T, HK, D).contiguous()
    cu = i32([i * T for i in range(B + 1)])
    z = i32([0] * B)
    ref_p = ref.transpose(1, 2).reshape(B * T, HQ, D)
    pre = {"contiguous (tools)": time_ms(lambda: attention_fixed_blocks(q, k, v, 0))}
    o = attention_varlen(qp, kp, vp, cu, cu, z, T)
    eq = {"varlen packed": mm_count(ref_p, o)}
    pre["varlen packed"] = time_ms(lambda: attention_varlen(qp, kp, vp, cu, cu, z, T))
    for P in PAGES:
        cache = PagedCache([T] * B, P, gen_cpu)
        for b in range(B):
            cache.write(b, k[b], v[b], 0)
        o = attention_paged(qp, cache.k, cache.v, cache.bt, cu, z, T)
        eq[f"paged P={P}"] = mm_count(ref_p, o)
        pre[f"paged P={P}"] = time_ms(lambda: attention_paged(qp, cache.k, cache.v, cache.bt, cu, z, T))
        # num_stages=1 (no software pipelining): same bits, timed separately
        o1 = attention_paged(qp, cache.k, cache.v, cache.bt, cu, z, T, num_stages=1)
        eq[f"paged P={P} num_stages=1"] = mm_count(ref_p, o1)
        pre[f"paged P={P} num_stages=1"] = time_ms(
            lambda: attention_paged(qp, cache.k, cache.v, cache.bt, cu, z, T, num_stages=1))
    out["prefill_B4_T2048_ms"] = pre
    out["prefill_B4_T2048_mismatch_vs_contiguous"] = eq
    # decode: 16 sequences, one query each at position 2047
    Bd = 16
    qd, kd, vd = rand(gen, Bd, HQ, 1, D), rand(gen, Bd, HK, T, D), rand(gen, Bd, HK, T, D)
    refd = attention_fixed_blocks(qd, kd, vd, T - 1)[:, :, 0]     # [Bd, Hq, D]
    qdp = qd[:, :, 0].contiguous()
    kdp = kd.transpose(1, 2).reshape(Bd * T, HK, D).contiguous()
    vdp = vd.transpose(1, 2).reshape(Bd * T, HK, D).contiguous()
    cuq, cuk, p0 = i32(list(range(Bd + 1))), i32([i * T for i in range(Bd + 1)]), i32([T - 1] * Bd)
    dec = {"contiguous (tools)": time_ms(lambda: attention_fixed_blocks(qd, kd, vd, T - 1), iters=50)}
    eqd = {"varlen packed": mm_count(refd, attention_varlen(qdp, kdp, vdp, cuq, cuk, p0, 1))}
    dec["varlen packed"] = time_ms(lambda: attention_varlen(qdp, kdp, vdp, cuq, cuk, p0, 1), iters=50)
    for P in PAGES:
        cache = PagedCache([T] * Bd, P, gen_cpu)
        for b in range(Bd):
            cache.write(b, kd[b], vd[b], 0)
        eqd[f"paged P={P}"] = mm_count(refd, attention_paged(qdp, cache.k, cache.v, cache.bt, cuq, p0, 1))
        dec[f"paged P={P}"] = time_ms(lambda: attention_paged(qdp, cache.k, cache.v, cache.bt, cuq, p0, 1),
                                      iters=50)
        eqd[f"paged P={P} num_stages=1"] = mm_count(
            refd, attention_paged(qdp, cache.k, cache.v, cache.bt, cuq, p0, 1, num_stages=1))
        dec[f"paged P={P} num_stages=1"] = time_ms(
            lambda: attention_paged(qdp, cache.k, cache.v, cache.bt, cuq, p0, 1, num_stages=1), iters=50)
    out["decode_B16_KV2048_ms"] = dec
    out["decode_B16_KV2048_mismatch_vs_contiguous"] = eqd
    out["gpu_after"] = gpu_state()
    print("paged timing:", json.dumps(out))
    res["paged_timing"] = out


# ============================================================================ 2. varlen
def build_pack(entries):
    """entries: list of (seq, pos0). Returns q, k, v, do packs, cu_q, cu_k, pos0, max_q and
    the q-row slice of each entry."""
    qs, ks, vs, dos, cu_q, cu_k, p0s, slices = [], [], [], [], [0], [0], [], []
    for s, p0 in entries:
        L = s["L"]
        qs.append(s["q"][:, p0:].transpose(0, 1))
        dos.append(s["do"][p0:])
        ks.append(s["k"].transpose(0, 1))
        vs.append(s["v"].transpose(0, 1))
        slices.append((cu_q[-1], cu_q[-1] + L - p0))
        cu_q.append(cu_q[-1] + L - p0)
        cu_k.append(cu_k[-1] + L)
        p0s.append(p0)
    cat = lambda xs: torch.cat(xs).contiguous()  # noqa: E731
    return dict(q=cat(qs), k=cat(ks), v=cat(vs), do=cat(dos), cu_q=i32(cu_q), cu_k=i32(cu_k),
                pos0=i32(p0s), max_q=max(b - a for a, b in slices), slices=slices, k_off=cu_k[:-1])


def vl_grads(pk, cu_key="cu_q"):
    q, k, v = (pk[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v"))
    o = attention_varlen_train(q, k, v, pk[cu_key], pk["max_q"])
    o.backward(pk["do"])
    torch.cuda.synchronize()
    return o.detach(), q.grad, k.grad, v.grad


def varlen_case(res):
    gen = torch.Generator(device=dev).manual_seed(20)
    out = {}
    probes = {L: rand_seq(gen, L) for L in (77, 200, 300)}
    # neighbour lengths before / after the probe; offsets include multiples of 64 (64, 128, 192)
    layouts = [((), ()), ((37,), ()), ((64,), ()), ((100, 28), (5,)), ((1,), (300,)),
               ((129, 63), ()), ((250,), (7, 90)), ((), (33, 33)), ((5, 6, 7), ()), ((511,), (64,))]

    # forward: BI (sequence-aligned blocks) vs naive (pack-aligned blocks), full prefill rows
    fwd = []
    for L, s in probes.items():
        ref = ref_attention(s)
        alone = build_pack([(s, 0)])
        o_alone = attention_varlen(alone["q"], alone["k"], alone["v"], alone["cu_q"], alone["cu_k"],
                                   alone["pos0"], alone["max_q"])
        fwd.append({"probe_len": L, "layout": "alone vs contiguous tools kernel",
                    "bi_mismatch": mm_count(ref, o_alone)})
        for before, after in layouts:
            ents = [(rand_seq(gen, n), 0) for n in before] + [(s, 0)] + [(rand_seq(gen, n), 0) for n in after]
            pk = build_pack(ents)
            a, b = pk["slices"][len(before)]
            koff = pk["k_off"][len(before)]
            o_bi = attention_varlen(pk["q"], pk["k"], pk["v"], pk["cu_q"], pk["cu_k"], pk["pos0"], pk["max_q"])
            o_nv = attention_varlen(pk["q"], pk["k"], pk["v"], pk["cu_q"], pk["cu_k"], pk["pos0"], pk["max_q"],
                                    align_pack=True)
            fwd.append({"probe_len": L, "before": list(before), "after": list(after), "offset": koff,
                        "offset_mod_64": koff % 64, "bi_mismatch": mm_count(o_alone, o_bi[a:b]),
                        "naive_mismatch": mm_count(o_alone, o_nv[a:b]), "of": o_alone.numel()})
    out["forward_prefill"] = fwd
    out["forward_prefill_bi_passed"] = all(r["bi_mismatch"] == 0 for r in fwd)
    print("varlen fwd BI all pass:", out["forward_prefill_bi_passed"])
    for r in fwd:
        print("  ", r)

    # forward with chunked queries in the pack (pos0 > 0): a 300-token sequence's rows 150..209
    s = probes[300]
    ref = ref_attention(s)
    chunk = {"q": s["q"][:, :210], "k": s["k"][:, :210], "v": s["v"][:, :210], "do": s["do"][:210], "L": 210}
    ch = []
    chunk_layouts = [([], []),
                     ([(rand_seq(gen, 100), 99)], []),                              # a decode row before
                     ([(rand_seq(gen, 64), 0)], [(rand_seq(gen, 513), 400)]),       # prefill before, chunk after
                     ([(rand_seq(gen, 1), 0), (rand_seq(gen, 70), 69)], [(rand_seq(gen, 30), 0)])]
    for before, after in chunk_layouts:
        pk = build_pack(before + [(chunk, 150)] + after)
        a, b = pk["slices"][len(before)]
        o = attention_varlen(pk["q"], pk["k"], pk["v"], pk["cu_q"], pk["cu_k"], pk["pos0"], pk["max_q"])
        o_nv = attention_varlen(pk["q"], pk["k"], pk["v"], pk["cu_q"], pk["cu_k"], pk["pos0"], pk["max_q"],
                                align_pack=True)
        ch.append({"k_offset": pk["k_off"][len(before)], "q_offset": a,
                   "bi_mismatch_vs_contiguous_prefill": mm_count(ref[150:210], o[a:b]),
                   "naive_mismatch": mm_count(ref[150:210], o_nv[a:b])})
    out["forward_chunk_in_pack"] = ch
    print("varlen chunk-in-pack:", ch)

    # lse variant
    pk = build_pack([(probes[200], 0), (probes[77], 0), (probes[300], 0)])
    lse = torch.empty(HQ, pk["q"].shape[0], device=dev, dtype=torch.float32)
    o1 = attention_varlen(pk["q"], pk["k"], pk["v"], pk["cu_q"], pk["cu_k"], pk["pos0"], pk["max_q"])
    o2 = attention_varlen(pk["q"], pk["k"], pk["v"], pk["cu_q"], pk["cu_k"], pk["pos0"], pk["max_q"], lse=lse)
    out["lse_variant_mismatch_vs_plain"] = mm_count(o1, o2)

    # backward: alone vs the tools kernel, alone vs packed, repeat
    names = ("o", "dq", "dk", "dv")
    bwd = []
    for L, s in probes.items():
        alone = vl_grads(build_pack([(s, 0)]))
        qt, kt, vt = (s[n][None].clone().requires_grad_(True) for n in ("q", "k", "v"))
        ot = attention_fixed_blocks_train(qt, kt, vt)
        ot.backward(s["do"].transpose(0, 1)[None].contiguous())
        tools = (ot.detach()[0].transpose(0, 1), qt.grad[0].transpose(0, 1), kt.grad[0].transpose(0, 1),
                 vt.grad[0].transpose(0, 1))
        bwd.append({"probe_len": L, "layout": "alone vs tools attention_fixed_blocks_train",
                    **{n: mm_count(t, a) for n, t, a in zip(names, tools, alone)}})
        for before, after in layouts:
            ents = [(rand_seq(gen, n), 0) for n in before] + [(s, 0)] + [(rand_seq(gen, n), 0) for n in after]
            pk = build_pack(ents)
            a, b = pk["slices"][len(before)]
            got = vl_grads(pk)
            bwd.append({"probe_len": L, "before": list(before), "after": list(after),
                        "offset": pk["k_off"][len(before)],
                        **{n: mm_count(r, g[a:b]) for n, r, g in zip(names, alone, got)}})
    out["backward"] = bwd
    out["backward_passed"] = all(all(r[n] == 0 for n in names) for r in bwd)
    print("varlen bwd all pass:", out["backward_passed"])
    for r in bwd:
        if any(r[n] for n in names):
            print("   FAIL", r)
    ents = [(rand_seq(gen, n), 0) for n in (300, 77, 1024, 5, 200, 640)]
    pk = build_pack(ents)
    ref_g = vl_grads(pk)
    rep = []
    for i in range(1, 5):
        got = vl_grads(pk)
        rep.append({n: mm_count(r, g) for n, r, g in zip(names, ref_g, got)})
    out["backward_repeat_L1"] = {"runs": 5, "tokens": pk["q"].shape[0], "mismatches": rep,
                                 "passed": all(all(v == 0 for v in r.values()) for r in rep)}
    print("varlen bwd L1:", out["backward_repeat_L1"]["passed"])
    res["varlen"] = out


def varlen_timing(res):
    gen = torch.Generator(device=dev).manual_seed(21)
    st = wait_idle()
    out = {"gpu_before": st}
    B, T = 4, 2048
    seqs = [rand_seq(gen, T) for _ in range(B)]
    pk = build_pack([(s, 0) for s in seqs])
    q = torch.stack([s["q"] for s in seqs])
    k = torch.stack([s["k"] for s in seqs])
    v = torch.stack([s["v"] for s in seqs])
    do = torch.stack([s["do"].transpose(0, 1) for s in seqs]).contiguous()

    def tools_fb():
        qq, kk, vv = (t.detach().requires_grad_(True) for t in (q, k, v))
        attention_fixed_blocks_train(qq, kk, vv).backward(do)

    def vl_fb():
        qq, kk, vv = (pk[n].detach().requires_grad_(True) for n in ("q", "k", "v"))
        attention_varlen_train(qq, kk, vv, pk["cu_q"], T).backward(pk["do"])

    out["fwd_B4_T2048_ms"] = {
        "contiguous (tools)": time_ms(lambda: attention_fixed_blocks(q, k, v, 0)),
        "varlen packed": time_ms(lambda: attention_varlen(pk["q"], pk["k"], pk["v"], pk["cu_q"], pk["cu_k"],
                                                          pk["pos0"], T))}
    out["fwd_bwd_B4_T2048_ms"] = {"contiguous (tools)": time_ms(tools_fb, iters=10),
                                  "varlen packed": time_ms(vl_fb, iters=10)}
    out["gpu_after"] = gpu_state()
    print("varlen timing:", out)
    res["varlen_timing"] = out


# ============================================================================ 3-6. deployment wrappers
H, INTER, T_BLK = HQ * D, 2048, 100
BLOCK_SIZES = tuple(s for s in DEFAULT_SIZES if s <= 256)


def block_sizes():
    return QUICK_BLOCK_SIZES if QUICK else BLOCK_SIZES


def block_weights(seed=30):
    gen = torch.Generator(device=dev).manual_seed(seed)

    def w(i, o):
        return rand(gen, i, o, scale=1 / math.sqrt(i))
    return {"g1": (1 + 0.1 * torch.randn(H, generator=gen, device=dev)).to(BF),
            "g2": (1 + 0.1 * torch.randn(H, generator=gen, device=dev)).to(BF),
            "wqkv": w(H, (HQ + 2 * HK) * D), "wo": w(HQ * D, H), "wgu": w(H, 2 * INTER), "wd": w(INTER, H)}


def make_block(W, mm, attn, norm, act):
    """Pre-norm block: h = x + Wo·attn(QKV(norm(x))); out = h + Wd·silu_mul(Wgu·norm(h))."""
    def block(x):
        B, T, Hd = x.shape
        h = x.reshape(B * T, Hd)
        qkv = mm(norm(h, W["g1"]), W["wqkv"])
        q = qkv[:, :HQ * D].view(B, T, HQ, D).transpose(1, 2)
        k = qkv[:, HQ * D:(HQ + HK) * D].view(B, T, HK, D).transpose(1, 2)
        v = qkv[:, (HQ + HK) * D:].view(B, T, HK, D).transpose(1, 2)
        o = attn(q, k, v).transpose(1, 2).reshape(B * T, HQ * D)
        h = h + mm(o, W["wo"])
        a = act(mm(norm(h, W["g2"]), W["wgu"]), INTER)
        return (h + mm(a, W["wd"])).view(B, T, Hd)
    return block


def _attn_bi(q, k, v):
    return attention_fixed_blocks(q, k, v, 0)


def _attn_bi_op(q, k, v):
    return bi_attn(q, k, v, 0)


def _attn_sdpa(q, k, v):
    g = HQ // HK
    return sdpa(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True)


def blocks(W):
    return {
        # BI kernels called directly (Dynamo traces the user-defined Triton kernels)
        "bi": make_block(W, mm_one_config, _attn_bi, rmsnorm_torch, silu_mul_torch),
        # BI kernels behind torch.library.custom_op; norm and SiLU still torch ops
        "bi_opaque_kernels": make_block(W, bi_mm, _attn_bi_op, rmsnorm_torch, silu_mul_torch),
        # every op that rounds (GEMMs, attention, norm, SiLU*mul) behind a custom op
        "bi_opaque_all": make_block(W, bi_mm, _attn_bi_op, rmsnorm_op, silu_mul_op),
        "cublas": make_block(W, torch.mm, _attn_bi, rmsnorm_torch, silu_mul_torch),
        "stock": make_block(W, torch.mm, _attn_sdpa, rmsnorm_torch, silu_mul_torch),
    }


def probe_rows(seed=31, R=2):
    gen = torch.Generator(device=dev).manual_seed(seed)
    return rand(gen, R, T_BLK, H)


def filler(kind, n, gen):
    if kind == "zeros":
        return torch.zeros(n, T_BLK, H, device=dev, dtype=BF)
    return rand(gen, n, T_BLK, H, scale=1e3 if kind == "large" else 1.0)


def digest(t):
    return hashlib.blake2b(bits(t).cpu().numpy().tobytes(), digest_size=8).hexdigest()


def eager_case(res):
    W = block_weights()
    bl = blocks(W)
    rows = probe_rows()
    out = {"shape": {"tokens_per_seq": T_BLK, "hidden": H, "inter": INTER, "heads": [HQ, HK, D]}}
    with torch.no_grad():
        for name in ("bi", "bi_opaque_all", "cublas", "stock"):
            rep = check_batch(bl[name], rows, block_sizes(), name=f"eager block {name}")
            fails = rep.failures
            out[name] = {"summary": rep.summary(), "passed": rep.passed, "checked": rep.checked,
                         "failing_sizes": sorted({f["size"] for f in fails}),
                         "max_mismatched_elements": max([f["mismatched_elements"] for f in fails], default=0),
                         "of": rows[0].numel(), "failures": fails[:5]}
            print(rep.summary(), out[name]["failing_sizes"])
        x = rows[:1]
        out["opaque_all_vs_bi_B1"] = mm_count(bl["bi"](x), bl["bi_opaque_all"](x))
        out["cublas_vs_bi_B1"] = mm_count(bl["bi"](x), bl["cublas"](x))
    res["eager"] = out


def compile_sweep(label, eager_fn, fn, rows, sizes, dynamic, mode=None, fullgraph=True):
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    out = {"variant": label, "dynamic": str(dynamic), "mode": mode or "default", "fullgraph": fullgraph}
    ro = mode == "reduce-overhead"

    def call(f, x):
        if ro:
            torch.compiler.cudagraph_mark_step_begin()
        return f(x).clone()
    gen = torch.Generator(device=dev).manual_seed(32)
    R = rows.shape[0]
    try:
        cfn = torch.compile(fn, dynamic=dynamic, fullgraph=fullgraph, mode=mode)
        t0 = time.time()
        refc = [call(cfn, rows[r:r + 1])[0] for r in range(R)]
        out["first_call_s"] = round(time.time() - t0, 1)
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {str(ex)[:1500]}"
        print(label, dynamic, "ERROR", out["error"][:500])
        return out
    refe = [eager_fn(rows[r:r + 1])[0] for r in range(R)]
    out["B1_compiled_vs_eager_mismatch"] = [mm_count(refe[r], refc[r]) for r in range(R)]
    groups = [defaultdict(set) for _ in range(R)]
    per_size, t0 = {}, time.time()
    for size in sizes:
        g0 = counters["stats"]["unique_graphs"]
        rec = {"checked": 0, "fail_vs_compiled_B1": 0, "fail_vs_eager": 0, "max_mm_vs_compiled_B1": 0,
               "max_mm_vs_eager": 0}
        for r in range(R):
            row = rows[r:r + 1]
            for pos in (("first", "middle", "last") if size > 1 else ("first",)):
                p = {"first": 0, "middle": size // 2, "last": size - 1}[pos]
                for kind in (("random", "zeros", "large") if size > 1 else ("random",)):
                    others = filler(kind, size - 1, gen)
                    o = call(cfn, torch.cat([others[:p], row, others[p:]]))[p]
                    rec["checked"] += 1
                    c, e = mm_count(refc[r], o), mm_count(refe[r], o)
                    rec["fail_vs_compiled_B1"] += c > 0
                    rec["fail_vs_eager"] += e > 0
                    rec["max_mm_vs_compiled_B1"] = max(rec["max_mm_vs_compiled_B1"], c)
                    rec["max_mm_vs_eager"] = max(rec["max_mm_vs_eager"], e)
                    groups[r][digest(o)].add(size)
        rec["new_graphs"] = counters["stats"]["unique_graphs"] - g0
        per_size[size] = rec
    out["sweep_s"] = round(time.time() - t0, 1)
    out["graphs_compiled"] = counters["stats"]["unique_graphs"]
    out["graph_breaks"] = sum(counters["graph_break"].values())
    out["L2_vs_compiled_B1"] = all(r["fail_vs_compiled_B1"] == 0 for r in per_size.values())
    out["equals_eager_all_sizes"] = all(r["fail_vs_eager"] == 0 for r in per_size.values()) and \
        all(m == 0 for m in out["B1_compiled_vs_eager_mismatch"])
    out["distinct_outputs_per_probe"] = [len(g) for g in groups]
    out["size_groups_probe0"] = sorted([sorted(s) for s in groups[0].values()], key=lambda s: s[0])
    out["failing_sizes_vs_compiled_B1"] = [s for s, r in per_size.items() if r["fail_vs_compiled_B1"]]
    out["failing_sizes_vs_eager"] = [s for s, r in per_size.items() if r["fail_vs_eager"]]
    out["per_size"] = per_size
    print(f"compile {label} dynamic={dynamic} mode={out['mode']}: L2={out['L2_vs_compiled_B1']} "
          f"eq_eager={out['equals_eager_all_sizes']} B1 vs eager={out['B1_compiled_vs_eager_mismatch']} "
          f"graphs={out['graphs_compiled']} groups={out['size_groups_probe0']} "
          f"fail_vs_eager={out['failing_sizes_vs_eager']}", flush=True)
    return out


def compile_case(res, variants=None):
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64          # default 8: past it, Dynamo silently runs eager
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    import torch._inductor.config as icfg
    icfg.fx_graph_cache = False                         # every graph compiled and autotuned afresh
    if hasattr(icfg, "autotune_local_cache"):
        icfg.autotune_local_cache = False
    res["compile_config"] = {"cache_size_limit": 64, "fx_graph_cache": icfg.fx_graph_cache,
                             "autotune_local_cache": getattr(icfg, "autotune_local_cache", None),
                             "sizes": list(block_sizes()), "probe_rows": 2,
                             "positions": ["first", "middle", "last"], "fillers": ["random", "zeros", "large"]}
    W = block_weights()
    bl = blocks(W)
    rows = probe_rows()
    out = res.setdefault("compile", {})
    plan = variants or (QUICK_COMPILE_PLAN if QUICK else
                        [(v, d, None) for v in ("bi", "bi_opaque_kernels", "bi_opaque_all")
                         for d in (None, True, False)] + [("bi", None, "reduce-overhead"),
                                                          ("bi_opaque_all", None, "reduce-overhead")])
    with torch.no_grad():
        for name, dyn, mode in plan:
            key = f"{name} dynamic={dyn}" + (f" mode={mode}" if mode else "")
            out[key] = compile_sweep(name, bl["bi"], bl[name], rows, block_sizes(), dyn, mode)
            if "error" in out[key]:                       # retry allowing graph breaks
                out[key + " fullgraph=False"] = compile_sweep(name, bl["bi"], bl[name], rows, block_sizes(),
                                                              dyn, mode, fullgraph=False)


def _attn_from_rows(x):
    """Attention on q, k, v cut from one sequence's rows x [T, H] (diagnostic only)."""
    q = x[:, :HQ * D].view(1, T_BLK, HQ, D).transpose(1, 2)
    k = x[:, :HK * D].view(1, T_BLK, HK, D).transpose(1, 2)
    v = x[:, HK * D:2 * HK * D].view(1, T_BLK, HK, D).transpose(1, 2)
    return _attn_bi(q, k, v)


def compile_diag(res):
    """Where compiled and eager part ways, op by op, and why Dynamo recompiles per size."""
    import torch._dynamo
    from torch._dynamo.utils import guard_failures
    torch._dynamo.config.cache_size_limit = 64
    W = block_weights()
    rows = probe_rows()
    out = {}
    with torch.no_grad():
        h = rows[0]                                                   # [T, H] tokens of one sequence
        y = mm_one_config(rmsnorm_torch(h, W["g1"]), W["wqkv"])[:, :H].contiguous()
        gu = mm_one_config(rmsnorm_torch(h, W["g2"]), W["wgu"])
        ops = {"rmsnorm": (lambda x: rmsnorm_torch(x, W["g1"]), h),
               "add + rmsnorm (residual feeding the norm)": (lambda x: rmsnorm_torch(x + y, W["g2"]), h),
               "add (residual alone)": (lambda x: x + y, h),
               "silu(gate) * up": (lambda g: silu_mul_torch(g, INTER), gu),
               "mm_one_config alone": (lambda x: mm_one_config(x, W["wqkv"]), h),
               "attention_fixed_blocks alone": (_attn_from_rows, h)}
        per_op = {}
        for name, (f, x) in ops.items():
            torch._dynamo.reset()
            try:
                per_op[name] = {"mismatch": mm_count(f(x), torch.compile(f)(x)), "of": f(x).numel()}
            except Exception as ex:  # noqa: BLE001
                per_op[name] = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
        out["per_op_compiled_vs_eager"] = per_op
        print("per-op compiled vs eager:", per_op, flush=True)

        # recompile reasons for the block (default dynamic=None), sizes 1, 2, 3, 4
        reasons = {}
        for name in ("bi", "bi_opaque_all"):
            torch._dynamo.reset()
            guard_failures.clear()
            cfn = torch.compile(blocks(W)[name])
            for b in (1, 2, 3, 4):
                cfn(rows[:1].expand(b, -1, -1).contiguous())
            reasons[name] = {k.co_name: [str(r)[:300] for r in v] for k, v in guard_failures.items()}
        out["recompile_reasons_sizes_1_to_4"] = reasons
        print("recompile reasons:", json.dumps(reasons, indent=1)[:3000], flush=True)
    res["compile_diag"] = out


class GraphRunner:
    """One CUDA graph per bucket; a batch of n is zero-padded to the smallest bucket >= n."""

    def __init__(self, fn, buckets):
        self.buckets, self.graphs = buckets, {}
        for b in buckets:
            sx = torch.zeros(b, T_BLK, H, device=dev, dtype=BF)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn(sx)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                so = fn(sx)
            self.graphs[b] = (g, sx, so)

    def __call__(self, batch):
        n = batch.shape[0]
        b = min(x for x in self.buckets if x >= n)
        g, sx, so = self.graphs[b]
        sx.zero_()
        sx[:n].copy_(batch)
        g.replay()
        return so[:n].clone(), b


def graphs_case(res):
    W = block_weights()
    bl = blocks(W)
    rows = probe_rows()
    buckets = (1, 2, 4, 8, 16, 32, 64)
    reals = (1, 2, 3, 5, 8, 9, 16, 17, 31, 33, 64)
    out = {"buckets": buckets, "real_batch_sizes": reals}
    gen = torch.Generator(device=dev).manual_seed(33)
    with torch.no_grad():
        for name in ("bi", "cublas", "stock"):
            fn = bl[name]
            runner = GraphRunner(fn, buckets)
            refe = [fn(rows[r:r + 1])[0] for r in range(rows.shape[0])]
            per_bucket = defaultdict(lambda: {"checked": 0, "fail_vs_eager_B1": 0, "max_mm": 0})
            groups = defaultdict(set)
            for n in reals:
                for r in range(rows.shape[0]):
                    for p in sorted({0, n - 1}):
                        others = filler("random", n - 1, gen)
                        o, b = runner(torch.cat([others[:p], rows[r:r + 1], others[p:]]))
                        m = mm_count(refe[r], o[p])
                        rec = per_bucket[b]
                        rec["checked"] += 1
                        rec["fail_vs_eager_B1"] += m > 0
                        rec["max_mm"] = max(rec["max_mm"], m)
                        if r == 0:
                            groups[digest(o[p])].add(b)
            # graph replay vs eager on the same padded batch
            same = {}
            for b in buckets:
                x = filler("random", b, gen)
                g, sx, so = runner.graphs[b]
                sx.copy_(x)
                g.replay()
                same[b] = mm_count(fn(x), so)
            out[name] = {"per_bucket": dict(per_bucket),
                         "passed": all(v["fail_vs_eager_B1"] == 0 for v in per_bucket.values()),
                         "bucket_groups_probe0": sorted([sorted(s) for s in groups.values()], key=lambda s: s[0]),
                         "graph_vs_eager_same_shape_mismatch": same}
            print(f"graphs {name}: passed={out[name]['passed']} groups={out[name]['bucket_groups_probe0']} "
                  f"same-shape={same}", flush=True)
            del runner
            torch.cuda.empty_cache()
    res["graphs"] = out


def wrappers_timing(res):
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    W = block_weights()
    bl = blocks(W)
    gen = torch.Generator(device=dev).manual_seed(34)
    sizes = (1, 8, 64)
    xs = {b: rand(gen, b, T_BLK, H) for b in sizes}
    out = {"tokens_per_seq": T_BLK}
    with torch.no_grad():
        torch._dynamo.reset()
        comp = {"compiled bi": torch.compile(bl["bi"]),
                "compiled bi_opaque_all": torch.compile(bl["bi_opaque_all"]),
                "compiled bi reduce-overhead": torch.compile(bl["bi"], mode="reduce-overhead"),
                "compiled bi_opaque_all reduce-overhead": torch.compile(bl["bi_opaque_all"], mode="reduce-overhead")}
        for f in comp.values():                          # compile every size before timing
            for b in sizes:
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    f(xs[b])
        runners = {n: GraphRunner(bl[n], sizes) for n in ("bi", "cublas", "stock")}
        out["gpu_before"] = wait_idle()
        for b in sizes:
            x = xs[b]
            row = {}
            for n in ("bi", "bi_opaque_all", "cublas", "stock"):
                row[f"eager {n}"] = time_ms(lambda n=n: bl[n](x), iters=20)
            for n, f in comp.items():
                if "reduce-overhead" in n:
                    def step(f=f):
                        torch.compiler.cudagraph_mark_step_begin()
                        f(x)
                    row[n] = time_ms(step, iters=20)
                else:
                    row[n] = time_ms(lambda f=f: f(x), iters=20)
            for n, rr in runners.items():
                g, sx, so = rr.graphs[b]
                sx.copy_(x)
                row[f"cuda graph {n}"] = time_ms(g.replay, iters=20)
            out[f"B={b}"] = row
            print(f"timing B={b}:", row, flush=True)
        out["gpu_after"] = gpu_state()
    res["wrappers_timing"] = out


# ============================================================================ main
SECTIONS = {"paged": paged_case, "paged_timing": paged_timing, "varlen": varlen_case,
            "varlen_timing": varlen_timing, "eager": eager_case, "compile": compile_case,
            "compile_diag": compile_diag, "graphs": graphs_case, "timing": wrappers_timing}


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--only", help="sections to run (default: all; with --quick: all but timing)")
    ap.add_argument("--quick", action="store_true", help="reduced sizes, for the regression suite")
    ap.add_argument("--bi-json", help="also write the checks as bi-results-v1 (tools/bi_report.py)")
    args = ap.parse_args()
    QUICK = args.quick
    if args.only is None:
        args.only = QUICK_SECTIONS if QUICK else ",".join(SECTIONS)
    res = {}
    if args.json and os.path.exists(args.json):
        with open(args.json) as fh:
            res = json.load(fh)
    import triton
    res.update({"device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__})
    for name in args.only.split(","):
        t0 = time.time()
        try:
            SECTIONS[name](res)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[name + "_error"] = f"{type(ex).__name__}: {str(ex)[:1000]}"
        res.setdefault("section_seconds", {})[name] = round(time.time() - t0, 1)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(res, fh, indent=1, default=str)
        print(f"== {name} done in {time.time() - t0:.1f}s", flush=True)
    if args.bi_json:
        sys.path.insert(0, os.path.join(HERE, "..", "..", "suite"))
        from adapters import emit
        sys.exit(emit("attention_serving", res, args.bi_json, quick=QUICK))


if __name__ == "__main__":
    main()
