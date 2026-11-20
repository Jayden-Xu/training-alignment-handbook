"""MoE layer: common implementations against a batch-invariant one, checked bitwise.

Variants (same math: softmax router, top-k, renormalized weights, SwiGLU experts):
  fused     one index_add_ of all T*K expert outputs into the token rows (atomics)
  fused_det the same under torch.use_deterministic_algorithms(True)
  hf_loop   loop over experts, one index_add_ per expert (HF Mixtral style)
  bi        every GEMM (router, experts) runs in fixed CHUNK-row pieces, so the
            library always sees M = CHUNK; combine sums each token's K outputs
            in top-k rank order in fp32, one row per token, no atomics
  bi_bmm    as bi, but the fixed-size chunks go through one batched matmul

Run (CUDA):  python moe_bi.py --json results.json
Regression suite:  python moe_bi.py --quick --bi-json out.json
                   (--quick: batch sizes 1, 2, 3, 8, 31, 33, 64, 127, 128 and 3 timing iterations;
                   --bi-json: the checks in the bi-results-v1 schema, through suite/adapters/moe_combine.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # needed by deterministic mode
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from bi_check import check_batch, check_repeat, first_mismatch  # noqa: E402

H, I, E, K = 1024, 512, 64, 8
CHUNK = 32
BF = torch.bfloat16
QUICK = False
QUICK_SIZES = [1, 2, 3, 8, 31, 33, 64, 127, 128]      # every documented failure shows at one of these


def make_params(seed=0, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s, sc: (torch.randn(*s, generator=g, device=dev) * sc).to(BF)
    return {"Wr": r(H, E, sc=H ** -0.5), "W1": r(E, H, I, sc=H ** -0.5),
            "W3": r(E, H, I, sc=H ** -0.5), "W2": r(E, I, H, sc=I ** -0.5)}


def route(x, P, mm):
    probs = mm(x, P["Wr"]).float().softmax(-1)
    w, ids = probs.topk(K, -1)
    return w / w.sum(-1, keepdim=True), ids


def expert(xe, P, e, mm):
    return mm(torch.nn.functional.silu(mm(xe, P["W1"][e])) * mm(xe, P["W3"][e]), P["W2"][e])


def mm_plain(a, b):
    return a @ b


@triton.jit(do_not_specialize=["M"])
def _mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """One tile config for every M; K walked in ascending BK steps into fp32; no split-K."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.bfloat16),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def mm_triton(a, b, BM=64, BN=64, BK=32):
    M, Kd = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=BF)
    if M == 0:
        return c
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mm_kernel[grid](a, b, c, M, N, Kd, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return c


def mm_fixed(a, b):
    """Run a @ b in fixed CHUNK-row pieces (zero-padded), one library call per piece."""
    M = a.shape[0]
    pad = (-M) % CHUNK
    if pad:
        a = torch.cat([a, a.new_zeros(pad, a.shape[1])])
    return torch.cat([a[i:i + CHUNK] @ b for i in range(0, a.shape[0], CHUNK)])[:M]


def mm_fixed_bmm(a, b):
    M = a.shape[0]
    pad = (-M) % CHUNK
    if pad:
        a = torch.cat([a, a.new_zeros(pad, a.shape[1])])
    return torch.matmul(a.view(-1, CHUNK, a.shape[1]), b).reshape(-1, b.shape[1])[:M]


def moe_fused(x, P, mm=mm_plain):
    T = x.shape[0]
    w, ids = route(x, P, mm)
    flat = ids.reshape(-1)
    order = flat.argsort()
    tok = order // K
    counts = torch.bincount(flat, minlength=E).tolist()
    xs, ys, s = x[tok], [], 0
    for e, c in enumerate(counts):
        if c:
            ys.append(expert(xs[s:s + c], P, e, mm))
            s += c
    y = torch.cat(ys) * w.reshape(-1)[order].unsqueeze(1).to(BF)
    out = torch.zeros_like(x)
    out.index_add_(0, tok, y)                       # all K contributions at once: atomics
    return out


def moe_fused_det(x, P, mm=mm_plain):
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return moe_fused(x, P, mm)
    finally:
        torch.use_deterministic_algorithms(prev)


def moe_hf_loop(x, P, mm=mm_plain):
    w, ids = route(x, P, mm)
    out = torch.zeros_like(x)
    for e in range(E):
        t, slot = (ids == e).nonzero(as_tuple=True)
        if t.numel():
            out.index_add_(0, t, expert(x[t], P, e, mm) * w[t, slot, None].to(BF))
    return out


def _moe_bi(x, P, mm):
    T = x.shape[0]
    w, ids = route(x, P, mm)
    flat = ids.reshape(-1)
    order = flat.argsort(stable=True)
    tok, slot = order // K, order % K
    counts = torch.bincount(flat, minlength=E).tolist()
    xs, ys, s = x[tok], [], 0
    for e, c in enumerate(counts):
        if c:
            ys.append(expert(xs[s:s + c], P, e, mm))
            s += c
    y = torch.cat(ys).float() * w[tok, slot].unsqueeze(1)
    per_tok = torch.empty(T, K, H, device=x.device, dtype=torch.float32)
    per_tok[tok, slot] = y                          # data movement: unique (token, rank) slots
    acc = per_tok[:, 0]
    for j in range(1, K):                           # fixed order: top-k rank 0..K-1, fp32
        acc = acc + per_tok[:, j]
    return acc.to(BF)


def moe_bi(x, P):
    return _moe_bi(x, P, mm_fixed)


def moe_bi_bmm(x, P):
    return _moe_bi(x, P, mm_fixed_bmm)


def moe_ref_fp32(x, P):
    """fp32 reference of the same math, for a numerical sanity check only."""
    Pf = {k: v.float() for k, v in P.items()}
    xf = x.float()
    probs = (xf @ Pf["Wr"]).softmax(-1)
    w, ids = probs.topk(K, -1)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(xf)
    for e in range(E):
        t, slot = (ids == e).nonzero(as_tuple=True)
        if t.numel():
            xe = xf[t]
            ye = (torch.nn.functional.silu(xe @ Pf["W1"][e]) * (xe @ Pf["W3"][e])) @ Pf["W2"][e]
            out.index_add_(0, t, ye * w[t, slot, None])
    return out


VARIANTS = {
    # stock GEMMs (cuBLAS)
    "fused": moe_fused, "fused_det": moe_fused_det, "hf_loop": moe_hf_loop,
    # one-config Triton GEMM, combine varied: attributes L1/L2 to GEMM vs combine
    "fused+triton": lambda x, P: moe_fused(x, P, mm_triton),
    "fused_det+triton": lambda x, P: moe_fused_det(x, P, mm_triton),
    "hf_loop+triton": lambda x, P: moe_hf_loop(x, P, mm_triton),
    # batch-invariant: fixed-order fp32 combine
    "bi_triton": lambda x, P: _moe_bi(x, P, mm_triton),
    "bi_chunk_loop": moe_bi, "bi_chunk_bmm": moe_bi_bmm}


def timed(fn, x, P, iters=10):
    if QUICK:
        iters = 3
    fn(x, P)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(x, P)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--sizes", type=int, nargs="*", default=[1, 2, 3, 8, 16, 31, 32, 33, 64, 127, 128, 256, 512])
    ap.add_argument("--quick", action="store_true", help="fewer batch sizes and timing iterations")
    ap.add_argument("--bi-json", help="also write the checks as bi-results-v1 (tools/bi_report.py)")
    args = ap.parse_args()
    QUICK = args.quick
    if QUICK:
        args.sizes = QUICK_SIZES
    gemm_sizes = list(args.sizes) + ([] if QUICK else [1024, 2048])
    torch.manual_seed(0)
    P = make_params()
    rows = torch.randn(2, H, device="cuda").to(BF)
    xb = torch.randn(256, H, device="cuda").to(BF)
    xt = torch.randn(4096, H, device="cuda").to(BF)
    ref = moe_ref_fp32(xb, P)
    gemm = {}
    for gname, g in (("cublas", mm_plain), ("triton_one_config", mm_triton)):
        for (kd, n) in ((H, I), (I, H), (H, E)):
            W = (torch.randn(kd, n, device="cuda") * kd ** -0.5).to(BF)
            rr = torch.randn(2, kd, device="cuda").to(BF)
            rep = check_batch(lambda x, W=W, g=g: g(x, W), rr, gemm_sizes,
                              name=f"{gname} K={kd} N={n}")
            gemm[rep.name] = rep.summary()
            print(rep.summary())
    W1 = P["W1"][0].contiguous()
    xg = torch.randn(4096, H, device="cuda").to(BF)
    gemm["ms_4096x1024x512"] = {"cublas": timed(lambda a, b: a @ b, xg, W1),
                                "triton_one_config": timed(mm_triton, xg, W1)}
    print("gemm ms", gemm["ms_4096x1024x512"])
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
           "config": {"H": H, "I": I, "E": E, "K": K, "CHUNK": CHUNK}, "gemm": gemm, "variants": {}}
    for name, fn in VARIANTS.items():
        f = lambda x, fn=fn: fn(x, P)
        entry = {}
        try:
            rep = check_repeat(f, xb, n=5, name=name)
            entry["L1_repeat"] = rep.to_dict() | {"summary": rep.summary()}
            bat = check_batch(f, rows, args.sizes, name=name)
            entry["L2_batch"] = bat.to_dict() | {"summary": bat.summary()}
            err = (fn(xb, P).float() - ref).abs()
            entry["vs_fp32_ref"] = {"max_abs": float(err.max()), "mean_abs": float(err.mean()),
                                    "ref_mean_abs": float(ref.abs().mean())}
            entry["ms_T4096"] = timed(fn, xt, P)
        except Exception as ex:  # report, do not hide
            entry["error"] = f"{type(ex).__name__}: {ex}"
        res["variants"][name] = entry
        print(f"{name:18s}", entry.get("error") or
              f"| {entry['L1_repeat']['summary'].split(': ',1)[1]} | {entry['L2_batch']['summary']} | "
              f"{entry['ms_T4096']:.1f} ms @T=4096 | max|Δ| vs fp32 {entry['vs_fp32_ref']['max_abs']:.2e}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    if args.bi_json:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "suite"))
        from adapters import emit
        sys.exit(emit("moe_combine", res, args.bi_json, quick=QUICK))


if __name__ == "__main__":
    main()
