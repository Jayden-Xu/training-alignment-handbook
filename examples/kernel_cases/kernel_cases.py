"""Batch-invariance cases beyond MoE, checked bitwise on one GPU.

1. attention   fixed-block Triton attention vs torch SDPA:
               L2 (a sequence alone vs among other sequences in the batch) and
               L3 (prefill vs per-token decode vs chunked prefill at 7/16/64/100)
2. fp8_scales  activation quantization scale granularity (emulated: FP8 values,
               dequantized to bf16, one-config GEMM): per-tensor dynamic vs
               per-token vs per-128-block
3. linear_attn chunked vs recurrent causal linear attention (fp32): two exact
               forms of the same math

Run (CUDA, Triton):  python kernel_cases.py --json results.json
Regression suite:    python kernel_cases.py --quick --bi-json out.json
                     (--quick: fewer FP8 batch sizes and timing iterations; --bi-json: the checks in
                     the bi-results-v1 schema, through suite/adapters/kernel_cases.py)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from bi_check import check_batch, check_modes, first_mismatch  # noqa: E402
from bi_kernels import BF, attention_fixed_blocks, mm_one_config  # noqa: E402

dev = "cuda"
sdpa = torch.nn.functional.scaled_dot_product_attention
QUICK = False


def timed(fn, iters=20):
    if QUICK:
        iters = 5
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


# ----------------------------------------------------------------------------- attention
def _expand(kv, g):
    return kv.repeat_interleave(g, dim=1)


def attn_sdpa(q, k, v, start):
    g = q.shape[1] // k.shape[1]
    Tq, Tk = q.shape[2], start + q.shape[2]
    k, v = _expand(k[:, :, :Tk], g), _expand(v[:, :, :Tk], g)
    if start == 0:
        return sdpa(q, k, v, is_causal=True)
    mask = torch.ones(Tq, Tk, device=q.device, dtype=torch.bool).tril(diagonal=start)
    return sdpa(q, k, v, attn_mask=mask)


def run_modes(attn, q, k, v, chunks=(7, 16, 64, 100)):
    T = q.shape[2]

    def chunked(c):
        def run():
            return torch.cat([attn(q[:, :, s:min(s + c, T)], k, v, s) for s in range(0, T, c)], 2)
        return run
    modes = {"prefill": lambda: attn(q, k, v, 0),
             "decode": lambda: torch.cat([attn(q[:, :, t:t + 1], k, v, t) for t in range(T)], 2)}
    for c in chunks:
        modes[f"chunk-{c}"] = chunked(c)
    return modes


def attention_case(res):
    torch.manual_seed(0)
    T, Hq, Hk, D = 256, 8, 2, 128
    q = torch.randn(1, Hq, T, D, device=dev).to(BF)
    k = torch.randn(1, Hk, T, D, device=dev).to(BF)
    v = torch.randn(1, Hk, T, D, device=dev).to(BF)
    out = {}
    for name, attn in (("sdpa", attn_sdpa), ("fixed_blocks", attention_fixed_blocks)):
        rep = check_modes(run_modes(attn, q, k, v), name=f"{name} L3 modes")
        out[f"{name}_L3"] = rep.to_dict() | {"summary": rep.summary()}
        print(rep.summary())

        # L2: the probe sequence alone vs inside batches of other sequences (prefill)
        fails, checked = [], 0
        ref = attn(q, k, v, 0)[0]
        gen = torch.Generator(device=dev).manual_seed(1)
        for B in (2, 3, 8, 16):
            for pos in (0, B - 1):
                qs, ks, vs = (torch.randn(B, *x.shape[1:], generator=gen, device=dev).to(BF) for x in (q, k, v))
                qs[pos], ks[pos], vs[pos] = q[0], k[0], v[0]
                checked += 1
                mm = first_mismatch(ref, attn(qs, ks, vs, 0)[pos])
                if mm:
                    fails.append({"batch": B, "position": pos, **mm})
        out[f"{name}_L2"] = {"checked": checked, "failures": fails[:5], "passed": not fails}
        print(f"{name} L2 batch: {'PASS' if not fails else 'FAIL %d/%d' % (len(fails), checked)}")

    # accuracy against fp32
    qf, kf, vf = q.float(), _expand(k.float(), Hq // Hk), _expand(v.float(), Hq // Hk)
    ref32 = sdpa(qf, kf, vf, is_causal=True)
    out["vs_fp32"] = {n: float((f(q, k, v, 0).float() - ref32).abs().max())
                      for n, f in (("sdpa", attn_sdpa), ("fixed_blocks", attention_fixed_blocks))}
    # cost: prefill and decode
    Bp, Tp = 4, 2048
    qp = torch.randn(Bp, Hq, Tp, D, device=dev).to(BF)
    kp = torch.randn(Bp, Hk, Tp, D, device=dev).to(BF)
    vp = torch.randn(Bp, Hk, Tp, D, device=dev).to(BF)
    ke, ve = _expand(kp, Hq // Hk), _expand(vp, Hq // Hk)
    Bd = 16
    qd = torch.randn(Bd, Hq, 1, D, device=dev).to(BF)
    kd = torch.randn(Bd, Hk, Tp, D, device=dev).to(BF)
    vd = torch.randn(Bd, Hk, Tp, D, device=dev).to(BF)
    kde, vde = _expand(kd, Hq // Hk), _expand(vd, Hq // Hk)
    out["ms"] = {
        "prefill_B4_T2048": {"sdpa": timed(lambda: sdpa(qp, ke, ve, is_causal=True)),
                             "fixed_blocks": timed(lambda: attention_fixed_blocks(qp, kp, vp, 0))},
        "decode_B16_KV2048": {"sdpa": timed(lambda: sdpa(qd, kde, vde)),
                              "fixed_blocks": timed(lambda: attention_fixed_blocks(qd, kd, vd, Tp - 1))},
    }
    print("attention ms", out["ms"], "max|Δ| vs fp32", out["vs_fp32"])
    res["attention"] = out


# ----------------------------------------------------------------------------- fp8 scales
FP8_MAX = 448.0


def _q(x, scale):
    return ((x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale).to(BF)


def quant_per_tensor(x):
    return _q(x, x.float().abs().amax().clamp_min(1e-12) / FP8_MAX)


def quant_per_token(x):
    return _q(x, x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / FP8_MAX)


def quant_per_block(x, blk=128):
    M, H = x.shape
    xb = x.view(M, H // blk, blk)
    s = xb.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    return _q(xb, s).view(M, H)


def fp8_case(res):
    torch.manual_seed(0)
    H, N = 1024, 1024
    W = (torch.randn(H, N, device=dev) * H ** -0.5).to(BF)
    Wq = quant_per_block(W.t().contiguous()).t().contiguous()      # static weight quantization
    rows = torch.randn(2, H, device=dev).to(BF)
    out = {}
    for name, qf in (("per_tensor_dynamic", quant_per_tensor), ("per_token", quant_per_token),
                     ("per_block_1x128", quant_per_block)):
        rep = check_batch(lambda x, qf=qf: mm_one_config(qf(x), Wq), rows,
                          sizes=(1, 2, 3, 8, 31, 64) if QUICK else (1, 2, 3, 8, 16, 31, 32, 64, 128, 256),
                          name=f"fp8 {name}")
        out[name] = rep.to_dict() | {"summary": rep.summary(),
                                     "fillers_failing": sorted({f["filler"] for f in rep.failures})}
        print(rep.summary(), out[name]["fillers_failing"])
    res["fp8_scales"] = out


# ----------------------------------------------------------------------------- linear attention
def lin_recurrent(q, k, v):
    B, Hh, T, D = q.shape
    S = torch.zeros(B, Hh, D, D, device=q.device)
    outs = []
    for t in range(T):
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]
        outs.append(torch.einsum("bhd,bhde->bhe", q[:, :, t], S))
    return torch.stack(outs, 2)


def lin_chunked(c):
    def run(q, k, v):
        B, Hh, T, D = q.shape
        S = torch.zeros(B, Hh, D, D, device=q.device)
        outs = []
        for s in range(0, T, c):
            e = min(s + c, T)
            qc, kc, vc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
            mask = torch.ones(e - s, e - s, device=q.device).tril()
            intra = ((qc @ kc.transpose(-1, -2)) * mask) @ vc
            outs.append(intra + qc @ S)
            S = S + kc.transpose(-1, -2) @ vc
        return torch.cat(outs, 2)
    return run


def linear_attn_case(res):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 128, 64, device=dev) * 0.1 for _ in range(3))
    rep = check_modes({"recurrent": lambda: lin_recurrent(q, k, v),
                       "chunk-16": lambda: lin_chunked(16)(q, k, v),
                       "chunk-64": lambda: lin_chunked(64)(q, k, v)}, name="linear attention modes")
    ref64 = lin_recurrent(q.double(), k.double(), v.double())
    res["linear_attention"] = rep.to_dict() | {
        "summary": rep.summary(),
        "max_abs_vs_fp64": {n: float((f(q, k, v).double() - ref64).abs().max())
                            for n, f in (("recurrent", lin_recurrent), ("chunk-16", lin_chunked(16)))}}
    print(rep.summary(), res["linear_attention"]["max_abs_vs_fp64"])


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true", help="fewer FP8 batch sizes and timing iterations")
    ap.add_argument("--bi-json", help="also write the checks as bi-results-v1 (tools/bi_report.py)")
    args = ap.parse_args()
    QUICK = args.quick
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__}
    import triton
    res["triton"] = triton.__version__
    for case in (attention_case, fp8_case, linear_attn_case):
        try:
            case(res)
        except Exception as ex:
            res[case.__name__] = {"error": f"{type(ex).__name__}: {ex}"}
            print(case.__name__, "ERROR", ex)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    if args.bi_json:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "suite"))
        from adapters import emit
        sys.exit(emit("kernel_cases", res, args.bi_json, quick=QUICK))


if __name__ == "__main__":
    main()
