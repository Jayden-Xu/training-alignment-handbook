"""examples/moe_combine/moe_bi.py -> bi-results-v1. Claims: handbook/3-kernels/moe.md."""
from . import dig, parse_summary

GEMMS = (("K=1024 N=512", "expert up/gate GEMM 1024->512"), ("K=512 N=1024", "expert down GEMM 512->1024"),
         ("K=1024 N=64", "router GEMM 1024->64"))
# variant -> (GEMM, combine, expect L1, expect L2). The chosen design and the fixed-piece oracle are
# required. A deterministic or per-expert index_add_ over the one-config GEMM passed, but its order
# is the library's: recorded, not judged.
VARIANTS = {
    "fused": ("cuBLAS", "one index_add_ (atomics)", "fail", "fail"),
    "fused_det": ("cuBLAS", "deterministic index_add_", "info", "fail"),
    "hf_loop": ("cuBLAS", "index_add_ per expert", "info", "fail"),
    "fused+triton": ("one-config Triton", "one index_add_ (atomics)", "fail", "fail"),
    "fused_det+triton": ("one-config Triton", "deterministic index_add_", "info", "info"),
    "hf_loop+triton": ("one-config Triton", "index_add_ per expert", "info", "info"),
    "bi_triton": ("one-config Triton", "per token, top-k rank order, fp32", "pass", "pass"),
    "bi_chunk_loop": ("cuBLAS on fixed 32-row pieces, one call each", "per token, fixed order", "pass", "pass"),
    "bi_chunk_bmm": ("cuBLAS, the pieces in one batched matmul", "per token, fixed order", "info", "fail"),
}
ATOMIC = ("fused", "fused+triton")


def convert(L, A):
    gemm = L.get("gemm") or {}
    for shape, what in GEMMS:
        for lib, impl, expect in (("cublas", "torch.mm (cuBLAS)", "fail"),
                                  ("triton_one_config", "one-config Triton GEMM", "pass")):
            s = parse_summary(gemm.get(f"{lib} {shape}"))
            A.check(f"{what}: a row alone vs inside batches", dig(s, "passed"), level="L2", impl=impl,
                    expect=expect, detail={"comparisons": dig(s, "checked")},
                    signature={"failing_sizes": dig(s, "sizes")})
    for lib, v in (gemm.get("ms_4096x1024x512") or {}).items():
        A.measure("GEMM 4096x1024x512", v, "ms", impl=lib)

    for name, (g, comb, e1, e2) in VARIANTS.items():
        v = dig(L, "variants", name)
        if isinstance(v, dict) and "error" in v:
            A.error(f"variant {name}: {v['error']}")
            v = None
        impl = f"{name} ({g}; {comb})"
        l1, l2 = dig(v, "L1_repeat"), dig(v, "L2_batch")
        A.check("the same batch, 5 runs", dig(l1, "passed"), level="L1", impl=impl, expect=e1,
                detail={"comparisons": dig(l1, "checked")})
        s = parse_summary(dig(l2, "summary")) or {}
        A.check("a token alone vs inside batches, three positions, three fillers", dig(l2, "passed"),
                level="L2", impl=impl, expect=e2, detail={"comparisons": dig(l2, "checked")},
                signature=None if name in ATOMIC else {"failing_sizes": s.get("sizes")})
        A.measure("forward, 4096 tokens", dig(v, "ms_T4096"), "ms", impl=name)
        A.measure("mean abs difference vs fp32 reference", dig(v, "vs_fp32_ref", "mean_abs"), "abs", impl=name)
