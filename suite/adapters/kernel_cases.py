"""examples/kernel_cases/kernel_cases.py -> bi-results-v1. Claims: handbook/3-kernels/attention.md, handbook/3-kernels/gemm-and-quantized.md and handbook/3-kernels/linear-attention.md."""
from . import dig, parse_summary


def convert(L, A):
    att = L.get("attention") or {}
    # 1. causal attention: prefill, decode, chunked prefill (L3) and a sequence among others (L2)
    for impl, expect in (("fixed_blocks", "pass"), ("sdpa", "fail")):
        l3 = att.get(f"{impl}_L3")
        s = parse_summary(dig(l3, "summary")) or {}
        A.check("prefill, per-token decode and chunked prefill give the same bits",
                dig(l3, "passed"), level="L3", impl=impl, expect=expect,
                detail={"comparisons": dig(l3, "checked")}, signature={"failing_modes": s.get("modes")})
        l2 = att.get(f"{impl}_L2")
        fails = dig(l2, "failures", default=[])
        A.check("a sequence alone vs among other sequences", dig(l2, "passed"), level="L2", impl=impl,
                expect=expect, detail={"comparisons": dig(l2, "checked")},
                signature={"failing_batches_first_5_failures": sorted({f.get("batch") for f in fails})})
    for group, unit in (("prefill_B4_T2048", "ms"), ("decode_B16_KV2048", "ms")):
        for impl, v in (dig(att, "ms", group) or {}).items():
            A.measure(group.replace("_", " "), v, unit, impl=impl)
    for impl, v in (att.get("vs_fp32") or {}).items():
        A.measure("max abs difference vs fp32 attention", v, "abs", impl=impl)

    # 2. FP8 activation scale granularity (emulated numerics, one-config GEMM)
    fp8 = L.get("fp8_scales") or {}
    for key, impl, expect in (("per_tensor_dynamic", "fp8 scale per tensor, dynamic (batch amax)", "fail"),
                              ("per_token", "fp8 scale per token", "pass"),
                              ("per_block_1x128", "fp8 scale per 1x128 block", "pass")):
        r = fp8.get(key)
        s = parse_summary(dig(r, "summary")) or {}
        A.check("a row alone vs inside batches, three positions, three fillers", dig(r, "passed"),
                level="L2", impl=impl, expect=expect, detail={"comparisons": dig(r, "checked")},
                signature={"failing_sizes": s.get("sizes"), "failing_fillers": dig(r, "fillers_failing")})

    # 3. linear attention: chunked vs recurrent, two exact forms of the same math
    lin = L.get("linear_attention")
    s = parse_summary(dig(lin, "summary")) or {}
    A.check("chunked (16, 64) and recurrent forms give the same bits", dig(lin, "passed"), level="L3",
            impl="linear attention, torch ops, fp32", expect="fail",
            detail={"note": "an algorithmic difference, not a kernel defect"},
            signature={"failing_modes": s.get("modes")})
    for impl, v in (dig(lin, "max_abs_vs_fp64") or {}).items():
        A.measure("linear attention max abs error vs fp64", v, "abs", impl=impl)
