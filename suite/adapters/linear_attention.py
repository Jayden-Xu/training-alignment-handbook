"""examples/linear_attention/gdn_bi.py -> bi-results-v1. Claims: handbook/3-kernels/linear-attention.md."""
from . import dig, parse_summary

IMPLS = ("recurrent", "chunked_bf16", "chunked_fp32")
DEFAULT_CFG = "BV=8,warps=1,prefetch=1"
NO_PREFETCH = "BV=8,warps=1,prefetch=0"


def _section(L, key, A):
    s = L.get(key)
    if isinstance(s, dict) and set(s) == {"error"}:
        return None
    return s


def _all_equal(d, prefix):
    rows = [v for k, v in (d or {}).items() if k.startswith(prefix)]
    return None if not rows else all(r.get("equal") for r in rows)


def _counts(d, prefix):
    return {k: dig(v, "o", "mismatched") for k, v in (d or {}).items() if k.startswith(prefix)}


def convert(L, A):
    rep, bat, modes = _section(L, "repeat", A), _section(L, "batch", A), _section(L, "modes", A)
    for impl in IMPLS:
        A.check("the same batch, 5 runs: output and final state", dig(rep, impl, "passed"), level="L1", impl=impl)
        A.check("a sequence alone vs among others, three positions, three fillers: output and final state",
                dig(bat, impl, "passed"), level="L2", impl=impl,
                detail={"comparisons": dig(bat, impl, "checked"), "sizes": dig(bat, impl, "sizes")})
    A.check("prefill, per-token decode and chunked prefill, each continuing from the saved state",
            _all_equal(dig(modes, "recurrent"), ""), level="L3", impl="recurrent",
            detail={"modes": sorted(dig(modes, "recurrent", default={}))})
    for prec in ("bf16", "fp32"):
        d = dig(modes, f"chunked_{prec}")
        A.check("calls at multiples of the chunk size equal the full prefill",
                _all_equal(d, "calls-at-multiples-of-64"), level="L3", impl=f"chunked_{prec}")
        A.check("aligned decode and chunked prefill of any size equal the full prefill",
                _all_equal(d, "aligned-"), level="L3", impl=f"chunked_{prec} with AlignedChunked",
                detail={"modes": sorted(k for k in (d or {}) if k.startswith("aligned-"))})
        A.check("decode and chunked prefill equal the full prefill", _all_equal(d, "call-relative-"),
                level="L3", impl=f"chunked_{prec}, call-relative chunks (contrast)", expect="fail",
                signature={"mismatched_output_elements": _counts(d, "call-relative-")})

    cfgs = _section(L, "configs", A) or {}
    if not cfgs:
        A.check("the same batch, 5 runs", None, level="L1", impl=f"recurrent {DEFAULT_CFG}")
    for cfg, r in cfgs.items():
        impl = f"recurrent {cfg}"
        A.check("the same batch, 5 runs", dig(parse_summary(r.get("L1")), "passed"), level="L1", impl=impl)
        A.check("a sequence alone vs among others", dig(parse_summary(r.get("L2")), "passed"), level="L2", impl=impl)
        A.check("prefill, per-token decode and chunked prefill", r.get("L3_pass"), level="L3", impl=impl)
        if cfg == DEFAULT_CFG:
            continue
        eq = dig(r, "vs_default", "equal")
        A.check("the configuration gives the bits of the default configuration", eq, level="other", impl=impl,
                expect="pass" if cfg == NO_PREFETCH else "fail",
                detail={"note": "tile width and warp count are part of the reduction tree; prefetch is not"},
                signature={"mismatched_output_elements": dig(r, "vs_default", "o", "mismatched")})

    cross = _section(L, "cross", A)
    for pair, expect in (("recurrent vs chunked_fp32", "fail"), ("recurrent vs chunked_bf16", "fail"),
                         ("chunked_fp32 vs chunked_bf16", "info")):
        r = dig(cross, "differ", pair)
        A.check("the two forms give the same output and final state", dig(r, "equal"), level="L3", impl=pair,
                expect=expect, detail={"note": "an algorithmic or precision difference, not a kernel defect"},
                signature={"mismatched_output_elements": dig(r, "o", "mismatched"),
                           "after_bf16_cast": dig(r, "o_bf16_mismatched")})
    for impl, e in (dig(cross, "error_vs_fp64") or {}).items():
        A.measure("output relative Frobenius error vs fp64", dig(e, "o", "rel_fro"), "rel", impl=impl)
        A.measure("final state relative Frobenius error vs fp64", dig(e, "state", "rel_fro"), "rel", impl=impl)

    t = _section(L, "timing", A) or {}
    for shape, row in (t.get("forward_ms") or {}).items():
        for name, v in row.items():
            if "/" not in name:                                   # ratios and the 3-phase split are derived
                A.measure(f"forward B x T = {shape}", v, "ms", impl=name)
    for b, row in (t.get("decode_step_ms") or {}).items():
        for name, v in row.items():
            A.measure(f"decode step {b}, CUDA graph", dig(v, "graph_ms"), "ms", impl=name)
            A.measure(f"decode step {b}, eager", dig(v, "eager_ms"), "ms", impl=name)
