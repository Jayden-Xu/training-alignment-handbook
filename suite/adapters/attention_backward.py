"""examples/attention_backward/attn_bwd.py -> bi-results-v1. Claims: handbook/3-kernels/attention.md."""
from . import dig

# (L1, L2 batch, L2 padded) per implementation. sdpa_efficient's batch failure is a handful of dQ
# elements at batch 8 and 16 only, and the strict-deterministic memory-efficient backend is a stock
# configuration that passed at the shapes tested: both are recorded, not judged.
EXPECT = {
    "fixed_blocks": ("pass", "pass", "pass"),
    "sdpa_flash": ("fail", "fail", "fail"),
    "sdpa_flash_det": ("info", "fail", "fail"),
    "sdpa_efficient": ("fail", "info", "info"),
    "sdpa_efficient_det": ("info", "info", "info"),
}
NONDETERMINISTIC = ("sdpa_flash", "sdpa_efficient")
CASES = (("repeat", "L1", "the same batch, 5 runs: O, dQ, dK, dV"),
         ("batch", "L2", "a sequence alone vs among others: O, dQ, dK, dV"),
         ("padded", "L2", "a sequence alone vs right-padded in a batch, zero gradient on the padding"))


def convert(L, A):
    fwd = L.get("forward")
    for key, what in (("with_lse_equals_plain", "the forward that stores log-sum-exp equals the plain forward"),
                      ("train_equals_plain", "the training forward (autograd Function) equals the plain forward")):
        ok = None if not fwd else all(v.get(key) is True for v in fwd.values())
        A.check(what, ok, level="other", impl="fixed_blocks", expect="pass",
                detail={"lengths": sorted(fwd) if fwd else None})
    for i, (case, level, name) in enumerate(CASES):
        for impl, exp in EXPECT.items():
            r = dig(L, case, impl)
            if isinstance(r, dict) and "error" in r:
                A.error(f"{case} {impl}: {r['error']}")
                r = None
            sig = None
            if impl not in NONDETERMINISTIC:      # an atomic backward fails in different places each run
                fails = dig(r, "failures", default=[])
                sig = {"failing_tensors_first_20_failures": sorted({f.get("tensor") for f in fails})}
                if dig(r, "failing") is not None:
                    sig["failing_cases"] = r["failing"]
            A.check(name, dig(r, "passed"), level=level, impl=impl, expect=exp[i],
                    detail={"comparisons": dig(r, "checked")}, signature=sig)
    for impl, acc in (L.get("accuracy") or {}).items():
        for t, e in acc.items():
            A.measure(f"relative Frobenius error vs fp64, {t}", e.get("rel_fro"), "rel", impl=impl)
    for impl, t in (L.get("timing") or {}).items():
        A.measure("forward B=4 T=2048", t.get("fwd_ms"), "ms", impl=impl)
        A.measure("forward+backward B=4 T=2048", t.get("fwd_bwd_ms"), "ms", impl=impl)
