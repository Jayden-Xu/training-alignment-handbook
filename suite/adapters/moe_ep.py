"""examples/moe_ep/moe_ep.py --compare (one node) -> bi-results-v1.
Claims: handbook/3-kernels/moe.md."""
import re

from . import dig, parse_summary

VARIANTS = ("arrival_atomic", "arrival_det", "fixed_slot", "fixed_slot_torch", "rank_partial", "capacity")
NONDETERMINISTIC = ("arrival_atomic",)
BLOCKS = (("mm_one_config router H->E", "router GEMM", "mm_one_config", "pass"),
          ("cublas router H->E", "router GEMM", "torch.mm (cuBLAS)", "fail"),
          ("mm_one_config expert up H->I", "expert up GEMM", "mm_one_config", "pass"),
          ("cublas expert up H->I", "expert up GEMM", "torch.mm (cuBLAS)", "fail"),
          ("mm_one_config expert down I->H", "expert down GEMM", "mm_one_config", "pass"),
          ("cublas expert down I->H", "expert down GEMM", "torch.mm (cuBLAS)", "fail"),
          ("router weights", "router weights", "router (one-config GEMM, softmax, stable top-k)", "pass"),
          ("router ids", "router ids", "router (one-config GEMM, softmax, stable top-k)", "pass"))


def expect(variant, level, K):
    """fixed_slot is the recipe. The atomic combine and the per-rank pre-reduce only fail from
    top-3 (a two-term sum commutes); the atomic combine's L5 result is its L1 noise."""
    if variant in ("fixed_slot", "fixed_slot_torch"):
        return "pass"
    if variant == "arrival_atomic":
        return "fail" if K >= 3 and level in ("L1", "L2") else "info"
    if variant == "rank_partial":
        return "fail" if K >= 3 and level == "L5" else "info"
    if variant == "capacity":
        return "fail" if level in ("L2", "L5") else "info"
    return "info"                                             # arrival_det: passed, order is the library's


def label(cfg):
    m = re.fullmatch(r"E(\d+)K(\d+)(?:T(\d+))?", cfg)
    if not m:
        return cfg, 0
    E, K, T = m.groups()
    return f"E={E} top-{K}" + (f", {T} global tokens" if T else ""), int(K)


def env_source(L):
    for entry in L.values():
        for r in (dig(entry, "runs") or {}).values():
            return {**r, "gpu_count": max(x.get("world", 1) for x in entry["runs"].values())}
    return {}


def convert(L, A):
    for cfg in sorted(k for k, v in L.items() if isinstance(v, dict) and "runs" in v):
        entry = L[cfg]
        lab, K = label(cfg)
        runs = entry.get("runs") or {}
        g = dig(runs, "ep1", "gemm_checks")
        for key, what, impl, exp in BLOCKS if g else ():
            s = parse_summary(dig(g, key, "summary"))
            A.check(f"{lab}: {what}: a row alone vs inside batches (EP 1)", dig(s, "passed"), level="L2",
                    impl=impl, expect=exp, detail={"comparisons": dig(s, "checked")},
                    signature={"failing_sizes": dig(s, "sizes")})
        for v in VARIANTS:
            l1 = {d: r["L1"][v] for d, r in runs.items() if v in (r.get("L1") or {})}
            if l1:
                A.check(f"{lab}: the same global batch, 5 runs, at every EP degree",
                        all(x["passed"] for x in l1.values()), level="L1", impl=v, expect=expect(v, "L1", K),
                        detail={"runs": sorted(l1)})
            l2 = {d: r["L2"][v] for d, r in runs.items() if v in (r.get("L2") or {})}
            if l2:
                A.check(f"{lab}: a probe group alone vs with co-batched tokens on its rank, on other ranks or both",
                        all(x["passed"] for x in l2.values()), level="L2", impl=v, expect=expect(v, "L2", K),
                        detail={"runs": sorted(l2), "comparisons": sum(x["checked"] for x in l2.values())},
                        signature=None if v in NONDETERMINISTIC else
                        {"failed_per_run": {d: x.get("failed") for d, x in l2.items()}})
            l5 = {d: recs for d, recs in (dig(entry, "L5", v) or {}).items() if recs}
            if len(l5) > 1:                                    # needs a degree other than EP 1
                worst = {d: max(x["mismatched_elements"] for x in recs.values()) for d, recs in l5.items()}
                A.check(f"{lab}: the same bits as EP 1 at every EP degree, sharding and GPU placement",
                        all(n == 0 for n in worst.values()), level="L5", impl=v, expect=expect(v, "L5", K),
                        detail={"runs": sorted(l5), "shardings": sorted({s for recs in l5.values() for s in recs})},
                        signature=None if v in NONDETERMINISTIC else {"max_mismatched_elements_per_run": worst})
        ids = [x.get("ids_equal") for recs in (dig(entry, "L5", "fixed_slot") or {}).values() for x in recs.values()]
        if len(ids) > 4:
            A.check(f"{lab}: router ids equal at every EP degree and sharding", all(ids), level="L5",
                    impl="router (one-config GEMM, softmax, stable top-k)")
        for other, exp in (("fixed_slot_torch", "pass"), ("rank_partial", "info")):
            n = dig(entry, "cross_variant_ep1", f"fixed_slot vs {other}", "mismatched_elements")
            A.check(f"{lab}: at EP 1 the output equals the Triton fixed-slot sum", None if n is None else n == 0,
                    level="other", impl=other, expect=exp, signature={"mismatched_elements": n})
        for d, r in sorted(runs.items()):
            for tokens, row in (r.get("timing_ms") or {}).items():
                for v, ms in row.items():
                    A.measure(f"{lab}: layer forward, {tokens} tokens per rank, {d}", ms, "ms", impl=v)
    if not any(isinstance(v, dict) and "runs" in v for v in L.values()):
        A.check("the same bits as EP 1 at every EP degree, sharding and GPU placement", None, level="L5",
                impl="fixed_slot")
