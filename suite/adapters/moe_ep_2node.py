"""examples/moe_ep/moe_ep_2node.py (16 ranks on two nodes) -> bi-results-v1.
Claims: handbook/3-kernels/moe.md, section "Two nodes (EP 16)"."""
from . import dig
from .moe_ep import NONDETERMINISTIC, VARIANTS, expect, label


def env_source(L):
    m = L.get("meta") or {}
    return {**m, "gpu_count": m.get("ranks", 16)}


def convert(L, A):
    meta = L.get("meta") or {}
    A.check("ranks 0-7 run on one node and ranks 8-15 on another", meta.get("ranks_0_7_one_node_8_15_another"),
            level="other", impl="two-node launch", detail={"ranks": meta.get("ranks")})
    if not L.get("L5"):
        A.check("the same bits as EP 1 on rank 0 in every layout, group and sharding", None, level="L5",
                impl="fixed_slot")
    for cfg, summ in (L.get("L5") or {}).items():
        lab, K = label(cfg)
        layouts = {k: v for k, v in summ.items() if isinstance(v, dict) and "variants" in v}
        for v in VARIANTS:
            worst = {lay: max(n for counts in d["variants"][v]["el"].values() for n in counts)
                     for lay, d in layouts.items() if v in d["variants"]}
            if not worst:
                continue
            A.check(f"{lab}: the same bits as EP 1 on rank 0 in every layout (EP 1 to 16, inside a node or "
                    "across two), group and sharding", all(n == 0 for n in worst.values()), level="L5",
                    impl=v, expect=expect(v, "L5", K), detail={"layouts": sorted(worst)},
                    signature=None if v in NONDETERMINISTIC else {"max_mismatched_elements_per_layout": worst})
        ids = [d["variants"]["fixed_slot"]["ids_equal"] for d in layouts.values() if "fixed_slot" in d["variants"]]
        if ids:
            A.check(f"{lab}: router ids equal in every layout, group and sharding", all(ids), level="L5",
                    impl="router (one-config GEMM, softmax, stable top-k)")
        link = summ.get("vs_single_node_ep1_dump")
        for v in VARIANTS if isinstance(link, dict) else ():
            if v in link:
                A.check(f"{lab}: EP 1 on rank 0 equals the EP 1 dump of the one-node run", link[v] == 0,
                        level="L1", impl=v, expect="pass" if expect(v, "L1", K) == "pass" else "info",
                        signature={"mismatched_elements": link[v]})
    for cfg, row in (L.get("L1_ep16") or {}).items():
        lab, K = label(cfg)
        for v in VARIANTS:
            if v in row:
                A.check(f"{lab}: the same global batch, 5 runs, at EP 16", row[v].get("passed"), level="L1",
                        impl=v, expect=expect(v, "L1", K))
    for cfg, row in (L.get("L2_ep16") or {}).items():
        lab, K = label(cfg)
        for v in VARIANTS:
            if v in row:
                A.check(f"{lab}: a probe group alone vs with co-batched tokens on its rank, on other ranks or "
                        "both, at EP 16", row[v].get("passed"), level="L2", impl=v, expect=expect(v, "L2", K),
                        detail={"comparisons": row[v].get("checked"), "tokens_per_rank": row.get("tokens_per_rank")},
                        signature=None if v in NONDETERMINISTIC else {"failed": row[v].get("failed")})
    for cfg, lays in (L.get("timing_ms") or {}).items():
        lab, _ = label(cfg)
        for lay, by_tokens in (lays or {}).items():
            for tokens, row in by_tokens.items():
                for v, ms in row.items():
                    A.measure(f"{lab}: layer forward, {tokens} tokens per rank, {lay}", ms, "ms", impl=v)
