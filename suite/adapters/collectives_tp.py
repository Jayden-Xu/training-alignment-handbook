"""examples/collectives_tp/collectives_tp.py --merge (one node or two) -> bi-results-v1.
Claims: handbook/3-kernels/collectives.md."""
from . import dig

SWEEP = {"rows": "a 4x4096 slice", "small": "a 512-element slice"}


def env_source(L):
    for r in (L.get("runs") or {}).values():
        return {**r, "gpu_count": r.get("world", 1)}
    return {}


def tp_expect(scheme, t, tag):
    """Row-parallel passes with two ranks (a two-term sum commutes) and under a pinned Tree; reduce-
    scatter (sequence parallelism) follows the owning rank; fallbacks and logical TP are the fixes."""
    if scheme == "row_parallel":
        return "info" if t == 2 else "pass" if tag.startswith("tree") else "fail"
    if scheme == "row_sp":
        return "fail"
    return "pass"


def convert(L, A):
    summary, runs = L.get("summary") or {}, L.get("runs") or {}
    if not summary:
        A.check("all-reduce bf16, a 4x4096 slice alone vs inside messages of many sizes", None, level="L2",
                impl="NCCL all-reduce [tree]")
    for key in sorted(summary):
        collectives(key.split("/")[0], summary[key], A)
    for k, n in sorted((L.get("fallback_distinct_results_across_all_runs") or {}).items()):
        dtype, sweep = k.split("/")[:2]
        form = "two-level fallback" if k.endswith("/hier") else "fallback (all-gather or all-to-all)"
        A.check(f"{dtype}, {SWEEP[sweep]}: one result across every NCCL setting and message size", n == 1,
                level="L2", impl=f"{form}: fp32 sum in rank order", signature={"distinct_results": n})
    for key in sorted(runs):
        if key.endswith("/tp"):
            tp(key.split("/")[0], runs[key], A)
    for key in sorted(runs):
        tag, part = key.split("/")
        t = dig(runs[key], "result", "timing") if part == "tp" else None
        for name, row in (dig(t, "collectives") or {}).items():
            for op, ms in (row.get("ms") or {}).items():
                A.measure(f"collective, {row.get('dtype')} {row.get('bytes')} bytes, {row.get('world')} ranks",
                          ms, "ms", impl=f"{op} [{tag}]")
        for name, ms in (dig(t, "tp") or {}).items():
            A.measure(f"TP linear {name}", ms, "ms", impl=f"[{tag}]")


def collectives(tag, s, A):
    tree = tag.startswith("tree")
    for dtype in ("bf16", "fp32"):
        for sweep, what in SWEEP.items():
            r = s.get(f"allreduce/{dtype}/{sweep}")
            if r:
                A.check(f"all-reduce {dtype}, {what} alone vs inside messages of many sizes", r["failing"] == 0,
                        level="L2", impl=f"NCCL all-reduce [{tag}]", expect="pass" if tree else "fail",
                        detail={"calls": r["calls"], "bytes": r.get("bytes"), "nccl_choices": r.get("nccl_choices")},
                        signature={"failing_calls": r["failing"], "distinct_results": r["distinct_results"]})
                A.check(f"all-reduce {dtype}, {what}: every result element equals one sequential or tree order",
                        r["unexplained_by_ring_orders"] == 0, level="other", impl=f"NCCL all-reduce [{tag}]",
                        expect="info")
            r = s.get(f"fallback/{dtype}/{sweep}")
            if r:
                A.check(f"{dtype}, {what} alone vs inside messages of many sizes; all-gather and all-to-all "
                        "forms agree and equal an fp32 rank-order sum",
                        r["failing"] == 0 and r["gather_equals_a2a"] and r["equals_fp32_rank_order_sum"],
                        level="L2", impl=f"fallback: fp32 sum in rank order [{tag}]", detail={"calls": r["calls"]},
                        signature={"failing_calls": r["failing"]})
                if "hier_failing" in r:
                    A.check(f"{dtype}, {what} alone vs inside messages of many sizes", r["hier_failing"] == 0,
                            level="L2", impl=f"two-level fallback: node sums, then node 0 + node 1 [{tag}]",
                            signature={"failing_calls": r["hier_failing"]})
        r = s.get(f"reduce_scatter/{dtype}")
        if r:
            impl = f"NCCL reduce-scatter [{tag}]"
            A.check(f"reduce-scatter {dtype}, a slice kept by one rank alone vs inside larger chunks",
                    r["failing"] == 0, level="L2", impl=impl, expect="info", detail={"calls": r["calls"]})
            A.check(f"reduce-scatter {dtype}: the same bits whichever rank keeps the slice",
                    r["owners_differ_from_owner0"] == 0, level="L2", impl=impl, expect="fail",
                    signature={"owners_that_differ_from_owner_0": r["owners_differ_from_owner0"]})
        r = s.get(f"fallback_rs/{dtype}")
        if r:
            A.check(f"reduce-scatter {dtype}: alone vs inside larger chunks, and the same bits whichever rank "
                    "keeps the slice", r["failing"] == 0 and r["owners_differ_from_owner0"] == 0, level="L2",
                    impl=f"fallback reduce-scatter: all-to-all, fp32 sum in rank order [{tag}]")
    rep = s.get("repeat") or {}
    if rep:
        A.check("5 identical all-reduce and reduce-scatter calls, every rank",
                all(v.get("passed") and not v.get("failures_all_ranks") for v in rep.values()), level="L1",
                impl=f"NCCL collectives [{tag}]", expect="info", detail={"cases": len(rep)})


def tp(tag, run, A):
    res = run.get("result") or {}
    world = run.get("world")
    f32 = res.get("f32_variant_rounds_to_mm_one_config")
    if f32 is not None:
        A.check("the fp32-output GEMM, rounded to bf16, equals mm_one_config", all(f32.values()), level="other",
                impl=f"mm_one_config_f32 [{tag}]", detail={"M": sorted(f32, key=int)})
    c0 = res.get("chunk0_of_8_M4096")
    if c0:
        A.check("splitting a chunk's N columns over two ranks keeps its fp32 partial", c0["split_N_in_two_equal"],
                level="L5", impl=f"mm_one_config_f32 [{tag}]")
        A.check("splitting a chunk's K range over two ranks keeps its fp32 partial",
                c0["split_K_in_two_mismatched"] == 0, level="L5", impl=f"mm_one_config_f32 [{tag}]", expect="fail",
                signature={"mismatched_elements": c0["split_K_in_two_mismatched"]})
    for key, r in (res.get("l2") or {}).items():
        scheme, t = key.split("/tp")
        A.check("a row alone vs inside batches", r["passed"], level="L2", impl=f"{scheme} TP {t} [{tag}]",
                expect=tp_expect(scheme, int(t), tag), detail={"comparisons": r["checked"]},
                signature={"failing_sizes": r["failing_sizes"], "failed": r["failed"]})
    for name, r in (res.get("l1") or {}).items():
        scheme, t = name.split(" ")[0], name.split("tp=")[1].split(" ")[0]
        A.check("the same batch, 5 runs", r["passed"], level="L1", impl=f"{scheme} TP {t} [{tag}]",
                expect="pass" if scheme.startswith("lt") else "info")
    by_scheme = {}
    for M, table in (res.get("l5") or {}).items():
        for key, r in table.items():
            if key == r["vs"]:
                continue                                      # the reference itself
            by_scheme.setdefault(key.split("/")[0], {})[f"M={M} {key.split('/')[1]}"] = r["mismatched"]
    for scheme, counts in by_scheme.items():
        if scheme in ("row_parallel", "row_fallback", "row_fbhier"):
            name, exp = "the same bits as the unsharded GEMM at every TP degree", "fail"
        elif scheme == "col_parallel":
            name, exp = "the same bits as the unsharded GEMM at every TP degree", "pass"
        elif scheme.endswith("_splitk"):
            name, exp = "the same bits as the 8-chunk scheme at TP 1 (chunks split along K)", "fail"
        elif scheme.endswith("_2d"):
            name, exp = "the same bits as the 8-chunk scheme at TP 1 (chunks split along N)", "pass"
        else:
            name, exp = "the same bits at every TP degree (logical TP)", "pass"
        A.check(name, all(n == 0 for n in counts.values()), level="L5", impl=f"{scheme} [{tag}]", expect=exp,
                detail={"world": world, "cases": sorted(counts)}, signature={"mismatched_elements": counts})
    for key, e in (res.get("accuracy") or {}).items():
        A.measure("relative Frobenius error vs fp64, M=256", e.get("rel_fro"), "rel", impl=f"{key} [{tag}]")
