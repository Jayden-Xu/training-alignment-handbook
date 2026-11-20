"""examples/attention_serving/serving_cases.py -> bi-results-v1.
Claims: handbook/3-kernels/attention.md."""
from . import dig, parse_summary

OPS_FAIL = ("rmsnorm", "add + rmsnorm (residual feeding the norm)", "silu(gate) * up")
OPS_PASS = ("mm_one_config alone", "attention_fixed_blocks alone")


def _zero(values):
    values = list(values)
    return None if not values or any(v is None for v in values) else all(v == 0 for v in values)


def convert(L, A):
    paged(L.get("paged"), L.get("paged_timing"), A)
    varlen(L.get("varlen"), L.get("varlen_timing"), A)
    eager(L.get("eager"), A)
    compiled(L.get("compile"), L.get("compile_diag"), A)
    graphs(L.get("graphs"), A)
    for b, row in (L.get("wrappers_timing") or {}).items():
        if b.startswith("B=") and isinstance(row, dict):
            for name, ms in row.items():
                A.measure(f"block call, {b[2:]} sequence(s) of 100 tokens", ms, "ms", impl=name)


def paged(p, timing, A):
    for key in sorted(dig(p, "L3", default={}), key=lambda k: int(k.split("=")[1])):
        impl = f"attention_paged, page size {key.split('=')[1]}"
        r = p["L3"][key]
        A.check("paged prefill, per-token decode and chunked prefill equal the contiguous prefill",
                dig(r, "passed"), level="L3", impl=impl, detail={"comparisons": dig(r, "checked")},
                signature={"failing_modes": (parse_summary(dig(r, "summary")) or {}).get("modes")})
        r = dig(p, "L2_mixed_batches", key)
        A.check("mixed batches (prefill, decode and chunk rows of different lengths) equal each "
                "sequence's contiguous prefill", dig(r, "passed"), level="L2", impl=impl,
                detail={"comparisons": dig(r, "checked")})
    if p is None:
        A.check("paged prefill, per-token decode and chunked prefill equal the contiguous prefill", None,
                level="L3", impl="attention_paged")
    c = dig(p, "contrast_BN_equals_P")
    if c is not None:
        impl = "paged attention, compute block = page size (contrast)"
        pairs = c.get("pairwise_mismatches") or {}
        A.check("the same bits at every page size", _zero(pairs.values()), level="L3", impl=impl,
                expect="fail", signature={"mismatched_elements_between_page_sizes": pairs})
        A.check("decode equals prefill at each page size (self-consistent)",
                _zero(v.get("decode_vs_prefill_same_BN") for k, v in c.items() if k.startswith("BN=P=")),
                level="L3", impl=impl, expect="info")
    if timing is not None:
        counts = {**{f"prefill {k}": v for k, v in (timing.get("prefill_B4_T2048_mismatch_vs_contiguous") or {}).items()},
                  **{f"decode {k}": v for k, v in (timing.get("decode_B16_KV2048_mismatch_vs_contiguous") or {}).items()}}
        A.check("at the timing shapes, every paged and varlen output equals the contiguous kernel",
                _zero(counts.values()), level="L3", impl="attention_paged, attention_varlen",
                detail={"outputs": len(counts)}, signature={"mismatched_elements": {k: v for k, v in counts.items() if v}})
        for group in ("prefill_B4_T2048_ms", "decode_B16_KV2048_ms"):
            for name, ms in (timing.get(group) or {}).items():
                A.measure(group[:-3].replace("_", " "), ms, "ms", impl=name)


def varlen(v, timing, A):
    fwd = dig(v, "forward_prefill")
    alone = [r for r in fwd or [] if "layout" in r]
    packed = [r for r in fwd or [] if "before" in r]
    A.check("a sequence alone equals the contiguous tools kernel", _zero(r["bi_mismatch"] for r in alone),
            level="other", impl="attention_varlen", detail={"probes": len(alone)})
    A.check("a sequence alone vs packed among neighbours at different offsets",
            _zero(r["bi_mismatch"] for r in packed), level="L2", impl="attention_varlen",
            detail={"layouts": len(packed)})
    ch = dig(v, "forward_chunk_in_pack")
    A.check("a chunk packed with decode, prefill and chunk rows of other sequences equals its contiguous "
            "prefill rows", _zero(r["bi_mismatch_vs_contiguous_prefill"] for r in ch or []), level="L3",
            impl="attention_varlen", detail={"layouts": len(ch or [])})
    naive = "attention_varlen, blocks aligned to the pack (contrast)"
    off = [r for r in packed if r.get("offset_mod_64")]
    on = [r for r in packed if not r.get("offset_mod_64")]
    if fwd is not None:
        A.check("a sequence alone vs packed at offsets that are not multiples of 64",
                _zero(r["naive_mismatch"] for r in off), level="L2", impl=naive, expect="fail",
                signature={"mismatched_elements_by_probe_len_and_offset":
                           {f"{r['probe_len']}@{r['offset']}": r["naive_mismatch"] for r in off}})
        A.check("a sequence alone vs packed at offsets that are multiples of 64",
                _zero(r["naive_mismatch"] for r in on), level="L2", impl=naive, expect="info")
    A.check("the forward that stores log-sum-exp equals the plain forward",
            _zero([dig(v, "lse_variant_mismatch_vs_plain")]), level="other", impl="attention_varlen")
    bwd = dig(v, "backward")
    names = ("o", "dq", "dk", "dv")
    alone = [r for r in bwd or [] if "layout" in r]
    packed = [r for r in bwd or [] if "before" in r]
    A.check("alone, O, dQ, dK, dV equal tools attention_fixed_blocks_train",
            _zero(r[n] for r in alone for n in names), level="other", impl="attention_varlen_train",
            detail={"probes": len(alone)})
    A.check("O, dQ, dK, dV of a sequence alone vs packed among neighbours",
            _zero(r[n] for r in packed for n in names), level="L2", impl="attention_varlen_train",
            detail={"layouts": len(packed)})
    A.check("the same pack, 5 runs: O, dQ, dK, dV", dig(v, "backward_repeat_L1", "passed"), level="L1",
            impl="attention_varlen_train", detail={"tokens": dig(v, "backward_repeat_L1", "tokens")})
    for group in ("fwd_B4_T2048_ms", "fwd_bwd_B4_T2048_ms"):
        for name, ms in (dig(timing, group) or {}).items():
            A.measure("varlen " + group[:-3].replace("_", " "), ms, "ms", impl=name)


def eager(e, A):
    name = "a sequence alone vs inside batches, three positions, three fillers"
    for variant, expect in (("bi", "pass"), ("bi_opaque_all", "pass"), ("cublas", "fail"), ("stock", "fail")):
        r = dig(e, variant)
        A.check(name, dig(r, "passed"), level="L2", impl=f"block {variant}, eager", expect=expect,
                detail={"comparisons": dig(r, "checked")}, signature={"failing_sizes": dig(r, "failing_sizes")})
    A.check("the block built from custom ops equals the direct block bitwise",
            _zero([dig(e, "opaque_all_vs_bi_B1")]), level="other", impl="block bi_opaque_all, eager")


def compiled(c, diag, A):
    if c is None:
        A.check("a sequence alone vs inside batches, under torch.compile", None, level="L2",
                impl="block bi, torch.compile")
    for key, r in (c or {}).items():
        if not isinstance(r, dict) or "variant" not in r:
            continue
        variant = r["variant"]
        impl = f"block {variant}, torch.compile dynamic={r.get('dynamic')}" + \
            (f" mode={r['mode']}" if r.get("mode") not in (None, "default") else "") + \
            ("" if r.get("fullgraph", True) else " fullgraph=False")
        if "error" in r:
            A.error(f"compile {key}: {r['error']}")
        A.check("a sequence alone vs inside batches, under torch.compile", r.get("L2_vs_compiled_B1"),
                level="L2", impl=impl, detail={"graphs_compiled": r.get("graphs_compiled")},
                signature={"failing_sizes": r.get("failing_sizes_vs_compiled_B1")})
        A.check("the compiled output equals eager at every batch size", r.get("equals_eager_all_sizes"),
                level="other", impl=impl, expect="pass" if variant == "bi_opaque_all" else "fail",
                signature={"mismatched_elements_at_batch_1": r.get("B1_compiled_vs_eager_mismatch")})
        A.measure("graphs compiled over the batch-size sweep", r.get("graphs_compiled"), "graphs", impl=impl)
    for op, r in (dig(diag, "per_op_compiled_vs_eager") or {}).items():
        if "error" in r:
            A.error(f"compile_diag {op}: {r['error']}")
            continue
        expect = "fail" if op in OPS_FAIL else "pass" if op in OPS_PASS else "info"
        A.check("compiled alone, the op equals eager", r["mismatch"] == 0, level="other",
                impl=f"torch.compile of {op}", expect=expect,
                signature={"mismatched_elements": r["mismatch"], "of": r.get("of")})


def graphs(g, A):
    for variant, expect in (("bi", "pass"), ("cublas", "fail"), ("stock", "fail")):
        r = dig(g, variant)
        if g is None and variant != "bi":
            continue
        A.check("padded to any CUDA-graph bucket, the probe gives the bits of eager batch 1",
                dig(r, "passed"), level="L2", impl=f"block {variant}, CUDA graphs", expect=expect,
                signature={"buckets_grouped_by_output": dig(r, "bucket_groups_probe0")})
        same = dig(r, "graph_vs_eager_same_shape_mismatch")
        if same is not None:
            A.check("a graph replay equals eager on the same padded batch", _zero(same.values()),
                    level="other", impl=f"block {variant}, CUDA graphs",
                    expect="pass" if variant == "bi" else "info")
