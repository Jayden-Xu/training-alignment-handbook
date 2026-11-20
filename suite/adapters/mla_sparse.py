"""examples/mla_sparse/mla_sparse.py -> bi-results-v1. Claims: handbook/3-kernels/mla-and-sparse.md."""
from . import dig

BASE, CORE, SDPA, TNORM, ALLBI = ("cublas+torchnorm+sdpa (baseline)", "cublas+torchnorm+bi_core",
                                  "bi_gemm+bi_norm+sdpa", "bi_gemm+torchnorm+bi_core", "all_bi")
# piece -> expect for (torch.mm / bmm / torch-op norm, one-config / grouped GEMM / Triton norm)
PIECES = {
    "W_q [2048->3072]": ("fail", "pass"),
    "W_dkv [2048->576]": ("fail", "pass"),
    "W_kvb decompress [512->4096]": ("fail", "pass"),
    "absorb q: per-head [128->512]": ("info", "pass"),
    "absorb o: per-head [512->128]": ("fail", "pass"),
    "W_o [2048->2048]": ("fail", "pass"),
    "rmsnorm [512]": ("info", "pass"),
    "DSA fused projection [1024->1664]": ("fail", "pass"),
    "DSA indexer logits fp32 [M x 4 heads x 64] . [2048 keys]": ("fail", None),
}
PIECE_IMPL = {"cublas": "torch.mm / bmm / torch-op norm", "bi": "one-config or grouped GEMM / Triton norm"}
# implementation -> expect for (explicit, absorbed)
MLA_L3 = {BASE: ("fail", "fail"), CORE: ("fail", "fail"), SDPA: ("fail", "info"), TNORM: ("pass", "pass"),
          ALLBI: ("pass", "pass")}
# implementation -> expect for (explicit prefill, explicit decode, absorbed prefill, absorbed decode)
MLA_L2 = {BASE: ("fail",) * 4, CORE: ("fail",) * 4, SDPA: ("info", "fail", "info", "info"),
          TNORM: ("pass",) * 4, ALLBI: ("pass",) * 4}
# sparse pipeline -> expect for (L3, L2 prefill, L2 decode step, L2 whole decode)
DSA = {
    "cublas+torch_indexer+topk+torch_attn (baseline)": ("fail", "info", "fail", "fail"),
    "cublas+torch_indexer+stable+bi_attn": ("fail", "info", "fail", "fail"),
    "cublas+bi_indexer+stable+bi_attn": ("fail", "info", "fail", "fail"),
    "bi_gemm+torch_indexer+topk+torch_attn": ("info",) * 4,
    "bi_indexer+topk+torch_attn": ("info",) * 4,
    "bi_indexer+stable+torch_attn": ("info",) * 4,
    "all_bi (bi_indexer+stable+bi_attn)": ("pass",) * 4,
}
DSA_TESTS = (("L3", "L3", "prefill, per-token decode and chunked prefill: the same selected sets and outputs"),
             ("L2_prefill", "L2", "a sequence alone vs among others, prefill: selected sets and outputs"),
             ("L2_decode", "L2", "one decode step alone vs among others: selected set and output"),
             ("L2_decode_sequence", "L2", "a whole decode alone vs in a batch at every step: selected sets and outputs"))


def convert(L, A):
    components(L.get("components"), A)
    mla(L.get("mla"), A)
    ties(L.get("topk_ties"), A)
    sparse(L.get("sparse"), A)
    timing(L.get("timing"), A)


def components(c, A):
    for piece, expects in PIECES.items():
        for which, expect in zip(("cublas", "bi"), expects):
            if expect is None:
                continue
            r = dig(c, f"{piece} | {which}")
            A.check(f"{piece}: a row alone vs inside batches", dig(r, "passed"), level="L2",
                    impl=PIECE_IMPL[which], expect=expect, detail={"comparisons": dig(r, "checked")},
                    signature={"failing_sizes": dig(r, "failing_sizes"),
                               "max_mismatched_elements": dig(r, "max_mismatched_elements")})
    A.check("the grouped GEMM equals a loop of mm_one_config over heads",
            dig(c, "grouped_gemm_equals_one_config_loop"), level="other", impl="grouped one-config GEMM")


def mla(m, A):
    for impl, expect in ((ALLBI, "fail"), (BASE, "info")):
        r = dig(m, "absorbed_vs_explicit", impl)
        n = dig(r, "layer_out_mismatch")
        A.check("the absorbed and explicit forms give the same layer output", None if n is None else n == 0,
                level="L3", impl=f"MLA {impl}", expect=expect,
                detail={"note": "an algorithmic difference: two forms of the same math"},
                signature={"mismatched_elements": n, "of": dig(r, "of_layer")})
        for form in ("explicit", "absorbed"):
            A.measure(f"MLA {form} form, layer output relative Frobenius error vs fp64",
                      dig(r, f"{form}_vs_fp64", "layer", "rel_fro"), "rel", impl=impl)
    for impl in MLA_L3:
        for i, form in enumerate(("explicit", "absorbed")):
            r = dig(m, "L3_modes", f"{form} | {impl}")
            A.check("prefill, per-token decode and chunked prefill give the same bits", dig(r, "passed"),
                    level="L3", impl=f"MLA {form} form, {impl}", expect=MLA_L3[impl][i],
                    detail={"comparisons": dig(r, "checked")},
                    signature={"mismatched_elements_vs_prefill": dig(r, "failures"), "of": dig(r, "of")})
            r = dig(m, "L2_batch", f"{form} | {impl}")
            for j, (mode, what) in enumerate((("prefill", "a sequence alone vs among others, prefill"),
                                              ("decode", "one decode step alone vs among others"))):
                x = dig(r, mode)
                A.check(what, dig(x, "passed"), level="L2", impl=f"MLA {form} form, {impl}",
                        expect=MLA_L2[impl][2 * i + j], detail={"comparisons": dig(x, "checked")},
                        signature={"failures": dig(x, "failures")})
    for impl, expect in ((ALLBI, "pass"), (BASE, "info")):
        for form in ("explicit", "absorbed"):
            r = dig(m, "L1_repeat", f"{form} | {impl}")
            A.check("the same batch, 5 runs", dig(r, "passed"), level="L1", impl=f"MLA {form} form, {impl}",
                    expect=expect)
    for nl in (1, 2):
        layers = f"{nl} layer{'s' if nl > 1 else ''}"
        for impl in (ALLBI, BASE):
            mm = dig(m, "mixed_forms", f"{impl} | {nl} layer(s)", "mismatched_elements")

            def n(eng_pre, eng_dec, trainer):
                return dig(mm, f"engine {eng_pre} prefill + {eng_dec} decode vs trainer {trainer} prefill")
            for form in ("explicit", "absorbed"):                   # one form everywhere
                v = n(form, form, form)
                A.check(f"engine (prompt prefill, then decode) equals a trainer prefill in the same form, {layers}",
                        None if v is None else v == 0, level="L3", impl=f"MLA {form} form everywhere, {impl}",
                        expect="pass" if impl == ALLBI else "fail", signature={"mismatched_elements": v})
            if impl != ALLBI:
                continue
            for trainer in ("explicit", "absorbed"):                # explicit prefill + absorbed decode
                v = n("explicit", "absorbed", trainer)
                # with one layer the latent cache does not depend on the form, so an absorbed trainer matches
                expect = "info" if (nl == 1 and trainer == "absorbed") else "fail"
                A.check(f"engine with explicit prefill and absorbed decode equals a trainer prefill, {layers}",
                        None if v is None else v == 0, level="L3",
                        impl=f"MLA mixed forms vs {trainer} trainer, {impl}", expect=expect,
                        signature={"mismatched_elements": v})


def ties(t, A):
    if t is None:
        return
    for key, what in (("failing_same_set_as_alone", "constructed ties: the selected set is the one of the row alone"),
                      ("failing_same_order_as_alone", "constructed ties: the returned order is the one of the row alone"),
                      ("failing_lowest_index_set", "constructed ties: the selected set is the lowest-index one"),
                      ("failing_sorted=False same_set", "constructed ties, sorted=False: the same set")):
        bad = t.get(key)
        A.check(what, None if bad is None else not bad, level="L2", impl="torch.topk", expect="info",
                detail={"trials": t.get("n_trials")}, signature={"failing_trials": bad})
    rows = t.get("rows") or {}
    A.check("on the constructed tie rows, the composite-key selection equals the stable-sort selection",
            all(r.get("stable_key_equals_stable_sort") for r in rows.values()) if rows else None,
            level="other", impl="stable top-k")


def sparse(s, A):
    v = dig(s, "stable_key_vs_stable_sort_rows_differ")
    A.check("on the pipeline's scores, the composite-key selection equals the stable-sort selection",
            None if v is None else v == 0, level="other", impl="stable top-k")
    v = dig(s, "torch_topk_vs_stable_rows_differ")
    if v is not None:
        A.check("on the pipeline's scores, torch.topk selects the (score desc, index asc) set", v == 0,
                level="other", impl="torch.topk", expect="info",
                detail={"positions_with_a_tie_at_the_boundary": dig(s, "natural_boundary_ties", "positions")})
    for impl, expects in DSA.items():
        for (key, level, what), expect in zip(DSA_TESTS, expects):
            r = dig(s, impl, "summary", key)
            A.check(what, None if r is None else r["tests_failing"] == 0, level=level, impl=f"sparse {impl}",
                    expect=expect, detail={"tests": dig(r, "tests")},
                    signature=None if r is None else {k: r[k] for k in (
                        "tests_failing", "max_positions_with_changed_set", "max_flipped_selections",
                        "max_out_mismatch")})
    for impl, expect in (("all_bi (bi_indexer+stable+bi_attn)", "pass"),
                         ("cublas+torch_indexer+topk+torch_attn (baseline)", "info")):
        A.check("the same batch, 5 runs", dig(s, "L1_repeat", impl, "passed"), level="L1",
                impl=f"sparse {impl}", expect=expect)


def timing(t, A):
    if not t:
        return

    def both(name, v, impl):
        if isinstance(v, dict):
            A.measure(name + ", eager", v.get("eager"), "ms", impl=impl)
            A.measure(name + ", CUDA graph", v.get("graph"), "ms", impl=impl)
    p = t.get("mla_prefill_ms") or {}
    for k, v in p.items():
        both(f"MLA prefill step B={p.get('B')} T={p.get('T')}", v, k)
    d = t.get("mla_decode_ms") or {}
    for k, v in d.items():
        if isinstance(v, dict):
            b, rest = k.split(" | ", 1)
            both(f"MLA decode step {b} context={d.get('context')}", v, rest)
    sp = t.get("sparse_prefill_ms") or {}
    for k, v in (sp.get("step") or {}).items():
        both(f"sparse prefill step B={sp.get('B')} T={sp.get('T')}", v, k)
    for k, v in (sp.get("pieces") or {}).items():
        A.measure(f"sparse piece at the prefill shape B={sp.get('B')} T={sp.get('T')}", v, "ms", impl=k)
    sd = t.get("sparse_decode_ms") or {}
    for k, v in sd.items():
        both(f"sparse decode step B={sd.get('B')} context={sd.get('context')}", v, k)
