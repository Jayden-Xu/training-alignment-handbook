"""Batch-invariant sampling: where stock sampling depends on the batch, and a reference
sampler (tools/bi_sampling.py) that does not.

A request's logit bits are identical in every comparison; only the batch around it, the
schedule, or the process changes.

Sections (--sections a,b,...; default all, --quick runs unit,chisq,l2,engine,nucleus at reduced sizes)
  rng      stock random streams: one generator for the batch, one generator per request,
           what n > 1 and the fill size do to a per-request stream, CPU vs CUDA streams
  ops      stock row-wise ops at vocab 32,000 and 151,936: temperature divide, softmax,
           log_softmax, logsumexp, exp, max, argmax, sum, cumsum; penalties and logit bias
  ties     exact ties in torch.sort, torch.topk and argmax; CPU vs GPU tie order
  nucleus  how often the top-p set of a row changes between a single-row call and a batch
  unit     reference sampler: Philox known answers, a scalar Python implementation, stable
           top-k, greedy ties, integer sums, exp modes, CPU vs GPU
  chisq    reference sampler: chi-square of empirical frequencies against the target
  l2       reference sampler: run-to-run (L1) and batch size, position, fillers (L2)
  engine   a toy decode loop: batched schedules, preemption, rollback, speculative
           verification in chunks of 1..8, resume in a new process; reference and stock
  cost     time per sampling step against torch.multinomial and per-request generators

The stock paths named "vLLM-style" or "SGLang-style" are re-implementations of the few lines
read from vllm 0.6.3.post1 and sglang 0.4.0; neither engine was run.

Run (CUDA):  python sampling_bi.py --quick --json out.json       about 2 minutes
             python sampling_bi.py --json results.json           full, about 15 minutes
             bash run_full.sh                                    full, one section per GPU lock
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import bi_sampling as bs  # noqa: E402
from bi_check import bit_equal, bits, check_batch, check_modes, check_repeat, first_mismatch  # noqa: E402
from bi_report import SCHEMA, Results  # noqa: E402

dev = "cuda"
VOCABS = (32000, 151936)
SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513)
SIZES_QUICK = (1, 2, 3, 8, 17, 64, 65)
SIZES_STOCK = (1, 2, 3, 4, 8, 15, 16, 17, 64, 65, 256, 257, 512, 513)
CONFIGS = {                                   # temperature, top_k, top_p
    "T=1": (1.0, 0, 1.0), "T=0.7": (0.7, 0, 1.0), "top_k=50": (1.0, 50, 1.0), "top_p=0.9": (1.0, 0, 0.9),
    "T=0.8,top_k=40,top_p=0.95": (0.8, 40, 0.95), "greedy": (0.0, 0, 1.0),
}
MIXED = [(1.0, 0, 1.0), (0.7, 0, 1.0), (1.0, 50, 1.0), (1.0, 0, 0.9), (0.8, 40, 0.95), (0.0, 0, 1.0),
         (1.3, 7, 0.5), (0.5, 1000, 0.99)]


def G(seed):
    return torch.Generator(device=dev).manual_seed(int(seed))


def brief(rep):
    """A Report without its failure list (results stay small)."""
    d = {"checked": rep.checked, "failures": len(rep.failures)}
    if rep.failures:
        sizes = sorted({f["size"] for f in rep.failures if f.get("size") is not None})
        if sizes:
            d["failing_sizes"] = sizes
            d["failing_positions"] = sorted({f["position"] for f in rep.failures})
        d["first"] = rep.failures[0]
    return d


def row_keys(batch):
    """Seed, position and a parameter id that travel with the row: taken from its own bits."""
    b = bits(batch[:, :2].contiguous()).to(torch.int64).cpu()
    return b[:, 0] * 2654435761 + b[:, 1], b[:, 1] & 0xFFFFF, ((b[:, 0] >> 3) ^ b[:, 1]) & 7


def planted_ties(n, V, gen, groups=6, group=5):
    """Random rows whose largest values are tie groups at random indices."""
    x = torch.randn(n, V, generator=gen, device=dev)
    vals = (10.0 - torch.arange(groups, device=dev).float()).repeat_interleave(group)
    for r in range(n):
        x[r, torch.randperm(V, generator=gen, device=dev)[: groups * group]] = vals
    return x


def probe_rows(V, gen, n_random=1):
    """Probe rows: random logits, a flat row, a row with planted ties, a bf16-quantized row."""
    r = torch.randn(n_random, V, generator=gen, device=dev) * 3
    flat = torch.randn(1, V, generator=gen, device=dev) * 1.5
    q = (torch.randn(1, V, generator=gen, device=dev) * 3).to(torch.bfloat16).float()
    return torch.cat([r, flat, planted_ties(1, V, gen), q])


# ============================================================================= stock paths
def stock_shared_multinomial(batch):
    return torch.multinomial(torch.softmax(batch, -1), 1, generator=G(0))


def stock_per_request_exponential(batch):
    """One generator per request, an exponential fill per row, argmax(probs / q)
    (the seeded path of vllm 0.6.3.post1's _multinomial with one sample)."""
    seeds, _, _ = row_keys(batch)
    probs = torch.softmax(batch, -1)
    q = torch.empty_like(probs)
    for r in range(batch.shape[0]):
        q[r:r + 1].exponential_(generator=G(seeds[r] & 0x7FFFFFFF))
    return probs.div_(q).argmax(-1, keepdim=True)


def stock_per_request_gumbel(batch):
    seeds, _, _ = row_keys(batch)
    out = torch.empty(batch.shape[0], 1, dtype=torch.long, device=dev)
    for r in range(batch.shape[0]):
        u = torch.rand(1, batch.shape[1], device=dev, generator=G(seeds[r] & 0x7FFFFFFF))
        out[r] = (batch[r:r + 1] - torch.log(-torch.log(u))).argmax(-1)
    return out


def vllm_style_top_k_top_p(logits, p, k):
    """_apply_top_k_top_p as read from vllm 0.6.3.post1 (model_executor/layers/sampler.py)."""
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    top_k_mask = logits_sort.size(1) - k.to(torch.long)
    top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
    top_k_mask = logits_sort < top_k_mask
    logits_sort.masked_fill_(top_k_mask, -float("inf"))
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)


def sglang_style_top_p_probs(probs, top_ps, top_ks):
    """top_k_top_p_min_p_sampling_from_probs_torch as read from sglang 0.4.0, up to the draw."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort[torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1) >= top_ks.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    return probs_sort, probs_idx


def vllm_style_penalties(batch):
    """_apply_penalties as read from vllm 0.6.3.post1, plus a logit bias. Each row's token
    history comes from its own bits and is padded with vocab_size to the batch's longest."""
    M, V = batch.shape
    hb = bits(batch[:, 3:35].contiguous()).to(torch.int64).abs()
    lens = 5 + hb[:, 0] % 27
    toks = torch.where(torch.arange(32, device=dev)[None, :] >= lens[:, None], V, hb % V)[:, : int(lens.max())]
    bin_counts = torch.zeros((M, V + 1), dtype=torch.long, device=dev)
    bin_counts.scatter_add_(1, toks, torch.ones_like(toks))
    bin_counts = bin_counts[:, :V]
    mask = bin_counts > 0
    rep = torch.full((M, V), 1.2, device=dev)
    rep[~mask] = 1.0
    logits = torch.where(batch > 0, batch / rep, batch * rep)
    logits -= 0.5 * bin_counts
    logits -= 0.3 * mask
    bias = torch.zeros(M, V + 1, device=dev).scatter_(1, toks[:, :3], 5.0)[:, :V]
    return logits.add_(bias)


# ============================================================================= rng
def section_rng(res, quick):
    gen = G(1)
    for V in VOCABS:
        rows = torch.randn(4, V, generator=gen, device=dev) * 1.5
        rep = check_batch(stock_shared_multinomial, rows, SIZES_STOCK, name="shared")
        res.check(f"torch.multinomial, one generator for the batch: row vs alone, V={V}", rep.passed,
                  level="L2", impl="stock_shared_stream", expect="fail", detail=brief(rep))
        first = check_batch(stock_shared_multinomial, rows, SIZES_STOCK, positions=("first",), name="shared-first")
        res.check(f"same, row placed first only, V={V}", first.passed, level="L2", impl="stock_shared_stream",
                  expect="info", detail=brief(first))
        for name, fn in (("exponential", stock_per_request_exponential), ("gumbel", stock_per_request_gumbel)):
            rep = check_batch(fn, rows[:2], SIZES_STOCK, name=name)
            res.check(f"one torch.Generator per request, {name} noise per row: row vs alone, V={V}", rep.passed,
                      level="L2", impl="stock_per_request_generator", expect="info", detail=brief(rep))
        probs = torch.softmax(torch.randn(256, V, generator=gen, device=dev) * 2, -1)
        same = bit_equal(torch.multinomial(probs, 1, generator=G(5)),
                         (probs / torch.empty_like(probs).exponential_(generator=G(5))).argmax(-1, keepdim=True))
        res.check(f"torch.multinomial(probs, 1) == argmax(probs / exponential_) with the same generator, V={V}", same,
                  level="other", impl="stock_shared_stream", expect="info")

    # the CUDA stream is Philox4x32-10, element i of a fill uses counter word 2 = i
    n = 4096
    r = torch.rand(n, device=dev, generator=G(77))
    i = torch.arange(n, dtype=torch.int64)
    z = torch.zeros(n, dtype=torch.int64)
    o = bs.philox4x32((z, z, i, z), (z + 77, z))
    mine = (o[0].to(torch.float32) * 2.0 ** -32 + 2.0 ** -33)
    mine = torch.where(mine == 1.0, torch.zeros_like(mine), mine)
    res.check("torch.rand on CUDA, element i == Philox4x32-10(counter=(0,0,i,0), key=seed) (4096 elements)",
              bit_equal(r.cpu(), mine), level="other", impl="stock_torch_generator", expect="info")
    c = torch.empty(1000).exponential_(generator=torch.Generator().manual_seed(1234))
    d = torch.empty(1000, device=dev).exponential_(generator=G(1234))
    res.check("a CPU torch.Generator and a CUDA one with the same seed give the same values", bit_equal(c, d.cpu()),
              level="L4", impl="stock_torch_generator", expect="fail")
    g = G(3)
    torch.empty(1, 1000, device=dev).exponential_(generator=g)
    st = g.get_state()
    a = torch.empty(1, 1000, device=dev).exponential_(generator=g)
    g2 = torch.Generator(device=dev)
    g2.set_state(st)
    res.check("CUDA generator get_state / set_state reproduces the next draw", bit_equal(a, torch.empty(1, 1000, device=dev).exponential_(generator=g2)),
              level="L1", impl="stock_torch_generator", expect="info", detail={"state_bytes": st.numel()})

    # how much a fill advances the generator: the next draw after a first fill of n elements
    def next_after(n):
        g = G(9)
        torch.empty(n, device=dev).exponential_(generator=g)
        return torch.empty(64, device=dev).exponential_(generator=g)
    base = next_after(1)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    table = {n: bit_equal(next_after(n), base) for n in (2, 32000, 151936, 16 * 32000, 516095, 516096, 516097, 17 * 32000, 4 * 151936, 1032192, 1032193)}
    res.extra["rng_next_draw_unchanged_after_fill_of_n"] = {str(k): v for k, v in table.items()}
    res.extra["sm_count"] = sm
    res.check("the draw after a fill does not depend on the fill's size, up to 516,096 float32 elements", table[516096] and table[2],
              level="other", impl="stock_torch_generator", expect="info", detail={"sm_count": sm, "sm_count*6144": sm * 6144})
    res.check("the draw after a fill of 516,097 elements equals the draw after a fill of 1", table[516097],
              level="other", impl="stock_torch_generator", expect="fail")

    # vLLM-style seeded _multinomial: every seeded request draws [len(seq_ids) * num_samples, V],
    # where num_samples is the largest n among the prompt-phase requests of the batch
    def vllm_seeded_step(probs_row, gen, num_samples, n_seqs=1):
        q = torch.empty(n_seqs * num_samples, probs_row.shape[-1], device=dev)
        q.exponential_(generator=gen)
        pr = probs_row.repeat_interleave(num_samples, dim=0) if probs_row.shape[0] == n_seqs else probs_row.expand(n_seqs * num_samples, -1)
        return (pr / q).argmax(-1).view(-1, num_samples)[:, 0], q
    for V, ns in ((32000, (2, 16, 17)), (151936, (2, 3, 4))):
        p2 = torch.softmax(torch.randn(2, V, generator=gen, device=dev) * 1.5, -1)
        for nmax in ns:
            g1, gn = G(42), G(42)
            t1, q1 = vllm_seeded_step(p2[:1], g1, 1)
            tn, qn = vllm_seeded_step(p2[:1], gn, nmax)
            same_now = bit_equal(q1[0], qn[0])
            nxt1 = torch.empty(1, V, device=dev).exponential_(generator=g1)
            nxtn = torch.empty(1, V, device=dev).exponential_(generator=gn)
            res.check(f"vLLM-style seeded draw: a decoding request's NEXT noise is unchanged when a co-batched prompt asks n={nmax}, V={V}",
                      bit_equal(nxt1, nxtn), level="L2", impl="stock_per_request_generator",
                      expect="fail" if nmax * V > 516096 else "info",
                      detail={"fill_elements": nmax * V, "noise_this_step_unchanged": same_now})
        # a request with 3 running samples: sibling j uses noise row j * num_samples
        g1, g2 = G(43), G(43)
        _, qa = vllm_seeded_step(p2[:1], g1, 1, n_seqs=3)
        _, qb = vllm_seeded_step(p2[:1], g2, 2, n_seqs=3)
        sib = [bit_equal(qa[j], qb[2 * j]) for j in range(3)]
        res.check(f"vLLM-style seeded draw, a request with 3 running samples: each sibling's noise is unchanged when a co-batched prompt asks n=2, V={V}",
                  all(sib), level="L2", impl="stock_per_request_generator", expect="fail", detail={"sibling_unchanged": sib})


# ============================================================================= ops
def batch_groups(fn, row, others, sizes):
    """Runs of batch sizes over which fn(batch)[0] keeps the same bits for the probe row."""
    groups, prev, start, last = [], None, None, None
    for B in sizes:
        key = fn(torch.cat([row, others[: B - 1]]))[0].cpu().numpy().tobytes()
        if key != prev:
            if prev is not None:
                groups.append([start, last])
            start, prev = B, key
        last = B
    groups.append([start, last])
    return groups


def distinct_results(fn, runs=20):
    seen = {}
    for _ in range(runs):
        k = fn().cpu().numpy().tobytes()
        seen[k] = seen.get(k, 0) + 1
    return len(seen)


def section_ops(res, quick):
    gen = G(2)
    for V in VOCABS:
        rows = torch.randn(3, V, generator=gen, device=dev) * 3
        t = torch.full((1, 1), 0.7, device=dev)
        for name, fn in (
            ("temperature divide", lambda x: x.clone().div_(t.expand(x.shape[0], 1))),
            ("softmax", lambda x: torch.softmax(x, -1)),
            ("log_softmax", lambda x: torch.log_softmax(x, -1)),
            ("logsumexp", lambda x: torch.logsumexp(x, -1, keepdim=True)),
            ("exp", lambda x: torch.exp(x)),
            ("max", lambda x: x.max(-1, keepdim=True).values),
            ("argmax", lambda x: x.argmax(-1, keepdim=True)),
            ("penalties and logit bias (vLLM-style gather/scatter)", vllm_style_penalties),
        ):
            rep = check_batch(fn, rows, SIZES, name=name)
            res.check(f"{name}: row vs alone, V={V}", rep.passed, level="L2", impl="stock_torch_op", expect="info", detail=brief(rep))
        rep = check_batch(lambda x: x.sum(-1, keepdim=True), rows, SIZES, name="sum")
        res.check(f"sum over the vocabulary: row vs alone, V={V}", rep.passed, level="L2", impl="stock_torch_op",
                  expect="fail" if V > 100000 else "info", detail=brief(rep))
        sp = lambda x: torch.sort(torch.softmax(x, -1), dim=-1, descending=True).values  # noqa: E731
        for name, fn, exp_ in (("cumsum fp32 of the sorted probabilities", lambda x: torch.cumsum(sp(x), -1), "fail"),
                               ("cumsum fp64 of the sorted probabilities", lambda x: torch.cumsum(sp(x).double(), -1), "fail")):
            rep = check_batch(fn, rows, SIZES, name=name)
            res.check(f"{name}: row vs alone, V={V}", rep.passed, level="L2", impl="stock_torch_cumsum", expect=exp_, detail=brief(rep))
        p1 = sp(rows[:1])
        n1 = distinct_results(lambda: torch.cumsum(p1, -1), 30)
        n1d = distinct_results(lambda: torch.cumsum(torch.softmax(rows[:1], -1).double(), -1), 30)
        res.check(f"cumsum fp32 of one row, 30 runs of the same tensor, V={V}", n1 == 1, level="L1", impl="stock_torch_cumsum",
                  expect="fail" if V > 100000 else "info", detail={"distinct_results": n1, "distinct_results_fp64": n1d})
        pm = sp(torch.randn(64, V, generator=gen, device=dev) * 3)
        nm = distinct_results(lambda: torch.cumsum(pm, -1), 20)
        res.check(f"cumsum fp32 of 64 rows, 20 runs of the same tensor, V={V}", nm == 1, level="L1", impl="stock_torch_cumsum",
                  expect="info", detail={"distinct_results": nm})
        nmax = 1100 if V == 32000 else 600
        others = sp(torch.randn(nmax, V, generator=gen, device=dev) * 3)
        res.extra[f"cumsum_fp32_batch_size_groups_V{V}"] = batch_groups(lambda b: torch.cumsum(b, -1), p1, others, range(2, nmax + 1))
        del others
        o2 = torch.randn(300, V, generator=gen, device=dev)
        res.extra[f"sum_fp32_batch_size_groups_V{V}"] = batch_groups(lambda b: b.sum(-1), rows[:1], o2, range(1, 301))
        del o2
    for V in (65536, 100000, 131072, 262144):
        row = torch.randn(1, V, generator=gen, device=dev)
        o2 = torch.randn(64, V, generator=gen, device=dev)
        res.extra[f"sum_fp32_batch_size_groups_V{V}"] = batch_groups(lambda b: b.sum(-1), row, o2, range(1, 65))
        res.extra[f"cumsum_fp32_one_row_distinct_in_30_runs_V{V}"] = distinct_results(lambda: torch.cumsum(row, -1), 30)
    p = torch.softmax(torch.randn(1, 32000, generator=gen, device=dev), -1)
    out = {}
    for warn_only in (True, False):
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                torch.cumsum(p, -1)
            out[f"warn_only={warn_only}"] = "runs; " + "; ".join(sorted({str(x.message)[:80] for x in w}))
        except RuntimeError as e:
            out[f"warn_only={warn_only}"] = "RuntimeError: " + str(e)[:80]
        finally:
            torch.use_deterministic_algorithms(False)
    res.extra["cumsum_deterministic_mode"] = out


# ============================================================================= ties
def section_ties(res, quick):
    gen = G(3)
    small = []
    for V in list(range(2, 70)) + [128, 1024]:
        x = torch.randint(0, 3, (512, V), generator=gen, device=dev).float()
        if any(not bit_equal(torch.sort(x, dim=-1, descending=d, stable=False).indices,
                             torch.sort(x, dim=-1, descending=d, stable=True).indices) for d in (True, False)):
            small.append(V)
    res.extra["unstable_sort_differs_from_stable_at_V"] = small
    res.check("torch.sort(stable=False) keeps tied elements in index order, V = 2..32 (3 distinct values)", not [v for v in small if v <= 32],
              level="other", impl="stock_torch_sort", expect="fail", detail={"V_with_differences": small})
    for V in VOCABS:
        rows = planted_ties(4, V, gen)
        sizes = (1, 2, 3, 8, 10, 11, 16, 64, 65, 256, 257, 512)
        ok_sort, ok_set, ok_order, ok_alone = True, True, True, True
        for k in (3, 7, 20, 28, 50):
            ref = torch.sort(rows, dim=-1, descending=True, stable=True).indices[:, :k]
            alone = torch.cat([torch.topk(rows[r:r + 1], k).indices for r in range(4)])
            ok_order &= bit_equal(alone, ref)
            ok_set &= bit_equal(alone.sort(-1).values, ref.sort(-1).values)
            for B in sizes[1:]:
                others = planted_ties(B - 1, V, gen)
                for pos in (0, B - 1):
                    batch = torch.cat([others[:pos], rows[:1], others[pos:]])
                    ib = torch.topk(batch, k).indices[pos:pos + 1]
                    ok_alone &= bit_equal(ib, alone[:1])
                    if k == 3:
                        for desc in (True, False):
                            a = torch.sort(batch, dim=-1, descending=desc).indices[pos:pos + 1]
                            ok_sort &= bit_equal(a, torch.sort(rows[:1], dim=-1, descending=desc, stable=True).indices)
        res.check(f"torch.topk with ties across the cut selects the lowest indices, V={V}", ok_set, level="other", impl="stock_torch_topk", expect="info")
        res.check(f"torch.topk returns tied values in index order, V={V}", ok_order, level="other", impl="stock_torch_topk", expect="fail")
        res.check(f"torch.topk indices of a row: alone vs in batches of 2..512, V={V}", ok_alone, level="L2", impl="stock_torch_topk", expect="info")
        res.check(f"torch.sort(stable=False) of a row with ties equals the stable order, alone and in batches of 2..512, V={V}", ok_sort,
                  level="L2", impl="stock_torch_sort", expect="info")
        g_idx = torch.topk(rows, 20).indices.cpu()
        c_idx = torch.topk(rows.cpu(), 20).indices
        res.check(f"torch.topk tie order, GPU vs CPU, V={V}", bit_equal(g_idx, c_idx), level="L4", impl="stock_torch_topk", expect="fail",
                  detail={"differing_indices": int((g_idx != c_idx).sum()), "of": g_idx.numel()})
        bad = total = 0
        for B in (1, 2, 3, 8, 64, 256, 512):
            x = torch.randn(B, V, generator=gen, device=dev)
            want = torch.empty(B, dtype=torch.long, device=dev)
            for r in range(B):
                idx = torch.randint(0, V, (int(torch.randint(2, 6, (1,), generator=gen, device=dev)),), generator=gen, device=dev)
                x[r, idx] = 99.0
                want[r] = idx.min()
            bad += int((torch.argmax(x, -1) != want).sum()) + int((x.max(-1).indices != want).sum())
            total += 2 * B
        res.check(f"torch.argmax and max().indices return the lowest index among tied maxima, batches 1..512, V={V}", bad == 0,
                  level="L2", impl="stock_torch_argmax", expect="info", detail={"wrong": bad, "of": total})


# ============================================================================= nucleus
def section_nucleus(res, quick):
    gen = G(4)
    flips = {}

    def keep_desc(x, top_p):                  # sglang 0.4.0 torch backend
        ps, _ = sglang_style_top_p_probs(torch.softmax(x, -1), torch.full((x.shape[0],), top_p, device=dev),
                                         torch.full((x.shape[0],), 1 << 30, device=dev))
        return (ps > 0).sum(-1)

    def keep_asc(x, top_p):                   # vllm 0.6.3.post1
        out = vllm_style_top_k_top_p(x.clone(), torch.full((x.shape[0],), top_p, device=dev),
                                     torch.full((x.shape[0],), x.shape[1], device=dev))
        return torch.isfinite(out).sum(-1)

    def keep_ref(x, top_p):
        return (bs.kept_weights(x, top_p=top_p)[0] > 0).sum(-1)

    for V, N in (((32000, 256), (151936, 64)) if quick else ((32000, 2048), (151936, 512))):
        for scale in (1.5, 3.0):
            x = torch.randn(N, V, generator=gen, device=dev) * scale
            for top_p in (0.9, 0.95):
                for name, fn, impl in (("descending cumsum (SGLang-style)", keep_desc, "stock_top_p_sglang_style"),
                                       ("ascending cumsum (vLLM-style)", keep_asc, "stock_top_p_vllm_style"),
                                       ("integer mass (reference)", keep_ref, "bi_sampling")):
                    alone = torch.cat([fn(x[i:i + 1], top_p) for i in range(N)])
                    again = torch.cat([fn(x[i:i + 1], top_p) for i in range(N)])
                    d = {"rows": N, "median_nucleus": int(alone.median()), "single_row_call_twice": int((alone != again).sum())}
                    for B in (2, 64, 256, 512):
                        inb = torch.cat([fn(x[i:i + B], top_p) for i in range(0, N, B)])
                        d[f"B={B}"] = int((inb != alone).sum())
                    changed = max(v for k, v in d.items() if k.startswith("B=") or k.startswith("single"))
                    flips[impl] = flips.get(impl, 0) + changed
                    res.check(f"top-p set of a row, single-row call vs batches of 2..512: {name}, V={V}, logits N(0,{scale}^2), top_p={top_p}",
                              changed == 0, level="L2", impl=impl, expect="info", detail=d)
            del x
    for impl, n in flips.items():
        res.check(f"top-p set of a row is the same in a single-row call and in batches of 2..512, over all 8 settings above", n == 0, level="L2",
                  impl=impl, expect="pass" if impl == "bi_sampling" else "fail", detail={"sum_of_max_rows_changed": n})
    # what a changed set does downstream (vLLM-style: mask, softmax again, log_softmax, argmax(probs / q))
    V, N, top_p = 32000, 256 if quick else 2048, 0.95
    x = torch.randn(N, V, generator=gen, device=dev) * 1.5
    pp, kk = torch.full((1,), top_p, device=dev), torch.full((1,), V, device=dev)

    def pipeline(xb, seeds):
        lg = vllm_style_top_k_top_p(xb.clone(), pp.expand(xb.shape[0]), kk.expand(xb.shape[0]))
        probs, logprobs = torch.softmax(lg, -1), torch.log_softmax(lg, -1)
        q = torch.empty_like(probs)
        for r in range(xb.shape[0]):
            q[r:r + 1].exponential_(generator=G(seeds[r]))
        tok = (probs / q).argmax(-1)
        return tok, logprobs.gather(1, tok[:, None])[:, 0], torch.isfinite(lg).sum(-1)
    seeds = list(range(N))
    a = [torch.cat(t) for t in zip(*[pipeline(x[i:i + 1], seeds[i:i + 1]) for i in range(N)])]
    b = [torch.cat(t) for t in zip(*[pipeline(x[i:i + 64], seeds[i:i + 64]) for i in range(0, N, 64)])]
    d = {"rows": N, "set_size_differs": int((a[2] != b[2]).sum()), "token_differs": int((a[0] != b[0]).sum()),
         "logprob_of_token_bits_differ": int((bits(a[1]) != bits(b[1])).sum())}
    res.check("vLLM-style top-p pipeline with per-request generators, single-row call vs batch of 64: token and its log-probability (V=32000, top_p=0.95)",
              d["token_differs"] == 0 and d["logprob_of_token_bits_differ"] == 0, level="L2", impl="stock_top_p_vllm_style", expect="info", detail=d)


# ============================================================================= unit
def philox_py(ctr, key):
    c, k, M = list(ctr), list(key), 0xFFFFFFFF
    for r in range(10):
        if r:
            k = [(k[0] + 0x9E3779B9) & M, (k[1] + 0xBB67AE85) & M]
        p0, p1 = 0xD2511F53 * c[0], 0xCD9E8D57 * c[2]
        c = [(p1 >> 32) ^ c[1] ^ k[0], p1 & M, (p0 >> 32) ^ c[3] ^ k[1], p0 & M]
    return c


def scalar_reference(row, seed, position, temperature, top_k, top_p, purpose=0, stream=0):
    """The exp="basic" sampler for one row in plain Python integers and numpy float32 scalars."""
    import numpy as np
    f, M = np.float32, 0xFFFFFFFF
    V = len(row)
    S = bs.scale_bits(V)
    if temperature == 0:
        m = max(row)
        return min(i for i, v in enumerate(row) if v == m), 0, 0
    z = [f(v) / f(temperature) for v in row]
    m = max(z)
    w = []
    for v in z:
        d = max(v - m, f(-(S + 2) * bs._LN2_HI))
        n = np.floor(d * f(bs._INV_LN2) + f(0.5))
        y = (d - n * f(bs._LN2_HI)) - n * f(bs._LN2_LO)
        acc = f(bs._EXP_COEFFS[0])
        for c in bs._EXP_COEFFS[1:]:
            acc = acc * y + f(c)
        w.append(int(np.floor(acc * f(2.0 ** (int(n) + S)))))
    order = sorted(range(V), key=lambda i: (-float(z[i]), i))
    kept = order[: V if top_k <= 0 else min(top_k, V)]
    bound = math.ceil(top_p * float(sum(w[i] for i in kept)))
    before, keep = 0, set()
    for i in kept:
        if before < bound:
            keep.add(i)
        before += w[i]
    w = [w[i] if i in keep else 0 for i in range(V)]
    total = sum(w)
    o = philox_py((position & M, (position >> 32) & M, purpose, stream), (seed & M, (seed >> 32) & M))
    u = (o[0] << 21) | (o[1] >> 11)
    r = min(int(math.floor(float(u) * 2.0 ** -53 * float(total))), total - 1)
    c = 0
    for i in range(V):
        c += w[i]
        if c > r:
            return i, w[i], total


def section_unit(res, quick):
    gen = G(5)
    kat = [((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
           ((0xffffffff,) * 4, (0xffffffff,) * 2, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
           ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0), (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1))]
    ok = all(tuple(int(o) for o in bs.philox4x32([torch.tensor([c]) for c in ctr], [torch.tensor([k]) for k in key])) == want
             and tuple(philox_py(ctr, key)) == want for ctr, key, want in kat)
    res.check("Philox4x32-10 known-answer vectors (Random123)", ok, level="other", impl="bi_sampling")
    n = 4096
    seeds = torch.randint(-2 ** 62, 2 ** 62, (n,), generator=torch.Generator().manual_seed(0))
    pos = torch.randint(0, 2 ** 40, (n,), generator=torch.Generator().manual_seed(1))
    allu = bs.uniform53(seeds, pos, bs.ACCEPT, 3)
    one = torch.cat([bs.uniform53(int(seeds[i]), int(pos[i]), bs.ACCEPT, 3) for i in range(0, n, 37)])
    res.check("uniform53: a row's draw alone equals its draw inside a vector of 4096", bit_equal(allu[::37], one), level="L2", impl="bi_sampling")
    purposes = torch.stack([bs.uniform53(seeds, pos, p, s) for p in (bs.TOKEN, bs.ACCEPT, bs.RESIDUAL, bs.DRAFT) for s in (0, 1)])
    res.check("uniform53: purposes and streams give different draws for the same (seed, position)",
              all(int((purposes[i] == purposes[j]).sum()) == 0 for i in range(8) for j in range(i)), level="other", impl="bi_sampling")

    # a scalar Python implementation of exp="basic" against the vectorized one
    try:
        import numpy  # noqa: F401
        V, nrow = 96, 48 if quick else 160
        x = (torch.randn(nrow, V, generator=gen, device=dev) * 3)
        x[: nrow // 2] = (x[: nrow // 2] * 2).round() / 2            # many exact ties
        x[1, 5] = float("-inf")
        mism = 0
        for i in range(nrow):
            t_, k_, p_ = MIXED[i % 8]
            info = bs.sample(x[i:i + 1], 1000 + i, 3 * i, temperature=t_, top_k=min(k_, 30), top_p=p_, exp="basic", return_info=True,
                             stream=i % 3)
            tok, wt, tot = scalar_reference([numpy.float32(v) for v in x[i].tolist()], 1000 + i, 3 * i, t_, min(k_, 30), p_, stream=i % 3)
            mism += int(tok != int(info.tokens)) + (0 if t_ == 0 else int(wt != int(info.weight)) + int(tot != int(info.total)))
        res.check(f"exp=basic equals a scalar Python implementation: token, weight and total on {nrow} rows with ties (V=96)", mism == 0,
                  level="L4", impl="bi_sampling", detail={"mismatches": mism})
    except ImportError:
        res.check("exp=basic equals a scalar Python implementation", True, level="L4", impl="bi_sampling", expect="info", detail="numpy missing: skipped")

    for V in VOCABS:
        rows = planted_ties(4, V, gen)
        bad = 0
        for k in (3, 7, 20, 28, 50):
            w, _ = bs.kept_weights(rows, top_k=k)
            ref = torch.sort(rows.cpu(), dim=-1, descending=True, stable=True).indices[:, :k].sort(-1).values
            got = (w > 0).nonzero()[:, 1].view(4, -1).cpu()
            bad += int(got.shape != ref.shape or not bit_equal(got, ref))
        res.check(f"top-k with ties across the cut keeps the lowest indices (k = 3..50), V={V}", bad == 0, level="other", impl="bi_sampling")
        bad = total = 0
        for B in (1, 2, 8, 64) if quick else (1, 2, 3, 8, 64, 256, 512):
            x = torch.randn(B, V, generator=gen, device=dev)
            want = torch.empty(B, dtype=torch.long, device=dev)
            for r in range(B):
                idx = torch.randint(0, V, (int(torch.randint(2, 6, (1,), generator=gen, device=dev)),), generator=gen, device=dev)
                x[r, idx] = 99.0
                want[r] = idx.min()
            bad += int((bs.sample(x, 1, 2, temperature=0.0) != want).sum()) + int((bs.greedy(x) != want).sum())
            total += 2 * B
        res.check(f"greedy: the lowest index among tied maxima, batches up to {B}, V={V}", bad == 0, level="L2", impl="bi_sampling",
                  detail={"wrong": bad, "of": total})
        rows = probe_rows(V, gen)
        sizes = SIZES_QUICK if quick else SIZES
        rep = check_batch(lambda b: bs._weights(b, "torch").cumsum(-1), rows[:2], sizes, name="int cumsum")
        res.check(f"int64 cumsum of the weights: row vs alone, V={V}", rep.passed, level="L2", impl="bi_sampling", detail=brief(rep))
        wt, wb = bs._weights(rows, "torch"), bs._weights(rows, "basic")
        res.check(f"int64 cumsum and sum on the GPU equal the CPU's, V={V}", bit_equal(wt.cumsum(-1).cpu(), wt.cpu().cumsum(-1))
                  and bit_equal(wt.sum(-1).cpu(), wt.cpu().sum(-1)), level="L4", impl="bi_sampling")
        res.check(f"the row maximum gets weight 2**S and totals stay below 2**52, V={V}", bool((wt.max(-1).values == 2 ** bs.scale_bits(V)).all())
                  and int(wt.sum(-1).max()) < 2 ** 52, level="other", impl="bi_sampling", detail={"S": bs.scale_bits(V)})
        big = torch.randn(16 if quick else 64, V, generator=gen, device=dev) * 3
        wt, wb = bs._weights(big, "torch"), bs._weights(big, "basic")
        wtc, wbc = bs._weights(big.cpu(), "torch"), bs._weights(big.cpu(), "basic")
        res.check(f"exp=basic weights: GPU == CPU, V={V}", bit_equal(wb.cpu(), wbc), level="L4", impl="bi_sampling",
                  detail={"elements": wb.numel()})
        res.check(f"exp=torch weights: GPU == CPU, V={V}", bit_equal(wt.cpu(), wtc), level="L4", impl="bi_sampling", expect="info",
                  detail={"differing": int((wt.cpu() != wtc).sum()), "of": wt.numel()})
        nz = wt > 0
        res.check(f"exp=torch and exp=basic give the same weights, V={V}", bit_equal(wt, wb), level="other", impl="bi_sampling", expect="info",
                  detail={"differing": int((wt != wb).sum()), "of": wt.numel(),
                          "max_relative_difference": float(((wt - wb).abs().double() / wt.clamp(min=1).double())[nz].max())})
        kw = dict(temperature=[MIXED[i % 8][0] for i in range(big.shape[0])], top_k=[MIXED[i % 8][1] for i in range(big.shape[0])],
                  top_p=[MIXED[i % 8][2] for i in range(big.shape[0])])
        seeds_, pos_ = list(range(50, 50 + big.shape[0])), list(range(big.shape[0]))
        tg = bs.sample(big, seeds_, pos_, exp="basic", **kw)
        tc = bs.sample(big.cpu(), seeds_, pos_, exp="basic", **kw)
        res.check(f"exp=basic tokens with mixed parameters: GPU == CPU, V={V}", bit_equal(tg.cpu(), tc), level="L4", impl="bi_sampling")
        one = torch.cat([bs.sample(big[i:i + 1], seeds_[i], pos_[i], temperature=kw["temperature"][i], top_k=kw["top_k"][i], top_p=kw["top_p"][i])
                         for i in range(big.shape[0])])
        res.check(f"a batch with mixed parameters equals row-by-row calls with each row's own parameters, V={V}",
                  bit_equal(bs.sample(big, seeds_, pos_, **kw), one), level="L2", impl="bi_sampling")


# ============================================================================= chisq
def target_dist(row, temperature, top_k, top_p):
    """The target distribution in float64 on the CPU, written independently of the sampler."""
    z = (row.double() / temperature).tolist()
    m = max(z)
    p = [math.exp(v - m) for v in z]
    order = sorted(range(len(z)), key=lambda i: (-z[i], i))
    kept = order[: len(z) if top_k <= 0 else top_k]
    total, mass, keep = sum(p[i] for i in kept), 0.0, []
    for i in kept:
        if mass < top_p * total:
            keep.append(i)
        mass += p[i]
    s = sum(p[i] for i in keep)
    out = torch.zeros(len(z), dtype=torch.float64)
    for i in keep:
        out[i] = p[i] / s
    return out


def chi_square(counts, probs, min_expected=10.0):
    N = counts.sum().double()
    inside = probs > 0
    outside_draws = int(counts[~inside].sum())
    O, E = counts[inside].double(), probs[inside] * N
    small = E < min_expected
    if int(small.sum()) > 0:
        O = torch.cat([O[~small], O[small].sum()[None]])
        E = torch.cat([E[~small], E[small].sum()[None]])
    stat = float(((O - E) ** 2 / E).sum())
    dof = O.numel() - 1
    pval = float(torch.special.gammaincc(torch.tensor(dof / 2.0, dtype=torch.float64), torch.tensor(stat / 2.0, dtype=torch.float64))) if dof > 0 else 1.0
    return {"chi2": round(stat, 2), "dof": dof, "p_value": pval, "draws": int(N), "draws_outside_support": outside_draws,
            "support": int(inside.sum())}


def draw_counts(row, N, keys, chunk, fn):
    counts = torch.zeros(row.numel(), dtype=torch.int64, device=dev)
    for s in range(0, N, chunk):
        idx = torch.arange(s, min(s + chunk, N))
        toks = fn(row[None].expand(idx.numel(), -1), idx)
        counts += torch.bincount(toks, minlength=row.numel())
    return counts.cpu()


def section_chisq(res, quick):
    gen = G(6)
    N = 1 << (18 if quick else 20)
    rows = {50: torch.randn(50, generator=gen, device=dev) * 2, 1000: torch.randn(1000, generator=gen, device=dev) * 2.5}
    cases = [(50, 1.0, 0, 1.0), (50, 0.7, 0, 1.0), (50, 1.5, 0, 1.0), (50, 1.0, 5, 1.0), (50, 0.8, 10, 1.0), (50, 1.0, 0, 0.9),
             (50, 1.3, 0, 0.5), (50, 1.0, 20, 0.8), (1000, 1.0, 0, 1.0), (1000, 0.9, 50, 1.0), (1000, 1.1, 0, 0.9), (1000, 0.8, 200, 0.95)]
    if quick:
        cases = cases[:8] + cases[9:10]
    for V, t, k, p in cases:
        row, chunk = rows[V], (1 << 18 if V == 50 else 1 << 15)
        tgt = target_dist(row.cpu(), t, k, p)
        for layout, exp in ((("seeds", "torch"), ("positions", "torch"), ("seeds", "basic")) if (V, t, k, p) in (cases[0], cases[7]) else (("seeds", "torch"),)):
            if layout == "seeds":
                fn = lambda x, idx: bs.sample(x, idx, 7, temperature=t, top_k=k, top_p=p, exp=exp)  # noqa: E731
            else:
                fn = lambda x, idx: bs.sample(x, 20261119, idx, temperature=t, top_k=k, top_p=p, exp=exp)  # noqa: E731
            d = chi_square(draw_counts(row, N, None, chunk, fn), tgt)
            res.check(f"chi-square, V={V}, T={t}, top_k={k}, top_p={p}, {N} draws over {layout}, exp={exp}",
                      d["p_value"] > 1e-3 and d["draws_outside_support"] == 0, level="other", impl="bi_sampling", detail=d)
    # the classical accept/reject verifier must reproduce the target whatever the draft
    row, drow = rows[50], rows[50] + torch.randn(50, generator=gen, device=dev)
    for t, k, p in ((1.0, 0, 1.0), (0.8, 10, 0.9)):
        def fn(x, idx):
            kw = dict(temperature=t, top_k=k, top_p=p)
            dl = drow[None].expand(idx.numel(), -1)
            dt = bs.sample(dl, idx, 7, purpose=bs.DRAFT, **kw)
            return bs.spec_verify_rejection(x, dl, dt, idx, 7, **kw)[0]
        d = chi_square(draw_counts(row, N, None, 1 << 17, fn), target_dist(row.cpu(), t, k, p))
        res.check(f"chi-square, spec_verify_rejection output against the target, V=50, T={t}, top_k={k}, top_p={p}, {N} draws",
                  d["p_value"] > 1e-3 and d["draws_outside_support"] == 0, level="other", impl="bi_sampling", detail=d)
        def fn2(x, idx):
            return bs.spec_verify_exact(x, torch.zeros(idx.numel(), dtype=torch.long), idx, 7, temperature=t, top_k=k, top_p=p)[0]
        d = chi_square(draw_counts(row, N, None, 1 << 18, fn2), target_dist(row.cpu(), t, k, p))
        res.check(f"chi-square, spec_verify_exact output against the target, V=50, T={t}, top_k={k}, top_p={p}, {N} draws",
                  d["p_value"] > 1e-3 and d["draws_outside_support"] == 0, level="other", impl="bi_sampling", detail=d)
    # control: the test must reject a slightly wrong target
    d = chi_square(draw_counts(rows[50], N, None, 1 << 18, lambda x, idx: bs.sample(x, idx, 7, temperature=0.7)), target_dist(rows[50].cpu(), 0.72, 0, 1.0))
    res.check(f"control: chi-square of T=0.7 draws against the T=0.72 target accepts ({N} draws)", d["p_value"] > 1e-3, level="other",
              impl="bi_sampling", expect="fail", detail=d)


# ============================================================================= l2
def ref_fn(cfg=None, exp="torch"):
    def fn(batch):
        seeds, pos, pid = row_keys(batch)
        if cfg is None:
            t, k, p = ([MIXED[i][j] for i in pid.tolist()] for j in range(3))
        else:
            t, k, p = cfg
        info = bs.sample(batch, seeds, pos, temperature=t, top_k=k, top_p=p, stream=pid & 3, exp=exp, return_info=True)
        return torch.stack([info.tokens, info.weight, info.total], 1)
    return fn


def section_l2(res, quick):
    gen = G(7)
    for V in VOCABS:
        sizes = SIZES_QUICK if quick else SIZES
        if quick and V > 100000:
            sizes = (1, 2, 4, 16, 17)
        rows = probe_rows(V, gen, n_random=1 if quick else 2)
        for name, cfg in list(CONFIGS.items()) + [("mixed parameters per row", None)]:
            probe = rows if cfg is None or not quick else rows[-3:]
            rep = check_batch(ref_fn(cfg), probe, sizes, name=name)
            res.check(f"token, weight, total: row vs alone, {name}, V={V}, sizes {sizes[0]}..{sizes[-1]}", rep.passed, level="L2", impl="bi_sampling",
                      detail=brief(rep))
        rep = check_batch(ref_fn(None, "basic"), rows[-3:], sizes if quick else SIZES_STOCK, name="basic")
        res.check(f"token, weight, total: row vs alone, mixed parameters, exp=basic, V={V}", rep.passed, level="L2", impl="bi_sampling", detail=brief(rep))
        x = torch.randn(16 if quick else 64, V, generator=gen, device=dev) * 2
        for name, cfg in (("mixed parameters per row", None), ("top_p=0.9", CONFIGS["top_p=0.9"])):
            rep = check_repeat(ref_fn(cfg), x, n=5, name=name)
            res.check(f"same batch, 5 runs, {name}, V={V}", rep.passed, level="L1", impl="bi_sampling", detail=brief(rep))
        if V > 100000 and not quick:
            big = torch.randn(1025, 32000, generator=gen, device=dev) * 2
            ref = torch.cat([ref_fn(None)(big[i:i + 1]) for i in (0, 512, 1024)])
            ok = all(bit_equal(ref_fn(None)(big[:n])[i], ref[j]) for n in (1024, 1025) for j, i in enumerate((0, 512, 1024)) if i < n)
            res.check("token, weight, total: row vs alone at batch 1024 and 1025, mixed parameters, V=32000", ok, level="L2", impl="bi_sampling")


# ============================================================================= engine
M61 = (1 << 61) - 1


class Toy:
    """Next-token logits are a table row picked by a hash of the prefix: every path that reaches
    the same prefix reads the same logit bits. `state` plays the KV cache."""

    def __init__(self, V, n_ctx=48, seed=0, scale=2.0, draft_noise=0.7):
        g = G(seed)
        self.V, self.n = V, n_ctx
        self.target = torch.randn(n_ctx, V, generator=g, device=dev) * scale
        self.draft = self.target + torch.randn(n_ctx, V, generator=g, device=dev) * draft_noise

    @staticmethod
    def h0(prompt):
        return (prompt * 1000003 + 12345) % M61

    @staticmethod
    def step(h, tok):
        return (h * 1000003 + tok + 1) % M61

    def prefill(self, prompt, toks):
        h = self.h0(prompt)
        for t in toks:
            h = self.step(h, t)
        return h

    def logits(self, hs, which="target"):
        return getattr(self, which)[torch.tensor([h % self.n for h in hs], device=dev)]


class Req:
    def __init__(self, rid, prompt, seed, temperature=1.0, top_k=0, top_p=1.0):
        self.id, self.prompt, self.seed = rid, prompt, seed
        self.params = dict(temperature=temperature, top_k=top_k, top_p=top_p)


class RefSampler:
    impl = "bi_sampling"

    def start(self, req):
        pass

    def sample(self, logits, reqs, positions):
        return bs.sample(logits, [r.seed for r in reqs], positions, temperature=[r.params["temperature"] for r in reqs],
                         top_k=[r.params["top_k"] for r in reqs], top_p=[r.params["top_p"] for r in reqs])


def _stock_probs(logits, reqs):
    t = torch.tensor([r.params["temperature"] for r in reqs], device=dev)[:, None]
    return torch.softmax(logits / t, -1)


class SharedStream:
    impl = "stock_shared_stream"

    def __init__(self):
        self.g = G(1234)

    def start(self, req):
        pass

    def sample(self, logits, reqs, positions):
        return torch.multinomial(_stock_probs(logits, reqs), 1, generator=self.g)[:, 0]


class PerRequestGen:
    """One generator per request. recreate=True re-creates it from the seed each time the
    request enters prefill, as vllm 0.6.3.post1's _prepare_seq_groups does; False keeps it."""

    def __init__(self, recreate=True):
        self.gens, self.recreate = {}, recreate
        self.impl = "stock_per_request_generator" + ("" if recreate else "_kept")

    def start(self, req):
        if self.recreate or req.id not in self.gens:
            self.gens[req.id] = G(req.seed)

    def sample(self, logits, reqs, positions):
        probs = _stock_probs(logits, reqs)
        q = torch.empty_like(probs)
        for i, r in enumerate(reqs):
            q[i:i + 1].exponential_(generator=self.gens[r.id])
        return probs.div_(q).argmax(-1)


def decode_alone(model, sampler, req, T, toks=None):
    toks = list(toks or [])
    sampler.start(req)
    h = model.prefill(req.prompt, toks)
    while len(toks) < T:
        tok = int(sampler.sample(model.logits([h]), [req], [len(toks)])[0])
        toks.append(tok)
        h = model.step(h, tok)
    return toks


def decode_batched(model, sampler, reqs, T, seed, preempt=None, rollback=0):
    """Random schedule: each step a random subset of the requests runs, in a random row order,
    among filler requests. preempt = {id: length}: at that length the request is evicted (state
    dropped, the last `rollback` tokens discarded); it returns a few steps later through
    prefill, and sampler.start() is called again."""
    rng = random.Random(seed)
    st = {r.id: {"h": model.h0(r.prompt), "toks": [], "out": 0, "done": False} for r in reqs}
    for r in reqs:
        sampler.start(r)
    nfill = 0
    while any(len(s["toks"]) < T for s in st.values()):
        runnable = []
        for r in reqs:
            s = st[r.id]
            if len(s["toks"]) >= T:
                continue
            if preempt and r.id in preempt and not s["done"] and len(s["toks"]) == preempt[r.id]:
                s["done"], s["out"], s["h"] = True, rng.randrange(1, 4), None
                del s["toks"][len(s["toks"]) - rollback:]
            if s["out"] > 0:
                s["out"] -= 1
                continue
            if s["h"] is None:
                s["h"] = model.prefill(r.prompt, s["toks"])
                sampler.start(r)
            runnable.append(r)
        pick = [r for r in runnable if rng.random() < 0.7] or runnable[:1]
        if not pick:
            continue
        fill = []
        for _ in range(rng.randrange(0, 5)):
            nfill += 1
            f = Req(f"fill{nfill}", rng.randrange(1 << 30), rng.randrange(1 << 30), *rng.choice(MIXED[:5]))
            if not isinstance(sampler, RefSampler):
                f.params.update(top_k=0, top_p=1.0)
            sampler.start(f)
            fill.append(f)
        rows = pick + fill
        rng.shuffle(rows)
        hs = [st[r.id]["h"] if r.id in st else model.h0(r.prompt) for r in rows]
        toks = sampler.sample(model.logits(hs), rows, [len(st[r.id]["toks"]) if r.id in st else rng.randrange(64) for r in rows]).tolist()
        for r, tok in zip(rows, toks):
            if r.id in st:
                st[r.id]["toks"].append(tok)
                st[r.id]["h"] = model.step(st[r.id]["h"], tok)
    return {r.id: st[r.id]["toks"] for r in reqs}


def decode_spec_ref(model, req, T, k, mode):
    """Speculative decoding with the reference sampler. The draft proposes k tokens; the target
    scores the k (+1) positions in one batched call. Returns (tokens, accepted, proposed)."""
    toks, h, acc_n, prop_n = [], model.h0(req.prompt), 0, 0
    if req.params["temperature"] == 0:
        mode = "exact"                         # a greedy request is verified by equality
    while len(toks) < T:
        t, kk = len(toks), min(k, T - len(toks))
        hd, dctx, dtoks = h, [], []
        for i in range(kk):
            x = int(bs.sample(model.logits([hd], "draft"), req.seed, t + i, purpose=bs.TOKEN if mode == "exact" else bs.DRAFT, **req.params)[0])
            dctx.append(hd)
            dtoks.append(x)
            hd = model.step(hd, x)
        if mode == "exact":
            tok_all, ok = bs.spec_verify_exact(model.logits(dctx + [hd]), torch.tensor(dtoks + [-1]), req.seed, list(range(t, t + kk + 1)), **req.params)
            ok = ok.tolist()
            j = next((i for i in range(kk) if not ok[i]), kk)
            emit = tok_all.tolist()[: j + 1]
        else:
            tok_all, ok = bs.spec_verify_rejection(model.logits(dctx), model.logits(dctx, "draft"), torch.tensor(dtoks), req.seed,
                                                   list(range(t, t + kk)), **req.params)
            ok = ok.tolist()
            j = next((i for i in range(kk) if not ok[i]), kk)
            emit = tok_all.tolist()[: min(j + 1, kk)]
        acc_n, prop_n = acc_n + j, prop_n + kk
        for x in emit[: T - t]:
            toks.append(x)
            h = model.step(h, x)
    return toks, acc_n, prop_n


def decode_spec_stock(model, req, T, k):
    """A per-request generator consumed in the order read from vllm 0.6.3.post1 (seeded request,
    batch-expansion scorer, torch rejection sampler): k draft draws of [1, V], one bonus draw of
    [1, V], torch.rand(1, k) for acceptance, one recovered-token draw of [k, V]."""
    g = G(req.seed)
    temp = req.params["temperature"]

    def draw(probs):
        return (probs / torch.empty_like(probs).exponential_(generator=g)).argmax(-1)
    toks, h = [], model.h0(req.prompt)
    while len(toks) < T:
        kk = min(k, T - len(toks))
        hd, dctx, dtoks, dprobs = h, [], [], []
        for i in range(kk):
            pr = torch.softmax(model.logits([hd], "draft") / temp, -1)
            x = int(draw(pr)[0])
            dctx.append(hd)
            dtoks.append(x)
            dprobs.append(pr)
            hd = model.step(hd, x)
        tp = torch.softmax(model.logits(dctx + [hd]) / temp, -1)
        bonus = int(draw(tp[kk:])[0])
        u = torch.rand(1, kk, device=dev, generator=g)[0]
        dp, dt = torch.cat(dprobs), torch.tensor(dtoks, device=dev)[:, None]
        ok = (u < torch.clamp(tp[:kk].gather(1, dt)[:, 0] / dp.gather(1, dt)[:, 0], max=1.0)).tolist()
        f = torch.clamp(tp[:kk] - dp, min=torch.finfo(torch.float32).tiny)
        rec = draw(f / f.sum(-1, keepdim=True)).tolist()
        j = next((i for i in range(kk) if not ok[i]), kk)
        for x in (dtoks[:j] + ([rec[j]] if j < kk else [bonus]))[: T - len(toks)]:
            toks.append(x)
            h = model.step(h, x)
    return toks


def tt(x):
    return torch.tensor(x, dtype=torch.int64)


def resume_child(args):
    ck = json.load(open(args.child_resume))
    model = Toy(ck["V"], seed=ck["model_seed"])
    req = Req("r", ck["prompt"], ck["seed"], *ck["params"])
    out = {"reference": decode_alone(model, RefSampler(), req, ck["T"], ck["toks"])}
    sreq = Req("r", ck["prompt"], ck["seed"], ck["stock_temperature"])
    out["stock_reseeded"] = decode_alone(model, PerRequestGen(), sreq, ck["T"], ck["stock_toks"])
    s = PerRequestGen()
    s.start(sreq)
    s.gens["r"].set_state(torch.tensor(ck["stock_state"], dtype=torch.uint8))
    s.recreate = False
    out["stock_state_restored"] = decode_alone(model, s, sreq, ck["T"], ck["stock_toks"])
    json.dump(out, open(args.child_out, "w"))


def section_engine(res, quick):
    T = 24 if quick else 64
    for V in (VOCABS[:1] if quick else VOCABS):
        model = Toy(V, seed=11)
        reqs = [Req(f"r{i}", 100 + i, 9000 + 17 * i, *MIXED[i % 8]) for i in range(8)]
        sreqs = [Req(f"r{i}", 100 + i, 9000 + 17 * i, (1.0, 0.7, 1.3)[i % 3]) for i in range(8)]   # temperature only
        ref = RefSampler()
        alone = {r.id: decode_alone(model, ref, r, T) for r in reqs}
        flat = lambda d, rs: tt([d[r.id] for r in rs])  # noqa: E731

        modes = {"alone": lambda: flat(alone, reqs)}
        for s in (1, 2, 3):
            modes[f"schedule {s}"] = lambda s=s: flat(decode_batched(model, ref, reqs, T, s), reqs)
        rep = check_modes(modes, name="batched")
        res.check(f"decode alone vs 3 random batched schedules with fillers, 8 requests x {T} tokens, V={V}", rep.passed, level="L2",
                  impl="bi_sampling", detail=brief(rep))
        pre = {r.id: 5 + 3 * i for i, r in enumerate(reqs)}
        modes = {"alone": lambda: flat(alone, reqs),
                 "preempt, recompute": lambda: flat(decode_batched(model, ref, reqs, T, 4, preempt=pre), reqs),
                 "preempt, roll back 3 tokens": lambda: flat(decode_batched(model, ref, reqs, T, 5, preempt=pre, rollback=3), reqs)}
        rep = check_modes(modes, name="preempt")
        res.check(f"replay after preemption (state recomputed from the tokens; with and without a 3-token rollback) vs uninterrupted, V={V}",
                  rep.passed, level="L3", impl="bi_sampling", detail=brief(rep))
        stats = {}
        for mode in ("exact", "rejection"):
            modes = {"sequential (no draft)": lambda: flat(alone, reqs)} if mode == "exact" else {}
            for k in range(1, 9):
                def run(k=k, mode=mode):
                    out = [decode_spec_ref(model, r, T, k, mode) for r in reqs]
                    stats[f"{mode} k={k}"] = round(sum(o[1] for o in out) / max(1, sum(o[2] for o in out)), 4)
                    return tt([o[0] for o in out])
                modes[f"k={k}"] = run
            rep = check_modes(modes, name=mode)
            what = "sequential decoding without a draft" if mode == "exact" else "k=1"
            res.check(f"speculative verification ({'spec_verify_exact' if mode == 'exact' else 'spec_verify_rejection'}) in chunks of k=1..8 vs {what}, V={V}",
                      rep.passed, level="L3", impl="bi_sampling", detail=brief(rep))
        rej1 = tt([decode_spec_ref(model, r, T, 4, "rejection")[0] for r in reqs])
        res.check(f"spec_verify_rejection tokens equal decoding without a draft, V={V}", bit_equal(rej1, flat(alone, reqs)), level="L3",
                  impl="bi_sampling_rejection", expect="fail", detail={"differing_tokens": int((rej1 != flat(alone, reqs)).sum()), "of": rej1.numel()})
        res.extra[f"draft_acceptance_rate_V{V}"] = stats
        # resume in a new process
        r0, s0 = reqs[4], sreqs[4]
        t0 = T // 3
        sp = PerRequestGen()
        stock_alone = decode_alone(model, sp, s0, T)
        sp2 = PerRequestGen()
        part = decode_alone(model, sp2, s0, t0)
        ck = {"V": V, "model_seed": 11, "prompt": r0.prompt, "seed": r0.seed, "params": [r0.params[k] for k in ("temperature", "top_k", "top_p")],
              "T": T, "toks": alone[r0.id][:t0], "stock_toks": part, "stock_temperature": s0.params["temperature"], "stock_state": sp2.gens[s0.id].get_state().tolist()}
        with tempfile.TemporaryDirectory() as td:
            json.dump(ck, open(os.path.join(td, "ck.json"), "w"))
            subprocess.run([sys.executable, os.path.abspath(__file__), "--child-resume", os.path.join(td, "ck.json"),
                            "--child-out", os.path.join(td, "out.json")], check=True)
            child = json.load(open(os.path.join(td, "out.json")))
        res.check(f"resume of a checkpointed request (prompt, seed, {t0} tokens) in a new process vs uninterrupted, V={V}",
                  child["reference"] == alone[r0.id], level="L3", impl="bi_sampling")
        res.check(f"resume in a new process, generator re-created from the seed, V={V}", child["stock_reseeded"] == stock_alone, level="L3",
                  impl="stock_per_request_generator", expect="fail")
        res.check(f"resume in a new process, generator state saved with get_state and restored, V={V}", child["stock_state_restored"] == stock_alone,
                  level="L3", impl="stock_per_request_generator_kept", expect="info")

        # stock paths through the same engine
        for mk, exp_batched, exp_pre, exp_roll in ((SharedStream, "fail", "fail", "fail"), (lambda: PerRequestGen(True), "info", "fail", "fail"),
                                                   (lambda: PerRequestGen(False), "info", "info", "fail")):
            s_alone = {r.id: decode_alone(model, mk(), r, T) for r in sreqs}
            impl = mk().impl
            for name, kw, e in (("3 random batched schedules", {}, exp_batched), ("preemption, recompute", {"preempt": pre}, exp_pre),
                                ("preemption with a 3-token rollback", {"preempt": pre, "rollback": 3}, exp_roll)):
                seeds = (1, 2, 3) if not kw else (4,)
                modes = {"alone": lambda: flat(s_alone, sreqs)}
                for s in seeds:
                    modes[f"schedule {s}"] = lambda s=s: flat(decode_batched(model, mk(), sreqs, T, s, **kw), sreqs)
                rep = check_modes(modes, name=name)
                res.check(f"decode alone vs {name}, 8 requests x {T} tokens, V={V}", rep.passed, level="L2" if not kw else "L3", impl=impl,
                          expect=e, detail=brief(rep))
        seq = tt([decode_alone(model, PerRequestGen(), r, T) for r in sreqs])
        modes = {"sequential": lambda: seq}
        for k in range(1, 9):
            modes[f"k={k}"] = lambda k=k: tt([decode_spec_stock(model, r, T, k) for r in sreqs])
        rep = check_modes(modes, name="stock spec")
        res.check(f"per-request generator consumed in vLLM's order, speculative chunks k=1..8 vs sequential decoding, V={V}", rep.passed, level="L3",
                  impl="stock_per_request_generator", expect="fail", detail=brief(rep))
        ks = [tt([decode_spec_stock(model, r, T, k) for r in sreqs]) for k in (2, 4, 8)]
        res.check(f"per-request generator consumed in vLLM's order: k=2, 4 and 8 give the same tokens, V={V}", bit_equal(ks[0], ks[1]) and bit_equal(ks[1], ks[2]),
                  level="L3", impl="stock_per_request_generator", expect="fail")
        del model


# ============================================================================= cost
def timed(fn, n=9):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[n // 2]


def section_cost(res, quick):
    gen = G(8)
    res.extra["gpu_processes_before_timing"] = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n")[:-1]
    for V in VOCABS:
        for B in (1, 8, 64, 256):
            x = torch.randn(B, V, generator=gen, device=dev) * 3
            seeds, pos = list(range(B)), [5] * B
            temp = torch.full((B, 1), 0.7, device=dev)
            g = G(1)
            gens = [G(i) for i in range(B)]
            pp, kk = torch.full((B,), 0.9, device=dev), torch.full((B,), 50, device=dev)

            def per_request(probs):
                q = torch.empty_like(probs)
                for i in range(B):
                    q[i:i + 1].exponential_(generator=gens[i])
                return probs.div_(q).argmax(-1)
            runs = {
                "temperature": {
                    "stock_multinomial": lambda: torch.multinomial(torch.softmax(x / temp, -1), 1, generator=g),
                    "stock_per_request_generator": lambda: per_request(torch.softmax(x / temp, -1)),
                    "bi_sampling": lambda: bs.sample(x, seeds, pos, temperature=0.7),
                    "bi_sampling_basic": lambda: bs.sample(x, seeds, pos, temperature=0.7, exp="basic"),
                },
                "top_k=50,top_p=0.9": {
                    "stock_multinomial": lambda: torch.multinomial(torch.softmax(vllm_style_top_k_top_p(x / temp, pp, kk), -1), 1, generator=g),
                    "stock_per_request_generator": lambda: per_request(torch.softmax(vllm_style_top_k_top_p(x / temp, pp, kk), -1)),
                    "bi_sampling": lambda: bs.sample(x, seeds, pos, temperature=0.7, top_k=50, top_p=0.9),
                    "bi_sampling_basic": lambda: bs.sample(x, seeds, pos, temperature=0.7, top_k=50, top_p=0.9, exp="basic"),
                },
                "greedy": {
                    "stock_multinomial": lambda: torch.argmax(x, -1),
                    "bi_sampling": lambda: bs.sample(x, seeds, pos, temperature=0.0),
                },
            }
            for case, impls in runs.items():
                ms = {name: timed(fn) for name, fn in impls.items()}
                for name, v in ms.items():
                    base = "torch.argmax" if case == "greedy" and name == "stock_multinomial" else name
                    res.measure(f"sampling step, {case}, B={B}, V={V}", round(v, 3), "ms", impl=base, B=B, V=V, case=case,
                                x_stock_multinomial=round(v / ms["stock_multinomial"], 2),
                                x_stock_per_request=round(v / ms["stock_per_request_generator"], 2) if "stock_per_request_generator" in ms else None)
            res.measure(f"noise only (uniform53 on the CPU), B={B}", round(timed(lambda: bs.uniform53(seeds, pos, n=B)), 3), "ms", impl="bi_sampling", B=B, V=V,
                        case="noise")
            del x
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(256, 151936, generator=gen, device=dev)
    base = torch.cuda.memory_allocated()
    bs.sample(x, list(range(256)), 5, top_k=50, top_p=0.9)
    res.measure("peak extra memory, top_k=50,top_p=0.9, B=256, V=151936", round((torch.cuda.max_memory_allocated() - base) / 2 ** 30, 2), "GiB", impl="bi_sampling")


SECTIONS = {"rng": section_rng, "ops": section_ops, "ties": section_ties, "nucleus": section_nucleus, "unit": section_unit,
            "chisq": section_chisq, "l2": section_l2, "engine": section_engine, "cost": section_cost}
QUICK = ("unit", "chisq", "l2", "engine", "nucleus")


def merge(paths, out):
    parts = [json.load(open(p)) for p in paths]
    m = {"schema": SCHEMA, "example": parts[0]["example"], "quick": any(p["quick"] for p in parts), "argv": [p["argv"] for p in parts],
         "env": parts[0]["env"], "checks": [c for p in parts for c in p["checks"]],
         "measurements": [x for p in parts for x in p["measurements"]], "extra": {}}
    for p in parts:
        m["extra"].update(p["extra"])
    m["counts"] = {s: sum(c["status"] == s for c in m["checks"]) for s in ("ok", "regression", "changed", "info")}
    json.dump(m, open(out, "w"), indent=1)
    print(f"merged {len(parts)} files into {out}: {m['counts']}")
    return 1 if m["counts"]["regression"] else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="every expect=pass check at reduced sizes")
    ap.add_argument("--sections", help="comma-separated subset of: " + ",".join(SECTIONS))
    ap.add_argument("--json", help="write the results here (bi-results-v1)")
    ap.add_argument("--merge", nargs="+", help="merge result files written by --sections runs into --json")
    ap.add_argument("--child-resume", help=argparse.SUPPRESS)
    ap.add_argument("--child-out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.merge:
        sys.exit(merge(args.merge, args.json))
    if args.child_resume:
        return resume_child(args)
    names = args.sections.split(",") if args.sections else (QUICK if args.quick else list(SECTIONS))
    res = Results("sampling", quick=args.quick)
    res.extra["sections_seconds"] = {}
    for name in names:
        t0 = time.time()
        SECTIONS[name](res, args.quick)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        res.extra["sections_seconds"][name] = round(time.time() - t0, 1)
        print(f"-- section {name}: {res.extra['sections_seconds'][name]} s", flush=True)
    res.write(args.json)
    sys.exit(res.exit_code())


if __name__ == "__main__":
    main()
