# MLA and sparse attention

## Where we are

The [attention page](attention.md) fixed the walk over keys so that every mode
gives the same bits. Two newer attention designs add a problem the walk does
not cover. MLA (latent attention) can be computed in two algebraically equal
forms, and engines often use one form in prefill and the other in decode.
Sparse attention first selects which keys to read, and a selection is a
discrete choice. Both pages' tools still apply: invariant GEMMs and a fixed
walk over keys.

## The idea

### Two equal forms round in different places

MLA caches a small latent vector per token instead of per-head keys and values.
The per-head keys come from it through a matrix W. The score of a query q
against a cached latent c can be computed in two orders:

| Form | Order | What is rounded to bf16 |
|---|---|---|
| explicit | decompress first: k = W·c, then q·k | the per-head keys and values |
| absorbed | fold W into the query first: q' = q·W, then q'·c | the absorbed query and the latent output |

(q·W)·c and q·(W·c) are the same number in exact arithmetic and different
numbers after rounding. Neither form is more accurate; they are different
choices. The rule is the attention rule one level up: **one form for every
mode and for both sides.** An engine that prefills explicitly and decodes
absorbed no longer matches a trainer that runs one form.

### A selection flips on a tiny change

A sparse attention scores the past keys with a small indexer, keeps the top k
and attends over those only. Suppose the k-th and the next score are 0.7001
and 0.7000. If a batch-variant GEMM moves the first score down by a few units
in the last place, the two keys swap, a different key enters the sum, and the
output changes by far more than one rounding step.

So the selection needs three things: invariant scores (every GEMM that feeds
them), a tie order that is written down (score descending, then index
ascending), and a sum over the selected keys in ascending index order rather
than in the order the selection returned them.

## The steps

### 1. Make every GEMM whose M changes with the step invariant

- **Goal:** the projections, the decompression and the per-head absorption see
  a different number of rows in prefill, chunks and decode; each must be
  invariant in M.
- **How:** the one-config GEMM from the [GEMM page](gemm-and-quantized.md), and
  a grouped one-config GEMM for the per-head products.
- **Success:** `check_batch` passes on each piece. Shown on one GPU type: an
  invariant attention core with library GEMMs around it still failed
  ([report](../../examples/mla_sparse/REPORT.md#what-this-shows)).

### 2. Choose one MLA form for every mode

- **Goal:** prompt, generated tokens and trainer all use the same form.
- **How:** an attention core that serves the chosen form with a fixed walk over
  the latent cache, as in [recipe 5](../../skills/batch-invariant-kernel/references/operator-recipes.md#5-mla-latent-attention)
  and [mla_sparse_kernels.py](../../examples/mla_sparse/mla_sparse_kernels.py).
- **Success:** all modes and batches bitwise equal within the form, and an
  engine run equal to a trainer run. Shown on one GPU type, with one and two
  stacked layers ([report](../../examples/mla_sparse/REPORT.md#part-a-mla-absorbed-vs-explicit)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Explicit prefill, absorbed decode (common in engines) | no match with a trainer | free | not for parity |
| Absorbed everywhere | L2, L3, one form on both sides | moderate in prefill | the usual parity choice |
| Explicit everywhere, decompressing the cache every step | L2, L3 | large in decode | short contexts only |
| Explicit everywhere with a per-head key/value cache | L2, L3 | moderate to large in decode, plus a large cache | when memory allows |

### 3. Make the selection stable

- **Goal:** the selected set and the output do not change with the batch or
  the mode.
- **How:** [recipe 6](../../skills/batch-invariant-kernel/references/operator-recipes.md#6-sparse-attention-with-selection-indexer-plus-top-k):
  invariant projection and indexer GEMMs, a fixed head order in the indexer
  sum, a stable top-k by (score descending, index ascending), the selected
  indices sorted ascending, and attention that visits them in that order.
- **Success:** no position changes its selected set, then no output element
  changes. Shown on one GPU type
  ([report](../../examples/mla_sparse/REPORT.md#part-b-sparse-attention-with-selection-dsa-like)).

| Piece | Option | Cost class |
|---|---|---|
| Projection GEMM | one-config | free to small; it removed every flip measured |
| Indexer | fixed-order fp32 sum over heads | free (faster than an unfused baseline) |
| Top-k | stable sort by (score, index) | small for the layer; the top-k step itself is slower than the library's |
| Sparse attention | ascending index order, fixed blocks | free |

### 4. Verify sets before values

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md),
and for sparse attention compare the selected index sets first: a value check
mixes discrete flips with rounding drift. Test whole decodes in a batch, not
only one step, and test at least two stacked layers.

## What can go wrong

!!! warning "One layer hides the mixed-form error"
    With one layer the latent cache does not depend on the form, so a trainer
    in the absorbed form can match by accident. From layer 2 the cache comes
    from the other form's outputs; test two or more layers.

!!! warning "The attention core alone is not enough"
    The decompression GEMM (its M is the number of cached tokens) and the
    per-head output absorption are easy to miss. Leave either on the library
    and the step-dependent M breaks L2 and L3.

!!! warning "The library's top-k tie order is not a contract"
    In every test it picked the lowest index among ties, but that order is not
    documented. Write the tie order down with a stable sort.

!!! warning "Most flips come from the scores, not the ties"
    A stable top-k did not remove a single flip while the projection was
    batch-variant. Fix the GEMMs before the top-k.

## Summary

- Absorbed and explicit MLA are equally accurate and bitwise different: pick
  one form for every mode and both sides.
- Every GEMM whose row count changes with the step must be invariant,
  including the decompression and the per-head absorption.
- A selection flips on a tiny score change: invariant scores, a stable
  (score, index) top-k, sorted accumulation.
- Compare selected sets before values, and test two layers.

## Go deeper

- Report: [MLA and sparse attention](../../examples/mla_sparse/REPORT.md).
- Code: [examples/mla_sparse/mla_sparse.py](../../examples/mla_sparse/mla_sparse.py),
  [mla_sparse_kernels.py](../../examples/mla_sparse/mla_sparse_kernels.py).
- Recipes: [5, MLA](../../skills/batch-invariant-kernel/references/operator-recipes.md#5-mla-latent-attention),
  [6, sparse attention](../../skills/batch-invariant-kernel/references/operator-recipes.md#6-sparse-attention-with-selection-indexer-plus-top-k).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
