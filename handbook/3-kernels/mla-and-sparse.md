# MLA and sparse attention

## Where we are

The [attention page](attention.md) fixed the walk over the keys so that every
mode gives the same bits. Two newer attention designs add problems that a
fixed walk does not solve. MLA (multi-head latent attention) can be computed
in two algebraically equal forms, and engines often use one form in prefill
and the other in decode. Sparse attention first selects which keys to read,
and that selection is a discrete choice. The tools from the last two pages
still apply: invariant GEMMs and a fixed walk over the keys.

## The idea

### The same math in two forms rounds in different places

MLA caches one small latent vector per token instead of separate keys and
values for each head. The keys for each head are computed from that vector
through a matrix W. The score of a query q against a cached latent c can then
be computed in two orders:

| Form | Order of operations | What is rounded to bf16 |
|---|---|---|
| explicit | decompress first, k = W·c, then compute q·k | the keys and values of each head |
| absorbed | fold W into the query first, q' = q·W, then compute q'·c | the absorbed query and the latent output |

(q·W)·c and q·(W·c) are the same number in exact arithmetic but different
numbers after rounding. Neither form is more accurate; they are simply two
different choices. The rule is the attention rule applied one level higher:
**use one form in every mode and on both sides.** An engine that runs prefill
in the explicit form and decode in the absorbed form no longer matches a
trainer that runs only one form.

### A tiny change in a score can change which keys are selected

A sparse attention layer scores the past keys with a small indexer, keeps the
top k and attends only over those. Suppose the k-th score and the next one
are 0.7001 and 0.7000. If a GEMM that varies with the batch lowers the first
score by a few units in the last place, the two keys swap places. A different
key then enters the sum, and the output changes by far more than one rounding
step.

So the selection needs three things: scores that are invariant, which means
that every GEMM feeding them is invariant; a tie order that is written down
(score descending, then index ascending); and a sum over the selected keys in
ascending index order, rather than in the order in which the selection
returned them.

## The steps

### 1. Make every GEMM invariant, especially those whose row count changes between steps

- **Goal:** the projections, the decompression and the per-head absorption
  see a different number of rows in prefill, in chunks and in decode, so each
  of them must be invariant in M.
- **How:** use the one-configuration GEMM from the
  [GEMM page](gemm-and-quantized.md), and a grouped one-configuration GEMM for
  the per-head products.
- **Success:** `check_batch` passes on each piece. This was shown on one GPU
  type, where an invariant attention core surrounded by library GEMMs still
  failed ([report](../../examples/mla_sparse/REPORT.md#what-this-shows)).

### 2. Use one MLA form in every mode and on both sides

- **Goal:** the prompt, the generated tokens and the trainer all use the same
  form.
- **How:** use an attention core that computes the chosen form with a fixed
  walk over the latent cache, as in
  [recipe 5](../../skills/batch-invariant-kernel/references/operator-recipes.md#5-mla-latent-attention)
  and [mla_sparse_kernels.py](../../examples/mla_sparse/mla_sparse_kernels.py).
- **Success:** within the chosen form, all modes and batches are bitwise
  equal, and an engine run equals a trainer run. This was shown on one GPU
  type, with one layer and with two stacked layers
  ([report](../../examples/mla_sparse/REPORT.md#part-a-mla-absorbed-vs-explicit)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Explicit form in prefill, absorbed form in decode (common in engines) | no match with a trainer | free | not when you need parity |
| Absorbed form everywhere | L2 and L3, with one form on both sides | moderate in prefill | the usual choice for parity |
| Explicit form everywhere, decompressing the cache at every step | L2, L3 | large in decode | short contexts only |
| Explicit form everywhere, with a key and value cache for each head | L2, L3 | moderate to large in decode, plus a large cache | when memory allows |

### 3. Make the selection of keys stable

- **Goal:** the set of selected keys and the output do not change with the
  batch or the mode.
- **How:** follow
  [recipe 6](../../skills/batch-invariant-kernel/references/operator-recipes.md#6-sparse-attention-with-selection-indexer-plus-top-k).
  Make the projection and indexer GEMMs invariant, sum over the indexer's
  heads in a fixed order, use a stable top-k ordered by score descending and
  then index ascending, sort the selected indices in ascending order, and let
  attention visit them in that order.
- **Success:** first, no position changes its selected set; then, no output
  element changes. This was shown on one GPU type
  ([report](../../examples/mla_sparse/REPORT.md#part-b-sparse-attention-with-selection-dsa-like)).

| Piece | Option | Cost class |
|---|---|---|
| Projection GEMM | one configuration | free to small; it removed every flip that was measured |
| Indexer | an fp32 sum over the heads in a fixed order | free (faster than an unfused baseline) |
| Top-k | a stable sort by (score, index) | small for the layer as a whole, although the top-k step itself is slower than the library's |
| Sparse attention | ascending index order, fixed blocks | free |

### 4. Compare the selected sets first, then the values

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md).
For sparse attention, compare the selected index sets first, because a check
on values mixes discrete flips with ordinary rounding drift. Test whole
decodes in a batch rather than a single step, and test at least two stacked
layers.

## What can go wrong

!!! warning "A single layer hides the error of mixing forms"
    With one layer, the latent cache does not depend on the form, so a
    trainer that uses the absorbed form can match by accident. From layer 2
    on, the cache is built from the outputs of the other form, so test two or
    more layers.

!!! warning "Fixing the attention core alone is not enough"
    The decompression GEMM, whose M is the number of cached tokens, and the
    per-head absorption of the output are easy to miss. If either one stays
    on the library, its M changes from step to step and breaks L2 and L3.

!!! warning "The library's order for ties in top-k is not guaranteed"
    In every test it picked the lowest index among tied scores, but that
    order is not documented. Write the tie order down by using a stable sort.

!!! warning "Most flips come from the scores, not from ties"
    While the projection still varied with the batch, a stable top-k did not
    remove a single flip. Fix the GEMMs before the top-k.

## Summary

- Absorbed and explicit MLA are equally accurate but bitwise different, so
  pick one form for every mode and for both sides.
- Every GEMM whose row count changes from step to step must be invariant,
  including the decompression and the per-head absorption.
- A tiny change in a score can flip the selection, so use invariant scores, a
  stable top-k ordered by (score, index), and accumulation in sorted order.
- Compare the selected sets before the values, and test with two layers.

## Go deeper

- Report: [MLA and sparse attention](../../examples/mla_sparse/REPORT.md).
- Code: [examples/mla_sparse/mla_sparse.py](../../examples/mla_sparse/mla_sparse.py),
  [mla_sparse_kernels.py](../../examples/mla_sparse/mla_sparse_kernels.py).
- Recipes: [5, MLA](../../skills/batch-invariant-kernel/references/operator-recipes.md#5-mla-latent-attention),
  [6, sparse attention](../../skills/batch-invariant-kernel/references/operator-recipes.md#6-sparse-attention-with-selection-indexer-plus-top-k).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
