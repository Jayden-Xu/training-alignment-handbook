# 3. Batch-invariant kernels: making one operator invariant

## Where we are

Part 1 ends at a first divergence and part 2 ends at an audit. Both can name
one operator that has no batch-invariant implementation: a new attention
variant, an MoE combine, a quantized GEMM, a collective. This part is the
workflow for that one operator. It takes the operator from "variant or
unknown" to "verified invariant at a stated level" (see the
[levels](../0-foundations/README.md)), or to a documented fallback.

<figure class="fig">
<div class="fig-chain"><span class="fig-node">1. Locate the variance</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">2. Choose the cheapest strategy</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">3. Keep the rounding points</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">4. Verify the matrix</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">5. Integrate</span></div>
<div class="fig-back">If step 4 fails, go back to step 1.</div>
</figure>

The agent workflow that applies these steps is the
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
The class pages after this one apply it to six operator classes, from the
simplest to the hardest.

## The idea: the order of a sum is part of its result

Floating-point addition rounds after every add, so the grouping changes the
answer. In fp32, 1e8 + 1 rounds back to 1e8, because the next number after
1e8 is 8 away:

| Grouping | Steps | Result |
|---|---|---|
| (1e8 + 1) − 1e8 | 1e8 + 1 rounds to 1e8; then 1e8 − 1e8 | 0 |
| (1e8 − 1e8) + 1 | 0; then 0 + 1 | 1 |

Same three numbers, two answers. A kernel makes this choice for every output
element, and it often makes it from the shape of the whole call. Take one row
of a matrix product. Alone, the batch has one row, so the library may split
the row's long dot product into four pieces on four programs and add the four
partial sums (split-K). In a batch of 64 rows there is enough work without
splitting, so it walks the dot product in one pass. The row's math is the same;
its grouping is not, so its last bits differ.

**The rule.** A reduction's order depends only on the element's own length.
The order, the grouping, the instruction and the precision of the sum that
produces an output element may depend on how long that sum is, and on nothing
else: not on the number of rows, the row's position, the other rows' values,
the grid size or GPU core count, a page or chunk boundary set by a scheduler,
or timing. This is the general principle of a pure function applied to the
reduction tree: the tree is a function of the element's own inputs.

**Determinism is not batch invariance.** A library GEMM gives the same bits on
every run of the same batch (L1), and still changes a row's bits when the
batch grows (L2), because it picks its kernel from the batch size. Atomics
break L1; shape-dependent choices break L2; a different kernel per serving mode
breaks L3 (measured: see [report](../../examples/kernel_cases/REPORT.md#stock-ops-determinism-is-not-batch-invariance)).

## The steps

### 1. Locate the variance

- **Goal:** know which choice inside the operator depends on something other
  than the element's own length.
- **How:** run `bi_check.check_batch` (and `check_modes` for attention-like
  operators) from [tools/bi_check.py](../../tools/bi_check.py); read the
  kernel's launch and dispatch code. For a whole model,
  [tools/bi_audit.py](../../tools/bi_audit.py) names the first variant module.
- **Success:** a failing pattern you can explain by one mechanism.

| Where variance hides | Example |
|---|---|
| A configuration chosen from the number of rows | tile size, split-K, a small-M fast path |
| A reduction split by the work available | split-KV decode, a row split across blocks |
| Accumulation in arrival order | float atomics, scatter-add |
| A statistic over the whole batch | a dynamic quantization scale, an MoE capacity limit |
| A grouping set by the layout | KV pages, packed sequences, message chunks |
| Two forms of the same math | absorbed and explicit MLA, chunked and recurrent linear attention |
| A layer above the kernel | compiler fusion, one CUDA graph per padded size |

The [verification reference](../../skills/batch-invariant-kernel/references/verification.md#reading-failures)
maps failure patterns to mechanisms.

### 2. Choose the cheapest strategy that holds

- **Goal:** remove the dependence with the least change and cost.
- **How:** try the options in this order; the
  [skill](../../skills/batch-invariant-kernel/SKILL.md#2-choose-the-cheapest-strategy-that-holds)
  details each one.
- **Success:** a candidate that you then verify in step 4.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Configure | a library knob fixes the order: one split, split-K off, a pinned algorithm | free to large, by operator | the library exposes the knob |
| Select | a ready-made batch-invariant op | small to moderate | one exists for your operator; re-verify on your shapes |
| Pin one configuration | one tile setup for every batch size | small to moderate; lower utilization at small batches | the dependence is a shape-keyed choice |
| Modify the kernel | a fixed reduction tree written down | moderate | no knob exists; follow the class recipe |
| One form everywhere | one algorithm in every mode and on both sides | moderate to large | two algebraically equal forms exist |
| Fall back | replace the kernel, or a slow row-by-row oracle for debugging | large | the kernel is closed; exclude it from parity claims |

### 3. Keep the rounding points

- **Goal:** the invariant kernel rounds at the same places as the side it must
  match: the same casts, fusion, accumulator dtype and epilogue order.
- **How:** write down the reference rounding sequence before editing; see
  [rounding points](../../skills/engine-parity/references/rounding-points.md).
- **Success:** the new kernel is bitwise equal to the old one on a batch where
  the old one was not yet variant, or both sides change together.

A kernel that is batch-invariant but rounds once differently from the other
engine passes L2 and still fails L4 ([whole model](../2-rl/whole-model.md)).

### 4. Verify the matrix

- **Goal:** evidence, not an argument from the code.
- **How:** the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md):
  batch sizes around every threshold, row positions, co-batched content
  (random, zeros, large), modes, layouts, repeats, backward, parallel degrees,
  compiled and graphed deployment, cross-engine.
- **Success:** zero mismatches, compared bitwise through integer views
  (`-0.0` is not `+0.0`), with no tolerance.

### 5. Integrate

- **Goal:** the verified kernel is what actually runs, and its cost is known.
- **How:** register it as an opaque custom op so a compiler cannot re-fuse it;
  gate it behind a flag; measure throughput at prefill and decode batch sizes
  next to the baseline; write the report entry the
  [skill](../../skills/batch-invariant-kernel/SKILL.md#6-report) asks for.
- **Success:** the deployed path passes the same matrix, and the cost is
  recorded.

## The operator classes, simplest to hardest

Each page opens from the one before it. The detailed recipe for each class,
with kernel parameters, is in the
[operator recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md).

| Page | What varies, by intuition | What fixes it | Level |
|---|---|---|---|
| [GEMM and quantized GEMM](gemm-and-quantized.md) | one dot product per element; the library picks its split from the batch; a batch-wide scale couples rows | one configuration for every batch size; per-token or static scales | L2 |
| [Softmax attention](attention.md) | a second reduction over keys, walked differently in prefill, decode, pages and packs | fixed key blocks from absolute position 0, the same kernel in every mode; an atomic-free backward | L2, L3 |
| [MLA and sparse attention](mla-and-sparse.md) | two equal forms round in different places; a selection flips on a tiny score change | one form everywhere; invariant scores and a stable top-k | L2, L3 |
| [Linear attention](linear-attention.md) | a carried state; where chunk boundaries fall | recurrent everywhere, or chunks at absolute boundaries | L3, L4 |
| [MoE](moe.md) | which rows an expert sees; the order of the combine; dropping | invariant expert GEMMs, a per-token sum in a fixed order from fixed slots, dropless routing | L2, L5 |
| [Collectives](collectives.md) | the library chooses the order of a cross-GPU sum; the parallel degree changes the partials | a pinned tree or a fixed-order fallback; a logical TP size | L2, L5 |

Classes taught in other parts: row-wise reductions
([recipe 1](../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy)),
[LoRA](../2-rl/lora.md), [sampling](../2-rl/sampling.md), the
[training step](../1-pretraining/training-step.md) and
[data-parallel gradients](../1-pretraining/data-parallel.md). The kernels
assembled into one model are in [whole model](../2-rl/whole-model.md) and
[parallelism](../2-rl/parallelism.md).

## What can go wrong

!!! warning "A deterministic mode is not batch invariance"
    Deterministic settings remove atomics and races, which gives L1 only. The
    row still changes with the batch, so verify L2 separately.

!!! warning "Two-term sums hide order bugs"
    a + b equals b + a exactly, so a test with two terms passes any order. Test
    with three or more terms: top-k of at least 3, three or more ranks.

!!! warning "Uniform fillers hide coupling"
    Zero or copied co-batched rows leave batch-wide statistics unchanged. Use
    random and large fillers as well.

!!! warning "A layout that agrees with itself can still be wrong"
    A kernel whose blocks follow the page size or the pack offset is consistent
    at any one setting and changes between settings. Vary the layout under test.

!!! warning "Invariance is a claim about what you tested"
    A pass covers the sizes, GPU and versions you ran. List them, and include
    sizes around every threshold found in the code.

## Summary

- A reduction's order must depend only on the element's own length.
- Locate the dependence, choose the cheapest strategy that removes it, keep
  the rounding points, verify bitwise on the full matrix, then integrate.
- L1 does not imply L2, and L2 and L3 do not imply L4.
- Cost is part of the result: record it next to the baseline.

## Go deeper

- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md),
  with the [operator recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md),
  their [evidence table](../../skills/batch-invariant-kernel/references/operator-recipes.md#evidence-behind-each-recipe)
  and the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md).
- Tools: [tools/bi_check.py](../../tools/bi_check.py) (the checker),
  [tools/bi_kernels.py](../../tools/bi_kernels.py) (reference kernels),
  [tools/bi_audit.py](../../tools/bi_audit.py) (per-module verdicts).
- Report: [kernel cases](../../examples/kernel_cases/REPORT.md), with the stock
  operators measured, the attention forward, FP8 scales and the plain
  linear-attention comparison.
