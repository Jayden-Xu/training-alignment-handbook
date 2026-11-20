# 3. Batch-invariant kernels

## Where we are

Part 1 ends when you have found the first divergence between two runs, and
part 2 ends with an audit of a whole model. Either one can leave you with a
single operator that has no batch-invariant implementation yet: a new
attention variant, an MoE combine, a quantized GEMM or a collective. This part
shows how to handle that one operator. You start with an operator whose result
changes with the batch, or that nobody has checked, and you finish with one
that has been verified invariant at a stated
[level](../0-foundations/README.md), or with a fallback that is written down.

<figure class="fx" data-fig="kernel-workflow" aria-label="The five steps for making one kernel batch-invariant" markdown>
<div class="fx-stage" markdown>
<div class="fx-seq" style="--n:5" markdown="span"><span class="fx-step" data-hot="0 5">[1. Find what varies](#1-locate-the-variance)</span><span class="fx-step" data-hot="1 5">[2. Choose the cheapest fix](#2-choose-the-cheapest-strategy-that-holds)</span><span class="fx-step" data-hot="2 5">[3. Round in the same places](#3-keep-the-rounding-points)</span><span class="fx-step" data-hot="3 6">[4. Verify bitwise](#4-verify-the-matrix)<span class="fx-swap"><span class="fx-note fx-fail" data-at="3" data-to="5">one case fails</span><span class="fx-note fx-pass" data-at="6">no mismatches</span></span></span><span class="fx-step" data-hot="7">[5. Put it into use](#5-integrate)</span><span class="fx-loop" style="grid-column: 1 / 5; --k: 4" data-hot="4">If step 4 fails, go back to step 1.</span></div>
</div>
<ol class="fx-say">
<li>1. Find out which choice inside the operator depends on something other than the length of the element's own sum.</li>
<li>2. Choose the cheapest strategy that removes that dependence.</li>
<li>3. Make the kernel round in the same places as the side it has to match.</li>
<li>4. Verify bitwise over the whole matrix of batch sizes, positions, modes, layouts and parallel degrees.</li>
<li data-hold="1.5">If one case fails, go back to step 1 with the pattern that failed.</li>
<li data-hold="1.5">Find the cause again, choose again, and keep the same rounding points.</li>
<li>4. Verify again until there are no mismatches.</li>
<li data-hold="2">5. Make the verified kernel the one that actually runs, and record what it costs.</li>
</ol>
<figcaption>Figure: the five steps for one operator. Step 4 asks for evidence rather than an argument from reading the code, and any failure sends you back to step 1.</figcaption>
</figure>

The agent workflow that carries out these steps is the
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
The pages after this one apply it to six classes of operator, from the
simplest to the hardest.

## The idea: the order of a sum is part of its result

Floating-point addition rounds after every add, so the way the terms are
grouped changes the answer. In fp32, 1e8 + 1 rounds back to 1e8, because the
nearest fp32 number above 1e8 is 8 larger:

<figure class="fx" data-fig="two-groupings" aria-label="The same three numbers added in two groupings">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label fx-mono">(1e8 + 1) − 1e8</span><div class="fx-chain is-tight is-grid"><span class="fx-swap" data-at="1"><span class="fx-node is-mono" data-at="1" data-to="2">1e8 + 1 = 100000001</span><span class="fx-node is-mono is-warn" data-at="2">1e8 + 1 → 1e8<span class="fx-note">rounded, since the next number is 8 away</span></span></span><span class="fx-arrow" data-at="3" aria-hidden="true"></span><span class="fx-node is-mono" data-at="3">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="4" aria-hidden="true"></span><span class="fx-node is-mono is-key" data-at="4">0</span></div></div>
<div class="fx-row"><span class="fx-label fx-mono">(1e8 − 1e8) + 1</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="3" aria-hidden="true"></span><span class="fx-node is-mono" data-at="3">0 + 1 = 1</span><span class="fx-arrow" data-at="4" aria-hidden="true"></span><span class="fx-node is-mono is-key" data-at="4">1</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>Take the same three fp32 numbers, 1e8, 1 and −1e8, and group them in two ways.</li>
<li>Each grouping first adds its first two numbers.</li>
<li data-hold="1.5">1e8 + 1 rounds back to 1e8, because the next number after 1e8 is 8 away.</li>
<li>Then each grouping adds the number that is left.</li>
<li data-hold="2">The same three numbers give two answers, 0 and 1.</li>
</ol>
<figcaption>Figure: floating-point addition rounds after every add, so the grouping changes the answer.</figcaption>
</figure>

A kernel has to choose a grouping like this for every output element, and it
often bases the choice on the shape of the whole call. Take one row of a
matrix product. When the row is computed alone, the batch has a single row,
so the library may split the row's long dot product into four pieces, run
them on four programs and add the four partial sums at the end (this is
split-K). In a batch of 64 rows there is enough work to keep the GPU busy
without splitting, so the library computes the same dot product in one pass.
The math for the row is the same in both cases, but the grouping is not, so
the last bits of the result differ.

**The rule.** The order of a reduction may depend only on the length of the
element's own sum. For the sum that produces one output element, the order,
the grouping, the instructions and the precision may depend on how many terms
the sum has, and on nothing else. They must not depend on the number of rows
in the batch, the row's position, the values in other rows, the grid size or
the number of GPU cores, a page or chunk boundary chosen by a scheduler, or
timing. This is the familiar idea of a pure function, applied to the
reduction tree: the tree is a function of the element's own inputs and of
nothing more.

**Determinism is not batch invariance.** A library GEMM returns the same bits
every time you run the same batch, so it is deterministic (L1). It can still
change a row's bits when the batch grows, because it picks its kernel from the
batch size, so it is not batch-invariant (L2). Each level has its own typical
cause of failure: float atomics break L1, choices made from the shape break
L2, and a different kernel for each serving mode breaks L3 (measured: see
[report](../../examples/kernel_cases/REPORT.md#stock-ops-determinism-is-not-batch-invariance)).

## The steps

### 1. Find what makes the result vary {#1-locate-the-variance}

- **Goal:** find the choice inside the operator that depends on something
  other than the length of the element's own sum.
- **How:** run `bi_check.check_batch` from
  [tools/bi_check.py](../../tools/bi_check.py), and also `check_modes` if the
  operator behaves like attention. Then read the code that launches the kernel
  and picks its configuration. If you are starting from a whole model,
  [tools/bi_audit.py](../../tools/bi_audit.py) names the first module whose
  output changes with the batch.
- **Success:** you have a failing pattern and one mechanism that explains it.

| Where the variance hides | Example |
|---|---|
| The library picks a configuration from the number of rows | a tile size, split-K, a fast path for small M |
| The kernel splits a reduction when there is little work | split-KV decode, one row split across several blocks |
| Values are accumulated in the order they arrive | float atomics, scatter-add |
| A statistic is computed over the whole batch | a dynamic quantization scale, an MoE capacity limit |
| The memory layout decides the grouping | KV pages, packed sequences, message chunks |
| The same math is written in two forms | absorbed and explicit MLA, chunked and recurrent linear attention |
| A layer above the kernel changes what runs | compiler fusion, one CUDA graph for each padded size |

The [verification reference](../../skills/batch-invariant-kernel/references/verification.md#reading-failures)
lists the common failure patterns and the mechanism behind each one.

### 2. Choose the cheapest fix that removes it {#2-choose-the-cheapest-strategy-that-holds}

- **Goal:** remove the dependence with the smallest change and the lowest
  cost.
- **How:** try the options in the order of the table below. The
  [skill](../../skills/batch-invariant-kernel/SKILL.md#2-choose-the-cheapest-strategy-that-holds)
  explains each one in detail.
- **Success:** you have a candidate kernel, which step 4 then verifies.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Configure the library | a setting fixes the order, for example a single split, split-K turned off or a pinned algorithm | free to large, depending on the operator | the library offers such a setting |
| Use an existing kernel | a batch-invariant op that someone has already written | small to moderate | one exists for your operator; verify it again on your own shapes |
| Pin one configuration | the same tile setup for every batch size | small to moderate; the GPU is less busy at small batches | the dependence comes from a choice made from the shape |
| Change the kernel | a reduction tree that you write down and fix | moderate | there is no such setting; follow the recipe for the operator's class |
| Use one form everywhere | the same algorithm in every mode and on both sides | moderate to large | the math has two algebraically equal forms |
| Fall back | replace the kernel, or keep a slow row-by-row oracle (a reference version) for debugging | large | the kernel's source is closed; leave it out of any parity claim |

### 3. Round in the same places as the side you must match {#3-keep-the-rounding-points}

- **Goal:** make the invariant kernel round in the same places as the side it
  has to match. That means the same casts, the same fusion, the same
  accumulator dtype and the same order of operations in the epilogue.
- **How:** before you edit anything, write down in order where the reference
  rounds; see [rounding points](../../skills/engine-parity/references/rounding-points.md).
- **Success:** on a batch where the old kernel did not yet vary, the new
  kernel is bitwise equal to it; otherwise both sides change together.

A kernel can be batch-invariant and still round in one place differently
from the other engine. It then passes L2 and fails L4
([whole model](../2-rl/whole-model.md)).

### 4. Verify bitwise across the whole test matrix {#4-verify-the-matrix}

- **Goal:** get evidence that the kernel is invariant, rather than an argument
  from reading the code.
- **How:** run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md).
  It varies the batch size around every threshold, the row's position, the
  content of the other rows in the batch (random, zeros, large values), the
  mode, the layout, repeated runs, the backward pass, the parallel degree,
  deployment under a compiler and CUDA graphs, and the engine.
- **Success:** there are no mismatches. Compare bitwise through integer views,
  so that `-0.0` and `+0.0` count as different, and allow no tolerance.

### 5. Put the verified kernel into the real code path {#5-integrate}

- **Goal:** make sure the verified kernel is the one that actually runs, and
  know what it costs.
- **How:** register it as an opaque custom op, so that a compiler cannot fuse
  it with its neighbours again, and put it behind a flag. Measure throughput
  at prefill and decode batch sizes and compare it with the baseline. Then
  write the report entry that the
  [skill](../../skills/batch-invariant-kernel/SKILL.md#6-report) asks for.
- **Success:** the deployed code path passes the same matrix, and the cost is
  recorded.

## The operator classes, from the simplest to the hardest

Each of the following pages builds on the one before it. The detailed recipe
for each class, including kernel parameters, is in the
[operator recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md).

| Page | What varies, in plain terms | What fixes it | Level |
|---|---|---|---|
| [GEMM and quantized GEMM](gemm-and-quantized.md) | Each element is one dot product. The library decides how to split it from the batch size, and a scale computed over the batch ties the rows together. | Use one configuration for every batch size, and per-token or static scales. | L2 |
| [Softmax attention](attention.md) | A second reduction runs over the keys, and prefill, decode, paging and packing each walk it differently. | Walk the keys in fixed blocks that start at absolute position 0, with the same kernel in every mode, and write a backward pass without atomics. | L2, L3 |
| [MLA and sparse attention](mla-and-sparse.md) | Two algebraically equal forms round in different places, and a tiny change in a score can change which keys are selected. | Use one form everywhere, make the scores invariant, and use a stable top-k. | L2, L3 |
| [Linear attention](linear-attention.md) | A state is carried from token to token, and the result depends on where the chunk boundaries fall. | Use the recurrent form everywhere, or put the chunk boundaries at absolute positions. | L3, L4 |
| [MoE](moe.md) | The result depends on which rows each expert sees, on the order in which the combine adds the experts' outputs, and on whether tokens are dropped. | Make the expert GEMMs invariant, sum each token's terms from fixed slots in a fixed order, and route without dropping. | L2, L5 |
| [Collectives](collectives.md) | The library chooses the order of a sum across GPUs, and the parallel degree changes the partial sums. | Pin the reduction tree or fall back to a fixed order, and compute with a fixed logical TP size. | L2, L5 |

Other parts of the handbook cover the remaining classes: row-wise reductions
([recipe 1](../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy)),
[LoRA](../2-rl/lora.md), [sampling](../2-rl/sampling.md), the
[training step](../1-pretraining/training-step.md) and
[data-parallel gradients](../1-pretraining/data-parallel.md). How these
kernels come together in one model is covered in
[whole model](../2-rl/whole-model.md) and [parallelism](../2-rl/parallelism.md).

## What can go wrong

!!! warning "A deterministic mode does not make a kernel batch-invariant"
    Deterministic settings remove atomics and races, and that gives you L1
    only. The row still changes when the batch changes, so verify L2
    separately.

!!! warning "A sum of two terms hides order bugs"
    a + b equals b + a exactly, so a test with two terms passes whatever the
    order. Test with three or more terms, for example a top-k of at least 3
    or three or more ranks.

!!! warning "Uniform filler rows hide coupling between rows"
    If the other rows in the batch are zeros or copies, a statistic over the
    batch does not change. Fill them with random and large values as well.

!!! warning "A kernel can be consistent within one layout and still vary across layouts"
    If a kernel's blocks follow the page size or the offset of a packed
    sequence, it gives the same result at any one setting and a different
    result when the setting changes. Vary the layout in your tests.

!!! warning "Invariance holds only for what you tested"
    A pass covers the sizes, the GPU and the software versions you ran. List
    them, and include sizes on both sides of every threshold you find in the
    code.

## Summary

- The order of a reduction must depend only on the length of the element's
  own sum.
- Find the dependence, choose the cheapest strategy that removes it, round
  in the same places as the side you match, verify bitwise on the full
  matrix, and only then integrate the kernel.
- L1 does not imply L2, and L2 and L3 together do not imply L4.
- Cost is part of the result, so record it next to the baseline.

## Go deeper

- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md),
  with the [operator recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md),
  their [evidence table](../../skills/batch-invariant-kernel/references/operator-recipes.md#evidence-behind-each-recipe)
  and the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md).
- Tools: [tools/bi_check.py](../../tools/bi_check.py) (the checker),
  [tools/bi_kernels.py](../../tools/bi_kernels.py) (reference kernels) and
  [tools/bi_audit.py](../../tools/bi_audit.py) (a verdict for each module).
- Report: [kernel cases](../../examples/kernel_cases/REPORT.md), which
  measures the stock operators, the attention forward pass, FP8 scales, and
  plain linear attention in its chunked and recurrent forms.
