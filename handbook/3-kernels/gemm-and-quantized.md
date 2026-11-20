# GEMM and quantized GEMM

## Where we are

The [workflow](README.md) is: locate the variance, choose the cheapest
strategy, keep the rounding points, verify, integrate. GEMM is the simplest
place to apply it. Each output element is one dot product of a fixed length K,
and nothing else in the operator reduces. Every later page builds on an
invariant GEMM: attention projections, MLA's per-head products, MoE experts and
tensor-parallel layers all contain one.

## The idea: the library chooses the split from the batch

An output element C[m, n] is the sum of K products. A GEMM library tiles the
output and decides, per call, how to walk K. That decision is keyed on the
problem shape, and the shape includes M, the number of rows in the batch.

<figure class="fig">
<div class="fig-rows">
<div class="fig-group"><span class="fig-label">batch of 1 row: little work</span><div class="fig-chain"><span class="fig-node">K split in 4 pieces on 4 programs</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">add 4 partial sums</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node is-key">row 0: value A</span></div></div>
<div class="fig-group"><span class="fig-label">batch of 64 rows: enough work</span><div class="fig-chain"><span class="fig-node">one program walks all of K in order</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node is-bad">row 0: value B ≠ A</span></div></div>
</div>
</figure>

With a small batch the library may split K to keep the GPU busy (split-K or
stream-K), or call a special small-M kernel. With a large batch it walks K in
one pass. Each choice is deterministic, so the same batch gives the same bits
on every run (L1). But the same row in another batch is summed in another
grouping (L2 fails).

**The rule for GEMM.** One tile configuration for every M, no split-K or
stream-K, K walked in ascending order into an fp32 accumulator, and the same
epilogue. Lower utilization at small M is the price.

**Quantized GEMM adds one more coupling: the scale.** An FP8 activation is
stored as a small number times a scale, and the scale is chosen from a maximum.
If the maximum is taken over the whole batch, a row's scale depends on its
neighbours:

| Row x = [0.5, 3] | Scale from | Scale | What x becomes |
|---|---|---|---|
| alone | its own maximum, 3 | fine | stored with full precision for its range |
| next to a row with 100 | the batch maximum, 100 | coarse | rounded on a coarser grid: other bits |

A per-token or per-block scale uses only the row's own values, so the
coupling is gone. A real FP8 GEMM also fixes, inside the kernel, how often the
partial sum is promoted to higher precision and in which order the scales are
applied; both are part of the reduction.

## The steps

### 1. Locate the variance

- **Goal:** know which GEMM shapes in your model change bits with M.
- **How:** `bi_check.check_batch` from [tools/bi_check.py](../../tools/bi_check.py)
  on each shape (K, N) the model runs, at batch sizes around powers of two and
  tile multiples, with random, zero and large fillers.
- **Success:** a list of failing shapes and sizes. Measured: the library GEMM
  failed on every shape tested, often from 2 rows
  ([report](../../examples/moe_combine/REPORT.md#1-locate-the-variance)).

### 2. Pin one configuration

- **Goal:** one reduction tree per element, whatever M is.
- **How:** the one-config GEMM `mm_one_config` in
  [tools/bi_kernels.py](../../tools/bi_kernels.py), or your own kernel built
  from [recipe 2](../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16).
  Several heads or experts use the same body in a loop or with a head index
  ([recipe 3](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)).
- **Success:** `check_batch` passes at every size. Shown on three shapes, on
  one GPU type ([report](../../examples/moe_combine/REPORT.md#1-locate-the-variance)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Library GEMM as is | L1 only | free | never for parity |
| One-config kernel | L2 at every M | moderate at most on the shapes measured; can be faster on small shapes | the default fix |
| Library with a pinned algorithm | L2 if the pin fixes the split | free to small | not yet shown here |
| Loop of the library GEMM over fixed row pieces | L2 | large | a correctness oracle for debugging, not a product |

A second, small-M kernel is allowed only if it reproduces the same per-element
K grouping and instruction.

### 3. Fix the quantization scales

- **Goal:** a row's quantized values depend only on that row.
- **How:** per-token or per-block activation scales, or static scales;
  [recipe 9](../../skills/batch-invariant-kernel/references/operator-recipes.md#9-quantized-gemm-fp8-block-scaled-mxfp8-nvfp4).
  Then the GEMM itself follows step 2, with a fixed promotion interval and
  scale order.
- **Success:** `check_batch` passes with large fillers. Shown with emulated
  FP8 numerics ([report](../../examples/kernel_cases/REPORT.md#fp8-activation-scales-l2)).

| Activation scale | L2 | Cost class | When to choose it |
|---|---|---|---|
| Per tensor, dynamic (batch maximum) | fails | free | not for parity |
| Per token | holds | small | the default |
| Per block along K | holds | small | block-scaled recipes |
| Static | holds | free | calibrated deployments |

**Not yet shown:** a real FP8 or FP4 GEMM's promotion interval on hardware
that runs it natively.

### 4. Verify and integrate

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
on every GEMM shape the model uses, then register the kernel as an opaque op
and measure it at prefill and decode batch sizes, as in the
[workflow](README.md#5-integrate).

## What can go wrong

!!! warning "Zero fillers hide a batch-wide scale"
    A zero row does not raise the batch maximum. The test passes, and real
    traffic changes bits; test with large fillers.

!!! warning "A batched matmul over fixed pieces is not invariant"
    One batched call over fixed row pieces still picks its kernel from the
    number of pieces. Loop over the pieces or use the one-config kernel.

!!! warning "A fast result on small shapes does not generalize"
    The untuned one-config kernel beat the library on small expert shapes; on
    large shapes it can be slower. Measure on your own shapes.

## Summary

- A GEMM element is one dot product; the library chooses its grouping from M.
- Fix: one configuration for every M, no split-K, ascending K into fp32, the
  same epilogue.
- A batch-wide quantization scale couples rows; use per-token, per-block or
  static scales.
- Every later class contains a GEMM: make it invariant first.

## Go deeper

- Reports: [MoE on one GPU](../../examples/moe_combine/REPORT.md) (the GEMM
  shapes and timings), [kernel cases](../../examples/kernel_cases/REPORT.md)
  (the stock GEMM and FP8 scales).
- Recipes: [2, dense GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16),
  [3, grouped GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm),
  [9, quantized GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#9-quantized-gemm-fp8-block-scaled-mxfp8-nvfp4).
- Code: [tools/bi_kernels.py](../../tools/bi_kernels.py),
  [examples/kernel_cases/kernel_cases.py](../../examples/kernel_cases/kernel_cases.py).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
