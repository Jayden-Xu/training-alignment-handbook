# GEMM and quantized GEMM

## Where we are

The [workflow](README.md) has five steps: find what varies, choose the
cheapest fix, round in the same places as the side you match, verify, and
integrate. GEMM is the simplest operator to practise it on. Each output
element is one dot product of a fixed length K, and nothing else in the
operator is a reduction. Every later page relies on an invariant GEMM,
because attention projections, MLA's per-head products, MoE experts and
tensor-parallel layers all contain one.

## The idea: the library looks at the batch to decide how to split the sum

An output element C[m, n] is the sum of K products. A GEMM library divides the
output into tiles and decides, on every call, how to walk along K. It makes
that decision from the shape of the problem, and the shape includes M, the
number of rows in the batch.

<figure class="fx" data-fig="gemm-split" aria-label="Row 0 of a matrix product in a batch of 1 and in a batch of 64">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">a batch of 1 row has little work<span class="fx-batch fx-mini"><span class="fx-rowbar is-mine"></span></span></span><div class="fx-tree" style="--n:4">
<div class="fx-level"><span class="fx-cell" data-hot="1">K/4</span><span class="fx-cell" data-hot="1">K/4</span><span class="fx-cell" data-hot="1">K/4</span><span class="fx-cell" data-hot="1">K/4</span></div>
<div class="fx-level" data-at="1"><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 0</span></span><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 1</span></span><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 2</span></span><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 3</span></span></div>
<div class="fx-level" data-at="2" data-hot="4"><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-key">row 0: value A</span></span></div>
</div></div>
<div class="fx-row"><span class="fx-label">a batch of 64 rows has enough work<span class="fx-batch fx-mini"><span class="fx-rowbar is-mine"></span><span class="fx-rowbar"></span><span class="fx-rowbar"></span><span class="fx-rowbar"></span><span class="fx-rowbar"></span></span></span><div class="fx-tree" style="--n:4">
<div class="fx-level fx-sweep" data-hot="3"><span class="fx-cell">K/4</span><span class="fx-cell">K/4</span><span class="fx-cell">K/4</span><span class="fx-cell">K/4</span></div>
<div class="fx-level" data-at="3"><span class="fx-walk" style="--w:4">one program walks all of K in order</span></div>
<div class="fx-level" data-at="4"><span class="fx-j" style="--w:4;--k:1"><span class="fx-node is-bad">row 0: value B ≠ A</span></span></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>Row 0 (highlighted) is the same dot product over K in both batches.</li>
<li data-hold="1.5">In a batch of 1 there is little work, so the library splits K into 4 pieces and runs them on 4 programs.</li>
<li data-hold="1.5">It then adds the 4 partial sums, and row 0 gets value A.</li>
<li data-hold="2">In a batch of 64 there is enough work without splitting, so one program walks all of K in order.</li>
<li data-hold="2">Now row 0 gets value B ≠ A. The math is the same, but the grouping is different.</li>
</ol>
<figcaption>Figure: the library picks how to split K from the number of rows, so the same row gets different bits in a different batch.</figcaption>
</figure>

With a small batch the library may split K to keep the GPU busy (split-K or
stream-K), or it may call a special kernel written for small M. With a large
batch it walks K in one pass. Each of these choices is deterministic, so the
same batch gives the same bits on every run (L1). But when the same row sits
in another batch, its sum is grouped differently, and L2 fails.

**The rule for GEMM.** Use one tile configuration for every M, do not use
split-K or stream-K, walk K in ascending order into an fp32 accumulator, and
keep the same epilogue. The price is lower utilization when M is small.

**Quantized GEMM adds one more coupling between rows: the scale.** An FP8
activation is stored as a small number multiplied by a scale, and the scale is
chosen from a maximum. If that maximum is taken over the whole batch, a row's
scale depends on the rows next to it:

| Row x = [0.5, 3] | Where the scale comes from | Scale | What happens to x |
|---|---|---|---|
| alone | its own maximum, 3 | fine | it is stored with full precision for its range |
| next to a row that contains 100 | the batch maximum, 100 | coarse | it is rounded on a coarser grid, so its bits change |

A per-token or per-block scale uses only the row's own values, so the
coupling disappears. A real FP8 GEMM also fixes two more things inside the
kernel: how often the partial sum is promoted to higher precision, and the
order in which the scales are applied. Both are part of the reduction.

## The steps

### 1. Find the GEMM shapes whose result changes with the batch

- **Goal:** know which GEMM shapes in your model give different bits when M
  changes.
- **How:** for each shape (K, N) that the model runs, call
  `bi_check.check_batch` from [tools/bi_check.py](../../tools/bi_check.py).
  Use batch sizes around powers of two and around multiples of the tile size,
  and fill the other rows with random values, zeros and large values.
- **Success:** you have a list of the shapes and sizes that fail. Measured:
  the library GEMM failed on every shape tested, often already with 2 rows
  ([report](../../examples/moe_combine/REPORT.md#1-locate-the-variance)).

### 2. Use one configuration for every batch size

- **Goal:** every element is summed by the same reduction tree, whatever M is.
- **How:** use the one-configuration GEMM `mm_one_config` in
  [tools/bi_kernels.py](../../tools/bi_kernels.py), or write your own kernel
  from [recipe 2](../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16).
  When several heads or experts need a GEMM, run the same kernel body in a
  loop or pass it a head index
  ([recipe 3](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)).
- **Success:** `check_batch` passes at every size. This was shown on three
  shapes, on one GPU type
  ([report](../../examples/moe_combine/REPORT.md#1-locate-the-variance)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Use the library GEMM as it is | L1 only | free | never, if you need parity |
| A one-configuration kernel | L2 at every M | at most moderate on the shapes measured, and sometimes faster on small shapes | the default fix |
| The library with a pinned algorithm | L2, if the pinned algorithm also fixes the split | free to small | not yet shown here |
| Call the library GEMM in a loop over fixed pieces of rows | L2 | large | as a correctness oracle for debugging, not for production |

You may add a second kernel for small M only if it groups K in exactly the
same way for each element and uses the same instructions.

### 3. Compute the quantization scales from each row alone

- **Goal:** a row's quantized values depend only on that row.
- **How:** use per-token or per-block activation scales, or static scales
  ([recipe 9](../../skills/batch-invariant-kernel/references/operator-recipes.md#9-quantized-gemm-fp8-block-scaled-mxfp8-nvfp4)).
  The GEMM itself then follows step 2, with a fixed promotion interval and a
  fixed order for applying the scales.
- **Success:** `check_batch` passes when the other rows hold large values.
  This was shown with emulated FP8 numerics
  ([report](../../examples/kernel_cases/REPORT.md#fp8-activation-scales-l2)).

| Activation scale | L2 | Cost class | When to choose it |
|---|---|---|---|
| Per tensor, computed at run time from the batch maximum | fails | free | not when you need parity |
| Per token | holds | small | the default |
| Per block along K | holds | small | block-scaled recipes |
| Static | holds | free | calibrated deployments |

**Not yet shown:** how often a real FP8 or FP4 GEMM promotes its partial sum,
on hardware that runs these formats natively.

### 4. Verify the kernel and put it into use

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
on every GEMM shape the model uses. Then register the kernel as an opaque op
and measure it at prefill and decode batch sizes, as described in the
[workflow](README.md#5-integrate).

## What can go wrong

!!! warning "Zero filler rows hide a scale computed over the batch"
    A row of zeros does not raise the batch maximum, so the test passes while
    real traffic still changes the bits. Put large values in the filler rows.

!!! warning "A batched matmul over fixed pieces is still not invariant"
    A single batched call over fixed pieces of rows still picks its kernel
    from the number of pieces. Loop over the pieces instead, or use the
    one-configuration kernel.

!!! warning "Speed on small shapes does not carry over to large ones"
    The untuned one-configuration kernel was faster than the library on small
    expert shapes, but on large shapes it can be slower. Measure on your own
    shapes.

## Summary

- Each element of a GEMM is one dot product, and the library chooses how to
  group it from M.
- To fix it, use one configuration for every M, no split-K, K in ascending
  order into fp32, and the same epilogue.
- A quantization scale computed over the batch couples the rows; use
  per-token, per-block or static scales instead.
- Every later class contains a GEMM, so make the GEMM invariant first.

## Go deeper

- Reports: [MoE on one GPU](../../examples/moe_combine/REPORT.md) (the GEMM
  shapes and their timings) and
  [kernel cases](../../examples/kernel_cases/REPORT.md) (the stock GEMM and
  FP8 scales).
- Recipes: [2, dense GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16),
  [3, grouped GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm),
  [9, quantized GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#9-quantized-gemm-fp8-block-scaled-mxfp8-nvfp4).
- Code: [tools/bi_kernels.py](../../tools/bi_kernels.py),
  [examples/kernel_cases/kernel_cases.py](../../examples/kernel_cases/kernel_cases.py).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
