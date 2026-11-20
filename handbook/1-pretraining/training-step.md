# The training step

## Where we are

You can capture a pair of runs and trace from the first divergence to a cause
([Localize with OpGuard](localize-with-opguard.md)). This page is for the case
where that trace leads into the training step, at step 5 (classify) of the
[pre-training workflow](README.md): into the backward of an operator, the
loss reduction, the optimizer, AMP, gradient accumulation or a resume from a
checkpoint. It also lists what to pin in a training step so that two trainers
agree, which belongs to step 2 (control), so read it before you capture a new
pair. The [preflight](determinism-and-preflight.md) checks the settings; this
page is about the arithmetic behind them.

## The idea: every floating-point sum in a step needs an order that is written down

A training step contains more sums than the forward pass does:

- the backward of every operator that scatters gradients back to its inputs
  (embedding, gather, index_select, an MoE combine);
- the loss reduction over tokens and sequences;
- the optimizer's update, which has its own rounding points;
- the accumulation of gradients across micro-batches.

**A small example with atomics.** Three tokens share one embedding row. Their
gradient rows g1, g2 and g3 are added into the weight gradient. With atomics,
the order is whichever row arrives first: (g1 + g2) + g3 in one run,
(g3 + g1) + g2 in the next. Part 0's three numbers showed that this can change
the bits. This breaks **L1** (run-to-run).

**A small example of a kernel that is deterministic but not shared.** A
scatter based on sorting uses no atomics, so it repeats itself. But its order
is a detail of its implementation. One such kernel rounds after each chunk of
rows when there are few indices, and sums once in fp32 when there are many;
another changes its order with the row width. Each of them repeats itself, but
two frameworks agree only if they use the same kernel. This breaks **L4**
(agreement between implementations), and it can make a token's gradient
depend on the batch.

**Rule.** For every floating-point sum in the step, the order must follow from
the definition of the sum, not from timing or from the kernel: for example,
add in ascending source position, accumulate in fp32, and round once. This is
part 0's principle applied to the backward pass, the loss and the optimizer.

Where each part of a step breaks, and what fixes it:

| Part | It breaks L1 (run-to-run) when | It breaks L4 (two trainers) when | Fix |
|---|---|---|---|
| A scatter-style backward (embedding, `index_select`, `gather` that picks the same row more than once, an MoE combine, a weighted `bincount`) | atomics add the rows | the deterministic kernel's order depends on the number of indices or on the row width | one written order for every scatter-style sum |
| A long 1-D `cumsum` | the order of the scan follows timing | the scan splits its input into blocks differently | a left-to-right scan in fp32 |
| The attention backward | dQ is added with atomics | the kernels differ | a kernel without atomics, or strict mode |
| The loss reduction | rarely | `mean` depends on the layout (padded, packed or split), and dividing by a Python number rounds differently from dividing by a tensor | a fixed reduction tree, with the count held in a tensor |
| The optimizer | rarely | the fused, foreach and single-tensor code paths round at different points | pin one code path and one build |
| AMP | in fp16, the GradScaler's scale decides which values stay normal | the scale schedule differs | bf16 needs no scaler; in fp16, pin the schedule |
| Gradient accumulation | rarely | the split into micro-batches, or their order, differs | pin the split and the order |
| Resume | any piece of state is missing | the same | restore every piece listed below |

Operators based on sorting (embedding backward, cross-entropy, the norms,
`linear`) repeat themselves in default mode; the ones based on atomics do not
(measured: see [report](../../examples/training_step/REPORT.md#results)).

## The step

### Pin down every part of the training step

**Goal:** two trainers, or one trainer run twice, do exactly the same
arithmetic.

**How:** go through the checklist below. The fixed-order kernels are in
[ts_kernels.py](../../examples/training_step/ts_kernels.py): `seg_sum` handles
every scatter-style sum (a stable sort by destination, then the rows added in
ascending source position in fp32, with one rounding), `seq_cumsum` is a
left-to-right scan, and `tree_mean_loss` is the loss reduction.

- [ ] **No atomics in any floating-point sum.** Use strict mode with
      `warn_only=False`, or the fixed-order kernels. Strict mode raises an
      error on a floating-point `cumsum`, a weighted `bincount` and `histc`,
      so replace those three first.
- [ ] **For agreement across frameworks, one written order for every
      scatter-style sum:** ascending source position, fp32, one rounding.
- [ ] **The attention backward:** use the kernel without atomics, or SDPA in
      strict mode ([attention](../3-kernels/attention.md)).
- [ ] **The optimizer's code path:** fused, foreach or single-tensor, with the
      same torch build. The default is not stable across versions.
- [ ] **The loss reduction** as a fixed tree, including how the count divides.
      Do not use `mean`, which depends on the layout.
- [ ] **The split into micro-batches, and their order.**
- [ ] **The fp16 GradScaler:** pin its scale schedule too. bf16 needs no
      scaler.
- [ ] **Every piece of checkpoint state:** the model; the optimizer with its
      step counters; the scheduler; the GradScaler;
      `torch.get_rng_state()`; `torch.cuda.get_rng_state()`; and the
      sampler's epoch, position and permutation.
- [ ] **TF32 off and the same `CUBLAS_WORKSPACE_CONFIG`** on both sides.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Stock torch in default mode | No guarantee at all; the operators based on atomics vary | free | Never, when you need alignment |
| Strict deterministic mode | L1 for every operator torch covers, and an error for the rest | moderate (under 2x) for a whole step, though single operators can be far slower | One framework, when you want something quick to turn on |
| Fixed-order kernels for attention only | L1 where the model actually varied | small (a few percent) | Attention is the only thing in the model that breaks L1 |
| Fixed-order kernels for attention, embedding and the loss | L1, and an order that another framework can reproduce (L4) | moderate (under 2x) | Two frameworks, or two codebases, must agree |

**Success:** two runs in separate processes give equal loss bits, equal
gradient fingerprints and equal weights at every step, and a run that resumes
from a checkpoint gives the same steps as a run that never stopped.

This was shown on one GPU type with one torch build. Every setting above gave
equal runs over the steps tested, and the fixed scatter order matched, bit for
bit, a NumPy loop written from the same definition, across row widths, index
counts and dtypes
([report](../../examples/training_step/REPORT.md#the-fix-one-written-order-for-every-scatter-style-sum)).

### Find which part of the step a first divergence points to

**Goal:** map a pivot inside the step to one of the parts in the table above.

**How:** look at where the pivot is.

| Where the pivot is | The likely part |
|---|---|
| The first operator that uses the gradients after the backward pass, while the forward pass is equal | a nondeterministic backward upstream (attention, a scatter) |
| A weight or an Adam moment after an update, while the gradients are equal | the optimizer's code path, or its state |
| The loss, while the per-token losses are equal | the loss reduction |
| An embedding lookup right after a resume, with equal token ids | the resumed weights or optimizer state |
| The loss at a dropout or reshuffle point after a resume | a piece of RNG or sampler state |

**Success:** you have named one part, and you have a test that changes only
that part.

## What can go wrong

!!! warning "A Python number as the divisor"
    `total / n`, with `n` a Python int, rounds differently from
    `total / tensor(n)`. Two reductions that look the same then give
    different bits. State the form of the divisor in the definition, and hold
    the count in a tensor.

!!! warning "Leaving the optimizer on its default"
    torch picks the optimizer's code path by itself, and the fused, foreach
    and single-tensor paths round at different points. A version upgrade can
    then change every parameter. Pin the code path and the build.

!!! warning "Assuming that accumulation changes nothing"
    One batch of four sequences and four micro-batches of one sequence each
    group the sum differently, and reversing the order of the micro-batches
    changes it too. The gradients then differ from the first step. Pin the
    split and the order on both sides.

!!! warning "Assuming that fp16 scaling is exact"
    In fp16, scaling preserves bits that would otherwise become subnormal or
    zero, so a scaled backward pass differs from an unscaled one. In bf16, a
    power-of-two scale changes nothing. In fp16, pin the scale schedule.

!!! warning "RNG state restored to the wrong device"
    `torch.set_rng_state` needs a CPU byte tensor, so loading the checkpoint
    onto the GPU fails to restore it. The run then diverges at the next
    reshuffle, several steps later. Load the RNG state on the CPU.

## Summary

- Every floating-point sum in a step needs one order that follows from its
  definition: scatters, scans, the attention backward and the loss.
- Determinism is not enough for two trainers to agree: orders specific to a
  kernel, the optimizer's code path and the form of the loss divisor all
  change the bits.
- Strict mode gives L1 at a moderate cost; fixed-order kernels give L1 and an
  order that others can reproduce.
- Pin the optimizer's code path, the loss tree, the split into micro-batches,
  and every piece of resume state.

## Go deeper

- [Report: the training step](../../examples/training_step/REPORT.md) covers
  every operator measured, the whole-step runs, the table of the optimizer's
  rounding points, the loss layouts, AMP, accumulation, clipping, resume and
  the cost.
- The code is in [examples/training_step/](../../examples/training_step/).
- [Data parallel](data-parallel.md) asks the same question for gradients that
  are summed across ranks.
- [Attention](../3-kernels/attention.md) covers the attention backward.
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
