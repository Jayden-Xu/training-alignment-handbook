# The training step: backward, optimizer, loss, resume

## Where we are

This page serves two steps of the [pre-training workflow](README.md). In step 2
(control) it lists what to pin in a training step so two trainers agree. In
step 5 (classify) it tells you which part of a step can make the first
divergence: a backward op, the loss reduction, the optimizer, AMP, gradient
accumulation or checkpoint resume. The [preflight](determinism-and-preflight.md)
checks settings; this page is about the arithmetic behind them.

## The idea: every float sum in a step needs one written order

A training step has more sums than the forward:

- the backward of every op that scatters gradients back (embedding, gather,
  index_select, an MoE combine);
- the loss reduction over tokens and sequences;
- the optimizer's update, with its own rounding points;
- accumulation across micro-batches.

**A tiny example: atomics.** Three tokens share one embedding row. Their
gradient rows g1, g2, g3 are added into the weight gradient. With atomics, the
order is whoever arrives first: (g1 + g2) + g3 on one run, (g3 + g1) + g2 on
the next. Part 0's three numbers showed that this can change the bits. This
breaks **L1** (run-to-run).

**A tiny example: deterministic but not shared.** A sort-based scatter has no
atomics, so it repeats. But its order is an implementation detail: one kernel
rounds after each chunk of rows when there are few indices and sums in fp32
once when there are many; another changes its order with the row width. Each
repeats; two frameworks agree only if they share that kernel. This breaks
**L4** (cross-implementation), and it can make a token's gradient depend on
the batch.

**Rule.** For every float sum in the step, the order must be a property of the
definition, not of timing or of the kernel: for example, ascending source
position, accumulated in fp32, rounded once. That is part 0's principle applied
to the backward, the loss and the optimizer.

Where each part of a step breaks, and what fixes it:

| Part | Breaks L1 (run-to-run) when | Breaks L4 (two trainers) when | Fix |
|---|---|---|---|
| Scatter-style backward (embedding, `index_select`, `gather` with repeated picks, MoE combine, weighted `bincount`) | atomics add the rows | the deterministic kernel's order depends on index count or row width | one written order for every scatter-style sum |
| Long 1-D `cumsum` | the scan's order follows timing | the scan's block layout differs | a left-to-right fp32 scan |
| Attention backward | dQ is added with atomics | the kernels differ | an atomic-free kernel, or strict mode |
| Loss reduction | rarely | `mean` depends on the layout (padded, packed, split), and a Python-number divisor rounds differently from a tensor divisor | a fixed tree, with the count held in a tensor |
| Optimizer | rarely | fused, foreach and single-tensor paths round at different points | pin one path and one build |
| AMP | fp16 GradScaler: the scale changes which values stay normal | the scale schedule differs | bf16 needs no scaler; in fp16 pin the schedule |
| Gradient accumulation | rarely | the micro-batch split or its order differs | pin the split and the order |
| Resume | any piece of state is missing | the same | restore every piece listed below |

Sort-based ops (embedding backward, cross-entropy, the norms, `linear`) repeat
in default mode; the atomics-based ones do not (measured: see
[report](../../examples/training_step/REPORT.md#results)).

## The step

### Pin the training step

**Goal:** two trainers, or one trainer run twice, do the same arithmetic.

**How:** go through the checklist. The fixed-order kernels are in
[ts_kernels.py](../../examples/training_step/ts_kernels.py): `seg_sum` (every
scatter-style sum: stable sort by destination, then rows added in ascending
source position in fp32, one rounding), `seq_cumsum` (left-to-right scan) and
`tree_mean_loss` (the loss reduction).

- [ ] **No atomics in any float sum.** Strict mode with `warn_only=False`, or
      the fixed-order kernels. Strict mode raises on float `cumsum`, weighted
      `bincount` and `histc`: replace those three first.
- [ ] **For agreement across frameworks, one written order for every
      scatter-style sum:** ascending source position, fp32, one rounding.
- [ ] **Attention backward:** the atomic-free kernel, or SDPA in strict mode
      ([attention](../3-kernels/attention.md)).
- [ ] **The optimizer path:** fused, foreach or single-tensor, with the same
      torch build. The default is not stable across versions.
- [ ] **The loss reduction** as a fixed tree, including how the count divides.
      Not `mean`, which depends on the layout.
- [ ] **The micro-batch split and its order.**
- [ ] **fp16 GradScaler:** the scale schedule too. bf16 needs none.
- [ ] **Every piece of checkpoint state:** model; optimizer with its step
      counters; scheduler; GradScaler; `torch.get_rng_state()`;
      `torch.cuda.get_rng_state()`; the sampler's epoch, position and
      permutation.
- [ ] **TF32 off and the same `CUBLAS_WORKSPACE_CONFIG`** on both sides.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Stock torch, default mode | Nothing guaranteed; atomics-based ops vary | free | Never for alignment |
| Strict deterministic mode | L1 for every op torch covers; raises on the rest | moderate (under 2x) on a whole step; single ops can be far slower | One framework, quick to turn on |
| Targeted fixed-order kernels (attention only) | L1 where the model actually varied | small (a few percent) | The model's only L1 breaker is attention |
| Fixed-order kernels for attention, embedding and loss | L1, and an order another framework can reproduce (L4) | moderate (under 2x) | Two frameworks or two codebases must agree |

**Success:** two runs, in separate processes, give equal loss bits, equal
gradient fingerprints and equal weights at every step; a resume from a
checkpoint gives the same steps as a straight run.

This was shown on one GPU type with one torch build: every setting above gave
equal runs over the steps tested, and the fixed scatter order matched a NumPy
loop of the same definition bit for bit, across row widths, index counts and
dtypes ([report](../../examples/training_step/REPORT.md#the-fix-one-written-order-for-every-scatter-style-sum)).

### Classify a first divergence inside a step

**Goal:** turn a pivot in the step into a part of the table above.

**How:** read where the pivot is.

| Pivot | Likely part |
|---|---|
| The first gradient consumer after the backward, with the forward equal | a nondeterministic backward upstream (attention, scatter) |
| A weight or Adam moment after an update, with gradients equal | the optimizer path or its state |
| The loss, with per-token losses equal | the loss reduction |
| An embedding lookup with equal token ids, right after a resume | resumed weights or optimizer state |
| The loss at a dropout or reshuffle point after a resume | an RNG or sampler piece |

**Success:** one part named, and a test that changes only that part.

## What can go wrong

!!! warning "A Python number as the divisor"
    `total / n` with `n` a Python int rounds differently from
    `total / tensor(n)`. Two reductions that look equal give different bits.
    State the divisor's form in the definition, and hold the count in a tensor.

!!! warning "The optimizer left to its default"
    torch picks the optimizer path itself, and fused, foreach and single-tensor
    paths round at different points. A version upgrade can then change every
    parameter. Pin the path and the build.

!!! warning "Accumulation treated as free"
    One batch of four and four micro-batches of one are different groupings of
    the sum, and a reversed micro-batch order differs too. The gradients differ
    from the first step. Pin the split and the order on both sides.

!!! warning "fp16 scaling assumed exact"
    In fp16, scaling keeps bits that would be subnormal or zero without it, so
    a scaled backward differs from an unscaled one. In bf16 a power-of-two
    scale changes nothing. Pin the scale schedule in fp16.

!!! warning "RNG state restored to the wrong device"
    `torch.set_rng_state` needs a CPU byte tensor; loading the checkpoint to
    the GPU fails to restore it. The run diverges at the next reshuffle, steps
    later. Load RNG state on the CPU.

## Summary

- Every float sum in a step needs one order fixed by its definition: scatters,
  scans, attention backward, loss.
- Deterministic is not enough for two trainers: kernel-specific orders, the
  optimizer path and the loss divisor's form all change bits.
- Strict mode gives L1 at a moderate cost; fixed-order kernels give L1 and a
  reproducible order.
- Pin the optimizer path, the loss tree, the micro-batch split and every piece
  of resume state.

## Go deeper

- [Report: the training step](../../examples/training_step/REPORT.md): every op
  measured, the whole-step runs, the optimizer rounding-point table, loss
  layouts, AMP, accumulation, clipping, resume and cost.
- Code: [examples/training_step/](../../examples/training_step/).
- [Data parallel](data-parallel.md): the same question for gradients summed
  across ranks.
- [Attention](../3-kernels/attention.md) for the attention backward.
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
