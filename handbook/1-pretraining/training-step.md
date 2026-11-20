# Making every training step repeat bit for bit

## Where we are

You can capture two runs with OpGuard and follow the first divergence back to
its cause ([Localize with OpGuard](localize-with-opguard.md)). Often the
forward pass of the two runs agrees, and the first difference appears only
after an update. Then the cause is inside the training step: the backward
pass, the loss, the optimizer, gradient accumulation, mixed precision, or a
resume from a checkpoint. This page explains why each of these can change the
bits, and what to pin so that a step repeats exactly. Read it before you
capture a new pair, too: the same list is what two trainers must agree on.

## The question

Start two runs from the same checkpoint, with the same data and the same
seeds. After one step, should their weights be identical, bit for bit?

Only if every sum inside the step is computed in the same order in both runs.
[Part 0](../0-foundations/README.md) showed that adding the same numbers in a
different order can give a different result. A training step contains far
more sums than the forward pass, and many of them have no fixed order unless
you give them one:

- the backward of every operator that scatters gradients back to its inputs
  (embedding, `gather`, `index_select`, the MoE combine);
- the loss, averaged over tokens and sequences;
- the optimizer update, which rounds at its own places;
- the sum of gradients over micro-batches.

These sums go wrong in two different ways.

**The same code gives a different result on each run (this breaks L1).**
Three tokens share one row of the embedding table, so their gradient rows g1,
g2 and g3 must be added into one weight-gradient row. A kernel that uses
atomic adds adds them in whatever order the threads finish: (g1 + g2) + g3 on
one run, (g3 + g1) + g2 on the next. The results can differ in the last bit.

**The same code repeats itself, but another implementation does not match it
(this breaks L4).** A sort-based scatter uses no atomics, so it repeats
exactly. But its order is a detail of that kernel: one kernel rounds after
each group of rows when there are few indices and sums in fp32 when there are
many, another changes its order with the row width. Each repeats itself; two
frameworks agree only if they happen to use the same kernel. The optimizer
and the loss behave the same way: the fused, foreach and single-tensor
optimizer paths round at different places, and `mean` gives different bits
for the same tokens padded or packed.

**Rule.** Give every floating-point sum in the step an order that follows
from its definition, not from timing or from a kernel's internals: for
example, add rows in ascending source position, accumulate in fp32, and
round once at the end. This is the principle of part 0 applied to the
backward pass, the loss and the optimizer.

| Part of the step | Can the same code differ between runs? | Can two implementations differ? | How to pin it |
|---|---|---|---|
| Backward of scatter-style operators (embedding, `index_select`, `gather` with repeated rows, MoE combine, weighted `bincount`) | Yes, with atomic adds | Yes, the order depends on the index count or row width | One written order for every scatter-style sum |
| A long 1-D `cumsum` | Yes, the scan order follows timing | Yes, the scan splits the input differently | A left-to-right scan in fp32 |
| Attention backward | Yes, dQ is accumulated with atomic adds | Yes, different kernels | An atomic-free kernel, or strict mode |
| Loss | Rarely | Yes: `mean` depends on padding, packing and splitting, and dividing by a Python number rounds differently from dividing by a tensor | A fixed reduction tree, with the count kept in a tensor |
| Optimizer | Rarely | Yes: the fused, foreach and single-tensor paths round in different places | One code path and one torch build |
| Mixed precision (fp16 GradScaler) | The scale decides which values stay normal | The scaling rule may differ | bf16 needs no scaler; with fp16, pin the rule |
| Gradient accumulation | Rarely | Yes, a different split or order of micro-batches | A fixed split and order |
| Resume from a checkpoint | Yes, if any piece of state is missing | Yes, the same | Restore every item in the list below |

Sort-based operators (embedding backward, cross-entropy, the norms, `linear`)
already repeat in default mode; atomic-based ones do not (measured: see the
[report](../../examples/training_step/REPORT.md#results)).

## The steps

### Make the same code repeat itself

Start here: two runs of one trainer must agree before two trainers can.
Remove every atomic floating-point sum, either with torch's strict mode or by
replacing the few operators that need it.

| Option | What you get | Cost class | When to choose it |
|---|---|---|---|
| Stock torch, default mode | No guarantee; atomic-based operators can differ on every run | Free | Never, when you need agreement |
| Strict deterministic mode | L1 for every operator torch covers; the others raise an error | Moderate (under 2x) for a whole step; single operators can be much slower | One framework, and you want a switch to turn on |
| Fixed-order kernel for attention only | L1 where the model actually varied | Small (a few percent) | Attention was the only operator that broke L1 |
| Fixed-order kernels for attention, embedding and the loss | L1, and an order another framework can reproduce (L4) | Moderate (under 2x) | Two frameworks or two code bases must agree |

- Strict mode: set it with `warn_only=False`. It raises on floating-point
  `cumsum`, weighted `bincount` and `histc`, so replace those three first.
- The fixed-order kernels are in
  [ts_kernels.py](../../examples/training_step/ts_kernels.py): `seg_sum` for
  every scatter-style sum (a stable sort by destination, then rows added in
  ascending source position in fp32, rounded once), `seq_cumsum` for a
  left-to-right scan, and `tree_mean_loss` for the loss.
- Attention backward: use an atomic-free kernel, or SDPA under strict mode
  ([Attention](../3-kernels/attention.md)).
- On both runs, turn TF32 off and use the same `CUBLAS_WORKSPACE_CONFIG`.

### Make two trainers compute the same arithmetic

Once each trainer repeats itself, pin what a second implementation would
otherwise do differently:

- every scatter-style sum in one written order: ascending source position,
  fp32 accumulation, one rounding;
- one optimizer code path (fused, foreach or single-tensor) and one torch
  build, since the default path can change between versions;
- the loss as a fixed reduction tree, with the division written out; not
  `mean`, whose result depends on the layout;
- the same split and order of micro-batches;
- with fp16, the same GradScaler rule (bf16 needs no scaler).

You are done when two runs in separate processes give the same loss bits at
every step, equal gradient fingerprints and equal weights. This was shown on
one GPU type and one torch build for every setting above, and the fixed
scatter order matched a NumPy loop written from the same definition at every
row width, index count and dtype tested
([report](../../examples/training_step/REPORT.md#the-fix-one-written-order-for-every-scatter-style-sum)).

### Save everything a resume needs

A resumed run matches an uninterrupted one only if the checkpoint holds all
the state the next steps read:

- [ ] the model;
- [ ] the optimizer, including its step counter;
- [ ] the learning-rate scheduler;
- [ ] the GradScaler, with fp16;
- [ ] the CPU random state, `torch.get_rng_state()`;
- [ ] the GPU random state, `torch.cuda.get_rng_state()`;
- [ ] the sampler's epoch, position and permutation.

You are done when a run resumed from a checkpoint and a run that never
stopped are identical at every step after the resume.

### Find which part of the step a first divergence points to

Look at where the first divergence lands and map it to a part of the step.
This is done when you can name one part and run a test that changes only
that part.

| Where the first divergence lands | Most likely part |
|---|---|
| The first operator that uses a gradient, while the forward results agree | A nondeterministic backward upstream (attention, scatter) |
| A weight or an Adam moment after the update, while the gradients agree | The optimizer code path or its state |
| The loss, while every token's loss agrees | The loss reduction |
| The embedding lookup right after a resume, while the token ids agree | The restored weights or optimizer state |
| The loss after a resume, at dropout or at the next reshuffle | A missing random state or sampler state |

## What can go wrong

!!! warning "Dividing by a Python number"
    When `n` is a Python int, `total / n` and `total / tensor(n)` round
    differently. Two reductions that look the same then differ in the last
    bit. Write the division into the definition and keep the count in a
    tensor.

!!! warning "Leaving the optimizer on its default path"
    torch picks the optimizer's code path itself, and the fused, foreach and
    single-tensor paths round in different places. One version upgrade can
    then change every parameter. Pin the path and the torch build.

!!! warning "Assuming gradient accumulation changes nothing"
    One batch of four sequences and four micro-batches of one sequence group
    the sums differently, and reversing the micro-batch order changes the
    result too. The gradients differ from the first step. Pin the split and
    the order on both sides.

!!! warning "Assuming fp16 scaling is exact"
    In fp16, scaling keeps bits that would otherwise become subnormal or zero,
    so a scaled backward differs from an unscaled one. In bf16, scaling by a
    power of 2 changes nothing. With fp16, pin the scaling rule.

!!! warning "Restoring the random state on the wrong device"
    `torch.set_rng_state` needs a byte tensor on the CPU, so loading the
    checkpoint onto the GPU fails to restore it. The runs then part a few
    steps later, at the next reshuffle. Load the random state on the CPU.

## Summary

- A training step adds far more numbers than the forward pass, and each of
  those sums needs an order that follows from its definition.
- Two failures look alike: atomic adds make the same code differ between
  runs (L1); fixed but different orders make two implementations differ
  (L4).
- First make one trainer repeat itself (strict mode or a few fixed-order
  kernels), then pin the optimizer path, the loss reduction and the
  micro-batch split so that two trainers agree.
- A resume matches only if the checkpoint saves every piece of state,
  including both random states and the sampler.

## Go deeper

- [Report: training step](../../examples/training_step/REPORT.md): every
  operator measured, whole-step runs, the optimizer's rounding points, loss
  layouts, mixed precision, accumulation, clipping, resume and costs.
- Code: [examples/training_step/](../../examples/training_step/).
- [Data parallel](data-parallel.md): the same question for gradients summed
  across ranks.
- Attention backward: [Attention](../3-kernels/attention.md).
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
