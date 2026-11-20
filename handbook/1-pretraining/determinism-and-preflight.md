# Determinism and preflight

## Where we are

This is step 2 of the [pre-training workflow](README.md). You have chosen a
reference run and written down the one difference you are testing. What you
do not know yet is whether the two runs also differ in something else. A
setting that nobody wrote down makes the comparison meaningless. So this step
first controls the avoidable nondeterminism, and then proves, from inside each
job, that only the declared difference remains.

## The idea: what you configured is not necessarily what ran

Here is a small example. Two runs resume from the same checkpoint. The suspect
run restores the model and Adam's moments, but not the optimizer's step
counter. Adam's bias correction divides by a factor computed from that
counter, so the first update after the restore has a different size in the
two runs. The model weights are equal right after the restore, but the next
forward pass is not.

Nothing in the launch command shows this, because the difference lives in
state inside the job. The same is true of a flag that a library sets when it
is imported, an environment variable that is set on only one node, or a
dataloader generator that was seeded differently.

**Rule.** Control the avoidable variation, then read it back from inside the
running job: on every rank, after the restore and before the first step. Then
diff the two runs. This applies part 0's principle of controlled and residual
nondeterminism: remove what makes two correct runs do different arithmetic,
and leave faults visible.

## The step

### 2a. Control the five sources of avoidable variation

**Goal:** remove the variation that makes two correct runs differ.

| What varies | What to set | Cost class |
|---|---|---|
| Randomness and data order | the CPU and CUDA RNG seeds, the dropout streams, the dataloader workers, the sampler state and the distributed initialization; seed numpy as well | free |
| Library kernels | `torch.use_deterministic_algorithms(True)` with `warn_only=False`, deterministic cuDNN, no autotuning, TF32 off and `CUBLAS_WORKSPACE_CONFIG=:16:8`; the deterministic variants of fused attention and MoE kernels; the same toolkit, GPU architecture and SM count | free to large: strict mode can cost a large share of a step, while targeted fixed-order kernels cost less ([Training step](training-step.md)) |
| Collectives | the algorithm, protocol, topology and bucketing, so that partial sums are added in the same tree | small to large, depending on the message size ([Data parallel](data-parallel.md)) |
| Checkpoint resume | the RNG streams, the optimizer state with its step counters, the scheduler, the GradScaler and the dataloader position, on every rank | free |
| TP degree | a TP-simulator mode, in which the logical TP stays above 1 while every rank runs the unsharded arithmetic, so that TP=N matches TP=1 bit for bit | not measured here |

Do **not** pin scheduling, or kernels written by you or by a third party. If
they change the bits, that change is the signal of a bug.

**Success:** every one of the five is set in code or in the launch
configuration, on every rank.

### 2b. Read the settings back from each run and diff them

**Goal:** prove that the settings in both runs are what you think they are.

**How:** in each job, call `determinism_preflight.report(...)` from
[determinism_preflight.py](../../tools/determinism_preflight.py) after the
restore and before the first step. It reads back, on each rank, and writes as
JSON: the deterministic flag and its `warn_only`, cuDNN, TF32 and matmul
precision, `CUBLAS_WORKSPACE_CONFIG`, the SDPA backends, `NCCL_*` and the NCCL
version, the RNG states, the dataloader's workers and generator, the
optimizer's code path (fused or foreach) and its state, the scheduler's
counters, AMP, whether compile and CUDA graphs are used, and the GPU, driver
and library versions. When OpGuard is installed, its `doctor` output is
included too. Then `diff(a, b)` lists what differs between the two runs and
which of those differences make a bitwise comparison meaningless, and
`findings` lists what a single run leaves uncontrolled.

**Success:** `findings` is empty, and the diff lists only the declared
difference.

In a known-answer walkthrough (one GPU, one rank, a few steps), the preflight
caught an attention backward running under `warn_only` and a missing step
counter before anything was captured, and each piece of resume state that was
left out showed up in its own field. A rounding point that had been moved did
not show up, because it is a change in code, not in a setting
([report](../../examples/pretraining_e2e/REPORT.md#the-determinism-preflight)).

## What can go wrong

!!! warning "`warn_only=True`"
    With `warn_only=True`, deterministic mode only warns when a kernel has no
    deterministic version, so attention's backward stays nondeterministic. The
    run looks controlled but is not. Use `warn_only=False`, or replace the
    kernel ([attention](../3-kernels/attention.md)).

!!! warning "A code difference is not a setting"
    The preflight reads settings and state. A cast that moved, or a different
    kernel, passes it unnoticed. Expect code differences to show up only in
    the capture.

!!! warning "The generator's state does not tell you the data position"
    The preflight sees the state of the loader's generator, not how many
    batches were consumed after it was seeded. A resume that skips the wrong
    number of batches therefore passes. Restore and compare the sampler's
    position explicitly.

!!! warning "A report taken at the wrong time"
    A report taken before the restore shows the fresh state, not the resumed
    one, so the diff is clean while the runs still differ. Take the report
    after the restore and before the first step.

!!! warning "A library RNG that nobody seeds"
    A random source that the workload never seeds (numpy is a common one)
    differs in every pair of runs, and it blocks every comparison. Seed it,
    even when you think nothing uses it.

## Summary

- Control the five sources of avoidable variation: randomness, library
  kernels, collectives, resume state and the TP degree. Leave scheduling and
  your own kernels alone.
- Read the settings back from inside each job, on every rank, after the
  restore.
- Diff the two runs; only the declared difference may remain.
- The preflight sees settings, not code, so a change in rounding shows up only
  in the capture.

## Go deeper

- [determinism_preflight.py](../../tools/determinism_preflight.py) is the
  tool.
- [Report: what the preflight caught, and what each left-out piece of resume state looks like](../../examples/pretraining_e2e/REPORT.md#the-determinism-preflight).
- [Training step](training-step.md) lists which settings and which state
  matter for a full training step, and what each fix costs.
- [Data parallel](data-parallel.md) covers the collective settings for
  gradients.
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md).
