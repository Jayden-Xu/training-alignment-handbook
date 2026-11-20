# Determinism and preflight

## Where we are

This is step 2 of the [pre-training workflow](README.md). You have chosen a
reference run and written down the one difference under test. What is still
open: whether the two runs differ in anything else. A setting nobody wrote down
makes the comparison meaningless, so this step controls the avoidable
nondeterminism and then proves, from inside each job, that only the declared
difference remains.

## The idea: what you set is not what ran

A tiny example. Two runs resume from the same checkpoint. The suspect restores
the model and Adam's moments, but not the optimizer's step counter. Adam's
bias correction divides by a factor computed from that counter, so the first
update after the restore is a different size on the two sides. The model
weights are equal at the restore; the next forward is not.

Nothing in the launch command shows this. The difference lives in state inside
the job. The same holds for a flag set by a library at import, an environment
variable on one node, or a dataloader generator seeded differently.

**Rule.** Control the avoidable variation, then read it back from inside the
running job, per rank, after the restore and before the first step, and diff
the two runs. This instances part 0's principle of controlled against residual
nondeterminism: remove what makes two correct runs do different arithmetic,
leave visible what is a fault.

## The step

### 2a. Set the five control surfaces

**Goal:** remove the variation that makes two correct runs differ.

| Surface | What to set | Cost class |
|---|---|---|
| Randomness and data order | CPU and CUDA RNG seeds, dropout streams, dataloader workers, sampler state, distributed initialization; seed numpy too | free |
| Library kernels | `torch.use_deterministic_algorithms(True)` with `warn_only=False`, deterministic cuDNN, no autotuning, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:16:8`; deterministic variants of fused attention and MoE kernels; the same toolkit, GPU architecture and SM count | free to large: strict mode can cost a large share of a step; targeted fixed-order kernels cost less ([Training step](training-step.md)) |
| Collectives | the algorithm, protocol, topology and bucketization, so partial sums follow the same tree | small to large, by message size ([Data parallel](data-parallel.md)) |
| Checkpoint resume | RNG streams, optimizer state with its step counters, scheduler, GradScaler, dataloader position, on every rank | free |
| TP degree | a TP-simulator mode: logical TP stays above 1 while every rank runs the unsharded arithmetic, so TP=N matches TP=1 bitwise | not measured here |

Do **not** fix scheduling, or user and third-party kernels. If they change the
bits, that is the bug signal.

**Success:** every surface is set in code or launch configuration, on every
rank.

### 2b. Report and diff

**Goal:** proof that the surfaces are what you think, on both runs.

**How:** call `determinism_preflight.report(...)` from
[determinism_preflight.py](../../tools/determinism_preflight.py) in each job,
after the restore and before the first step. It reads back, per rank, as JSON:
the deterministic flag and its `warn_only`, cuDNN, TF32 and matmul precision,
`CUBLAS_WORKSPACE_CONFIG`, SDPA backends, `NCCL_*` and the NCCL version, RNG
states, the dataloader's workers and generator, the optimizer's path (fused,
foreach) and state, scheduler counters, AMP, compile and CUDA-graph use, GPU,
driver and library versions. When OpGuard is installed its `doctor` output is
embedded. Then `diff(a, b)` lists what differs between the two runs and which
differences make a bitwise comparison meaningless; `findings` lists what one
run leaves uncontrolled.

**Success:** `findings` is empty and the diff lists only the declared
difference.

In a known-answer walkthrough (one GPU, one rank, a few steps) the preflight
caught a `warn_only` attention backward and a missing step counter before any
capture, and each left-out resume piece showed in its own field; a moved
rounding point did not show, because it is code
([report](../../examples/pretraining_e2e/REPORT.md#the-determinism-preflight)).

## What can go wrong

!!! warning "`warn_only=True`"
    Deterministic mode with `warn_only=True` only warns when a kernel has no
    deterministic version, so attention's backward stays nondeterministic. The
    run looks controlled and is not. Use `warn_only=False`, or replace the
    kernel ([attention](../3-kernels/attention.md)).

!!! warning "A code difference is not a setting"
    The preflight reads settings and state. A moved cast or a different kernel
    passes it unseen. Expect code differences to show only in the capture.

!!! warning "Generator state is not data position"
    The preflight sees the loader's generator state, not how many batches were
    consumed after it. A resume that skips the wrong number of batches passes.
    Restore and compare the sampler's position explicitly.

!!! warning "A report taken at the wrong time"
    Before the restore, the report shows the fresh state, not the resumed one.
    The diff is then clean while the runs differ. Take it after the restore,
    before the first step.

!!! warning "An unseeded library RNG"
    A random source the workload never seeds (numpy is common) differs on every
    pair. It blocks every comparison. Seed it, even when you think nothing uses
    it.

## Summary

- Set the five surfaces: randomness, library kernels, collectives, resume
  state, TP degree. Leave scheduling and user kernels alone.
- Read them back from inside each job, per rank, after the restore.
- Diff the two runs; only the declared difference may remain.
- The preflight sees settings, not code: a rounding change shows only in the
  capture.

## Go deeper

- [determinism_preflight.py](../../tools/determinism_preflight.py): the tool.
- [Report: what the preflight caught, and each resume piece left out alone](../../examples/pretraining_e2e/REPORT.md#the-determinism-preflight).
- [Training step](training-step.md): which settings and state matter for a full
  step, and what each fix costs.
- [Data parallel](data-parallel.md): the collective settings for gradients.
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md).
