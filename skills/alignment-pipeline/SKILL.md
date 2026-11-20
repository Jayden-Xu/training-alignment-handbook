---
name: alignment-pipeline
description: Route a bitwise-alignment question to the right workflow and run it end to end. Pre-training (two runs that should agree, a loss spike after a change, a resume that drifts, suspected SDC) goes preflight, reference, OpGuard capture or graph probes, first divergence, fix, confirm, regression entry. RL train–inference (rollout and trainer log-probs differ) goes model audit, per-op batch invariance, whole-model L1–L3, L5, sampling, L4 against the engine, online records, staleness/numeric split, offline localization, regression. Use first, before the more specific skills it calls.
---

# Alignment pipeline

This skill only routes. Each step names its entry condition, the tool or skill
that does the work, the evidence it must leave, and when to stop. The work and
its rules live in the linked skills and handbook parts; do not repeat them here.

Equality is bitwise everywhere: no tolerance, no rounding to force equality.
Every claim names its GPU and versions, or says "untested".

## Which flow

| The user has | Flow |
|---|---|
| Two training runs that should agree; a spike or drift after an upgrade, migration, config change or resume | A. Pre-training |
| A machine suspected of silent data corruption | A, step A0 only |
| Rollout log-probs that differ from the trainer's; an RL run that is silently off-policy | B. RL train–inference |
| A single operator without a batch-invariant implementation | [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) directly |
| A framework migration, checkpoint conversion, precomputed reference log-probs, or another pair | [part 5](../../handbook/5-scenarios/README.md); conversions: [tools/ckpt_compare.py](../../tools/ckpt_compare.py) |
| A new stack (torch, Triton, NCCL, driver, CUDA, GPU) | [regression](../../handbook/4-operations/regression.md): run the suite first |

Before either flow, on a new machine or stack, check what was measured there:
[hardware coverage](../../handbook/4-operations/hardware.md). What is untested or open (no real
engine record hooks, no blind trial, FSDP emulated, ...) is listed in
[open items](../../handbook/open-items.md).

## A. Pre-training

| Step | Entry | Tool or skill | Evidence to leave | Stop when |
|---|---|---|---|---|
| A0. SDC | a device is suspected, or before a long run | [tools/sdc_scan.py](../../tools/sdc_scan.py) (`--quick`, then fleet and golden); in-job `sdc_guard`, `cross_rank_check` ([case](../../handbook/1-pretraining/sdc.md)) | the scan's result JSON naming device, op and repeat | a device disagrees within its group (explain software differences first), or all pass |
| A1. Preflight | always, both runs | [tools/determinism_preflight.py](../../tools/determinism_preflight.py) `report` per rank after restore; `diff(a, b, declared=[...])` | both reports and the diff, with only the declared difference left | an undeclared difference that makes a bitwise comparison meaningless: fix the setting, rerun |
| A1'. Data | preflight clean | [tools/data_fingerprint.py](../../tools/data_fingerprint.py): `Recorder(dir, rank)` and `record(step, batch)` at the loader's exit on both runs; `compare` (`--global` when the DP sizes differ) ([page](../../handbook/1-pretraining/data-alignment.md)) | both record directories and the compare result | a record differs: fix it by its class (order, samples, packing, masks) and compare again |
| A2. Reference | preflight clean | [part 1, step 1](../../handbook/1-pretraining/README.md#step-1-pick-a-reference-run) | the one difference under test, written down | — |
| A3. Capture | eager run | OpGuard through [tools/opguard_bridge.py](../../tools/opguard_bridge.py) `run_pair`; check `coverage(run)` | both traces and the coverage report | coverage misses the region in question: fix wrapping first |
| A3'. Capture | `torch.compile` or CUDA graphs | [tools/graph_probe.py](../../tools/graph_probe.py) with `isolate=True` ([case](../../handbook/1-pretraining/capture-compiled-code.md)) | probe traces; a compile with and without probes giving the same bits | probes change the compiled bits: isolate them |
| A4. First divergence | two traces | `opguard_bridge.first_divergence` | the pivot facts: operator, line, call, differing fields, unaligned events before it | no event differs and final parameters are equal: the runs agree |
| A5. Cause | a pivot | [pretraining-alignment](../pretraining-alignment/SKILL.md) (steps 5–6) with OpGuard's harness skills; `suggest()` for leads | a hypothesis tested by the smallest intervention | the intervention removes the difference at the pivot |
| A6. Fix | a confirmed cause | a setting or state: [training step](../../handbook/1-pretraining/training-step.md), [DP and FSDP-style (emulated: flat gradient, reduce-scatter, sharded AdamW)](../../handbook/1-pretraining/data-parallel.md); a kernel: [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | the change and its cost (warm-up, synchronize, median) | — |
| A7. Confirm | a fix | capture the pair again | no differing event, none unaligned, final parameters bit-identical | confirmed; otherwise back to A4 |
| A8. Regression | confirmed | an example with `--quick`/`--json` in `bi-results-v1`, one entry in [suite/manifest.json](../../suite/manifest.json) ([regression](../../handbook/4-operations/regression.md#how-to-add-a-check-or-an-example)) | the example, its result file, a baseline | the suite passes on the stack |

## B. RL train–inference

Order matters: each level assumes the ones before it hold. Skip a step only
when its entry condition is false, and say so.

| Step | Entry | Tool or skill | Evidence to leave | Stop when |
|---|---|---|---|---|
| B1. Audit | the trainer's model | [tools/bi_audit.py](../../tools/bi_audit.py) on the model | the operator inventory and per-module L2 verdicts | every module is L2, or the variant ones are listed |
| B2. Per-op L2 | a variant operator | [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | its verification matrix and cost | each operator passes, or has a documented fallback |
| B3. Whole model L1–L3 | all ops L2 | the [whole model case](../../handbook/2-rl/whole-model.md) method: every serving mode, activation gradients | per-mode logprob equality | L1–L3 in every mode the engine uses |
| B4. L5 | the trainer and the engine use different TP, DP or EP degrees | [whole-model-tp](../../handbook/2-rl/parallelism.md), [training-dp-fsdp](../../handbook/1-pretraining/data-parallel.md), recipe 10 | equality across degrees, forward and backward | equal at every degree used |
| B5. Sampling | tokens are sampled, replayed, preempted or speculated | [tools/bi_sampling.py](../../tools/bi_sampling.py) ([case](../../handbook/2-rl/sampling.md)) | replay after recompute, rollback, n, speculative chunks, new process | tokens replay bitwise |
| B6. L4 | B3 holds on each side | [engine-parity](../engine-parity/SKILL.md) ([vLLM case](../../handbook/2-rl/engine-parity-vllm.md)); LoRA: [case](../../handbook/2-rl/lora.md) | both rounding sequences; per-op, per-layer equality | 0 logprobs differ, or each remaining difference is a recorded decision with its cost |
| B7. Online records | the RL job | OpGuard's `opguard_harness.rl_records.RecordWriter`: rollout batches, trainer log-probs, forwards, weight syncs, snapshots ([part 2](../../handbook/2-rl/README.md#synchronous-and-asynchronous-runs)) | record files per rank | — |
| B8. Split | records | `rl.weight_sync_check` first, then `rl.decompose` | unmatched weight rows (none), per-token `numeric` and `staleness`, mixed-version tokens set aside | `numeric` is bit-zero: the infrastructure is exonerated, the gap is staleness |
| B9. Localize | `numeric` ≠ 0 | OpGuard's `rl-alignment` skill (`harness/skills/rl-alignment`): `rl.replay_request`, recapture with `OPGUARD_XOR_ROWS=1`, `vectors.*`; [example](../../handbook/2-rl/rl-loop.md) | the divergent op with other-side evidence; its cause class | a cause tested by the smallest change; then back to B2, B4, B5 or B6 for the fix |
| B10. Regression | a fix | as A8 | the example and baseline | the suite passes on the stack |

## When to stop and ask

- A fix costs more than the user agreed to: report the measured cost and the
  cheaper levels that already hold.
- A difference is algorithmic (two exact forms of the same math): it is a
  choice, not a bug. Report both forms and their costs.
- The engine or kernel is closed and has no knob: report the fallback options
  (batch-invariant-kernel, step 2).
- A required regression check fails on a new stack: that is a finding to
  report, not to tune away.
