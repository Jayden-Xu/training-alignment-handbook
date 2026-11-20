---
name: pretraining-alignment
description: Find where two pre-training runs stop agreeing bit for bit and why. Choose the reference run, check and diff the determinism settings of both runs, capture both with OpGuard, take the first divergence, hand it to OpGuard's harness skills, map the operator to the handbook's operator classes, fix it through a determinism setting or the batch-invariant-kernel skill, confirm the identical prefix extends, and record a regression entry. Use when a training run went wrong (loss spike, drift after an upgrade or config change, a resume that does not continue the same trajectory) or when two runs should agree and do not.
---

# Pre-training alignment

Own one question: two training runs that should agree bit for bit do not.
Return where they first stop agreeing, the operator class and cause (or a
bounded inconclusive result), the fix with its cost, and a regression entry.

The method is [part 1](../../handbook/1-pretraining/README.md). The measured
walkthrough is the [pre-training case](../../handbook/1-pretraining/localize-with-opguard.md).
Tools:
- [tools/determinism_preflight.py](../../tools/determinism_preflight.py):
  `report`, `findings`, `diff`;
- [tools/opguard_bridge.py](../../tools/opguard_bridge.py): `run_pair`,
  `first_divergence`, `coverage`, `suggest`, fingerprints, `TraceWriter`;
- OpGuard itself (`python -m opguard.tools`, `opguard-harness`), which this
  skill drives and never modifies.

## Who owns what

| Step | Owner | Returns |
|---|---|---|
| Reference, preflight, diff | this skill | the question, both reports, the diff |
| Capture and comparison | OpGuard backend, through the harness (`run_pair`) | per-rank Perfetto exports, `verdict: not_evaluated` |
| First divergence facts | OpGuard harness read-only tools (`first_divergence` calls them) | pivot event, previous ok event, unaligned counts, fingerprints |
| Correspondence, upstream walk | OpGuard skill `investigate-first-diff`, then `trace-upstream` | a narrowed boundary, or the smallest missing observation |
| Testing a cause | OpGuard skill `validate-hypothesis` | supported, refuted or inconclusive, with the control |
| Operator class, recipe | this skill (`suggest` gives leads) | class, recipe, fix options |
| Kernel fix | skill [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | changed kernel, verification, cost |
| Confirmation, regression entry | this skill | rerun with no divergence, entry |

OpGuard produces facts. Judgments (which events correspond, which difference
is first, what caused it) are yours and carry evidence. `suggest` gives leads,
never a verdict.

## 1. Choose the reference

Pick it for the question (part 1, step 1 table): self-replay for residual
nondeterminism or SDC, the last good commit for a regression, one flag off for
a feature, the old version for an upgrade. Both runs use the same data, seeds
and checkpoint. Write down the one difference under test.

Evidence: both launch commands, the commit of each, the difference under test.

## 2. Preflight and diff

In each run, after the model, optimizer, scheduler and loader are built and any
checkpoint is restored, just before the first step:

```python
rep = determinism_preflight.report(optimizer=opt, dataloader=loader, scheduler=sched,
                                   autocast=amp, model=model)
```

Then `determinism_preflight.diff(rep_a, rep_b, declared=[...])`, declaring the
difference under test (for example `launch.argv`, `stack.torch`).

- `bitwise_comparison: not_meaningful`: a blocking difference outside the
  declared one. Fix it or declare it before capturing; otherwise the first
  divergence only shows that configuration.
- `at_risk` (from `findings`): surfaces left uncontrolled on a side. The
  common one: `warn_only=True` keeps SDPA's backward nondeterministic
  (measured). Fix these first; a self-replay may then already agree.
- On a resume, a difference in `optimizer.state`, `scheduler`, `seeds` or the
  loader position after the restore is the bug itself (measured: the step
  counter showed up here before any capture).

Evidence: both report files and the diff.

### Data at the loader's exit

On every rank of both runs, record each batch as the loader returns it: `rec = data_fingerprint.Recorder(dir, rank=dp_rank, meta={...})`, then `rec.record(step, batch)` ([data alignment](../../handbook/1-pretraining/data-alignment.md)). `python tools/data_fingerprint.py compare A B` (`--global` when the DP sizes differ) must report no differing record before capture; otherwise the first divergence is the data. Fix by its class. Evidence: both record directories and the compare output.

## 3. Capture both runs with OpGuard

Use dedicated checkouts per side (`stage_checkouts`), then `run_pair`: four
workload launches (discovery and capture on each side) plus a comparison.
Keep runs short: a few steps, one or two GPUs. Give both sides the same
`CUDA_VISIBLE_DEVICES` index: OpGuard compares device strings.

Then `coverage(run)`: call sites discovered in the checkout but not planned
for instrumentation have no boundary record. Measured: with `torch.nn.Linear`
the GEMMs are launched from torch's `linear.py`, outside the checkout, so they
are not captured; with a Linear that calls `F.linear` in model code they are.
OpGuard captures eager execution only. Boundaries inside `torch.compile` or a
CUDA graph need fingerprints taken there (the bridge's plain-torch functions
work inside both) and written with `TraceWriter`.

Evidence: experiment ID, stage list, launches and wall time, coverage.

## 4. Take the first divergence

`facts = first_divergence(report, root, og)`. Read, in this order:
1. `status_counts` and `unaligned_before`: unaligned events are gaps, not
   differences. With unaligned events before the pivot, an earlier divergence
   may be hidden.
2. `pivot`: operator, file, line, call number, `diff_fields`, inputs and
   outputs on both sides, differing regions. If inputs already differ, the
   pivot is a consumer; the origin is upstream.
3. `site_statuses` / `sites`: which calls of a site agree and which differ.
   A run of `o` then `d` at a gradient consumer means some gradients agree:
   the origin is a backward op feeding only the ones that differ.
4. `previous_ok`: the last agreeing event, not necessarily a producer.

No diff is not proof of equivalence: check coverage, and compare final
parameters with `bi_check.bit_equal`.

## 5. Hand over to OpGuard's harness skills

In OpGuard's `harness/skills/`:
- `investigate-first-diff`: owns the boundary and evidence record. Returns a
  root-cause explanation or a bounded inconclusive result.
- `trace-upstream`: when inputs already differ or the producer is unclear.
  Returns a narrowed boundary, a candidate mechanism, or the missing
  observation.
- `validate-hypothesis`: one candidate cause, one discriminating
  intervention. Returns supported / refuted / inconclusive with the control.

`suggest(facts, preflight=diff)["handoff"]` says which one to enter.

## 6. Map to an operator class

`suggest` maps the pivot to the handbook's classes
([part 3](../../handbook/3-kernels/README.md),
[operator recipes](../batch-invariant-kernel/references/operator-recipes.md))
and to part 1's root-cause patterns, with reasons. See
[references/leads.md](references/leads.md) for what each lead kind means and
what it took to confirm it in the measured case.

## 7. Fix

Cheapest first:
1. a determinism setting or restore fix (preflight finding, missing state);
2. a rounding point aligned with the reference (both sides round at the same
   place);
3. a batch-invariant or deterministic kernel through
   [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md).
Record the cost next to the fix.

## 8. Confirm

Rerun the same pair through OpGuard with the fix. Done when OpGuard labels no
event `diff`, no event is unaligned, and final parameters are bit-identical.
"The identical prefix extends" means the first divergence moved later or
disappeared over the same steps; if it only moved, return to step 4.

## 9. Regression entry

Add a check to an example script through `tools/bi_report.py`: the pair as
`expect="fail"` (documented variance) and the fixed pair as `expect="pass"`.
Write a case entry: scenario, symptom, pivot, class, fix, cost, launches.

## When to stop

- Supported cause plus a confirmed rerun: done.
- Preflight shows a blocking difference that is the cause (resume state,
  `warn_only`): fix and confirm; a capture may be unnecessary.
- Pivot preceded by unaligned events you cannot pair, or coverage misses the
  region: report the narrowed boundary and the smallest missing observation
  (a boundary to add, a rank, a step). Do not guess.
- Launch budget used: report what is established and what is not.

## Guardrails

- Bitwise only. No tolerance, no rounding of outputs.
- Never modify OpGuard or the workload's numerics to make captures agree.
- An unpaired event is a gap, not a difference.
- Equal XOR fingerprints are not proof of equal tensors; a zero fingerprint
  is never evidence.
- Count launches: one `run_pair` is four workload launches.
