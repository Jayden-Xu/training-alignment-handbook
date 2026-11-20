---
name: pretraining-alignment
description: Find the first point where two pre-training runs stop agreeing bit for bit, and find out why. Choose the reference run; check the determinism settings of both runs and diff them; capture both runs with OpGuard; take the first divergence and hand it to OpGuard's harness skills; map the operator to the handbook's operator classes; fix it with a determinism setting or with the batch-invariant-kernel skill; confirm that the identical prefix of the two runs now extends further; and record a regression entry. Use when a training run went wrong (a loss spike, drift after an upgrade or a config change, a resume that does not continue the same trajectory), or when two runs that should agree do not.
---

# Pre-training alignment

This skill answers one question: two training runs should agree bit for bit,
and they do not. Return:

- where they first stop agreeing;
- the operator class and the cause, or, if the cause cannot be established, an
  inconclusive result that states clearly what is and is not known;
- the fix and its cost;
- a regression entry.

The method is explained in [part 1](../../handbook/1-pretraining/README.md),
and the [pre-training case](../../handbook/1-pretraining/localize-with-opguard.md)
walks through it with measurements. The tools are:

- [tools/determinism_preflight.py](../../tools/determinism_preflight.py), with
  `report`, `findings` and `diff`;
- [tools/opguard_bridge.py](../../tools/opguard_bridge.py), with `run_pair`,
  `first_divergence`, `coverage`, `suggest`, the fingerprint functions and
  `TraceWriter`;
- OpGuard itself (`python -m opguard.tools`, `opguard-harness`), which this
  skill drives and never modifies.

## Who does each part of the work

| Step | Done by | Returns |
|---|---|---|
| Choosing the reference, preflight and diff | this skill | the question under test, both reports and the diff |
| Capture and comparison | the OpGuard backend, driven through the harness (`run_pair`) | Perfetto exports for each rank, with `verdict: not_evaluated` |
| The facts about the first divergence | the read-only tools of the OpGuard harness, which `first_divergence` calls | the pivot event, the last event that agreed before it, the counts of unaligned events, and fingerprints |
| Deciding which events correspond, and walking upstream | the OpGuard skill `investigate-first-diff`, then `trace-upstream` | a narrower boundary, or the smallest observation still missing |
| Testing a cause | the OpGuard skill `validate-hypothesis` | supported, refuted or inconclusive, together with the control |
| Finding the operator class and its recipe | this skill, with leads from `suggest` | the class, the recipe and the fix options |
| Fixing a kernel | the [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) skill | the changed kernel, its verification and its cost |
| Confirming the fix and adding a regression entry | this skill | a rerun with no divergence, and the entry |

OpGuard produces facts. The judgments are yours: which events correspond,
which difference comes first, and what caused it. Each judgment must carry its
evidence. `suggest` gives leads, never a verdict.

## 1. Choose the reference run

Choose the reference according to the question you are asking (see the table
in part 1, step 1). For residual nondeterminism or SDC, replay the same run
against itself. For a regression, use the last good commit. For a new feature,
use the same run with that one flag off. For an upgrade, use the old version.
Both runs use the same data, seeds and checkpoint. Write down the single
difference under test.

Evidence to keep: both launch commands, the commit of each run, and the
difference under test.

## 2. Preflight both runs and diff the reports

In each run, call the preflight just before the first step, after the model,
optimizer, scheduler and loader are built and any checkpoint has been
restored:

```python
rep = determinism_preflight.report(optimizer=opt, dataloader=loader, scheduler=sched,
                                   autocast=amp, model=model)
```

Then compare the two reports with
`determinism_preflight.diff(rep_a, rep_b, declared=[...])`, and declare the
difference under test (for example `launch.argv` or `stack.torch`).

- `bitwise_comparison: not_meaningful` means there is a blocking difference
  besides the declared one. Fix it or declare it before you capture; otherwise
  the first divergence will only show you that configuration difference.
- `at_risk` (from `findings`) lists the surfaces, the sources of
  nondeterminism, that one side leaves uncontrolled. The most common one is
  `warn_only=True`, which leaves SDPA's backward nondeterministic (measured).
  Fix these first; a self-replay may then agree with no further work.
- When the question is a resume, a difference after the restore in
  `optimizer.state`, `scheduler`, `seeds` or the loader position is the bug
  itself. In the measured case, the step counter showed up here before
  anything was captured.

Evidence to keep: both report files and the diff.

### Check the data at the loader's exit

On every rank of both runs, record each batch as the loader returns it. Create
a recorder with `rec = data_fingerprint.Recorder(dir, rank=dp_rank, meta={...})`
and call `rec.record(step, batch)` for each batch (see
[data alignment](../../handbook/1-pretraining/data-alignment.md)). Before you
capture, `python tools/data_fingerprint.py compare A B` must report no
differing record; add `--global` when the two runs use different DP sizes. If
a record differs, the first divergence is in the data, and you fix it
according to its class. Evidence to keep: both record directories and the
output of compare.

## 3. Capture both runs with OpGuard

Give each side its own checkout (`stage_checkouts`), then call `run_pair`. It
launches the workload four times, a discovery run and a capture run on each
side, and then compares the captures. Keep the runs short: a few steps on one
or two GPUs. Give both sides the same `CUDA_VISIBLE_DEVICES` index, because
OpGuard compares device strings.

Then call `coverage(run)`. It lists the call sites that discovery found in the
checkout but that were not planned for instrumentation; these have no boundary
record. In the measured case, with `torch.nn.Linear` the GEMMs are launched
from torch's `linear.py`, which lies outside the checkout, so they are not
captured. With a Linear whose model code calls `F.linear`, they are. OpGuard
captures eager execution only. For boundaries inside `torch.compile` or a CUDA
graph, take fingerprints there (the bridge's plain-torch functions work inside
both) and write them with `TraceWriter`.

Evidence to keep: the experiment ID, the list of stages, the number of
launches and the wall time, and the coverage report.

## 4. Read the first divergence

Call `facts = first_divergence(report, root, og)` and read the result in this
order:

1. `status_counts` and `unaligned_before`. An unaligned event is a gap in the
   pairing, not a difference. If there are unaligned events before the pivot,
   an earlier divergence may be hidden among them.
2. `pivot`, the first event that differs. It gives the operator, file, line,
   call number, `diff_fields`, the inputs and outputs on both sides, and the
   regions that differ. If the inputs already differ, the pivot only consumes
   a difference made earlier, and the origin lies upstream.
3. `site_statuses` / `sites`, which show which calls of a call site agree and
   which differ. If a gradient consumer shows a run of `o` followed by `d`,
   some gradients agree and others do not, so the origin is a backward op that
   feeds only the gradients that differ.
4. `previous_ok`, the last event that agreed. It is not necessarily the
   producer of the pivot's inputs.

Finding no difference does not prove that the runs are equivalent. Check the
coverage, and compare the final parameters with `bi_check.bit_equal`.

## 5. Hand the divergence to OpGuard's harness skills

OpGuard's `harness/skills/` has three skills for this:

- `investigate-first-diff` keeps the record of the boundary and its evidence.
  It returns a root-cause explanation, or an inconclusive result that states
  clearly what is and is not known.
- `trace-upstream` is for the case where the inputs already differ or the
  producer is unclear. It returns a narrower boundary, a candidate mechanism,
  or the observation that is missing.
- `validate-hypothesis` tests one candidate cause with one intervention that
  can tell the alternatives apart. It returns supported, refuted or
  inconclusive, together with the control.

`suggest(facts, preflight=diff)["handoff"]` tells you which of the three to
start with.

## 6. Map the pivot to an operator class

`suggest` maps the pivot to the handbook's operator classes
([part 3](../../handbook/3-kernels/README.md),
[operator recipes](../batch-invariant-kernel/references/operator-recipes.md))
and to the root-cause patterns of part 1, and gives its reasons.
[references/leads.md](references/leads.md) explains what each kind of lead
means and what it took to confirm it in the measured case.

## 7. Fix the cause

Try the cheapest fix first:

1. a determinism setting, or a fix to what is saved and restored (for a
   preflight finding or missing state);
2. a rounding point moved to match the reference, so that both sides round at
   the same place;
3. a batch-invariant or deterministic kernel, built with the
   [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) skill.

Record the cost next to the fix.

## 8. Confirm the fix

Rerun the same pair through OpGuard with the fix applied. You are done when
OpGuard labels no event `diff`, no event is unaligned, and the final
parameters are bit-identical. Over the same steps, a working fix moves the
first divergence later or makes it disappear; in both cases the identical
prefix of the two runs extends. If the divergence only moved later, return to
step 4.

## 9. Add a regression entry

Add a check to an example script with `tools/bi_report.py`. Register the
original pair with `expect="fail"`, because its variance is now documented,
and the fixed pair with `expect="pass"`. Then write a case entry that records
the scenario, the symptom, the pivot, the operator class, the fix, its cost
and the number of launches.

## When to stop

- When the cause is supported and a rerun confirms the fix, you are done.
- When the preflight shows a blocking difference that is itself the cause
  (resume state, `warn_only`), fix it and confirm. A capture may not be
  needed.
- When the pivot is preceded by unaligned events that you cannot pair, or the
  coverage misses the region, report the narrowed boundary and the smallest
  observation still missing (a boundary to add, a rank, a step). Do not guess.
- When the launch budget is used up, report what is established and what is
  not.

## Guardrails

- Compare bitwise only: no tolerance, and no rounding of outputs.
- Never modify OpGuard or the workload's numerics to make captures agree.
- An unpaired event is a gap, not a difference.
- Equal XOR fingerprints do not prove that the tensors are equal, and a zero
  fingerprint is never evidence.
- Count the launches: one `run_pair` launches the workload four times.
