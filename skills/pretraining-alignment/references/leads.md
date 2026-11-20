# Leads from `suggest` and how to confirm them

`opguard_bridge.suggest(facts, preflight)` returns ordered leads. Each has a
`kind`, the evidence it rests on, a `check` and a handbook reference. None is
a cause until an intervention confirms it.

| Kind | Rests on | Confirm with |
|---|---|---|
| `preflight` | a blocking difference or uncontrolled surface in the reports | make the runs agree on it and recapture |
| `alignment` | unaligned events before the pivot | inspect the first unaligned event; pair only with evidence (`review_pair`) |
| `rounding-point` | the same call records a different dtype, or one side has call sites the other never runs | write both rounding sequences; move the suspect's cast to the reference's place; rerun |
| `layout` | different shapes at the same call | per-vector fingerprints row by row; batch invariance of the path (L2) |
| `nondeterministic-backward` | a gradient consumer with differing inputs after agreeing forward events | `bi_check.check_repeat` on the suspected backward; strict deterministic mode |
| `consumer` | inputs already differ | `trace-upstream` |
| `operator-class` | a class name in the operator, its line or its callers | `check_repeat` (L1), `check_batch` (L2) on the op |
| `rng` | a random op with equal inputs and different outputs | RNG digests in the preflight reports |
| `state` | a lookup with equal indices and a different result | optimizer and scheduler state after the restore |
| `data` | the first difference is at the data boundary | seeds, sampler state, loader position |

In the measured case ([pretraining-opguard-e2e](../../../handbook/1-pretraining/localize-with-opguard.md))
each planted cause was the first non-preflight lead; what each needed beyond
`suggest` is listed there.
