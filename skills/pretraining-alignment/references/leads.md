# What the leads from `suggest` mean and how to confirm them

`opguard_bridge.suggest(facts, preflight)` returns a ranked list of leads.
Each lead has a `kind`, the evidence it rests on, a `check` to run, and a
reference into the handbook. No lead is a cause until an intervention confirms
it.

| Kind | What it rests on | How to confirm it |
|---|---|---|
| `preflight` | a blocking difference, or a surface left uncontrolled, in the preflight reports | make the two runs agree on that setting and capture again |
| `alignment` | unaligned events before the pivot | inspect the first unaligned event, and pair events (`review_pair`) only when you have evidence that they correspond |
| `rounding-point` | the same call records a different dtype on each side, or one side has call sites that the other never runs | write down both rounding sequences, move the suspect run's cast to where the reference casts, and rerun |
| `layout` | the same call sees different shapes on the two sides | compare per-vector fingerprints row by row, and check that the path is batch-invariant (L2) |
| `nondeterministic-backward` | a gradient consumer receives differing inputs although the forward events before it agree | run `bi_check.check_repeat` on the suspected backward op, and try strict deterministic mode |
| `consumer` | the inputs already differ | follow the difference upstream with `trace-upstream` |
| `operator-class` | a class name appears in the operator, its source line or its callers | run `check_repeat` (L1) and `check_batch` (L2) on the op |
| `rng` | a random op receives equal inputs and gives different outputs | compare the RNG digests in the preflight reports |
| `state` | a lookup receives equal indices and returns a different result | compare the optimizer and scheduler state after the restore |
| `data` | the first difference is at the data boundary | compare the seeds, the sampler state and the loader position |

In the measured case ([pretraining-opguard-e2e](../../../handbook/1-pretraining/localize-with-opguard.md)),
each planted cause came out as the first lead after the preflight leads. That
page also lists what each cause needed, beyond `suggest`, to be confirmed.
