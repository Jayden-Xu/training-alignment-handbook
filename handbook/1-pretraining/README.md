# 1. Pre-training: find where a run went wrong

## Where we are

[Part 0](../0-foundations/README.md) showed that a sum's bits depend on its
order, and that two runs can be compared bitwise at operator boundaries with
fingerprints. This part applies that to two training runs that should agree
and do not. Typical triggers:

- a loss spike or a slow drift after an upgrade, a migration or a config change;
- a run that does not resume to the same trajectory;
- a machine suspected of silent data corruption (SDC).

The workflow takes you from the symptom to a confirmed fix and a regression
check. It is the method OpGuard takes, and the
[pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)
applies it as an agent.

<figure class="fig">
<div class="fig-steps"><span class="fig-node">1. Choose the reference</span><span class="fig-node">2. Preflight and control</span><span class="fig-node">3. Capture: eager or compiled</span><span class="fig-node">4. First divergence</span><span class="fig-node is-key">5. Classify<span class="fig-note">⇢ hardware: SDC scan</span></span><span class="fig-node">6. Fix</span><span class="fig-node">7. Confirm</span><span class="fig-node">8. Regression</span></div>
</figure>

| Step | What it settles | Page |
|---|---|---|
| 1. Choose the reference run | Which question the comparison answers | this page |
| 2. Preflight and control | That the two runs differ only in the one thing under test | [Determinism and preflight](determinism-and-preflight.md); for the training step, [Training step](training-step.md); for gradient collectives, [Data parallel](data-parallel.md) |
| 2. Preflight and control: the data | That every rank sees the same token bits at every step | [Data alignment](data-alignment.md) |
| 3. Capture | Fingerprints at every model-level boundary, eager or compiled | [Localize with OpGuard](localize-with-opguard.md); compiled code: [Capture compiled code](capture-compiled-code.md) |
| 4. First divergence | The first boundary whose bits differ | [Localize with OpGuard](localize-with-opguard.md) |
| 5. Classify | Which kind of cause: setting, training-step part, collective, compiled code, kernel, hardware | this page, then the page for that class |
| 6. Fix | The smallest change that removes the cause | the page for that class; kernels in [part 3](../3-kernels/README.md) |
| 7. Confirm | The pair, captured again, agrees everywhere | [Localize with OpGuard](localize-with-opguard.md) |
| 8. Regression | The pair stays in a suite and keeps agreeing | [Regression](../4-operations/regression.md) |

## The idea: two runs, one difference

Hold everything equal except one declared thing, and the first bitwise
difference is either that thing or a fault. A tiny example: two runs are
identical except that the suspect's norm multiplies by its weight before
rounding to bf16. Their boundary fingerprints, in call order:

| Boundary | Reference | Suspect | Equal? |
|---|---|---|---|
| token embedding | e | e | yes |
| attention norm | n | n′ | **no: first divergence** |
| q projection | q | q′ | no |
| … every later boundary | | | no |

Everything after the norm differs too, because its input differs. So read only
the first difference, then walk outward from it to the cause.

**Rule.** Make the runs differ in one written-down thing, compare bitwise in
call order, and start at the first difference. This is a controlled experiment:
the reference run is the control.

## The steps

### Step 1: pick a reference run {#step-1-pick-a-reference-run}

**Goal:** a reference that isolates your question. One captured suspect run and
one reference run are enough, and the suspect need not be deterministic.

| Reference | Answers |
|---|---|
| Self-replay (same everything) | Is there residual nondeterminism, a race or SDC? |
| Last stable commit | Which change introduced the drift? |
| Config tweak (one flag off: overlap, offload, recompute, fusion) | Does this feature change the computation? |
| Library or driver version | Did the upgrade change the numerics? |
| Cross-stack (other framework, same model) | Where do two implementations disagree? |
| Single GPU, or a TP simulator at logical TP above 1 | Is the parallel implementation wrong? |

**How:** both runs use the same data, seeds and checkpoint. Write down the one
difference under test; the preflight diff takes it as `declared`.

**Success:** one sentence names the difference, and nothing else is meant to
differ.

### Step 2: preflight and control

**Goal:** remove the avoidable variation and prove it is gone. Five surfaces:
randomness and data order, library kernels, collectives, checkpoint resume, TP
degree. Do not fix scheduling or user kernels: if they change the bits, that is
the signal.

**How:** set the surfaces, then read them back from inside each job with
[determinism_preflight.py](../../tools/determinism_preflight.py) and diff the
two reports ([Determinism and preflight](determinism-and-preflight.md)). For
the parts of a training step see [Training step](training-step.md); for
gradient collectives, [Data parallel](data-parallel.md).

**Success:** the diff lists only the declared difference.

### Step 3: capture

**Goal:** fingerprints of the inputs and outputs of every model-level boundary,
on both runs, computed on the op's stream with no added synchronization.

**How:** [opguard_bridge.py](../../tools/opguard_bridge.py) runs both sides
through OpGuard's harness: discovery, plan, capture, comparison. For code under
`torch.compile` or CUDA graphs, eager capture does not show the bits production
computes; use [graph_probe.py](../../tools/graph_probe.py)
([Capture compiled code](capture-compiled-code.md)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Eager OpGuard capture | Every discovered boundary, with call sites | small per boundary (a few device kernels, no synchronization); the whole capture's overhead is not measured here | The run is eager, or an eager replay shows the symptom |
| Graph probe in compiled code | The bits compiled code and graph replays produce | small per probe inside a CUDA graph, large outside one; many probes on a small step add up to large | Production runs `torch.compile` or CUDA graphs |
| Full tensor dump | Values, not fingerprints | large (2x or more) | Only for a region a fingerprint already named |

**Success:** `opguard_bridge.coverage(run)` shows the region in question
captured, including the GEMMs.

### Step 4: take the first divergence

**Goal:** the first boundary, in call order, whose fingerprints differ.

**How:** the traces never line up position by position: overlap, fusion and
rewrites move events, and one side may have extra events. OpGuard pairs them
with order-preserving matching anchored on unambiguous boundaries and leaves
uncertain events as gaps. `opguard_bridge.first_divergence(...)` collects the
facts about the pivot ([Localize with OpGuard](localize-with-opguard.md)).

**Success:** a pivot with no unaligned event before it. An unpaired event is
not a difference.

### Step 5: classify {#step-5-classify}

**Goal:** name the kind of cause, because each kind has its own fix and page.

**How:** read the pivot's source line, its producers and the call-site
statuses; `opguard_bridge.suggest(...)` lists leads with reasons. The root
causes OpGuard has pinpointed in production fall into these classes:

| Class | Examples | Go to |
|---|---|---|
| Missing or wrong setting | deterministic mode unset for one backward kernel | [Determinism and preflight](determinism-and-preflight.md) |
| Racy or nondeterministic kernel | `torch.embedding` backward, a fused embedding backward, a Triton kernel missing a barrier | [Training step](training-step.md); [part 3](../3-kernels/README.md) |
| State bookkeeping | an optimizer step variable overwritten on resume | [Training step](training-step.md) |
| Topology-dependent reduction | reduce-scatter under a different NCCL topology | [Data parallel](data-parallel.md) |
| Compiled code | fusion or a graph bucket changes a reduction | [Capture compiled code](capture-compiled-code.md) |
| Stream misuse | an fp32 RMSNorm stream race, activation offload | [Localize with OpGuard](localize-with-opguard.md) |
| Layout assumption | QKV projections relying on implicit contiguity | [Localize with OpGuard](localize-with-opguard.md) |
| MoE routing | a histogram bin mismatch causing routing drift | [part 3: MoE](../3-kernels/moe.md) |
| Hardware SDC | a boolean scatter misrouting one token, seen as a shape mismatch many ops later | [SDC](sdc.md) |

**Success:** one hypothesis, with the evidence that points to it.

### Step 6: fix

**Goal:** the smallest change that removes the cause.

**How:** a setting goes back to the preflight; a training-step part follows
its checklist; a kernel goes through the
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)
and its [recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md).
OpGuard's harness skill `validate-hypothesis` tests one cause at a time.

**Success:** the intervention changes the pivot as the hypothesis predicts.

### Step 7: confirm

**Goal:** show that the fix, and nothing else, made the runs agree.

**How:** capture the pair again with the fix.

**Success:** no event differs, none is unaligned, and the final parameters are
bit-identical.

### Step 8: keep it as a regression check

**Goal:** the agreement stays true after the next upgrade.

**How:** add the pair, or the smallest case that reproduces it, to the
regression suite ([Regression](../4-operations/regression.md)), and record the
stack it holds on.

**Success:** the suite runs the case and reports it equal.

## What can go wrong

!!! warning "More than one difference between the runs"
    Two undeclared differences produce one first divergence that neither
    explains alone. You chase a mixture. Diff the preflight reports and remove
    every difference but the declared one before capturing.

!!! warning "Capture that misses the region"
    A boundary that was not instrumented cannot be the first divergence, so the
    pivot lands later, in the wrong operator. Check `coverage` before reading a
    first difference.

!!! warning "The pivot taken for the cause"
    The first difference is where the bits first show, not where they were
    made: a nondeterministic backward without a boundary shows first at the
    next consumer. Walk upstream from the pivot and test the hypothesis.

!!! warning "Disagreement between devices called SDC too early"
    Different drivers or libraries give different bits without any fault. You
    drain a healthy node. Rule out software differences first ([SDC](sdc.md)).

## Summary

- Choose a reference run that isolates one question, and write that one
  difference down.
- Control the avoidable nondeterminism and check it with the preflight diff.
- Capture model-level boundaries, check coverage, and take the first bitwise
  difference with no unaligned event before it.
- Classify the cause, fix it with the smallest change, confirm by capturing
  again, and keep the pair as a regression check.

## Go deeper

- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md):
  the workflow as an agent runs it.
- [opguard_bridge.py](../../tools/opguard_bridge.py),
  [determinism_preflight.py](../../tools/determinism_preflight.py),
  [graph_probe.py](../../tools/graph_probe.py),
  [sdc_scan.py](../../tools/sdc_scan.py).
- [Report: the OpGuard walkthrough, and the earlier version of this overview with the production numbers](../../examples/pretraining_e2e/REPORT.md).
- [Part 3](../3-kernels/README.md) when the cause is a kernel with no invariant
  implementation.
