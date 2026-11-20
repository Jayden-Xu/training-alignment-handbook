# 1. Pre-training: finding where a run went wrong

## Where we are

[Part 0](../0-foundations/README.md) showed that the bits of a sum depend on
the order of its additions, and that two runs can be compared bit for bit at
operator boundaries using fingerprints. This part applies those ideas to two
training runs that should agree but do not. Typical reasons to start are:

- a loss spike or a slow drift after an upgrade, a migration or a config
  change;
- a run that, after a resume, does not follow the same trajectory as before;
- a machine suspected of silent data corruption (SDC).

The workflow takes you from the symptom to a confirmed fix and a regression
check. It is the method OpGuard is built around, and the
[pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)
lets an agent carry it out.

<figure class="fx" data-fig="pre-workflow" aria-label="The pre-training workflow" markdown>
<div class="fx-stage" markdown>
<div class="fx-steps" markdown="span"><span class="fx-step" data-hot="0">[1. Choose the reference](#step-1-pick-a-reference-run)</span><span class="fx-step" data-hot="1">[2. Preflight and control](#step-2-preflight-and-control)</span><span class="fx-step" data-hot="2">[3. Capture](#step-3-capture)</span><span class="fx-step" data-hot="3">[4. First divergence](#step-4-take-the-first-divergence)</span><span class="fx-step is-key" data-hot="4">[5. Classify](#step-5-classify)</span><span class="fx-step" data-hot="6">[6. Fix](#step-6-fix)</span><span class="fx-step" data-hot="7">[7. Confirm](#step-7-confirm)</span><span class="fx-step" data-hot="8">[8. Regression](#step-8-keep-it-as-a-regression-check)</span></div>
<div class="fx-seq is-col" markdown="span"><span class="fx-node is-key fx-root" data-hot="4">5. Classify</span><span class="fx-fork" style="--n:4"><span class="fx-leg" data-hot="5"><span class="fx-when">backward, optimizer, loss or resume</span><span class="fx-node" data-hot="5 6">[Training step](training-step.md)</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">collectives that sum gradients</span><span class="fx-node" data-hot="5 6">[Data parallel](data-parallel.md)</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">fused code or graph buckets</span><span class="fx-node" data-hot="2 5 6">[Compiled code](capture-compiled-code.md)</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">the device itself</span><span class="fx-node" data-hot="5 6">[SDC](sdc.md)</span></span></span></div>
</div>
<ol class="fx-say">
<li>1. Choose the reference run. It decides which question the comparison answers.</li>
<li>2. Preflight and control: make sure the two runs differ only in the one thing you are testing.</li>
<li data-hold="1.5">3. Capture fingerprints at every boundary on both runs. If the job runs compiled code, read that page first.</li>
<li>4. Find the first boundary whose bits differ.</li>
<li>5. Classify: the first divergence points to a kind of cause.</li>
<li data-hold="1.5">Each common kind of cause has its own page.</li>
<li>6. Fix the cause there, with the smallest change that removes it.</li>
<li>7. Confirm: capture the pair again and check that it agrees everywhere.</li>
<li data-hold="2">8. Keep the pair in the regression suite, so that it goes on agreeing.</li>
</ol>
<figcaption>Figure: steps 1 to 8 are the main line. At step 5 the first divergence points to a kind of cause, and each common kind has its own page; fix the cause there, then confirm the fix (step 7). Compiled code also changes how you capture: if the production job runs compiled code, read its page before step 3.</figcaption>
</figure>

| Step | What the step establishes | Page |
|---|---|---|
| 1. Choose the reference run | Which question the comparison answers | this page |
| 2. Preflight and control | That the two runs differ only in the one thing you are testing | [Determinism and preflight](determinism-and-preflight.md); for the parts of a training step, [Training step](training-step.md); for the collectives that sum gradients, [Data parallel](data-parallel.md) |
| 2. Preflight and control, for the data | That every rank sees the same token bits at every step | [Data alignment](data-alignment.md) |
| 3. Capture | That both runs have fingerprints at every model-level boundary | [Localize with OpGuard](localize-with-opguard.md); if the job runs `torch.compile` or CUDA graphs, also [When the job runs compiled code](capture-compiled-code.md) |
| 4. First divergence | Which boundary is the first whose bits differ | [Localize with OpGuard](localize-with-opguard.md) |
| 5. Classify | Which kind of cause it is: a setting, a part of the training step, a collective, compiled code, a kernel or the hardware | this page, then the page for that kind of cause |
| 6. Fix | The smallest change that removes the cause | the page for that kind of cause, or [part 3](../3-kernels/README.md) for a kernel |
| 7. Confirm | That the pair, captured again, agrees everywhere | [Localize with OpGuard](localize-with-opguard.md) |
| 8. Regression | That the pair stays in a suite and goes on agreeing | [Regression](../4-operations/regression.md) |

The pages of this part follow the workflow. Step 2 comes first, in
[Determinism and preflight](determinism-and-preflight.md) and then in
[Data alignment](data-alignment.md). Next comes the core loop,
[Localize with OpGuard](localize-with-opguard.md), where you capture both runs,
find the first divergence, trace it to its cause, fix the cause and confirm
the fix. The pages after it cover the places a first divergence most often
points to: [Training step](training-step.md) for the backward pass, the
optimizer, the loss and resume; [Data parallel](data-parallel.md) for the
collectives that sum gradients;
[When the job runs compiled code](capture-compiled-code.md) for
`torch.compile` and CUDA graphs; and [SDC](sdc.md) for the hardware.

## The idea: make the two runs differ in exactly one thing {#the-idea-two-runs-one-difference}

If you keep everything equal except one thing that you have declared, the
first bitwise difference between the runs is either that thing or a fault.
Here is a small example. Two runs are identical except that, in the suspect
run, the norm multiplies by its weight before rounding to bf16. These are
their fingerprints at each boundary, in call order:

<figure class="fx" data-fig="first-divergence" aria-label="Two runs compared boundary by boundary">
<div class="fx-stage">
<div class="fx-grid" style="--cols:3">
<span class="fx-th">Boundary</span><span class="fx-th">Reference</span><span class="fx-th">Suspect</span><span class="fx-th">Equal?</span>
<span class="fx-label" data-hot="1">token embedding</span><span class="fx-node is-mono" data-hot="1">e</span><span class="fx-node is-mono" data-hot="1">e</span><span class="fx-pair is-ok" data-at="1">yes</span>
<span class="fx-label" data-hot="2 5">attention norm</span><span class="fx-node is-mono" data-hot="2 5">n</span><span class="fx-node is-mono to-bad" data-hot="2 5">n′</span><span class="fx-pair is-bad" data-at="2">no, the first divergence</span>
<span class="fx-label" data-hot="3">q projection</span><span class="fx-node is-mono" data-hot="3">q</span><span class="fx-node is-mono to-bad" data-hot="3">q′</span><span class="fx-pair" data-at="3">no</span>
<span class="fx-label" data-hot="4">… every later boundary</span><span class="fx-node is-mono" data-hot="4">…</span><span class="fx-node is-mono to-bad" data-hot="4">…</span><span class="fx-pair" data-at="4">no</span>
</div>
</div>
<ol class="fx-say">
<li>The two runs are identical except where the suspect's norm multiplies by its weight. Their fingerprints are listed in call order.</li>
<li>The token embedding is equal on both runs.</li>
<li data-hold="1.5">The attention norm differs. This is the first divergence.</li>
<li>The q projection differs too, because its input already differs.</li>
<li>So does every later boundary.</li>
<li data-hold="2">Read only the first difference, and trace from it to the cause.</li>
</ol>
<figcaption>Figure: when everything except one declared thing is held equal, the first bitwise difference is either that thing or a fault.</figcaption>
</figure>

Everything after the norm differs as well, because its input already differs.
So read only the first difference, and trace from it to the cause.

**Rule.** Make the two runs differ in exactly one thing, and write that thing
down. Compare them bit for bit in call order, and start from the first
difference. This is a controlled experiment, and the reference run is its
control.

## The steps

### Step 1: choose a reference run {#step-1-pick-a-reference-run}

**Goal:** choose a reference run that isolates your question. One captured
suspect run and one reference run are enough, and the suspect run does not
have to be deterministic.

| Reference run | The question it answers |
|---|---|
| The same job run again, with everything the same | Is there residual nondeterminism, a race or SDC? |
| The last commit that was stable | Which change introduced the drift? |
| The same config with one feature turned off (overlap, offload, recompute or fusion) | Does this feature change the computation? |
| The previous library or driver version | Did the upgrade change the numerics? |
| The same model in another framework | Where do the two implementations disagree? |
| One GPU, or a TP simulator that keeps logical TP above 1 | Is the parallel implementation wrong? |

**How:** give both runs the same data, seeds and checkpoint. Write down the
one difference you are testing; you pass it to the preflight diff as
`declared`.

**Success:** you can name the difference in one sentence, and nothing else is
meant to differ.

### Step 2: control the avoidable variation and check it {#step-2-preflight-and-control}

**Goal:** remove the avoidable variation and prove that it is gone. It comes
from five surfaces: randomness and data order, library kernels, collectives,
checkpoint resume and the TP degree. Do not pin scheduling or user kernels; if
they change the bits, that change is the signal you are looking for.

**How:** set the five surfaces, then read them back from inside each job with
[determinism_preflight.py](../../tools/determinism_preflight.py) and diff the
two reports ([Determinism and preflight](determinism-and-preflight.md)). For
the parts of a training step, see [Training step](training-step.md); for the
collectives that sum gradients, see [Data parallel](data-parallel.md).

**Success:** the diff lists only the declared difference.

### Step 3: capture fingerprints on both runs {#step-3-capture}

**Goal:** record fingerprints of the inputs and outputs at every model-level
boundary on both runs. They are computed on the operator's own stream and add
no synchronization.

**How:** [opguard_bridge.py](../../tools/opguard_bridge.py) runs both sides
through OpGuard's harness, which discovers the boundaries, plans where to
instrument, captures and compares. If the code runs under `torch.compile` or
CUDA graphs, an eager capture does not show the bits that production
computes; use [graph_probe.py](../../tools/graph_probe.py) instead
([When the job runs compiled code](capture-compiled-code.md)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| An eager OpGuard capture | Every boundary it discovers, with its call site | small at each boundary (a few device kernels and no synchronization); the overhead of a whole capture is not measured here | The run is eager, or replaying it in eager shows the symptom |
| A graph probe inside compiled code | The bits that compiled code and graph replays actually produce | small for each probe inside a CUDA graph, large outside one; many probes on a small step add up to large | Production uses `torch.compile` or CUDA graphs |
| A full tensor dump | The values themselves, not fingerprints | large (2x or more) | Only for a region that a fingerprint has already named |

**Success:** `opguard_bridge.coverage(run)` shows that the region you care
about, GEMMs included, was captured.

### Step 4: find the first divergence {#step-4-take-the-first-divergence}

**Goal:** find the first boundary, in call order, whose fingerprints differ.

**How:** the two traces never line up position by position, because overlap,
fusion and rewrites move events around, and one side may have events that the
other lacks. OpGuard therefore pairs the events with an order-preserving
match, anchored on boundaries it can identify without doubt, and it leaves the
uncertain events unpaired, as gaps. The first pair that differs is the
**pivot**. `opguard_bridge.first_divergence(...)` collects the facts about it
([Localize with OpGuard](localize-with-opguard.md)).

**Success:** you have a pivot with no unaligned event before it. Remember that
an unpaired event is not a difference.

### Step 5: decide what kind of cause it is {#step-5-classify}

**Goal:** name the kind of cause, because each kind has its own fix and its
own page.

**How:** read the pivot's source line, the operators that produced its
inputs, and the status of each call at its call site.
`opguard_bridge.suggest(...)` lists possible leads, each with its reason. The
root causes that OpGuard has pinpointed in production fall into these
classes:

| Kind of cause | Example | Where to go |
|---|---|---|
| A setting that is missing or wrong | deterministic mode not set for one backward kernel | [Determinism and preflight](determinism-and-preflight.md) |
| A racy or nondeterministic kernel | the backward of `torch.embedding`, a fused embedding backward, a Triton kernel that lacks a barrier | [Training step](training-step.md); [part 3](../3-kernels/README.md) |
| State that is restored wrongly | an optimizer's step variable overwritten on resume | [Training step](training-step.md) |
| A reduction that depends on the topology | a reduce-scatter under a different NCCL topology | [Data parallel](data-parallel.md) |
| Compiled code | a fusion or a graph bucket that changes a reduction | [When the job runs compiled code](capture-compiled-code.md) |
| Streams used wrongly | a stream race in an fp32 RMSNorm; activation offload | [Localize with OpGuard](localize-with-opguard.md) |
| A wrong assumption about layout | QKV projections that silently assume a contiguous tensor | [Localize with OpGuard](localize-with-opguard.md) |
| MoE routing | mismatched histogram bins that make the routing drift | [part 3: MoE](../3-kernels/moe.md) |
| Silent data corruption in hardware | a boolean scatter that sends one token to the wrong place, which shows up as a shape mismatch many operators later | [SDC](sdc.md) |

**Success:** you have one hypothesis and the evidence that points to it.

### Step 6: fix the cause {#step-6-fix}

**Goal:** make the smallest change that removes the cause.

**How:** the fix depends on the kind of cause. Correct a setting in the
preflight. Fix a part of the training step with that page's checklist. Fix a
kernel with the
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)
and its [recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md).
OpGuard's harness skill `validate-hypothesis` tests one cause at a time.

**Success:** the change affects the pivot in the way the hypothesis predicts.

### Step 7: confirm the fix {#step-7-confirm}

**Goal:** show that the fix, and nothing else, made the runs agree.

**How:** capture the pair again, with the fix applied.

**Success:** no event differs, no event is unaligned, and the final
parameters are bit-identical.

### Step 8: keep the pair as a regression check {#step-8-keep-it-as-a-regression-check}

**Goal:** make sure the runs still agree after the next upgrade.

**How:** add the pair, or the smallest case that reproduces it, to the
regression suite ([Regression](../4-operations/regression.md)), and record the
stack on which the agreement holds.

**Success:** the suite runs the case and reports it equal.

## What can go wrong

!!! warning "More than one difference between the runs"
    Two undeclared differences produce one first divergence that neither of
    them explains on its own, and you end up chasing a mixture of causes. Diff
    the preflight reports, and remove every difference except the declared
    one before you capture.

!!! warning "A capture that misses the region"
    A boundary that was not instrumented cannot be the first divergence, so
    the pivot lands later, in the wrong operator. Check `coverage` before you
    read a first difference.

!!! warning "Mistaking the pivot for the cause"
    The first difference is where wrong bits first become visible, not where
    they were produced. A nondeterministic backward pass, for example, has no
    boundary of its own, so its effect shows first at the next operator that
    consumes its output. Trace upstream from the pivot, and test the
    hypothesis.

!!! warning "Calling a disagreement between devices SDC too early"
    Different drivers or libraries give different bits without any fault. If
    you call that SDC, you take a healthy node out of service. Rule out
    software differences first ([SDC](sdc.md)).

## Summary

- Choose a reference run that isolates one question, and write down the one
  difference between the runs.
- Control the avoidable nondeterminism, and check it with the preflight diff.
- Capture the model-level boundaries, check the coverage, and take the first
  bitwise difference that has no unaligned event before it.
- Decide what kind of cause it is, fix it with the smallest change, confirm
  the fix by capturing again, and keep the pair as a regression check.

## Go deeper

- The [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)
  runs this workflow as an agent.
- The tools: [opguard_bridge.py](../../tools/opguard_bridge.py),
  [determinism_preflight.py](../../tools/determinism_preflight.py),
  [graph_probe.py](../../tools/graph_probe.py) and
  [sdc_scan.py](../../tools/sdc_scan.py).
- [Report: the OpGuard walkthrough, and an earlier version of this overview that includes the production numbers](../../examples/pretraining_e2e/REPORT.md).
- [Part 3](../3-kernels/README.md) covers the case where the cause is a kernel
  that has no invariant implementation.
