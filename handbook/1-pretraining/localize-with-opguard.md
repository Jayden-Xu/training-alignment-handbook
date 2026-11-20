# Localize with OpGuard

## Where we are

This page is the core loop of the [pre-training workflow](README.md). It
covers steps 3, 4, 6 and 7: capture both runs, find the first divergence, fix
the cause and confirm the fix. You have a reference run and a suspect run that
differ only in the declared thing ([preflight](determinism-and-preflight.md))
and that read the same batches ([data alignment](data-alignment.md)). What you
do not know yet is where the two runs first part, and why. Deciding what kind
of cause it is (step 5) is covered on the [workflow page](README.md#step-5-classify).
The pages after this one cover the causes that a first divergence most often
points to: [the training step](training-step.md),
[the gradient collectives](data-parallel.md),
[compiled code](capture-compiled-code.md) and [the hardware](sdc.md).

## The idea: the first difference tells you where to start, not what the cause is

OpGuard records a fingerprint at every model-level boundary in both runs,
pairs up the two traces, and reports the first pair that differs. That pair is
called the **pivot**.

<figure class="fx" data-fig="pairing" aria-label="Pairing two traces and finding the pivot">
<div class="fx-stage">
<div class="fx-grid" style="--cols:5">
<span class="fx-label">reference</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-empty" data-hot="3"></span><span class="fx-node" data-hot="4">attn</span><span class="fx-node">mlp</span>
<span></span><span class="fx-pair is-ok" data-at="1">equal</span><span class="fx-pair is-ok" data-at="2">equal</span><span class="fx-pair" data-at="3">a gap, left unpaired</span><span class="fx-pair is-bad" data-at="4">first difference, the pivot</span><span></span>
<span class="fx-label">suspect</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-ghost" data-hot="3">extra cast</span><span class="fx-node to-bad" data-hot="4">attn</span><span class="fx-node">mlp</span>
</div>
</div>
<ol class="fx-say">
<li>Both traces are listed in call order, with a fingerprint at every boundary.</li>
<li>embed pairs with embed, and they are equal.</li>
<li>norm pairs with norm, and they are equal.</li>
<li data-hold="1.5">The suspect has an extra cast. The order-preserving match leaves it unpaired: it is a gap, not a difference.</li>
<li data-hold="2">attn differs. The first pair that differs is the pivot; trace from it to the cause.</li>
</ol>
<figcaption>Figure: both traces in call order. The match pairs the equal boundaries, leaves the extra cast unpaired, and reports the first pair that differs.</figcaption>
</figure>

Two things in this picture matter.

- **A gap is not a difference.** The suspect has an event that the reference
  does not have. The order-preserving match leaves it unpaired instead of
  forcing it into a pair, and only pairs whose fingerprints differ are
  reported.
- **The pivot is not always the cause.** Here is a small example. Attention's
  backward pass adds up dQ with atomics, and it has no boundary of its own. The
  first boundary that sees its result is the gradient norm, after the whole
  backward pass. So the pivot is the norm, while the cause lies upstream. What
  points to the cause is the call-site status, which says for each
  parameter's gradient whether it is equal or different. In the last block
  only the gradient of the q projection differs, and that gradient is dQ.

**Rule.** Take the first bitwise difference that has no unaligned event before
it. From there, trace upstream through the operators that produced its inputs,
and downstream to the symptom, and test the suspected cause with the smallest
intervention you can make. This applies part 0's advice to compare at operator
boundaries: a boundary localizes the first visible effect, and reasoning about
the producers localizes the cause.

## The step

### 1. Check that the region you suspect is captured

**Goal:** the region you care about is captured.

**How:** `opguard_bridge.coverage(run)` reads OpGuard's discovery and plan
files, and lists which boundaries were discovered and which were planned for
capture.

**Success:** the operators you suspect, GEMMs included, are planned.

### 2. Capture both runs

**Goal:** fingerprints of both runs at every planned boundary.

**How:** `opguard_bridge.run_pair(reference, suspect, ...)` in
[opguard_bridge.py](../../tools/opguard_bridge.py) runs both sides through
OpGuard's harness, which discovers the boundaries, plans the
instrumentation, captures and compares. Discovery and capture each launch both
sides once. If the production job runs `torch.compile` or CUDA graphs, this
capture runs it in eager mode instead, and it may not show the bits that
production computes; see
[When the job runs compiled code](capture-compiled-code.md).

**Success:** both traces exist, and the summary counts the events that are
equal and the events that differ.

### 3. Collect the facts about the first divergence

**Goal:** everything that is known about the pivot.

**How:** `opguard_bridge.first_divergence(export, ...)` calls the harness's
`trace_summary`, `first_diff`, `inspect_event`, `compare_events` and
`trace_neighbors`. It returns the operator, the source line, the call number,
which fields differ, the inputs and outputs on both sides, the regions that
differ, the unaligned events before the pivot, and the status of every call at
the pivot's call site. It makes no judgment of its own.

**Success:** you have a pivot whose `unaligned_before` is zero.

### 4. Trace from the pivot to the cause, and test it

**Goal:** the cause, with the evidence for it.

**How:** use OpGuard's harness skills. `investigate-first-diff` owns the
boundary and the record of the evidence, `trace-upstream` follows the
producers when the inputs already differ, and `validate-hypothesis` tests one
cause. `opguard_bridge.suggest(facts)` maps the pivot onto the handbook's
operator classes and
[recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md),
and gives each lead with its reason.

| Evidence | What it tells you | Cost class | When to use it |
|---|---|---|---|
| The pivot's source line and dtypes | A moved rounding point or a cast, directly | free | The output dtype or shape differs at the pivot |
| The call-site statuses (one letter per call) | Which calls of one call site differ, and so which parameter or which step | free | The pivot consumes the output of many producers (for example, a norm over the gradients) |
| Equal inputs but a different output | The operator's own state or weights changed earlier | free | Lookups, and other operators that look stateless |
| The `suggest` leads | The operator class and its recipe, with reasons | free | Before you start reading kernels |
| A recapture of one region's full tensors | The values, to see how they differ | large (2x or more) for that region | A fingerprint is equal but you suspect a collision, or you need the size of the difference |
| Repeating one operator on fixed inputs | Whether that operator is nondeterministic | small | Your hypothesis is a nondeterministic kernel |

**Success:** you have one hypothesis, and a single intervention confirms it.

### 5. Fix the cause and confirm the fix

**Goal:** show that the fix, and nothing else, makes the runs agree.

**How:** apply the fix (a setting, a part of the training step, or a kernel
through the
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)),
then call `run_pair` again.

**Success:** every event is equal, no event is unaligned, and the final
parameters are bit-identical.

In a known-answer walkthrough with three planted causes (a moved rounding
point in a norm, an attention backward running under `warn_only`, and a
resume without the optimizer's step counter), on one GPU and one rank, OpGuard
placed each first difference with no unaligned event before it. Two of the
three pivots were not the cause. After each fix, the rerun was equal
everywhere and the final parameters were identical
([report](../../examples/pretraining_e2e/REPORT.md#what-this-shows)). This is a
known-answer demonstration, not a blind evaluation.

## What can go wrong

!!! warning "`torch.nn.Linear` hides the GEMMs"
    The GEMM is launched from torch's own source file, so the line that calls
    the projection is discovered but not planned. No projection gets
    captured, and the pivot lands later. Call `F.linear` in the model code,
    and check `coverage`.

!!! warning "One-letter names in the model code"
    OpGuard's instrumenter matches call sites by the names around them, and it
    ignores one-letter names. Attention and the projections are then planned
    but never wrapped. Use descriptive names, and check `coverage`.

!!! warning "A boundary placed on a trailing cast"
    On a line like `q = q_norm(x).float()`, the cast is not the computation,
    yet the pivot then names the cast. Make the operator (`q_norm`) the
    boundary.

!!! warning "Reading an unpaired event as a difference"
    One side may have extra events, from a fusion or a rewrite. Forcing them
    into pairs invents differences that are not there. Leave unpaired events
    as gaps.

!!! warning "Mistaking the pivot for the cause"
    A producer without a boundary (a backward pass, an optimizer step, a
    resume) shows up first at the next operator that consumes its output. You
    then fix the wrong operator. Trace upstream and test the hypothesis before
    you fix anything.

## Summary

- Check coverage first: an operator that was not captured cannot be the
  pivot.
- The pivot is the first pair that differs and has no unaligned event before
  it; gaps are not differences.
- The pivot is where you start. Use the source line, the call-site statuses
  and the inputs to trace from it to the cause.
- Confirm a fix by capturing again: every event equal, none unaligned, and the
  parameters bit-identical.

## Go deeper

- [Report: the OpGuard walkthrough](../../examples/pretraining_e2e/REPORT.md)
  describes the setup, what OpGuard reported for each pair, the fixes and
  reruns, the cost of each investigation, the coverage, and the checks of the
  fingerprints.
- The code is in [pretraining_e2e.py](../../examples/pretraining_e2e/pretraining_e2e.py)
  and [opguard_bridge.py](../../tools/opguard_bridge.py).
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md).
- [When the job runs compiled code](capture-compiled-code.md) covers runs that
  use `torch.compile` or CUDA graphs.
