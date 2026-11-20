# Localize with OpGuard

## Where we are

This page is the core loop of the [pre-training workflow](README.md): steps 3,
4, 6 and 7, capture, first divergence, fix and confirm. You have a reference run
and a suspect run that differ only in the declared thing
([preflight](determinism-and-preflight.md)) and read the same batches
([data alignment](data-alignment.md)). What is still open: where the two runs
first part, and why. Classification (step 5) is on the
[workflow page](README.md#step-5-classify); the pages after this one cover the
causes a first divergence most often points to:
[the training step](training-step.md),
[the gradient collectives](data-parallel.md),
[compiled code](capture-compiled-code.md) and [the hardware](sdc.md).

## The idea: the first difference is where to start, not the cause

OpGuard records a fingerprint at every model-level boundary on both runs,
pairs the two traces, and reports the first pair that differs: the **pivot**.

<figure class="fx" data-fig="pairing" aria-label="Pairing two traces and finding the pivot">
<div class="fx-stage">
<div class="fx-grid" style="--cols:5">
<span class="fx-label">reference</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-empty" data-hot="3"></span><span class="fx-node" data-hot="4">attn</span><span class="fx-node">mlp</span>
<span></span><span class="fx-pair is-ok" data-at="1">equal</span><span class="fx-pair is-ok" data-at="2">equal</span><span class="fx-pair" data-at="3">gap: unpaired</span><span class="fx-pair is-bad" data-at="4">first difference: pivot</span><span></span>
<span class="fx-label">suspect</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-ghost" data-hot="3">extra cast</span><span class="fx-node to-bad" data-hot="4">attn</span><span class="fx-node">mlp</span>
</div>
</div>
<ol class="fx-say">
<li>Both traces in call order, a fingerprint at every boundary.</li>
<li>embed pairs with embed: equal.</li>
<li>norm pairs with norm: equal.</li>
<li data-hold="1.5">The suspect has an extra cast. Order-preserving matching leaves it unpaired: a gap, not a difference.</li>
<li data-hold="2">attn differs: the first pair that differs is the pivot. Walk from it to the cause.</li>
</ol>
<figcaption>Figure: both traces in order. Matching pairs equal boundaries, leaves the extra cast unpaired, and reports the first pair that differs.</figcaption>
</figure>

Two things in this picture matter.

- **Gaps are not differences.** The suspect has an event the reference has
  not. Order-preserving matching leaves it unpaired instead of forcing a pair,
  and the walk only reports pairs whose fingerprints differ.
- **The pivot is not always the cause.** A tiny example: attention's backward
  adds dQ with atomics, and has no boundary of its own. The first boundary that
  sees its result is the gradient norm, after the whole backward. The pivot is
  the norm; the cause is upstream. What names it is the call-site status: for
  each parameter's gradient, equal or different. In the last block only the
  q projection's gradient differs, which is dQ.

**Rule.** Take the first bitwise difference with no unaligned event before it,
then walk upstream through producers and downstream to the symptom, and test
the cause with the smallest intervention. This instances part 0's "compare at
operator boundaries": a boundary localizes the first visible effect, and
reasoning about producers localizes the cause.

## The step

### 1. Check coverage

**Goal:** the region in question is captured.

**How:** `opguard_bridge.coverage(run)` reads OpGuard's discovery and plan
files and lists what was discovered and what was planned.

**Success:** the operators you suspect, GEMMs included, are planned.

### 2. Capture the pair

**Goal:** fingerprints of both runs at every planned boundary.

**How:** `opguard_bridge.run_pair(reference, suspect, ...)` in
[opguard_bridge.py](../../tools/opguard_bridge.py) runs both sides through
OpGuard's harness: discovery, instrumentation plan, capture, comparison.
Discovery and capture launch each side once. If the production job runs
`torch.compile` or CUDA graphs, this capture runs it in eager and may not show
the bits production computes: see
[When the job runs compiled code](capture-compiled-code.md).

**Success:** both traces exist, and the summary counts equal and differing
events.

### 3. Take the first divergence

**Goal:** the facts about the pivot.

**How:** `opguard_bridge.first_divergence(export, ...)` calls the harness's
`trace_summary`, `first_diff`, `inspect_event`, `compare_events` and
`trace_neighbors`, and returns: operator, line, call number, which fields
differ, inputs and outputs on both sides, differing regions, unaligned events
before it, and the status of every call at its call site. It judges nothing.

**Success:** a pivot with `unaligned_before` equal to zero.

### 4. Walk outward and test

**Goal:** the cause, with evidence.

**How:** OpGuard's harness skills: `investigate-first-diff` owns the boundary
and the evidence record, `trace-upstream` follows producers when inputs already
differ, `validate-hypothesis` tests one cause. `opguard_bridge.suggest(facts)`
maps the pivot to the handbook's operator classes and
[recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md)
as leads with reasons.

| Evidence | What it gives | Cost class | When to use it |
|---|---|---|---|
| The pivot's source line and dtypes | A moved rounding point or a cast, directly | free | The output dtype or shape differs at the pivot |
| Call-site statuses (one letter per call) | Which calls of one site differ: which parameter, which step | free | The pivot is a consumer of many producers (a norm over gradients) |
| Inputs equal, output differs | The operator's own state or weights changed earlier | free | Lookups and other stateless-looking ops |
| `suggest` leads | Operator class and recipe, with reasons | free | Before reading kernels |
| Recapture full tensors of one region | Values, to see how they differ | large (2x or more) for that region | A fingerprint is equal but you suspect a collision, or you need magnitudes |
| A repeat check of one op on fixed inputs | Whether that op is nondeterministic | small | A nondeterministic kernel is the hypothesis |

**Success:** one hypothesis that a single intervention confirms.

### 5. Fix and confirm

**Goal:** the fix, and nothing else, makes the runs agree.

**How:** apply the fix (a setting, a training-step part, a kernel through the
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)),
then `run_pair` again.

**Success:** every event equal, none unaligned, final parameters
bit-identical.

In a known-answer walkthrough with three planted causes (a moved norm rounding
point, a `warn_only` attention backward, a resume without the optimizer's step
counter), on one GPU and one rank, OpGuard placed each first difference with no
unaligned event before it; two of the three pivots were not the cause; after
each fix the rerun was equal everywhere and the final parameters identical
([report](../../examples/pretraining_e2e/REPORT.md#what-this-shows)). It is a
known-answer demonstration, not a blind evaluation.

## What can go wrong

!!! warning "`torch.nn.Linear` hides the GEMMs"
    The GEMM is launched from torch's own file, so the line calling the
    projection is discovered but not planned. No projection is captured, and
    the pivot lands later. Call `F.linear` in model code, and check `coverage`.

!!! warning "One-letter names in model code"
    OpGuard's instrumenter matches call sites by the names around them and
    ignores one-letter names. Attention and projections are planned but not
    wrapped. Use descriptive names, and check `coverage`.

!!! warning "The boundary on a trailing cast"
    On a line like `q = q_norm(x).float()` the cast is not the computation.
    The pivot then names the cast. Choose the operator (`q_norm`) as the
    boundary.

!!! warning "An unpaired event read as a difference"
    One side may have extra events (fusion, a rewrite). Forcing a pairing
    invents differences. Leave unpaired events as gaps.

!!! warning "The pivot read as the cause"
    A boundary-less producer (a backward, an optimizer step, a resume) shows
    first at its next consumer. You fix the wrong operator. Walk upstream and
    test before fixing.

## Summary

- Check coverage first: an uncaptured operator cannot be the pivot.
- The pivot is the first differing pair with no unaligned event before it;
  gaps are not differences.
- The pivot is where to start: use the source line, the call-site statuses and
  the inputs to walk to the cause.
- Confirm a fix by capturing again: all equal, none unaligned, parameters
  bit-identical.

## Go deeper

- [Report: the OpGuard walkthrough](../../examples/pretraining_e2e/REPORT.md):
  setup, what OpGuard reported for each pair, fixes and reruns, cost of each
  investigation, coverage, the fingerprint checks.
- Code: [pretraining_e2e.py](../../examples/pretraining_e2e/pretraining_e2e.py)
  and [opguard_bridge.py](../../tools/opguard_bridge.py).
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md).
- [When the job runs compiled code](capture-compiled-code.md) for runs under
  `torch.compile` or CUDA graphs.
