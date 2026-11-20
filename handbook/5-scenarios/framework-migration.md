# Framework migration and library upgrades

## Where we are

You have a model that one stack trains or serves, and you are moving it to
another stack: from Megatron to TorchTitan, from an in-house trainer to a
library, or to the same framework after an upgrade of torch, CUDA or a
fused-kernel library. Two questions follow. Does the new stack compute the
same thing? And where it does not, is the difference a bug or a choice? This
page answers both. It uses the [pre-training workflow](../1-pretraining/README.md)
with the old stack as the reference, and the rounding-point method of
[engine parity](../2-rl/engine-parity-vllm.md) to decide each difference.

## The idea: the same model written as two programs

Two stacks that implement the same model are two different programs. They
draw module boundaries in different places, fuse different operators, and
round at different points. Here is a tiny example, one attention block:

| Step | Stack A | Stack B |
|---|---|---|
| Before attention | RMSNorm, then three projections q, k and v | One fused kernel that does the norm and a packed QKV projection |
| Residual add | Done in bf16 | Done in fp32, and cast once later |
| Loss | log_softmax over the full vocabulary | Vocab-parallel cross entropy, summed across ranks |

Three things follow from this table.

1. **The traces do not line up.** Stack A has a norm boundary and three
   projection boundaries, while stack B has a single boundary. If you pair
   events by their position, you invent differences. Pair them by content
   instead: a row that holds the same bits on both sides is the same value,
   wherever it appears.
2. **Each difference is either a bug or a choice.** Using the wrong RoPE
   variant is a bug; keeping the residual in fp32 is a choice. Both show up
   as different bits, so you decide each difference on its own and write
   down what it costs.
3. **An upgrade is the easy case.** The same framework at two versions has the
   same boundaries and one change, and the first divergence usually names the
   kernel that the new version picked.

So the rule is: **compare at the boundaries both stacks share, pair rows by
their content, and turn every remaining difference into a named rounding point
with a decision.** This is the controlled experiment of part 1 applied to two
implementations, together with the engine-parity principle that equal results
need equal arithmetic, not just equal formulas.

<figure class="fx" data-fig="migration" aria-label="The framework-migration step, on one attention block">
<div class="fx-stage">
<div class="fx-steps"><span class="fx-step" data-hot="0">1. Diff the preflight reports</span><span class="fx-step" data-hot="1">2. Align data and weights</span><span class="fx-step" data-hot="2">3. Capture both stacks</span><span class="fx-step is-key" data-hot="3">4. Find the first divergence</span><span class="fx-step" data-hot="4">5. List the rounding points</span><span class="fx-step" data-hot="5">6. Decide each difference</span><span class="fx-step" data-hot="6">7. Keep a regression check</span></div>
<div class="fx-grid" style="--cols:4">
<span class="fx-label">stack A</span><span class="fx-node" data-hot="2">RMSNorm</span><span class="fx-node" data-hot="2">q, k, v</span><span class="fx-node" data-hot="3">attention</span><span class="fx-node to-bad" data-hot="3 4">residual add, bf16</span>
<span></span><span class="fx-pair" data-at="2" style="grid-column: span 2">no shared boundary, so not compared</span><span class="fx-pair" data-at="3">shared boundary, rows paired by content</span><span class="fx-swap fx-pair" data-at="3"><span data-to="4">shared boundary, rows paired by content</span><span class="fx-pair is-bad" data-at="4">a rounding point: bf16 against fp32</span></span>
<span class="fx-label">stack B</span><span class="fx-node is-ghost" data-hot="2" style="grid-column: span 2">fused norm + packed QKV</span><span class="fx-node" data-hot="3">attention</span><span class="fx-node to-bad" data-hot="3 4">residual add, fp32</span>
</div>
</div>
<ol class="fx-say">
<li>1. Diff the preflight reports to confirm that the two runs differ only in the stack.</li>
<li>2. Make both stacks see the same tokens and start from the same weights.</li>
<li data-hold="1.5">3. Capture both stacks at the boundaries they share. Stack B fused the norm and the projections, so those are not compared.</li>
<li data-hold="1.5">4. Pair rows by content at the shared boundaries, and find the first boundary where the same input rows give different output rows.</li>
<li data-hold="1.5">5. List every difference between the two forwards as a rounding point, such as a residual add in bf16 on one side and in fp32 on the other.</li>
<li>6. Decide each difference: either it is a bug to fix, or it is a choice to keep, recorded with its measured effect.</li>
<li data-hold="2">7. Keep the pair as a regression check, so that the next upgrade shows what moved.</li>
</ol>
<figcaption>Figure: compare the two stacks only at boundaries they share, pair rows by their content, and turn every remaining difference into a named rounding point with a decision.</figcaption>
</figure>

## The step

1. **Diff the preflight reports.**
   - Goal: confirm that the two runs differ only in the stack.
   - How: run [determinism_preflight.py](../../tools/determinism_preflight.py)
     `report` inside each job and diff the two reports, declaring the stack as
     the intended difference
     ([determinism and preflight](../1-pretraining/determinism-and-preflight.md)).
     Two frameworks often name the same control differently (deterministic
     flags, collective algorithms, fused-kernel switches), so compare the
     settings by meaning and list what each side sets.
   - Success: the diff lists the stack and its own switches, and nothing else.
2. **Align the data and the weights.**
   - Goal: both stacks see the same tokens in the same order and start from
     the same weights.
   - How: align the token stream as described in
     [data alignment](../1-pretraining/data-alignment.md). When the
     checkpoint formats differ, convert once and check the conversion bit for
     bit with [ckpt_compare.py](../../tools/ckpt_compare.py)
     ([checkpoint conversion](checkpoint-conversion.md)).
   - Success: the token ids of the first steps are equal on both sides, and
     every row of every weight has a matching row.
3. **Capture both stacks.**
   - Goal: fingerprints at every boundary both stacks have: the embedding
     output, each layer's attention and MLP outputs, the residual stream, the
     logits, and, in training, the gradient of each parameter.
   - How: use `opguard_bridge.run_pair` in
     [opguard_bridge.py](../../tools/opguard_bridge.py) with OpGuard's
     per-vector fingerprints switched on (`OPGUARD_XOR_ROWS=1`), so that rows
     can be paired by content. For code that runs under torch.compile or CUDA
     graphs, use the [graph probe](../1-pretraining/capture-compiled-code.md).
   - Success: `opguard_bridge.coverage(run)` shows those boundaries captured
     on both sides.
4. **Find the first divergence.**
   - Goal: the first shared boundary where the same input rows give a
     different output row.
   - How: OpGuard pairs events automatically when the two call orders agree
     ([localize with OpGuard](../1-pretraining/localize-with-opguard.md)).
     Across stacks they often do not. In that case, pair rows by their
     fingerprints with OpGuard's token matching, and walk the data flow from an
     anchor where both sides agree, such as the embedding output, the way its
     `rl-alignment` skill does. Record every correspondence you rely on as an
     entry in an OpGuard alignment map, together with its evidence (OpGuard's
     `docs/AlignmentMaps.md`).
   - Success: a pivot whose inputs are equal on both sides and whose output
     differs, and an alignment-map entry that says why the two boundaries
     correspond.
5. **Write the rounding-point table.**
   - Goal: every difference between the two forwards, and between the two
     backwards in training, written as a named line.
   - How: follow steps 1 and 2 of the
     [engine-parity skill](../../skills/engine-parity/SKILL.md). Take an
     inventory of what actually runs on each side, then write down both
     rounding sequences, using the
     [rounding points](../../skills/engine-parity/references/rounding-points.md)
     table as the checklist, and mark each line "read from source" or
     "measured".
   - Success: the difference at the pivot is one of the lines.
6. **Decide each difference.**
   - Goal: every line has a decision and a measured effect.
   - How: change one side, one rounding point at a time, and capture again, as
     in [engine parity](../2-rl/engine-parity-vllm.md). After each change, the
     first differing boundary should move later in the model.
   - Success: no difference is left without a decision.
7. **Keep it as a regression check.**
   - Goal: the next upgrade shows what moved.
   - How: keep the smallest pair that shows each decision (one layer, one
     step) as an example in the [regression](../4-operations/regression.md)
     suite, with the decisions written next to it.
   - Success: the suite runs the pair and reports it equal, or reports the
     line that moved.

| Decision | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Change the new stack to round like the old one | Bitwise equality with the reference | small to moderate, per rounding point | The old stack is the validated reference, or the difference is a bug |
| Change the old stack to round like the new one | A new reference that both stacks agree on | moderate, because the old results must be established again | The new form is better (fewer roundings, or faster) and you will retire the old stack |
| Accept the difference and document it | A known, measured difference | free | The two forms are both exact versions of the same math, and neither side can change (a closed kernel, a fusion done by the compiler) |

These cost classes are judgements based on the work each decision adds, not
measurements from a real migration.

For an **upgrade within one framework**, steps 1, 3 and 4 are usually enough.
Run the [regression](../4-operations/regression.md) suite first, because it
tells you which stock kernels changed; then capture the old and the new
version and read the pivot.

## What can go wrong

!!! warning "Pairing events by position across stacks"
    The two stacks have different boundaries, so the n-th event on one side
    is not the n-th event on the other. Pairs forced this way report
    differences that do not exist. Pair rows by content, and leave an event
    without a partner as a gap in the record.

!!! warning "Weights or data that are not actually the same"
    A conversion that changed a few rows, or a different shuffle, makes every
    boundary differ from the first one on. The pivot then names the embedding
    or the first projection instead of the real cause. Check the conversion
    and the token ids before you capture.

!!! warning "Treating a choice as a bug"
    A fused kernel and an unfused chain that compute the same math round
    differently by design. Forcing them to agree can cost more than the
    difference is worth. Name both forms, measure the gap, and decide.

!!! warning "Using loss curves as the only evidence"
    Two loss curves can agree while one boundary is wrong, and they can drift
    apart after many steps because of a single harmless rounding point. A
    curve does not tell you where the difference is. Compare at boundaries.

## Summary

- Two stacks are two programs whose traces do not line up, so pair rows by
  content and record each correspondence with its evidence.
- Remove every other difference first: diff the preflight reports, use the
  same data, and check the weights bit for bit.
- Find the first boundary where equal inputs give a different output, and
  name that difference as a rounding point.
- Decide each difference: either change one side, or accept and document it
  with its measured effect. Keep the pair as a regression check.

## Go deeper

- The [pre-training workflow](../1-pretraining/README.md) and
  [localize with OpGuard](../1-pretraining/localize-with-opguard.md) cover
  capture, pairing and the first divergence.
- [Engine parity](../2-rl/engine-parity-vllm.md) and the
  [engine-parity skill](../../skills/engine-parity/SKILL.md) describe the
  rounding-point method that this page reuses.
- [Checkpoint conversion](checkpoint-conversion.md) covers the weights, and
  [data alignment](../1-pretraining/data-alignment.md) covers the tokens.
- OpGuard's alignment maps (`docs/AlignmentMaps.md`) and its `rl-alignment`
  skill (`harness/skills/rl-alignment`) handle pairing across
  implementations.
- Not yet shown: a full migration between two training frameworks. The
  [report on the cross-engine case](../../examples/cross_engine/REPORT.md)
  shows the rounding-point method between a trainer and an engine.
