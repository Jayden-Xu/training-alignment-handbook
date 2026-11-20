# Framework migration and library upgrades

## Where we are

You have a model that one stack trains or serves, and you are moving it to
another: Megatron to TorchTitan, an in-house trainer to a library, or the same
framework after an upgrade of torch, CUDA or a fused-kernel library. Two
questions follow. Does the new stack compute the same thing? Where it does
not, is the difference a bug or a choice? This page answers both with the
[pre-training workflow](../1-pretraining/README.md), the old stack serving as
the reference, and the rounding-point method of
[engine parity](../2-rl/engine-parity-vllm.md) to decide each difference.

## The idea: one model, two programs

Two stacks that implement one model are two programs. They draw different
module boundaries, fuse different operators and round at different places. A
tiny example, one attention block:

| Step | Stack A | Stack B |
|---|---|---|
| Before attention | RMSNorm, then three projections q, k, v | one fused kernel: norm and a packed QKV projection |
| Residual add | in bf16 | in fp32, cast once later |
| Loss | log_softmax over the full vocabulary | vocab-parallel cross entropy, summed across ranks |

Three things follow from this table.

1. **The traces do not line up.** Stack A has a norm boundary and three
   projection boundaries; stack B has one. Pairing events by position invents
   differences. Pair them by content instead: a row that holds the same bits
   on both sides is the same value, wherever it appears.
2. **A difference is a bug or a choice.** A wrong RoPE variant is a bug; an
   fp32 residual is a choice. Both show up as different bits, so the decision
   is made per difference, with its cost written down.
3. **An upgrade is the easy case.** The same framework at two versions has the
   same boundaries and one change; the first divergence usually names the
   kernel the new version picked.

The rule: **compare at boundaries both stacks share, pair rows by content,
and turn every remaining difference into a named rounding point with a
decision.** It is the controlled experiment of part 1 applied to two
implementations, and the engine-parity principle that equal results need
equal arithmetic, not equal formulas.

<figure class="fig">
<div class="fig-steps"><span class="fig-node">1. Preflight diff</span><span class="fig-node">2. Align data and weights</span><span class="fig-node">3. Capture both</span><span class="fig-node is-key">4. First divergence</span><span class="fig-node">5. Rounding-point table</span><span class="fig-node">6. Decide per difference</span><span class="fig-node">7. Regression</span></div>
</figure>

## The step

1. **Diff the preflight reports.**
   - Goal: the two runs differ only in the stack.
   - How: run [determinism_preflight.py](../../tools/determinism_preflight.py)
     `report` inside each job and diff the two reports, declaring the stack as
     the difference ([determinism and preflight](../1-pretraining/determinism-and-preflight.md)).
     Two frameworks name the same control differently (deterministic flags,
     collective algorithms, fused-kernel switches): compare by meaning and
     list what each side sets.
   - Success: the diff lists the stack and its own switches, nothing else.
2. **Align the data and the weights.**
   - Goal: both stacks see the same tokens in the same order and start from
     the same weights.
   - How: align the token stream as in
     [data alignment](../1-pretraining/data-alignment.md). When the checkpoint
     formats differ, convert once and check the conversion bit for bit with
     [ckpt_compare.py](../../tools/ckpt_compare.py)
     ([checkpoint conversion](checkpoint-conversion.md)).
   - Success: the token ids of the first steps are equal on both sides, and
     every row of every weight is matched.
3. **Capture both stacks.**
   - Goal: fingerprints at the boundaries both stacks have: the embedding
     output, each layer's attention and MLP outputs, the residual stream, the
     logits, and in training the gradients of each parameter.
   - How: `opguard_bridge.run_pair` in
     [opguard_bridge.py](../../tools/opguard_bridge.py), with OpGuard's
     per-vector fingerprints on (`OPGUARD_XOR_ROWS=1`) so that rows can be
     paired by content; for compiled code, the
     [graph probe](../1-pretraining/capture-compiled-code.md).
   - Success: `opguard_bridge.coverage(run)` shows those boundaries captured
     on both sides.
4. **Take the first divergence.**
   - Goal: the first shared boundary where the same input rows give a
     different output row.
   - How: OpGuard pairs events automatically when the call orders agree
     ([localize with OpGuard](../1-pretraining/localize-with-opguard.md)).
     Across stacks they often do not: pair rows by their fingerprints with
     OpGuard's token matching, walking the data flow from an agreed anchor such
     as the embedding output, as its `rl-alignment` skill does. Record each
     correspondence you rely on as an OpGuard alignment map entry with its
     evidence (OpGuard's `docs/AlignmentMaps.md`).
   - Success: a pivot with equal inputs and a different output on both sides,
     and an alignment-map entry that says why the two boundaries correspond.
5. **Write the rounding-point table.**
   - Goal: every difference between the two forwards, and backwards in
     training, as a named line.
   - How: the [engine-parity skill](../../skills/engine-parity/SKILL.md),
     steps 1 and 2: inventory what actually runs on each side, then write both
     rounding sequences with the
     [rounding points](../../skills/engine-parity/references/rounding-points.md)
     table as the checklist; mark each line "read from source" or "measured".
   - Success: the pivot's difference is one of the lines.
6. **Decide per difference.**
   - Goal: each line has a decision and its measured effect.
   - How: change one side one rounding point at a time and capture again, as
     in [engine parity](../2-rl/engine-parity-vllm.md); the first differing
     boundary should move later after each change.
   - Success: no difference is left without a decision.
7. **Keep it as a regression check.**
   - Goal: the next upgrade shows what moved.
   - How: keep the smallest pair that shows each decision (one layer, one
     step) as an example in the [regression](../4-operations/regression.md)
     suite, with the decisions written next to it.
   - Success: the suite runs it and reports it equal, or reports the line that
     moved.

| Decision | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Change the new stack to the old one's rounding | bitwise equality with the reference | small to moderate, per rounding point | the old stack is the validated reference, or the change is a bug |
| Change the old stack to the new one's | a new reference both agree on | moderate: the old results must be re-established | the new form is better (fewer roundings, faster) and you will retire the old stack |
| Accept and document | a known, measured difference | free | two exact forms of the same math, neither side can move (a closed kernel, a compiler fusion) |

These cost classes are judgements from the work each decision adds, not
measurements of a migration.

For an **upgrade within one framework**, steps 1, 3 and 4 usually suffice: run
the [regression](../4-operations/regression.md) suite first, since it tells you
which stock kernels changed, then capture the old and new versions and read
the pivot.

## What can go wrong

!!! warning "Pairing events by position across stacks"
    The stacks have different boundaries, so the n-th event on one side is
    not the n-th on the other. Forced pairs report differences that do not
    exist. Pair rows by content, and leave an event without a partner as a gap.

!!! warning "Weights or data that are not the same"
    A conversion that changed a few rows, or a different shuffle, makes every
    boundary differ from the first one. The pivot then names the embedding or
    the first projection, not the cause. Check the conversion and the token
    ids before capturing.

!!! warning "A choice treated as a bug"
    A fused kernel and an unfused chain of the same math round differently by
    design. Forcing them to agree can cost more than the difference is worth.
    Name both forms, measure the gap, and decide.

!!! warning "Loss curves as the only evidence"
    Two loss curves can agree while one boundary is wrong, and disagree after
    many steps for a single benign rounding point. A curve does not say where.
    Compare at boundaries.

## Summary

- Two stacks are two programs: their traces do not line up, so pair rows by
  content and record each correspondence with its evidence.
- Remove every other difference first: preflight diff, same data, weights
  checked bit for bit.
- Take the first boundary where equal inputs give a different output, then
  name the difference as a rounding point.
- Decide each difference: change one side, or accept and document it with
  its measured effect; keep the pair as a regression check.

## Go deeper

- [Pre-training workflow](../1-pretraining/README.md) and
  [localize with OpGuard](../1-pretraining/localize-with-opguard.md):
  capture, pairing and the first divergence.
- [Engine parity](../2-rl/engine-parity-vllm.md) and the
  [engine-parity skill](../../skills/engine-parity/SKILL.md): the
  rounding-point method this page reuses.
- [Checkpoint conversion](checkpoint-conversion.md) for the weights, and
  [data alignment](../1-pretraining/data-alignment.md) for the tokens.
- OpGuard's alignment maps (`docs/AlignmentMaps.md`) and its `rl-alignment`
  skill (`harness/skills/rl-alignment`) for pairing across implementations.
- Not yet shown: a full migration between two training frameworks; the
  [report on the cross-engine case](../../examples/cross_engine/REPORT.md)
  shows the rounding-point method between a trainer and an engine.
