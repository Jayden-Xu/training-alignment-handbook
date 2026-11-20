# 5. Other alignment scenarios

## Where we are

Parts 1 to 4 taught one method in two workflows: find where two pre-training
runs part ([part 1](../1-pretraining/README.md)), make a rollout engine and a
trainer agree ([part 2](../2-rl/README.md)), make one operator invariant
([part 3](../3-kernels/README.md)), and keep the results true when the stack
changes ([part 4](../4-operations/README.md)). Many other needs are the same
question with a different pair of programs: a framework migration, a
checkpoint conversion, a reference model run in an inference engine. This part
shows how to apply the method you already have to seven such scenarios. Three
have their own page; the other four have a short section here.

## The idea: name the pair, the inputs and the claim

Every scenario is a reference, a suspect, the inputs both receive, and a claim
about which outputs should be equal and how strictly. Once those four are
written down, an existing workflow applies. A few pairs:

| Pair | Same inputs | What should be equal | Expected |
|---|---|---|---|
| A checkpoint and its converted copy | none: nothing is computed | every row of every weight | bitwise, always |
| One model in two training frameworks | data, weights, seed | every boundary both share | bitwise only after aligning every rounding point |
| A trainer and a precomputed reference log-prob | the same token ids and weights | each token's log-prob | bitwise with engine parity |
| One request alone and inside a busy server | the request | its tokens and log-probs | bitwise with batch-invariant serving |
| The same job on two GPU generations | everything | nothing at the bit level | different bits; localize instead |

The last row matters as much as the others. Where bitwise equality is not
promised, the goal is the first operator whose bits differ and a decision
about it, not equality at any cost. The rule: **state the pair, the inputs and
the claim before comparing anything.** It is the controlled experiment of
[part 1](../1-pretraining/README.md#the-idea-two-runs-one-difference) with the
claim made explicit.

## The scenarios

| Scenario | Workflow to reuse | Tools and skills | What is specific |
|---|---|---|---|
| [Framework migration and library upgrades](framework-migration.md) | [part 1](../1-pretraining/README.md), with the other stack as the reference; the rounding-point method of [engine parity](../2-rl/engine-parity-vllm.md) | [determinism_preflight.py](../../tools/determinism_preflight.py), [opguard_bridge.py](../../tools/opguard_bridge.py), [engine-parity](../../skills/engine-parity/SKILL.md) | two programs whose traces do not line up; each difference is a bug or a choice |
| [Checkpoint conversion and resharding](checkpoint-conversion.md) | none computes: a copy check, then a forward | [ckpt_compare.py](../../tools/ckpt_compare.py) | rows survive repacking; placement needs a forward |
| [Offline post-training](offline-post-training.md): reference log-probs for SFT and DPO, teacher logits for distillation | [part 2](../2-rl/README.md) at L4, without sampling | [engine-parity](../../skills/engine-parity/SKILL.md), OpGuard's RL records | which tokens are scored, templates, padding |
| [Reward models and evaluation](#reward-models-and-evaluation) | [whole model](../2-rl/whole-model.md), [sampling](../2-rl/sampling.md), [regression](../4-operations/regression.md) | [bi_audit.py](../../tools/bi_audit.py), [bi_sampling.py](../../tools/bi_sampling.py) | a score near a threshold turns one bit into a different decision |
| [Deterministic serving](#deterministic-serving) | [whole model](../2-rl/whole-model.md) on the engine side; [part 3](../3-kernels/README.md) | [bi_audit.py](../../tools/bi_audit.py), [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md) | every serving mode a request can meet |
| [Speculative decoding and MTP](#speculative-decoding-and-mtp) | [sampling](../2-rl/sampling.md), [whole model](../2-rl/whole-model.md) at L3 | [bi_sampling.py](../../tools/bi_sampling.py) | verification is a chunk, decoding is one token |
| [Hardware migration and mixed fleets](#hardware-migration-and-mixed-fleets) | [hardware](../4-operations/hardware.md), [localize with OpGuard](../1-pretraining/localize-with-opguard.md) | [sdc_scan.py](../../tools/sdc_scan.py), [bi_suite.py](../../tools/bi_suite.py) | bitwise is not expected across GPU generations |

## Reward models and evaluation

**Goal:** a reward model gives a (prompt, response) pair the same score in any
batch, and an evaluation gives the same number when run again.

A reward model is a whole model with a scalar head, so it has the same
batch and mode dependence as a policy. What is different is the use: a score
is compared with a threshold, ranked against other responses, or averaged into
a pass rate. A score one bit away from a threshold changes a decision, and a
changed decision changes the training signal or the reported number.

How to apply the existing pages:

1. Make the reward model invariant like any model: audit its operators with
   [bi_audit.py](../../tools/bi_audit.py) and check one pair alone against
   inside batches, padded against packed ([whole model](../2-rl/whole-model.md)).
2. If the score is computed in an engine and recomputed by the trainer, that is
   L4: follow [engine parity](../2-rl/engine-parity-vllm.md).
3. If the evaluation samples, make its tokens replayable from the logits and a
   seed ([sampling](../2-rl/sampling.md), [bi_sampling.py](../../tools/bi_sampling.py)).
4. Keep a fixed evaluation set and its scores as a
   [regression](../4-operations/regression.md) entry.

**Success:** the same score bits for every pair across batch sizes, and the
same evaluation result in a rerun with a different batch composition.

Not yet shown: no reward model or evaluation harness was measured here.

## Deterministic serving

**Goal:** a request's tokens and log-probs do not depend on which other
requests share its batch (L2), or on how the server chunks, caches or decodes
it (L3).

This is the engine side of the [whole-model step](../2-rl/whole-model.md),
seen by a user: two identical requests at temperature 0 get different answers
because the server was busy differently. The causes and fixes are the same.

How to apply the existing pages:

1. Audit the served model and test every mode a request can meet: a changing
   decode batch, chunked prefill, a prefix-cache hit, preemption
   ([whole model](../2-rl/whole-model.md)).
2. Replace each batch-variant operator with an invariant kernel
   ([part 3](../3-kernels/README.md)); keep the sampler replayable
   ([sampling](../2-rl/sampling.md)).
3. Measure the cost at your decode batch sizes. If it is too high, offer the
   invariant path to the requests that need it rather than to all.

**Success:** each request's log-probs are bit-equal served alone and under
load, in every mode the server uses.

Not yet shown: a production server under real traffic, multi-node serving.

## Speculative decoding and MTP

**Goal:** a speculative or multi-token-prediction (MTP) decoder emits exactly
the tokens and log-probs the target model emits on its own.

Verification runs the target model over several draft tokens at once, like a
short prefill; plain decoding runs one token at a time. If the two paths round
differently, an accepted token's log-prob differs from the plain decode's,
and the acceptance test itself can flip. The draft's own numerics only change
speed, not the result, provided acceptance compares against the target's own
sample.

How to apply the existing pages:

1. Make the target's verification forward equal its decode forward: chunk
   against one token is the L3 case of the [whole model](../2-rl/whole-model.md),
   and attention's prefill and decode use one split of the keys
   ([attention](../3-kernels/attention.md)).
2. Make acceptance replayable: with exact-equality verification, speculative
   decoding gives the tokens of decoding without a draft
   ([sampling](../2-rl/sampling.md), [bi_sampling.py](../../tools/bi_sampling.py)).
3. Treat MTP heads used as drafts the same way; an MTP loss in pre-training is
   part of the training step ([part 1](../1-pretraining/README.md)).

**Success:** the same tokens and log-prob bits with and without speculation,
at every draft length.

Not yet shown: a real engine's speculative decoding, MTP heads.

## Hardware migration and mixed fleets

**Goal:** know which results carry over to new GPUs, and when two machines
disagree, find where.

Across GPU generations bitwise equality is not expected: libraries pick other
kernels and accumulation orders per architecture ([hardware](../4-operations/hardware.md)).
So the goal changes from equality to localization. The first operator whose
bits differ should be a library GEMM or a reduction whose order the library
chooses; anything else is a finding.

How to apply the existing pages:

1. Qualify the new hardware: run the regression suite against the committed
   baselines and the SDC scan across groups ([hardware](../4-operations/hardware.md),
   [bi_suite.py](../../tools/bi_suite.py), [sdc_scan.py](../../tools/sdc_scan.py)).
2. Within one generation and stack, compare devices bitwise: a disagreement
   there is a fault ([SDC](../1-pretraining/sdc.md)).
3. Across generations, capture both runs with OpGuard on the same inputs and
   take the first divergence ([localize with OpGuard](../1-pretraining/localize-with-opguard.md)).
   Classify it: an architecture-dependent kernel choice (expected) or anything
   else (report it).
4. For RL with the trainer and the engine on different generations, only
   kernels whose reduction order is written in their source can carry over,
   and that is untested across architectures. Otherwise measure the numeric
   gap and record it as a decision, as
   [engine parity](../2-rl/engine-parity-vllm.md) does for differences
   neither side can move.

**Success:** every difference between generations is classified, and the
expected ones are recorded with the stack they belong to.

Not yet shown: the handbook's kernels across GPU architectures, a
mixed-generation RL run.

## What can go wrong

!!! warning "Expecting bitwise equality where none is promised"
    Two GPU generations, or two libraries' kernels, are not built to agree in
    the last bit. Chasing equality there spends time on expected differences.
    State the claim first; where it is not equality, localize and decide.

!!! warning "A tolerance standing in for a bit check"
    "Close enough" outputs hide faults that happen to be small: a cast that
    rounds small weights can leave every logit unchanged on a few inputs
    ([checkpoint conversion](checkpoint-conversion.md)). Check copies and
    recomputations bitwise; use a tolerance only for pairs that cannot agree.

!!! warning "Comparing two different computations"
    A reference log-prob scored over other tokens, a template applied on one
    side only, a score after a different post-processing: the numbers differ
    though nothing is wrong numerically. Check that both sides computed the
    same thing before comparing values.

## Summary

- Each scenario is a pair, the inputs both receive, and a claim; write those
  down first.
- A copy must be bitwise; two programs can be made bitwise by aligning
  rounding points; two GPU generations are localized, not equalized.
- The existing workflows apply unchanged: preflight, capture, first
  divergence, rounding points, decide, regression.
- Each section says what is not yet shown; treat those as open.

## Go deeper

- Pages: [framework migration](framework-migration.md),
  [checkpoint conversion](checkpoint-conversion.md),
  [offline post-training](offline-post-training.md).
- [Report: checkpoint conversion](../../examples/ckpt_conversion/REPORT.md),
  with the planted faults and what each check saw.
- Skills: [alignment-pipeline](../../skills/alignment-pipeline/SKILL.md),
  [engine-parity](../../skills/engine-parity/SKILL.md),
  [pretraining-alignment](../../skills/pretraining-alignment/SKILL.md).
- [Open items](../open-items.md) for everything still untested.
