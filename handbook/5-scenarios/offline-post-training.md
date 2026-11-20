# Offline post-training

## Where we are

Several post-training methods compute one term of the loss ahead of time,
often in an inference engine because that is faster. DPO and its variants use
a reference model's logprobs for the chosen and the rejected response; SFT
with a KL term uses reference logprobs; distillation uses a teacher's logits
or its top-k logprobs. The trainer then uses the stored numbers as if it had
computed them itself. If the engine and the trainer compute different numbers
for the same tokens, the loss carries an error that nobody chose. This is the
L4 problem of [part 2](../2-rl/README.md) without the sampler: the dataset
fixes the tokens, and the reference weights never change, so there is no
staleness. Only bookkeeping errors and numeric error are left.

## The idea: step 0 gives you a free check

DPO compares the policy with the reference on each pair of responses, a
preferred response y_w and a rejected response y_l:

```
margin = (logp_policy(y_w) - logp_ref(y_w)) - (logp_policy(y_l) - logp_ref(y_l))
loss   = -log sigmoid(beta * margin)
```

At step 0 the policy is the reference, so every margin is exactly 0 and the
loss is exactly log 2. If the stored reference logprobs came from an engine,
step 0 therefore tells you how well they match:

| Where logp_ref came from | Margin at step 0 |
|---|---|
| The trainer's own forward | Exactly 0 for every pair |
| An engine that rounds at different places | Small but nonzero, and different for each pair |
| An engine that scored different tokens (another template, other positions) | Not small, because a different sequence was scored |

So at step 0, recompute the reference logprobs in the trainer for a sample of
pairs and compare their bits with the stored ones. If they are equal, the
precomputation is exact. If they are not, first ask whether the same tokens
were scored, then whether the same weights were used, and only then debug the
numerics.

So the rule is: **the trainer must be able to reproduce a precomputed number
bit for bit from the same tokens and the same weights; check the tokens first,
then the weights, then the numbers.** This is part 2's split of a gap applied
here: separate bookkeeping errors from numeric error before you debug either.

## The step

1. **Check that the same tokens were scored**, at the same positions. Store
   every precomputed value together with its token ids, its positions and the
   exact bits of each logprob, using the fields of OpGuard's `rollout_batch`
   record (`opguard_harness.rl_records.RecordWriter`); the trainer writes its
   own values as a `trainer_logprob` record. Compare the ids before any value,
   and check the things that usually differ: the chat template (a doubled
   BOS, a system prompt, the generation prompt), which tokens are scored (the
   response only, with or without the end-of-sequence token), the padding
   side and the attention mask, the position ids in packed sequences, and
   truncation. The ids and the positions must be equal for every stored
   sequence.
2. **Check that the same weights were used**: the engine must hold the
   reference or teacher weights that the trainer loads. Write a `weight_sync`
   record with per-row fingerprints from both sides, and check it with
   OpGuard's `rl.weight_sync_check`; every row must have a match. If the
   engine loads a converted checkpoint, run
   [ckpt_compare.py](../../tools/ckpt_compare.py) on the two files instead
   ([checkpoint conversion](checkpoint-conversion.md)).
3. **Compare the numbers.** Feed the stored token ids through the trainer's
   forward with teacher forcing, and compare the fp32 bits with no tolerance.
   `rl.decompose` at the reference version reports the numeric error for each
   token (the staleness is zero by construction) ([RL loop](../2-rl/rl-loop.md)).
   You end with a numeric error of zero for every scored token, or with a list
   of the tokens where it is not.
4. **If the numbers differ, make the engine agree or move the computation.**
   The engine batches sequences of different lengths, so first make it
   invariant on its own ([whole model](../2-rl/whole-model.md)), then align
   its rounding points with the trainer's
   ([engine parity](../2-rl/engine-parity-vllm.md),
   [engine-parity skill](../../skills/engine-parity/SKILL.md)), and localize
   whatever remains with per-vector fingerprints as in the
   [RL loop](../2-rl/rl-loop.md). Alternatively, compute the reference in the
   trainer. You are done when the check in step 3 passes, or the measured gap
   is recorded as a decision.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Compute the reference in the trainer | Exact by construction | moderate (under 2x): one extra forward of the reference model per example | The default, when throughput allows it |
| Use an engine that has parity with the trainer | Exact, at the engine's speed | the cost of the engine's invariant mode (moderate to large), plus a one-time alignment | Large datasets that are scored once and reused |
| Use the engine as it is and measure the gap | A known, documented error | free | The measured numeric error is acceptable for this use |

These cost classes follow from the work each option adds; they were not
measured for post-training here. The cost of the engine's invariant mode is
measured in the [engine parity report](../../examples/cross_engine/REPORT.md).

For **distillation**, two more things decide whether the numbers can match:

- **The top-k selection.** When only the teacher's top-k logprobs are stored,
  the selection itself can differ, because two values are tied, or nearly
  tied, at the k-th place. Compare the selected ids first, then the values.
- **The normalization.** A logprob normalized over the full vocabulary is a
  different number from one renormalized over the top k, and a temperature
  can be applied before or after normalizing. Write down which one each side
  computes.

Not yet shown: no DPO or distillation run has been measured here. The parts
have been shown separately: engine parity on one model
([report](../../examples/cross_engine/REPORT.md)), and the records, the weight
check and the split on a small loop ([report](../../examples/rl_loop/REPORT.md)).

## What can go wrong

!!! warning "A different template on one side"
    The engine applies its own chat template, or adds a BOS that the trainer
    then adds again. The two sides then score different token sequences, and
    the gap is large yet looks numeric. Compare the token ids before the
    values.

!!! warning "Padding that changes the numbers"
    An engine or trainer that is not batch-invariant gives a sequence
    different bits when it is padded or packed differently. The gap then
    depends on which sequences happened to share a batch. First check each
    side alone against batched ([whole model](../2-rl/whole-model.md)).

!!! warning "Scoring different positions"
    One side sums over the prompt and the response, the other over the
    response only, or one side counts the end-of-sequence token. The totals
    then differ even though every per-token value agrees. Store and compare
    per-token values with their positions, not sums.

!!! warning "Processed logprobs from the engine"
    An engine can report logprobs after temperature, penalties or other
    processing. Those are not the model's logprobs, so no trainer can match
    them. Ask the engine for raw logprobs, and check one of them against the
    log_softmax of the engine's own logits.

## Summary

- A precomputed reference logprob or teacher logit is a rollout without the
  sampler, and the trainer must reproduce it bit for bit.
- At step 0 the DPO margin must be exactly zero, so recomputing a sample of
  pairs there is a free check.
- Check the tokens first, then the weights, then the numbers; most large gaps
  are bookkeeping errors.
- If the numbers differ, make the engine agree through engine parity, or
  compute the reference in the trainer.

## Go deeper

- [Engine parity](../2-rl/engine-parity-vllm.md) and the
  [engine-parity skill](../../skills/engine-parity/SKILL.md).
- [Whole model](../2-rl/whole-model.md): making each side invariant to the
  batch and the mode.
- [RL loop](../2-rl/rl-loop.md): the records, the weight check and the split
  that this page reuses; OpGuard's protocol is in its `docs/RLAlignment.md`.
- [Checkpoint conversion](checkpoint-conversion.md), when the engine loads a
  converted reference model.
