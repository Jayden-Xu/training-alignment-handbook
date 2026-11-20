# Offline post-training: reference log-probs and teacher logits

## Where we are

Several post-training methods compute one term of the loss ahead of time,
often in an inference engine because it is faster. DPO and its variants use a
reference model's log-probs of the chosen and the rejected response; SFT with
a KL term uses reference log-probs; distillation uses a teacher's logits or
its top-k log-probs. The trainer then uses the stored numbers as if it had
computed them itself. If the engine and the trainer compute different numbers
for the same tokens, the loss carries an error nobody chose. This is the L4
problem of [part 2](../2-rl/README.md) with the sampler removed: the tokens
are fixed by the dataset, and the reference weights do not change, so there
is no staleness. Only bookkeeping and numeric error are left.

## The idea: a free check at step 0

DPO compares the policy with the reference on each pair of responses, a
preferred one y_w and a rejected one y_l:

```
margin = (logp_policy(y_w) - logp_ref(y_w)) - (logp_policy(y_l) - logp_ref(y_l))
loss   = -log sigmoid(beta * margin)
```

At step 0 the policy is the reference, so every margin is exactly 0, and the
loss is exactly log 2. If the stored reference log-probs came from an engine,
step 0 says how well they match:

| Where logp_ref came from | Margin at step 0 |
|---|---|
| The trainer's own forward | exactly 0 for every pair |
| An engine that rounds at other places | small and nonzero, different per pair |
| An engine that scored other tokens (another template, other positions) | not small: another sequence was scored |

So recompute the reference log-probs in the trainer for a sample of pairs at
step 0 and compare bits with the stored ones. Equal: the precomputation is
exact. Not equal: first ask whether the same tokens were scored, then whether
the same weights were used, and only then debug numerics.

The rule: **a precomputed number must be reproducible bit for bit by the
trainer from the same tokens and the same weights; check the tokens, then the
weights, then the numbers.** It is part 2's split of a gap applied here:
separate bookkeeping from numeric error before debugging either.

## The step

1. **Check that the same tokens were scored.**
   - Goal: both sides scored the same token ids at the same positions.
   - How: store, with every precomputed value, the token ids, positions and
     the exact bits of each log-prob, in the fields of OpGuard's
     `rollout_batch` record (`opguard_harness.rl_records.RecordWriter`); the
     trainer writes its own values as a `trainer_logprob` record. Compare the
     ids before any value. Check what usually differs: the chat template
     (a doubled BOS, a system prompt, the generation prompt), which tokens are
     scored (response only, end-of-sequence included or not), padding side and
     attention mask, position ids in packed sequences, truncation.
   - Success: ids and positions are equal for every stored sequence.
2. **Check that the same weights were used.**
   - Goal: the engine held the reference or teacher weights the trainer
     loads.
   - How: a `weight_sync` record with per-row fingerprints from both sides,
     checked with OpGuard's `rl.weight_sync_check`; or, when the engine loads
     a converted checkpoint, [ckpt_compare.py](../../tools/ckpt_compare.py)
     on the two files ([checkpoint conversion](checkpoint-conversion.md)).
   - Success: no unmatched row.
3. **Compare the numbers.**
   - Goal: the trainer's log-prob equals the stored one for every scored
     token, bit for bit.
   - How: teacher-force the stored token ids through the trainer's forward
     and compare fp32 bits with no tolerance; `rl.decompose` at the reference
     version reports the numeric error per token (staleness is zero by
     construction) ([RL loop](../2-rl/rl-loop.md)).
   - Success: numeric error zero for every token, or a list of tokens where it
     is not.
4. **If the numbers differ, make the engine agree or move the computation.**
   - Goal: zero numeric error, or a decision.
   - How: the engine batches sequences of different lengths, so first make it
     invariant on its own ([whole model](../2-rl/whole-model.md)), then align
     its rounding points with the trainer's
     ([engine parity](../2-rl/engine-parity-vllm.md),
     [engine-parity skill](../../skills/engine-parity/SKILL.md)); localize
     what remains with per-vector fingerprints as in the
     [RL loop](../2-rl/rl-loop.md). Or compute the reference in the trainer.
   - Success: the check of step 3 passes, or the measured gap is a recorded
     decision.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Compute the reference in the trainer | exact by construction | moderate (under 2x): one extra forward of the reference per example | the default when throughput allows |
| Engine with parity | exact, at engine speed | the engine's invariant mode (moderate to large) and a one-time alignment | large datasets scored once and reused |
| Engine as it is, gap measured | a known, documented error | free | the measured numeric error is accepted for this use |

The cost classes follow from the work each option adds; they were not measured
for post-training here. The engine's invariant mode is measured in the
[engine parity report](../../examples/cross_engine/REPORT.md).

For **distillation** two more things decide whether the numbers can match:

- **Top-k selection.** When only the teacher's top-k log-probs are stored, the
  selection itself can differ: two values tied, or nearly tied, at the k-th
  place. Compare the selected ids first, then the values.
- **The normalization.** A log-prob normalized over the full vocabulary is not
  the same number as one renormalized over the top k, and a temperature can be
  applied before or after. Write down which one each side computes.

Not yet shown: no DPO or distillation run was measured here. The parts are
shown separately: engine parity on one model ([report](../../examples/cross_engine/REPORT.md))
and the records, the weight check and the split on a small loop
([report](../../examples/rl_loop/REPORT.md)).

## What can go wrong

!!! warning "Another template on one side"
    The engine applies its own chat template, or adds a BOS the trainer adds
    again. The two sides then score different token sequences, and the gap is
    large and looks numeric. Compare token ids before values.

!!! warning "Padding that changes the numbers"
    A batch-variant engine or trainer gives a sequence other bits when it is
    padded or packed differently. The gap then depends on which sequences
    shared a batch. Check each side alone against batched first
    ([whole model](../2-rl/whole-model.md)).

!!! warning "Scoring different positions"
    One side sums over prompt and response, the other over the response only,
    or one counts the end-of-sequence token. The totals differ while every
    per-token value agrees. Store and compare per-token values with their
    positions, not sums.

!!! warning "Processed log-probs from the engine"
    An engine can report log-probs after temperature, penalties or other
    processing. They are not the model's log-probs, so no trainer can match
    them. Ask the engine for raw log-probs and check one against the
    log_softmax of its own logits.

## Summary

- A precomputed reference log-prob or teacher logit is a rollout without the
  sampler: the trainer must reproduce it bit for bit.
- At step 0 the DPO margin must be exactly zero; recomputing a sample there is
  a free check.
- Check tokens, then weights, then numbers; most large gaps are bookkeeping.
- If the numbers differ, make the engine agree through engine parity, or
  compute the reference in the trainer.

## Go deeper

- [Engine parity](../2-rl/engine-parity-vllm.md) and the
  [engine-parity skill](../../skills/engine-parity/SKILL.md).
- [Whole model](../2-rl/whole-model.md): batch and mode invariance of each
  side.
- [RL loop](../2-rl/rl-loop.md): the records, the weight check and the split
  this page reuses; OpGuard's protocol is in its `docs/RLAlignment.md`.
- [Checkpoint conversion](checkpoint-conversion.md) when the engine loads a
  converted reference.
