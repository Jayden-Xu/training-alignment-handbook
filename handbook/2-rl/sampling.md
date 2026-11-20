# Sampling: a token you can replay

## Where we are

After the [whole-model](whole-model.md) and [parallelism](parallelism.md)
steps, a request's logits no longer depend on its batch, its mode or its TP
degree. The sampler comes after the logits. To replay a rollout for debugging,
the same logit bits must give the same token whatever the request is
co-batched with, and whether it was preempted, verified speculatively or
resumed elsewhere. This step makes the sampled token a function of the row
alone.

## The idea

A sampler draws noise from a random stream and picks a token. Two things can
tie the token to something other than the row:

1. **Where the noise comes from.** With one generator for the batch, a row's
   draw depends on its row index. With one generator per request, the draw
   depends on how much of the stream the request has already used. Picture a
   request that drew noise u1, u2, u3 for tokens 1–3, is preempted and
   recomputed: if its generator restarts from the seed, token 4 gets u1 again;
   if it is kept, a rollback of token 3 gets u4 for token 3. Either way the
   noise for a position moved. Other requests' n > 1, the speculative chunk
   size and a restart in a new process move it the same way.
2. **How the probabilities are summed.** Top-p needs a cumulative sum over the
   sorted vocabulary. A cumulative sum over a long row is a reduction whose
   order a GPU kernel picks by shape, and one bit in it moves the nucleus
   boundary.

So a per-request stream is batch-invariant but not replayable. The rule: key
the noise by what the token is (seed, position, purpose, sample index), not by
how many draws came before, and make every sum exact. The principle is the same
as for kernels: the result may depend only on the row's own data.

| Piece | Stock form | Replayable form |
|---|---|---|
| Noise | a stateful generator, shared or per request | counter-based: one uniform per (seed, position, purpose, stream), no state |
| Sums | floating-point softmax and cumulative sum | integer weights, so every sum is exact |
| Order | unstable sort | (value descending, index ascending), stable |
| Speculation | the draft's consumption of the stream shapes later draws | accept a draft token iff it equals the target's own sample |

With exact-equality verification, speculative decoding gives exactly the
tokens of decoding without a draft, at every chunk size.

## The step

1. **Find what your engine's sampler depends on.**
   - Goal: the list of ways a request's token depends on its neighbours or its
     history.
   - How: read the engine's sampler for where generators are created, how much
     noise each request draws per step, what happens on preemption, and how
     top-p sums; [the report](../../examples/sampling/REPORT.md#what-the-engines-do-read-from-source-not-run)
     has this read for two engines.
   - Success: each dependence is named.
2. **Use a counter-keyed, exact sampler on both sides of the replay.**
   - Goal: a token that is a function of the logit bits, seed, position,
     purpose, stream and sampling parameters only.
   - How: the reference sampler [tools/bi_sampling.py](../../tools/bi_sampling.py)
     (`sample`, `spec_verify_exact`, `spec_verify_rejection`).
   - Success: the same tokens under batching, preemption with recompute and
     rollback, chunked speculation, and resume in a new process.
3. **Check that it samples the right distribution.**
   - Goal: invariance without bias.
   - How: compare empirical frequencies against an independently computed
     target with a chi-square test, including a deliberately wrong target as
     a control.
   - Success: no rejection on the true target, a clear rejection on the
     wrong one.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Shared stream | Nothing replayable; a draw depends on the row index | free | Never, if you need replay |
| Per-request generator | Batch-invariant; breaks on preemption, rollback, other requests' n, speculation, restart | free to small | Replay of simple, uninterrupted requests only |
| Counter-keyed, integer-weight sampler | Replayable under every schedule tested | moderate (under 2x) at large batch; large (2x or more) at small batch until the noise is fused into a kernel | Any rollout you may need to replay |
| Same, with a polynomial exp | Also equal across CPU and GPU | free to large (2x or more) over the default exp, growing with batch | Replay on another device |

Shown: the reference sampler is L1–L3 under batching, preemption, rollback,
speculation and resume, and samples the intended distribution, on one GPU type
with a toy engine ([report](../../examples/sampling/REPORT.md#results)). Not
yet shown: a patched production engine, min-p, beam search, CUDA graphs, bf16
logits.

## What can go wrong

!!! warning "The cumulative sum is the weak op"
    A floating-point cumulative sum over the vocabulary changes with the batch
    size, and for one long row it is not even run-to-run deterministic.
    Higher precision does not fix the order. Use integer weights so the sum is
    exact.

!!! warning "Re-creating a generator from the seed"
    Engines often rebuild a request's generator from its seed when it
    re-enters prefill or a new process. The stream restarts while the tokens
    continue, so every later draw changes. Key the noise by position instead,
    or save the generator state.

!!! warning "Other requests' n changes your draws"
    An engine that fills noise for the largest n in the batch advances every
    seeded request's stream by that amount. A request's tokens then depend on
    who it was batched with. Draw per (request, sample index) only.

!!! warning "Rejection sampling ties tokens to the draft"
    The classical speculative rule is reproducible but its tokens depend on the
    draft model. If replay must not depend on speculation, verify by equality
    with the target's own sample, and accept the lower acceptance rate.

!!! warning "Two exp implementations are two samplers"
    A library exp and a hand-written polynomial give different weights, and
    CPU and GPU library exps differ too. Pick one per deployment and keep it.

## Summary

- A per-request random stream is batch-invariant but not replayable.
- Key the noise by (seed, position, purpose, stream); no state, nothing to
  restore.
- Make every sum over the vocabulary exact with integer weights, and break
  ties by index.
- Verify speculation by equality with the target's sample to make it invisible.

## Go deeper

- [Lab report](../../examples/sampling/REPORT.md): every source of variance,
  the nucleus flips, tie behaviour, the sampler's design, its cost, and what
  two engines do.
- Code: [tools/bi_sampling.py](../../tools/bi_sampling.py),
  [sampling_bi.py](../../examples/sampling/sampling_bi.py).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
- Next step: [LoRA](lora.md).
