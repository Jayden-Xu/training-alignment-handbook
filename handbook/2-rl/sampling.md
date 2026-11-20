# Replayable sampling

## Where we are

After the [whole-model](whole-model.md) and [parallelism](parallelism.md)
steps, a request's logits no longer depend on its batch, its serving mode or
its TP degree. The sampler runs after the logits. To replay a rollout while
debugging, the same logit bits must give the same token, whichever requests
it was batched with, and whether or not it was preempted, verified
speculatively or resumed on another machine. This step makes the sampled token
depend on the row alone.

## The idea

A sampler draws noise from a random stream and uses it to pick a token. Two
things can make the token depend on something other than the row:

1. **Where the noise comes from.** If the whole batch shares one generator, the
   noise a row draws depends on its row index. If each request has its own
   generator, the noise depends on how much of the stream the request has
   already used. Picture a request that drew the noise values u1, u2 and u3
   for tokens 1–3, and is then preempted and recomputed. If its generator
   restarts from the seed, token 4 gets u1 again. If the generator is kept and
   token 3 is rolled back, the new token 3 gets u4. Either way, the noise at a
   given position has moved. Another request with n > 1 in the same batch, a
   different speculative chunk size, or a restart in a new process moves it in
   the same way.
2. **How the probabilities are summed.** Top-p needs a cumulative sum over the
   sorted vocabulary. A cumulative sum over a long row is a reduction, a GPU
   kernel chooses its order from the shape, and a single bit of difference in
   it can move the boundary of the nucleus.

A random stream per request is therefore batch-invariant but not replayable.
The rule is to derive the noise from what the token is (its seed, position,
purpose and sample index), not from how many draws came before it, and to make
every sum exact. This is the same principle as for kernels: the result may
depend only on the row's own data.

| Piece | Stock form | Replayable form |
|---|---|---|
| Noise | A generator with state, shared by the batch or one per request | Counter-based: one uniform number per (seed, position, purpose, stream), with no state |
| Sums | A floating-point softmax and cumulative sum | Integer weights, so that every sum is exact |
| Order | An unstable sort | A stable order: value descending, then index ascending |
| Speculation | The draft's use of the stream changes the draws that follow | A draft token is accepted if and only if it equals the target model's own sample |

When speculation is verified by exact equality, speculative decoding produces
exactly the tokens that decoding without a draft would, at every chunk size.

## The step

1. **Find out what your engine's sampler depends on**: every way in which a
   request's token depends on its neighbours in the batch or on its own
   history. Read the engine's sampler and note where generators are created,
   how much noise each request draws per step, what happens on preemption,
   and how top-p adds up probabilities;
   [the report](../../examples/sampling/REPORT.md#what-the-engines-do-read-from-source-not-run)
   contains this reading for two engines. You are done when every dependence
   is named.
2. **Use a counter-keyed sampler with exact sums on both sides of the
   replay**, so that the token depends only on the logit bits, the seed, the
   position, the purpose, the stream and the sampling parameters. The
   reference sampler is [tools/bi_sampling.py](../../tools/bi_sampling.py)
   (`sample`, `spec_verify_exact`, `spec_verify_rejection`). The tokens must
   stay the same under batching, under preemption with recompute and with
   rollback, under chunked speculation, and after resuming in a new process.
3. **Check that it samples the right distribution**, because a sampler can be
   invariant and still biased. Compare the observed frequencies with an
   independently computed target distribution using a chi-square test, and
   include a deliberately wrong target as a control. The test must not reject
   the true target, and must clearly reject the wrong one.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| One stream shared by the batch | Nothing is replayable; a draw depends on the row index | free | Never, if you need replay |
| One generator per request | Batch-invariant, but breaks on preemption, rollback, other requests' n, speculation and restarts | free to small | Only to replay simple requests that are never interrupted |
| A counter-keyed sampler with integer weights | Replayable under every schedule tested | moderate (under 2x) at large batch sizes; large (2x or more) at small batch sizes until the noise is fused into a kernel | Any rollout you may need to replay |
| The same, with a polynomial exp | Also equal between CPU and GPU | free to large (2x or more) compared with the default exp, growing with the batch | Replay on a different device |

What has been shown: the reference sampler reaches L1–L3 under batching,
preemption, rollback, speculation and resume, and it samples the intended
distribution. This was shown on one GPU type with a toy engine
([report](../../examples/sampling/REPORT.md#results)). Not yet shown: a patched
production engine, min-p, beam search, CUDA graphs and bf16 logits.

## What can go wrong

!!! warning "The cumulative sum is the weak spot"
    A floating-point cumulative sum over the vocabulary changes with the batch
    size, and for a single long row it is not even deterministic from run to
    run. Higher precision does not fix the order. Use integer weights so that
    the sum is exact.

!!! warning "Re-creating a generator from its seed"
    Engines often rebuild a request's generator from its seed when the request
    re-enters prefill or moves to a new process. The stream starts over while
    the tokens carry on, so every later draw changes. Derive the noise from
    the position instead, or save the generator's state.

!!! warning "Another request's n changes your draws"
    An engine that draws noise for the largest n in the batch advances the
    stream of every seeded request by that amount. A request's tokens then
    depend on which requests it was batched with. Draw noise only per
    (request, sample index).

!!! warning "Rejection sampling ties the tokens to the draft model"
    The classical speculative rule is reproducible, but the tokens it produces
    depend on the draft model. If a replay must not depend on speculation,
    accept a draft token only when it equals the target's own sample, and
    accept the lower acceptance rate that comes with it.

!!! warning "Two exp implementations make two different samplers"
    A library exp and a hand-written polynomial give different weights, and
    the library exps on CPU and GPU differ as well. Pick one for each
    deployment and keep it.

## Summary

- A random stream per request is batch-invariant but not replayable.
- Derive the noise from (seed, position, purpose, stream). There is no state,
  so there is nothing to restore.
- Make every sum over the vocabulary exact with integer weights, and break
  ties by index.
- Verify speculative tokens by equality with the target's own sample, so that
  speculation leaves no trace in the tokens.

## Go deeper

- [Lab report](../../examples/sampling/REPORT.md): every source of variation,
  the flips at the nucleus boundary, how ties behave, the sampler's design and
  cost, and what two engines do.
- Code: [tools/bi_sampling.py](../../tools/bi_sampling.py),
  [sampling_bi.py](../../examples/sampling/sampling_bi.py).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
- Next step: [LoRA](lora.md).
