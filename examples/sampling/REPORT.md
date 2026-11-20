# Batch-invariant sampling: lab report

This is the full record behind the handbook page
[Sampling](../../handbook/2-rl/sampling.md). The page teaches the step;
this report keeps the setup, every measurement and how the case was built.

## Summary

- Shared and per-request random streams break replay under batching,
  preemption, n > 1 and speculation: a per-request stream is batch-invariant
  but not replayable.
- `cumsum` over the vocabulary is the weak op: batch-variant at 32,000 and
  151,936, and not run-to-run deterministic at 151,936. fp64 does not fix it.
- **Fix:** noise keyed by (seed, position, purpose, stream), integer weights so
  sums are exact, stable (value, index) order. The reference sampler
  [tools/bi_sampling.py](../../tools/bi_sampling.py) passes L1–L3.
- **Cost:** 1.3x the stock sampler at B = 256 (A40). Speculation verified by
  equality with the target's own sample gives the tokens of decoding without a
  draft, at every k.

## How

The other cases make the logits of a request independent of its batch. The
sampler comes after them. A rollout can be replayed for debugging only if the
same logit bits give the same token whatever the request is co-batched with,
and whether it was preempted, verified speculatively or resumed elsewhere.
### Where the variance comes from

Measured on the A40 unless marked "read from source".

**The random stream.**
- `torch.multinomial` with one generator for the batch: a row's draw depends
  on its row index. A row placed first matched its single-row draw at every
  size; in the middle or last it failed at every batch size 2–513 (312 of 472
  comparisons at each vocabulary).
- One `torch.Generator` per request, with exponential or Gumbel noise filled
  per row: batch-invariant in every test (236 comparisons per vocabulary). The
  draw then depends on how much of the stream the request already used:
  - **Preemption with recompute.** vLLM 0.6.3.post1 re-creates the generator
    from the seed when a request re-enters prefill (read from source). In a toy
    engine with the same logits per prefix, every preempted request then
    produced different tokens. Keeping the generator fixed this, but a
    preemption that discards the last tokens still diverged: the stream had
    moved on.
  - **n > 1.** vLLM draws `[n_running x num_samples, V]` noise per seeded
    request, where `num_samples` is the largest `n` among the prompts in the
    batch (read from source). Re-implemented here: a decoding request's next
    draw changed when a co-batched prompt asked n = 17 (V = 32,000) or n = 4
    (V = 151,936). The torch CUDA generator advances its offset by fill size,
    with steps every 516,096 fp32 elements here (84 SMs x 6,144; the relation
    to the SM count is a guess). A request with 3 running samples got new noise
    for samples 2 and 3 whenever the batch's largest `n` changed.
  - **Speculative decoding.** Consumed in vLLM's order (k draft draws, a bonus
    draw, `rand(1, k)` for acceptance, a `[k, V]` recovery draw), the same
    request gave different tokens for every chunk size k = 1–8 and differed
    from sequential decoding at every k.
  - **Resume in a new process.** Re-creating the generator from the seed
    diverged. Restoring `get_state()` (16 bytes) worked.
  - A CPU and a CUDA generator with the same seed give different values.

**Row-wise ops.** Re-measured at both vocabularies, batch 1–513, three
positions, three fillers:

| Op | V = 32,000 | V = 151,936 |
|---|---|---|
| temperature divide, softmax, log_softmax, logsumexp, exp, max, argmax | L2 | L2 |
| repetition, frequency and presence penalties, logit bias (vLLM's gather/scatter) | L2 | L2 |
| `sum` over the vocabulary | L2 | fails from batch 14 |
| `cumsum` fp32 or fp64 of the sorted probabilities | fails at every batch size from 2 | fails, and not even L1 |

- `cumsum` of one row changes with the batch size in groups: 2–64, 65–256,
  257–1,024 at V = 32,000. A row alone differs from the same row in a batch of
  2 in 83% of elements.
- From about 80,000 elements one row's `cumsum` is not run-to-run
  deterministic: 30 distinct results in 30 runs at V = 131,072 and 151,936
  (in a probe, 1 at 65,537 and 6 at 80,000). Many rows at once were
  deterministic. `torch.use_deterministic_algorithms(True, warn_only=False)`
  raises on `cumsum`; with `warn_only=True` it warns and runs the same kernel.
- `sum` at V ≥ 131,072 gives up to 19 different results over batch 1–64.

**The nucleus boundary.** One bit in the cumulative sum moves the top-p cut.
Rows (out of 2,048 at V = 32,000, 512 at V = 151,936) whose kept set differed
between a single-row call and batches of 2–512:

| Logits, top_p | SGLang-style (descending `cumsum`) | vLLM-style (ascending `cumsum`) | reference |
|---|---|---|---|
| V 32,000, N(0, 1.5²), 0.90 | 48 | 4 | 0 |
| V 32,000, N(0, 1.5²), 0.95 | 96 | 1 | 0 |
| V 32,000, N(0, 3²), 0.90 / 0.95 | 5 / 15 | 2 / 1 | 0 |
| V 151,936, N(0, 1.5²), 0.90 / 0.95 | 61 / 126 | 3 / 2 | 0 |
| V 151,936, N(0, 3²), 0.90 / 0.95 | 6 / 26 | 0 / 2 | 0 |

- The SGLang-style rule also changed 4 of 512 rows between two identical
  single-row calls at V 151,936, N(0, 1.5²), 0.95 (the `cumsum` L1 failure);
  0 in every other configuration.
- The flipped tokens sit at the edge of the nucleus, with tiny probability. In
  2,048 rows of a vLLM-style pipeline with per-request generators, no sampled
  token and no log-probability changed between batch 1 and 64. A flip changes
  the distribution's normalizer in the last bits; it rarely changes a token.

**Ties.**
- `torch.argmax` and `max().indices` returned the lowest index among tied
  maxima in all 1,692 rows tried (batch 1–512, both vocabularies). Not documented.
- `torch.topk` with ties across the cut kept the lowest indices, the same in
  every batch. Its sorted output does not list tied values in index order, and
  its tie order differs from the CPU's (57 of 80 indices at V = 32,000).
- `torch.sort(stable=False)` differs from the stable order for rows of 2–32
  elements with ties. For longer rows it matched the stable order in every test
  here. Not documented.

### The fix: a reference sampler

`tools/bi_sampling.py` samples a token that is a function of the row's logit
bits, seed, absolute position, purpose, stream index and parameters, and of
nothing else.

| Piece | How |
|---|---|
| Noise | Counter-based Philox4x32-10: one 53-bit uniform per (seed, position, purpose, stream). No state, so recompute, rollback, chunked verification and resume reproduce the draw. `purpose` separates token, draft, accept and residual draws; `stream` separates the n samples of a request |
| Weights | `w = floor(exp(z - max z) * 2^S)` in int64, S = 52 − ceil(log2 V). Every sum and cumulative sum is an exact integer, so no reduction order exists to fix |
| Top-k, top-p | order by (value desc, index asc) from `torch.sort(stable=True)`; keep a token iff the integer mass before it is < top_p x total |
| Draw | r = floor(u x total); the first index in vocabulary order whose cumulative weight exceeds r |
| Greedy | the lowest index among the maxima, by construction |
| Speculative | `spec_verify_exact`: the token is always the target's sample; a draft token is accepted iff equal. `spec_verify_rejection`: the classical rule with counter-based accept and residual draws |

API:
```python
import bi_sampling as bs
tok = bs.sample(logits, seeds, positions, temperature=t, top_k=k, top_p=p, stream=j)  # [B] int64
info = bs.sample(..., return_info=True)            # tokens, integer weight, integer total
tok, acc = bs.spec_verify_exact(target_logits, draft_tokens, seeds, positions, **params)
tok, acc = bs.spec_verify_rejection(target_logits, draft_logits, draft_tokens, seeds, positions, **params)
w, total = bs.kept_weights(logits, t, k, p)        # the exact distribution
u = bs.uniform53(seeds, positions, purpose, stream)
```

**Floating-point functions it relies on.** With `exp="torch"` (default):
`torch.exp` on fp32 must give the same bits for the same input wherever the
element sits. No `log`. The rest is one IEEE operation per element, integer
arithmetic and a stable sort. With `exp="basic"`: no transcendental function.
exp is a degree-7 polynomial after a two-constant range reduction, each
multiply and add a separate op, and 2^n built from exponent bits. Its result
equals a scalar Python implementation (token, weight and total on 160 rows) and
is identical on CPU and GPU. `exp="torch"` weights differ between CPU and GPU in
1.6–8% of elements and from `exp="basic"` by up to 0.5% relative, so the two are
different samplers: pick one per deployment. `exp="basic"` costs 0.9–3.6x
`exp="torch"`.

**Fidelity.** A token below 2^-S of the row maximum (2^-34 at V = 151,936) gets
weight 0 and is never drawn. NaN and +inf logits are not supported.

## Evidence

### Setup, code and results

Taken through the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
- Code: the reference sampler [tools/bi_sampling.py](../../tools/bi_sampling.py);
  the checks [examples/sampling/sampling_bi.py](sampling_bi.py).
- Results: [results-a40.json](results-a40.json).
- Setup: one NVIDIA A40 (84 SMs), driver 550.144.03, torch 2.5.1+cu121. Logits
  in fp32, vocabularies 32,000 and 151,936, batch 1–513 (1,024 and 1,025 for
  the reference sampler at 32,000).
- In every comparison the request's logit bits are identical. Only the batch
  around it, the schedule or the process changes.

### Results

**Invariance of the reference sampler** (all bitwise, all pass).

| Check | Level | Tested |
|---|---|---|
| token, weight and total, row vs alone | L2 | 7 settings (T 1, T 0.7, top-k 50, top-p 0.9, all three, greedy, mixed per row) x 2 vocabularies; batch 1–513, first/middle/last, random/zero/large fillers; probe rows with ties, flat logits, bf16 values; 1,220 comparisons each; also batch 1,024 and 1,025 |
| same batch, 5 runs | L1 | mixed and top-p, both vocabularies |
| top-p set, single-row call vs batches 2–512 | L2 | the 8 settings of the nucleus table: 0 rows changed |
| 3 random batched schedules with random fillers vs decoding alone | L2 | 8 requests x 64 tokens, both vocabularies |
| preemption with recompute, and with a 3-token rollback | L3 | same |
| `spec_verify_exact` in chunks k = 1–8 vs sequential decoding without a draft | L3 | same |
| `spec_verify_rejection` in chunks k = 1–8 vs k = 1 | L3 | same |
| resume of a checkpointed request (prompt, seed, tokens) in a new process | L3 | both vocabularies |
| `exp="basic"`: GPU vs CPU, and vs a scalar Python implementation | L4 (device) | 64 rows per vocabulary; 160 rows at V = 96 |

The same toy engine with stock paths: the shared stream failed every schedule
and preemption; per-request generators passed the schedules and failed
preemption, rollback, speculative chunks and resume from the seed.

**Correctness.** Chi-square of empirical frequencies against a float64 target
computed independently: 12 settings at V = 50 and 1,000 (temperature 0.7–1.5,
top-k 5–200, top-p 0.5–0.95, combined), with draws over seeds and over
positions, both exp modes, and the outputs of both speculative verifiers;
2^20 draws each. All 20 had p between 0.016 and 0.98, and no draw outside the
support. Control: T = 0.7 draws against the T = 0.72 target gave p = 2e-232.

**Cost** per sampling step, ms, median of 9 after warm-up, A40 alone:

| Case | B | V = 32,000: multinomial / per-request gen / reference / basic | V = 151,936: multinomial / per-request gen / reference / basic |
|---|---|---|---|
| temperature | 1 | 0.18 / 0.08 / 2.56 / 2.37 | 0.21 / 0.11 / 2.17 / 2.41 |
| | 8 | 0.19 / 0.18 / 2.10 / 2.38 | 0.20 / 0.18 / 2.38 / 2.96 |
| | 64 | 0.23 / 0.97 / 2.15 / 2.51 | 1.04 / 1.20 / 2.46 / 6.05 |
| | 256 | 0.87 / 3.59 / 2.41 / 5.06 | 3.95 / 4.94 / 6.33 / 22.7 |
| top-k 50, top-p 0.9 | 1 | 0.49 / 0.38 / 2.53 / 2.86 | 0.52 / 0.40 / 2.67 / 3.01 |
| | 8 | 0.54 / 0.50 / 2.64 / 2.85 | 1.15 / 1.15 / 3.01 / 3.30 |
| | 64 | 1.24 / 1.33 / 2.75 / 3.80 | 5.56 / 5.51 / 7.72 / 11.8 |
| | 256 | 5.24 / 5.65 / 7.11 / 10.6 | 24.8 / 24.8 / 32.2 / 48.8 |
| greedy (vs `torch.argmax`) | 1 / 256 | 0.03 / – / 0.19 / – ; 0.08 / – / 0.50 / – | 0.03 / – / 0.23 / – ; 0.31 / – / 2.00 / – |

- About 1.5 ms of the reference's time at small batch is the Philox noise,
  computed on the CPU in torch integer ops (1.47–1.70 ms for B = 1–256).
  A fused kernel would remove it; not written.
- At B = 256 with top-k and top-p it costs 1.30x the stock path at
  V = 151,936 and 1.36x at 32,000. With temperature only at V = 32,000 it is
  faster than per-request generators (2.41 against 3.59 ms).
- Peak extra memory at B = 256, V = 151,936, top-k and top-p: 1.71 GiB.

### What the engines do (read from source, not run)

**vLLM 0.6.3.post1** (`model_executor/layers/sampler.py`,
`rejection_sampler.py`, `spec_decode_base_sampler.py`, `sampling_metadata.py`,
`core/scheduler.py`, `spec_decode/`):
- A request with `seed` gets its own `torch.Generator`, created when the
  request is in prefill and stored per request id. Requests without a seed
  share the global CUDA stream, one `exponential_()` fill over all their rows:
  batch-variant.
- The draw is `argmax(probs / Exp(1))`. Each seeded request fills
  `len(seq_ids) x max_n_in_batch` rows, where `max_n_in_batch` is the largest
  `n` among prompt-phase requests in the batch: other requests' `n` changes how
  far the stream moves.
- Preemption: recompute by default for single-sequence groups. The sequence
  goes back to prefill and the generator is re-created from the seed, so the
  stream restarts while the tokens continue. Groups with several sequences are
  swapped and keep it.
- Top-k/top-p: ascending unstable sort, softmax, `cumsum` (variance measured
  above). FlashInfer sampling (`VLLM_USE_FLASHINFER_SAMPLER`) draws
  `[32, batch]` uniforms per group: untested.
- Speculative decoding: scoring positions other than the bonus use default
  (unseeded) parameters; the bonus token, the acceptance `rand(1, k)` and the
  `[k, V]` recovery draw use the request's generator, so k changes what is
  consumed. Speculation is switched off above `speculative_disable_by_batch_size`,
  which changes the consumption with load. `generators` is a class attribute of
  `ModelRunnerBase`, so draft and target runners may share one dict (not
  confirmed).

**SGLang 0.4.0** (tag v0.4.0; `layers/sampler.py`, `sampling/`,
`managers/`):
- No per-request seed: `SamplingParams` has none and the OpenAI `seed` field is
  not used. `--random-seed` seeds the global torch RNG once per process.
- Every step draws from that shared stream: `torch.rand(32, batch)` for
  FlashInfer (the default when installed), or `torch.multinomial` over the
  batch for the torch backend. A request's draws depend on the batch, its row
  and the whole history of the server.
- `n > 1` becomes n separate requests with new ids; retracted requests
  recompute later with whatever the stream gives then. No speculative decoding
  in this version.
- Top-p in the torch backend: descending sort, `cumsum` (the SGLang-style
  column above). Repetition, frequency and presence penalties are ignored under
  the default overlap scheduler (it logs a warning).

Newer versions of both engines are untested.

### What this shows

1. **A per-request stream is batch-invariant but not replayable.** It survives
   co-batching. It breaks on everything that changes how many draws a request
   consumes: recompute, rollback, other requests' `n`, speculative chunk size,
   resume from the seed.
2. **Key the noise by what the token is, not by how many draws came before.**
   A counter-based draw keyed by (seed, position, purpose, stream) passed every
   replay test with no state to save.
3. **`cumsum` is the weak op.** It is not batch-invariant at either vocabulary,
   and not run-to-run deterministic for one row at 151,936. fp64 does not fix
   it. Integer weights make the sum exact and remove the question.
4. **Row-wise softmax, log_softmax and temperature stayed L2** at these sizes;
   a plain `sum` did not at 151,936.
5. **Speculation can be made invisible.** Verifying by equality with the
   target's own sample gives the tokens of decoding without a draft, at every
   k. The classical rejection rule is also exactly reproducible, but its tokens
   depend on the draft (446 of 512 differed from no-draft decoding).

## Limits

- One GPU (A40), torch 2.5.1. `exp="torch"` assumes `torch.exp` is elementwise
  and position-independent; verified at these shapes only. Use `exp="basic"`
  for CPU–GPU or cross-GPU agreement.
- The toy engine stands in for a real one: logits come from a table keyed by
  the prefix. No engine was patched or run.
- Not tested: min-p, typical sampling, beam search, logprob outputs of the
  reference sampler, FlashInfer kernels, `torch.compile`, CUDA graphs (the CPU
  noise is not capturable), bf16 or fp16 logits.
- The noise is on the CPU and costs about 1.5 ms per step.
- `spec_verify_exact` accepts only draft tokens equal to the target's sample:
  23.6% at k = 1 in the toy against 58.3% for the rejection rule.
- Tie and stability behaviour of `torch.sort`, `torch.topk` and `argmax` is
  measured, not documented; the reference sampler does not rely on it beyond
  `stable=True`.

## How it was built

An agent wrote the probes, the sampler and the checks. What took more than one
try:
- The first sampler used fp64 `exp` and computed the noise on the GPU: 4.5 ms
  per step at B = 1. Moving the noise to the CPU and `exp` to fp32 halved it.
- `exp="basic"` first used an fp64 degree-12 polynomial; it was replaced by
  the fp32 degree-7 form above.
- The nucleus flips are rare and depend on the sample: two stock settings
  showed no flip in one run. The per-setting checks became "info", with one
  required check over all settings.
- One probe script failed on a `torch.sort` argument order and was rerun.
