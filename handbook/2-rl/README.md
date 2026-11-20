# 2. RL: train–inference alignment

## Where we are

In RL the rollout engine (vLLM, SGLang) samples tokens and reports their
logprobs, and the trainer (FSDP, Megatron, TorchTitan) recomputes the same
logprobs for the policy-gradient ratio. The two run one policy with different
code, kernels, batching, decode mode and parallelism, so their logprobs differ
even with identical weights, and the run becomes off-policy without anyone
choosing it. This part teaches how to make the two sides agree bit for bit, and
how to find what broke when they do not. It assumes the levels L1–L5 from
[Part 0](../0-foundations/README.md).

Treat parity first as a debugging and audit tool: a short window in which the
infrastructure must give a gap of exactly zero. Whether to keep it on in
production is a cost decision.

## The workflow

<figure class="fx" data-fig="rl-workflow" aria-label="The RL train–inference workflow" markdown>
<div class="fx-stage" markdown>
<div class="fx-phases" style="--cols: minmax(0, 1fr) minmax(0, 0.75fr) minmax(0, 1.25fr)" markdown>
<div class="fx-phase" markdown="span"><span class="fx-label">Each side invariant</span><span class="fx-seq is-col"><span class="fx-step" data-hot="0">[1. Audit the operators](whole-model.md)</span><span class="fx-step" data-hot="1">[2. Whole model](whole-model.md)<span class="fx-note">L1–L3</span></span><span class="fx-step" data-hot="2">[3. Parallel degrees](parallelism.md)<span class="fx-note">L5</span></span><span class="fx-step" data-hot="3">[4. Sampling](sampling.md)</span><span class="fx-step" data-hot="4">[5. LoRA](lora.md)</span></span></div>
<div class="fx-phase" markdown="span"><span class="fx-label">Both sides agree</span><span class="fx-seq is-col"><span class="fx-step" data-hot="5 9">[6. Engine parity](engine-parity-vllm.md)<span class="fx-note">L4</span></span></span></div>
<div class="fx-phase" markdown="span"><span class="fx-label">Watch the running loop</span><span class="fx-seq is-col"><span class="fx-step" data-hot="6">[7. Record online](rl-loop.md)</span><span class="fx-step" data-hot="7">[8. Split the gap](rl-loop.md)<span class="fx-note">staleness and numeric error</span></span><span class="fx-fork is-pair"><span class="fx-leg" data-hot="8"><span class="fx-when">numeric ≠ 0</span><span class="fx-step to-bad" data-hot="8">[9. Localize offline](rl-loop.md)<span class="fx-note" data-hot="9">then back to 6</span></span></span><span class="fx-leg" data-hot="10"><span class="fx-when">numeric = 0</span><span class="fx-step to-ok" data-hot="10">[10. Regression](../4-operations/regression.md)</span></span></span></span></div>
</div>
</div>
<ol class="fx-say">
<li>1. Audit the operators: the list of modules whose output depends on the batch or the mode.</li>
<li>2. Whole model: one side gives a sequence the same bits in every batch and serving mode (L1–L3).</li>
<li>3. Parallel degrees: the same bits at every TP degree, and across disaggregated prefill and decode (L5).</li>
<li>4. Sampling: a sampled token that can be replayed after batching, preemption, n > 1 and speculation.</li>
<li>5. LoRA: adapters that round the same way in a training batch and in a mixed serving batch.</li>
<li>6. Engine parity: trainer logprobs equal to the engine's (L4).</li>
<li>7. Record the running loop online.</li>
<li>8. Split every token's gap into staleness and numeric error.</li>
<li>numeric ≠ 0: 9. localize it offline, down to an operator.</li>
<li>Then go back to step 6 with the cause it found.</li>
<li data-hold="2">numeric = 0: 10. keep the result as a baseline that a new version must reproduce.</li>
</ol>
<figcaption>Figure: steps 1–5 make each side invariant on its own, step 6 makes the two agree, and steps 7–10 watch the running loop. Each step links to its page.</figcaption>
</figure>

The [alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md) routes
a question to the right step.

## The idea: one side first, then both

Two numbers can differ for two reasons: each side is unstable on its own, or
both are stable and round differently. Fix them in that order. An engine whose
logprob for a token changes when another request joins the batch cannot match
any trainer, so steps 2–5 make each side invariant on its own, and only step 6
compares the two.

Some differences are not kernel bugs but two exact forms of the same math that
round differently: chunked against recurrent linear attention, absorbed
against explicit MLA, split-KV decode against single-pass prefill. Tuning a
kernel cannot remove them. Pick one form for both sides and pay its cost;
[Part 3](../3-kernels/README.md) covers writing the kernels.

## The idea: split the gap before you debug it

In an asynchronous loop the rollout used policy version v and the trainer is
at version c. The observed gap mixes two things, and they split exactly:

```
gap       = logp_rollout(v) - logp_trainer(c)
numeric   = logp_rollout(v) - logp_trainer(v)   # same weights: an infrastructure fault, should be 0
staleness = logp_trainer(v) - logp_trainer(c)   # policy lag: corrected by IS / TIS / masking
```

A tiny example. A token was sampled at version 3; the trainer is now at
version 5. Recompute the token's logprob on the trainer at version 3 and
compare:

| Token | logp_rollout(3) | logp_trainer(3) | logp_trainer(5) | gap | numeric | staleness |
|---|---|---|---|---|---|---|
| A | -1.20 | -1.20 | -1.35 | 0.15 | 0 | 0.15 |
| B | -1.20 | -1.21 | -1.35 | 0.15 | 0.01 | 0.14 |

Both tokens show the same gap. For A it is all policy lag, which importance
sampling is designed to correct; the infrastructure is exonerated. For B a
part of it is an infrastructure fault that no correction should absorb, and it
is the part worth localizing. Only `numeric` is the infrastructure's to fix,
and only a replay at the rollout's own version can measure it.

## Synchronous and asynchronous runs

- **Synchronous (staleness 0).** `numeric` is the whole gap. With parity on
  both sides the target is a difference of exactly 0 for every token over a
  short on-policy window. If the gap is 0 and training is still broken, the
  problem is in data or algorithm.
- **Asynchronous (staleness > 0).** Record online, split each token's gap, and
  localize only tokens with nonzero `numeric`. Tokens generated across a weight
  update (KV cache kept) belong to two versions and cannot be split; set them
  aside.

## What can go wrong

!!! warning "Comparing before each side is stable"
    A trainer is compared with an engine that is not batch-invariant. Every
    trainer change then looks partly right, because the engine's own variance
    is as large as the gap. Check the engine against itself first.

!!! warning "Correcting a numeric error as if it were staleness"
    Importance-sampling corrections absorb any gap, including an
    infrastructure fault. The fault then trains the policy silently. Split
    the gap at the rollout's version before applying a correction.

!!! warning "Trusting the version label"
    A rollout batch labelled with one version may have run on the weights of
    another. Its numeric error then looks real but does not reproduce on
    replay. Record a fingerprint of the weights each batch actually used.

## Summary

- Make each side invariant on its own (L1–L3, L5, sampling, LoRA), then make
  the two agree (L4).
- Some mismatches are two exact forms of the same math: choose one for both
  sides.
- Split every gap into staleness and numeric error; only numeric error is an
  infrastructure fault.
- A nonzero numeric error is localized offline along the data flow, then kept
  as a regression baseline once fixed.

## Go deeper

- Step pages: [whole model](whole-model.md), [parallelism](parallelism.md),
  [sampling](sampling.md), [LoRA](lora.md),
  [engine parity](engine-parity-vllm.md), [RL loop](rl-loop.md).
- Reported results from other groups (IsoExec, verl-omni, TorchTitan RL) and a
  worked single-element localization: [report](../../examples/cross_engine/REPORT.md#context-moved-from-the-part-2-overview);
  sources in [references](../references.md).
- Skills: [alignment-pipeline](../../skills/alignment-pipeline/SKILL.md),
  [engine-parity](../../skills/engine-parity/SKILL.md),
  [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md);
  OpGuard's [`rl-alignment` skill](https://github.com/ziming-zh/OpGuard/tree/main/harness/skills/rl-alignment) (`harness/skills/rl-alignment` in the OpGuard
  repository) and its protocol, `docs/RLAlignment.md`.
