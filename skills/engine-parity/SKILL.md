---
name: engine-parity
description: Make a trainer's forward pass and an inference engine (vLLM, SGLang, a custom server) produce bit-identical results on a real model (cross-engine invariance, L4). List what each forward actually runs, write down where each side rounds, check first that the engine agrees with itself (L1/L2/L3), then move the trainer's rounding points to the engine's one at a time and count the differing tokens after each change, localize what remains operator by operator and layer by layer, and decide for each remaining difference whether to change the trainer, change the engine, or accept and document it. Use when RL rollout logprobs and trainer logprobs disagree, or before claiming that the trainer and the engine compute the same policy.
---

# Engine parity

This skill works on one pair at a time: one trainer forward and one engine
configuration, on one model, GPU and software stack. At the start you do not
know whether the two agree. The goal is L4: the logprob of every realized token
is bit-equal on both sides, on a prompt set and a range of batch sizes that you
state. If that cannot be reached, the goal is to name the operator that still
differs, with evidence, and with the cost of each way of removing the
difference. Return:

- the rounding sequence of each side;
- the step table;
- the decision taken for each difference;
- the throughput cost of the engine configuration you end with.

For background and the definition of the levels, see
[skills/batch-invariant-kernel/SKILL.md](../batch-invariant-kernel/SKILL.md).
[references/rounding-points.md](references/rounding-points.md) lists every
rounding point found so far and what each side does at it.
A worked case on Qwen2.5-0.5B with vLLM 0.19.1 on one GPU is described in
[handbook/2-rl/engine-parity-vllm.md](../../handbook/2-rl/engine-parity-vllm.md) and
implemented in [examples/cross_engine/](../../examples/cross_engine/). Its
`trainer.py` and `cross_engine.py` are the templates for steps 4 and 5.

## Measure the quantity RL actually uses

RL consumes the log-probability of every realized token, so that is what you
compare: both the prompt tokens (prompt logprobs) and the sampled tokens.
Teacher-force the engine's own tokens through the trainer, so that a different
token can never hide a numeric difference. Compare the fp32 bits and never use
a tolerance. Report the fraction of tokens that differ, the largest |Δlogp|,
and the mean KL when you can get both full distributions (to get the engine's,
hook its logits processor). Equal logprobs are what you claim; equal logits are
the stronger fact that explains the claim.

Before anything else, check one thing. The logprob the engine reports must
equal `log_softmax(fp32(engine logits))` computed on the logits you captured,
and computed the way the engine computes it, on the same device (CPU and GPU
`log_softmax` differed by about 2.5e-5 on the values checked here). If the two
are not equal, the engine post-processes the logits (temperature, penalties, a
"processed" logprobs mode), and you are comparing the wrong quantity.

## 1. List what each forward actually runs

For each side, collect the following, both from the source and from what
actually runs:

- the model code: which modules it has, which projections are packed into one
  GEMM (QKV, gate/up), which ops are fused (residual add + norm), and where the
  biases are added;
- how ops are dispatched: custom CUDA ops, a compiler (`torch.compile`/Inductor
  rewrites native ops, and vLLM turns its custom ops off when it compiles),
  CUDA graphs, and the attention backend that is actually chosen;
- every switch that changes which kernels run: batch-invariant mode, eager
  mode, the attention backend, split-KV, cascade attention, prefix caching,
  chunked prefill;
- what each switch really covers on your GPU. Read the code path for your
  GPU's compute capability. Measured: vLLM 0.19.1's batch-invariant mode
  overrides `aten::mm/addmm/linear` only on a listed set of compute
  capabilities (`enable_batch_invariant_mode` in vLLM's `batch_invariant.py`).
  The GPU tested is outside that set, so its lm_head stays on cuBLAS while the
  decoder's linear layers use vLLM's Triton GEMM.

Print the module tree of the model the engine actually loaded and the class
name of the backend it chose. Do not trust the configuration you passed in.

## 2. Write down the rounding sequence of each side

For each side, write one line per rounding point, in execution order. Each line
gives the op, the dtype of each input and of the accumulator, every cast, what
is fused into what, and the reduction order if you know it. Use the table in
[references/rounding-points.md](references/rounding-points.md) as the
checklist. Mark each entry "read from source (file, version)" or "measured".
Where the source is not available (compiled CUDA kernels), say so, and settle
the entry by op replay (step 5).

## 3. Check the engine against itself first

Before you compare anything with the trainer, measure the engine on its own,
on the prompt set:

- L1: run the same batch twice;
- L2: run each prompt alone and compare it with batches of 4, 16 and 64;
- L3: compare decode with a prefill of the same tokens (replaying the
  attention op in step 5 does this layer by layer).

If the engine is not L2, the trainer and the engine cannot reach L4 on batched
rollouts, whatever the trainer does. Report this as variance inside the
engine. Then either switch the engine to a mode that is L2, or restrict the
claim to batch size 1. Measured on Qwen2.5-0.5B
([report](../../examples/cross_engine/REPORT.md)): between batch 1 and batch
64, default vLLM changed 96% of the realized-token logprobs. Batch-invariant
eager mode changed 7%, all of them caused by the lm_head, which stays on cuBLAS
because the mode does not cover it on the GPU tested. Once the lm_head was
moved to vLLM's own batch-invariant GEMM, no logprob changed.

## 4. Align one rounding point at a time

Start from the trainer as it is. Change one rounding point so that it does what
the engine does, re-run the teacher-forced comparison, and write down how many
tokens differ. Then move on to the next point. In the trainer, prefer importing
the engine's own kernel to re-implementing it: the same source code gives the
same bits only if the tile shapes, the reduction order and the compiler also
match. When you reach the final state, run a leave-one-out pass: revert each
point on its own and count again. If reverting a point changes nothing on this
prompt set, that does not prove the point is irrelevant; say so in the report.

The count can stay flat for several changes and then drop at the last one,
because no token can be equal until every rounding point upstream of the
logits matches. Progress therefore shows in the per-layer localization below,
not in the token count.

## 5. Localize the differences that remain

Use two tools, in this order:

- **Op replay.** Hook every module of the engine with forward pre- and
  post-hooks, and clone the inputs before in-place kernels overwrite them.
  Record one prefill and a few decode steps. Then feed each captured input to
  each trainer candidate and count the outputs that are bit-equal. This settles
  each rounding point independently of the others. It also shows L3 effects
  when you compare the decode steps with a trainer prefill of the whole
  sequence.
- **Per-layer residual stream.** Run the trainer on the captured prompt and
  find the first layer whose residual stream or MLP output differs from the
  engine's.

Hooks only fire in eager mode. When the engine is compiled or runs captured
CUDA graphs, compare per-vector fingerprints of the tensors at operator
boundaries instead. OpGuard computes them as `lastdim-u32le-v1` vectors, and
the same vectors are available in plain torch as
`tools/opguard_bridge.py xor_vectors`. OpGuard's `rl-alignment` skill
([OpGuard repo](https://github.com/ziming-zh/OpGuard/tree/main/harness/skills/rl-alignment), `harness/skills/rl-alignment/SKILL.md`)
works with these vectors: it splits a logprob gap into staleness and numeric
error, and locates the first divergence at a specific operator and token.

## 6. Decide what to do about each difference

| Choice | When to choose it | Cost to state |
|---|---|---|
| Change the trainer | The engine's op is batch-invariant and the trainer can call it (a Triton kernel, a custom op) | Trainer throughput. For training, note that the engine's kernel may have no backward |
| Change the engine | The engine's op is not batch-invariant, or the trainer cannot call it (a closed kernel, compiler output) | Engine throughput, measured at decode batch sizes |
| Accept and document | The two sides use algorithmically different forms (compiler fusion, a different attention algorithm) and neither side can move | The measured gap: the fraction of tokens that differ, the max \|Δlogp\| and the KL |

Write each decision next to its rounding point, together with its measured
effect.

## 7. Report

The report contains:

- the model, the GPU, the software versions on both sides, the engine flags,
  the prompt set and the batch sizes;
- the engine's L1/L2 results for each configuration;
- the two rounding sequences;
- the step table and the leave-one-out table;
- the op replay table;
- the final L4 result: tokens compared, tokens that differ, max |Δlogp| and KL;
- the throughput of the final engine configuration compared with the default;
- known limits: the prompts and lengths not tested, the modes not covered
  (prefix caching, speculative decoding, TP), and the backward pass, which is
  not covered.

## Guardrails

- Never claim L4 unless the engine is L2 on the same prompts and batches.
- Never compare the logprobs of different tokens. Teacher-force the engine's
  tokens through the trainer.
- A flag's name does not tell you what it covers. Read the code path for your
  GPU's compute capability, and measure what the flag changes.
- A compiled engine is a different engine. Inductor fuses native ops and drops
  intermediate roundings, so results measured in eager mode do not carry over
  to compiled mode.
- The same source does not give the same bits. Re-implementing a kernel in
  another framework or with another tile shape changes the reduction order, so
  import the kernel instead.
- A rounding point that matches on one model may not match on another. A bias,
  a tied embedding, a norm weight of exactly 1 or a power-of-two scale can hide
  a difference, so test with the real checkpoint.
- The engine's batch-invariant mode costs throughput. Measure the cost at
  decode batch sizes and report it next to the parity claim.
- Leave production defaults unchanged. Parity modes and patches must be opt-in.
