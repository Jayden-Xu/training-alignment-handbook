---
name: engine-parity
description: Bring a trainer forward and an inference engine (vLLM, SGLang, a custom server) to cross-engine bit equality (L4) on a real model. Inventory both forwards, write both rounding sequences, check the engine's own L1/L2/L3 first, move the trainer's rounding points to the engine's one at a time with a measured token count after each, localize what remains per op and per layer, and decide per difference whether to change the trainer, change the engine, or accept and document it. Use when RL rollout logprobs and trainer logprobs disagree, or before claiming the trainer and the engine compute the same policy.
---

# Engine parity

Own one pair: one trainer forward and one engine configuration, on one model,
GPU and software stack. Take the pair from "unknown" to "realized-token
logprobs bit-equal on a stated prompt set and batch range" (L4), or to a named
remaining op with evidence and a cost for each way out. Return the rounding
sequence of both sides, the step table, the decision per difference, and the
throughput cost of the engine configuration you ended on.

Background and levels: [skills/batch-invariant-kernel/SKILL.md](../batch-invariant-kernel/SKILL.md).
Every rounding point found so far, with what each side does:
[references/rounding-points.md](references/rounding-points.md).
A worked case (Qwen2.5-0.5B, vLLM 0.19.1, one GPU):
[handbook/2-rl/engine-parity-vllm.md](../../handbook/2-rl/engine-parity-vllm.md) and
[examples/cross_engine/](../../examples/cross_engine/), whose `trainer.py`
and `cross_engine.py` are the templates for steps 4 and 5.

## Measure what RL uses

Compare the log-probability of every realized token: prompt tokens (prompt
logprobs) and sampled tokens. Teacher-force the engine's own tokens through the
trainer, so a different token never hides a numeric difference. Compare fp32
bits, never with a tolerance. Report the fraction of tokens that differ, the
largest |Δlogp|, and mean KL where you can get both full distributions (hook
the engine's logits processor). Logprob equality is the claim; logits equality
is the stronger fact that explains it.

Check one thing before anything else: the engine's reported logprob equals
`log_softmax(fp32(engine logits))` of the logits you captured, computed the way
the engine computes it (same device: CPU and GPU `log_softmax` differed by
about 2.5e-5 on the values checked here). If not, the engine post-processes logits (temperature,
penalties, a "processed" logprobs mode) and you are comparing the wrong thing.

## 1. Inventory both forwards

For each side, from source and from what actually runs:
- the model code: which modules, which are packed (QKV, gate/up), which are fused
  (residual add + norm), where biases enter;
- the dispatch: custom CUDA ops, a compiler (`torch.compile`/Inductor rewrites
  native ops; vLLM turns its custom ops off when it compiles), CUDA graphs,
  the attention backend actually chosen;
- every switch that changes kernels: batch-invariant mode, eager mode, the
  attention backend, split-KV, cascade attention, prefix caching, chunked prefill;
- what the switch really covers on your GPU. Read the code path for your
  compute capability. Measured: vLLM 0.19.1's batch-invariant mode overrides
  `aten::mm/addmm/linear` only on a listed set of compute capabilities
  (`enable_batch_invariant_mode` in vLLM's `batch_invariant.py`); on the GPU
  tested, outside that set, the lm_head stays on cuBLAS while the decoder's
  linears use vLLM's Triton GEMM.

Print the module tree of the loaded engine model and the backend class name;
do not trust the config you passed.

## 2. Write both rounding sequences

One line per rounding point, in order, for each side: the op, the dtype of
each input and of the accumulator, every cast, what is fused into what, the
reduction order if known. Use the table in
[references/rounding-points.md](references/rounding-points.md) as the
checklist. Mark each entry "read from source (file, version)" or "measured".
Where the source is not available (compiled CUDA kernels), say so and settle it
by op replay (step 5).

## 3. Check the engine against itself first

Before any trainer comparison, measure the engine alone on the prompt set:
- L1: the same batch twice;
- L2: each prompt alone against batches of 4, 16, 64;
- L3: decode against a prefill of the same tokens (op replay of the attention
  in step 5 does this per layer).

If the engine is not L2, trainer–engine L4 cannot hold for batched rollouts,
whatever the trainer does. Report it as engine variance, then either switch the
engine to a mode that is L2 or restrict the claim to batch 1. Measured
(Qwen2.5-0.5B, [report](../../examples/cross_engine/REPORT.md)): default vLLM changed 96% of realized-token logprobs
between batch 1 and batch 64; batch-invariant eager mode changed 7% (all in a
cuBLAS lm_head the mode does not cover on the GPU tested), and none once the lm_head was
moved to vLLM's own batch-invariant GEMM.

## 4. Align one rounding point at a time

Start from the trainer as it is. Change one rounding point toward the engine,
re-run the teacher-forced comparison, write down the number of differing
tokens, then the next. Prefer the engine's own kernel in the trainer (import
it) over a re-implementation: equal source is not equal bits unless the tile
shapes, the reduction order and the compiler match. Then run leave-one-out
from the final state: revert each point alone and count again. A point whose
revert changes nothing at this prompt set is not proven irrelevant; say so.

The count can stay flat for several steps and drop at the last one. Every
rounding point upstream of the logits must match before any token is equal,
so the per-layer localization below, not the token count, shows progress.

## 5. Localize what remains

Two tools, in this order:
- **Op replay.** Hook every module of the engine (forward pre- and post-hooks,
  clone inputs before in-place kernels run), record one prefill and a few
  decode steps, then feed each captured input to each trainer candidate and
  count bit-equal outputs. This settles each rounding point independently of
  the others and shows L3 effects (feed decode steps against a trainer prefill
  of the whole sequence).
- **Per-layer stream.** Run the trainer on the captured prompt and find the first
  layer whose residual stream or MLP output differs from the engine's.

Hooks only fire in eager mode. For compiled or graph-captured engines, compare
per-vector fingerprints of boundary tensors instead: OpGuard's
`lastdim-u32le-v1` vectors, available in plain torch as
`tools/opguard_bridge.py xor_vectors`, and OpGuard's `rl-alignment` skill
(OpGuard repo, `harness/skills/rl-alignment/SKILL.md`), which splits a logprob
gap into staleness and numeric error and walks the first divergence to an
operator and token.

## 6. Decide per difference

| Choice | When | Cost to state |
|---|---|---|
| Change the trainer | The engine's op is batch-invariant and callable from the trainer (a Triton kernel, a custom op) | trainer throughput; backward: the engine's kernel may have none |
| Change the engine | The engine's op is not batch-invariant, or the trainer cannot call it (closed kernel, compiler output) | engine throughput, measured at decode batch sizes |
| Accept and document | Two algorithmically different forms (compiler fusion, a different attention algorithm) and neither side can move | the measured gap: fraction of tokens, max |Δlogp|, KL |

Write the decision next to each rounding point with its measured effect.

## 7. Report

- model, GPU, versions of both sides, engine flags, prompt set and batch sizes;
- engine L1/L2 per configuration;
- the two rounding sequences;
- the step table and leave-one-out table;
- op replay table;
- final L4 result (tokens compared, tokens differing, max |Δlogp|, KL);
- throughput of the final engine configuration against the default;
- known limits: prompts and lengths not tested, modes not covered (prefix
  caching, speculative decoding, TP), backward not covered.

## Guardrails

- Never claim L4 without engine L2 on the same prompts and batches.
- Never compare logprobs of different tokens: teacher-force the engine's tokens.
- An engine flag is not its coverage. Read the code path for your GPU's compute
  capability; measure what it changes.
- A compiled engine is a different engine. Inductor fuses native ops and drops
  intermediate roundings; eager results do not carry over to compiled mode.
- Equal source does not mean equal bits: re-implementing a kernel in another
  framework or tile shape changes the reduction order. Import the kernel.
- A rounding point that matches on one model may not on another: a bias, a
  tied embedding, a norm weight of exactly 1 or a power-of-two scale hides
  differences. Use the real checkpoint.
- Batch-invariant engine mode costs throughput. Measure it at decode batch
  sizes and report it next to the parity claim.
- Leave production defaults unchanged. Parity modes and patches are opt-in.
