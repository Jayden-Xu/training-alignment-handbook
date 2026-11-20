# Make both sides agree: engine parity (L4)

## Where we are

After the previous steps each side is invariant on its own: the
[whole model](whole-model.md) in every serving mode, at every
[parallel degree](parallelism.md), with a replayable [sampler](sampling.md)
and [LoRA](lora.md). In RL the engine is usually not your code: it is vLLM or
SGLang, with its own model code, kernels and compiler. This step makes the
trainer's logprobs equal the engine's, bit for bit (L4).

## The idea

Two invariant sides can still differ, because they round at different places.
A **rounding point** is any place where a value is cast to a narrower type or
where operations are grouped into one rounding. A tiny example from RMSNorm,
with x̂ the normalized row in fp32 and w the bf16 weight:

| Side | What it computes | Roundings |
|---|---|---|
| A | bf16(x̂) · w, then bf16 | two: the cast of x̂, then the product |
| B | bf16(x̂ · w) | one: the product in fp32, then the cast |

Both are correct; they give different bits on many elements. A model has
dozens of such points: how projections are packed, where a bias is added, the
norm, the RoPE table and formula, the attention kernel, the lm_head, the
log_softmax. Every one upstream of the logits must match before any logprob is
equal, which is why the number of differing tokens can stay near all of them
for several fixes and then drop to zero at the last one.

The method follows from that:

1. **Check the engine against itself first.** If its logprob for a token
   changes when other requests join the batch, no trainer can match it.
2. **Write both rounding sequences**, one line per rounding point, in order.
3. **Align one rounding point at a time**, measuring after each, and watch
   the first layer that differs move forward, not only the token count.
4. **Prefer the engine's own kernel** in the trainer over a re-implementation.
   Equal source is not equal bits unless tile shapes, reduction order and
   compiler also match.

The general principle is the same as for kernels: equal results need equal
arithmetic, not equal formulas.

## The step

The [engine-parity skill](../../skills/engine-parity/SKILL.md) runs these
actions as a workflow.

1. **Inventory both forwards.**
   - Goal: what actually runs on each side.
   - How: from source and from the loaded model: packed and fused modules,
     custom ops, compiler and CUDA graphs, the attention backend chosen, every
     switch that changes kernels (skill section 1).
   - Success: the module tree and backend printed from the running engine,
     not taken from the config you passed.
2. **Write both rounding sequences.**
   - Goal: a checklist of differences.
   - How: the table in
     [rounding points](../../skills/engine-parity/references/rounding-points.md);
     mark each entry "read from source" or "measured".
   - Success: every difference between the sides is a named line.
3. **Check the engine against itself.**
   - Goal: engine L1 and L2 before any trainer comparison.
   - How: the same batch twice; each prompt alone against batches of several
     sizes; switch to the engine's batch-invariant mode if it has one, and
     read which ops that mode covers on your GPU.
   - Success: every prompt's logprobs equal alone and batched. If not, report
     it as engine variance and fix it in the engine.
4. **Align the trainer one rounding point at a time.**
   - Goal: L4 on realized-token logprobs.
   - How: teacher-force the engine's own tokens through the trainer, compare
     fp32 logprob bits with no tolerance, change one point, count again;
     [trainer.py](../../examples/cross_engine/trainer.py) and
     [cross_engine.py](../../examples/cross_engine/cross_engine.py) are
     templates.
   - Success: zero differing tokens; then a leave-one-out pass shows which
     points matter.
5. **Localize what remains.**
   - Goal: the first operator that still differs.
   - How: op replay (feed each captured engine input to each trainer
     candidate) and a per-layer comparison of the residual stream; for
     compiled engines, boundary fingerprints instead of hooks
     ([RL loop](rl-loop.md)).
   - Success: a named operator with evidence from both sides.

| Approach | What it does | Cost class | When to choose it |
|---|---|---|---|
| Move the trainer to the engine's kernels | The trainer calls the engine's own GEMM, norm, attention and log_softmax | moderate (under 2x) to large on the engine side from its invariant mode; the forward is free on the trainer | The engine is fixed and the trainer is yours; the trainer also needs backward kernels with the same forward bits |
| Move the engine to the trainer's rounding | Unfuse and reorder engine ops so casts follow the trainer; invariant ops on both sides | moderate (under 2x) to large on rollout throughput (reported) | The trainer is fixed and you can patch the engine |
| One model definition for both | The engine keeps only cache management; the forward is shared, so only kernels remain to align | large (2x or more) on trainer throughput for some architectures (reported) | You can change both frameworks |
| An execution contract | Pin every rounding-relevant choice on both sides and check it by hash; parallel-invariant kernels | moderate (under 2x) (reported) | Several frameworks must stay aligned over time |

Shown: a trainer that calls the engine's own kernels matched a real engine's
realized-token logprobs bit for bit across batch sizes, on one model and one
GPU type, in eager mode, forward only
([report](../../examples/cross_engine/REPORT.md#results)). Reported results
for the other approaches are collected in the
[report](../../examples/cross_engine/REPORT.md#three-ways-to-reach-parity). Not
yet shown: a compiled engine, tensor parallelism at L4, a training step with
matched backward kernels.

## What can go wrong

!!! warning "The engine's batch-invariant mode may not cover every op on every GPU"
    A mode flag can switch some ops to invariant kernels and leave others on a
    library GEMM, depending on the GPU architecture. The engine then still
    varies with the batch, and the trainer chases noise. Read the code path for
    your GPU and check engine L2 yourself.

!!! warning "Comparing the wrong logprob"
    An engine may report logprobs after temperature, penalties or other
    processing, or compute log_softmax on another device than you do. The
    comparison then shows a gap that is not in the model. First check that the
    reported logprob equals the log_softmax of the captured logits, computed
    the engine's way.

!!! warning "Hooks do not fire inside compiled graphs"
    A compiler fuses norms, RoPE and activations into neighbouring ops and the
    engine may turn its custom ops off when compiling. Module-level capture
    sees nothing there. Use boundary fingerprints, or match the eager engine
    first.

!!! warning "A flat token count is not a failed step"
    Every rounding point upstream of the logits must match before any token is
    equal. Judging each fix by the token count abandons correct fixes. Track
    the first differing layer after each change.

!!! warning "A revert that changes nothing proves little"
    In leave-one-out, a point whose revert changes no token may matter on other
    prompts or longer positions (a RoPE table can differ only past some
    position). Say "not shown to matter here", not "irrelevant".

## Summary

- Invariant sides can still differ: they must round at the same places.
- Check the engine against itself first; an engine that is not L2 cannot be
  matched.
- Write both rounding sequences, align one point at a time, and track the first
  differing layer.
- Call the engine's own kernels from the trainer rather than re-implementing
  them.

## Go deeper

- [Lab report](../../examples/cross_engine/REPORT.md): one real engine and
  model, the step table, leave-one-out, engine self-variance, which ops its
  invariant mode covers on which GPU, cost, and reported results of other
  groups.
- Skill: [engine-parity](../../skills/engine-parity/SKILL.md) and
  [rounding points](../../skills/engine-parity/references/rounding-points.md).
- Code: [examples/cross_engine/](../../examples/cross_engine/).
- Next step: [RL loop](rl-loop.md).
