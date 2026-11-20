# LoRA with mixed-adapter batches

## Where we are

The [whole-model](whole-model.md), [parallelism](parallelism.md) and
[sampling](sampling.md) steps made the base model and the sampler invariant on
each side. RL fine-tuning often trains LoRA adapters instead of the full
weights, and every adapter brings kernels of its own. The two sides run those
kernels very differently. The trainer applies one adapter to every row of its
batch. The engine serves many adapters in the same batch: it either runs
grouped or gathered kernels, in which each token picks up its own adapter, or
it merges each adapter into the base weights ahead of time. This step makes a
LoRA layer round the same way on both sides.

## The idea

A LoRA layer computes y = x·W + s·(x·A)·B. Besides the base GEMM x·W, it has a
small low-rank update (x·A)·B, a scale s, and a point where the update is added
to the base result. On top of the base GEMM, each of these three can make the
two sides disagree bit for bit:

| Source | Tiny picture | What it breaks |
|---|---|---|
| How many tokens share the adapter | An engine groups the tokens that use the same adapter and runs one GEMM per group. A request with 3 tokens forms a group of 3 when it runs alone, but in a busy batch its adapter's group may hold 300 tokens, and the GEMM library may choose a different kernel for that size | L2 |
| Where the scale s is multiplied in | s·(x·A)·B, (x·A)·(s·B) and (s·x·A)·B are equal in exact arithmetic. When every product is rounded to bf16, they are three different numbers, unless s is a power of two | L4 |
| Where the update is added to the base result | Adding two bf16 values rounds twice. Adding the fp32 update into the base GEMM's fp32 accumulator and casting once rounds only once | L4, and accuracy |

Engines also often use one LoRA kernel for prefill and another for decode,
which breaks L3 as well.

Merging the adapter into the base weights is a design choice, not a bug, but it
costs accuracy. When W + s·B·A is stored in bf16, every update element smaller
than half of the last bit of the weight it is added to is rounded away. An
engine that serves merged weights therefore cannot match a trainer that keeps
the adapter separate, and the smaller the update, the more of it is lost.

So the rule is to run the same LoRA kernel on both sides and to build it in
three ways. Each row looks up its own adapter. The reduction order never
depends on which other rows are in the batch. The update goes into the base
accumulator with one fp32 fused multiply-add, and the result is cast to bf16
once, at the end. Casting only once is also the most accurate form: it reaches
the accuracy of rounding the exact result to bf16 a single time.

## The step

1. **Turn off what the engine cannot reproduce.**
   - Goal: when the trainer recomputes logprobs, it computes the same function
     the engine served.
   - How: switch LoRA dropout off for the trainer's logprob pass, and have the
     engine serve the adapters unmerged.
   - Success: neither side applies dropout or merges the adapter.
2. **Use the same LoRA linear layer in the trainer and the engine.**
   - Goal: the trainer and the engine produce the same bits, and prefill
     produces the same bits as decode.
   - How: use `lora_linear` and `lora_linear_train` from
     [lora_kernels.py](../../examples/lora/lora_kernels.py). In the forward,
     each row selects its own adapter, and the shrink x·A adds up in a fixed
     fp32 order; one fp32 FMA then adds the update into the base accumulator,
     and the result is cast once. The backward adds in the same order and skips
     rows whose gradient is zero, so padding cannot move the chunk boundaries.
   - Success: the checks in [lora_bi.py](../../examples/lora/lora_bi.py) pass.
3. **Test the batch placements that break grouped kernels.**
   - Goal: a request gets the same result (L2) whichever other adapters share
     its batch.
   - How: compare a request run alone with the same request placed next to
     other adapters, next to rows that use no adapter, at other positions in
     the batch, and among filler rows. Choose group sizes on both sides of
     every size at which the GEMM library switches kernels, and include
     adapters whose rank is padded with zeros.
   - Success: every placement gives the same result.
4. **Check that padding does not change the trainer's backward.**
   - Goal: the gradients are batch-invariant (L2), not only repeatable from run
     to run (L1).
   - How: compare the gradients of a sequence run alone with those of the same
     sequence padded on the right (the padding gets a zero output gradient),
     and with those of the sequence batched together with other sequences.
   - Success: dX, dA and dB do not change.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Merge the adapter into the weights | A single GEMM, the fastest form. Small updates are lost in bf16, so it cannot match a trainer that keeps the adapter separate | free | Serving alone, when nothing has to match the trainer |
| Grouped or gathered library GEMMs | The adapter stays separate, but a request's result depends on which other tokens share its batch | free to moderate (the host-side overhead grows with the number of adapters) | When trainer and engine do not need to agree |
| One invariant LoRA kernel on both sides | L1 to L4 in the forward and the backward, at the best accuracy bf16 allows | moderate (under 2x) to large (2x or more) for a forward; large for a training step; faster than grouped GEMMs when there are many adapters | RL with LoRA, when trainer and engine must agree |

What has been shown: with the shared LoRA kernel, the trainer and the engine
produced identical bits in both prefill and decode, every placement test
passed, and padding did not change the backward. This was shown on one GPU
type with square layers ([report](../../examples/lora/REPORT.md#results)). Not
yet shown: a real engine's LoRA path (the engine's kernel forms were modelled
from its source code, not run), packed QKV slices, tensor parallelism, and
quantized base weights.

## What can go wrong

!!! warning "Making the base GEMM invariant is not enough"
    The LoRA GEMMs have their own M, the number of tokens that share the
    adapter, just as the GEMMs of MoE experts do. If only the base GEMM is
    invariant, the placement tests still fail. Make the update path invariant
    as well.

!!! warning "A power-of-two scale hides where the scale is applied"
    Multiplying by 2 is exact wherever it happens, so every placement agrees.
    A test that uses such a scale passes, and a real adapter then fails. Test
    with a scale that is not a power of two.

!!! warning "Split-K with atomic adds is not even repeatable (L1)"
    If the engine's shrink splits K and adds the partial sums into an fp32
    buffer with atomics, the buffer holds different values on every run. The
    bf16 cast hides most of the difference, but not all of it. Use a reduction
    with a fixed order.

!!! warning "Deterministic does not mean invariant"
    A trainer backward that goes through a library GEMM can give identical
    results run after run and still change when the batch is padded. Test the
    gradients for L2, not only for L1.

!!! warning "Padding the rank changes K"
    Padding a small adapter's rank with zeros changes the K of its GEMM, and a
    library GEMM may choose a different kernel for the new K. A fixed-order
    FMA loop only adds exact zeros, so it is not affected.

## Summary

- LoRA adds three places where the rounding can differ: how many tokens share
  an adapter, where the scale is multiplied in, and where the update is added
  to the base result.
- Merging an adapter into bf16 weights loses part of the update, so an engine
  that serves merged weights cannot match a trainer that keeps the adapter
  separate.
- Run one kernel on both sides that adds the update with one fp32 FMA and
  casts once. It is invariant, and it is also the most accurate form.
- Turn LoRA dropout off when the trainer recomputes logprobs.

## Go deeper

- [Lab report](../../examples/lora/REPORT.md): a comparison of every form, a
  table of what merging loses, the placement sweep, the rounding points,
  accuracy and cost, and how vLLM's LoRA kernels round (worked out from the
  source code).
- Code: [lora_kernels.py](../../examples/lora/lora_kernels.py),
  [lora_bi.py](../../examples/lora/lora_bi.py).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
- Regression baselines: [regression](../4-operations/regression.md).
- Next step: [engine parity](engine-parity-vllm.md).
