# LoRA: one adapter in a training batch, many in a serving batch

## Where we are

After the [whole-model](whole-model.md), [parallelism](parallelism.md) and
[sampling](sampling.md) steps, the base model and the sampler are invariant on
each side. RL fine-tuning often trains LoRA adapters, and the adapters bring
their own kernels. The trainer runs one adapter over a whole batch. The engine
serves many adapters in one batch with grouped or gathered kernels, or serves
merged weights. This step makes a LoRA layer round the same way in both.

## The idea

A LoRA layer computes y = x·W + s·(x·A)·B: a base GEMM, a small update, a
scale s, and a point where the two meet. It adds three new places where the
bits can move, on top of the base GEMM:

| Source | Tiny picture | What it breaks |
|---|---|---|
| The adapter group's size | An engine groups the tokens that share an adapter and runs one GEMM per group. A request with 3 tokens is in a group of 3 alone and a group of 300 in a busy batch, and the GEMM may pick another kernel | L2 |
| Where s enters | s·(x·A)·B, (x·A)·(s·B) and (s·x·A)·B are equal in exact math; with a bf16 rounding after each product they are three different numbers (unless s is a power of two) | L4 |
| Where base and update meet | Adding two bf16 values rounds twice; adding the fp32 update to the fp32 base accumulator and casting once rounds once | L4, and accuracy |

Prefill and decode often use different LoRA kernels in an engine, which breaks
L3 too.

Merging is a choice of form, not a fault. W + s·B·A stored in bf16 loses every
update element smaller than half of W's last bit. An engine that serves merged
weights therefore cannot match an unmerged trainer, and the smaller the update,
the more of it is lost.

The rule: one LoRA kernel for both sides, in which a row selects its own
adapter, the reduction order never depends on which other rows are present,
and base and update meet in one fp32 fused multiply-add followed by the only
cast. The single cast is also the most accurate form: it reaches the bf16
rounding of the exact result.

## The step

1. **Turn off what the engine cannot reproduce.**
   - Goal: the trainer's logprob pass computes what the engine computes.
   - How: run the trainer's logprob pass with LoRA dropout off, and serve
     unmerged adapters.
   - Success: no dropout and no merge on either side.
2. **Use one LoRA linear in trainer and engine.**
   - Goal: trainer bits = engine bits, prefill = decode.
   - How: `lora_linear` and `lora_linear_train` in
     [lora_kernels.py](../../examples/lora/lora_kernels.py): a row-selected
     shrink with a fixed fp32 order, one fp32 FMA into the base accumulator, one
     cast; the backward uses the same order and skips zero-gradient rows so
     padding cannot shift chunk boundaries.
   - Success: the checks in [lora_bi.py](../../examples/lora/lora_bi.py) pass.
3. **Check the placements that break grouped kernels.**
   - Goal: L2 for a request among other adapters.
   - How: a request alone against the same request with other adapters,
     rows with no adapter, other positions and fillers, and group sizes around
     every kernel threshold of the GEMM library; ranks padded with zeros.
   - Success: every placement equal.
4. **Check the trainer's backward for padding.**
   - Goal: L2 for gradients, not only L1.
   - How: a sequence alone against right-padded with zero output gradient,
     and with other sequences in the batch.
   - Success: dX, dA and dB unchanged.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Merged weights | One GEMM, fastest; loses small updates in bf16 and cannot match an unmerged trainer | free | Serving only, no parity needed |
| Grouped or gathered library GEMMs | Unmerged, but a request's bits depend on its co-batched tokens | free to moderate (host overhead grows with the number of adapters) | Parity not needed |
| One invariant LoRA kernel on both sides | L1–L4 forward and backward, at the bf16 accuracy floor | moderate (under 2x) to large (2x or more) for a forward; large for a training step; faster than grouped GEMMs with many adapters | RL with LoRA where trainer and engine must agree |

Shown: with one LoRA kernel, trainer and engine gave identical bits in prefill
and decode, every placement passed, and the backward was invariant to padding,
on one GPU type with square layers
([report](../../examples/lora/REPORT.md#results)). Not yet shown: a real
engine's LoRA path (the engine forms were modelled from source), packed
QKV slices, tensor parallelism, quantized base weights.

## What can go wrong

!!! warning "An invariant base GEMM is not enough"
    The LoRA GEMMs have their own M, the number of tokens that share the
    adapter, just as MoE experts do. With only the base made invariant,
    placements still fail. Make the update path invariant too.

!!! warning "Power-of-two scales hide the scale placement"
    A scale of 2 is exact wherever it is applied, so every placement agrees.
    A test with such scales passes and a real adapter fails. Test with scales
    that are not powers of two.

!!! warning "Split-K with atomics is not even L1"
    An engine shrink that splits K and adds into an fp32 buffer with atomics
    changes that buffer on every run. The bf16 cast hides most of it, not all.
    Use a fixed-order reduction.

!!! warning "Deterministic is not invariant"
    A trainer backward through a library GEMM can be identical run to run and
    still change when the batch is padded. Test gradients for L2, not only L1.

!!! warning "Rank padding changes K"
    Padding a small adapter's rank with zeros changes the GEMM's K, and a
    library GEMM may pick another kernel for it. A fixed-order FMA loop adds
    exact zeros and is unaffected.

## Summary

- LoRA adds three rounding sources: the group size, where the scale enters,
  and where base and update meet.
- Merging in bf16 is a lossy choice of form; an engine serving merged weights
  cannot match an unmerged trainer.
- One kernel on both sides, one fp32 FMA, one cast: invariant and the most
  accurate form.
- Run the trainer's logprob pass with LoRA dropout off.

## Go deeper

- [Lab report](../../examples/lora/REPORT.md): every form compared, the
  merge-loss table, the placement sweep, rounding points, accuracy, cost, and
  how vLLM's LoRA kernels round (read from source).
- Code: [lora_kernels.py](../../examples/lora/lora_kernels.py),
  [lora_bi.py](../../examples/lora/lora_bi.py).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
- Regression baselines: [regression](../4-operations/regression.md).
- Next step: [engine parity](engine-parity-vllm.md).
