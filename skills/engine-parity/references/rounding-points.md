# Where the trainer and vLLM round differently

This page lists every place in a Llama/Qwen2-style decoder where the trainer
and the engine can round differently, and what each side does there.
"Trainer" means [examples/whole_model/model.py](../../../examples/whole_model/model.py),
which follows the transformers sequence of operations. "vLLM" means vLLM 0.19.1
serving Qwen2.5-0.5B-Instruct in bf16 on one GPU type, with torch
2.10.0+cu128. The hardware and the results are in
[examples/cross_engine/REPORT.md](../../../examples/cross_engine/REPORT.md).

The Evidence column uses three kinds of evidence:

- **src**: read from vLLM 0.19.1's Python source (the file is named).
- **replay**: measured by op replay. The engine's captured input is fed to the
  trainer's candidate implementation, and the outputs are compared bit for bit
  ([cross_engine.py](../../../examples/cross_engine/cross_engine.py) `op_replay`,
  on one 50-token prompt: a prefill and two decode steps through all 24 layers).
- **tokens**: measured on the logprobs of realized tokens (64 prompts, batch
  sizes 1–64).

The wheel does not ship the CUDA sources of vLLM's custom ops
(`torch.ops._C.*`) or of FlashAttention, so their behaviour below was
measured, not read from source.

## Rounding points, op by op

| # | Rounding point | Trainer (model.py, stock) | vLLM eager (custom ops) | vLLM batch-invariant eager (`VLLM_BATCH_INVARIANT=1`, on the GPU tested) | Evidence |
|---|---|---|---|---|---|
| 1 | Embedding | a gather, which is exact | a gather, which is exact | a gather, which is exact | src |
| 2 | Residual add before a norm | `h = dt(h + o)`; the norm then reads the rounded `h` | fused into `fused_add_rms_norm`, which works in place | `x + residual` computed in bf16 (`layernorm.py fused_add_rms_norm`), then the norm | src; replay shows that all three round the sum |
| 3 | RMSNorm variance | `mean(fp32(x)^2)` with a torch reduction, then `* rsqrt(var + eps)` | a CUDA kernel; it equals the trainer's formula in 137 of 141 calls, so its reduction order differs | Triton `_rms_norm_kernel`, which takes `tl.sum` over a 1024-wide block, computes `sum / n`, then `1 / sqrt(mean + eps)` | src (BI); replay |
| 4 | Norm weight | cast to bf16 first, then `* w` in bf16 | the same as the trainer (replay is equal) | `(x * inv) * w` in fp32, then a single cast | src (BI); replay: 74% of elements equal the trainer's |
| 5 | Q/K/V packing | three GEMMs (N = 896, 128, 128) | one packed GEMM, N = 1152 | one packed GEMM, N = 1152 | src (`qwen2.py` QKVParallelLinear); replay: separate cuBLAS GEMMs give different bits in 30 of 72 calls |
| 6 | Q/K/V bias | added in the GEMM epilogue before rounding, `dt(sum + b)` (cuBLAS addmm) | the same (`F.linear(x, W, b)`) | added after the cast, `dt(dt(sum) + b)` (`linear_batch_invariant`) | src; replay is equal in 72/72 calls for the matching choice and in 0/72 for the other |
| 7 | GEMM kernel (q/k/v, o, gate/up, down) | cuBLAS, which picks a kernel per shape | cuBLAS, which picks a kernel per shape (not L2) | `matmul_persistent` (Triton, 128x128x64 tiles, fp32 accumulation in K order). At M = 50, cuBLAS gave the same bits for q/k/v and o, but not for gate/up (25/72) or down (24/72) | src; replay |
| 8 | Gate/up packing | two GEMMs | one packed GEMM, N = 9728 | one packed GEMM, N = 9728 | src; replay: separate cuBLAS GEMMs are equal in 48/72 calls, the packed GEMM in 72/72 (eager) |
| 9 | RoPE table | transformers computes `inv_freq` on the CPU and the angles and cos/sin on the GPU in fp32, then casts to bf16 | `cos_sin_cache` is built in fp32 on the GPU (`base ** (arange/D)` on the device), then cast to bf16 | the same as eager | src (`rotary_embedding/base.py`); measured: 26 of 262,144 table entries differ, the first at position 459 |
| 10 | RoPE formula | `dt(dt(q cos) + dt(rot(q) sin))` | `_C.rotary_embedding`, in place | the same as eager | replay: bit-equal to the trainer's formula (72/72 with either table, at 50 tokens) |
| 11 | Attention | SDPA, with K/V expanded to the number of query heads | FlashAttention 2 (vllm_flash_attn) on paged KV (block 16), with `num_splits=0`, so a heuristic picks the split | the same, but with `num_splits=1` | src (`flash_attn.py`); replay: SDPA is equal in 11–12 of 24 layers, FA2 called from the trainer in 24/24 |
| 12 | Attention in decode and in prefill | the trainer only runs prefill | each decode step attends to the cache | the same | replay: the FA2 decode output equals an FA2 prefill of the whole sequence in the trainer, in 48/48 calls (num_splits 0 and 1) |
| 13 | SiLU and multiply | `dt(silu(g)) * u` in bf16 | `_C.silu_and_mul` | the same | replay: bit-equal (72/72) |
| 14 | Final norm | as in rows 2–4 | the fused add + norm | the Triton norm applied to the bf16 sum | src; replay |
| 15 | lm_head | cuBLAS on all T positions | cuBLAS (`F.linear`), only on the rows it samples from (M = batch size at decode) | **cuBLAS**. The mode overrides `aten::mm/addmm/linear` only on a listed set of compute capabilities (`batch_invariant.py enable_batch_invariant_mode`), and the GPU tested is not in that set. The lm_head goes through `F.linear`, so it stays on cuBLAS; the mode only pins the cuBLAS workspace | src; tokens: this breaks the engine's own L2 (1,116 of 15,231 logprobs differ between running alone and batched) |
| 16 | Logits dtype entering the sampler | the bf16 logits are cast to fp32, then `log_softmax` runs in fp32 | `logits.log_softmax(-1, dtype=float32)` on the GPU | `aten::_log_softmax` is overridden by a Triton kernel that takes the max per 1024-wide block, accumulates `sum(tl.exp(x - max))` block by block, and returns `x - max - log(sum)` | src (`sampler.py compute_logprobs`, `batch_invariant.py`); measured: in eager mode the engine's logprob equals GPU `log_softmax(fp32(logits))` bit for bit |
| 17 | log_softmax kernel | torch | torch | Triton (the row above) | tokens: reverting it to torch changes 56% of logprobs, by up to 1.9e-6 |
| 18 | Prompt logprobs | computed in the same forward | a separate `compute_logits` call on the prompt rows (M = chunk size) | the same | src |
| 19 | Compiled mode (the default) | eager | custom ops are turned off; RMSNorm, RoPE and SiLU run `forward_native` through Inductor, which fuses them; CUDA graphs are used | the same, with Triton GEMMs | src (`config/vllm.py` sets custom_ops to "none" under Inductor); tokens: no trainer choice we tried matched it |
| 20 | Cascade attention | not used | can be used for shared prefixes | disabled by the mode | src (`config/vllm.py`) |
| 21 | CPU vs GPU `log_softmax` | GPU | GPU | GPU | measured: the CPU result differed by about 2.5e-5 on the values checked, so compute both sides on the GPU |

## The rounding sequence that reached L4 on the GPU tested

On the engine side, run with `VLLM_BATCH_INVARIANT=1`, `enforce_eager=True`
and `attention_backend="FLASH_ATTN"`, turn prefix caching off, and route the
lm_head through vLLM's own `linear_batch_invariant`. The routing is a runtime
patch of `UnquantizedEmbeddingMethod.apply`, applied by
`engine_run.py --patch-lmhead`.

On the trainer side, use the same weights and make these choices
(`trainer.forward_matched` with `TARGETS["bi_eager_lmhead"]`):

- pack Q/K/V and gate/up;
- use `matmul_persistent` for every linear layer;
- add the bias after the bf16 cast;
- apply vLLM's Triton RMSNorm to the bf16 residual sum;
- use vLLM's `cos_sin_cache`;
- compute RoPE and SiLU in either form, since the two forms are bit-equal;
- run FA2 on paged KV with `num_splits=1`;
- use `matmul_persistent` for the lm_head;
- run vLLM's Triton `log_softmax` on the fp32 logits.

Starting from that state, a leave-one-out pass reverted one choice at a time
and counted the tokens that differ, out of 76,155:

- GEMMs back to separate cuBLAS calls: 76,155;
- the norm reverted: 76,155;
- the attention reverted: 75,585;
- the RoPE table reverted: 9,470 (the prompts longer than 459 tokens);
- log_softmax reverted: 42,515;
- the lm_head back on cuBLAS: 1,510;
- the RoPE formula or SiLU reverted: 0.

## Checklist for a new model or engine

For a new model or engine, check each of these:

- How the projections are packed (QKV, gate/up, fused experts), and where each
  bias is added.
- Residual add: is the sum rounded before the norm, or passed to the norm in
  fp32? Fused kernels and compiled `forward_native` paths differ here.
- Norm: the reduction order, `sum/n` or `sum * (1/n)`, `rsqrt` or `1/sqrt`,
  and whether the weight is applied before or after the cast.
- RoPE: the device and dtype the table is built with, the formula, and
  whether the rotation is interleaved or half-split.
- Attention: the kernel, the split-KV setting, the paged layout, whether
  decode and prefill take different paths, and cascade attention.
- Activation: whether `silu(g)` is rounded before the multiply.
- lm_head: which kernel actually runs on your GPU, and at which M.
- Logprob: the dtype of the logits, the log_softmax kernel, the device, the
  logprobs mode (raw or processed), and whether temperature is applied before
  or after.
- The compile path: which ops a compiler fuses, and whether it keeps the
  intermediate casts.
