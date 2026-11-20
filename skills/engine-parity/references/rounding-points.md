# Rounding points: trainer (model.py) against vLLM

Every place where the trainer and the engine can round differently in a
Llama/Qwen2-style decoder, with what each side does. "Trainer" is
[examples/whole_model/model.py](../../../examples/whole_model/model.py) (the
transformers sequence). "vLLM" is vLLM 0.19.1 serving Qwen2.5-0.5B-Instruct in
bf16 on one GPU type, torch 2.10.0+cu128 (hardware and results:
[examples/cross_engine/REPORT.md](../../../examples/cross_engine/REPORT.md)).

Evidence column:
- **src** — read from vLLM 0.19.1's Python source (file named).
- **replay** — measured by op replay: the engine's captured input fed to the
  trainer's candidate, output compared bit for bit
  ([cross_engine.py](../../../examples/cross_engine/cross_engine.py) `op_replay`;
  one 50-token prompt, prefill and two decode steps, 24 layers).
- **tokens** — measured on realized-token logprobs (64 prompts, batches 1–64).

CUDA sources of vLLM's custom ops (`torch.ops._C.*`) and FlashAttention are not
in the wheel; their behaviour below is measured, not read.

## Per op

| # | Rounding point | Trainer (model.py, stock) | vLLM eager (custom ops) | vLLM batch-invariant eager (`VLLM_BATCH_INVARIANT=1`, on the GPU tested) | Evidence |
|---|---|---|---|---|---|
| 1 | Embedding | gather, exact | gather, exact | gather, exact | src |
| 2 | Residual add before a norm | `h = dt(h + o)`, then norm reads `h` | fused into `fused_add_rms_norm` (in place) | `x + residual` in bf16 (`layernorm.py fused_add_rms_norm`), then norm | src; replay: rounded sum on all three |
| 3 | RMSNorm variance | `mean(fp32(x)^2)` (torch reduction), `* rsqrt(var + eps)` | CUDA kernel; equals the trainer's formula in 137 of 141 calls, so the reduction order differs | Triton `_rms_norm_kernel`: `tl.sum` over a 1024-wide block, `sum / n`, `1 / sqrt(mean + eps)` | src (BI); replay |
| 4 | Norm weight | cast to bf16, then `* w` in bf16 | same as trainer (replay equal) | `(x * inv) * w` in fp32, one cast | src (BI); replay: 74% of elements equal to the trainer's |
| 5 | Q/K/V packing | three GEMMs (N = 896, 128, 128) | one GEMM, N = 1152 | one GEMM, N = 1152 | src (`qwen2.py` QKVParallelLinear); replay: separate cuBLAS GEMMs give other bits in 30 of 72 calls |
| 6 | Q/K/V bias | in the GEMM epilogue: `dt(sum + b)` (cuBLAS addmm) | same (`F.linear(x, W, b)`) | after the cast: `dt(dt(sum) + b)` (`linear_batch_invariant`) | src; replay 72/72 for the matching choice, 0/72 for the other |
| 7 | GEMM kernel (q/k/v, o, gate/up, down) | cuBLAS, kernel chosen per shape | cuBLAS, kernel chosen per shape (not L2) | `matmul_persistent` (Triton, 128x128x64 tiles, fp32 accumulate in K order); cuBLAS gave the same bits for q/k/v and o at M = 50, not for gate/up (25/72) and down (24/72) | src; replay |
| 8 | Gate/up packing | two GEMMs | one GEMM, N = 9728 | one GEMM, N = 9728 | src; replay: cuBLAS separate 48/72 vs packed 72/72 (eager) |
| 9 | RoPE table | transformers: `inv_freq` on the CPU, angles and cos/sin on the GPU in fp32, cast to bf16 | `cos_sin_cache` built in fp32 on the GPU (`base ** (arange/D)` on the device), cast to bf16 | same as eager | src (`rotary_embedding/base.py`); measured: 26 of 262,144 entries differ, first at position 459 |
| 10 | RoPE formula | `dt(dt(q cos) + dt(rot(q) sin))` | `_C.rotary_embedding`, in place | same as eager | replay: bit-equal to the trainer's formula (72/72 with either table at 50 tokens) |
| 11 | Attention | SDPA on K/V expanded to the query heads | FlashAttention 2 (vllm_flash_attn), paged KV (block 16), `num_splits=0` (heuristic) | same, `num_splits=1` | src (`flash_attn.py`); replay: SDPA equal in 11–12 of 24 layers; FA2 called from the trainer 24/24 |
| 12 | Attention, decode vs prefill | trainer is prefill only | decode step against the cache | same | replay: FA2 decode output equals FA2 prefill of the whole sequence in the trainer, 48/48 calls (num_splits 0 and 1) |
| 13 | SiLU and multiply | `dt(silu(g)) * u` in bf16 | `_C.silu_and_mul` | same | replay: bit-equal (72/72) |
| 14 | Final norm | as 2–4 | fused add + norm | Triton norm on the bf16 sum | src; replay |
| 15 | lm_head | cuBLAS on all T positions | cuBLAS (`F.linear`) on the rows it samples only (M = batch at decode) | **cuBLAS**: the mode overrides `aten::mm/addmm/linear` only on a listed set of compute capabilities (`batch_invariant.py enable_batch_invariant_mode`), which leaves out the GPU tested, and the lm_head goes through `F.linear`; only the cuBLAS workspace is pinned | src; tokens: breaks engine L2 (1,116 of 15,231 logprobs, alone vs batched) |
| 16 | Logits dtype into the sampler | fp32 `log_softmax` of the bf16 logits cast to fp32 | `logits.log_softmax(-1, dtype=float32)` on the GPU | `aten::_log_softmax` overridden by a Triton kernel: max per 1024 block, `sum(tl.exp(x - max))` block by block, `x - max - log(sum)` | src (`sampler.py compute_logprobs`, `batch_invariant.py`); measured: engine logprob = GPU `log_softmax(fp32(logits))` bit for bit in eager mode |
| 17 | log_softmax kernel | torch | torch | Triton (above) | tokens: reverting it changes 56% of logprobs by up to 1.9e-6 |
| 18 | Prompt logprobs | same forward | a separate `compute_logits` on prompt rows (M = chunk) | same | src |
| 19 | Compiled mode (default) | eager | custom ops off; RMSNorm, RoPE, SiLU run `forward_native` through Inductor, fused; CUDA graphs | same with Triton GEMMs | src (`config/vllm.py`: custom_ops "none" under Inductor); tokens: not matched by any trainer choice tried |
| 20 | Cascade attention | none | possible for shared prefixes | disabled by the mode | src (`config/vllm.py`) |
| 21 | CPU vs GPU `log_softmax` | GPU | GPU | GPU | measured: the CPU result differed by about 2.5e-5 on the values checked; compute both sides on the GPU |

## The sequence that gives L4 on the GPU tested

Engine: `VLLM_BATCH_INVARIANT=1`, `enforce_eager=True`,
`attention_backend="FLASH_ATTN"`, prefix caching off, and the lm_head routed
through vLLM's own `linear_batch_invariant` (a runtime patch of
`UnquantizedEmbeddingMethod.apply`; `engine_run.py --patch-lmhead`).

Trainer: the same weights with these choices (`trainer.forward_matched`,
`TARGETS["bi_eager_lmhead"]`):
packed Q/K/V and gate/up; `matmul_persistent` for every linear; bias after the
bf16 cast; vLLM's Triton RMSNorm on the bf16 residual sum; vLLM's
`cos_sin_cache`; RoPE and SiLU in either form (bit-equal); FA2 paged with
`num_splits=1`; `matmul_persistent` for the lm_head; vLLM's Triton
`log_softmax` on the fp32 logits.

Leave-one-out from that state (tokens differing of 76,155):
GEMMs back to separate cuBLAS 76,155; norm back 76,155; attention back
75,585; RoPE table back 9,470 (prompts longer than 459 tokens); log_softmax
back 42,515; lm_head back to cuBLAS 1,510; RoPE formula or SiLU back 0.

## Checklist for a new model or engine

- Packing of projections (QKV, gate/up, fused experts) and where each bias enters.
- Residual add: rounded before the norm, or fp32 into the norm (fused kernels
  and compiled `forward_native` paths differ).
- Norm: reduction order, `sum/n` vs `sum * (1/n)`, `rsqrt` vs `1/sqrt`, weight
  before or after the cast.
- RoPE: table device and dtype, formula, interleaved vs half-split.
- Attention: kernel, split-KV setting, paged layout, decode vs prefill path,
  cascade attention.
- Activation: rounding of `silu(g)` before the multiply.
- lm_head: which kernel actually runs on your GPU, and with which M.
- Logprob: dtype of the logits, log_softmax kernel, device, logprobs mode (raw
  or processed), temperature applied before or after.
- The compile path: which ops a compiler fuses, and whether it keeps
  intermediate casts.
