# bi_audit: hf LlamaForCausalLM (transformers, random weights, sdpa), 2 layers, rows of 16 tokens

- NVIDIA A40, torch 2.5.1+cu121. Batch of 34 rows, row shapes [16]; target row 0.
- Per-module test: the target row alone against sub-batches of [2, 3, 4, 8, 9, 16, 17, 33], positions ['first', 'middle', 'last'], fillers ['recorded'], with identical input bits.
- Verdicts: no difference: 9, not tested: 1, variant: 22.

## First divergence in the full forward (alone against the batch)

`model.layers.0.self_attn.q_proj` is the first module whose target-row output differs (40.48% of elements; its inputs were equal). 29 of 31 module calls differ.

Leaf modules whose inputs agree and whose output differs (where variance enters): `model.layers.0.self_attn.q_proj` (40.48%), `model.layers.0.self_attn.k_proj` (26.95%), `model.layers.0.self_attn.v_proj` (25.95%).

## Modules

| Module | Type | Class | Verdict | First batch size that changes it | Fraction of elements | Recipe |
|---|---|---|---|---|---|---|
| `model.embed_tokens` | Embedding | Data movement | no difference |  |  |  |
| `model.rotary_emb` | LlamaRotaryEmbedding | Batched or grouped GEMM | not tested: no batch-first tensor output |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm) |
| `model.layers.0.input_layernorm` | LlamaRMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.0.self_attn.q_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 40.50% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.self_attn.k_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 4 | 40.41% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.self_attn.v_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 4 | 42.04% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.self_attn.o_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 40.36% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.post_attention_layernorm` | LlamaRMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.0.mlp.gate_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 39.12% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.mlp.act_fn` | SiLU | Elementwise | no difference |  |  |  |
| `model.layers.0.mlp.up_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 39.08% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.mlp.down_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 48.78% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.input_layernorm` | LlamaRMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.1.self_attn.q_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 40.00% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.self_attn.k_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 4 | 41.46% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.self_attn.v_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 4 | 41.28% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.self_attn.o_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 39.88% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.post_attention_layernorm` | LlamaRMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.1.mlp.gate_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 38.96% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.mlp.act_fn` | SiLU | Elementwise | no difference |  |  |  |
| `model.layers.1.mlp.up_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 38.76% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.mlp.down_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 49.65% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.norm` | LlamaRMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `lm_head` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |

## Operators by class

| Class | Default risk | Forward ops (calls) | Backward ops (calls) |
|---|---|---|---|
| Softmax attention | high: SDPA changed a sequence from batch 3 and between prefill and decode (A40) | `_scaled_dot_product_flash_attention` (2) | `_scaled_dot_product_flash_attention_backward` (2) |
| Dense GEMM | high: cuBLAS bf16 rows changed from M = 31, per shape (A40) | `mm` (15) | `mm` (30) |
| Batched or grouped GEMM | high: bmm runs other kernels than mm and varies with the piece count | `bmm` (1) |  |
| Scatter-add / combine | high: float atomics break run-to-run determinism |  | `scatter_add` (1), `embedding_dense_backward` (1) |
| Row-wise reduction | medium: torch's sum/mean changes its tree below 16 rows (fp32 variance of a 1024-wide row differed from 8 rows, 4096-wide from 2); softmax and log_softmax passed | `mean` (5), `_log_softmax` (1), `sum` (1) | `sum` (14), `_log_softmax_backward_data` (1) |
| Elementwise | none expected: one element at a time | `mul` (22), `add` (13), `pow` (5), `rsqrt` (5), `neg` (5), `silu` (2), `cos` (1), `sin` (1) | `mul` (52), `slice_backward` (29), `add` (23), `pow` (10), `neg` (5), `div` (5), `silu_backward` (2) |
| Cast (rounding point) | none for batch invariance; a rounding point across engines | `_to_copy` (15) | `_to_copy` (11) |
| Data movement | none: exact | `slice` (33), `view` (25), `_unsafe_view` (20), `t` (15), `unsqueeze` (13), `transpose` (9), `detach` (8), `clone` (8), `expand` (7), `cat` (5), `embedding` (1), `arange` (1) | `t` (60), `detach` (50), `view` (41), `transpose` (8), `slice` (8), `expand` (6), `clone` (6), `_unsafe_view` (6), `squeeze` (4), `ones_like` (1), `unsqueeze` (1), `new_zeros` (1) |

The audit sees operators that pass through the torch dispatcher or are reported with `bi_audit.note`. Compiled graphs and kernels called outside any module are not visible. "No difference" is an observation at the sizes and rows tested.
