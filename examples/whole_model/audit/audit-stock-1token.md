# bi_audit: stock model.py dense, 2 layers, rows of 1 token

- NVIDIA A40, torch 2.5.1+cu121. Batch of 34 rows, row shapes [1]; target row 0.
- Per-module test: the target row alone against sub-batches of [2, 3, 4, 8, 9, 16, 17, 33], positions ['first', 'middle', 'last'], fillers ['recorded'], with identical input bits.
- Verdicts: no difference: 17, variant: 17, variant inside: 1.
- Warning: a non-leading dimension equals the batch size 34: a tensor may be mistaken for batch-first

## First divergence in the full forward (alone against the batch)

`model.layers.1.self_attn.q_proj` is the first module whose target-row output differs (0.20% of elements; its inputs were equal). 8 of 35 module calls differ.

Leaf modules whose inputs agree and whose output differs (where variance enters): `model.layers.1.self_attn.q_proj` (0.20%), `model.layers.1.self_attn.v_proj` (0.39%), `model.layers.1.mlp.down_proj` (0.10%), `lm_head` (0.06%).

## Modules

| Module | Type | Class | Verdict | First batch size that changes it | Fraction of elements | Recipe |
|---|---|---|---|---|---|---|
| `model.embed_tokens` | Embedding | Data movement | no difference |  |  |  |
| `model.layers.0.input_layernorm` | RMSNorm | Row-wise reduction | variant inside: `aten::mean` (float32) differs, the output did not at these rows | 8 | 100.00% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.0.self_attn.q_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 39.55% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.self_attn.k_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.self_attn.v_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.self_attn.rope` | RoPE | Elementwise | no difference |  |  |  |
| `model.layers.0.self_attn.core` | AttnCore | Softmax attention | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv) |
| `model.layers.0.self_attn.o_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 41.11% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.post_attention_layernorm` | RMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.0.mlp.gate_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 0.03% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.mlp.up_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.mlp.act` | SiluMul | Elementwise | no difference |  |  |  |
| `model.layers.0.mlp.down_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 38.09% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.input_layernorm` | RMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.1.self_attn.q_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 38.57% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.self_attn.k_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.self_attn.v_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 0.39% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.self_attn.rope` | RoPE | Elementwise | no difference |  |  |  |
| `model.layers.1.self_attn.core` | AttnCore | Softmax attention | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv) |
| `model.layers.1.self_attn.o_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 39.75% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.post_attention_layernorm` | RMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.1.mlp.gate_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.mlp.up_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.mlp.act` | SiluMul | Elementwise | no difference |  |  |  |
| `model.layers.1.mlp.down_proj` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 39.16% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.norm` | RMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `lm_head` | Linear | Dense GEMM | variant (first op: `aten::mm`) | 2 | 0.06% | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |

## Operators by class

| Class | Default risk | Forward ops (calls) | Backward ops (calls) |
|---|---|---|---|
| Softmax attention | high: SDPA changed a sequence from batch 3 and between prefill and decode (A40) | `_scaled_dot_product_flash_attention` (2) | `_scaled_dot_product_flash_attention_backward` (2) |
| Dense GEMM | high: cuBLAS bf16 rows changed from M = 31, per shape (A40) | `mm` (15) | `mm` (30) |
| Scatter-add / combine | high: float atomics break run-to-run determinism |  | `scatter_add` (1), `embedding_dense_backward` (1) |
| Row-wise reduction | medium: torch's sum/mean changes its tree below 16 rows (fp32 variance of a 1024-wide row differed from 8 rows, 4096-wide from 2); softmax and log_softmax passed | `mean` (5), `_log_softmax` (1), `sum` (1) | `sum` (14), `_log_softmax_backward_data` (1) |
| Elementwise | none expected: one element at a time | `mul` (20), `add` (13), `pow` (5), `rsqrt` (5), `neg` (5), `silu` (2) | `mul` (52), `add` (23), `slice_backward` (18), `pow` (10), `neg` (5), `div` (5), `silu_backward` (2) |
| Cast (rounding point) | none for batch invariance; a rounding point across engines | `_to_copy` (11) | `_to_copy` (11) |
| Data movement | none: exact | `view` (23), `_unsafe_view` (19), `slice` (19), `t` (15), `unsqueeze` (9), `detach` (8), `transpose` (8), `expand` (5), `index` (4), `cat` (4), `clone` (4), `arange` (1) | `t` (60), `detach` (50), `view` (47), `transpose` (8), `slice` (8), `expand` (6), `squeeze` (4), `ones_like` (1), `unsqueeze` (1), `new_zeros` (1) |

The audit sees operators that pass through the torch dispatcher or are reported with `bi_audit.note`. Compiled graphs and kernels called outside any module are not visible. "No difference" is an observation at the sizes and rows tested.
