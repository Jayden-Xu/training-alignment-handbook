# bi_audit: bi model.py dense, 2 layers, rows of 16 tokens

- NVIDIA A40, torch 2.5.1+cu121. Batch of 34 rows, row shapes [16]; target row 0.
- Per-module test: the target row alone against sub-batches of [2, 3, 4, 8, 9, 16, 17, 33], positions ['first', 'middle', 'last'], fillers ['recorded'], with identical input bits.
- Verdicts: no difference: 35.

## First divergence in the full forward (alone against the batch)

None: the target row's output is bitwise equal at all 35 module calls.

## Modules

| Module | Type | Class | Verdict | First batch size that changes it | Fraction of elements | Recipe |
|---|---|---|---|---|---|---|
| `model.embed_tokens` | Embedding | Data movement | no difference |  |  |  |
| `model.layers.0.input_layernorm` | RMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.0.self_attn.q_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.self_attn.k_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.self_attn.v_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.self_attn.rope` | RoPE | Batched or grouped GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm) |
| `model.layers.0.self_attn.core` | AttnCore | Softmax attention | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv) |
| `model.layers.0.self_attn.o_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.post_attention_layernorm` | RMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.0.mlp.gate_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.mlp.up_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.0.mlp.act` | SiluMul | Elementwise | no difference |  |  |  |
| `model.layers.0.mlp.down_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.input_layernorm` | RMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.1.self_attn.q_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.self_attn.k_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.self_attn.v_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.self_attn.rope` | RoPE | Elementwise | no difference |  |  |  |
| `model.layers.1.self_attn.core` | AttnCore | Softmax attention | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv) |
| `model.layers.1.self_attn.o_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.post_attention_layernorm` | RMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `model.layers.1.mlp.gate_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.mlp.up_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.layers.1.mlp.act` | SiluMul | Elementwise | no difference |  |  |  |
| `model.layers.1.mlp.down_proj` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |
| `model.norm` | RMSNorm | Row-wise reduction | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy) |
| `lm_head` | Linear | Dense GEMM | no difference |  |  | [recipe](../../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16) |

## Operators by class

| Class | Default risk | Forward ops (calls) | Backward ops (calls) |
|---|---|---|---|
| Softmax attention | high: SDPA changed a sequence from batch 3 and between prefill and decode (A40) | `attention_fixed_blocks` (2) |  |
| Dense GEMM | high: cuBLAS bf16 rows changed from M = 31, per shape (A40) | `mm_one_config` (15) |  |
| Batched or grouped GEMM | high: bmm runs other kernels than mm and varies with the piece count | `bmm` (3) |  |
| Scatter-add / combine | high: float atomics break run-to-run determinism |  | `scatter_add` (1), `embedding_dense_backward` (1) |
| Row-wise reduction | medium: torch's sum/mean changes its tree below 16 rows (fp32 variance of a 1024-wide row differed from 8 rows, 4096-wide from 2); softmax and log_softmax passed | `row_sum` (5), `_log_softmax` (1), `sum` (1) | `sum` (10), `_log_softmax_backward_data` (1) |
| Elementwise | none expected: one element at a time | `mul` (26), `add` (13), `rsqrt` (5), `neg` (5), `cos` (3), `sin` (3), `silu` (2), `div` (1), `pow` (1), `reciprocal` (1) | `mul` (57), `add` (23), `new_empty_strided` (15), `slice_backward` (10), `neg` (5), `pow` (5), `silu_backward` (2) |
| Cast (rounding point) | none for batch invariance; a rounding point across engines | `_to_copy` (29) | `_to_copy` (11) |
| Data movement | none: exact | `view` (52), `detach` (38), `clone` (38), `t` (30), `empty` (22), `unsqueeze` (17), `slice` (17), `transpose` (11), `expand` (7), `cat` (7), `_unsafe_view` (5), `index` (4) | `view` (51), `t` (45), `clone` (34), `empty` (32), `detach` (20), `copy_` (15), `transpose` (8), `slice` (8), `empty_like` (6), `_unsafe_view` (2), `ones_like` (1), `expand` (1) |

The audit sees operators that pass through the torch dispatcher or are reported with `bi_audit.note`. Compiled graphs and kernels called outside any module are not visible. "No difference" is an observation at the sizes and rows tested.
