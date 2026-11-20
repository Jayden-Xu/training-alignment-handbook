# Checkpoint conversion: lab report

This is the record behind the handbook page
[Checkpoint conversion](../../handbook/5-scenarios/checkpoint-conversion.md).
The page teaches the check; this report keeps the setup, every measurement and
what the demo does and does not show.

## Summary

- A Hugging Face Llama checkpoint converted to a Megatron-style layout (QKV
  fused and interleaved per query group, gate and up fused per rank, TP 2 and
  TP 4 shards, vocab-parallel embedding and output layer) and back is
  bit-exact: 39 of 39 tensors equal, position by position, at both degrees.
- [ckpt_compare.py](../../tools/ckpt_compare.py) with the map
  [mcore_map.json](mcore_map.json) matched every row of the 39 logical
  parameters between the HF and the Megatron-style checkpoints (19,465 rows),
  between TP 2 and TP 4, and after a bf16 → fp32 widening (reported for all 39
  parameters, every row exact). One comparison took about 0.19 s on CPU after
  loading.
- Five planted faults were each named by logical parameter and row, with the
  physical tensor and row: a transposed shard (1,024 rows), a wrong QKV
  interleave (12 parameters, 1,280 rows), a bf16 → fp16 → bf16 cast (2,081
  rows in 30 parameters), one flipped bit (1 row), a missing row (1 row).
- The forward check (target layout against the source weights run with the
  same reduction) gave equal logits, bit for bit, for the correct conversion
  at TP 2 and TP 4, and differed for four of the five faults. It did **not**
  see the fp16 cast: the 2,081 changed rows moved no logit on the fixed inputs.
- Two faults a row multiset cannot see, by design, were seen by the forward
  check: two q heads swapped inside one parameter, and row-parallel shards
  joined in the wrong order. The second is also seen by the default row
  digest, not by XOR row fingerprints.
- XOR row fingerprints (OpGuard's `lastdim-u32le-v1`) missed 11 of the 2,081
  rows the cast changed: in each, two elements in the same 16-bit lane changed
  by the same bits, and the changes cancel in the XOR. This is why the tool's
  default row key is a digest of the row's bytes in order.

## How

### The conversion under test

[convert.py](convert.py) converts a state dict with `LlamaForCausalLM`'s names
to the names and packing of Megatron-core's GPT model, one file per TP rank
(`mp_rank_XX/model_optim_rng.pt`, the state dict under `"model"`):

| Logical parameter (HF) | Megatron-style tensor, per rank r | How it is packed |
|---|---|---|
| `embed_tokens`, `lm_head` | `embedding.word_embeddings`, `output_layer` | vocabulary rows split across ranks |
| `q_proj`, `k_proj`, `v_proj` | `linear_qkv` | per query group: its q heads, its k head, its v head; groups split across ranks |
| `o_proj` | `linear_proj` | columns split across ranks (row-parallel) |
| `gate_proj`, `up_proj` | `linear_fc1` | the rank's gate rows, then its up rows |
| `down_proj` | `linear_fc2` | columns split across ranks (row-parallel) |
| `input_layernorm`, `post_attention_layernorm`, `model.norm` | `linear_qkv.layer_norm_weight`, `linear_fc1.layer_norm_weight`, `final_layernorm` | one copy per rank |

`qkv_layout="concat"` is a deliberate bug: all q, then all k, then all v, cut
into TP shards.

### The two checks

1. **Rows.** [ckpt_compare.py](../../tools/ckpt_compare.py) reads both
   checkpoints, builds each logical parameter on each side from the map, and
   matches the rows (last-dimension vectors) of the two sides as multisets by
   their exact bits. Rows split along the last dimension (row-parallel shards)
   are joined first (`"dim": -1` in the map). Replicated copies must be
   bit-equal. Every leftover row is listed with its logical index and its
   physical origin.
2. **Forward.** [ckpt_conversion.py](ckpt_conversion.py) runs one Llama forward
   twice on fixed token ids (2 × 16): once reading the Megatron-style shards
   the way Megatron's layers read them (QKV split per query group, gate/up
   split per rank, vocab-parallel lookup, row-parallel partial sums), once on
   the HF tensors. Both sides sum row-parallel partials in rank order in fp32
   with one cast, so the only difference left is where each weight sits.
   Column-parallel outputs are computed from the shards reassembled per the
   layout, one GEMM per projection, so the GEMM shapes are the source's.

The plain forward equals `transformers`' `LlamaForCausalLM` (eager attention,
bf16) bit for bit on these inputs, when the model is built the way
`from_pretrained` builds it (see Findings).

### Ground truth for the faults

Each HF row was replaced by its own id and pushed through the same converter,
so every physical row of the Megatron-style checkpoint knows which HF row it
holds. For each fault, the rows whose bits changed were compared with the
rows the tool named: HF side as (parameter, row), Megatron side as physical
tensor and row.

## Evidence

### Setup, code and results

- Code: [ckpt_conversion.py](ckpt_conversion.py) (the demo),
  [convert.py](convert.py) (the converter), [mcore_map.json](mcore_map.json)
  (the map), [ckpt_compare.py](../../tools/ckpt_compare.py) (the tool).
- Results: [results-cpu.json](results-cpu.json) (full mode, `bi-results-v1`).
- Setup: CPU only, 24 torch threads (the host's GPU was not used; the result's
  `env` lists it); torch 2.5.1, transformers 4.49.0, Python 3.10.
- Full mode: vocabulary 2,048, hidden 512, intermediate 1,024, 4 layers, 16
  heads, 4 KV heads, head dimension 32, bf16 weights drawn with std 0.02, norm
  weights 1 + 0.1 · N(0, 1). Quick mode: vocabulary 512, hidden 128,
  intermediate 256, 2 layers, 8 heads, 4 KV heads, head dimension 16; the same
  checks, with the same outcomes except the one noted below.
- Run: `python examples/ckpt_conversion/ckpt_conversion.py [--quick] --json FILE`
  (27 ok, 0 regressions, 6 info in both modes; about 10 s in full mode).

### Correct conversions

| Check | Result (full mode) |
|---|---|
| Round trip HF → TP 2 → HF, and HF → TP 4 → HF | 39 of 39 tensors bit-equal at each degree |
| HF vs TP 2, HF vs TP 4, with the map | 39 of 39 logical parameters, every row matched (all also equal by position) |
| TP 2 vs TP 4, with the map on both sides | every row matched |
| HF vs a TP 2 checkpoint saved in fp32 | 39 dtype changes reported as exact widenings; every row matched after widening |
| Forward at the target layout, TP 2 and TP 4 | 0 of 65,536 logits differ |
| Command line, clean conversion / flipped bit | exit status 0 / 1 |
| Time for one comparison (19,465 rows each side, after loading) | 0.18–0.19 s |

Two information checks show why the map and the matched reduction matter:

- **Global scope without a map** (every row of A anywhere in B): the 4,096
  rows of the 8 row-parallel weights (`o_proj`, `down_proj`) have no partner,
  because their shards hold half rows; on the Megatron side 8,201 rows are
  left over (those half rows, plus the 9 extra copies of replicated norms).
- **The reduction changes the logits.** The source forward with its
  row-parallel layers summed as at TP 2 differs from the same weights at TP 1
  in 52,363 of 65,536 logits (max |Δ| 0.0195); at TP 4, in 52,342 (max |Δ|
  0.0176). A forward check that compares TP 2 against TP 1 bits would report a
  correct conversion as wrong. The demo gives both sides the same reduction
  instead.

### Planted faults (TP 2)

| Fault | Rows changed (HF rows) | ckpt_compare (default digest) | XOR rows | Forward check |
|---|---|---|---|---|
| Transposed shard: rank 1's `linear_fc1` of layer 0 stored as its transpose, reshaped | 1,024 | named exactly: `gate_proj` and `up_proj` of layer 0, rows 512–1,023 each, origin `mp_rank_01/…linear_fc1.weight[j]` | same rows | 65,297 logits differ |
| Wrong QKV interleave (`qkv_layout="concat"`) | — | named q, k and v of all 4 layers and nothing else; 1,280 rows | same rows | 65,458 differ |
| bf16 → fp16 → bf16 cast of every tensor | 2,081 in 30 parameters | named exactly | 2,070: **11 missed** | **0 differ** |
| One flipped bit (rank 0, layer 1 `linear_qkv`, row 144, mantissa bit 3) | 1 | `k_proj` of layer 1, row 16, origin `mp_rank_00/…linear_qkv.weight[144]` | same row | 20,353 differ |
| Missing row (last row of rank 1's `output_layer`) | 1 | `lm_head`, row 2,047 unmatched on the HF side | same row | shape differs |

In quick mode the cast changed 89 rows and XOR missed none; that is why the XOR
and forward checks of the cast are recorded as information, not claimed.

### What a row multiset cannot see

| Fault | Row digest | XOR rows | Forward check |
|---|---|---|---|
| Two q heads swapped inside group 0 of layer 0 (rank 0) | all rows matched (by design) | all rows matched | 58,975 of 65,536 logits differ |
| Rank 0 and rank 1 `linear_fc2` shards swapped, layer 1 | 512 rows of `down_proj` unmatched on each side | all rows matched | 65,263 differ |

## Findings

- **XOR row fingerprints cancel equal changes in one lane.** Row 361 of layer
  0's `down_proj` changed in its last mantissa bit in columns 32 and 160 under the
  fp16 round trip. Both elements sit in the low half of their 32-bit words, so
  the two changes (0x0001 each) cancel and the row's XOR is unchanged. A cast
  rounds many small values the same way, so this is not rare under a cast: 11
  of 2,081 changed rows here. Recorded XOR fingerprints (OpGuard's weight-sync
  records) remain the cheap live check; an offline conversion check should
  hash the row.
- **A forward check on a few inputs is not a bit check.** The cast changed
  values below about 7.6e-6 in magnitude; their effect on every logit rounded
  away in bf16. The row check is what proves a copy exact; the forward check
  proves placement.
- **`.to(torch.bfloat16)` on a freshly built transformers model rounds the RoPE
  `inv_freq` buffer to bf16** (transformers 4.49). The plain forward then
  differed from `LlamaForCausalLM` in 7,238 of 16,384 logits (quick mode).
  Built under a bf16 default dtype, as `from_pretrained` does, the buffer stays
  fp32 and the two are bit-equal.

## Limits

- The Megatron-style layout is written by this demo's converter, with
  Megatron-core's names and packing; Megatron-core's own loader and
  `dist_checkpointing` formats were not run.
- The target forward is emulated in plain torch with the source's arithmetic.
  A real target framework runs its own kernels; compare it against the source
  with its own reduction, or within the bounds of the
  [parallelism page](../../handbook/2-rl/parallelism.md).
- Not exercised: pipeline-parallel resharding (it moves whole tensors between
  files, which the directory loader covers), vocabulary padding (the map's
  `slice`), tied embeddings, biases, MoE experts, optimizer state, a real
  model's weights, a narrowing cast between formats with an overflow.
- One CPU stack. The checks are bitwise copies and CPU forwards; nothing here
  depends on a GPU.

## Reproduce

```bash
python examples/ckpt_conversion/ckpt_conversion.py --quick --json /tmp/ckpt.json
python examples/ckpt_conversion/ckpt_conversion.py --quick --keep /tmp/ckpt-work
python tools/ckpt_compare.py /tmp/ckpt-work/hf /tmp/ckpt-work/fault_one_flipped_bit \
    --map examples/ckpt_conversion/mcore_map.json --layouts hf mcore --set b:r=2
```
