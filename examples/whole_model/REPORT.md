# Whole model, trainer against engine (L1–L3): lab report

This is the full record behind the handbook page
[A whole model, trainer against engine](../../handbook/2-rl/whole-model.md).
The page gives the answer and what to do; this report gives the setup, every
configuration, every table and how the case was built.

## Summary

- A Llama-architecture model built from the handbook's kernels kept L1, L2 and
  L3 in every serving mode tested (decode in changing batches, chunks, prefix
  hit, speculative, preempt, varlen) and in activation gradients, for four
  architectures (A40).
- With stock ops nearly every realized log-probability differs between trainer
  and engine: KL 1e-3 to 1.4e-2 at staleness zero.
- Rounding points matter as much as batch invariance: one rounding change moved
  86–100% of logits. L4 needs the engine to follow the trainer's rounding
  sequence.
- Parameter gradients sum over the batch and can only be L1.
  [tools/bi_audit.py](../../tools/bi_audit.py) lists a model's operators and
  finds its batch-variant modules.

## Setup

Every kernel case checked one operator alone. RL train–inference alignment
needs more: a whole model must give a sequence the same logits alone and in a
batch, and in trainer mode (teacher-forced prefill) and engine mode (decode
with a KV cache). This case builds a Llama-architecture model from the
handbook's kernels and checks exactly that. It also adds a tool that lists a
model's operators and finds its batch-variant modules.

### The model

A decoder-only LM that follows transformers' `LlamaForCausalLM`: RMSNorm, SwiGLU
MLP, GQA, RoPE (theta 10000) on q and k, no biases, untied `lm_head`. Parameter
names are HF's, so the state dict loads into `LlamaForCausalLM` with
`strict=True`. Options cover Qwen2: bias on q/k/v, tied embeddings,
configurable eps and theta, and a loader from an HF state dict.

| Variant | Shape | Parameters |
|---|---|---|
| dense (default) | hidden 1024, 8 layers, 16/4 heads, MLP 3584, vocab 32000 | 175M |
| MoE | the dense model with `moe_bi.py`'s layer: 64 experts, top-8, inner 512 | 892M |
| wide | hidden 4096, 2 layers, 32/8 heads, MLP 14336 (Llama-3-8B layers) | 698M |
| qwen2 | Qwen2.5-0.5B shapes (hidden 896, 14/2 heads, q/k/v bias, tied, theta 1e6), 4 layers | 196M |

Random weights, fixed seed. The `lm_head` is scaled so logits have std 4.5:
the dense model's mean next-token entropy is 3.27 nats (uniform is 10.4) and
the realized log-probability of sampled tokens averages -3.25. Targets are
sequences sampled from the model itself. The qwen2 variant is more peaked
(entropy 0.30 nats) because its tied embedding carries the logit scale.

### Two op sets, one rounding sequence

The top of `model.py` writes down
every cast, which ops run in fp32, the RoPE formula and where the norm weight
multiplies. Both op sets follow it and differ only in kernels:

| Op | `stock` | `bi` |
|---|---|---|
| GEMM | `F.linear` (cuBLAS) | `mm_one_config` (tools) |
| attention, trainer | SDPA on K/V expanded to the query heads (as HF) | `attention_fixed_blocks`, `_train` under autograd |
| attention, engine | SDPA with a padded batch and a mask | the per-sequence kernel of `examples/attention_serving` (copied), reading a slot KV cache |
| RMSNorm variance | `x.pow(2).mean(-1)` | a fixed-tree row sum (new, in `model.py`) |
| q/k/v bias (Qwen2) | `F.linear(x, W, b)` (addmm) | fp32-output GEMM, fp32 bias add, one cast |

### Execution modes

`forward_cached` appends any number of tokens per row to
a slot cache, so one engine loop drives all modes: decode in a batch whose
membership changes (fillers join and finish, 1 to 65 rows per step); chunked
prefill (chunks 7, 16, 64, 100); a prefix-cache hit whose prefix KV was computed
inside another request's batch; speculative appends of k = 1..8 tokens with
rollback of rejected tokens; preemption and recompute from scratch; right
padding; packed varlen prefill.

### Code, results and machine

- Code: [examples/whole_model/model.py](model.py) (the
  model, two op sets), [whole_model.py](whole_model.py)
  (the checks), [hf_check.py](hf_check.py) (architecture
  check against transformers), [audit_model.py](audit_model.py)
  and [tools/bi_audit.py](../../tools/bi_audit.py) (the audit).
- Results: [results-a40.json](results-a40.json) (275
  checks: 263 ok, 0 regressions, 12 info),
  [results-hf-a40.json](results-hf-a40.json), audit
  tables in [examples/whole_model/audit/](audit/).
- Setup: one NVIDIA A40 (shared, under a lock), torch 2.5.1+cu121, Triton
  3.1.0; transformers 4.49.0 for the architecture check and the audit of an HF
  model. bf16 weights and activations, fp32 log_softmax.

## Where the variance comes from

- **GEMMs.** cuBLAS changes a row's result with M at every shape of these
  models, often from M = 2. For example 1024→1024 failed at 24 of 26 batch sizes
  tested, 4096→14336 at 2–7 and 63–128 only. In the full model the audit names
  the layer-0 q_proj as the first module that differs.
- **RMSNorm's fp32 variance.** torch's `mean` over a row changes its reduction
  tree with the number of rows: hidden 1024 differed from 8 rows on, hidden
  4096 from 2. The bf16 output hid it here: of 20,000 rows normalized one by
  one against all at once, 310 variances differed and 0 bf16 outputs did
  (hidden 1024). It is a latent risk, so `bi` replaces it.
- **Attention.** SDPA runs other kernels for prefill, masked chunks and
  decode.
- **The other ops passed alone.** fp32 `log_softmax` at vocab 32000, rows
  1–257. The MoE router's softmax, top-k and renormalization on invariant
  GEMM scores, rows 1–2048. RoPE and SiLU are elementwise.

## Results

### Invariance

"Equal" means every logit of every position, and every realized-token
log-probability, bitwise.

| Check | `bi` (dense, MoE, wide; qwen2 for L2, L3 and lengths only) | `stock` |
|---|---|---|
| L1: trainer forward ×5, engine decode in a changing batch ×3 | equal | equal |
| L2 trainer: alone vs batch 2–65, first/middle/last, random, constant and copied fillers; T = 65, 200, 2048 | equal | differs from batch 2 |
| L2 engine: each mode in a changing batch vs the same mode alone | equal | differs |
| L3: decode, chunks 7/16/64/100, speculative k = 1..8, prefix hit, preempt, packed varlen, vs trainer forward | equal | differs |
| L3 at T = 2048: decode and chunk 64 in a changing batch, chunk 100 alone (dense) | equal | differs |
| lengths 1, 2, 63–65, 127–129, 191–193, 257: decode, chunk 7, chunk 64 | equal | differs |
| right padding to 201/256/320 in a batch | equal | differs |
| trainer row 8 of 16 vs engine decode in a changing batch: what RL compares | equal | differs |
| backward: d loss / d activations and logits of one sequence (14 taps), alone vs batch 2–16 (dense), 2–8 (MoE, wide) | equal | differs in every tap |
| backward: parameter gradients, B = 4, 5 runs (L1) | equal | dense and MoE equal; wide: 12 of 21 tensors differ run to run |

The rounding variants below were each checked too: in every variant, decode
equals that variant's own trainer forward.

### What RL would see with stock ops

T = 200 tokens sampled from the model. Fraction of tokens whose realized
log-probability differs, largest |Δ log p|, and KL(trainer ‖ other) between the
two next-token distributions (mean over tokens, maximum).

| Comparison | dense | MoE | wide | qwen2 |
|---|---|---|---|---|
| trainer: alone vs co-batched | 100%, 0.255, 2.1e-3 (max 1.2e-2) | 100%, 1.24, 1.4e-2 (0.27) | 100%, 0.172, 9.6e-4 (5.8e-3) | 99.9%, 0.134, 3.3e-4 (4.2e-3) |
| decode alone vs trainer | 100%, 0.168, 2.0e-3 | 100%, 0.673, 1.5e-2 | 100%, 0.171, 1.1e-3 | 99.5%, 0.124, 3.4e-4 |
| decode in a changing batch vs alone | 96.5%, 0.236, 2.0e-3 | 96.0%, 0.846, 1.5e-2 | 96.0%, 0.154, 1.1e-3 | 95.5%, 0.168, 2.4e-4 |
| chunk 100 in a busy batch vs alone | 49.7%, 0.18, 1.1e-3 | 49.7%, 1.16, 6.9e-3 | 49.7%, 0.124, 4.5e-4 | 49.7%, 0.119, 1.5e-4 |
| trainer (row 8 of 16) vs engine decode in a changing batch | 100%, 0.185, 2.1e-3 (5.3e-3) | 100%, 1.06, 1.4e-2 (0.13) | 100%, 0.135, 9.9e-4 (5.6e-3) | 100%, 0.14, 3.5e-4 (3.9e-3) |

- Almost every token's log-probability differs. The KL is 1e-3 to 2e-3 for the
  dense models and 1.4e-2 for the MoE, where a changed router score moves a
  token to another expert. IsoExec reports a train–inference KL of about 0.013
  on a GLM run (see the [RL part](../../handbook/2-rl/README.md)). The MoE
  number here is of that order, at staleness 0 and identical weights.
- Chunk 100 in a busy batch differs on 49.7% of tokens: its second chunk runs
  as a different GEMM shape when other rows join. The first chunk ran alone in
  both runs.

### Rounding points

Measured with the `bi` op set, dense model, B = 4, T = 256. One choice was
changed at a time, against the reference sequence (HF's).

| Choice changed | Logits changed | Tokens whose log p changed | max \|Δ log p\| | KL mean |
|---|---|---|---|---|
| norm weight multiplies in fp32, then cast | 88.3% | 100% | 0.282 | 2.7e-3 |
| norm entirely in bf16 | 89.0% | 100% | 0.409 | 2.9e-3 |
| residual add in fp32 into the next norm (fused add+norm) | 87.4% | 100% | 0.321 | 2.4e-3 |
| residual stream kept in fp32 | 86.3% | 100% | 0.232 | 2.1e-3 |
| RoPE in fp32 with one rounding | 86.9% | 99.6% | 0.284 | 2.3e-3 |
| RoPE `inv_freq` rounded to bf16 (what `.to(bfloat16)` does to an HF model) | 87.4% | 99.6% | 0.245 | 2.4e-3 |
| RoPE cos/sin computed on the CPU | 0 | 0 | 0 | 0 |
| `lm_head` output kept in fp32 | 100% | 100% | 0.093 | 3.6e-4 |
| `log_softmax` in bf16 | 0 (logits) | 100% | 0.031 | — |

Every rounding point except the RoPE table's device moves almost every logit,
by about as much as stock batch variance does. An engine that is L2 and L3 but
rounds once differently is as far from the trainer as a stock engine. The CPU
and GPU fp32 tables differ in 49,304 of 262,144 entries, but those vanish when
cast to bf16 (2 of 262,144 entries differ, at positions no test reached).

### Architecture check against transformers

This is not a bitwise claim, but it is measured.

| | max \|logit − HF logit\| |
|---|---|
| Llama, fp32, `stock`, B = 4, T = 512, 8 layers; and 128 cached decode steps against HF's cache | 0 (bitwise equal) |
| Llama, bf16, `stock`, prefill and cached decode | 0 (bitwise equal) |
| Llama, bf16, `bi` | 0.34 (13% of logits equal); other kernels |
| Qwen2 (random `Qwen2ForCausalLM`, Qwen2.5-0.5B shapes, 4 layers, biases set to N(0, 0.01)), fp32 and bf16, `stock`, prefill B = 2, T = 256 | 0 (bitwise equal) |
| Qwen2, `stock`, 32 cached decode steps against HF **prefill** (not HF's cache; another composition, recorded as info) | 5.48e-6 (fp32), 0.031 (bf16) |
| Qwen2, bf16, `bi` | 0.031 |
| HF bf16 model made with `.to(bfloat16)` against `from_pretrained(torch_dtype=bf16)` | 0.41 |

- `export_hf()` then `from_pretrained()` gives back every weight bitwise.
- Two findings came from this check. A first version added the q/k/v bias after
  the GEMM's bf16 cast; HF's `F.linear` runs `addmm`, which adds it before. That
  left fp32 differences of 4.5e-6 (from that first version; not in the saved
  results). Second, an HF model cast with `.to(bfloat16)`
  rounds its RoPE `inv_freq` buffer to bf16, while `from_pretrained` keeps it
  fp32. The two HF models differ by up to 0.41 in a logit, so say which one
  an engine is compared against.

### Cost

Tokens/s on the A40, median of 5; prefill at T = 512, decode at context 512.

| | prefill B=1 | B=8 | B=64 | decode B=1 | B=8 | B=64 |
|---|---|---|---|---|---|---|
| dense stock | 85,909 | 180,170 | 196,678 | 149 | 1,152 | 6,568 |
| dense bi | 66,848 | 153,322 | 91,282 | 117 | 935 | 7,235 |
| bi time / stock | 1.29x | 1.18x | **2.16x** | 1.28x | 1.23x | 0.91x |
| wide bi time / stock | 1.51x | 2.34x | — | 1.09x | 0.87x | — |
| MoE bi time / stock | 0.41x | 0.48x | — | 0.56x | 0.43x | — |

- The untuned one-config GEMM dominates large prefill: 2.2–2.3x at 4,096 and
  32,768 tokens. Decode is host-bound at these sizes (eager, no CUDA graphs).
- The MoE `stock` path runs `moe_bi.py`'s loop with cuBLAS per expert, which is
  slower than the Triton GEMMs at these small expert shapes. It is not a
  production MoE kernel.
- The fixed-tree RMSNorm cost 0.85x torch at 16,384 rows and 1.38x at one row
  (hidden 1024: 1.09 against 1.28 ms, 0.111 against 0.081 ms).
- Another job used the GPU during the cost runs (utilization 26–91% before and
  after, one process; recorded in the JSON). Treat the ratios as indicative.

## The audit tool

`tools/bi_audit.py` takes any `nn.Module` with batch-first tensors and a
function that builds a batch from rows. It
1. lists every dispatcher operator of the forward and the backward per module,
   with shapes and dtypes, in the operator classes of part 3 with each
   class's default risk;
2. records each module's inputs in one batched run, replays the module on the
   target row alone and in sub-batches with identical input bits, and gives
   every module its own verdict. It also compares the module's inner
   operators in their own dtype, so a varying fp32 reduction is reported even
   when the bf16 output hides it ("variant inside");
3. names the first module, in forward order, whose target-row output differs
   between the alone and the batched forward, and the leaf modules whose inputs
   agree but whose outputs differ.

Output is JSON and a markdown table. Runs (batch of 34 rows, sub-batches 2–33,
2 layers):

| Model | Rows | Verdicts (module calls) | First divergence |
|---|---|---|---|
| `model.py` stock | 16 tokens | 22 variant, 13 no difference | layer 0 `q_proj` (40% of elements); k_proj and v_proj too |
| `model.py` stock | 1 token | 17 variant, 1 variant inside (`input_layernorm`: fp32 `mean` from 8 rows), 17 no difference | layer 1 `q_proj` |
| `model.py` bi | 16 and 1 token | 35 no difference | none |
| transformers `LlamaForCausalLM`, SDPA | 16 tokens | 22 variant, 9 no difference, 1 not tested (`rotary_emb`: no batch-first output) | layer 0 `q_proj` |

Every variant leaf module was a GEMM. SDPA did not change at 16-token rows,
consistent with earlier cases where it changed from batch 3 at T = 256: test
the lengths you run. The HF run needed `use_cache=False`, otherwise the
recorded `DynamicCache` object is replayed with the wrong batch.

### What it assumes and cannot see

- Tensors whose leading size equals the batch size are batch rows; choose a
  batch size no other dimension equals.
- Modules are pure in eval mode (no cache update, no randomness).
- Kernels called outside the dispatcher (Triton) are invisible unless the code
  calls `bi_audit.note()`; `model.py` does. The module is still tested as a
  black box.
- Compiled graphs, CUDA graphs and exported engines are not visible.
- Code between modules is tested only as part of its parent. After an op that
  regroups rows (sort, nonzero, index_add) inner operators are not compared.
- A "no difference" is an observation at the rows and sizes tested.
- Recipe links in the markdown are relative to the repository root.

## What this shows

1. **A whole model built from verified kernels kept L1, L2 and L3 in every
   mode tested**, forward and activation gradients, for four architectures.
   Nothing failed, so no open limit came from the model.
2. **Kernels alone were not enough.** RMSNorm's torch variance was replaced
   although its bf16 output never moved in these tests; the engine needed a
   per-sequence attention reading a slot cache. The audit found both kinds.
3. **With stock ops, nearly every realized log-probability differs** between
   the trainer and the engine, with KL 1e-3 to 1.4e-2 at staleness zero.
4. **Rounding points matter as much as batch invariance.** Each single
   rounding change moved 86–100% of logits. L4 needs the engine to follow the
   trainer's rounding sequence, which `model.py` writes down.
5. **Parameter gradients sum over the batch** and so can only be L1. Stock
   SDPA backward was L1 at hidden 1024 but not at hidden 4096 (12 of 21
   parameter gradients differed run to run).

## Limits

- One GPU (A40), eager mode. No `torch.compile`, no CUDA graphs.
- Random weights. Qwen2 checked with random weights at real shapes, 4 of 24
  layers; the real checkpoint is not on the machine.
- Not tested here: L4 against an inference engine (see
  [cross-engine-vllm](../../handbook/2-rl/engine-parity-vllm.md)); tensor parallelism (see
  [whole-model-tp](../../handbook/2-rl/parallelism.md)); sampling (the comparison uses
  teacher-forced tokens; see [sampling](../../handbook/2-rl/sampling.md)); sliding windows; batch 64 for
  the MoE and wide cost; backward through the KV-cache path; the MoE backward
  beyond activation gradients; `bi` backward of the q/k/v bias in fp32 (under
  autograd the bias is added after the cast, unlike the forward).
- `model.py` builds its RoPE table lazily. Built inside `torch.compile`, the
  table comes from Inductor's math (178 of 131,072 entries differ from eager)
  and is kept for later eager runs. Build it in eager before compiling
  (measured in [compile-graph-capture](../../handbook/1-pretraining/capture-compiled-code.md)).
- `stock` engine attention pads ragged rows and masks them in one SDPA call;
  a real engine (paged FlashAttention, split-KV) may vary differently.
- The MoE layer is `moe_bi.py`, whose module constants are set per call: one
  MoE geometry per process at a time.

## How it was built

An agent wrote the model, the checks, the audit tool and the original handbook
page (now this report); no human edits. What took more than one try:
- The first architecture check against HF in bf16 differed, from casting HF
  with `.to(bfloat16)` (see the architecture check). The Qwen2 check then exposed the bias
  rounding point; `model.py` was changed to follow `addmm`.
- The first logit scale (std 3.5) gave entropy near 5 nats; raised to 4.5.
- The audit's forward pre-hooks first returned values, which torch takes as
  replacement arguments; its inner-op comparison first flagged `aten::empty`
  outputs (uninitialized memory) as variant.
- The `bi` op set passed every check on its first full run; nothing was fixed
  in the kernels.
- Network drops to the machine interrupted two runs; the full run ran under
  `nohup` in pieces of under 5 minutes each (`run_all.sh`).
