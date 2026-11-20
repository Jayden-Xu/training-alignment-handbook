# Trainer against vLLM (L4): lab report

This is the full record behind the handbook page
[Engine parity](../../handbook/2-rl/engine-parity-vllm.md). The page teaches the step;
this report keeps the setup, every measurement and how the case was built.

## Summary

- Qwen2.5-0.5B on an A16: default vLLM differs from a transformers-style
  trainer at ~100% of logprobs (KL ~1e-3), and is itself not batch-invariant.
- **Fix:** `VLLM_BATCH_INVARIANT=1`, eager, an lm_head patch (on sm86 the mode
  leaves it on cuBLAS) and a trainer that calls vLLM's own kernels: all 76,155
  logprobs are bit-equal at batch 1–64.
- Most rounding points matter: GEMM packing, bias placement and kernel, RMSNorm,
  attention, the RoPE table, lm_head and log_softmax each break equality when
  reverted.
- **Cost:** 0.45–0.73x default vLLM throughput. The matched trainer is forward
  only, and compiled vLLM could not be matched.

## How

The whole-model case showed a trainer and an engine built from the same
kernels agree bit for bit. In RL the engine is not ours: it is vLLM, with its
own model code, kernels and compiler. This case measures how far a trainer is
from vLLM on a real checkpoint, finds each place where the two round
differently, and moves the trainer to vLLM's rounding one point at a time.
With vLLM's batch-invariant mode, eager execution and one engine patch, the
trainer's realized-token logprobs equal vLLM's bit for bit on every token
tested, at batch sizes 1 to 64.
### The fix

Engine: `VLLM_BATCH_INVARIANT=1`, `enforce_eager=True`,
`attention_backend="FLASH_ATTN"`, and the lm_head patch above. Trainer: the
same weights through `trainer.forward_matched` with vLLM's kernels called
directly (`TARGETS["bi_eager_lmhead"]`): packed projections on vLLM's
`matmul_persistent` with the bias after the cast; vLLM's Triton RMSNorm;
vLLM's `cos_sin_cache`; FlashAttention 2 from `vllm.vllm_flash_attn` on a
paged cache with `num_splits=1`; `matmul_persistent` for the lm_head; vLLM's
Triton `log_softmax`.

Steps, one rounding point at a time (tokens differing of 76,155; batch-1 run of
15,231; first engine tensor the trainer does not reproduce, from the per-layer
comparison on a captured prompt):

| Step | All runs | Batch 1 | First mismatch |
|---|---|---|---|
| model.py stock | 76,150 | 15,230 | layer 0, stream after attention (11% of elements equal) |
| + packed Q/K/V, gate/up (cuBLAS) | 76,155 | 15,231 | layer 0 (43%) |
| + bias after the cast | 76,155 | 15,231 | layer 0 (51%) |
| + vLLM's Triton GEMM | 76,155 | 15,231 | layer 0 (51%) |
| + vLLM's RMSNorm | 75,985 | 15,197 | layer 0 (99.1%) |
| + vLLM's RoPE table | 75,985 | 15,197 | layer 0 (99.1%) |
| + vLLM's RoPE kernel | 75,985 | 15,197 | layer 0 (99.1%) |
| + FlashAttention 2, `num_splits=1` | 43,110 | 8,622 | none: all 24 layers, final norm and logits equal |
| + vLLM's `silu_and_mul` | 43,110 | 8,622 | none |
| + lm_head on `matmul_persistent` | 42,515 | 8,503 | none |
| + vLLM's Triton `log_softmax` | **0** | **0** | none |

The token count stays near 100% until the attention step, because every
rounding point upstream of the logits must match first. The per-layer column
shows the progress.

Leave-one-out from the final state (tokens differing of 76,155):

| Reverted to model.py's choice | Differing | Max \|Δlogp\| |
|---|---|---|
| GEMMs (separate cuBLAS) | 76,155 | 0.71 |
| RMSNorm | 76,155 | 0.45 |
| attention (SDPA) | 75,585 | 0.24 |
| log_softmax (torch) | 42,515 | 1.9e-6 |
| RoPE table (transformers') | 9,470 | 0.16 |
| lm_head (cuBLAS, M = sequence) | 1,510 | 4.2e-4 |
| RoPE formula (transformers') | 0 | 0 |
| SiLU (torch) | 0 | 0 |

The RoPE table matters only for prompts past position 459. Under eager vLLM
without batch-invariant mode, the same procedure matches every layer of a
prefill, but 5,616 of 15,231 batch-1 logprobs still differ: cuBLAS chooses
other kernels for the engine's M (1 at decode, the sampled rows at the
lm_head) than for the trainer's M (the whole sequence). That is engine L2
variance; no trainer choice removes it.

## Evidence

### Setup, code and results

- Code: [examples/cross_engine/](.):
  `engine_run.py` (one vLLM configuration per process: generate, record
  logprobs, logits, per-module captures, throughput), `trainer.py` (model.py
  with Qwen2.5 weights, and `forward_matched`, whose rounding points can be
  switched one at a time), `cross_engine.py` (all checks), `tput.sh`.
- Results: [results-A16.json](results-A16.json)
  (torch 2.10 trainer and all engines),
  [results-A16-torch251.json](results-A16-torch251.json)
  (the same engine records against model.py under torch 2.5.1).
- Skill: [skills/engine-parity](../../skills/engine-parity/SKILL.md); every
  rounding point: [rounding-points.md](../../skills/engine-parity/references/rounding-points.md).
- Setup: one NVIDIA A16 (sm86, 15 GB), driver 535. Engine: vLLM 0.19.1, torch
  2.10.0+cu128, Triton 3.6.0; also vLLM 0.6.3.post1 (torch 2.4.0). Trainer:
  model.py in the same torch 2.10 environment, and in torch 2.5.1+cu121 /
  Triton 3.1.0. Model: Qwen2.5-0.5B-Instruct, bf16.
- Workload: 64 prompts of 4 to 703 tokens (seeded synthetic text), 48 greedy
  tokens each. Each engine configuration runs every prompt alone, in batches of
  4, 16 and 64, and batch 64 twice: 76,155 realized-token logprobs per
  configuration (prompt tokens and generated tokens), 15,231 of them in the
  batch-1 run. Prefix caching off. The trainer teacher-forces each run's own
  tokens. KL uses full logits at 192 generated positions (4 prompts, batch 1).

### Where the variance comes from

Three separate sources, measured in this order.

**1. The engine against itself.** Default vLLM is L1 (the same batch twice
gives the same bits) but not L2: a prompt's logprobs change when it is batched.

| vLLM config | Attention | L1 | Alone vs batch 64: sequences with other tokens | Logprobs differing | Max \|Δlogp\| |
|---|---|---|---|---|---|
| default (compiled, CUDA graphs) | FlashAttention 2 | yes | 23 / 64 | 13,673 / 14,305 | 0.41 |
| `enforce_eager` | FlashAttention 2 | yes | 31 / 64 | 13,167 / 14,000 | 0.45 |
| default | Triton | yes | 25 / 64 | 13,625 / 14,261 | 0.30 |
| default | FlashInfer | yes | 19 / 64 | 13,806 / 14,440 | 0.49 |
| default | FlexAttention | yes | 24 / 64 | 13,658 / 14,296 | 0.37 |
| `VLLM_BATCH_INVARIANT=1` (compiled) | FlashAttention 2 | yes | 0 / 64 | 1,133 / 15,231 | 0.0049 |
| batch-invariant, eager | FlashAttention 2 | yes | 0 / 64 | 1,116 / 15,231 | 0.0040 |
| batch-invariant, eager | Triton | yes | 0 / 64 | 1,137 / 15,231 | 0.0018 |
| batch-invariant, eager, lm_head patched | FlashAttention 2 | yes | 0 / 64 | **0 / 15,231** | 0 |
| vLLM 0.6.3 default / eager | FlashAttention 2 | yes | 26 / 64 | 8,649 / 14,215 | 0.68 |

Batches of 4 and 16 give the same picture. Logprobs are compared up to the
first token where two runs diverge.

vLLM 0.19.1 names its mode `VLLM_BATCH_INVARIANT=1` (`vllm/envs.py`). Read from
`model_executor/layers/batch_invariant.py` and the layers that check the flag:
- it replaces RMSNorm (with and without the residual add) by a Triton kernel,
  every `UnquantizedLinearMethod` GEMM by a persistent Triton matmul with the
  bias added after the cast, `aten::_log_softmax` and `mean.dim` by Triton
  kernels, `softmax` by a torch composition, and `bmm`;
- it sets FlashAttention's `num_splits=1` and turns cascade attention off;
- it overrides `aten::mm`, `addmm`, `matmul` and `linear` **only on sm80, sm89
  and sm100**. On the A16 (sm86) the lm_head, which reaches cuBLAS through
  `F.linear`, stays on cuBLAS with a pinned workspace;
- it refuses to start without an explicit attention backend (FLASH_ATTN,
  TRITON_ATTN or two MLA backends).

The remaining 1,116 differences in batch-invariant eager mode are all in that
lm_head. Routing it through vLLM's own `linear_batch_invariant` (a runtime
patch of `UnquantizedEmbeddingMethod.apply`, `engine_run.py --patch-lmhead`)
removes them: alone and batched agree on all 15,231 logprobs per run.

**2. Trainer against engine: the rounding sequence.** model.py follows
transformers. vLLM's Qwen2 differs at these points (full table in
[rounding-points.md](../../skills/engine-parity/references/rounding-points.md)):
- Q/K/V and gate/up are packed into one GEMM each; the kernel cuBLAS picks
  depends on N, so packing alone changes bits;
- in batch-invariant mode the q/k/v bias is added after the bf16 cast, not
  in the GEMM's fp32 sum;
- in batch-invariant mode RMSNorm multiplies the weight in fp32 before the
  single cast, divides by n, and uses `1/sqrt`; eager vLLM's CUDA norm uses
  the transformers formula with another reduction order;
- the RoPE table is built on the GPU (transformers computes `inv_freq` on the
  CPU); 26 of 262,144 bf16 entries differ, the first at position 459;
- attention is FlashAttention 2 on a paged cache;
- the lm_head runs on the sampled rows only (M = batch size at decode);
- in batch-invariant mode `log_softmax` is a Triton kernel.

Op replay settled the rest: vLLM's RoPE kernel and `silu_and_mul` give the same
bits as the transformers formulas (72 of 72 calls each).

**3. The trainer's own stack.** model.py's stock op set under torch 2.5.1 and
under torch 2.10 differs in 14,548 of 15,231 realized logprobs (cuBLAS and SDPA
change between versions). model.py's `bi` op set gives the same bits in both
(0 of 15,231) under Triton 3.1 and 3.6.

### Results

Trainer against engine, realized-token logprobs, all runs (76,155 tokens):

| Trainer | vLLM default | vLLM eager | vLLM BI eager | vLLM BI eager + lm_head patch |
|---|---|---|---|---|
| model.py stock | 76,154 (max 1.28) | 76,129 (0.63) | 76,150 (0.69) | 76,150 (0.69) |
| model.py bi | 76,152 (1.28) | 76,060 (0.73) | 76,155 (0.60) | 76,155 (0.60) |
| matched to eager | 76,155 (1.28) | 64,285 (0.73) | 76,155 (0.51) | 76,155 (0.51) |
| matched to BI eager (lm_head on cuBLAS) | 76,154 (1.33) | 76,151 (0.92) | 2,527 (0.0040) | 1,510 (4.2e-4) |
| matched to BI eager + patch | 76,154 (1.33) | 76,151 (0.92) | 1,116 (0.0040) | **0** |

Mean KL(engine ‖ trainer) over 192 generated positions: model.py stock against
vLLM default 1.5e-3, model.py bi 9.6e-4, the best matched trainer 9.5e-4;
against eager vLLM, model.py stock 1.4e-3; against compiled batch-invariant
vLLM, model.py stock 1.5e-3 and the best matched trainer 1.2e-3. The matched
trainer against the patched batch-invariant engine: KL 0, all 192 logit rows
bit-equal.

No trainer choice tried matches the compiled engines (default, and
batch-invariant without `enforce_eager`): 76,154–76,155 of 76,155 differ.
With compilation vLLM turns its custom ops off, and Inductor fuses RMSNorm,
RoPE and SiLU into neighbouring ops.

Cost (output tokens/s; 128-token prompts, 128 new tokens, greedy, no
logprobs; one configuration at a time on one GPU; warm-up 1, median of 3):

| vLLM config | B = 1 | B = 8 | B = 64 |
|---|---|---|---|
| default | 139 | 952 | 4,067 |
| `enforce_eager` | 100 | 706 | 3,928 |
| batch-invariant (compiled) | 86 | 639 | 3,313 |
| batch-invariant, eager | 66 | 470 | 3,007 |
| batch-invariant, eager, lm_head patch | 63 | 471 | 2,959 |

The L4 configuration costs 0.45x the default's throughput at batch 1, 0.49x at
8 and 0.73x at 64. The trainer side costs nothing extra in the forward:
forward and log-probabilities of one 512-token sequence take 66.2 ms with the
matched trainer, 68.3 ms with model.py stock and 71.0 ms with model.py bi
(warm-up 2, median of 5).

### What this shows

- Batch-invariant engine mode is necessary but not sufficient. On sm86 it
  leaves the lm_head on cuBLAS, which alone breaks engine L2 at 1,116 of 15,231
  logprobs. Read what the mode covers on your GPU.
- Bit equality with a real engine was reached by making the trainer call the
  engine's own kernels, not by re-implementing them. model.py's own
  batch-invariant kernels are L1–L3 but differ from vLLM's at every token.
- Each rounding point is small and most of them matter. GEMMs (packing, bias
  placement, kernel), RMSNorm, attention, RoPE table, lm_head and log_softmax
  each break equality when reverted; the RoPE formula and SiLU were equal
  already. The log_softmax kernel alone moves 56% of logprobs by up to 1.9e-6.
- Default vLLM is far from any trainer: KL about 1e-3, max |Δlogp| above 1,
  and its own batch-to-batch variance is of the same size.

## Limits

- One model (0.5B, 24 layers), one GPU type, single GPU (no TP). Sequences up to
  751 tokens; positions where the RoPE tables differ beyond that are untested.
- Prefix caching, speculative decoding, sampling with temperature, and CUDA
  graphs with batch-invariant mode are untested at L4. Chunked prefill was on
  (vLLM's default); which prompts were split across steps was not recorded.
- The matched trainer is forward only: vLLM's Triton GEMM, RMSNorm and
  `log_softmax` and the FlashAttention build in vLLM have no backward here. A
  training step needs backward kernels with the same forward bits (untested).
- The lm_head patch is a runtime patch in the engine process, not a vLLM option.
- Compiled vLLM was not matched. Localizing it needs boundary fingerprints
  (OpGuard per-vector XOR, `tools/opguard_bridge.py xor_vectors`), because
  module hooks do not fire inside compiled graphs.
- vLLM 0.6.3 ran (its default and eager modes gave identical bits); it was
  compared with model.py only. Its eager mode differs from 0.19.1's eager in
  10,049 of 14,805 batch-1 logprobs.
- Throughput was measured in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, needed
  for the hooks), which can lower small-batch throughput against the default
  multiprocess engine.

## How it was built (what took more than one try)

- Batch-invariant mode failed to start until an attention backend was named.
- vLLM 0.6.3 ran out of memory at batch 64: its V0 engine materializes fp32
  logits for every prompt row of a step for prompt logprobs. Capping
  `max_num_batched_tokens` at 2,048 fixed it.
- FlashInfer needed its JIT toolchain on `PATH` (ninja from the environment,
  CUDA 12.1 `nvcc`); the first build failed without them.
- The engine's reported logprob first looked 2.5e-5 off the captured logits:
  the check ran `log_softmax` on the CPU. On the GPU the two agree bit for bit.
- The first analysis counted the batch-invariant L2 checks as regressions:
  the expectation assumed the mode covers the lm_head on sm86. It does not;
  the checks now expect failure without the patch and success with it.
- Engine records hold vLLM objects; a torch-only environment cannot unpickle
  them, so `--strip` writes copies without the captures for the torch 2.5.1 run.


## Context moved from the part 2 overview

These sections were on the overview page [RL: train–inference alignment](../../handbook/2-rl/README.md). Numbers from other groups are as reported there ([references](../../handbook/references.md)).

### Why it matters, and what is reported

- IsoExec (SkyRL) cites a GLM run at train–inference KL ≈ 0.013 where clipping
  dropped about 45% of tokens and the reward collapsed around step 20; with
  bitwise alignment there was no clipping and the run was stable.
- The same authors saw no clear reward gain from removing the mismatch in a
  50-step run, and Yichuan Wang's async study found gains depend on the
  configuration. What holds consistently is the numerics: a gap of exactly 0 at
  staleness 0, and a gap that does not run away as the off-policy window widens.

Treat parity as a **debugging and audit tool** first. Whether to run it in
production is a cost decision.


### Three ways to reach parity

| Approach | What it does | Reported cost |
|---|---|---|
| Match the engine to the trainer op by op (verl-omni, HF-numerics mode) | Unfuse and reorder engine ops so casts and rounding follow the trainer: norm cast before the weight, residual add in bf16, the RoPE formula, router weight dtype; batch-invariant ops on both sides; the same log-prob formula | ≈37% rollout throughput (small prototype) |
| One model definition for both (TorchTitan RL + vLLM) | The engine keeps only cache management; the forward is shared, so only kernels remain to align | 2–3x trainer throughput for GDN models (recurrent forward), about 5x on an agent workload |
| A cross-framework execution contract (IsoExec) | Pin every rounding-relevant choice and check it by hash; parallel-invariant kernels (a fixed reduction tree for GEMM, fixed-order expert combine, one GDN algorithm) | +31.3% generation, +18.6% training, +25.3% RL step (Qwen3.5-35B-A3B) |


### Synchronous, on-policy (staleness 0)

With parity enabled on both sides, the target is a log-probability difference of
**exactly 0** for every token. Run it for a short window, for example 20 on-policy
steps, as Yichuan Wang suggests. If the gap is 0 and training is still broken,
the infrastructure is exonerated and the problem is in data or algorithm. If the
gap is not 0, localize it (below).


### Worked example: one element at q_norm

A vLLM rollout against a training stack, 38 tokens, batch-invariant vLLM:
- within the request's forward, per-vector fingerprints showed all 1,216 head
  vectors of layer-0 `q_norm`'s input equal on both sides;
- its output differed in exactly one vector: token 12, head 17;
- a full-tensor compare found one bf16 element different (`1.5859375` against
  `1.578125`).

The likely cause class is a rounding-point difference between the two RMSNorm
implementations, in line with the norm-cast-order mismatch verl-omni reports.
This has not been confirmed by an intervention. Two traps from this case:
- the training side's capture first landed on a trailing `.float()` rather than
  the norm;
- a norm fused away on the trainer looked like a divergence on the engine side
  until other-side evidence was required.

Measured end to end ([trainer vs vLLM](REPORT.md)): on
Qwen2.5-0.5B, default vLLM differs from a transformers-style trainer at nearly
every realized token (KL ~1e-3) and is itself not batch-invariant. Bit equality
needed vLLM's batch-invariant eager mode, the lm_head moved to its
batch-invariant GEMM, and a trainer that calls vLLM's own kernels; it costs
0.45x throughput at batch 1 and 0.73x at 64. The matched trainer is forward
only, and compiled vLLM could not be matched.


### Measured in the RL loop (summary from the overview)

Measured in the [RL loop](../rl_loop/REPORT.md), a small RL loop run through the
harness on A16s: with batch-invariant op sets on both sides, numeric error is
bit-zero on every non-mixed token (sync and async lag 1–2), so the whole gap is
staleness. Records and row fingerprints cost about 13 ms per step on the tiny
model, plus snapshots. Four planted faults were localized end to end in a known-answer walk (the
data-flow order was written from `model.py`, not discovered). The page
also lists the harness gaps found on the way.
