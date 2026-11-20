# Verification

A kernel is batch-invariant at a level only if it passes this matrix, measured
on the GPU, dtype and versions you ship. It passes with zero bitwise mismatches.
The checker is [tools/bi_check.py](../../../tools/bi_check.py).

## Compare bits, not values

`torch.equal` and `==` treat `-0.0` and `+0.0` as equal, and NaN as unequal to
itself. Compare integer views of the same width instead:
`bi_check.bit_equal(a, b)`. Sign-of-zero differences are real: a masked tail that
adds `+0.0` in one path and `-0.0` in another is a variance, and it can matter
downstream (`1/x`, `copysign`).

## The matrix

| Axis | Values | Why |
|---|---|---|
| Batch size M | 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 1024, 2048; plus the SM count and its multiples ±1; every threshold seen in the code; every CUDA-graph capture size | dispatch thresholds sit at powers of two, tile multiples and SM-count multiples |
| Position of the row | first, middle, last | alignment and vector-width paths, tail handling |
| Co-batched content | random, zeros, large (×1e3) | content-dependent statistics: dynamic scales, capacity, top-k |
| Modes (attention-like) | prefill; per-token decode; chunked prefill at several chunk sizes (including ones not dividing the block size); several KV page sizes | split-KV and block iteration, backend selection by arguments |
| Repeat | at least 5 runs of the same batch | atomics and races (L1) |
| Backward | the same matrix on gradients, when training uses the op; plus a sequence alone vs right-padded in a batch with zero output gradient on the padding; deterministic mode with `warn_only=False` | deterministic-backward variants; atomics in dQ; padding handling |
| Layout | page sizes 16, 32, 64, 128 with random placement; pack offsets and lengths that are not multiples of the block | blocks that follow the page or the pack instead of absolute positions |
| Parallel degree (L5) | TP and EP 1, 2, 4, 8; several token shardings; collectives on messages of 1 KB–256 MB; top-k ≥ 3 and 3 or more ranks | reductions grouped by rank or shard; ring orders that follow chunk boundaries |
| Deployment wrappers | eager, `torch.compile` with the deployed settings, CUDA graphs at the deployed sizes; compiled against eager bitwise | per-shape specialization, re-fusion of norms and activations |
| Cross-engine (L4) | the same rows through the trainer's op and the engine's op | rounding points, kernel identity |

## Using the checker

```python
from bi_check import check_batch, check_modes, check_repeat

rep = check_batch(lambda x: my_gemm(x, W), rows)    # rows: [R, K] on the GPU
print(rep.summary())                                # PASS, or failing sizes

rep = check_modes({"prefill": run_prefill, "decode": run_decode,
                   "chunk-16": run_chunked(16), "chunk-7": run_chunked(7)})
```

`python tools/bi_check.py --demo` checks a few stock torch ops, with results in a
demo JSON under `tools/`. Measured results, on one GPU type with torch 2.5.1 unless
noted (multi-GPU rows: nodes of 8 GPUs of a second type); details, including the
hardware, are in the case studies and each case's `examples/<case>/REPORT.md`. To re-check every row on your own stack,
run the [regression suite](../../../handbook/4-operations/regression.md):

| Op | Result |
|---|---|
| `torch.mm` bf16 (cuBLAS) | L1 yes; L2 no: fails from M=31 (4/4096 elements differ in a row) |
| matmul row by row | L2 (control: invariant by construction) |
| RMSNorm from torch ops, fp32 `log_softmax` | L2 at the shapes tested |
| SDPA causal | L3 no: decode vs prefill ~10% of elements differ; chunked (explicit mask) vs prefill 1 element |
| SDPA causal, a sequence among others (GQA 8/2, T=256) | L2 no: the same sequence changes at batch 3, 8 and 16 (17.6% of elements) but not at 2 |
| Fixed-block Triton attention (`tools/bi_kernels.py`) | L2 and L3 yes: prefill, decode and chunks 7/16/64/100 bitwise equal |
| Its atomic-free backward | L1 and L2 yes on dQ/dK/dV: batch 2–16, and right padding to 256/320 |
| SDPA backward | flash: L1 no (dQ). Strict deterministic flash: L1 yes, L2 no, because its forward changes from batch 3. Strict deterministic memory-efficient: L1 and L2 yes at the shapes tested |
| Paged KV (pages 16–128) and varlen packing, fixed-block attention | L2 and L3 yes; blocks that follow the page or the pack offset fail |
| MLA, one form with every piece invariant | L2 and L3 yes; explicit prefill with absorbed decode fails against a trainer from layer 2 |
| Sparse attention, one-config projection plus stable top-k | L2 and L3 yes; a cuBLAS projection flips selected sets |
| GDN recurrent / chunked | recurrent L1–L3; chunked L3 only with chunk boundaries at absolute positions |
| MoE fixed-slot combine, EP 1–16 (two nodes of 8 GPUs; [report](../../../examples/moe_ep/REPORT.md)) | L5 yes on one node and across two; per-rank pre-reduce fails L5 from top-3 |
| NCCL all-reduce (8 and 16 ranks, one and two nodes) | L2 no with defaults or Ring pinned; yes with Tree on one or two nodes; yes with the rank-order fallback |
| Compiled block (`torch.compile`) | L2 yes; equal to eager only with every rounding op opaque |
| Fingerprint probes in compiled code and CUDA graphs | probe = opguard_bridge fingerprints; stock graph = eager; stock buckets 8/16 vs 1 differ at o_proj; bi buckets 1/8/16 equal; bi_opaque_all compile = eager (3 modes); probes leave compiled logits unchanged (isolated) |
| Sampling: per-request generator / torch.cumsum / bi_sampling | generator: L2 yes, replay after preemption, rollback, spec chunks: no; cumsum: L2 no, L1 no at V=151,936; bi_sampling: L1–L3 yes, chi-square pass |
| LoRA linear, one-config base + row-selected adapter, one cast | L1, L2, prefill/decode, trainer vs engine yes; grouped cuBLAS and gathered bmm fail L2; vLLM-style split-K atomics fail L1 in the fp32 buffer |
| Training step, 4-layer LM, 20 steps | default: first difference at step 0 (SDPA dQ); strict and fixed: equal across processes |
| Embedding backward | L1 yes; L2 no: two orders split at 3072 indices |
| DP gradient all-reduce / FSDP-style reduce-scatter (emulated; flat gradient, reduce-scatter, sharded AdamW), DP 1–16 (two nodes of 8 GPUs; [report](../../../examples/training_dp/REPORT.md)) | packing L2: no under Ring, yes under Tree or the ordered all-reduce; owner L2: no for NCCL reduce-scatter, yes ordered; L5: no for any per-rank batched backward; yes with logical shards, end to end with AdamW and elastic resume |
| Whole model (dense, MoE, wide, Qwen2 shapes), bi kernels | L1–L3 in all engine modes; activation-gradient L2; stock: every log p differs, KL 1e-3 to 1.4e-2 |
| Whole model, TP 1–16 (nodes of 8 GPUs, two nodes for 16; [report](../../../examples/whole_model_tp/REPORT.md)) | logical TP C=16 + fixed-order vocab log_softmax: L5 yes in the forward, engine modes, prefill/decode at different TP and activation gradients (TP 2, 4); stock NCCL TP: every realized log-prob differs at every degree |
| vLLM 0.19.1 vs model.py, Qwen2.5-0.5B ([report](../../../examples/cross_engine/REPORT.md)) | default, eager, Triton, FlashInfer, Flex: L1 yes, L2 no (96% of logprobs alone vs batch 64); VLLM_BATCH_INVARIANT=1: L2 no on the GPU tested, where the mode leaves the lm_head on cuBLAS (7%), yes with the lm_head on vLLM's Triton GEMM; L4 yes (0/76,155) against a trainer calling vLLM's kernels, eager only |

## Reading failures

| Pattern | Likely mechanism |
|---|---|
| Passes below some M, fails at and above it, the same at every position and filler | shape-dispatched kernel or tile configuration (heuristic threshold) |
| Fails at scattered sizes | split-K or stream-K driven by tile count; grid-size-dependent splitting |
| Fails only at some positions | alignment-dependent vector width; tail handling |
| Fails only with some fillers | content-dependent statistics: batch-wide amax, capacity dropping, top-k over the batch |
| Fails on repeat with an identical batch | atomics or a race: L1 broken, fix that first |
| Modes differ in many elements | different block iteration or split-KV in decode |
| Modes differ in a few elements | different backend chosen by call arguments, or partially masked block handling |
| Passes in eager, fails under compile or graphs | per-shape specialization or re-fusion; register the op as opaque |
| Compiled is L2 but differs from eager | re-fusion moved rounding points (norms, activations); make every op that rounds opaque |
| Differs between page sizes or pack offsets, each self-consistent | compute blocks follow the layout instead of absolute positions |
| Passes at top-2 or with 2 ranks, fails from 3 | an order-dependent sum, hidden because a two-term sum commutes |
| L1 and L2 pass, but results change with the parallel degree | a reduction grouped by rank or shard (L5) |
| A collective fails inside larger messages with its algorithm pinned | chunk boundaries move with message size and change each element's ring order |
| Discrete outputs flip (selected keys, routed experts, dropped tokens) | an upstream batch-variant score or a capacity rule; compare sets before values |
| Passes in steady state, fails at the first iteration or after a resume | torch DDP reduces its first iteration (and every restart) in a single bucket; under Ring that is another order |
| Sampled tokens pass across batch sizes but change after preemption, rollback, another request's n or a speculative chunk | the noise comes from a generator's running state, not from (seed, position, purpose, sample) |
| Each side is L2/L3, but trainer and engine differ (L4) | a different rounding point: packing, bias, norm form, RoPE table, or a different kernel; compare the two rounding sequences |

## After the operator passes

1. Run the model-level check: the same sequence alone and inside batches of
   several sizes must give identical per-token log-probabilities.
2. For L4, the trainer and the engine on the same weights at staleness zero
   must give identical log-probabilities.
3. If not, localize with per-vector fingerprints (OpGuard,
   `OPGUARD_XOR_ROWS=1`) along the data flow, as in the RL part.
4. Measure throughput at representative prefill M and decode M (1–64), and
   record it with the result.
