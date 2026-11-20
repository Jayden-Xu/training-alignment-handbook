# Verification

A kernel counts as batch-invariant at a given level only if it passes the
matrix below, measured on the GPU, dtype and versions that you ship. Passing
means zero bitwise mismatches. The checker is
[tools/bi_check.py](../../../tools/bi_check.py).

## Compare bits, not values

`torch.equal` and `==` treat `-0.0` and `+0.0` as equal, and treat NaN as
unequal to itself. Compare integer views of the same width instead, with
`bi_check.bit_equal(a, b)`. A difference in the sign of zero is a real
difference: if a masked tail adds `+0.0` in one path and `-0.0` in another, the
kernel varies, and the difference can matter downstream (`1/x`, `copysign`).

## The test matrix

| Axis | Values | What it catches |
|---|---|---|
| Batch size M | 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 1024, 2048; also the SM count and its multiples, each ±1; every threshold you see in the code; every CUDA-graph capture size | Dispatch thresholds, which sit at powers of two, at multiples of the tile size and at multiples of the SM count |
| Position of the row | First, middle, last | Paths that depend on alignment or vector width, and tail handling |
| Content of the other rows | Random, zeros, large values (×1e3) | Statistics that depend on content: dynamic scales, capacity, top-k |
| Modes (attention-like ops) | Prefill; decode one token at a time; chunked prefill at several chunk sizes, including sizes that do not divide the block size; several KV page sizes | Split-KV and block iteration, and backends selected by the call arguments |
| Repeat | At least 5 runs of the same batch | Atomics and races (L1) |
| Backward | The same matrix on gradients, when training uses the op; also a sequence alone against the same sequence right-padded in a batch, with zero output gradient on the padding; deterministic mode with `warn_only=False` | Backward kernels that differ in deterministic mode, atomics in dQ, and padding handling |
| Layout | Page sizes 16, 32, 64, 128 with random placement; pack offsets and lengths that are not multiples of the block | Blocks that follow the page or the pack instead of absolute positions |
| Parallel degree (L5) | TP and EP 1, 2, 4, 8; several ways of sharding the tokens; collectives on messages of 1 KB–256 MB; top-k ≥ 3 and 3 or more ranks | Reductions grouped by rank or shard, and ring orders that follow chunk boundaries |
| Deployment wrappers | Eager, `torch.compile` with the deployed settings, CUDA graphs at the deployed sizes; the compiled output compared with eager bitwise | Specialization for each shape, and norms and activations fused again by the compiler |
| Cross-engine (L4) | The same rows through the trainer's op and through the engine's op | Rounding points, and whether the two sides run the same kernel |

## Using the checker

```python
from bi_check import check_batch, check_modes, check_repeat

rep = check_batch(lambda x: my_gemm(x, W), rows)    # rows: [R, K] on the GPU
print(rep.summary())                                # PASS, or failing sizes

rep = check_modes({"prefill": run_prefill, "decode": run_decode,
                   "chunk-16": run_chunked(16), "chunk-7": run_chunked(7)})
```

`python tools/bi_check.py --demo` checks a few stock torch ops and writes its
results to a demo JSON under `tools/`. The measured results below were taken on
one GPU type with torch 2.5.1, unless a row says otherwise (the multi-GPU rows
ran on nodes of 8 GPUs of a second type). The details, including the hardware,
are in the case studies and in each case's `examples/<case>/REPORT.md`. To
re-check every row on your own stack, run the
[regression suite](../../../handbook/4-operations/regression.md).

| Op | Result |
|---|---|
| `torch.mm` bf16 (cuBLAS) | L1 holds. L2 fails from M=31 (4/4096 elements of a row differ) |
| matmul one row at a time | L2 holds (a control that is invariant by construction) |
| RMSNorm built from torch ops, and fp32 `log_softmax` | L2 holds at the shapes tested |
| SDPA causal | L3 fails: decode differs from prefill in ~10% of elements, and chunked prefill with an explicit mask differs from prefill in 1 element |
| SDPA causal, one sequence batched with others (GQA 8/2, T=256) | L2 fails: the same sequence changes at batch 3, 8 and 16 (17.6% of elements), but not at batch 2 |
| Fixed-block Triton attention (`tools/bi_kernels.py`) | L2 and L3 hold: prefill, decode and chunks of 7/16/64/100 are bitwise equal |
| Its backward without atomics | L1 and L2 hold for dQ/dK/dV, at batch 2–16 and with right padding to 256/320 |
| SDPA backward | Flash fails L1 (in dQ). Flash in strict deterministic mode holds L1 but fails L2, because its forward changes from batch 3. Memory-efficient in strict deterministic mode holds L1 and L2 at the shapes tested |
| Paged KV (pages of 16–128) and varlen packing with fixed-block attention | L2 and L3 hold. Blocks that follow the page or the pack offset fail |
| MLA in one form with every piece invariant | L2 and L3 hold. Explicit prefill with absorbed decode stops matching a trainer from layer 2 |
| Sparse attention with a one-config projection and a stable top-k | L2 and L3 hold. A cuBLAS projection flips the selected sets |
| GDN, recurrent and chunked | The recurrent form holds L1–L3. The chunked form holds L3 only when chunk boundaries sit at absolute positions |
| MoE fixed-slot combine at EP 1–16 (two nodes of 8 GPUs; [report](../../../examples/moe_ep/REPORT.md)) | L5 holds on one node and across two. Pre-reducing on each rank fails L5 from top-3 |
| NCCL all-reduce (8 and 16 ranks, one and two nodes) | L2 fails with the defaults and with Ring pinned. It holds with Tree on one or two nodes, and with the fallback that sums in rank order |
| Compiled block (`torch.compile`) | L2 holds. The block equals eager only when every op that rounds is opaque |
| Fingerprint probes in compiled code and CUDA graphs | The probes give the same fingerprints as opguard_bridge, and a stock graph equals eager. Stock buckets 8/16 differ from bucket 1 at o_proj, while bi buckets 1/8/16 are equal. With bi_opaque_all, compile equals eager in 3 modes. Isolated probes leave the compiled logits unchanged |
| Sampling with a per-request generator, torch.cumsum and bi_sampling | The per-request generator holds L2 but fails replay after preemption, rollback and speculative chunks. cumsum fails L2, and fails L1 at V=151,936. bi_sampling holds L1–L3 and passes a chi-square test |
| LoRA linear with a one-config base, an adapter selected per row and a single cast | L1, L2, prefill/decode and trainer vs engine all hold. Grouped cuBLAS and a gathered bmm fail L2. vLLM-style split-K atomics fail L1 in the fp32 buffer |
| Training step of a 4-layer LM over 20 steps | By default the first difference appears at step 0 (SDPA dQ). With strict mode and the fixed kernels, the runs are equal across processes |
| Embedding backward | L1 holds. L2 fails, because the kernel switches between two orders at 3072 indices |
| DP gradient all-reduce and FSDP-style reduce-scatter (emulated; flat gradient, reduce-scatter, sharded AdamW) at DP 1–16 (two nodes of 8 GPUs; [report](../../../examples/training_dp/REPORT.md)) | Bucket packing: L2 fails under Ring and holds under Tree or the ordered all-reduce. Row ownership: L2 fails for NCCL reduce-scatter and holds for the ordered version. L5 fails for any backward that batches per rank, and holds with logical shards, end to end with AdamW and an elastic resume |
| Whole model (dense, MoE, wide, Qwen2 shapes) with bi kernels | L1–L3 hold in all engine modes, and activation gradients hold L2. With stock kernels every log p differs, with KL from 1e-3 to 1.4e-2 |
| Whole model at TP 1–16 (nodes of 8 GPUs, two nodes for 16; [report](../../../examples/whole_model_tp/REPORT.md)) | With logical TP C=16 and a fixed-order vocab log_softmax, L5 holds in the forward, in the engine modes, for prefill and decode at different TP, and for activation gradients (TP 2, 4). With stock NCCL TP, every realized log-prob differs at every degree |
| vLLM 0.19.1 against model.py, Qwen2.5-0.5B ([report](../../../examples/cross_engine/REPORT.md)) | Default, eager, Triton, FlashInfer and Flex hold L1 but fail L2 (96% of logprobs differ between a prompt alone and batch 64). With VLLM_BATCH_INVARIANT=1, L2 fails on the GPU tested, because there the mode leaves the lm_head on cuBLAS (7%); it holds once the lm_head runs on vLLM's Triton GEMM. L4 holds (0/76,155) against a trainer that calls vLLM's kernels, in eager mode only |

## Reading failures

| Pattern | Likely mechanism |
|---|---|
| Passes below some M and fails at and above it, the same at every position and with every filler | A kernel or tile configuration dispatched by shape (a heuristic threshold) |
| Fails at scattered sizes | Split-K or stream-K driven by the tile count, or a split that depends on the grid size |
| Fails only at some positions | A vector width that depends on alignment, or tail handling |
| Fails only with some fillers | Statistics that depend on content: a batch-wide amax, capacity dropping, a top-k over the batch |
| Fails when an identical batch is repeated | Atomics or a race. L1 is broken, so fix that first |
| Modes differ in many elements | A different block iteration, or split-KV in decode |
| Modes differ in a few elements | A different backend chosen by the call arguments, or different handling of partially masked blocks |
| Passes in eager, fails under compile or graphs | Specialization for each shape, or fusion by the compiler; register the op as opaque |
| The compiled version is L2 but differs from eager | Fusion moved rounding points (norms, activations); make every op that rounds opaque |
| Results differ between page sizes or pack offsets, though each one is consistent with itself | The compute blocks follow the layout instead of absolute positions |
| Passes at top-2 or with 2 ranks, fails from 3 | A sum that depends on order, hidden because a two-term sum commutes |
| L1 and L2 pass, but the results change with the parallel degree | A reduction grouped by rank or shard (L5) |
| A collective with its algorithm pinned fails inside larger messages | Chunk boundaries move with the message size and change each element's ring order |
| Discrete outputs flip (selected keys, routed experts, dropped tokens) | A batch-variant score upstream, or a capacity rule; compare the sets before the values |
| Passes in steady state, fails at the first iteration or after a resume | torch DDP reduces its first iteration (and every restart) in a single bucket, which under Ring is a different order |
| Sampled tokens agree across batch sizes but change after preemption, rollback, another request's n or a speculative chunk | The noise comes from a generator's running state, not from (seed, position, purpose, sample) |
| Each side is L2/L3, but the trainer and the engine differ (L4) | A rounding point differs (packing, bias, norm form, RoPE table), or the kernel differs; compare the two rounding sequences |

## After the operator passes

1. Run the model-level check: the same sequence, run alone and inside batches
   of several sizes, must give identical log-probabilities for every token.
2. For L4, the trainer and the engine, with the same weights and zero
   staleness, must give identical log-probabilities.
3. If they do not, localize the difference with per-vector fingerprints
   (OpGuard, `OPGUARD_XOR_ROWS=1`) along the data flow, as the RL part does.
4. Measure throughput at a representative prefill M and at decode M (1–64),
   and record it with the result.
