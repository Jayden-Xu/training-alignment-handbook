# Kernel cases: lab report

This is the full record behind the handbook page [Batch-invariant kernels](../../handbook/3-kernels/README.md). The page teaches the step and what to do; this report keeps the setup, every table, every number, the limits and how the case was built. It also backs the [attention](../../handbook/3-kernels/attention.md) forward, the FP8 part of [GEMM and quantized GEMM](../../handbook/3-kernels/gemm-and-quantized.md) and the plain linear-attention comparison of [linear attention](../../handbook/3-kernels/linear-attention.md).

Code: [kernel_cases.py](kernel_cases.py) and the reference kernels in [tools/bi_kernels.py](../../tools/bi_kernels.py). Results: [results-a40.json](results-a40.json).
Setup: one NVIDIA A40, torch 2.5.1+cu121, Triton 3.1.0, bf16 unless noted.

## Stock ops: determinism is not batch invariance

Measured on an A40 with torch 2.5.1 (`tools/bi_check.py --demo`):

| Op | Run-to-run (L1) | Batch (L2) / mode (L3) |
|---|---|---|
| `torch.mm` bf16 (cuBLAS) | yes | no: a row changes once the batch reaches 31 (4 of 4096 elements) |
| Row-by-row matmul (control) | yes | L2 yes |
| RMSNorm from torch ops, fp32 `log_softmax` | yes | L2 yes at these shapes |
| SDPA causal attention | yes | L3 no: decode vs prefill ≈10% of elements; chunked with an explicit mask vs prefill, 1 element |

The run is recorded in [tools/demo-a40.json](../../tools/demo-a40.json).

The same demo as quoted on the GEMM page (GEMM rows only, L2 column):


| Op | Run-to-run (L1) | Batch (L2) |
|---|---|---|
| `torch.mm` bf16 (cuBLAS) | yes | no: a row changes once the batch reaches 31 (4 of 4096 elements) |
| Row-by-row matmul (control) | yes | yes |

Cost of parity modes, as reported elsewhere (not measured here): the reported costs range from about 25% per RL step (IsoExec) to 2–3x trainer throughput (recurrent GDN forward); see [references](../../handbook/references.md). Record throughput at prefill M and at decode M=1–64 next to the baseline.

Where each kernel page was measured:

| Page | Covers | Measured on |
|---|---|---|
| [Attention](../../handbook/3-kernels/attention.md) | Forward in every mode (prefill, decode, chunked prefill), atomic-free backward, paged KV, varlen packing, `torch.compile` and CUDA graphs | A40 |
| [MLA and sparse attention](../../handbook/3-kernels/mla-and-sparse.md) | Absorbed against explicit MLA; DSA-like top-k selection | A40 |
| [Linear attention](../../handbook/3-kernels/linear-attention.md) | Chunked against recurrent GDN, aligned chunks, cost by sequence length | A40 |
| [MoE](../../handbook/3-kernels/moe.md) | Expert GEMMs and combine on one GPU; expert parallelism at EP 1–16 | A40; 2x8 A16 |
| [GEMM and quantized GEMM](../../handbook/3-kernels/gemm-and-quantized.md) | cuBLAS against a one-config GEMM; FP8 activation scales (emulated) | A40 |
| [Collectives](../../handbook/3-kernels/collectives.md) | NCCL algorithms, reduce-scatter, fallbacks, logical TP size | 2x8 A16 |

## Forward: prefill, decode and chunked prefill (L2, L3)

**What made it invariant.** Every query visits the KV cache in 64-token blocks
from absolute position 0, with an online-softmax update per block and no
split-KV. Prefill, chunked prefill and decode run the same kernel with the
query's absolute offset, so a position's arithmetic is the same in every mode.
Blocks past a query's position are fully masked and are exact no-ops: the
rescale factor is exactly 1 and the added probability exactly 0.

Results: [results-a40.json](../../examples/kernel_cases/results-a40.json).
Setup: one NVIDIA A40, torch 2.5.1+cu121, Triton 3.1.0, bf16 unless noted.

Setup: 8 query heads, 2 KV heads (GQA), head dim 128, a 256-token sequence. L3
compares prefill against per-token decode, and against chunked prefill with
chunk sizes 7, 16, 64 and 100, where each chunk attends to the cached keys plus
itself. L2 places the sequence first or last among 2–16 random sequences.

| | L3: modes vs prefill | L2: batch of sequences | Prefill B=4, T=2048 | Decode B=16, KV=2048 |
|---|---|---|---|---|
| torch SDPA | fail: decode and chunks 7/16/64 differ in 10.2% of elements, chunk 100 in 12.6% | fail at batch 3, 8, 16 (17.6% of elements); pass at 2 | 0.38 ms | 0.23 ms* |
| fixed-block Triton | **pass**: all six modes bitwise equal | **pass** | 0.62 ms | 0.19 ms |

\* SDPA has no native GQA path here, so its KV was expanded 4x in memory. That
favours the Triton kernel in the decode timing.

**What SDPA shows.** Its result for one sequence changed with the batch size
(between 2 and 3). This is consistent with a kernel that decides how to split
work from batch × heads. Decode and chunked prefill differed from prefill in
every case.

Both kernels have the same maximum error against fp32 attention (9.2e-3). The
Triton kernel is untuned: 1.6x slower in prefill here.

### Other hardware

- On 8x A16 (sm86, driver 535.288.01, torch 2.5.1+cu121) the SDPA L2 failure of section 1 did not
  reproduce at the regression suite's quick sizes (2026-11-19, see
  [regression](../../handbook/4-operations/regression.md#baselines-recorded-here)). The failure is
  claimed for the A40 only. Full sizes were not run on the A16.

### Limits

**Not covered:** sliding windows, long-context decode with split-KV at a fixed
split size.

## FP8 activation scales (L2)

- **Quantized:** a batch-wide dynamic FP8 activation scale breaks L2; per-token
  and per-1x128-block scales keep it (emulated numerics).

Setup: one NVIDIA A40, torch 2.5.1+cu121, Triton 3.1.0, bf16 unless noted.

The A40 has no FP8 tensor-core path, so this emulates the numerics. Activations
are quantized to `float8_e4m3fn` with a scale, dequantized to bf16 and fed to the
one-config GEMM. Weights are quantized statically per 128-block. Only the scale
granularity varies.

| Activation scale | L2 | Fails with |
|---|---|---|
| per tensor, dynamic (amax over the batch) | **fail** from 2 rows (98 of 164 cases): 911–988 of 1024 elements differ in the 50 recorded failures | random or large co-batched rows |
| per token | pass | — |
| per 1x128 block | pass | — |

A batch-wide amax couples every row to its neighbours. That is invisible with
zero fillers, so test with large ones. Per-token and per-block scales keep rows
independent. Real FP8 GEMMs add another requirement, a fixed promotion interval
and scale order inside the GEMM, which this emulation does not exercise. It
needs sm89+ hardware.

### Limits

- FP8 is emulated: the A40 has no FP8 tensor-core path. A real FP8 GEMM's
  promotion interval needs sm89+; NVFP4 needs sm100.

### How it was built

The FP8 checks are part of [examples/kernel_cases/kernel_cases.py](../../examples/kernel_cases/kernel_cases.py),
written by an agent through the
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md). The
one-config GEMM was added to [tools/bi_kernels.py](../../tools/bi_kernels.py) in
the second run of the MoE case ([MoE, how it was built](../moe_combine/REPORT.md#how-it-was-built)).

## Plain linear attention: chunked vs recurrent (an algorithmic difference)

Setup: one NVIDIA A40, torch 2.5.1+cu121, Triton 3.1.0, bf16 unless noted.

Causal linear attention in fp32: 4 heads, dim 64, 128 tokens.

| Comparison | Elements that differ |
|---|---|
| chunked (16) vs recurrent | 84% |
| chunked (64) vs recurrent | 88% |

Maximum error against fp64: recurrent 8.1e-8, chunked 5.4e-8.

Both forms are equally accurate and bitwise different. This is not a kernel
defect, so tuning cannot remove it: the two algorithms sum in different orders.
To reach L3 or L4, use one form for the forward in every mode. A recurrent
kernel with one program per sequence is naturally batch-invariant. The reported
cost of recurrent-only forward on GDN models is 2–3x trainer throughput
(reported by Wang et al., "Closing the Training–Inference Mismatch for Linear Attention in RL"; see [references](../../handbook/references.md#rl-traininference-alignment); not measured here).
