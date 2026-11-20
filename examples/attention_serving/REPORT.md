# Attention serving layers: lab report

This is the full record behind the handbook page [Softmax attention](../../handbook/3-kernels/attention.md). The page teaches the step and what to do; this report keeps the setup, every table, every number, the limits and how the case was built.

## Serving layers: paged KV, varlen packing, compile and CUDA graphs

The [forward](../kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3) and [backward](../attention_backward/REPORT.md#backward-dq-dk-dv) cases made
attention invariant on contiguous, padded batches. A serving engine adds three
layers around the kernel:
- a paged KV cache;
- packed variable-length batches (`cu_seqlens`);
- `torch.compile` and CUDA graphs.

Each layer can bring back a dependence on the batch even when the kernel itself
is invariant. This case checks each one.

### Where the serving-layer variance comes from

- **Paged KV.** A kernel that walks one page per loop iteration makes the
  online-softmax block the page, whose size is a deployment setting. The
  reduction tree then changes with the page size.
- **Varlen packing.** A kernel that lays its block grid over the whole pack, as
  a document mask over one long sequence does, puts a sequence's block
  boundaries wherever its neighbours' lengths leave them.
- **Compile and graphs.**
  - Inductor fuses the torch ops between the kernels, and can move rounding points.
  - Dynamo specializes graphs per shape.
  - A CUDA graph fixes the kernels of one bucket, so a padded batch runs at the
    bucket's size, not its real size.

### What was done

**Kernels.** Queries are packed as `[NQ, Hq, D]` with `cu_q`. Each sequence also
carries `pos0`, the absolute position of its first query (its cached length).
Its KV length is `pos0 + q_len`. Prefill, decode, chunked prefill and mixed
batches are all one launch of the same kernel.

| Kernel | K/V layout | Compute blocks |
|---|---|---|
| `attention_paged` | pages `[num_pages, P, Hk, D]` and a block table | 64 tokens from absolute position 0, whatever P is. For P < 64, a block gathers 64/P pages through a per-row page lookup. For P ≥ 64, it reads a slice of one page. |
| `attention_varlen` | packed `[NK, Hk, D]` and `cu_k` | 64 tokens from the sequence's own position 0 |
| `attention_varlen_train` | packed, self-attention | backward as in the [backward case](../attention_backward/REPORT.md#backward-dq-dk-dv), with query and key blocks aligned to each sequence's start; no atomics |
| contrast: `BN=P` | paged | the compute block equals the page size |
| contrast: `align_pack=True` | packed | blocks on the pack's 64-token grid, so a sequence's first block also covers masked rows of its neighbour |

- **Same per-row arithmetic.** The kernels keep the reference's tile shapes
  (64 × 64, D = 128), warps, dot order, `exp2` path and masking. Only the
  addresses change.
- **Unwritten slots are never read.** Unwritten cache slots hold NaN. Loads are
  masked by each sequence's KV length, so the NaN never enters the result.
  Decode and chunked prefill write the current chunk's K/V into the cache
  before the kernel runs.

**Block for the wrappers.** A pre-norm transformer block on 100 tokens per
sequence, hidden 1024, MLP intermediate 2048:
- h = x + O-proj(attention(QKV(RMSNorm(x))));
- out = h + down(SiLU(gate) · up(RMSNorm(h))).

RMSNorm and SiLU are torch ops. Five variants:

| Variant | GEMMs | Attention | RMSNorm, SiLU·mul |
|---|---|---|---|
| `bi` | `mm_one_config` | `attention_fixed_blocks` | torch ops |
| `bi_opaque_kernels` | the same, as `torch.library.custom_op` | the same, as a custom op | torch ops |
| `bi_opaque_all` | custom op | custom op | custom ops wrapping the same torch ops |
| `cublas` | `torch.mm` | `attention_fixed_blocks` | torch ops |
| `stock` | `torch.mm` | SDPA (KV expanded to 8 heads) | torch ops |

## Results

Taken through the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
- Code: [examples/attention_serving/serving_kernels.py](../../examples/attention_serving/serving_kernels.py)
  (paged and varlen kernels, varlen backward, custom-op registration) and
  [serving_cases.py](../../examples/attention_serving/serving_cases.py) (the checks).
  Both build on [tools/bi_kernels.py](../../tools/bi_kernels.py) without changing it.
- Results: [results-a40.json](../../examples/attention_serving/results-a40.json).
- Setup: one NVIDIA A40 (shared with other jobs), torch 2.5.1+cu121, Triton 3.1.0,
  bf16, causal, 8 query heads and 2 KV heads (GQA), head dim 128.

### Paged KV

- **L3.** One 300-token sequence; pages randomly permuted. Paged prefill,
  per-token decode, and chunked prefill (7, 16, 100) at `num_stages` 2 and 1,
  all against the contiguous `attention_fixed_blocks` prefill.
- **L2.** Mixed batches of 11 sequences, lengths 1, 2, 37, 64, 65, 129, 300, 511,
  1000, 77 and 200, in random order. Five batches per page size: all decode,
  all prefill, and three mixes of prefill, decode and chunk rows. Every
  sequence's rows were compared with its contiguous prefill, at both stage
  counts.

| Page size P | L3 (10 paged modes) | L2 mixed batches | Contrast BN = P: elements differing from BN = 64 |
|---|---|---|---|
| 16 | **pass** | **pass** (110/110) | 65,800 of 307,200 (21.4%) |
| 32 | **pass** | **pass** (110/110) | 45,505 (14.8%) |
| 64 | **pass** | **pass** (110/110) | 0 (the same kernel) |
| 128 | **pass** | **pass** (110/110) | 40,266 (13.1%) |

- **The contrast is self-consistent.** With BN = P, decode equals prefill at
  every P. Every pair of page sizes still disagrees, in 40k–77k elements. The
  largest difference from the BN = 64 result was 0.0078.
- **Large shapes too.** At the timing shapes, every paged and varlen output was
  bitwise equal to the contiguous kernel (prefill 4 × 2048, decode 16 × 2048).

**The cost of the gather** (ms; median of 3 runs of 20–50 calls). The
contiguous tools kernel and the varlen kernel run at `num_stages=2`.

| | contiguous (tools) | varlen packed | P = 16 | P = 32 | P = 64 | P = 128 |
|---|---|---|---|---|---|---|
| prefill B=4, T=2048, `num_stages=2` | 0.617 | 0.631 | 0.848 | 0.846 | 0.684 | 0.694 |
| prefill, `num_stages=1` | — | — | 0.643 | 0.641 | 0.568 | 0.611 |
| decode B=16, KV=2048, `num_stages=2` | 0.189 | 0.191 | 0.269 | 0.266 | 0.206 | 0.210 |
| decode, `num_stages=1` | — | — | 0.195 | 0.192 | 0.160 | 0.198 |

- **The stage count changed the time, not the bits.** Every cell above was
  bitwise equal to the contiguous kernel.
- **Two-level gather (P < 64).** At `num_stages=2` it cost 37% in prefill and
  42% in decode over the contiguous kernel. At `num_stages=1` it cost 4% and 3%.
- **Same stage count, across page sizes.** At `num_stages=1`, P = 16 and 32
  cost 13% (prefill) and 20–22% (decode) more than P = 64.
- **Not timed:** the contiguous kernel at `num_stages=1`. Paged P = 64 at
  `num_stages=1` (0.568 ms) beat the contiguous kernel at 2 (0.617 ms), so the
  contiguous kernel may also be faster at 1.

These numbers are from a separate run with the GPU otherwise idle. During the
full run another job joined the GPU (`paged_timing_run5_overlapped` in the JSON
records two compute processes and 99% utilization). The contiguous decode there
took 0.38 ms instead of 0.19 ms.

### Varlen packing

| Check | Result |
|---|---|
| A sequence (77, 200 or 300 tokens) alone vs the contiguous tools kernel | bitwise equal |
| Alone vs packed: 10 layouts per probe, offsets 0–511, neighbours of 1–511 tokens before and after | **pass**, 30 of 30 |
| A chunk (rows 150–209 of a 300-token sequence) packed with decode, prefill and chunk rows of other sequences | **pass**, 4 of 4 |
| Contrast, pack-aligned blocks | equal only when the offset is a multiple of 64 (0, 64, 128, 192). Otherwise 0.3–20.5% of elements differ, e.g. 300 tokens at offset 37: 63,070 of 307,200; at offset 1: 7,751 |
| Forward storing log-sum-exp vs plain | bitwise equal |
| Backward alone vs the tools `attention_fixed_blocks_train` | o, dQ, dK, dV bitwise equal |
| Backward alone vs packed (the same 30 layouts) | **pass**: o, dQ, dK, dV |
| Backward L1: 5 runs of a 6-sequence pack (2,246 tokens) | **pass** |

**Cost** at 4 × 2048 tokens: forward 0.630 vs 0.620 ms for the contiguous tools
kernel (+2%); forward + backward 2.64 vs 2.55 ms (+4%).

### Deployment wrappers

**Eager.** `check_batch` over batch sizes 1–256 (the default sizes up to 256),
first, middle and last position, and random, zero and ×1000 fillers. Two probe
sequences; 308 comparisons per variant.

| Variant | L2 |
|---|---|
| `bi`, `bi_opaque_all` | **pass**, and equal to each other bitwise |
| `cublas` | fail from batch 2: up to 50,133 of 102,400 output elements (49%) |
| `stock` | fail from batch 2: up to 50,154 elements |

**`torch.compile`.** The same matrix, plus a check against eager. Each graph
compiled and autotuned afresh (FX-graph and autotune caches off), and Dynamo's
cache limit raised from 8 to 64. Without that raise, Dynamo would silently run
eager after 8 recompiles.

| Variant | `dynamic` | Graphs for 18 sizes | L2 under compile | Compiled = eager |
|---|---|---|---|---|
| `bi` | None (default), True, False | 18 each | **pass** | no: 66,251 and 66,438 of 102,400 elements (65%) for the two probes, at every size |
| `bi_opaque_kernels` | None, True | 2 | **pass** | no: 66,251 (65%) |
| `bi_opaque_kernels` | False | 18 | **pass** | no: 66,251 (65%) |
| `bi_opaque_all` | None, True | 2 | **pass** | **yes**, at every size |
| `bi_opaque_all` | False | 18 | **pass** | **yes** |
| `bi`, `mode="reduce-overhead"` | None | 18 | **pass** | no: 66,251 (65%) |
| `bi_opaque_all`, `mode="reduce-overhead"` | None | 2 | **pass** | **yes** |

- No graph break in any variant (`fullgraph=True` compiled).
- Every variant gave one output for each probe across all 18 sizes, so its
  separate graphs agreed.

**Where compiled and eager part.** Each op was compiled alone and compared with
eager on the block's own inputs:

| Op | Elements differing |
|---|---|
| RMSNorm | 25,337 of 102,400 (25%) |
| residual add feeding RMSNorm | 34,639 of 102,400 (34%) |
| residual add alone | 0 |
| SiLU(gate) · up | 56,105 of 204,800 (27%) |
| `mm_one_config` alone | 0 |
| `attention_fixed_blocks` alone | 0 |

**Why `bi` recompiles at every size.** Dynamo's guard-failure log shows the
batch dimension guarded to its exact value after each recompile: "size mismatch
at index 0. expected 2, actual 3". So the graph never became dynamic, even with
`dynamic=True`. With the kernels behind custom ops, the second compile was
dynamic: one graph for batch 1 (Dynamo specializes 0 and 1) and one for all
other sizes. The likely cause is that tracing the launch of a user-defined
Triton kernel specializes the size passed to it (not confirmed).

**CUDA graphs.** One graph per bucket (1, 2, 4, 8, 16, 32 and 64 sequences). Real
batches of 1, 2, 3, 5, 8, 9, 16, 17, 31, 33 and 64 sequences were zero-padded to
the bucket, with the probe first or last.

| Block | Probe output across buckets | Graph replay vs eager, same padded shape |
|---|---|---|
| `bi` | **one output in all 7 buckets**, equal to eager batch 1 | bitwise equal |
| `cublas` | 6 different outputs over 7 buckets (16 and 64 agree); up to 50,133 elements differ from batch 1 | bitwise equal |
| `stock` | 6 different outputs over 7 buckets (16 and 64 agree) | bitwise equal |

**Timing** (ms per block call, 100 tokens per sequence; the GPU showed idle
before and only this process after):

| | 1 seq | 8 seqs | 64 seqs |
|---|---|---|---|
| eager `bi` | 0.629 | 0.461 | 3.021 |
| eager `bi_opaque_all` | 0.788 | 0.787 | 3.079 |
| eager `cublas` | 0.370 | 0.374 | 2.367 |
| eager `stock` | 0.360 | 0.377 | 2.469 |
| compiled `bi` | 0.158 | 0.261 | 2.217 |
| compiled `bi_opaque_all` | 0.816 | 0.829 | 3.099 |
| compiled `bi`, `reduce-overhead` | 0.161 | 0.334 | 2.291 |
| compiled `bi_opaque_all`, `reduce-overhead` | 0.199 | 0.416 | 3.143 |
| CUDA graph `bi` | 0.128 | 0.340 | 3.014 |
| CUDA graph `cublas` | 0.118 | 0.343 | 2.337 |
| CUDA graph `stock` | 0.121 | 0.347 | 2.436 |

- **Small batches are host-bound.** At 1 and 8 sequences, eager time is
  launch overhead. It moved between runs: eager `cublas` at 1 sequence took
  0.63 ms in an earlier run (not in the saved results), likely from CPU load by other jobs. Graphs remove
  it: the BI block is within 8% of cuBLAS at 1 sequence and 1% at 8.
- **At 64 sequences (6,400 tokens), the untuned one-config GEMM shows.** The BI
  block costs 1.29x the cuBLAS block, both graphed.
- **Custom ops add host time.** Compiled `bi_opaque_all` took 0.82 ms at 1
  sequence, against 0.16 ms for compiled `bi`. The likely cost is the
  custom-op dispatch path on the host. Under `reduce-overhead`, a CUDA graph
  replays the ops without host work: 0.20 ms. At 64 sequences it keeps none of
  the fused elementwise ops and runs at eager speed (3.1 ms).
- **Compiled `bi` beat eager `bi` at 64 sequences** (2.22 vs 3.02 ms). The
  reason was not investigated.

### What the serving-layer results show

1. **A paged layout costs no invariance if compute blocks ignore the page
   size.** With 64-token blocks on absolute positions, the result was the same
   bits at every page size, in every mode and in mixed batches. That held with
   random page placement and NaN in unwritten slots. Making the block follow the
   page instead changed 13–21% of the elements, and every page size gave a
   different answer. Each answer was self-consistent across modes, so an L3 test
   at a single page size would not catch it.
2. **Varlen packing is invariant when blocks start at the sequence.** A
   pack-aligned grid failed exactly when the sequence's offset was not a
   multiple of 64. A test with neighbours that happen to sum to multiples of 64
   would pass by luck. The atomic-free backward carried over unchanged, and
   gradients were identical alone and packed.
3. **`torch.compile` kept L2 here but did not keep eager's bits.**
   - Compiled alone, RMSNorm, the add feeding it, and SiLU · up each differed
     from eager in 25–34% of their elements. The BI GEMM, the attention and a
     lone residual add came through compile bit-exact.
   - The likely cause is that inductor fuses these torch ops and moves their
     rounding points: it keeps fused intermediates in fp32 and uses its own
     reduction order for the mean. This was not confirmed from the generated code.
   - Making only the BI kernels opaque fixed the recompiles (18 graphs down to
     2) but not the bits. Making every op that rounds opaque gave compiled =
     eager at every size, with and without CUDA graphs (`reduce-overhead`).
   - That configuration keeps eager's rounding points, which an engine needs
     before it can match an eager trainer. L4 itself was not measured.
4. **CUDA graphs are invariant only with invariant kernels.** With BI kernels,
   padding to any bucket gave the same bits. With cuBLAS, the probe's output
   depended on the bucket. The padded batch runs at the bucket's size, and
   cuBLAS likely picks its kernel by that size, as in the eager check. A graph
   replayed the same bits as eager at the same shape, so the variance comes
   from the kernels the bucket runs, not from capture.

## Limits

- One GPU (A40) and one head configuration (GQA 8/2, D = 128).
- Paged L3 at one length (300); mixed batches up to 1,000 tokens per sequence.
- Not tested:
  - page sizes other than 16–128;
  - fixed-size split-KV for long decode;
  - sliding windows;
  - a paged backward;
  - FP8 KV caches;
  - writing the cache from a fused kernel (here, torch indexing).
- The pack-aligned contrast models a document-mask kernel over the pack. No
  library kernel (e.g. FlexAttention) was measured.
- Compile was checked at one block shape with torch 2.5.1 defaults. Inductor's
  choices may differ at other shapes or versions.
- `reduce-overhead` was checked only with `dynamic=None`.
- Not tested under compile: the paged and varlen kernels, and any backward.
- L4 (against a trainer) was not measured.
- Untuned kernels. The P < 64 gather prints a Triton 3.1 pipeliner diagnostic
  ("operation scheduled before its operands") at `num_stages=2`. It still
  compiles and passes, but runs slower than `num_stages=1`.
- The A40 was shared. Timing waited for the GPU to show idle and recorded
  nvidia-smi before and after (in the JSON); timings may still overlap other jobs.

## How it was built

An agent wrote the kernels and the checks; there were no human edits. The first
run passed every paged and varlen check. Later runs added measurements, not
fixes:
- per-op compiled-vs-eager checks and Dynamo's recompile reasons, to locate the
  compile difference;
- the `num_stages=1` variants, after the pipeliner diagnostic;
- fresh compiles (caches off) for the final run.

Runs on the A40:
1. paged and varlen;
2. eager and CUDA graphs;
3. the compile sweep;
4. compile diagnostics and timing;
5. the full run from an empty inductor cache, which produced the results file;
6. and 7. paged timing alone. Another job joined the GPU during run 6, so it
   was discarded. Run 7 was clean and replaced the overlapped numbers.

One more compile run was killed by the agent's own mistake: a `pkill` pattern
also matched its shell.

One environment note: when Dynamo loads its backends, importing `onnxruntime`
from `~/.local` prints a numpy 1.x/2.0 ABI traceback. Dynamo catches the
resulting `ImportError`, and compile proceeds.
