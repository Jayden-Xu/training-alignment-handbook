# MoE on one GPU: lab report

This is the full record behind the handbook page [MoE](../../handbook/3-kernels/moe.md). The page teaches the step and what to do; this report keeps the setup, every table, every number, the limits and how the case was built. The same GEMM table backs [GEMM and quantized GEMM](../../handbook/3-kernels/gemm-and-quantized.md).

## Summary

- **One GPU:** cuBLAS expert GEMMs vary with the number of tokens per expert,
  and an atomic combine breaks even L1 (17% of output elements). One-config
  Triton expert GEMMs plus a per-token fixed-order fp32 combine pass L1 and L2,
  at 10.7 ms against 9.6 ms for the same GEMMs with an atomic combine (A40,
  4096 tokens).
- Top-2 hides combine-order bugs, and capacity-based dropping makes a token's
  output depend on its neighbours: test with top-k ≥ 3 and use dropless routing.

## How

The fix in one line: an invariant router, a stable top-k, dropless routing,
expert GEMMs with one configuration for every M, and a per-token sum in a fixed
order (top-k rank, fp32) over rows placed in fixed slots, with no atomics.

## One GPU

This applies the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)
to an operator with no ready-made invariant version: an MoE layer's routing,
expert GEMMs and combine. Code: [examples/moe_combine/moe_bi.py](../../examples/moe_combine/moe_bi.py).
Results: [results-a40.json](../../examples/moe_combine/results-a40.json).

## One GPU: setup

**Setup.**
- Layer: hidden 1024, expert intermediate 512, 64 experts, top-8, a softmax
  router with renormalized weights, SwiGLU experts, bf16.
- Hardware and software: one NVIDIA A40, torch 2.5.1+cu121, Triton 3.1.0.
- L2 test: 2 probe tokens, each placed first, middle and last in batches of
  1–512 tokens, among random, zero and ×1000 co-batched tokens.
- L1 test: 5 runs of a 256-token batch.
- Timing: a forward of 4096 tokens.

### 1. Locate the variance

There were two sources: the expert's M, and the combine.

**The expert's M.** Each expert's GEMM sees M_e, the number of tokens routed to
it. M_e changes with whatever else is in the batch, and cuBLAS picks kernels by
M. Measured on the three GEMM shapes of this layer:

| GEMM (K→N) | cuBLAS (L2) | One-config Triton (L2) |
|---|---|---|
| 1024 → 512 (expert up/gate) | fails at M = 8, 16, 64, 127, 128, 256 | passes, M = 1–2048 |
| 512 → 1024 (expert down) | fails from M = 2 | passes |
| 1024 → 64 (router) | fails at M = 31, 32, 33, 64, 1024 | passes |

**The combine.** One `index_add_` of all T×K expert outputs into the token rows
uses atomics. Even with invariant GEMMs it broke L1: 17% of output elements
changed from run to run.

### 2. Strategies tried

| Variant | GEMM | Combine | L1 | L2 | ms @ T=4096 |
|---|---|---|---|---|---|
| fused (common) | cuBLAS | one `index_add_` (atomics) | fail | fail from 2 | 22.8 |
| fused, deterministic mode | cuBLAS | the same under `use_deterministic_algorithms` | pass | fail from 2 | 22.6 |
| HF loop | cuBLAS | one `index_add_` per expert, in expert order | pass | fail from 2 | 29.3 |
| fused + Triton | one-config Triton | atomics | fail | fail from 3 | 9.6 |
| deterministic + Triton | one-config Triton | deterministic `index_add_` | pass | pass | 10.4 |
| HF loop + Triton | one-config Triton | per-expert, expert order | pass | pass | 16.7 |
| **BI (chosen)** | **one-config Triton** | **per token, top-k rank order, fp32** | **pass** | **pass** | **10.7** |
| row-chunk oracle | cuBLAS on fixed 32-row pieces, one call each | per token, fixed order | pass | pass | 349.8 |
| row-chunk bmm | the same pieces in one batched matmul | per token, fixed order | pass | fail at 33, 64, 127, 128, 256, 512 | 30.3 |

All variants agree with an fp32 reference to a mean |Δ| of about 2.4–2.6e-3
(output mean |y| 0.19). This is a correctness sanity check, not parity.

### 3. What the results say

- **Determinism is not batch invariance.** The deterministic mode fixed L1 and
  left L2 broken, because the GEMMs still changed with M_e.
- **The GEMM is the main L2 breaker in MoE.** Every combine passed L2 once the
  GEMM had one configuration for all M, except the atomic one.
- **Atomic combine breaks L1, and so L2, whatever the GEMM.**
- **Two correct combines exist once the GEMMs are invariant:** expert-ordered
  accumulation (the HF loop, or a deterministic `index_add_` over expert-sorted
  rows) and an explicit per-token fixed-order sum. We chose the explicit sum for
  two reasons:
  1. Its order (top-k rank) and precision (fp32) are stated in the code. The
     deterministic `index_add_` gets its order from the library's sort, an
     implementation detail.
  2. For L4 the trainer must use the same order and precision, and an explicit
     sum is easy to reproduce there.
- **Batched matmul over fixed pieces is not invariant.** The batched call's
  kernel changed with the number of pieces. This matches the reported `bmm`
  against `mm` router mismatch.
- **A fixed-piece loop around a closed GEMM is a valid oracle, not a product.**
  It was 33x slower here.
- **Cost.** Against the same Triton GEMM with an atomic combine, the invariant
  combine cost about 11% (10.7 vs 9.6 ms). Against the common cuBLAS version,
  the invariant layer was faster here (10.7 vs 22.8 ms), because the untuned
  one-config Triton GEMM beat cuBLAS on these small shapes (0.075 vs 0.112 ms
  for 4096×1024×512). Do not generalize that: check on your own shapes.

### Notes from the GEMM page

- `torch.mm` in bf16 (cuBLAS) is deterministic run to run but not
  batch-invariant: on an A40 a row changes once the batch reaches 31 (4 of 4096
  elements). cuBLAS was also variant on all three GEMM shapes of the MoE layer.
- **Fix:** one tile configuration for every M, no split-K or stream-K, ascending
  K into fp32, the same epilogue. The one-config Triton GEMM
  (`mm_one_config` in [tools/bi_kernels.py](../../tools/bi_kernels.py)) is L2 at
  M = 1–2048 on those three shapes.
- **Cost:** lower utilization at small M. On the MoE layer's small shapes the
  untuned one-config GEMM was faster than cuBLAS (0.075 vs 0.112 ms for
  4096×1024×512); do not generalize that.

At 4096×1024×512 the untuned one-config Triton GEMM took 0.075 ms against
0.112 ms for cuBLAS. The one-config GEMM has 64×64×32 tiles, 4 warps and 3
stages, and no fused epilogue.

## Limits

- Forward only, one GPU, no expert parallelism. Under EP, place returned rows
  into fixed slots before the per-token sum.
- Dropless routing. Capacity-based dropping would add a batch dependence this
  test does not cover.
- No tie-break rule was needed in top-k: it was not exercised beyond zero-filled
  co-batched tokens. Add (score, expert id) ordering for exact ties.
- L4 (matching a trainer's MoE) was not tested here.
- The Triton GEMM is untuned (64×64×32 tiles, 4 warps, 3 stages) and has no fused
  epilogue.
- On 8x A16 (sm86, driver 535.288.01, torch 2.5.1+cu121) the cuBLAS 1024→512 variance did not
  reproduce at the regression suite's quick sizes (2026-11-19, see
  [regression](../../handbook/4-operations/regression.md#baselines-recorded-here)). The failure is
  claimed for the A40 only. Full sizes were not run on the A16.

- The GEMM evidence is on sm86 (A40, A16) only. Tuning, large shapes, fused
  epilogues, cuBLASLt pinning and Hopper are open
  ([evidence table](../../skills/batch-invariant-kernel/references/operator-recipes.md#evidence-behind-each-recipe)).
- The one-config Triton GEMM is untuned.

## How it was built

- The agent's effort took two runs and no human edits.
  1. The first run located the variance, showed the row-chunk loop correct but
     slow, and found the batched-matmul version variant.
  2. The second added the one-config Triton GEMM and the attribution variants.

The one-config GEMM was added to [tools/bi_kernels.py](../../tools/bi_kernels.py) in the second run above.
