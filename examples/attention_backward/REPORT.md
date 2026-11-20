# Attention backward: lab report

This is the full record behind the handbook page [Softmax attention](../../handbook/3-kernels/attention.md). The page teaches the step and what to do; this report keeps the setup, every table, every number, the limits and how the case was built.

## Backward (dQ, dK, dV)

The forward cases made attention invariant across batch and mode. Training also
needs the backward to be invariant: the same sequence must get the same
gradients whether it runs alone, in a larger batch, or padded.

### Where the backward variance comes from

In the forward, one program owns each output row. In the backward it does not:
- a key's dK and dV sum over every later query, and over every query head in
  its GQA group;
- a query's dQ sums over every earlier key.

Common kernels parallelize over key blocks and add each block's share of dQ
with atomics. The order of those additions depends on timing, so L1 breaks. A
kernel that also chooses its work split from batch × heads breaks L2 as well.

### The kernel: two passes, no atomics

| Kernel | One program per | Walks, in a fixed order | Writes |
|---|---|---|---|
| Delta | sequence, query head, 64 rows | the head dimension (`rowsum(dO * O)` in fp32) | Delta |
| dK/dV | sequence, KV head, 64 keys | the GQA group's query heads, then the query blocks from the first one that sees these keys | dK, dV once |
| dQ | sequence, query head, 64 queries | key blocks from position 0, as the forward does | dQ once |

- **No atomics.** Each program accumulates in fp32 registers and writes its
  output once.
- **GQA inside the program.** The sum over a group's query heads happens inside
  the dK/dV program, in fp32.
- **Log-sum-exp from the forward.** The forward stores each row's log-sum-exp
  (log2 units) as a compile-time variant. Its output was checked bitwise equal
  to the plain forward at T = 200, 256 and 1000. The decode and chunked-prefill
  path therefore keeps exactly the arithmetic verified in the
  [forward case](../kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3).
- **Batch only adds programs.** A program's loop depends only on its own
  sequence, its head, its block and T. A larger batch never changes what any
  program computes.

**The price.** The dQ pass recomputes the scores P and dP that the dK/dV pass
also computes.

### Why right padding is exact here

- Causality keeps padding after a sequence out of its tokens' forward.
- In the backward, padding rows carry zero output gradient. So dP = 0,
  Delta = 0 and dS = 0 for them, and their contributions to dK and dV are exact
  zeros. Extra query blocks add zeros as well.
- One theoretical exception: adding +0.0 to an accumulator that is exactly −0.0
  gives +0.0. It was not observed.
- Left padding shifts absolute positions and changes the block grouping, so it
  is not covered by this argument. Neither is varlen packing.

## Results

Taken through the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
- Code: [examples/attention_backward/attn_bwd.py](../../examples/attention_backward/attn_bwd.py)
  and the kernels in [tools/bi_kernels.py](../../tools/bi_kernels.py)
  (`attention_fixed_blocks_train`).
- Results: [results-a40.json](../../examples/attention_backward/results-a40.json).
- Setup: one NVIDIA A40, torch 2.5.1+cu121, Triton 3.1.0, bf16, causal, 8 query
  heads and 2 KV heads (GQA), head dim 128.

**Invariance.**
- L1: the same batch (B=4, T=1024) run 5 times.
- L2 batch: one 256-token sequence alone, vs first or last among 2, 3, 8 or 16
  sequences.
- L2 padded: a 200-token sequence alone, vs right-padded to 256 or 320 inside a
  batch of 3–4, with zero output gradient on the padding (as a loss mask gives).

| | L1 | L2 batch | L2 padded | fwd / fwd+bwd, B=4, T=2048 |
|---|---|---|---|---|
| **fixed-block Triton** | **pass** | **pass** | **pass** | 0.63 / 2.94 ms |
| SDPA flash | fail: dQ, ~150 of 4.2M elements per run | fail: batch 2 only in dQ (1–3 elements); from batch 3, O 17.2%, dQ 11.3%, dK 17.3%, dV 0.06% | fail: O 14.3%, dQ 10.0%, dK 16.7%, dV 0.06% | 0.47 / 2.13 ms |
| SDPA flash, strict deterministic | pass | fail from batch 3 (the forward already differs) | fail (same) | 0.47 / 5.20 ms |
| SDPA memory-efficient | fail: dQ, ~100 elements per run | fail at batch 8 and 16, dQ only (13 and 45 elements) | pass | 0.74 / 10.73 ms |
| SDPA memory-efficient, strict deterministic | pass | pass | pass | 0.74 / 9.42 ms |

- "Strict deterministic" means `torch.use_deterministic_algorithms(True, warn_only=False)`.
- SDPA ran on K/V expanded to the query heads, as a model's `repeat_kv` does.
  Its invariance was checked on the gradients of the expanded K/V. The timing
  includes the expansion.

**Accuracy** against an fp64 reference (B=2, T=500, GQA 8/2). The table gives
the relative Frobenius error, with the maximum absolute error in parentheses.

| | O | dQ | dK | dV |
|---|---|---|---|---|
| fixed-block Triton | 1.96e-3 (7.2e-3) | 2.51e-3 (1.5e-2) | 2.49e-3 (1.6e-2) | 2.29e-3 (2.0e-2) |
| SDPA flash | 1.96e-3 (7.2e-3) | 2.51e-3 (1.5e-2) | 3.02e-3 (2.1e-2) | 2.84e-3 (2.5e-2) |
| SDPA memory-efficient | 1.97e-3 (7.2e-3) | 2.97e-3 (1.3e-2) | 3.40e-3 (2.3e-2) | 2.84e-3 (2.5e-2) |

SDPA's dK and dV pass through the expansion's backward, which sums
per-head bf16 gradients. The Triton kernel sums the group in fp32. That likely
explains its slightly smaller dK and dV error.

### What the backward results show

1. **`warn_only=True` does not make SDPA's backward deterministic.** With it,
   torch only warns that flash and memory-efficient attention "default to a
   non-deterministic algorithm", then runs that algorithm. A first run measured
   dQ still differing run to run. Use `warn_only=False`. It raises an error on
   any op without a deterministic implementation, which is what a parity mode
   wants.
2. **Deterministic is not batch-invariant, again.** Flash in strict
   deterministic mode passed L1 and failed L2: its forward output for a sequence
   already changes between batch 2 and 3. That is consistent with a work split
   chosen from batch × heads (not confirmed). A backward cannot be more invariant
   than the forward it differentiates.
3. **A configure-only option exists on this GPU.** SDPA memory-efficient in
   strict deterministic mode passed L1 and both L2 tests at these shapes
   (strategy 1 in the skill). It costs 4.4x default flash for forward plus
   backward, and 3.2x the Triton kernel. It was not tested across modes (L3).
   An inference engine does not run it, so it does not help cross-engine
   agreement (L4).
4. **The atomic-free Triton backward** reached L1 and L2, including right
   padding. It costs 1.38x default flash and is 1.8x faster than deterministic
   flash. Its forward is the one already verified across prefill, decode and
   chunked prefill.

## Limits

- One GPU (A40) and one head configuration (GQA 8/2, D=128). Lengths up to 1024
  for L1 and 2048 for timing.
- Not tested:
  - packed varlen (`cu_seqlens`);
  - left padding;
  - sliding windows;
  - dropout;
  - `torch.compile`;
  - CUDA graphs;
  - agreement with an inference engine's forward (L4).
- Untuned: 64 × 64 blocks, 8 warps for dK/dV and 4 for dQ.

## How it was built

An agent wrote the three kernels and the test in one pass, with no human edits.
The first run on the A40 passed every check for the Triton kernel. One change
to the test harness followed: strict deterministic mode for SDPA (finding 1).
