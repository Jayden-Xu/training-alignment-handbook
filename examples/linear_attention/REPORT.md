# Linear attention (GDN): lab report

This is the full record behind the handbook page [Linear attention](../../handbook/3-kernels/linear-attention.md). The page teaches the step and what to do; this report keeps the setup, every table, every number, the limits and how the case was built.

## Summary

- Chunked and recurrent forms are equally accurate and bitwise different (84–88%
  of elements in plain causal linear attention): an algorithmic difference, not
  a kernel defect.
- Both forms are batch-invariant by construction. Recurrent is mode-invariant
  for free; chunked is mode-invariant only with chunk boundaries at absolute
  multiples of C (`AlignedChunked`).
- The chunked–recurrent gap that matters in practice is mostly matmul
  precision: bf16-operand chunked differs from recurrent in 55.5% of bf16
  outputs, fp32 chunked in 0.03%.
- **Cost:** the recurrent forward is 0.9–1.5x the chunked kernel at T ≤ 4096
  and 2.5–3.6x at T = 8192 (A40). The
  [recommendation](#recommendation-with-the-measured-cost-of-each-option)
  weighs the options.

## How

Linear attention layers keep a fixed-size state per sequence and head. Training
and prefill usually run a chunked form, and decode runs a recurrent step.
[Kernel cases, section 3](#plain-linear-attention-chunked-vs-recurrent-an-algorithmic-difference)
showed that the two forms give different bits at equal accuracy. This case
builds both forms as Triton kernels for a Gated DeltaNet-style layer. It
verifies each form across runs, batches and modes, and measures what each way
of reaching trainer–engine agreement costs.
### Where the variance comes from

- **Not from the batch.** In both forms a program owns one sequence and head (and
  a tile). No reduction crosses sequences, so a larger batch only adds programs.
- **From where chunk boundaries fall.** In the chunked form, a token's output is
  the decayed chunk-start state plus terms from earlier tokens in its chunk.
  Suppose a call starts its own chunks: a decode step, or a prefill chunk of 7
  tokens. Then a token's terms are grouped differently from a full prefill, and
  the reduction order changes.
- **From the algorithm.** Recurrent and chunked are two exact forms of the same
  math, and they round differently. The chunked form also builds the decay from
  a cumulative sum of log-gates, where the recurrent form multiplies by one gate
  per step.
- **From the tile configuration.** Inside the recurrent kernel, the sum over Dk
  depends on the Dv tile width and the warp count. That does not matter for
  L1–L3 at a fixed configuration. It does matter for L4: two configurations are
  two reduction trees.

### What was built

| Kernel | One program per | Does, in a fixed order |
|---|---|---|
| recurrent | sequence, head, Dv tile (8 columns, 1 warp) | steps t in order. Its state tile stays in fp32 registers: decay, delta correction, rank-1 write, output. It loads the initial state and stores the final state. |
| chunked: prepare | sequence, head, chunk of 64 | in-chunk cumulative log-decay G; A = β_i (k_i·k_j) e^{G_i−G_j} for i > j; T = (I + A)⁻¹ by row-by-row forward substitution in fp32; W = T(βe^G k), U = T(βv) (the WY/UT form) |
| chunked: state | sequence, head, Dv tile | walks the chunks in order. It stores the chunk-start state, then v_new = U − W S, then S ← e^{G_last} S + (k e^{G_last−G})ᵀ v_new |
| chunked: out | sequence, head, chunk, Dv tile | o = (e^G q) S_start + ((q kᵀ) ⊙ e^{G_i−G_j}, j ≤ i) v_new |
| `AlignedChunked` (Python) | — | keeps the fp32 state at the last absolute multiple of C, plus the tokens since then. Every call (a decode step or a prefill chunk of any size) reruns the chunked kernels from that boundary over the pending and the new tokens, and returns the new tokens' outputs. |

The chunked kernels have two precisions:
- **bf16**: the matmuls take bf16 operands with fp32 accumulation. The state and
  the intermediates are rounded to bf16 at each matmul, as common chunked GDN
  kernels do.
- **fp32**: fp32 operands with `input_precision="ieee"`. These are FMA dots, not
  tensor cores, and they are untuned.

A partial last chunk masks its tail to zero. In it, every row does the same
arithmetic as in a full chunk: masked rows contribute exact zeros, and the
forward substitution never reads them. The `AlignedChunked` checks below are
what verify this.

## Evidence

### Setup, code and results

Taken through the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md), recipe 7.
- Code: [examples/linear_attention/gdn_bi.py](../../examples/linear_attention/gdn_bi.py)
  and the kernels in [gdn_kernels.py](../../examples/linear_attention/gdn_kernels.py).
- Results: [results-a40.json](../../examples/linear_attention/results-a40.json).
- Setup:
  - one NVIDIA A40, torch 2.5.1+cu121, Triton 3.1.0;
  - 8 heads, Dk = Dv = 128, chunk size C = 64;
  - inputs: q and k L2-normalized, in bf16; v and β in bf16; the log-decay in
    fp32. The state is fp32.
- Outputs are compared in fp32, before the cast to bf16. This is stricter than
  comparing the bf16 output.

The layer, per head, with S ∈ R^{Dk×Dv}:

    S_t = g_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ,   o_t = S_tᵀ q_t / √Dk

Here g_t = exp(glog_t) is in (0, 1). This is the gated delta rule. The plain
gated variant (no delta correction) was not built.

### Plain linear attention: chunked vs recurrent (an algorithmic difference)

Taken through the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
Code: [examples/kernel_cases/kernel_cases.py](../../examples/kernel_cases/kernel_cases.py) and the reference kernels in [tools/bi_kernels.py](../../tools/bi_kernels.py).
Results: [results-a40.json](../../examples/kernel_cases/results-a40.json).
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

### Results

**Invariance.**
- L1: B=4, T=1024, 5 runs.
- L2: `bi_check.check_batch` on a T=200 sequence (a partial last chunk) with a
  nonzero initial state. The sequence runs alone, then among 1–64 others (batch
  1–65, at the sizes around powers of two), first, middle or last, with random,
  zero or ×1e3 fillers. That is 218 comparisons per kernel.
- L3: T=300, B=2, with a nonzero initial state. Every mode continues from the
  saved fp32 state. The output (614,400 elements) and the final state (262,144)
  are compared bitwise against the full prefill.

| Kernel | L1 | L2 | L3 |
|---|---|---|---|
| **recurrent** | **pass** | **pass** | **pass**: per-token decode and chunked prefill 7/16/64/100 |
| chunked, bf16 or fp32 | **pass** | **pass** | depends on the chunk boundaries (next table) |

| Chunked mode vs chunked full prefill | bf16: output / state differ | bf16: after the cast to bf16 | fp32: output / state differ | fp32: after the cast |
|---|---|---|---|---|
| calls of 64, 128, 64, 44 tokens (boundaries at multiples of 64) | **0 / 0** | 0 | **0 / 0** | 0 |
| call-relative chunks, decode (1-token calls) | 99.7% / 100% | 56.5% | 88.9% / 88.1% | 208 elements |
| call-relative, calls of 7 | 97.7% / 100% | 53.3% | 84.0% / 83.5% | 137 |
| call-relative, calls of 16 | 94.7% / 100% | 41.2% | 78.0% / 81.7% | 124 |
| call-relative, calls of 100 | 66.7% / 100% | 22.9% | 55.7% / 81.9% | 86 |
| **aligned** (`AlignedChunked`): decode, calls of 7, 16, 64, 100 | **0 / 0** | 0 | **0 / 0** | 0 |

With calls of 100, only the first call's tokens (positions 0–99) match. Its
partial chunk [64, 100) reproduces the prefix of the full chunk [64, 128), so
the other two thirds differ: 66.7% of output elements.

**Recurrent tile configurations.** These are the same T=300 test, the same L1
and L2 matrix, and bits compared against the default (8 columns, 1 warp,
prefetch).

| Configuration | L1, L2, L3 | Output differs from the default | after the cast to bf16 | State differs |
|---|---|---|---|---|
| 8 columns, 1 warp, no prefetch | pass | 0 | 0 | 0 |
| 16 columns, 1 warp | pass | 74.2% | 82 elements | 35.3% |
| 16 columns, 2 warps | pass | 69.8% | 65 | 29.6% |
| 32 columns, 2 warps | pass | 76.1% | 83 | 35.4% |
| 32 columns, 4 warps | pass | 70.0% | 69 | 29.9% |
| 64 columns, 4 warps | pass | 76.0% | 92 | 35.5% |

**Accuracy and difference between the forms.** One sequence, T=2048, against
an fp64 recurrence on the same bf16 inputs. The table gives the maximum absolute
error and the relative Frobenius error.

| | Output (fp32) | Output after the cast to bf16 | Final state |
|---|---|---|---|
| recurrent | 3.0e-8, 2.50e-7 | 2.4e-4, 1.66e-3 | 2.6e-7, 1.92e-7 |
| chunked fp32 | 4.9e-8, 2.32e-7 | 2.4e-4, 1.66e-3 | 3.8e-7, 2.33e-7 |
| chunked bf16 | 4.6e-4, 3.01e-3 | 4.8e-4, 3.44e-3 | 4.8e-3, 2.86e-3 |

| Pair (T=2048) | Output elements that differ (fp32) | after the cast to bf16 | State | Max \|Δo\| |
|---|---|---|---|---|
| recurrent vs chunked fp32 | 90.6% | 653 of 2,097,152 (0.03%) | 91.5% | 6.0e-8 |
| recurrent vs chunked bf16 | 100.0% (18 equal) | 55.5% | 100.0% | 4.6e-4 |

**Cost: forward.** These are GDN-layer kernels only, not a trainer step. H=8,
D=128. Each figure is the median of 5 calls, in ms. The recurrent kernel is
shown at its default configuration and at the fastest of the 7 configurations.

| B × T | recurrent, default | recurrent, fastest | chunked bf16 (prepare / state / out) | chunked fp32 | recurrent ÷ chunked bf16 |
|---|---|---|---|---|---|
| 8 × 2048 | 2.65 | 1.87 (16 columns, 1 warp) | 2.03 (0.68 / 0.73 / 0.59) | 45.0 | 1.31 (fastest: 0.92) |
| 4 × 4096 | 3.01 | 2.96 (16 columns, 1 warp) | 2.07 (0.69 / 0.79 / 0.60) | 44.0 | 1.45 (1.42) |
| 2 × 8192 | 5.35 | 5.35 | 2.17 (0.69 / 0.85 / 0.60) | 43.6 | 2.47 |
| 1 × 8192 | 5.22 | 5.22 | 1.44 (0.38 / 0.73 / 0.31) | 20.4 | 3.61 |

Issuing the next token's loads early (prefetch) cut the default recurrent time
by 2–17% (2.72 → 2.65 ms at 8 × 2048, 6.43 → 5.35 ms at 2 × 8192). The bits
did not change.

**Cost: one decode step.** All 8 heads, in ms. The first figure is CUDA-graph
replay (GPU time). The figure in parentheses is eager, which includes Python
and launch overhead. Each aligned step times the chunked call on p + 1 tokens;
the cache's concatenation is not included.

| Decode batch | recurrent step | chunked bf16, 1-token call (call-relative, not bitwise) | aligned, mean over 0/15/31/47/63 pending | aligned, 63 pending |
|---|---|---|---|---|
| 1 | 0.012 (0.037) | 0.022 (0.099) | 0.040 (0.099) | 0.058 (0.099) |
| 16 | 0.036 (0.038) | 0.088 (0.101) | 0.134 (0.138) | 0.182 (0.183) |
| 64 | 0.134 (0.133) | 0.289 (0.291) | 0.464 (0.466) | 0.641 (0.642) |

The timing ran when no other process held the GPU: none appeared in 6
`nvidia-smi` samples taken during the run. An earlier timing run overlapped
another agent's process. Its forward times were within 3% of these; a few of
its chunked decode times differed by up to 38%.

### What this shows

1. **Both forms are batch-invariant by construction.** No reduction crosses
   sequences, and L1 and L2 passed for every kernel and every recurrent
   configuration. This confirms recipe 7's two L2 claims on a delta-rule layer.
2. **Mode invariance is about where chunk boundaries fall.** The recurrent
   kernel is L3 for free, because prefill, chunked prefill and decode run the
   same loop body. The chunked algorithm is L3 when every call's chunks sit at
   absolute multiples of C:
   - prefill in whole chunks reproduced full prefill bitwise;
   - call-relative decode and call-relative chunks of 7, 16 or 100 tokens did
     not (56–100% of elements differ);
   - recomputing the open chunk from the last boundary state (`AlignedChunked`)
     made decode and every chunk size bitwise equal to full prefill.

   So "chunked everywhere, with decode as chunk steps of the same algorithm"
   can be made bitwise. A naive chunk-1 decode is not bitwise.
3. **The precision of the chunked matmuls matters more than the algorithm.**
   - Chunked fp32 and recurrent are equally accurate (2.3e-7 and 2.5e-7 relative
     error). After the cast to bf16, they differ in only 0.03% of outputs.
   - Chunked with bf16 operands, like common chunked kernels, has 3.0e-3
     relative error, about 12,000 times larger. It differs from recurrent in
     55.5% of bf16 outputs.

   That error is 1.8 times the error of rounding the output to bf16 (1.66e-3).
   The chunk-versus-recurrent gap that matters in practice is therefore mostly
   a precision gap.
4. **The recurrent cost tracks T, not B × T.** At a fixed 16,384 tokens, the
   default recurrent kernel took 2.65, 3.01 and 5.35 ms as T grew from 2048 to
   8192. 1 × 8192, with half the tokens, took the same 5.22 ms. Each step costs
   about 0.64 µs and depends on the previous one. Against this untuned chunked
   kernel, the recurrent forward costs:
   - 0.9–1.5× at T ≤ 4096;
   - 2.5–3.6× at T = 8192.

   The reported 2–3× (Wang et al., "Closing the Training–Inference Mismatch
   for Linear Attention in RL"; see [references](../../handbook/references.md#rl-traininference-alignment))
   is in range at long sequences. That report measures trainer
   throughput, which this does not.
5. **Each choice also fixes a configuration.** The recurrent configurations
   differ from each other in 70–76% of output elements. Only 65–92 of 614,400
   bf16 outputs differ, but that is still not bitwise. Trainer and engine must
   run the same tile width and warp count (L4).

### Recommendation, with the measured cost of each option

| Option | Bitwise trainer–engine (L3 measured here) | Trainer forward, GDN layer | Engine decode step (graph, B = 1/16/64) | Engine bookkeeping |
|---|---|---|---|---|
| (a) recurrent forward everywhere | yes | 1.31× / 1.45× / 2.47× / 3.61× the chunked bf16 kernel at the four shapes (0.92× / 1.42× with the fastest configuration at T ≤ 4096) | 0.012 / 0.036 / 0.134 ms: the cheapest | none; sequences at mixed positions decode together |
| (b) chunked everywhere, aligned decode | yes | 1× (unchanged) | 0.040 / 0.134 / 0.464 ms on average: 3.4–3.7× recurrent, up to 5.0× | fp32 state at the last boundary, plus up to 63 pending tokens per sequence; mixed positions need per-sequence lengths (not built) |
| (c) accept the difference | no: 55.5% of bf16 outputs differ (bf16 chunked); 0.03% (fp32 chunked) | 1× (bf16); 14–22× as built (fp32) | recurrent: cheapest | none |

**What the data supports.** For a parity mode, use (a). It is bitwise with no
engine-side logic, and its decode step is the fastest measured. At T ≤ 4096 its
trainer-side cost on this layer's forward is 0.9–1.5×. Choose (b) when
sequences are long: at T = 8192 the recurrent forward costs 2.5–3.6×. (b) keeps
the trainer unchanged and moves the cost to decode, at 3.4–3.7× per GDN step
here. Which of the two is cheaper overall depends on how the workload splits
time between trainer forward and rollout decode. This layer benchmark does not
measure that split.

Choose (c) only when parity is not required. If you do, the precision of the
chunked kernel's matmuls decides most of the mismatch.

## Limits

- **No backward.** Option (a) needs a backward for the recurrent forward. The
  reported approach is a chunked backward, which is neither built nor measured
  here.
- **Untuned kernels, no production comparison.** `fla` (flash-linear-attention)
  is not installed on the host, so there is no comparison against production
  kernels.
  - A tuned chunked kernel is likely faster than this one, which would raise the
    recurrent ÷ chunked ratios.
  - The chunked kernels keep W, U, v_new and the chunk-start states in fp32 in
    memory. The aligned decode path writes all chunk-start states and the final
    state at every step. That likely explains much of its 3.4–3.7×.
  - The fp32 chunked kernel uses FMA dots that spill registers. It serves as an
    accuracy reference, not as a performance candidate.
- **One shape.** One GPU (A40), 8 heads, Dk = Dv = 128, C = 64. The gate was
  sampled with decay mostly between 0.88 and 0.98. Tested lengths: T = 200 for
  L2, 300 for L3, 2048 for accuracy, up to 8192 for timing.
- **Not tested:**
  - packed variable-length sequences (`cu_seqlens`);
  - a decode batch whose sequences sit at different positions, for
    `AlignedChunked`, which assumes one common position;
  - `torch.compile`;
  - the bits of CUDA-graph replays (graphs were only timed);
  - the plain gated (no delta) variant;
  - agreement with an inference engine's kernel (L4).

## How it was built

An agent wrote the kernels and the test, with no human edits. Runs on the A40:

1. **Smoke run** (`--quick`). The chunked state kernel needed 131 KB of shared
   memory, over the 101 KB limit, with the default pipelining. It was set to
   `num_stages=1`. Every other check ran.
2. **Second smoke run.** Every check passed at small shapes. Three performance
   edits followed, none of which changes the arithmetic:
   - The recurrent kernel loads the next token before the current token's
     arithmetic (prefetch). It is checked bitwise against the version without
     prefetch.
   - The forward substitution loops over the chunk's valid rows only. A 1-token
     decode chunk ran 63 iterations before.
   - The fp32 W/U products are computed in 32-column tiles. A register report
     had shown heavy spills in the fp32 prepare kernel.
3. **Third smoke run, then a full correctness run.** The results matched those
   in the tables. A one-off diagnostic found that the printout had rounded the
   653 differing bf16 outputs to "0.0%". The script now prints counts.
4. **A first timing run** overlapped another agent's process. A second timing
   run waited until no other process held the GPU for 15 s; its numbers are the
   ones above.
5. **A final correctness run** added L1 and L2 for every recurrent
   configuration. All results were identical to the first correctness run.
