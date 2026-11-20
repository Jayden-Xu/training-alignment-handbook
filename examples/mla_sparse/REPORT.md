# MLA and sparse attention: lab report

This is the full record behind the handbook page [MLA and sparse attention](../../handbook/3-kernels/mla-and-sparse.md). The page teaches the step and what to do; this report keeps the setup, every table, every number, the limits and how the case was built.

## Summary

- The absorbed and explicit MLA forms are equally accurate and differ in 65% of
  output bits. Within one form, prefill, decode, chunked prefill and batches are
  bitwise equal; an engine that prefills in one form and decodes in the other no
  longer matches a trainer.
- The attention core alone is not enough: every GEMM whose M changes with the
  step must be invariant, including the explicit form's decompression and the
  per-head output absorption (`bmm`).
- In the sparse (DSA-like) pipeline every selection flip came from the
  projection GEMM, not from top-k ties. With the one-config projection
  everything passed at these shapes (A40).
- Check selected sets before values: most output mismatches sat at positions
  whose set did not change.

## How

Recipes 5 and 6 of the [operator recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md)
were untested. This case takes both through the
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md):
- MLA, DeepSeek's latent attention, in its two algebraically equal forms;
- a DSA-like sparse attention: an indexer scores past keys, top-k selects 64,
  and attention runs over the selected keys only.
## Evidence

### Setup, code and results

Files:
- Code: [examples/mla_sparse/mla_sparse.py](../../examples/mla_sparse/mla_sparse.py)
  (the harness) and [mla_sparse_kernels.py](../../examples/mla_sparse/mla_sparse_kernels.py)
  (the new Triton kernels). The projections use `mm_one_config` from
  [tools/bi_kernels.py](../../tools/bi_kernels.py).
- Results: [results-a40.json](../../examples/mla_sparse/results-a40.json).
- Setup: one NVIDIA A40 (sm86), torch 2.5.1+cu121, Triton 3.1.0, bf16, random
  weights and inputs.

### Part A. MLA: absorbed vs explicit

#### Setup

The layer has the shape of DeepSeek-V2-Lite's attention:
- hidden 2048 and 16 heads;
- kv_lora_rank 512, qk_nope 128, qk_rope 64, v_head 128;
- the query projected directly (no query LoRA), RMSNorm on the latent, and
  rotate-half RoPE on the 64 rope dims;
- a latent KV cache: per token, the normalized latent c_kv (512) and the roped
  key k_pe (64), shared by all heads.

Two forms of the same math:

| Form | What it does | Attention dims (QK / V) |
|---|---|---|
| explicit | decompress per-head K and V from the cached latent with W_UK and W_UV (one GEMM over the cached tokens), then standard attention | 128 + 64 / 128, 16 KV heads |
| absorbed | fold W_UK into the query (per head, q_nope · W_UK^T → 512) and W_UV into the output (per head, 512 → 128); attend over the latent plus k_pe | 512 + 64 / 512, one KV head shared by 16 query heads |

The softmax scale is 1/sqrt(192) in both forms. Every engine step projects the
new tokens, writes their latent into the cache, and attends over the cache.
Explicit-form steps decompress the whole cached latent. A variant,
`explicit_kv`, decompresses only the new tokens into a per-head K/V cache.

Implementations, from the baseline to all invariant pieces:

| Name | Projections and per-head GEMMs | Latent RMSNorm | Attention core |
|---|---|---|---|
| baseline | `torch.matmul` / `torch.bmm` (cuBLAS) | torch ops | SDPA |
| cuBLAS + BI core | cuBLAS | torch ops | the new Triton kernel |
| BI GEMMs + SDPA | `mm_one_config`, grouped one-config GEMM | Triton | SDPA |
| BI GEMMs + torch norm + BI core | `mm_one_config`, grouped one-config GEMM | torch ops | the new Triton kernel |
| all BI | `mm_one_config`, grouped one-config GEMM | Triton | the new Triton kernel |

SDPA ran as an engine without an MLA kernel would run it:
- explicit form: V padded with zeros from 128 to 192 to match the QK head dim.
  Flash ran for causal prefill and maskless decode, memory-efficient for
  chunks with a boolean mask.
- absorbed form: K and V expanded to 16 heads. Head dim 576 exceeds flash's
  256, so the memory-efficient backend ran in every mode.

FlashMLA (Hopper only) and FlashInfer were not available.

#### Where the variance comes from

1. **GEMMs whose M depends on the step.** Every projection sees M = the number
   of tokens in the step. That is T·B in prefill, c·B in chunked prefill and B
   in decode. The explicit form's decompression sees M = the number of cached
   tokens. A row alone vs inside batches of 1–2048 rows (`check_batch`,
   3 positions, 3 fillers, 398 comparisons each):

   | Piece | cuBLAS / torch ops | One-config Triton |
   |---|---|---|
   | W_q 2048 → 3072 | fail at 20 of 22 sizes from M = 2, up to 1229 of 3072 elements | pass |
   | W_dkv 2048 → 576 | fail at 20 of 22 sizes from M = 2, up to 244 of 576 | pass |
   | W_kvb decompression 512 → 4096 (explicit) | fail at M = 63, 64, 1 element | pass |
   | absorb q, per head 128 → 512 (`bmm`) | pass | pass |
   | absorb o, per head 512 → 128 (`bmm`) | fail at all 22 sizes from M = 2, 1 element | pass |
   | W_o 2048 → 2048 | fail at 21 of 22 sizes, up to 1009 of 2048 | pass |
   | RMSNorm over 512 | pass (torch ops) | pass |

   The grouped kernel, one launch for the 16 per-head GEMMs, is the one-config
   kernel's body with a head index. Its output was bitwise equal to a loop of
   `mm_one_config` over heads.
2. **The attention core.** SDPA picks a backend from the call's arguments and
   head dims, so prefill, chunks and decode can run different kernels.
3. **The form itself.** The two forms round in different places: explicit
   rounds the per-head K and V to bf16, absorbed rounds the absorbed query and
   the latent output. This is an algorithmic difference, not a defect.

#### What was built

**An attention kernel for both forms** (`mla_attention`):
- **Two dot products.** The score is s = q_nope · k_nope + q_rope · k_rope,
  computed in that order into one fp32 accumulator. Neither 192 nor 576 has to
  be a power of two, and no padding is needed.
- **A shared rope key.** k_pe has one head, read by every query head.
- **Rows are (position, head) pairs.** In the absorbed form one program holds
  2 positions × 16 heads, so each latent block is loaded once for all heads.
  V is the latent tile itself.
- **A fixed walk over the cache.** KV is visited in fixed blocks from absolute
  position 0 with an online softmax and no split-KV, as in
  `attention_fixed_blocks`. Blocks past a row's position are exact no-ops.
- **One configuration per form, for every mode.** Explicit: 64 × 64 blocks,
  4 warps. Absorbed: 32 rows × 32 keys, 8 warps, 1 stage. With 576/512-dim
  tiles, 32 × 64 and 64 × 32 tiles needed 110,592 bytes of shared memory,
  above the A40's 101,376-byte limit. 16 × 64 fit but was 3x slower in
  prefill.

**Invariant GEMMs and norm:**
- `mm_one_config` for every projection;
- a grouped one-config GEMM for the per-head absorption;
- a row-per-program Triton RMSNorm. It was not needed: the torch-op RMSNorm
  passed at 512, and the "BI GEMMs + torch norm + BI core" row below passes
  everything.

#### Results

**1. Absorbed vs explicit on the same input** (prefill, T = 512, one sequence,
same weights). The fp64 reference is the explicit math with no intermediate
rounding.

| | all BI | baseline |
|---|---|---|
| layer output elements that differ | 681,049 of 1,048,576 (65.0%) | 694,268 (66.2%) |
| attention output (before W_o) | 629,241 (60.0%) | 631,040 (60.2%) |
| residual stream (x + layer output) | 191,951 (18.3%) | 200,891 (19.2%) |
| explicit vs fp64, layer: max abs / relative Frobenius | 1.54e-2 / 5.16e-3 | 1.54e-2 / 5.28e-3 |
| absorbed vs fp64, layer | 1.41e-2 / 5.21e-3 | 1.53e-2 / 5.34e-3 |

The two forms are equally accurate and differ in most output bits.

**2. L3: modes within one form.** One 512-token sequence. Prefill vs per-token
decode vs chunked prefill with chunks of 7, 16 and 64. Explicit also has
decode with a per-head K/V cache. The table gives elements that differ from
prefill, of 1,048,576.

| Implementation | explicit | absorbed |
|---|---|---|
| baseline | **fail**: decode 56.6%, K/V-cache decode 56.6%, chunks 72.2–73.2% | **fail**: decode 54.9%, chunks 71.6–72.7% |
| cuBLAS + BI core | **fail**: decode 52.7%, chunks 72.1–73.2% | **fail**: decode 54.8%, chunks 71.5–72.6% |
| BI GEMMs + SDPA | **fail**: decode 40.6%, chunks 43.3–43.4% | pass |
| BI GEMMs + torch norm + BI core | pass | pass |
| all BI | **pass**: all 6 modes bitwise equal | **pass**: all 5 modes bitwise equal |

**3. L2: one sequence among others.**
- Prefill: the 512-token sequence first or last among 2, 3, 8 or 16 random
  sequences (8 tests per cell).
- Decode: one step at position 511, first or last among 2, 3, 8, 16, 32 or 64
  sequences (12 tests per cell). The probe's latent cache is built alone; the
  others' caches are random.

| Implementation | explicit prefill | explicit decode | absorbed prefill | absorbed decode |
|---|---|---|---|---|
| baseline | fail 8/8: 62.9% at B = 2, 26.9% from B = 3 | fail 12/12: 1073–1135 of 2048 | fail 8/8: 68.4% at B = 2, 26.8–27.8% from 3 | fail 12/12: 867–996 of 2048 |
| cuBLAS + BI core | fail 8/8 (same pattern) | fail 12/12: 502–859 | fail 8/8 (same pattern) | fail 12/12: 832–980 |
| BI GEMMs + SDPA | pass | **fail 12/12**: 908–1006 | pass | pass |
| BI GEMMs + torch norm + BI core | pass | pass | pass | pass |
| all BI | **pass** | **pass** | **pass** | **pass** |

L1 (the same batch of 4 × 512 tokens, 5 runs) passed for the baseline and for
all BI, in both forms.

**4. Mixing forms.**
- The engine prefills a 256-token prompt, then decodes 256 tokens one at a
  time.
- The "trainer" recomputes all 512 tokens in one prefill.
- The comparison is on the residual stream h = x + MLA(x) of the 256 decoded
  positions (524,288 elements). It covers one layer, and two stacked layers
  with their own caches.
- All BI pieces in every run.

| Engine (prompt + decode) | vs trainer explicit, 1 layer / 2 layers | vs trainer absorbed, 1 layer / 2 layers |
|---|---|---|
| explicit + absorbed (what engines often do) | 77,830 (14.8%) / 129,035 (24.6%) | 0 / 93,116 (17.8%) |
| explicit + explicit | **0 / 0** | 77,830 / 138,849 |
| absorbed + absorbed | 77,830 / 138,849 | **0 / 0** |

With the baseline pieces, every pairing failed, including the same form
everywhere: 93,295–99,451 elements at one layer and 158,347–163,503 at two.

**5. Throughput.** One layer step, ms, CUDA-graph replay. Both timing runs
started with the GPU at 0% and no other process on it, and they agree within
2%. The GPU was shared with other jobs during the session, so an overlap
cannot be fully excluded. A first timing run that did overlap another job was
discarded.

Prefill, B = 4, T = 1024:

| | explicit | absorbed | absorbed / explicit |
|---|---|---|---|
| baseline (cuBLAS + SDPA) | 2.51 | 4.69 | 1.87x |
| all BI | 2.43 | 4.31 | 1.77x |

Decode, one step at context 1024:

| B | baseline: explicit, re-decompress | baseline: explicit, K/V cache | baseline: absorbed | BI: explicit, re-decompress | BI: explicit, K/V cache | BI: absorbed |
|---|---|---|---|---|---|---|
| 1 | 0.25 | 0.21 | 0.22 | 0.23 | 0.17 | 0.34 |
| 16 | 2.11 | 1.38 | 0.32 | 2.41 | 0.41 | 0.35 |
| 64 | 7.76 | 5.14 | 0.80 | 10.15 | 1.11 | 0.37 |

Cache per token and layer: 576 elements for the latent, 4160 for a per-head K/V
cache (7.2x).

**The cost of forcing one form** (all BI):
- **Absorbed in prefill:** 1.77x the explicit prefill.
- **Explicit in decode, re-decompressing the cached latent every step:** 6.9x
  the absorbed step at B = 16, 27x at B = 64.
- **Explicit in decode with a per-head K/V cache:** 1.17x at B = 16 and 3.0x at
  B = 64, plus 7.2x the cache memory.
- **At B = 1, explicit is faster** (0.17 vs 0.34 ms). The absorbed kernel puts
  all 16 heads of a sequence in one program and does not split KV, so a single
  sequence walks its 1024-token cache on one SM. This is the split-KV cost that
  recipe 4 warns about.

**The cost of the invariant pieces**, against cuBLAS + SDPA in the same form:
- **Prefill:** 0.97x (explicit) and 0.92x (absorbed).
- **Absorbed decode:** 1.52x at B = 1, 1.08x at B = 16, 0.47x at B = 64. The
  baseline expands the latent to 16 heads.

The baseline is not FlashMLA, so these ratios do not give the cost against a
production MLA kernel.

### Part B. Sparse attention with selection (DSA-like)

#### Setup

- One fused projection, hidden 1024 → 1664 columns:
  - the query, 8 heads × 128;
  - one shared K and one shared V head of 128;
  - the indexer query, 4 heads × 64;
  - the indexer key, 64;
  - 4 indexer head weights;
  - padding.
- The indexer score, fp32: score(t, s) = Σ_h w_h(t) · ReLU(q_I,h(t) · k_I(s)),
  with w scaled by 64^-1/2 · 4^-1/2. Keys after t get −inf.
- Top-64 of up to 2048 causal keys, one selection per token shared by its
  8 query heads.
- Attention over the selected keys only: one KV head, head dim 128, no RoPE.
- A 2048-token sequence. K, V and k_I are cached, as an engine would.

Pieces compared:

| Piece | Baseline | Invariant |
|---|---|---|
| projection | `torch.matmul` (cuBLAS) | `mm_one_config` |
| indexer | fp32 `torch.matmul` over the cached k_I (materializes [B, 4, T, T]), ReLU, weighted head sum | Triton: each tile of scores is computed for the 4 heads in a fixed order, in fp32, with the causal mask applied |
| top-k | `torch.topk` | stable descending sort, so ties go to the lower index; the 64 indices are then sorted ascending, with −1 for empty slots |
| sparse attention | gather K and V, `torch.matmul`, fp32 softmax, `torch.matmul`, in `torch.topk`'s order | Triton: one program per token, selected keys visited in ascending index order in 32-key blocks with an online softmax |

A second stable selection was kept as a cross-check: `torch.topk` over a
composite int64 key (an order-preserving int of the score bits, then the
inverted index). On every row tested it selected the same set as the sort.

#### Where the variance comes from

- **The projection GEMM.** The fused projection failed `check_batch` at 13 of
  22 sizes (M = 2–16, 64, 65, 255, 256, 1024, 2048), with up to 647 of 1664
  elements differing. A different q_I or k_I moves scores, and a moved score
  near the 64th place flips a selection.
- **The indexer GEMM.** The fp32 logits computed as one 2-D `torch.matmul`
  ([4M, 64] · [64, 2048]) failed `check_batch` from M = 4.
- **Top-k ties.** If `torch.topk` broke ties differently with batch size or
  row length, equal scores could flip selections with no score change at all.

#### Results

**Ties in `torch.topk`, constructed on purpose.** Two kinds of rows of 2048:
- 32 integer levels: 69 values tied at the 64th place, all 64 drawn from the
  tie group;
- 40 large values above a plateau of 2008 equal values, 24 drawn from the
  plateau.

60 trials varied:
- the batch: 2–4096 rows, the probe first or last;
- the other rows: ties or random;
- `sorted=True` and `False`;
- the row length: −inf padding to 2049, 4096, 8192 and 32768;
- the dtype: bf16.

In every trial `torch.topk` returned the same set and the same order as for
the row alone, and the set was the lowest-index one. The profiler confirmed
that both of its paths ran:
- single-block (`sbtopk::gatherTopK`) for one 2048-row;
- multi-block (`mbtopk::*`) for 4096 rows and for one 32768-row.

In the pipeline, 85 of 2048 query positions had a tie at the 64th place, mostly
exact zeros: 6.3% of valid scores are 0 because all four ReLU terms vanish.
`torch.topk` agreed with the (score desc, index asc) rule at every one. This is
observed behaviour of torch 2.5.1 on this GPU, not a documented contract.

**Selections and outputs.**
- L3: prefill vs per-token decode vs chunks of 7, 16 and 64.
- L2 prefill: the 2048-token sequence first or last among 2, 3, 8 or 16 others.
- L2 decode step: the last token, first or last among 2–64 others, with the
  probe's cache built alone.
- L2 whole decode: the probe generated token by token, every step a batch of
  2 (probe last), 8 (probe first) or 32 (probe last), vs decoded alone.
- Counts: query positions whose selected set changed (flipped selections), and
  output elements that differ, of 2,097,152 (1024 per position).

| Pipeline: projection / indexer / top-k / attention | L3 | L2 prefill | L2 decode step | L2 whole decode |
|---|---|---|---|---|
| cuBLAS / torch fp32 / `topk` / torch (baseline) | **fail**: decode 2 positions (2 flips), 58,383 elements; chunks 7 and 16: 546 positions (668 flips); chunk 64: 537 (569); 77% of elements | pass (8/8) | fail 10/12: 655 of 1024 at B = 2–16, 615 at 64, pass at 32; no flips | **fail**: B = 2 and 8: 544 positions (666 flips), all 2048 positions' outputs; B = 32: 2 positions (2 flips) |
| cuBLAS / torch fp32 / stable / BI | fail: same flips | pass | fail: 632 | fail: same flips |
| cuBLAS / BI / stable / BI | fail: same flips | pass | fail: 632 | fail: same flips |
| one-config / torch fp32 / `topk` / torch | pass | pass | pass | pass |
| one-config / BI / `topk` / torch | pass | pass | pass | pass |
| one-config / BI / stable / torch | pass | pass | pass | pass |
| **all BI**: one-config / BI / stable / BI | **pass** | **pass** | **pass** | **pass** |

L1 (a batch of 2 × 2048, 5 runs) passed for the baseline and for all BI.

**Throughput.** ms, from the same timing runs as Part A.

| | baseline | all BI |
|---|---|---|
| prefill step, B = 4, T = 2048 (CUDA graph) | 6.93 | 2.56 |
| decode step, B = 16, context 2048 (CUDA graph) | 0.31 | 0.10 |
| indexer, at the prefill shape (eager) | 3.73: fp32 matmul, materializes [4, 4, 2048, 2048] | 0.20 |
| top-k (eager) | 0.90: `torch.topk` | 1.76: stable sort (composite key: 5.70) |
| sparse attention (eager) | 2.07: gather + matmul | 0.16 |

The invariant pipeline is faster here because the baseline is plain torch:
- it materializes the per-head logits and the gathered K/V;
- production DSA kernels are fused and use FP8, and none was available on
  sm86.

The one piece that invariance itself adds is the stable top-k: +0.87 ms, 2.0x
`torch.topk`.

### What this shows

1. **The MLA forms are a choice, and every mode must make the same one.** The
   absorbed and explicit forms are equally accurate against fp64 and differ in
   65% of output bits. Within one form, all BI pieces gave bitwise-equal
   prefill, decode, chunked prefill and batches. Engines often prefill with
   the explicit form and decode with the absorbed one. Measured that way, the
   decoded tokens no longer match a trainer:
   - in either form, once there are two layers;
   - with one layer, a trainer in the absorbed form happened to match, because
     the latent cache does not depend on the form. With two, layer 2 caches
     the prompt from explicit-form outputs.

   Only one form for prompt, generated tokens and trainer gave 0.
2. **The attention core alone is not enough.** A batch-invariant core with
   cuBLAS around it still failed L3 in 53–73% of elements, and L2 in prefill
   and decode. Every GEMM whose M changes with the step had to go, including
   two that are easy to miss:
   - the explicit form's decompression, whose M is the number of cached
     tokens;
   - the per-head output absorption (`bmm`).
3. **SDPA behaved differently per form.** With invariant GEMMs:
   - explicit form (flash for causal and maskless calls, memory-efficient for
     masked chunks): failed L3 (41–43%) and decode L2;
   - absorbed form (memory-efficient in every mode at head dim 576): passed L2
     and L3 at these shapes.

   That is an observation about one backend at one shape, not a guarantee.
4. **In the sparse pipeline every flip came from the projection GEMM, not from
   top-k ties.**
   - With cuBLAS projections, chunked prefill changed the selected set at about
     a quarter of positions, and so did batched decode at batch 2 and 8. A
     stable top-k or an invariant indexer did not remove a single flip.
   - With the one-config projection, everything passed, even with torch's fp32
     indexer matmul, `torch.topk` and the torch gather attention, at these
     shapes.
   - The stable top-k changed nothing measurable here, because `torch.topk`
     already picked the lowest index among ties. It is still the safe choice:
     that tie order is not documented.
5. **Check sets before values.** Most baseline output mismatches sat at
   positions whose set did not change, e.g. 1.08M of the 1.62M under
   chunk 64. A value-only comparison mixes discrete flips with rounding drift.

## Limits

- One GPU (A40, sm86), bf16, random weights and inputs, one MLA shape and one
  sparse shape.
- MLA:
  - invariance tested at T = 512, timing at T = 1024;
  - no query LoRA, no YaRN RoPE scaling;
  - no FlashMLA or FlashInfer baseline (not available on sm86 here), so the
    absorbed baseline is SDPA's memory-efficient kernel on expanded K/V.
- The absorbed core's configuration:
  - BM and the warp count did not change the bits at a fixed 32-key block, in
    one test each at prefill and decode shapes; the key block size did;
  - so a per-mode BM might be possible, but it is not verified.
- Sparse:
  - top-64 rather than DSA's 2048;
  - a bf16/fp32 indexer rather than FP8;
  - an MQA attention without RoPE, instead of MLA over the selected latents;
  - T = 2048.
- Not tested:
  - paged KV;
  - varlen packing;
  - prompt lengths and chunk boundaries other than those listed;
  - sequences of different lengths in one batch;
  - the backward pass;
  - `torch.compile`;
  - invariance under CUDA graphs (graphs were used for timing only);
  - tensor parallelism;
  - agreement with a real inference engine (L4).
- L2 prefill for the sparse baseline passed only because every batch had
  M = 2048·B. The component check shows cuBLAS changes at other M, e.g. with
  varlen packing.
- On 8x A16 (sm86, driver 535.288.01, torch 2.5.1+cu121) the fp32 indexer matmul failure and one SDPA decode L2 failure did not
  reproduce at the regression suite's quick sizes (2026-11-19, see
  [regression](../../handbook/4-operations/regression.md#baselines-recorded-here)). The failure is
  claimed for the A40 only. Full sizes were not run on the A16.

## How it was built

An agent wrote the kernels and the harness, with no human edits. Runs on the
A40: 2 smoke runs at small shapes, 1 configuration probe, 6 full correctness
runs (one lost to a harness `KeyError`), and 3 timing runs (one discarded).
1. Smoke runs. All BI pieces passed L2 and L3 on the first run; nothing in the
   kernels needed fixing afterwards.
2. A configuration probe. 64-key blocks for the absorbed core exceeded shared
   memory, so the block was pinned at 32 keys, with 32 rows and 8 warps,
   fastest at prefill. That probe's timings may have overlapped another job.
3. The mixed-form test moved to a 2-layer stack. With one layer, a trainer in
   the absorbed form matched exactly, which would have hidden the prompt-cache
   effect.
4. Isolating configurations were added to find the source of the flips: cuBLAS
   projections with the invariant indexer, and the reverse. A whole-sequence
   decode L2 test was added, since a single decode step checks only one query.
5. The composite-key top-k took 5.7 ms vs 1.8 ms for a stable sort, with
   identical sets. The pipeline switched to the sort, and the sparse case was
   re-run, with the same results.
6. Timing. The first timing run overlapped another job on the shared GPU and
   was discarded. Two runs on an idle GPU followed, with CUDA graphs for the
   launch-bound decode steps.
7. A final run of every correctness case with the final script, which wrote
   [results-a40.json](../../examples/mla_sparse/results-a40.json). The earlier
   idle timing run is kept there as `timing_idle_run_1`. The configuration
   probe's entry comes from step 2; the kernels did not change after it.
