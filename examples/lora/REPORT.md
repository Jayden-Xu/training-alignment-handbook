# LoRA and multi-adapter batches: lab report

This is the full record behind the handbook page
[LoRA](../../handbook/2-rl/lora.md). The page teaches the step;
this report keeps the setup, every measurement and how the case was built.

## Summary

- Grouped and gathered LoRA GEMMs vary with the adapter's group size, and the
  scale placement and the point where base and update meet add rounding points:
  with an invariant base GEMM alone, 22–100 of 723 placements still failed.
- Merging is a lossy choice of form in bf16 (about half the output bits
  change): an engine serving merged weights cannot match an unmerged trainer.
- **Fix:** one LoRA kernel (row-selected adapter, one fp32 FMA, one cast) in
  trainer and engine gives trainer = engine bits and prefill = decode. The
  single cast is also the most accurate form.
- **Cost:** 1.2–2.7x the stock forward at 4096 tokens (A40).

## How

RL fine-tuning often trains LoRA adapters. The trainer runs one adapter over a
whole batch. The engine serves many adapters in one batch, with grouped or
gathered kernels, or it serves merged weights. This case measures where those
forms change the bits. It then builds one LoRA linear that the trainer and the
engine can both call.
### The fix: one LoRA linear for the trainer and the engine

`lora_linear(x, W, A, B, idx, scale)` in [lora_kernels.py](lora_kernels.py):

| Kernel | One program per | Reduction, in a fixed order | Output |
|---|---|---|---|
| shrink | row × 16 ranks | its own adapter's A (selected inside the kernel from `idx`); K ascending in chunks of 256; exact fp32 products, one fixed tree per chunk, fp32 accumulator | u, fp32, never rounded |
| base + expand | 64 rows × 64 columns | base: `mm_one_config`'s loop (64×64×32 tiles, K ascending, fp32); update: rank j ascending, one fp32 FMA per rank with each row's own B; then y = fma(s, update, base) | bf16, one cast |
| backward dX | as the forward | the forward kernel on (g, Wᵀ, du = shrink(g, B), A) | bf16 |
| backward dA, dB | adapter × 16 ranks × 64 columns | that adapter's rows with a nonzero output gradient, in ascending row order, in chunks of 32 through `tl.dot`; the fp32 factor enters as two bf16 terms (its rounding, then the rest) | fp32, written once, no atomics |

- **No regrouping.** Rows are never sorted or grouped for a reduction. A row of
  another adapter adds programs and never changes one.
- **One meeting point.** Base and update meet in one fp32 FMA, then the only
  cast. Rows with no adapter get exactly `mm_one_config(x, W)`.
- **Rank padding is exact.** Zero ranks add fma(0, 0, acc) = acc.
- **Trainer and engine call the same kernels.** `lora_linear_train` runs the
  same forward with autograd, on the bf16 values of the trainer's fp32 master
  weights.
- **Zero-gradient rows are skipped in dA and dB.** That makes right padding
  exact: without it, padding rows would shift the 32-row chunk boundaries.

## Evidence

### Setup, code and results

Taken through the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
- Code: [examples/lora/lora_bi.py](lora_bi.py) (the harness)
  and [lora_kernels.py](lora_kernels.py) (the kernels, plus
  models of vLLM's LoRA kernels).
- Results: [results-a40.json](results-a40.json) (full mode).
- Setup:
  - one NVIDIA A40 (sm86, driver 550.144.03), torch 2.5.1+cu121, Triton 3.1.0;
  - bf16 base weights, adapters and activations;
  - a square base layer, hidden 1024 and 4096;
  - 64 adapters with ranks 8, 16 and 64, zero-padded to rank 64 in one stack;
  - s = alpha/r or alpha/sqrt(r) (rsLoRA). The probe adapters have
    s = 8.49, 1.5 and 3.

A power-of-two scale is exact wherever it is applied. With s = 2 every scale
placement below gave the same bits. A first run used only power-of-two scales and
hid that source of variance, so the probes now use scales that are not.

### Where the variance comes from

The forms compared:

| Form | What runs |
|---|---|
| trainer | PEFT's unmerged path: `F.linear(x, W) + F.linear(F.linear(x, A), B) * s`, one adapter over the batch, cuBLAS, a bf16 result at every step |
| merged | `F.linear(x, W + s·B·A)`, the merge done in bf16 as PEFT's `merge` does |
| grouped | tokens sorted by adapter; the trainer's two GEMMs run once per adapter group (cuBLAS). M = the tokens sharing the adapter |
| gathered | per-row A_i and B_i gathered, then `torch.bmm` (or `einsum`) over rows |
| seg / row model | Triton models of vLLM 0.6.3.post1's SGMV (prefill) and BGMV (decode) kernels. They copy vLLM's grid, block sizes, split-K with fp32 atomics, dtypes, and the scale folded into B. They are models, not vLLM |

#### 1. Merged against unmerged: a choice of form

On the same 256 tokens (r = 16, s = 1.5, update about 1% of the base output),
merged and unmerged results differ in:
- 128,612 of 262,144 elements (49%) at hidden 1024;
- 604,021 of 1,048,576 (58%) at hidden 4096.

Both are about equally accurate. The difference is a choice, not a fault.

**What a bf16 merge loses.** W + s·B·A is stored in bf16. Where an element of
s·B·A is below half of W's last bit, it is lost from the merged weight. rho is
the update's size against the base output.

| rho | ΔW elements lost (merged = W) | relative error of the effective ΔW |
|---|---|---|
| 0.1 | 1.8% | 1.7e-2 |
| 0.01 | 18% | 0.17 |
| 0.001 | 78% | 0.80 |

- Hidden 1024 and 4096 give the same fractions to two digits.
- Merging in fp32 and casting once loses the same amount, 0.0182 against
  0.0183 at rho = 0.1. The loss comes from storing W + ΔW in bf16, not from the
  merge arithmetic.
- The output has the same limit. A bf16 output cannot hold an update smaller
  than its last bit either. At rho = 0.001, the update part of y (y minus x·W in
  fp64) had relative error 1.6–2.5 in every form, ours included. The merged
  weight loses the update for good, for every token. An unmerged form loses
  only what each output's rounding loses.

#### 2. Grouped GEMM: a request's bits depend on its co-batched tokens

The check: a 3-token request alone, against the same request co-batched. The
co-batched cases vary:
- 1–63 other adapters, with 1, 3 or 17 tokens each;
- first, middle or last position;
- random, zero or ×1000 fillers;
- rows with no adapter;
- extra tokens of the request's own adapter, for group sizes 3–4096.

That is 723 cases per hidden size.

| Form | hidden 1024 | hidden 4096 |
|---|---|---|
| grouped cuBLAS | **fail** 486/723 | **fail** 506/723 |
| grouped cuBLAS LoRA GEMMs, one-config base GEMM | **fail** 22/723, at group sizes 17–1024 | **fail** 100/723, at group sizes 8–4096 |
| gathered `bmm` | **fail** 474/711 | **fail** 492/711 |
| **ours** | **pass** 0/723 | **pass** 0/723 |

- With a cuBLAS base GEMM most failures come from the base: its M is the whole
  batch.
- With the base made invariant, the LoRA GEMMs alone still fail. Their M is the
  number of tokens that share the adapter, as with MoE experts.
- The group sizes that failed were 3 (grouped cuBLAS and gathered bmm), 8,
  15–17, 31–34, 63–65, 127–129, 255–257, 511, 512, 1024, 2048 and 4096 (the
  union over all configurations; only the listed sizes, not the gaps between them). The exact set depends on the
  hidden size and the rank: see `failing_group_sizes` in the JSON.

#### 3. Gathered against grouped against the trainer

- `einsum` and `bmm` gave identical bits.
- Gathered against grouped, on a batch of 12 adapters × 9 tokens: identical at
  hidden 1024, and 9,220 of 442,368 elements differ at 4096.
- Against the trainer's `F.linear` path (a 7-token sequence), grouped and
  gathered differed in the same elements:
  - 2,686–2,746 of 7,168 at hidden 1024;
  - 14–1,611 of 28,672 at hidden 4096.

  Both differences came from the base GEMM's M and the LoRA GEMMs' M.

#### 4. Rounding points

One adapter, 256 tokens. The table counts elements that differ from PEFT's form
(of 262,144 at hidden 1024 and 1,048,576 at 4096).

| Variant | hidden 1024 | hidden 4096 | rel. error vs fp64 (1024 / 4096) |
|---|---|---|---|
| PEFT: s on the B output, every step in bf16 | 0 | 0 | 2.35e-3 / 2.90e-3 |
| s folded into B in bf16 (vLLM at load) | 13,005–15,497 | 52,803–63,195 | 2.35e-3 / 2.89e-3 |
| s on the A output | 13,099–15,608 | 52,784–63,157 | 2.35e-3 / 2.89e-3 |
| fp32 update, added to the bf16 base in fp32, one cast | 15,305–19,786 | 61,736–79,981 | 2.34e-3 / 2.89e-3 |
| any of the above with s = 2 | 0 | 0 | |
| **ours: fp32 base + fp32 update, one fp32 FMA, one cast** | 72,673–74,663 | 462,701–464,384 | **1.66e-3 / 1.66e-3** |
| ours with two bf16 values added (rejected) | 10,230–14,921 | 371,375–375,711 | 2.35e-3 / 2.35e-3 |

- **Where s enters** changes 5–6% of the outputs.
- **Where base and update meet** matters more. Ours keeps the base GEMM's fp32
  accumulator and adds the scaled update to it before the only cast. Its error
  equals the bf16 rounding of the fp64 result (1.662e-3 against a floor of
  1.662e-3). Every form that adds two bf16 values sits at 2.3–2.9e-3.
- **Rank padding with zeros.** A rank-8 or rank-16 adapter padded to 64 with
  zeros, through cuBLAS `F.linear`:
  - identical at hidden 1024;
  - 72,737–75,626 elements (7%) differ at 4096, because the GEMM's K changed.

  Ours gave identical bits at both sizes. A zero rank adds fma(0, 0, acc) = acc.
- **Dropout.** The trainer's dropout (p = 0.05, train mode, on the LoRA input)
  changed 84% of the outputs. In eval mode it changed nothing. The engine has no
  dropout, so the trainer's log-prob pass must run with dropout off.

#### 5. Run to run, and the trainer's backward

Repeats of a 300-token mixed batch (20 runs; 5 for the backward at T = 1024):

| | hidden 1024 | hidden 4096 |
|---|---|---|
| seg model, fp32 shrink buffer (split-K 8, atomics) | differs in 19/19 runs, up to 4,853 of 19,200 | 19/19, up to 4,679 |
| row model, fp32 shrink buffer (split-K 64, atomics) | 19/19, up to 1,035 | 19/19, up to 4,900 |
| seg model, bf16 output | same in 19/19 | differs in 13/19 runs (1 element) |
| row model, bf16 output | same in 19/19 | differs in 1/19 runs (31 elements) |
| grouped cuBLAS, trainer backward (dX, dA, dB) | same | same |

- The atomics change the fp32 buffer on every run. The cast to bf16 hides most
  of it, but not all of it.
- Prefill (seg model) against decode (row model) on the same tokens:
  645–692 of 7,168 elements differ at hidden 1024, 2,769–2,904 of 28,672 at
  4096.
- **The trainer's backward is L1 but not L2.** A 37-token sequence alone,
  against right-padded with zero output gradient:
  - dX, dA and dB changed at 6 of 8 pad lengths (hidden 1024) and 7 of 8 (4096);
  - up to 42,100 dX elements and 37,926 dA elements changed, because the GEMMs'
    M and K grew;
  - dX also changed when other sequences joined the batch.

### Results

**Invariance** (ours, hidden 1024 and 4096, every check `expect="pass"`, 84 ok):

| Check | Result |
|---|---|
| L1: 20 repeats of a 300-token batch with 64 adapters and rows with no adapter | pass |
| L2: 723 request placements per hidden size (1–63 other adapters, 3 positions, 3 fillers, mixed ranks 8/16/64, group sizes 3–4096 around every cuBLAS threshold found) | pass |
| Rank padding r = 8 or 16 to 64 | pass |
| L3: prefill (M = 40) against decode (M = 6 per step), 3 adapters | pass |
| L4: trainer (one adapter, 2 sequences, `lora_linear_train`) against the engine (prefill with 7 adapters and rows with no adapter, or decode) | pass, 3 adapters × 2 sizes × 2 modes |
| Backward L1: dX, dA, dB, 5 runs at T = 1024 | pass |
| Backward L2: a 37-token sequence alone vs right-padded to 38–1024; dX among 1–31 other sequences; dA, dB with other adapters' rows reordered around it | pass |

**Accuracy** against fp64 (relative Frobenius error; mixed batch of 12 adapters
plus rows with no adapter):

| | forward y (1024 / 4096) | dX | dA | dB |
|---|---|---|---|---|
| floor: bf16 of the fp64 result | 1.662e-3 / 1.662e-3 | | | |
| **ours** | **1.662e-3 / 1.662e-3** | 1.66e-3 | 3.7e-6 (fp32) | 3.7e-6 (fp32) |
| grouped cuBLAS (the trainer's rounding) | 2.34e-3 / 2.35e-3 | 2.35e-3 / 2.63e-3 | 2.9–3.4e-3 (bf16) | 3.0–3.4e-3 (bf16) |
| gathered `bmm` | 2.34e-3 / 2.35e-3 | | | |
| seg / row models | 2.35e-3 / 2.35e-3 | | | |

The stock backward's dA and dB are bf16 parameter gradients. Ours are fp32. The
gradient columns compare what each path delivers, not like with like.

**Cost** (ms per forward, median of 15 after warm-up, `torch.cuda.synchronize`,
GPU otherwise idle). Tokens sit in contiguous runs per adapter. Columns are
1 / 8 / 64 adapters; tokens/s is in the JSON.

| hidden, rank, tokens | merged (1 GEMM) | grouped cuBLAS | seg model | row model | gathered `bmm` | **ours** |
|---|---|---|---|---|---|---|
| 1024, 16, 64 | 0.025 | 0.25 / 1.07 / 7.32 | 0.23 / 0.23 / 0.23 | 0.11 / 0.11 / 0.11 | 0.21 / 0.20 / 0.21 | **0.090 / 0.089 / 0.097** |
| 1024, 16, 4096 | 0.110 | 0.49 / 1.17 / 7.48 | 0.32 / 0.32 / 0.32 | 2.39 / 2.39 / 2.42 | 1.69 / 1.69 / 1.69 | **0.224 / 0.227 / 0.236** |
| 1024, 64, 4096 | 0.110 | 0.49 / 1.19 / 7.59 | 0.35 / 0.35 / 0.36 | 11.3 / 11.4 / 11.6 | 5.49 / 5.50 / 5.55 | **0.405 / 0.424 / 0.479** |
| 4096, 16, 64 | 0.097 | 0.35 / 1.12 / 7.59 | 0.31 / 0.27 / 0.29 | 0.13 / 0.13 / 0.13 | 0.23 / 0.24 / 0.25 | **0.146 / 0.147 / 0.151** |
| 4096, 16, 1024 | 0.331 | 0.72 / 1.41 / 7.68 | 0.55 / 0.55 / 0.57 | 0.93 / 0.93 / 0.94 | 1.95 / 1.94 / 1.96 | **0.617 / 0.619 / 0.620** |
| 4096, 16, 4096 | 1.18 | 2.33 / 2.64 / 8.80 | 1.57 / 1.53 / 1.56 | 3.58 / 3.56 / 3.56 | 7.48 / 7.45 / 7.46 | **2.76 / 2.76 / 3.32** |
| 4096, 64, 4096 | 1.19 | 2.37 / 2.78 / 8.86 | 1.67 / 1.61 / 1.67 | 12.8 / 12.8 / 12.9 | not run (memory) | **3.17 / 4.22 / 4.46** |

- **Against the merged single GEMM** (one adapter only): 1.5–3.6x at 64
  tokens, 2.0–3.7x at 4096 tokens. Most of the gap at 4096 tokens is the
  untuned one-config base GEMM against cuBLAS: about 2.0 against 1.2 ms at
  hidden 4096, measured in a probe during development; not in the saved results.
- **Against grouped cuBLAS:**
  - with 1 adapter at 4096 tokens: 1.2–1.3x slower at hidden 4096, 2x faster
    at hidden 1024;
  - faster from 8 adapters at small batches, and with 64 adapters at every
    size: its per-group loop costs about 0.11 ms per adapter of host time.
- **Against the seg model,** the closest form to the engine:
  - faster at hidden 1024, and at 64–256 tokens;
  - 1.1–1.3x slower at hidden 4096 and 1024 tokens;
  - 1.8–2.7x slower at hidden 4096 and 4096 tokens.
- **A training step of one layer** (forward + backward, one adapter):

  | hidden, rank, tokens | trainer cuBLAS | ours | ratio |
  |---|---|---|---|
  | 4096, 16, 1024 | 1.01 | 2.26 | 2.2x |
  | 4096, 16, 4096 | 3.55 | 6.82 | 1.9x |
  | 4096, 64, 1024 | 1.04 | 3.67 | 3.5x |
  | 4096, 64, 4096 | 3.62 | 8.93 | 2.5x |

  Values are ms. At hidden 1024 the stock step was bound by host time
  (0.76–1.11 ms at any size), and ours took 1.8–2.7 ms.

### What vLLM 0.6.3.post1 does (read from source)

Read from `vllm/lora/{punica,layers,lora,models}.py` and
`vllm/lora/ops/{sgmv,bgmv}_{shrink,expand}*.py` of vLLM 0.6.3.post1. It was not
run: torch does not import in that environment. Newer versions are untested.

- **Unmerged and gathered.** The base layer runs first, as the model's normal
  GEMM over all tokens. `PunicaWrapper.add_lora` then adds the update in place
  into that bf16 output. There is no merge and no per-adapter cuBLAS GEMM.
- **Two kernels by mode.** Prefill (`mapping.is_prefill`) uses SGMV. Decode uses
  BGMV, so a token's LoRA update rounds differently in prefill and decode.
- **SGMV.** `compute_meta` merges consecutive tokens with the same adapter into
  one segment. The shrink tiles 32×16×32 with `SPLIT_K = 8` and
  `tl.atomic_add` into a zeroed fp32 buffer. The expand casts that buffer to
  bf16, accumulates in fp32, casts to bf16 and adds it to the bf16 output.
- **BGMV.** The shrink uses `SPLIT_K = 64` with `BLOCK_K = 256`. Its products
  and sums are in bf16 (`tl.sum(a * b)`), followed by `tl.atomic_add`. The
  expand computes bf16 products and a bf16 sum, plus the output.
- **The scale is folded at load.** `LoRALayerWeights.optimize` multiplies
  `lora_b` by alpha/r in the LoRA dtype. The kernels get scale 1.0.
- **Rank padding.** `lora_a_stacked` and `lora_b_stacked` have
  `max_lora_rank` rows. A smaller adapter is copied into `[:r]`, with zeros
  above.
- **Consequences, measured on our models:**
  - the fp32 buffer is not L1, from split-K atomics;
  - the segment's M depends on its neighbours;
  - prefill and decode differ;
  - folding s into B in bf16 and adding two bf16 values move the rounding away
    from PEFT's.

### What this shows

1. **LoRA adds three new ways to lose parity on top of the base GEMM:**
   - the adapter group's M;
   - where the scale enters;
   - where base and update meet.

   An invariant base GEMM alone still left 22–100 of 723 placements failing.
2. **Merging is a choice of form, and a lossy one in bf16.** It changed about
   half of the output bits. It also dropped 18% of the update's weight elements
   when the update was 1% of the output, and 78% at 0.1%. An RL engine that
   serves merged weights cannot match an unmerged trainer.
3. **The property RL needs holds with one kernel.** One adapter over a whole
   training batch and the same adapter in a mixed serving batch gave identical
   bits, in prefill and decode.
4. **The single cast is also the accurate choice.** Keeping base and update in
   fp32 until one cast reached the bf16 rounding floor. Every form that adds two
   bf16 values was 40–75% further from fp64.
5. **Deterministic is not invariant, again.** The trainer's cuBLAS backward was
   identical run to run, and still changed with padding.

## Limits

- One GPU (A40). Square base layers at hidden 1024 and 4096, ranks 8, 16 and
  64, 64 adapters. No packed QKV or gate/up slices, no tensor parallelism, no
  LoRA on embeddings or the LM head.
- The vLLM forms are Triton models written from its source, not vLLM itself.
  Engine-level parity with vLLM was not tested.
- Not tested:
  - `torch.compile`;
  - CUDA graphs;
  - fp16;
  - DoRA;
  - quantized base weights (QLoRA);
  - adapters with different scales folded at load;
  - fp32 LoRA weights in the engine.
- Untuned: the base GEMM is `mm_one_config` (64×64×32), the main cost at large
  batches. The dA/dB kernel walks each adapter's rows in one program per tile,
  which is slow with one adapter and many tokens.
- The dA and dB factor uses about 16 bits of the fp32 shrink output (two bf16
  terms). The result is deterministic and invariant, but it is not an exact fp32
  product.
- The trainer must run LoRA dropout off when it recomputes log-probabilities.
- On 8x A16 (sm86, driver 535.288.01, torch 2.5.1+cu121) 7 of the stock cuBLAS failures did not
  reproduce at the regression suite's quick sizes (2026-11-19, see
  [regression](../../handbook/4-operations/regression.md#baselines-recorded-here)). The failure is
  claimed for the A40 only. Full sizes were not run on the A16.

## How it was built

An agent wrote the kernels, the vLLM models and the harness. There were no
human edits. Steps that took more than one try:
1. **The shrink.** The first version used a 16-row tile with `tl.dot`, one pass
   per distinct adapter in the tile. With 64 interleaved adapters it was
   2.6–7x slower than one program per row (0.60 against 0.085 ms at hidden 4096
   and 64 tokens). It became one program per row.
2. **The expand.** Unrolling the rank loop by 4 was slower than the plain loop,
   so the plain loop stayed.
3. **dA and dB.** The first version did one FMA per row, serially, and took
   1.5–1.8 ms with one adapter at 4096 tokens. It became `tl.dot` over 32-row
   chunks with the fp32 factor split into two bf16 terms.
   - Triton's pipeliner failed on the row-indirect loads ("operation scheduled
     before its operands"), so the kernel runs with `num_stages = 1`.
   - Skipping zero-gradient rows was added so that right padding cannot move
     the chunk boundaries.
4. **The autograd backward thread.** It had no current CUDA context for
   Triton: the launch failed with "invalid device context". `torch.cuda.set_device`
   did not help, because torch caches the device. A stream query in the
   backward binds the context.
5. **The first harness run** used power-of-two scales only. It hid the
   scale-placement variance, so the probe adapters were changed.
