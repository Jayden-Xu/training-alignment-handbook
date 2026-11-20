# The training step: backward, optimizer, loss, resume: lab report

This is the full record behind the handbook page [Training step](../../handbook/1-pretraining/training-step.md). The page teaches the idea and the step; this report keeps the setup, every configuration, every table, the limits and how the case was built.

## Summary

- On one A40, every run-to-run (L1) failure was a float sum ordered by timing:
  atomics in `index_add_` and `scatter_add_` and what is built on them, a long
  1-D `cumsum`, SDPA flash's dQ. Sort-based ops (embedding backward,
  cross-entropy, the norms, `linear`) repeated in default mode.
- Deterministic is not enough for two trainers to agree (L4): embedding backward
  and strict-mode scatters have orders that depend on the index count or the
  row width, and the optimizer path and the form of the loss divisor change bits.
- **Fix:** one written order for every scatter-style sum (`seg_sum`: stable
  sort, ascending source position, fp32, one rounding), matched bit for bit by a
  NumPy loop; pin the optimizer path and the loss reduction; restore seven
  pieces of state on resume ([checklist](#what-to-pin-so-two-trainers-agree)).
- **Cost:** 20-step runs repeat in strict mode at 1.44x the default step, or
  with targeted fixes at 1.03x (attention only) to 1.18x (attention, embedding
  and loss).

## How

The other cases made forward kernels and attention's backward invariant. A
training step has more parts:
- the backward of every other op;
- the loss reduction;
- the optimizer;
- AMP, clipping and gradient accumulation;
- checkpoint resume.

Pre-training alignment needs two answers for each part. Which ones change bits
from run to run (L1)? And which ones change bits between two correct
implementations (L4)?
### Where the variance comes from

Every L1 failure found here is a float sum whose order depends on timing:
- atomics in `index_add_` and `scatter_add_`;
- everything built on them: the backward of `index_select`, `gather` with
  repeated picks along dim 0, an MoE combine, a weighted `bincount`;
- the scan in a long 1-D `cumsum`;
- SDPA flash's dQ (from the attention case).

Ops that take a sort-based path were deterministic in default mode:
- embedding backward;
- `W[idx]` backward and `index_put_(accumulate=True)`;
- cross-entropy, the norms, `linear`, `topk`, `histc` on integer data.

A deterministic op can still have an order that is a detail of the
implementation. Three were measured:
- **Embedding backward has two orders.** We read torch's `Embedding.cu`, then
  reproduced its bits with a NumPy model of each path:
  - up to 3072 indices: the source rows go in chunks of 32, each chunk's sum is
    rounded to the weight dtype, then added to the weight;
  - above 3072: each destination's rows are summed in pieces of 10 in fp32,
    with one rounding at the end.

  So a token's gradient row depends on the batch. The test put a token's 100
  rows into batches of 128 to 65536 tokens, at the front, spread or at the
  back:
  - at or below 3072 tokens, the row equalled the token alone only when its
    rows came first;
  - above 3072, it never equalled the token alone, but every batch agreed with
    the others.

  It is L1, not L2. The fixed kernel passed every one of these cases.
- **Strict mode's accumulation order depends on the slice width.** In strict
  mode `index_add_`, `scatter_add_` and `index_put_(accumulate=True)` all use
  the sort-based kernel. Its order (written as NumPy models, matched bit for
  bit) depends on the row width H:
  - H = 1: 32 lanes, then a shuffle tree, then the remaining rows, then the
    base;
  - H = 2–32: the rows summed from zero in fp32, then added to the base;
  - H > 32: added one at a time into the destination, rounding to its dtype each
    time.

  Two frameworks that both "use deterministic scatter" agree only if they
  share this kernel.
- **A Python-number divisor is not a tensor divisor.** `total / n`, with `n` a
  Python int, gave different bits from `total / tensor(n)` in 2 of 4 loss
  cases (B = 16 and 32). A reduction that must be reproduced elsewhere should state which form
  it uses.

### The fix: one written order for every scatter-style sum

`ts_kernels.seg_sum(src, index, num_dst, base)` is a Triton kernel. Its order:

1. Sort the destination index stably. Each destination's source rows are then
   in ascending source position.
2. Run one program per (destination row, block of columns), with an fp32
   accumulator that starts at the base row.
3. Add the source rows one at a time, in ascending source position. Round once
   to the output dtype.

No atomics, and one write per element. The order is a property of the
definition, not of the kernel. The same NumPy loop that states it
(`for i ascending: out[index[i]] += src[i]` in fp32) matched it bit for bit:
- H = 1, 2, 31, 32, 33, 64 and 256;
- N = 3071, 3072, 3073 and 20000;
- fp32 and bf16 sources;
- with and without a base;
- column blocks of 16, 64 and 128, strided source rows, int32 indices.

That is the L4 claim: any framework that implements this loop gets these bits.
It replaces:
- `index_add_`, `scatter_add_` and `index_put_(accumulate=True)`;
- a weighted `bincount`;
- the backward of the embedding, `index_select`, `gather`, `W[idx]` and an MoE
  dispatch (`gather_rows_fixed`).

Two more pieces:
- `seq_cumsum` is a left-to-right fp32 scan, one program per row.
- `tree_mean_loss` is the loss reduction (section 5).

### What to pin so two trainers agree

- [ ] **No atomics in any float sum.** Use strict mode with
      `warn_only=False`, or the fixed-order kernels. Strict mode raises on
      `cumsum` (float), weighted `bincount` and `histc`. Replace those three
      first.
- [ ] **For agreement across frameworks, one written order for every
      scatter-style sum:** ascending source position, fp32, one rounding.
      Embedding backward and strict-mode scatters are deterministic, but their
      order depends on the index count or the row width.
- [ ] **Attention backward:** the atomic-free kernel, or SDPA in strict mode
      (see the attention backward case).
- [ ] **The optimizer path**: fused, foreach or single-tensor, with the same
      torch build. The default choice is not stable across versions.
- [ ] **The loss reduction**, as a fixed tree, including how the count divides.
      Not `mean`, which depends on the layout.
- [ ] **The micro-batch split and its order.**
- [ ] **fp16 GradScaler**: the scale schedule too. bf16 needs none.
- [ ] **The checkpoint's seven pieces of state.** The RNG pieces matter when
      dropout or reshuffling uses them.
- [ ] **TF32 off and the same `CUBLAS_WORKSPACE_CONFIG`** on both sides.

## Evidence

### Setup, code and results

- Code: [examples/training_step/](./).
  - `train_step.py` is the driver. Each section runs in its own process, so
    that strict mode and `CUBLAS_WORKSPACE_CONFIG` are set before CUDA starts.
  - `ts_ops.py`: the per-op tests.
  - `ts_kernels.py`: the fixed-order kernels.
  - `ts_model.py`: the model and the trainer.
  - `ts_optim.py`: the optimizer paths and their order models.
  - `ts_sections.py`: loss, AMP and clipping.
- Results: [results-a40.json](results-a40.json).
- Setup: one NVIDIA A40, torch 2.5.1+cu121, Triton 3.1.0, CUDA 12.1, cuDNN 9.1.
  TF32 off. bf16 autocast with fp32 weights unless a row says otherwise.
- Modes:
  - **default**: stock torch.
  - **strict**: `torch.use_deterministic_algorithms(True, warn_only=False)` with
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
  - **fixed**: stock settings, with the fixed-order kernels below swapped in.
- SDPA's backward is not re-measured here. See the
  [attention backward case](../../handbook/3-kernels/attention.md#backward-dq-dk-dv): with `warn_only=True` it
  only warns and stays non-deterministic.

### Results

#### 1–2. Ops: L1 over 20 runs, forward plus backward

Times are medians of 9 runs after 3 warm-ups, synchronized, on the same GPU.
"differs" gives the most elements that changed in any run, out of the total.

| Op (case) | Default L1 | Strict L1 | Strict raises? | Fix | Time: default / strict / fix (ms) |
|---|---|---|---|---|---|
| embedding backward, fp32/bf16, N = 512–65536 indices (also 64 distinct rows) | repeats | repeats | no | `gather_rows_fixed` (for L2 and L4; stock is L1) | N=3072: 0.37 / 0.33 / 2.10; N=65536: 1.06 / 1.05 / 3.16 |
| tied embedding + head, bf16 autocast, N=4096 | repeats | repeats | no | `gather_rows_fixed` | 3.68 / 4.67 / 4.58 |
| `index_select` backward, N=4096 / 65536 | **differs**: up to 6,628 / 377,962 of 8.2M dW elements (bf16) | repeats | no | `gather_rows_fixed` | N=65536 bf16: 26.5 / 4.15 / 3.20 |
| `W[idx]` backward (`index_put_` accumulate) | repeats | repeats | no | `gather_rows_fixed` | N=65536 bf16: 4.21 / 3.93 / 3.19 |
| `index_add_`, R=1024, N=16384, H=256 (and 3 per row; 1-D) | **differs**: 48,760 of 262,144 (fp32), 101,833 (bf16); 64/64 in 1-D | repeats | no | `index_add_fixed` | fp32: 0.09 / 0.21 / 0.56 |
| `scatter_add_`, same cases | **differs**: 43,482 of 262,144 (fp32); 63/64 in 1-D | repeats | no | `scatter_add_fixed` | fp32: 0.06 / 4.29 / 0.55 |
| `index_put_(accumulate=True)`, same cases | repeats | repeats | no | `index_add_fixed` (one order for all widths) | 0.23 / 0.20 / 0.55 |
| `gather` backward, one pick per row (LM loss) | repeats | repeats | no | not needed | 1.20 / 2.87 / — |
| `gather` backward, repeated picks along dim 0 (R=64, N=32768) | **differs**: 64/64 | repeats | no | `seg_sum` | 0.20 / 0.66 / 0.53 |
| `gather` backward, repeated picks inside a row (T=4096, K=8) | repeats | repeats | no | `seg_sum` (order) | 0.21 / 1.43 / 0.60 |
| `cross_entropy` and `log_softmax`+`nll_loss` backward, N=4096, V=32000 | repeats | repeats | no | not needed | 8.18 / 8.17 / — |
| RMSNorm (torch ops) backward, bf16 and fp32, 4096×1024 | repeats | repeats | no | not needed | bf16: 2.10 / 1.29 / — |
| LayerNorm backward, bf16 and fp32 | repeats | repeats | no | not needed | bf16: 0.30 / 0.34 / — |
| `linear` backward (dW, db, dX), bf16 autocast and fp32, 4096×1024→2048 | repeats | repeats | no | not needed | bf16: 1.60 / 2.59 / — |
| `cumsum` [64, 4096], fp32 and bf16, forward+backward | repeats | — | **yes** | `seq_cumsum` (forward) | fp32: 0.23 / — / 0.50 |
| `cumsum` [1048576] fp32, forward+backward | **differs**: 745,870 of 1M (y), 637,047 (dx) | — | **yes** | `seq_cumsum` (one row; untested at 1M) | 0.25 / — / — |
| `cumsum` int64 | repeats | repeats | no | not needed | 0.05 / 0.06 / — |
| `topk` backward, k=8 of 64, with and without tied scores | repeats | repeats | no | not needed | 0.23 / 1.31 / — |
| `bincount`, no weights | repeats | repeats | no | not needed | 0.10 / 0.17 / — |
| `bincount`, fp32 weights | **differs**: 54/64 | — | **yes** | `bincount_fixed` | 0.10 / — / 0.53 |
| `histc`, 64 bins, integer-valued data | repeats | — | **yes** | not needed on this data; `bincount_fixed` with unit weights | 0.05 / — / — |
| MoE dispatch → 16 experts → combine, top-4, backward (`x[tok]` + `index_add_`) | **differs**: output 2,755 of 524,288 | repeats | no | `gather_rows_fixed` + `index_add_fixed` | 16.0 / 41.1 / 19.4 |
| MoE, `index_select` + `scatter_add_` | **differs**: output 6,873, dx 2,933 | repeats | no | the same | 16.8 / 41.5 / 19.6 |
| SDPA backward | flash fails (dQ) | repeats | no (`warn_only=False` needed) | atomic-free Triton kernel | see [attention backward](../../handbook/3-kernels/attention.md#backward-dq-dk-dv) |

Not measured as separate rows:
- the GQA `repeat_kv` backward (expand or `repeat_interleave`): it repeated in
  both modes;
- a full `sum` and `mean`: they repeated.

#### 3. Whole step

The model has 4 layers, d = 1024, 8 query heads and 2 KV heads, SwiGLU 2048,
vocab 8192 and an untied head: 53.5M parameters in 40 tensors. Training uses
B = 4, T = 1024, AdamW and a warmup-cosine LambdaLR, with clipping at 1.0.

Each run lasts 20 steps from the same seed and data. Two separate processes are
compared at every step:
- the loss bits;
- an XOR plus wrap-around sum fingerprint of every gradient (after unscaling,
  before clipping);
- the same fingerprint of every weight after the update.

| Setting | First difference | Step time (median) |
|---|---|---|
| default | step 0, gradient of `blocks.3.wq` (the first tensor the backward reaches after SDPA's dQ); 31 of 40 gradients differ at step 0; the loss is still equal | 43.0 ms |
| strict | none in 20 steps | 61.8 ms (1.44x) |
| fixed: the atomic-free attention only (`fix_l1`) | none in 20 steps | 44.2 ms (1.03x) |
| fixed: attention, embedding backward and loss reduction (`fix_all`) | none in 20 steps | 50.7 ms (1.18x) |

- In this model the only L1 breaker was SDPA's backward. The embedding
  backward and the loss repeat in default mode.
- `fix_all` also makes the embedding backward and the loss follow a written
  order, which `fix_l1` does not. That matters for L4, not for L1.
- The three passing settings give different bits from each other: for example,
  a different step-0 loss for the SDPA settings and the fixed attention. Each
  is a choice.

#### 4. Optimizer paths

The same gradients and initial state went through each path: 4.35M parameters
in 7 tensors, including odd shapes and a 1-element tensor. Gradients mixed
magnitudes from 1e-5 to 1 and had 2% exact zeros. AdamW used lr 3e-4,
betas (0.9, 0.95), eps 1e-8 and weight decay 0.1. SGD used lr 0.1, momentum 0.9
and weight decay 1e-4.

Each path repeats bit for bit over 100 steps. Between paths:

| Pair | After 1 step: parameters differing (max ulp) | After 100 steps | Moments |
|---|---|---|---|
| AdamW fused vs foreach | 1,046,446 / 4.35M (4929) | 4,348,375 (99.96%) | exp_avg, exp_avg_sq differ from step 1 |
| AdamW fused vs single-tensor | 1,047,104 (4929) | 4,348,403 | differ from step 1 |
| AdamW foreach vs single-tensor | 3,759 (1573) | 419,261 (9.6%) | equal at every step |
| SGD fused vs foreach | 242,336 (23246) | 3,894,203 | buffer differs from step 1 |
| SGD foreach vs single-tensor | 0 | 0 | equal |

**The rounding-point table.** The table below is written from the torch 2.5.1
sources (`torch/optim/adamw.py`, `torch/optim/sgd.py`, `fused_adam_utils.cuh`,
`FusedSgdKernel.cu`). It was then checked by replaying every step with a model
of each order and comparing all parameter and state bits.

| Path | Weight decay | Moments | Bias correction and eps | Update | Model reproduces |
|---|---|---|---|---|---|
| AdamW single-tensor | `p *= fl32(1 - lr*wd)` | `lerp_` (one fused multiply-add), `mul_` then `addcmul_` (fused multiply-add) | `sqrt(v)` × fl32(1/sqrt(bc2)) (division by a Python scalar runs as multiply by the reciprocal), `+ eps` | `addcdiv_`: fma(−lr/bc1, m/denom, p) | every bit, 100 steps |
| AdamW foreach | the same | the same | `_foreach_div_` by sqrt(bc2): a true division, then `+ eps` | the same | every bit, 100 steps |
| AdamW fused | `p - lr*wd*p` in fp64 (hyperparameters are doubles), one rounding | `beta*m + (1-beta)*g` in fp64, one rounding | bias corrections from the fp32 step count, through double; `sqrt(v)/bc2_sqrt` in fp32, `+ eps` in fp64 | `p - (step_size*m)/denom` in fp32 | step 1 exactly (parameters, m, v); later steps all but 12,690 element-steps (1 ulp in m first). The remaining difference is not identified |
| SGD single-tensor and foreach | `grad + wd*p` as one fused multiply-add | `buf*momentum + g` | none | fma(−lr, buf, p) | every bit, 100 steps |
| SGD fused | `g + wd*p` in fp64, one rounding | `momentum*buf + g` in fp64 | none | `p - lr*g` in fp64, one rounding | all but about 35 of 1M parameters per step (1 ulp) |

What this explains:
- **foreach against single-tensor AdamW** differ only in how `sqrt(v)` is
  divided by the bias correction: a true division against a multiply by the
  reciprocal. The moments stay equal and the parameters drift apart: 0.09% at
  step 1, 9.6% at step 100.
- **The fused kernels** compute in fp64 where the hyperparameters are doubles,
  and round once per line. They differ from the other paths in the moments
  from step 1.

**How two trainers agree.** Pin the same path and the same torch build. Do not
leave it to the default, which picks foreach on CUDA today. Within one path the
results repeat. The cost of each choice is the step time at 53.5M parameters
in 40 tensors:

| | fused | foreach | single-tensor |
|---|---|---|---|
| AdamW | 4.40 ms | 7.98 ms | 7.67 ms |
| SGD | 3.54 ms | 4.45 ms | 4.26 ms |

To agree with a different framework, that framework must implement the same
order. For the single-tensor and foreach paths, the table above is that order.

#### 5. Loss reduction

The same per-token fp32 losses were averaged from several layouts:
- packed;
- padded, and padded further;
- flat;
- per-sequence sums;
- two half batches.

The cases were B = 3, 8, 16 and 32 sequences, up to 100–2048 tokens each.

| Reduction | Same bits in every layout? |
|---|---|
| `.mean()` / `sum()/n` over the same contiguous buffer, viewed as [B, T] or flat | yes (it is the same memory) |
| padded further, per-sequence sums, two half batches | no: each changed in at least one of the 4 cases |
| `F.cross_entropy(mean)` padded with `ignore_index` vs packed | no in 3 of 4 cases. The per-token losses themselves were equal |
| `sum()/n` padded, flat [B·T], per-sequence sums (B = 3) | no |
| **`tree_mean_loss`** | **yes: padded, padded further, packed, and two half batches whenever B/2 is a power of two** |

`tree_mean_loss` defines the order:
1. A pairwise tree over each sequence's absolute positions, padded to a power
   of two with +0.0.
2. The same tree over the sequences, in batch order.
3. One fp32 division by the token count, held in a tensor.

Padding adds +0.0, which changes no value except −0.0. A packed layout is
first moved to rows, which is data movement only.

Cost: 0.49 ms against 0.03 ms for `.mean()` on 64 × 4096 values. In the whole
step it is part of `fix_all`.

#### 6. AMP: GradScaler

- **bf16 autocast:** the gradients unscaled from a scaled backward were bitwise
  equal to an unscaled backward at every scale tested, 2^0 to 2^40. That holds
  because the scale is a power of two and bf16 has fp32's exponent range, so
  nothing overflows or becomes subnormal.
- **fp16 autocast:** scaling changes the bits. At 2^4 to 2^16, 25.8M gradient
  elements differed from the unscaled backward, and the number of exact zeros
  changed (8,089,468 unscaled against 8,085,504–8,085,736 scaled). Values that
  are subnormal or zero in fp16 without scaling keep more bits with it. From
  2^24 the scaled backward overflowed to inf. So scaling is exact only while
  every scaled value stays normal and finite.
- **The skip path is exact.** With an inf placed in one gradient, all three
  AdamW paths (single, foreach, fused) left the parameters, `exp_avg` and the
  step count bitwise unchanged. The scale halved (65536 → 32768).
- **Consequence for resume:** in bf16, a wrong GradScaler scale changes no bits
  (section 9).

#### 7. Gradient accumulation

The `fix_all` model was run for 10 steps. The same 4 sequences per step were
grouped differently:

| Comparison | Result |
|---|---|
| 4 micro-batches of 1, run twice | equal at every step |
| one batch of 4 vs 4 micro-batches of 1 | differ at step 0: the loss and all 40 gradients |
| 2 micro-batches of 2 vs 4 of 1 | differ at step 0 |
| 4 micro-batches of 1, reversed order | differ at step 0: all 40 gradients. The loss bits happened to be equal |

Accumulation is a different grouping of the sum, and its order matters. It is
a choice: fix the micro-batch split and the order on both sides.

#### 8. Gradient clipping

`clip_grad_norm_(max_norm=1.0)` ran on 53.5M parameters in 40 tensors:
- foreach and per-tensor repeated over 20 runs;
- they gave the same norm and the same clipped gradients bitwise;
- a fixed-tree norm (`clip_fixed_`) also repeated, and happened to give the
  same norm.

Times: foreach 2.9 ms, per-tensor 3.2 ms, fixed tree 12.1 ms (4.2x). On this
stack clipping needs no fix. The fixed tree only matters for agreement with
another framework, whose norm order is unknown.

#### 9. Resume

The setup:
- `fix_all`, dropout 0.1;
- GradScaler on, with a growth interval of 3 so that the scale changes;
- the sampler reshuffles from the CPU RNG every 16 steps.

One run took 20 steps straight. The other took 10, saved, and continued in a
new process for 10 more. Then one saved piece was left out at a time:

| State omitted on resume | First differing step | First differing tensor |
|---|---|---|
| nothing | none (20 steps equal) | — |
| model | 10 | loss |
| optimizer (moments and step) | 10 | weight `head.weight` (the gradients are still equal) |
| only the optimizer's step counter | 10 | weight `head.weight` |
| scheduler | 11: step 10 still uses the learning rate stored in the optimizer's param groups; the new scheduler sets a wrong one from step 11 | weight `head.weight` |
| GradScaler | none: in bf16 the scale changes no bits (section 6) | — |
| CPU RNG | 16, at the sampler's reshuffle | loss |
| CUDA RNG | 10: dropout masks | loss |
| data position | 10 | loss |
| CUDA RNG, with dropout 0 | none | — |

The checkpoint that resumes bitwise holds seven pieces:
- model;
- optimizer, step counters included;
- scheduler;
- GradScaler;
- `torch.get_rng_state()`;
- `torch.cuda.get_rng_state()`;
- the sampler's epoch, position and permutation.

#### 10. Cost

| Setting | Step (B=4, T=1024, 4 layers) | Against default |
|---|---|---|
| default | 43.0 ms | 1.00x |
| strict | 61.8 ms | 1.44x |
| fixed attention only | 44.2 ms | 1.03x |
| fixed attention, embedding and loss | 50.7 ms | 1.18x |

Per-op costs are in the op table. The largest strict-mode costs:
- `scatter_add_`: 4.3 ms against 0.06 ms (72x);
- the MoE block backward: 2.6x (the fixed kernels: 1.2x);
- `topk` backward: 5.8x;
- `gather` backward: 2.4–7x.

## Limits

- One GPU (A40), one torch build (2.5.1+cu121). No compile, no CUDA graphs,
  no distributed training (gradient all-reduce is covered in the
  [collectives case](../../handbook/3-kernels/collectives.md) only for the forward).
- L2 (batch invariance) was tested only for the embedding backward. The
  whole-step test is L1, across processes.
- The fused AdamW and fused SGD order models are close, not exact. Their exact
  order is not established.
- The cross-framework (L4) claim rests on the NumPy model of the fixed order.
  No second framework was run.
- `seq_cumsum` covers the forward only. A `cumsum` backward would be the same
  kernel run in reverse (untested). `topk` backward repeated in default mode.
  Its scatter has no repeated destination.
- The fixed kernels are untuned and launch-bound at small sizes (about 0.5 ms
  minimum).
- The GPU was shared with other agents' jobs during the runs. Every timed
  command held a lock, but a foreign process was resident on the GPU at times.

## How it was built

An agent wrote the code. Things that took more than one try:
1. **Reading the order from source alone was not enough.** The first NumPy
   models of strict-mode scatter matched only some row widths. Three models
   were needed, one per width class, and that is the finding.
2. **The fused optimizer kernels.** An fp32 model of fused AdamW was far off.
   An fp64 model matched step 1 exactly, but not every later bit. The source was
   read through a summarizing web fetch (no torch source on the host), so the
   exact types of a few intermediates are unconfirmed.
3. **The loss tree failed once:** packed against padded at B = 32. The cause was
   the divisor: a Python int against a tensor count. The definition now fixes
   that form.
4. **Resume:** the first run failed to restore the CPU RNG, because the
   checkpoint was loaded to the GPU and `set_rng_state` needs a CPU byte
   tensor. Omitting the GradScaler state changed nothing, which led to the
   bf16 exactness check in section 6.
5. **Quick mode** took 12.7 minutes of wall time while another job shared the
   GPU lock. The full run took 12.7 minutes (764 s) on the second attempt.
