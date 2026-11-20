---
name: batch-invariant-kernel
description: Write, adapt or verify a batch-invariant GPU kernel when no ready-made one exists. Fix the reduction tree; remove split-K, split-KV, atomics and shape-dependent dispatch; make MoE combine and quantized GEMM order-stable; then prove bitwise invariance across batch size, row position, co-batched content, prefill/decode and chunking. Use when a model brings an operator without a batch-invariant implementation: a new attention variant (MLA, sparse/DSA, linear/GDN), MoE dispatch and combine, FP8/FP4 GEMM, or a closed-source kernel.
---

# Batch-invariant kernel

Own one operator at a time. Take it from "unknown or variant" to "verified
invariant at a stated level", or to a documented fallback. Return the change
(configuration, replacement or kernel edit), the verification report, the
throughput cost, and a handbook entry. Selecting a ready-made batch-invariant
op is step 2 below. This skill is for when that is not enough.

Recipes per operator class: [references/operator-recipes.md](references/operator-recipes.md).
The verification protocol and how to read failures:
[references/verification.md](references/verification.md).
The checker: [tools/bi_check.py](../../tools/bi_check.py).
Reference kernels to start from: [tools/bi_kernels.py](../../tools/bi_kernels.py),
verified on one GPU type:
- a one-config GEMM;
- fixed-block causal attention with an atomic-free backward.

More verified kernels come with the case studies, under `examples/`:
- paged and varlen attention;
- recurrent and chunked GDN;
- the absorbed MLA core;
- a stable top-k sparse pipeline;
- a fixed-slot expert-parallel combine;
- an fp32-output GEMM for a logical TP size, and a fixed-order vocab-parallel
  log_softmax;
- a LoRA linear with row-selected adapters;
- a counter-keyed sampler ([tools/bi_sampling.py](../../tools/bi_sampling.py));
- a sorted, ascending segmented sum for backward scatters;
- logical-shard data-parallel gradients.

[tools/bi_audit.py](../../tools/bi_audit.py) lists every operator a module runs
and gives per-module L2 verdicts: use it to find which operators need this skill.
The [evidence table](references/operator-recipes.md#evidence-behind-each-recipe)
says which recipes are verified and which are untested. Treat untested recipes
as hypotheses, and verify them before relying on them.

## Levels: say which one you deliver

| Level | Holds when | Typical breakers |
|---|---|---|
| L1 run-to-run | Same input, same bits on every run | atomics, races, uninitialized reads |
| L2 batch-invariant | A row's result does not depend on batch size, its position, or what the other rows hold | shape-dependent tiles or dispatch, split-K, dynamic batch-wide scales |
| L3 mode-invariant | Prefill, decode, chunked prefill and paged layouts give the same bits per position | split-KV, different block iteration, different kernels per mode, algebraically equal but different forms |
| L4 cross-engine | The trainer and the inference engine give the same bits | different rounding points, fusion, kernels or formulas, even when each side is L2/L3 |
| L5 parallel-invariant | Different TP/EP/SP degrees give the same bits | sharded reductions, collective trees, expert combine order |

L1 does not imply L2. Measured on one GPU type with torch 2.5.1: `torch.mm` in bf16
was bit-identical run to run, yet a row's result changed once the batch reached
31 rows (4 of 4096 elements differed). L2 and L3 do not imply L4. Both engines
must also round at the same places (see the RL part of the handbook).

## The rule

Every output element's floating-point reduction must be a fixed function of that
element's own reduction length. That covers the order of operations, their
grouping, the hardware instruction and the precision at every step. It must not
depend on the number of rows or tokens, the row's position, other rows' values,
grid size, SM count or occupancy, page or chunk boundaries decided by the
scheduler, or timing.

## 1. Locate the variance

Read the kernel's launch and dispatch code, then run `bi_check.check_batch`, and
`check_modes` for attention-like ops. Look for:

- tile, warp or stage configuration chosen from M, or autotuning keyed on shape;
- split-K, stream-K or persistent scheduling, whose K split depends on tile count;
- split-KV (flash-decoding) with the number of splits derived from batch or SM count;
- shape-dispatched fast paths: a GEMV for M=1, a small-M kernel, library heuristics;
- argument-dispatched backends, e.g. an attention call with an explicit mask
  taking a different backend than `is_causal=True`;
- float atomics: `atomicAdd`, `index_add_`/`scatter_add_` into shared outputs;
- a row split across blocks when there are few rows;
- statistics over the whole batch: per-tensor dynamic amax for FP8, a dynamic
  global scale for NVFP4, MoE capacity limits, top-k over the batch;
- collectives whose algorithm or protocol depends on message size;
- alignment-dependent vector width, when a row's start alignment depends on its
  position in the batch;
- compiler or graph layers that specialize per shape: `torch.compile` buckets,
  one CUDA graph per padded batch size.

The failure pattern points to the cause. The table in
[verification.md](references/verification.md#reading-failures) maps patterns to
likely mechanisms.

## 2. Choose the cheapest strategy that holds

1. **Configure.** Use knobs the library exposes: one split (`num_splits=1`),
   split-K off, a pinned algorithm, a deterministic backward, a fixed collective
   algorithm and protocol. Then re-verify.
2. **Select** an existing batch-invariant implementation, e.g. Thinking Machines'
   `batch_invariant_ops` or vLLM's `VLLM_BATCH_INVARIANT=1` set. Re-verify on your
   own shapes, dtype and GPU: coverage differs by operator.
3. **Pin one configuration** for every M. Remove the shape key from autotuning and
   accept lower utilization at small M.
4. **Modify the kernel** following the recipe for its class.
5. **Choose one algorithm for every mode** when two algebraically equal forms
   exist: chunked against recurrent linear attention, absorbed against explicit
   MLA. State the cost: it can be large (2–3x trainer throughput was reported for
   recurrent-only GDN forward).
6. **Fall back** when the kernel is closed and has no knobs. Replace it, or keep
   a slow row-by-row wrapper as a correctness oracle for debugging only, and
   exclude the op from parity claims.

## 3. Keep the arithmetic identical to what you must match

Batch invariance must not move rounding points. Before editing, write down the
reference sequence from the engine you must match:
- where casts happen;
- what is fused and what is a separate op;
- the accumulator dtype;
- epilogue order: bias, activation, residual, norm weight.

Keep it unless both sides change together. A batch-invariant kernel that rounds
differently from the other engine passes L2 and still fails L4.

## 4. Verify

Run the full matrix in [verification.md](references/verification.md):
- batch sizes, including threshold neighbours;
- row positions;
- filler content;
- modes;
- repeated runs;
- backward, when training uses it;
- the deployed wrappers (`torch.compile`, CUDA-graph capture sizes).

Comparison is bitwise through integer views. Zero mismatches is a pass;
anything else fails, with no tolerance.

## 5. Integrate

- Register it so compilers cannot re-fuse it: a `torch.library` override or an
  opaque custom op. Measured: called directly, Triton kernels made Dynamo
  recompile per batch size. Norms and activations left to inductor kept L2 but
  changed 65% of elements against eager. To match eager, make every op that
  rounds opaque.
- Gate it behind a flag, default off unless the user decides otherwise.
- For L4, make both engines call the same implementation, or kernels proven to
  share the reduction tree. Matching a real engine is the
  [engine-parity skill](../engine-parity/SKILL.md).
- Measure throughput at representative prefill M and at decode M (1–64), and
  record the cost next to the baseline.

## 6. Report

Return a handbook entry with:
- operator and shapes;
- level delivered;
- source of variance;
- strategy and change;
- verification matrix and result, with GPU and versions;
- throughput cost;
- known limits, such as sizes not tested or modes not covered.

If an agent wrote or edited the kernel, record the iterations and the human edits
needed. Public reports rarely state this.

## Guardrails

- Never claim invariance from a few sizes. List what was tested, and include
  sizes around every threshold you found in the code.
- Never round or post-process outputs to force equality, and never use a tolerance.
- Deterministic modes (`torch.use_deterministic_algorithms`) give L1, not L2,
  and only with `warn_only=False`. With `warn_only=True`, SDPA's backward only
  warns and stays non-deterministic (measured on one GPU type). Deterministic is also
  not a written order: strict-mode scatters and the embedding backward change
  order with row width or index count. For cross-framework agreement, define
  the order yourself.
- Test sums with at least three terms: MoE top-k ≥ 3, three or more ranks. A
  two-term sum commutes, so it hides order bugs. Measured: an atomic combine
  and a per-rank pre-reduce both passed at top-2 and failed at top-3; DP
  gradients handed to NCCL failed from 3 ranks, even under Tree.
- Test scales and factors that are not powers of two: a power-of-two scale is
  exact anywhere it is applied and hides rounding-point differences.
- Vary the layout under test: page sizes, pack offsets, message sizes. A kernel
  whose blocks follow the page or the pack agrees with itself at any one
  setting and changes between settings.
- Pinning a collective's algorithm is not invariance. Check a fixed slice
  inside messages of many sizes. Measured: Ring pinned under every protocol
  still failed.
- Deterministic is not parallel-invariant. Check L5 separately when the
  parallel degree can differ between the runs you compare. DP invariance needs
  a fixed micro-batch count and a fixed tree that includes the cross-rank sum.
  An L5 fix in the forward does not fix the backward: every column-parallel
  layer's input gradient is a sum over ranks.
- Check the first iteration and a resume, not only steady state: torch DDP
  reduces its first iteration (and every restart) in a single bucket, which
  changes gradient bits under Ring.
- Pin the optimizer implementation (fused, foreach or single-tensor) and the
  loss reduction: each is different arithmetic, and mean is not
  layout-invariant.
- A random stream is part of the op. Check replay after recompute, rollback,
  other requests' n, speculative chunk sizes and a new process, not only batch
  size. `torch.cumsum` is neither batch-invariant nor, for long rows,
  deterministic: use integer sums.
- Check the whole model in every serving mode, and compare its rounding
  sequence with the trainer's. L2/L3 kernels with one different rounding point
  moved 86–100% of logits.
- An engine's batch-invariant flag does not cover every op on every GPU.
  Measured: on some GPU architectures vLLM 0.19.1's mode leaves the lm_head on
  cuBLAS, which breaks its L2 at 7% of logprobs. Check which ops it covers on yours.
- Probing inside `torch.compile` can change what you probe. Give the probe an
  exact pointwise copy (`tools/graph_probe.py`, `isolate=True`) and confirm by
  compiling with and without probes. Build lazy caches (RoPE tables) in eager
  before compiling: built inside a graph, they come from Inductor's math and are
  kept for later eager runs.
- An algorithmic difference (two forms of the same math) is a choice, not a
  bug. Report it as a choice with its cost.
- Leave production defaults unchanged. Parity modes are opt-in.
