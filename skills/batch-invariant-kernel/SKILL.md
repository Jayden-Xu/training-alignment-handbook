---
name: batch-invariant-kernel
description: Write, adapt or verify a batch-invariant GPU kernel when no ready-made one exists. The work is to fix the reduction tree, to remove split-K, split-KV, atomics and dispatch that depends on the shape, and to give MoE combine and quantized GEMM a stable order; then to prove that the result stays bitwise identical across batch sizes, row positions, the content of the other rows, prefill and decode, and chunking. Use it when a model brings an operator that has no batch-invariant implementation, such as a new attention variant (MLA, sparse/DSA, linear/GDN), MoE dispatch and combine, an FP8/FP4 GEMM, or a closed-source kernel.
---

# Batch-invariant kernel

Work on one operator at a time. You start from an operator whose invariance is
unknown or known to be broken, and you finish with the operator verified
invariant at a stated level, or with a documented fallback. Return four things:
the change (a configuration, a replacement or an edit to the kernel), the
verification report, the throughput cost, and a handbook entry. Picking a
ready-made batch-invariant op is step 2 below; this skill is for the cases
where that is not enough.

[references/operator-recipes.md](references/operator-recipes.md) has a recipe
for each class of operator.
[references/verification.md](references/verification.md) describes the
verification protocol and how to read failures.
The checker is [tools/bi_check.py](../../tools/bi_check.py).
Two reference kernels to start from are in [tools/bi_kernels.py](../../tools/bi_kernels.py),
verified on one GPU type:
- a GEMM with a single configuration;
- fixed-block causal attention with a backward that uses no atomics.

More verified kernels come with the case studies, under `examples/`:
- paged and varlen attention;
- GDN in recurrent and in chunked form;
- the core of MLA in absorbed form;
- a sparse-attention pipeline with a stable top-k;
- an expert-parallel combine that puts rows into fixed slots;
- a GEMM with fp32 output for a logical TP size, and a vocab-parallel
  log_softmax that adds in a fixed order;
- a LoRA linear that selects each row's adapter;
- a sampler keyed by a counter ([tools/bi_sampling.py](../../tools/bi_sampling.py));
- a segmented sum for backward scatters that sorts by index and adds in
  ascending order;
- data-parallel gradients computed over logical shards.

[tools/bi_audit.py](../../tools/bi_audit.py) lists every operator a module runs
and gives an L2 verdict for each module. Use it to find the operators that need
this skill. The [evidence table](references/operator-recipes.md#evidence-behind-each-recipe)
says which recipes have been verified and which are untested. Treat an untested
recipe as a hypothesis, and verify it before you rely on it.

## Say which invariance level you deliver {#levels-say-which-one-you-deliver}

| Level | Holds when | What usually breaks it |
|---|---|---|
| L1 run-to-run | The same input gives the same bits on every run | Atomics, races, reads of uninitialized memory |
| L2 batch-invariant | A row's result does not depend on the batch size, on the row's position, or on what the other rows hold | Tiles or dispatch chosen from the shape, split-K, dynamic scales computed over the whole batch |
| L3 mode-invariant | Prefill, decode, chunked prefill and paged layouts give the same bits at each position | Split-KV, a different block iteration or a different kernel in each mode, forms that are algebraically equal but computed differently |
| L4 cross-engine | The trainer and the inference engine give the same bits | Different rounding points, fusions, kernels or formulas, even when each side is L2/L3 on its own |
| L5 parallel-invariant | Different TP/EP/SP degrees give the same bits | Reductions split across shards, the tree a collective uses, the order in which expert outputs are combined |

L1 does not imply L2. On one GPU type with torch 2.5.1, `torch.mm` in bf16 gave
bit-identical results from run to run, yet a row's result changed once the batch
reached 31 rows (4 of its 4096 elements differed). L2 and L3 do not imply L4
either: the two engines must also round at the same places (see the RL part of
the handbook).

## The rule

The floating-point reduction that produces each output element must be decided
by that element's own reduction length and by nothing else. "Decided" covers
the order of the operations, how they are grouped, the hardware instruction and
the precision at every step. None of these may depend on the number of rows or
tokens, the row's position, the values in other rows, the grid size, the SM
count or occupancy, page or chunk boundaries chosen by the scheduler, or
timing.

## 1. Find where the result starts to depend on the batch

Read the kernel's launch and dispatch code, then run `bi_check.check_batch`,
and also `check_modes` for attention-like ops. Look for:

- a tile, warp or stage configuration chosen from M, or autotuning keyed on the
  shape;
- split-K, stream-K or persistent scheduling, where the K split depends on the
  tile count;
- split-KV (flash-decoding), where the number of splits is derived from the
  batch size or the SM count;
- fast paths dispatched by shape: a GEMV for M=1, a kernel for small M, library
  heuristics;
- backends dispatched by argument; for example, an attention call with an
  explicit mask can take a different backend from a call with `is_causal=True`;
- float atomics: `atomicAdd`, and `index_add_`/`scatter_add_` into shared
  outputs;
- one row split across several blocks when there are few rows;
- statistics over the whole batch: a dynamic per-tensor amax for FP8, a dynamic
  global scale for NVFP4, MoE capacity limits, a top-k over the batch;
- collectives whose algorithm or protocol depends on the message size;
- a vector width that depends on alignment, when the alignment of a row's start
  depends on the row's position in the batch;
- compiler or graph layers that specialize for each shape: `torch.compile`
  buckets, or one CUDA graph per padded batch size.

The pattern of the failures points to the cause. The table in
[verification.md](references/verification.md#reading-failures) maps each
pattern to its likely mechanism.

## 2. Choose the cheapest strategy that holds

1. **Configure.** Use the knobs the library exposes: a single split
   (`num_splits=1`), split-K turned off, a pinned algorithm, a deterministic
   backward, a fixed collective algorithm and protocol. Then verify again.
2. **Select** an existing batch-invariant implementation, such as Thinking
   Machines' `batch_invariant_ops` or the set that vLLM's
   `VLLM_BATCH_INVARIANT=1` turns on. Verify it again on your own shapes, dtype
   and GPU, because coverage differs from one operator to another.
3. **Pin one configuration** for every M. Take the shape out of the autotuning
   key, and accept lower utilization at small M.
4. **Modify the kernel**, following the recipe for its class.
5. **Choose one algorithm for every mode** when two algebraically equal forms
   exist, such as chunked and recurrent linear attention, or absorbed and
   explicit MLA. State the cost, which can be large: running the GDN forward
   only in recurrent form was reported to cost 2–3x in trainer throughput.
6. **Fall back** when the kernel is closed and has no knobs. Replace it, or keep
   a slow row-by-row wrapper as a correctness oracle for debugging only, and
   leave the op out of every parity claim.

## 3. Keep the arithmetic identical to what you must match

Making a kernel batch-invariant must not move its rounding points. Before you
edit it, write down the reference sequence, that is, how the engine you must
match computes the op:
- where the casts happen;
- which ops are fused and which run separately;
- the accumulator dtype;
- the order of the epilogue: bias, activation, residual, norm weight.

Keep that sequence unless both sides change together. A batch-invariant kernel
that rounds differently from the other engine passes L2 and still fails L4.

## 4. Verify

Run the full matrix in [verification.md](references/verification.md), which
covers:
- batch sizes, including the neighbours of every threshold;
- row positions;
- the content of the other rows (filler);
- modes;
- repeated runs;
- the backward, when training uses it;
- the wrappers used in deployment (`torch.compile`, CUDA-graph capture sizes).

Compare bits through integer views. Zero mismatches is a pass and anything else
is a failure; there is no tolerance.

## 5. Integrate

- Register the kernel so that compilers cannot fuse it again: use a
  `torch.library` override or an opaque custom op. Measured: when Triton
  kernels were called directly, Dynamo recompiled for every batch size. Norms
  and activations left to Inductor stayed L2 but changed 65% of elements
  compared with eager. To match eager, make every op that rounds opaque.
- Put the kernel behind a flag that is off by default, unless the user decides
  otherwise.
- For L4, make both engines call the same implementation, or kernels proven to
  share the same reduction tree. Matching a real engine is the subject of the
  [engine-parity skill](../engine-parity/SKILL.md).
- Measure throughput at a representative prefill M and at decode M (1–64), and
  record the cost next to the baseline.

## 6. Report

Return a handbook entry with:
- the operator and its shapes;
- the level delivered;
- where the variance came from;
- the strategy and the change;
- the verification matrix and its result, with the GPU and the versions;
- the throughput cost;
- known limits, such as sizes not tested or modes not covered.

If an agent wrote or edited the kernel, record how many iterations it took and
which human edits were needed. Public reports rarely state this.

## Guardrails

- Never claim invariance from a few sizes. List what you tested, and include
  the sizes around every threshold you found in the code.
- Never round or post-process outputs to force equality, and never use a
  tolerance.
- Deterministic modes (`torch.use_deterministic_algorithms`) give L1, not L2,
  and only with `warn_only=False`. With `warn_only=True`, SDPA's backward only
  warns and stays nondeterministic (measured on one GPU type). Deterministic
  also does not mean that the order is written down anywhere: in strict mode,
  scatters and the embedding backward change their order with the row width or
  the index count. If you need agreement across frameworks, define the order
  yourself.
- Test sums of at least three terms: MoE top-k ≥ 3, and three or more ranks. A
  two-term sum commutes, so it hides order bugs. Measured: an atomic combine
  and a per-rank pre-reduce both passed at top-2 and failed at top-3, and DP
  gradients handed to NCCL failed from 3 ranks, even under Tree.
- Test scales and factors that are not powers of two. A power-of-two scale is
  exact wherever it is applied, so it hides differences in rounding points.
- Vary the layout under test: page sizes, pack offsets, message sizes. A kernel
  whose blocks follow the page or the pack agrees with itself at any one
  setting and changes between settings.
- Pinning a collective's algorithm does not make it invariant. Check a fixed
  slice inside messages of many sizes. Measured: Ring pinned under every
  protocol still failed.
- Deterministic does not mean parallel-invariant. Check L5 separately when the
  parallel degree can differ between the runs you compare. DP invariance needs
  a fixed number of micro-batches and a fixed tree that includes the sum across
  ranks. An L5 fix in the forward does not fix the backward, because the input
  gradient of every column-parallel layer is a sum over ranks.
- Check the first iteration and a resume, not only the steady state. torch DDP
  reduces its first iteration (and every restart) in a single bucket, which
  changes the gradient bits under Ring.
- Pin the optimizer implementation (fused, foreach or single-tensor) and the
  loss reduction. Each choice is different arithmetic, and a mean is not
  layout-invariant.
- A random stream is part of the op. Check replay after recompute and
  rollback, with other requests' n, with other speculative chunk sizes and in a
  new process, not only across batch sizes. `torch.cumsum` is not
  batch-invariant, and for long rows it is not even deterministic: use integer
  sums.
- Check the whole model in every serving mode, and compare its rounding
  sequence with the trainer's. L2/L3 kernels with a single different rounding
  point moved 86–100% of the logits.
- An engine's batch-invariant flag does not cover every op on every GPU.
  Measured: on some GPU architectures, vLLM 0.19.1's mode leaves the lm_head on
  cuBLAS, which breaks L2 for 7% of logprobs. Check which ops it covers on
  your GPU.
- Probing inside `torch.compile` can change what you probe. Give the probe an
  exact pointwise copy (`tools/graph_probe.py`, `isolate=True`), and confirm by
  compiling with and without the probes. Build lazy caches (RoPE tables) in
  eager before compiling: if they are built inside a graph, they come from
  Inductor's math and are kept for later eager runs.
- An algorithmic difference (two forms of the same math) is a choice, not a
  bug. Report it as a choice, with its cost.
- Leave production defaults unchanged. Parity modes are opt-in.
