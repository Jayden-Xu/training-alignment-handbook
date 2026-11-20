# Operator recipes

Each recipe states what makes the operator batch-variant and what fixes it. The
level it reaches is from [SKILL.md](../SKILL.md#levels-say-which-one-you-deliver).
Every recipe ends in the same verification ([verification.md](verification.md));
none is proven by reading the code alone.

## Evidence behind each recipe

Measurements: one GPU type for single-GPU cases. Multi-GPU cases (collectives,
expert parallelism, DP gradients, the whole model under TP) ran on one or two
nodes of 8 GPUs each, of a second type: PCIe inside a node, 25 GbE
sockets between nodes, no RDMA. All with torch 2.5.1 and Triton
3.1. Exceptions are named in the row (the cross-engine row ran vLLM's own
stack: torch 2.10.0, Triton 3.6.0). The hardware and raw results of each case
are in its `examples/<case>/REPORT.md`. Case studies:
- [MoE](../../../examples/moe_combine/REPORT.md#one-gpu);
- [MoE under expert parallelism](../../../examples/moe_ep/REPORT.md#under-expert-parallelism);
- [attention, FP8, linear attention](../../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3);
- [attention backward](../../../examples/attention_backward/REPORT.md#backward-dq-dk-dv);
- [paged KV, varlen, compile and CUDA graphs](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs);
- [MLA and sparse attention](../../../handbook/3-kernels/mla-and-sparse.md);
- [linear attention (GDN)](../../../handbook/3-kernels/linear-attention.md);
- [collectives and TP](../../../handbook/3-kernels/collectives.md);
- [whole model](../../../handbook/2-rl/whole-model.md) and [under TP](../../../handbook/2-rl/parallelism.md);
- [DP and FSDP-style (emulated: flat gradient, reduce-scatter, sharded AdamW) gradients](../../../handbook/1-pretraining/data-parallel.md);
- [sampling](../../../handbook/2-rl/sampling.md);
- [LoRA and multi-adapter batches](../../../handbook/2-rl/lora.md);
- [training step](../../../handbook/1-pretraining/training-step.md);
- [trainer vs vLLM](../../../handbook/2-rl/engine-parity-vllm.md).

| # | Class | Status | Evidence | Still open |
|---|---|---|---|---|
| 1 | Row-wise reductions | stock observed | torch-op RMSNorm and fp32 log_softmax are L2 at hidden 4096 and vocab 32000, batch 1–2048; torch mean/sum in fp32 changes its tree with the row count (hidden 1024 from 8 rows, 4096 from 2) while bf16 RMSNorm output did not change; a fixed-tree Triton row sum is L2 at rows 1–2048 and widths 8–4096 (whole-model case) | the few-rows split case not reproduced; LayerNorm, cross-entropy |
| 2 | Dense GEMM | **built and verified** | one-config Triton GEMM is L2 at M=1–2048 on 3 shapes; cuBLAS is variant on all 3 | tuning, large shapes, fused epilogues, cuBLASLt pinning, Hopper |
| 3 | Grouped GEMM, `bmm` vs `mm` | **built (loop) and verified** | per-expert one-config GEMM is L2 on both GPU types ([one GPU](../../../examples/moe_combine/REPORT.md), [8 GPUs](../../../examples/moe_ep/REPORT.md)); cuBLAS expert GEMMs are variant on both; `bmm` over fixed row pieces is variant | a single-launch grouped kernel |
| 4 | Softmax attention | **built and verified** (forward, backward, paged, varlen) | fixed-block Triton is L2 and L3 across prefill, decode and chunks; SDPA fails L2 (batch ≥ 3) and L3. Atomic-free backward is L1 and L2, right padding included, at 1.38x SDPA flash; SDPA flash fails L1 (dQ) and, in strict deterministic mode, L2. Paged KV (pages 16–128) and varlen packing (forward and backward) keep the bits when blocks follow each sequence's absolute positions; blocks that follow the page or the pack offset do not | fixed-size split-KV, sliding window, paged backward |
| 5 | MLA absorbed vs explicit | **built and verified** | the forms differ in 65% of elements at equal accuracy; one form with every piece invariant is L2 and L3; explicit prefill with absorbed decode no longer matches a trainer from layer 2; an invariant core alone is not enough (decompression GEMM, output `bmm`) | FlashMLA, split-KV for the absorbed form, backward, L4 against an engine |
| 6 | Sparse attention (top-k) | **built and verified** | every selection flip came from the cuBLAS projection GEMM (about a quarter of positions); with a one-config projection and a stable top-k, the selected sets and outputs are L2 and L3; `torch.topk` took the lowest index among ties in every test, an undocumented order | DSA scale (top-2048), FP8 indexer, backward |
| 7 | Linear attention / SSM | **built and verified** (GDN forward) | recurrent kernel is L1–L3; chunked is L1/L2, and L3 only with chunk boundaries at absolute multiples of C; bf16 chunked matmuls differ from recurrent in 55.5% of outputs, fp32 ones in 0.03%; recurrent costs 0.9–1.5x chunked at T ≤ 4096, 2.5–3.6x at 8192 | backward, production chunked kernels (fla), varlen |
| 8 | MoE routing and combine | **built and verified** (1 GPU; EP 1–16 on two nodes of 8 GPUs, [report](../../../examples/moe_ep/REPORT.md)) | fixed-order fp32 combine with one-config GEMMs is L1 and L2; under EP a fixed-slot combine gives the same bits at EP 1, 2, 4, 8 and 16, on one node or split across two (L5); a per-rank pre-reduce is L1/L2 and fails L5 from top-3; atomic combine breaks L1 (from top-3 in the EP case); capacity dropping fails L2 and L5 | backward under EP, shared expert, DeepEP-style kernels, NVLink, RDMA, more than two nodes, L4 against a trainer |
| 9 | Quantized GEMM | scales verified (emulated) | per-tensor dynamic scales fail L2; per-token and 1x128 pass | real FP8 GEMM promotion interval (on GPUs with FP8 tensor cores), NVFP4 (on GPUs with FP4 tensor cores) |
| 10 | Collectives | **measured** (nodes of 8 GPUs, one and two nodes; [report](../../../examples/collectives_tp/REPORT.md)) | all-reduce fails L2 with defaults and with Ring pinned under every protocol; `NCCL_ALGO=Tree` passes on one node (1.0–1.3x, 5.3x at 1 MB) and on two (16 ranks: each node's chain, then one two-operand add); reduce-scatter's order follows the owning rank, so TP with sequence parallelism fails; all-to-all plus an fp32 rank-order sum passes at 1.04–1.20x on one node; across nodes a two-level fixed order (1.06–1.8x) replaces flat fallbacks (2.1–8.8x); a logical TP size gives L5: 8 across TP 1–8, 16 across TP 1–16 over two nodes; in a whole model, logical TP (C = 16) plus a fixed-order vocab log_softmax gives one result at TP 1–16 and across prefill/decode at different TP. DP gradients: bucket layout changes bits under Ring (Tree and an ordered all-reduce at 1.05–1.20x fix it); logical shards (S=16, fixed tree) give identical gradients, norms and 10-step AdamW weights at DP 1–16 on one or two nodes, replicated or sharded, and across elastic resume | three or more nodes, NVLink/NVLS, RDMA, torch.distributed.fsdp itself, TP backward beyond activation gradients |
| 11 | Closed kernels | measured | cuBLAS and SDPA variance mapped; a fixed-piece cuBLAS loop works as an oracle (about 33x slower); SDPA memory-efficient backward in strict deterministic mode passed L1 and L2 at the shapes tested (4.4x flash) | cuDNN, FlashAttention binaries |
| 12 | Deployment wrappers | **measured** | a compiled block stayed L2 but differed from eager in 65% of elements until every rounding op was an opaque custom op; Triton kernels called directly recompiled per batch size (18 graphs; 2 as custom ops); CUDA-graph buckets keep invariant kernels invariant, while cuBLAS gave 6 results over 7 buckets; probes inside compiled code and graphs (`tools/graph_probe.py`) leave the compiled bits unchanged when isolated | compile of the paged, varlen and backward paths; `max-autotune` |
| 13 | Sampling | **built and verified** | per-request generators are L2 but fail recompute, rollback, n>1 and speculative chunks; cumsum fails L2 (and L1 at V=151,936); a counter-keyed sampler with int64 weights passes L1–L3 and chi-square at 1.3x stock (B=256) | fused noise kernel, min-p, compile/graphs, real engines |
| 14 | LoRA / multi-adapter | **built and verified** (forward, backward) | grouped cuBLAS fails L2 (486–506/723), with an invariant base still 22–100/723; gathered bmm fails; bf16 merge changes ~50% of bits and drops 18–78% of ΔW at update/output 1e-2–1e-3; scale placement changes 5–6%; ours passes L1, L2, prefill/decode, trainer vs engine, backward padding | vLLM itself, packed slices, TP, compile, QLoRA |
| 15 | Training step (backward ops, optimizer, loss, resume) | **built and verified** (1 GPU) | index_add_/scatter_add_/index_select bwd/weighted bincount/1-D cumsum fail L1; a sort-plus-ascending fp32 segmented sum is L1 and matches a NumPy loop bitwise; strict mode raises on cumsum, bincount(weights), histc; AdamW fused/foreach/single differ (foreach vs single only in the bias-correction division); 7 pieces of state give a bitwise resume | fused optimizer order exact (1 ulp in 0.15% of exp_avg), multi-GPU (see row 10), compile |
| 16 | Cross-engine rounding (L4) | **measured** (vLLM 0.19.1, [report](../../../examples/cross_engine/REPORT.md)) | packing, bias placement, norm form, RoPE table, attention kernel, lm_head kernel and log_softmax kernel each break equality; RoPE formula and SiLU already equal; trainer calling vLLM's kernels gives 0/76,155 | compiled vLLM, backward, TP, other models |

## 1. Row-wise reductions: RMSNorm, LayerNorm, softmax, log_softmax, row sums, cross-entropy

Variance comes from:
- splitting one row across blocks when there are few rows, to fill the GPU;
- block size or vector width chosen from the row count or from pointer alignment;
- a fast path for small batches.

Fix:
- One row per program (block or warp), whatever the batch size.
- Fixed block size along the row; fixed per-thread accumulation, then a fixed
  in-block tree.
- If a row is too long for one block, split it into chunks whose number depends
  only on the row length, and combine them in a fixed order in a second pass.
  Never use atomics.
- Fixed vector width. If the row's start address alignment varies with its
  position in the batch (hidden size not a multiple of the vector width), use the
  scalar path everywhere, or pad the row stride.
- Keep the reference rounding points: e.g. whether the normalized value is cast
  to bf16 before or after multiplying by the weight.

Reaches L2. Observed on one GPU type with torch 2.5.1, hidden 4096 and vocab 32000: a
torch-op RMSNorm and an fp32 `log_softmax` were already L2 across batch sizes
1–2048. That is not guaranteed for other shapes, versions or GPUs, so verify.

## 2. Dense GEMM (bf16/fp16)

Variance comes from:
- tile, instruction or kernel chosen from M;
- split-K or stream-K;
- library heuristics.

Observed on one GPU type with torch 2.5.1: `torch.mm` (cuBLAS) was L2 up to M=16 and
variant from M=31 on.

Fix:
- One tile configuration (block M/N/K, warps, stages, MMA instruction) for every
  M.
- No split-K, no stream-K. The K loop runs in ascending order into an fp32
  accumulator, with the same epilogue.
- A second, small-M kernel is allowed only if it reproduces the same per-element
  K chunking and instruction. Prove it bitwise against the main kernel.
- cuBLASLt can pin an algorithm and its split-K setting. Still verify across M:
  a pinned algorithm is not a guarantee.
- TF32 off, unless both sides use it identically.

Reaches L2. L4 needs both engines on the same kernel. Cost: lower utilization at
small M (decode).

## 3. Batched and grouped GEMM: MoE experts, `bmm` vs `mm`

- An expert's GEMM sees M_e, the number of tokens routed to it, and that changes
  with what else is in the batch. Each expert GEMM must be invariant in M_e
  (recipe 2). Grouped kernels often size tiles from the largest or total M: pin
  them.
- One side calling `bmm` on a 3-D tensor and the other calling `mm` on 2-D runs
  different kernels. One reported case is the MoE router: different gate scores
  can route a token to different experts, a discrete divergence. Route both
  through the same invariant GEMM.

## 4. Softmax attention: prefill, decode, chunked prefill, paged KV

Observed on one GPU type with torch 2.5.1 SDPA:
- prefill against per-token decode differed in about 10% of output elements;
- chunked prefill with an explicit mask differed in 1 element. The masked call
  likely took a different backend from `is_causal=True`.

Fix, for a query at position t:
- **Same key iteration everywhere.** KV is visited in blocks of a fixed size B,
  in absolute position order from 0, with the online-softmax update applied
  block by block. Prefill, chunked prefill and decode use the same B and the same
  order.
- **No split-KV** (`num_splits=1`). Or split by a fixed size in tokens, aligned
  to absolute positions, never by a number of splits. Then use the same split
  and merge in prefill as in decode.
- **Paged KV:** block boundaries follow absolute positions (multiples of B),
  independent of page size and of where a prefill chunk starts. Write the
  current chunk's K/V into the cache before the kernel runs, so the kernel reads
  one uniform source.
- **Masking:** masked scores contribute exactly zero. Partially masked blocks
  compute the running max and normalizer the same way in every mode.
- The same exponent path (e.g. `exp2` with a log2(e) prescale), the same scale,
  and the same GQA head mapping in every mode.
- Variable-length packing (`cu_seqlens`): a sequence's iteration does not depend
  on its neighbours in the pack.
- **Backward:** common kernels add dQ contributions from key-block programs with
  atomics, which breaks L1. Use two atomic-free passes:
  - dK/dV: one program per key block walks the GQA group's query heads, then
    the query blocks, in a fixed order;
  - dQ: one program per query block walks key blocks from position 0.

  Both accumulate in fp32 registers and write once. Sum the GQA group inside the
  program in fp32. Store the forward's log-sum-exp as a variant whose output is
  checked bitwise against the plain forward. Measured in the
  [attention backward case](../../../examples/attention_backward/REPORT.md#backward-dq-dk-dv):
  - L1 and L2, including right padding with a masked loss;
  - 1.38x SDPA flash forward+backward.

  A vendor deterministic backward needs `warn_only=False`, and gives no more
  invariance than its forward: flash failed L2 that way.

Reaches L2 and L3. Cost: losing split-KV hurts long-context decode with few
sequences.

Measured on one GPU type ([paged, varlen and compile case](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)):
- **Paged KV.** At page sizes 16–128 with random page placement, the contiguous
  result came back bitwise in every mode. Computing in blocks of the page size
  instead gave a different result for every page size. Each result agreed with
  itself across modes, so test several page sizes. A page smaller than the
  block needs a gather; at `num_stages=1` it cost +4%.
- **Varlen packing.** Forward and backward were identical alone and packed when
  blocks start at each sequence. A grid aligned to the pack offset failed
  whenever the offset was not a multiple of the block.

## 5. MLA (latent attention)

Implementations often use an absorbed form for decode (weights folded into the
query/output projections) and an explicit K/V form for prefill. They are equal
in exact arithmetic and round differently. That is an algorithmic difference:
use one form in every mode, and do the same in the trainer for L4. Then apply
recipe 4 to the attention core. State the throughput cost of the form chosen.

Measured on one GPU type ([MLA and sparse case](../../../handbook/3-kernels/mla-and-sparse.md)):
- **The forms differ.** They disagree in 65% of output elements, with the same
  accuracy against fp64.
- **One form is invariant.** With every GEMM, the norm and the attention core
  invariant, each form alone is L2 and L3.
- **Mixed forms break trainer agreement.** Explicit prefill with absorbed decode
  no longer matched a trainer from the second layer on, whichever form the
  trainer used. With one layer it can match by accident, because the latent
  cache does not depend on the form.
- **The core alone is not enough.** An invariant attention core with cuBLAS
  projections still failed. The explicit form's decompression, whose M is the
  number of cached tokens, and the per-head output `bmm` must be invariant too.
- **Costs.**
  - Absorbed prefill: 1.77x explicit (B=4, T=1024).
  - Explicit decode that re-decompresses K/V every step: 6.9–27x absorbed at
    B=16–64.
  - Explicit decode with a per-head K/V cache: 1.17–3.0x, but the cache holds
    7.2x the memory per token.

## 6. Sparse attention with selection (indexer plus top-k)

A batch-variant score turns into a discrete flip: a different selected key set,
which is a large divergence.

- Make the indexer's scores invariant with recipes 1–2.
- Top-k with a deterministic tie-break: score first, then index.
- Accumulate over the selected keys in sorted index order with recipe 4.
- Check that the selected sets are identical across batches before checking
  values.

Measured on one GPU type ([MLA and sparse case](../../../handbook/3-kernels/mla-and-sparse.md)):
- **The flips came from the projection GEMM.** Every selection flip, at about a
  quarter of positions under chunked prefill and batched decode, came from the
  cuBLAS projection GEMM. A stable top-k or an invariant indexer alone removed
  none of them. With a one-config projection everything passed.
- **Keep a stable sort anyway.** `torch.topk` chose the lowest index among ties
  in every test, but that order is not documented. The stable sort cost
  0.87 ms extra at T=2048.
- **Compare sets before values.** Most baseline value mismatches sat at
  positions whose set had not changed.

## 7. Linear attention and SSMs (GDN, Mamba-style)

The chunked form (training, prefill) and the recurrent form (decode) are
different algorithms with different reduction orders. A per-sequence recurrent
kernel is naturally batch-invariant: it has no cross-sequence reduction.

Options:
- recurrent forward everywhere, with chunked only for backward. This was
  reported to reach logprob difference 0 against vLLM, at 2–3x trainer
  throughput;
- or accept the difference, and report it as a choice rather than a fault.

A chunked kernel is L2 when the chunk size and the in-chunk order are fixed.

Measured on one GPU type with a gated-delta-rule layer ([linear attention case](../../../handbook/3-kernels/linear-attention.md)):
- **Recurrent.** A recurrent Triton kernel (one program per sequence, head and
  value tile) is L1, L2 and L3 with no extra work.
- **Chunked.** A chunked kernel is L1 and L2. It is L3 only if every call's
  chunks sit at absolute multiples of C. Decode can recompute the open chunk
  from the last boundary state and match full prefill bitwise, at 3.4–3.7x a
  recurrent decode step.
- **Precision matters more than the algorithm.** With fp32 matmul operands,
  chunked and recurrent differ in 0.03% of bf16 outputs. With bf16 operands, as
  common chunked kernels use, they differ in 55.5%.
- **Recurrent cost grows with T.** Against an untuned chunked kernel, the
  recurrent forward cost 0.9–1.5x at T ≤ 4096 and 2.5–3.6x at T = 8192.
- **Configurations differ too.** Two recurrent tile configurations differ from
  each other, so trainer and engine must share one.

## 8. MoE routing, dispatch and combine

- **Router:** an invariant GEMM (recipes 2–3), a row-wise softmax or sigmoid
  (recipe 1), stable top-k (score, then expert id). Keep the reference's dtype
  for routing weights: one reported mismatch came only from casting them to
  bf16.
- **No batch-dependent dropping.** Capacity limits make a token's fate depend on
  its neighbours. Use dropless routing, or a capacity rule that does not depend
  on other tokens.
- **Dispatch** only moves data, so it is exact. Padding rows in expert segments
  must not change real rows; they do not when the expert GEMM is invariant.
- **Combine:** for each token, sum its top-k expert outputs in a fixed order (by
  top-k rank or expert id) and at a fixed precision, in one program per token.
  Never scatter-add or atomically index-add into the output. Under expert
  parallelism, place returned rows into fixed slots before summing, so arrival
  order cannot matter. Add the shared expert in a fixed position in that sum.
- **Measured** ([MoE case study](../../../examples/moe_combine/REPORT.md#one-gpu), one GPU type):
  - cuBLAS expert GEMMs broke L2 from 2 tokens;
  - the deterministic mode fixed only L1;
  - an atomic combine broke L1 even with invariant GEMMs;
  - one-config Triton GEMMs with a per-token fixed-order fp32 combine passed L1
    and L2, at about +11% over the same GEMMs with an atomic combine;
  - a batched matmul over fixed-size pieces was not invariant.
- **Measured under expert parallelism** ([EP case](../../../examples/moe_ep/REPORT.md#under-expert-parallelism),
  one and two nodes of 8 GPUs):
  - the fixed-slot combine gave the same bits at EP 1, 2, 4, 8 and 16, on one
    node and across two, whatever the GPU placement (L5). No NCCL setting was
    needed, because the all-to-alls contain no reduction;
  - summing per rank before returning was L1 and L2, but failed L5 from top-3:
    the grouping follows which experts share a rank. Its mismatches depended on
    the EP degree, not on where the ranks sat;
  - at top-2 any order gives the same bits, so test with top-k ≥ 3;
  - capacity dropping changed which tokens survived with the neighbours, the EP
    degree, or the local order.

## 9. Quantized GEMM (FP8 block-scaled, MXFP8, NVFP4)

- **Activation scales must not couple rows.** Use per token, per token-block
  (e.g. 1x128) or static scales, never a per-tensor amax computed over the batch.
  The same applies to NVFP4's second-level scale: per token or static.
- The quantization kernel is itself a row-wise reduction (amax): recipe 1.
- The GEMM follows recipe 2, plus a fixed promotion interval: how many MMA steps
  accumulate in tensor-core precision before adding into the fp32 accumulator.
  Apply scales at fixed points in a fixed order. No split-K.
- Verify that the quantized values themselves (data and scales) are identical
  per row across batches before checking the GEMM output.

## 10. Collectives (TP / EP / SP)

Measured on 8 GPUs (one PCIe node, NCCL 2.21.5, [collectives case](../../../handbook/3-kernels/collectives.md)):

- **Pinning the algorithm and protocol is not enough.** With Ring pinned under
  every protocol, a fixed slice of rows still got a different reduction order
  inside messages of different sizes. Each element's start rank follows the
  chunk boundaries, and those move with message size and offset. Row-parallel
  TP 4 and 8 failed L2 from M = 2.
- **A single-order algorithm works on one node.** On that node,
  `NCCL_ALGO=Tree` gave one order for every element at every size and passed
  L2. It cost 1.0–1.3x all-reduce, and 5.3x at 1 MB.
- **It also held across two nodes.** With 16 ranks over 25 GbE, pinned Tree
  still passed. Each element is each node's chain, then one add of the two
  node sums, and a two-operand add commutes. Three or more nodes likely break
  this (not tested): verify on your topology.
- **Reduce-scatter has one order per output chunk, which is not enough.** The
  order depends on which rank owns the row. Under sequence parallelism that
  depends on position and batch, and TP with SP failed under every setting.
- **Fallback.** Exchange partials (all-to-all, or all-gather) and sum them
  locally in rank order in fp32. That is invariant by construction. The
  all-to-all form cost 1.04–1.20x all-reduce from 1 MB on. It changes the
  arithmetic, so both engines must use it for L4.
- **Across nodes, use two levels.** The flat fallbacks cost 2.1–8.8x all-reduce
  there. A two-level fixed order was just as invariant at 1.06–1.8x: sum inside
  each node in rank order, then one fp32 exchange between nodes.
- **Two ranks are order-free**, because a two-operand sum commutes. Test with 3
  or more.
- **A different parallel degree is a different tree.** Row-parallel differed
  from unsharded in 37.5–59% of elements at TP 2–8. A logical TP size reaches
  L5:
  - split K into a fixed number of chunks (8 here);
  - compute each chunk's partial with the same kernel;
  - gather the partials and sum them in chunk order.

  TP 1, 2, 4 and 8 then agreed bitwise, at 1.6–1.9x row-parallel at TP 8 in
  tree form. With 16 chunks, TP 1, 2, 4, 8 and 16 agreed across two nodes.
- **Fix the chunk count everywhere.** Logical TP 8 and 16 differ from each
  other. Logical TP 8 reached TP 16 only by splitting each chunk's output
  columns. Splitting K instead changed the result.
- **Whole model** (case whole-model-tp): logical TP on every row-parallel layer
  (C = 16) plus a vocab-parallel log_softmax with fixed chunk sums gave one
  result at TP 1, 2, 4, 8 and 16 and for prefill at TP a with decode at TP b.
  Column-parallel input gradients in the backward are also a cross-rank sum:
  reduce them the same way (NCCL all-reduce broke L5 at all 9 points checked).
  Cost: 1.4–1.6x prefill, 1.6–1.9x decode at TP 2–8 on 8 GPUs over PCIe. Logical TP is
  a different arithmetic from single-GPU (84.7% of logits differ at TP 1), so
  the trainer and the engine must both use it.
- **Data-parallel gradients** (case training-dp-fsdp, two nodes of 8 GPUs): under Ring, the
  bucket layout changes the bits, and torch DDP reduces its first iteration
  (and every restart) in a single bucket. Tree or an ordered all-reduce fixes
  packing (1.05–1.20x). Reduce-scatter's order follows the owner. For L5 across
  DP degrees, split the batch into a fixed number of logical shards (16 here),
  compute each shard's gradient alone, and sum them in a fixed tree that
  includes the cross-rank sum; handing that sum to NCCL failed from 3 ranks,
  even under Tree.

## 11. Closed or precompiled kernels

- Map the invariant region with `check_batch` over many sizes. Padding to a
  fixed M helps only if every call really runs at that M: CUDA-graph buckets
  break this across buckets.
- If no knob or region holds, replace the kernel. Otherwise keep a row-by-row
  wrapper as a debugging oracle, and exclude the op from parity claims.

## 12. Deployment wrappers: `torch.compile` and CUDA graphs

Measured on one GPU type ([paged, varlen and compile case](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)):
- **Register invariant kernels as opaque custom ops.** Called directly, the
  Triton kernels made Dynamo recompile at every batch size: 18 graphs, even with
  `dynamic=True`. As custom ops: 2.
- **Compile can keep L2 and still change the bits.** Inductor fused RMSNorm, the
  residual add and SiLU·up, and the compiled block differed from eager in 65% of
  elements. Making every op that rounds opaque gave compiled = eager. An engine
  that must match an eager trainer needs that.
- **CUDA graphs keep invariance only with invariant kernels.** With padding to
  buckets, cuBLAS gave the same sequence 6 different results across 7 buckets.
  Graph replay itself gave eager's bits at the same shape.
- **Custom ops cost host time:** about 0.65 ms per call at small batch here,
  unless graphed.
- **Fingerprints inside compiled code** ([compile-graph-capture case](../../../handbook/1-pretraining/capture-compiled-code.md)):
  `tools/graph_probe.py` records OpGuard-format fingerprints from compiled code
  and graph replays. Give the probe an exact pointwise copy (`isolate=True`): a
  non-pointwise user can switch Inductor to `addmm` and change the bits. Build
  lazy caches (RoPE tables) in eager before compiling.

## 13. Sampling

Key the noise by (seed, absolute position, purpose, sample index) with a
counter-based generator, never by a generator's running state. Make the
distribution exact: integer weights floor(exp(z−max)·2^S) so sums and cumsums
are order-free. Order ties by (value desc, index asc) with a stable sort; greedy
picks the lowest index. Verify speculative decoding by equality with the
target's own sample if tokens must match no-draft decoding. Measured on one GPU type
([sampling case](../../../handbook/2-rl/sampling.md); reference sampler
[tools/bi_sampling.py](../../../tools/bi_sampling.py)).

## 14. LoRA and multi-adapter batches

Select each row's adapter inside the kernel; never regroup rows for a
reduction. Run the base GEMM with one configuration, shrink and expand in fixed
rank and K order, keep the update in fp32, and meet the base in one fp32 FMA
before a single cast. Use the same kernel in the trainer (one adapter) and the
engine (mixed adapters); pin where the scale enters; zero-pad ranks inside fp32
accumulation; run trainer log-probs with dropout off; do not merge in bf16 when
the engine must match an unmerged trainer. Backward: dX by the same kernel,
dA/dB per adapter over its nonzero-gradient rows in ascending order, no
atomics. Measured on one GPU type ([LoRA case](../../../handbook/2-rl/lora.md)).

## 15. Training step: backward scatters, optimizer, loss, resume

- Replace `index_add_`, `scatter_add_`, the embedding backward and weighted
  `bincount` with a written order: sort by index, then an ascending fp32
  segmented sum. Strict mode alone is not a written order: its scatters and
  the embedding backward change order with row width or index count.
- Pin the optimizer path (fused, foreach or single-tensor) and the loss
  reduction. Mean is not layout-invariant.
- Save and restore every piece of state a step reads; the case lists 7.
- Measured on one GPU type ([training step case](../../../handbook/1-pretraining/training-step.md));
  multi-GPU gradients are in recipe 10.

## 16. Cross-engine rounding (L4)

L2 and L3 on each side do not give L4. Write down both rounding sequences
(packing, bias placement, norm form, RoPE table, attention, lm_head and
log_softmax kernels), check the engine's own L1 and L2 first, then align one
rounding point at a time. The workflow is the
[engine-parity skill](../../engine-parity/SKILL.md). Measured with vLLM 0.19.1
on one GPU type ([trainer vs vLLM case](../../../handbook/2-rl/engine-parity-vllm.md)):
equality needed the trainer to call vLLM's own kernels.

## Sources for reported numbers

- Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025): batch
  invariance as the root cause of inference nondeterminism; invariant RMSNorm,
  matmul and attention.
- Yichuan Wang, "Closing the Training–Inference Mismatch for Linear Attention in
  RL": GDN recurrent forward, batch-invariant `bmm` for the MoE router, 2–3x
  trainer cost.
- verl-omni issue #637: vLLM matched to HF numerics op by op (norm cast order,
  residual add precision, RoPE formula, router weight cast); about 37% rollout
  throughput.
- IsoExec (SkyRL, vLLM blog): a cross-framework execution contract and
  parallel-invariant kernels.

Links are in the handbook's [references](../../../handbook/references.md). The
observations above come from `tools/bi_check.py --demo` (results in a
demo JSON under `tools/`) and from the case studies, each with its script and
results JSON under `examples/`.
