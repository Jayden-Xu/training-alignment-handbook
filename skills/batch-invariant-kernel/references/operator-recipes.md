# Operator recipes

Each recipe explains what makes an operator batch-variant and how to fix it.
The level a recipe reaches uses the scale defined in
[SKILL.md](../SKILL.md#levels-say-which-one-you-deliver). Every recipe ends with
the same verification ([verification.md](verification.md)), because reading the
code alone never proves that a fix works.

## Evidence behind each recipe

The single-GPU cases were measured on one GPU type. The multi-GPU cases
(collectives, expert parallelism, DP gradients, the whole model under TP) ran
on a second GPU type, on one or two nodes of 8 GPUs each. Inside a node the
GPUs are connected by PCIe, and the nodes talk over 25 GbE sockets without
RDMA. Everything ran with torch 2.5.1 and Triton 3.1, except where a row says
otherwise (the cross-engine row ran on vLLM's own stack: torch 2.10.0, Triton
3.6.0). Each case keeps its hardware and raw results in its
`examples/<case>/REPORT.md`. The case studies are:
- [MoE](../../../examples/moe_combine/REPORT.md#one-gpu);
- [MoE under expert parallelism](../../../examples/moe_ep/REPORT.md#under-expert-parallelism);
- [attention, FP8 and linear attention](../../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3);
- [the attention backward](../../../examples/attention_backward/REPORT.md#backward-dq-dk-dv);
- [paged KV, varlen packing, compile and CUDA graphs](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs);
- [MLA and sparse attention](../../../handbook/3-kernels/mla-and-sparse.md);
- [linear attention (GDN)](../../../handbook/3-kernels/linear-attention.md);
- [collectives and TP](../../../handbook/3-kernels/collectives.md);
- [the whole model](../../../handbook/2-rl/whole-model.md) and [the whole model under TP](../../../handbook/2-rl/parallelism.md);
- [gradients under DP and FSDP-style sharding (emulated: flat gradient, reduce-scatter, sharded AdamW)](../../../handbook/1-pretraining/data-parallel.md);
- [sampling](../../../handbook/2-rl/sampling.md);
- [LoRA and batches that mix adapters](../../../handbook/2-rl/lora.md);
- [the training step](../../../handbook/1-pretraining/training-step.md);
- [the trainer against vLLM](../../../handbook/2-rl/engine-parity-vllm.md).

| # | Class | Status | Evidence | Still open |
|---|---|---|---|---|
| 1 | Row-wise reductions | observed in stock torch | An RMSNorm built from torch ops and an fp32 log_softmax are L2 at hidden 4096 and vocab 32000 for batches of 1–2048. In fp32, torch's mean and sum change their reduction tree with the row count (from 8 rows at hidden 1024, from 2 rows at 4096), yet the bf16 RMSNorm output did not change. A Triton row sum with a fixed tree is L2 for 1–2048 rows and widths 8–4096 (whole-model case). | The split of one row across blocks that happens with few rows has not been reproduced. LayerNorm and cross-entropy are not covered yet. |
| 2 | Dense GEMM | **built and verified** | A Triton GEMM with one configuration is L2 at M=1–2048 on 3 shapes. cuBLAS is batch-variant on all 3. | Tuning, large shapes, fused epilogues, pinning cuBLASLt, and Hopper. |
| 3 | Grouped GEMM, `bmm` vs `mm` | **built (as a loop) and verified** | A one-config GEMM per expert is L2 on both GPU types ([one GPU](../../../examples/moe_combine/REPORT.md), [8 GPUs](../../../examples/moe_ep/REPORT.md)). The cuBLAS expert GEMMs are batch-variant on both, and so is `bmm` over fixed pieces of rows. | A grouped kernel that runs in a single launch. |
| 4 | Softmax attention | **built and verified** (forward, backward, paged, varlen) | A fixed-block Triton kernel is L2 and L3 across prefill, decode and chunks, while SDPA fails L2 (batch ≥ 3) and L3. The atomic-free backward is L1 and L2, right padding included, and costs 1.38x SDPA flash. SDPA flash fails L1 (in dQ), and in strict deterministic mode it fails L2. Paged KV (pages of 16–128) and varlen packing (forward and backward) keep the result bit for bit when blocks follow each sequence's absolute positions, but not when blocks follow the page or the pack offset. | Split-KV with a fixed split size, sliding windows, and the paged backward. |
| 5 | MLA, absorbed vs explicit | **built and verified** | The two forms differ in 65% of elements while being equally accurate. One form with every piece invariant is L2 and L3. Explicit prefill combined with absorbed decode stops matching a trainer from layer 2 on. An invariant attention core alone is not enough: the decompression GEMM and the output `bmm` must be invariant too. | FlashMLA, split-KV for the absorbed form, the backward, and L4 against an engine. |
| 6 | Sparse attention (top-k) | **built and verified** | Every selection flip (at about a quarter of positions) came from the cuBLAS projection GEMM. With a one-config projection and a stable top-k, the selected sets and the outputs are L2 and L3. `torch.topk` took the lowest index among ties in every test, but that order is not documented. | DSA at full scale (top-2048), an FP8 indexer, and the backward. |
| 7 | Linear attention / SSM | **built and verified** (GDN forward) | The recurrent kernel is L1–L3. The chunked kernel is L1/L2, and it is L3 only when chunk boundaries sit at absolute multiples of C. With bf16 matmuls the chunked form differs from the recurrent one in 55.5% of outputs; with fp32 matmuls, in 0.03%. The recurrent form costs 0.9–1.5x the chunked one at T ≤ 4096 and 2.5–3.6x at 8192. | The backward, production chunked kernels (fla), and varlen. |
| 8 | MoE routing and combine | **built and verified** (on 1 GPU, and at EP 1–16 on two nodes of 8 GPUs; [report](../../../examples/moe_ep/REPORT.md)) | A fixed-order fp32 combine with one-config GEMMs is L1 and L2. Under EP, a fixed-slot combine gives bit-identical results at EP 1, 2, 4, 8 and 16, whether on one node or split across two (L5). Pre-reducing on each rank is L1/L2 but fails L5 from top-3. An atomic combine breaks L1 (from top-3 in the EP case). Dropping tokens at a capacity limit fails L2 and L5. | The backward under EP, a shared expert, DeepEP-style kernels, NVLink, RDMA, more than two nodes, and L4 against a trainer. |
| 9 | Quantized GEMM | scales verified (emulated) | Dynamic per-tensor scales fail L2; per-token scales and 1x128 block scales pass. | The promotion interval of a real FP8 GEMM (on GPUs with FP8 tensor cores), and NVFP4 (on GPUs with FP4 tensor cores). |
| 10 | Collectives | **measured** (on one and two nodes of 8 GPUs; [report](../../../examples/collectives_tp/REPORT.md)) | All-reduce fails L2 with the defaults, and also with Ring pinned under every protocol. `NCCL_ALGO=Tree` passes on one node (at 1.0–1.3x, and 5.3x at 1 MB) and on two nodes (16 ranks: each node sums along its chain, then the two node sums are added once). Reduce-scatter's order follows the rank that owns the row, so TP with sequence parallelism fails. An all-to-all followed by an fp32 sum in rank order passes at 1.04–1.20x on one node. Across nodes, a two-level fixed order (1.06–1.8x) replaces the flat fallbacks (2.1–8.8x). A logical TP size gives L5: a size of 8 holds across TP 1–8, and 16 holds across TP 1–16 over two nodes. In a whole model, logical TP (C = 16) plus a fixed-order vocab log_softmax gives one result at TP 1–16, and also when prefill and decode run at different TP. For DP gradients, the bucket layout changes the bits under Ring (Tree, or an ordered all-reduce at 1.05–1.20x, fixes it). Logical shards (S=16, fixed tree) give identical gradients, norms and AdamW weights after 10 steps at DP 1–16, on one or two nodes, replicated or sharded, and across an elastic resume. | Three or more nodes, NVLink/NVLS, RDMA, torch.distributed.fsdp itself, and the TP backward beyond activation gradients. |
| 11 | Closed kernels | measured | Where cuBLAS and SDPA vary has been mapped. A cuBLAS loop over fixed pieces works as an oracle, about 33x slower. SDPA's memory-efficient backward in strict deterministic mode passed L1 and L2 at the shapes tested, at 4.4x the cost of flash. | cuDNN and the FlashAttention binaries. |
| 12 | Deployment wrappers | **measured** | A compiled block stayed L2 but differed from eager in 65% of elements until every op that rounds was made an opaque custom op. Triton kernels called directly were recompiled for every batch size (18 graphs, against 2 when registered as custom ops). CUDA-graph buckets keep invariant kernels invariant, while cuBLAS gave 6 different results over 7 buckets. Probes inside compiled code and graphs (`tools/graph_probe.py`) leave the compiled bits unchanged when they are isolated. | Compiling the paged, varlen and backward paths, and `max-autotune`. |
| 13 | Sampling | **built and verified** | Per-request generators are L2 but fail under recompute, rollback, n>1 and speculative chunks. cumsum fails L2 (and L1 at V=151,936). A sampler keyed by a counter, with int64 weights, passes L1–L3 and a chi-square test at 1.3x the stock sampler's cost (B=256). | A fused noise kernel, min-p, compile and CUDA graphs, and real engines. |
| 14 | LoRA with many adapters | **built and verified** (forward, backward) | Grouped cuBLAS fails L2 in 486–506/723 co-batching cases, and still in 22–100/723 with an invariant base GEMM. A gathered bmm fails as well. Merging in bf16 changes ~50% of the bits and loses 18–78% of ΔW when the update is 1e-2–1e-3 of the output. Moving the point where the scale is applied changes 5–6% of the outputs. Our kernel passes L1, L2, prefill against decode, trainer against engine, and the backward with padding. | vLLM itself, packed slices, TP, compile, and QLoRA. |
| 15 | Training step (backward ops, optimizer, loss, resume) | **built and verified** (1 GPU) | index_add_, scatter_add_, the index_select backward, weighted bincount and 1-D cumsum fail L1. Sorting by index and then taking an ascending fp32 segmented sum is L1 and matches a NumPy loop bit for bit. Strict mode raises an error on cumsum, bincount(weights) and histc. The fused, foreach and single-tensor AdamW paths differ from each other (foreach and single-tensor only in the bias-correction division). Saving and restoring 7 pieces of state gives a bitwise resume. | Matching the fused optimizer's order exactly (it is off by 1 ulp in 0.15% of exp_avg), multi-GPU (see row 10), and compile. |
| 16 | Cross-engine rounding (L4) | **measured** (vLLM 0.19.1, [report](../../../examples/cross_engine/REPORT.md)) | Packing, bias placement, the norm form, the RoPE table, the attention kernel, the lm_head kernel and the log_softmax kernel each break equality on their own. The RoPE formula and SiLU were already equal. A trainer that calls vLLM's kernels gives 0/76,155 differing tokens. | Compiled vLLM, the backward, TP, and other models. |

## 1. Row-wise reductions: RMSNorm, LayerNorm, softmax, log_softmax, row sums, cross-entropy

A row-wise reduction becomes batch-variant when the kernel:
- splits one row across several blocks when there are few rows, to fill the GPU;
- chooses the block size or the vector width from the row count or from
  pointer alignment;
- takes a fast path for small batches.

To fix it:
- Give each row its own program (a block or a warp), whatever the batch size.
- Fix the block size along the row and the way each thread accumulates, then
  combine the threads with a fixed tree inside the block.
- If a row is too long for one block, split it into chunks whose number depends
  only on the row length, and combine the chunks in a fixed order in a second
  pass. Never use atomics.
- Fix the vector width. If the alignment of a row's start address varies with
  its position in the batch (because the hidden size is not a multiple of the
  vector width), use the scalar path everywhere, or pad the row stride.
- Keep the reference's rounding points: for example, whether the normalized
  value is cast to bf16 before or after it is multiplied by the weight.

This recipe reaches L2. On one GPU type with torch 2.5.1, at hidden 4096 and
vocab 32000, an RMSNorm built from torch ops and an fp32 `log_softmax` were
already L2 across batch sizes 1–2048. Nothing guarantees this for other shapes,
versions or GPUs, so verify it on yours.

## 2. Dense GEMM (bf16/fp16)

A dense GEMM becomes batch-variant when:
- the tile, the instruction or the kernel is chosen from M;
- it uses split-K or stream-K;
- a library heuristic picks the kernel.

On one GPU type with torch 2.5.1, `torch.mm` (cuBLAS) was L2 up to M=16 and
batch-variant from M=31 on.

To fix it:
- Use one tile configuration (block M/N/K, warps, stages, MMA instruction) for
  every M.
- Use no split-K and no stream-K. The K loop runs in ascending order into an
  fp32 accumulator, and every M uses the same epilogue.
- A second kernel for small M is allowed only if it reproduces the same
  per-element K chunking and the same instruction. Prove bit for bit that it
  matches the main kernel.
- cuBLASLt can pin an algorithm and its split-K setting. Verify across M
  anyway, because a pinned algorithm is not a guarantee.
- Turn TF32 off, unless both sides use it in exactly the same way.

This recipe reaches L2. For L4, both engines must also run the same kernel. The
cost is lower utilization at small M, that is, in decode.

## 3. Batched and grouped GEMM: MoE experts, `bmm` vs `mm`

- An expert's GEMM sees M_e, the number of tokens routed to that expert, and
  M_e changes with whatever else is in the batch. So each expert GEMM must be
  invariant in M_e (recipe 2). Grouped kernels often size their tiles from the
  largest or the total M; pin the tile size instead.
- When one side calls `bmm` on a 3-D tensor and the other calls `mm` on a 2-D
  tensor, the two sides run different kernels. One reported case is the MoE
  router: different gate scores can send a token to different experts, which is
  a discrete divergence. Route both sides through the same invariant GEMM.

## 4. Softmax attention: prefill, decode, chunked prefill, paged KV

On one GPU type with torch 2.5.1, SDPA varied in two ways:
- prefill and per-token decode differed in about 10% of output elements;
- chunked prefill with an explicit mask differed in 1 element. The masked call
  probably took a different backend from `is_causal=True`.

To fix it, a query at position t must be computed the same way in every mode:
- **Same key iteration everywhere.** Visit KV in blocks of a fixed size B, in
  absolute position order starting from 0, and apply the online-softmax update
  block by block. Prefill, chunked prefill and decode use the same B and the
  same order.
- **No split-KV** (`num_splits=1`). If you must split, split by a fixed number
  of tokens aligned to absolute positions, never by a number of splits, and use
  the same split and merge in prefill as in decode.
- **Paged KV:** block boundaries follow absolute positions (multiples of B),
  whatever the page size and wherever a prefill chunk starts. Write the current
  chunk's K/V into the cache before the kernel runs, so that the kernel reads
  everything from one uniform source.
- **Masking:** masked scores contribute exactly zero. Partially masked blocks
  compute the running max and the normalizer in the same way in every mode.
- Every mode uses the same exponent path (for example `exp2` with a log2(e)
  prescale), the same scale and the same GQA head mapping.
- With variable-length packing (`cu_seqlens`), a sequence's iteration must not
  depend on its neighbours in the pack.
- **Backward:** common kernels add up the dQ contributions from the key-block
  programs with atomics, which breaks L1. Use two passes without atomics
  instead:
  - dK/dV: one program per key block walks the query heads of its GQA group,
    then the query blocks, in a fixed order;
  - dQ: one program per query block walks the key blocks from position 0.

  Both passes accumulate in fp32 registers and write once. Sum over the GQA
  group inside the program, in fp32. Store the forward's log-sum-exp from a
  variant of the forward, and check bitwise that this variant's output equals
  the plain forward's. Measured in the
  [attention backward case](../../../examples/attention_backward/REPORT.md#backward-dq-dk-dv):
  - L1 and L2 hold, including right padding with a masked loss;
  - forward plus backward costs 1.38x SDPA flash.

  A vendor's deterministic backward needs `warn_only=False`, and it is no more
  invariant than its forward: flash still failed L2 in that mode.

This recipe reaches L2 and L3. The cost is that, without split-KV, long-context
decode with few sequences runs slower.

Measured on one GPU type ([paged, varlen and compile case](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)):
- **Paged KV.** At page sizes 16–128 with pages placed at random, every mode
  reproduced the contiguous result bit for bit. Computing in blocks of the page
  size instead gave a different result for each page size. Each of those
  results agreed with itself across modes, so a test with a single page size
  cannot catch this: test several. A page smaller than the block needs a
  gather, which cost +4% at `num_stages=1`.
- **Varlen packing.** When blocks start at each sequence's own start, forward
  and backward gave identical results alone and packed. A grid aligned to the
  pack offset failed whenever the offset was not a multiple of the block size.

## 5. MLA (latent attention)

Implementations often use an absorbed form for decode, with the weights folded
into the query and output projections, and an explicit K/V form for prefill.
The two forms are equal in exact arithmetic but round differently. This is an
algorithmic difference, so choose one form and use it in every mode; for L4,
use it in the trainer too. Then apply recipe 4 to the attention core, and state
the throughput cost of the form you chose.

Measured on one GPU type ([MLA and sparse case](../../../handbook/3-kernels/mla-and-sparse.md)):
- **The forms differ.** They disagree in 65% of output elements, and both are
  equally accurate against fp64.
- **One form is invariant.** When every GEMM, the norm and the attention core
  are invariant, each form on its own is L2 and L3.
- **Mixing forms breaks agreement with the trainer.** Explicit prefill with
  absorbed decode stopped matching a trainer from the second layer on,
  whichever form the trainer used. A model with one layer can match by
  accident, because the latent cache does not depend on the form.
- **The core alone is not enough.** An invariant attention core with cuBLAS
  projections still failed. The explicit form's decompression, whose M is the
  number of cached tokens, and the per-head output `bmm` must be invariant too.
- **Costs.**
  - Absorbed prefill costs 1.77x explicit prefill (B=4, T=1024).
  - Explicit decode that decompresses K/V again at every step costs 6.9–27x
    absorbed decode at B=16–64.
  - Explicit decode with a per-head K/V cache costs 1.17–3.0x, but the cache
    takes 7.2x the memory per token.

## 6. Sparse attention with selection (indexer plus top-k)

In this operator a batch-variant score becomes a discrete flip: the kernel
selects a different set of keys, and the output diverges by a large amount.

- Make the indexer's scores invariant with recipes 1–2.
- Use a top-k with a deterministic tie-break: order by score first, then by
  index.
- Accumulate over the selected keys in sorted index order, following recipe 4.
- Check that the selected sets are identical across batches before you compare
  values.

Measured on one GPU type ([MLA and sparse case](../../../handbook/3-kernels/mla-and-sparse.md)):
- **The flips came from the projection GEMM.** Under chunked prefill and
  batched decode, about a quarter of positions flipped their selection, and
  every flip came from the cuBLAS projection GEMM. A stable top-k alone, or an
  invariant indexer alone, removed none of them. With a one-config projection,
  everything passed.
- **Keep a stable sort anyway.** `torch.topk` chose the lowest index among ties
  in every test, but that order is not documented. The stable sort cost an
  extra 0.87 ms at T=2048.
- **Compare sets before values.** Most of the baseline's value mismatches were
  at positions whose selected set had not changed.

## 7. Linear attention and SSMs (GDN, Mamba-style)

The chunked form, used in training and prefill, and the recurrent form, used in
decode, are different algorithms with different reduction orders. A recurrent
kernel that handles one sequence at a time is batch-invariant by nature,
because it never reduces across sequences.

There are two options:
- run the recurrent forward everywhere and use the chunked form only for the
  backward. This was reported to reach a logprob difference of 0 against vLLM,
  at a cost of 2–3x in trainer throughput;
- or accept the difference, and report it as a choice rather than a fault.

A chunked kernel is L2 as long as the chunk size and the order inside each
chunk are fixed.

Measured on one GPU type with a gated-delta-rule layer ([linear attention case](../../../handbook/3-kernels/linear-attention.md)):
- **Recurrent.** A recurrent Triton kernel (one program per sequence, head and
  value tile) is L1, L2 and L3 with no extra work.
- **Chunked.** A chunked kernel is L1 and L2. It is L3 only if every call
  places its chunks at absolute multiples of C. Decode can recompute the open
  chunk from the state at the last boundary and match a full prefill bit for
  bit, at 3.4–3.7x the cost of a recurrent decode step.
- **Precision matters more than the algorithm.** With fp32 matmul operands, the
  chunked and recurrent forms differ in 0.03% of bf16 outputs. With bf16
  operands, which common chunked kernels use, they differ in 55.5%.
- **The recurrent cost grows with T.** Against an untuned chunked kernel, the
  recurrent forward cost 0.9–1.5x at T ≤ 4096 and 2.5–3.6x at T = 8192.
- **Configurations differ too.** Two recurrent tile configurations give
  different results from each other, so the trainer and the engine must share
  one.

## 8. MoE routing, dispatch and combine

- **Router:** use an invariant GEMM (recipes 2–3), a row-wise softmax or sigmoid
  (recipe 1), and a stable top-k (by score, then by expert id). Keep the
  reference's dtype for the routing weights: one reported mismatch came only
  from casting them to bf16.
- **No batch-dependent dropping.** A capacity limit makes a token's fate depend
  on its neighbours. Use dropless routing, or a capacity rule that does not
  depend on other tokens.
- **Dispatch** only moves data, so it is exact. Padding rows in the expert
  segments must not change the real rows, and they do not when the expert GEMM
  is invariant.
- **Combine:** for each token, sum its top-k expert outputs in a fixed order (by
  top-k rank or by expert id) and at a fixed precision, with one program per
  token. Never scatter-add or atomically index-add into the output. Under expert
  parallelism, put the returned rows into fixed slots before summing, so that
  the order in which they arrive cannot matter. Add the shared expert at a fixed
  position in that sum.
- **Measured** ([MoE case study](../../../examples/moe_combine/REPORT.md#one-gpu), one GPU type):
  - cuBLAS expert GEMMs broke L2 from 2 tokens;
  - the deterministic mode fixed only L1;
  - an atomic combine broke L1 even with invariant GEMMs;
  - one-config Triton GEMMs with a fixed-order fp32 combine per token passed L1
    and L2, at about +11% over the same GEMMs with an atomic combine;
  - a batched matmul over fixed-size pieces was not invariant.
- **Measured under expert parallelism** ([EP case](../../../examples/moe_ep/REPORT.md#under-expert-parallelism),
  one and two nodes of 8 GPUs):
  - the fixed-slot combine gave bit-identical results at EP 1, 2, 4, 8 and 16,
    on one node and across two, wherever the GPUs were placed (L5). No NCCL
    setting was needed, because the all-to-alls contain no reduction;
  - summing on each rank before returning the rows was L1 and L2 but failed L5
    from top-3, because the grouping follows which experts share a rank. Its
    mismatches depended on the EP degree, not on where the ranks sat;
  - at top-2 every order gives the same bits, so test with top-k ≥ 3;
  - with capacity dropping, which tokens survived changed with the neighbours,
    the EP degree or the local order.

## 9. Quantized GEMM (FP8 block-scaled, MXFP8, NVFP4)

- **Activation scales must not couple rows.** Use per-token, per-token-block
  (for example 1x128) or static scales, never a per-tensor amax computed over
  the batch. The same applies to NVFP4's second-level scale: make it per token
  or static.
- The quantization kernel is itself a row-wise reduction (amax), so it follows
  recipe 1.
- The GEMM follows recipe 2 and also fixes the promotion interval: how many MMA
  steps accumulate in tensor-core precision before the partial result is added
  into the fp32 accumulator. Apply the scales at fixed points in a fixed order,
  and use no split-K.
- Before you check the GEMM output, verify that the quantized values themselves
  (data and scales) are identical for each row across batches.

## 10. Collectives (TP / EP / SP)

Measured on 8 GPUs (one PCIe node, NCCL 2.21.5, [collectives case](../../../handbook/3-kernels/collectives.md)):

- **Pinning the algorithm and protocol is not enough.** With Ring pinned under
  every protocol, a fixed slice of rows was still reduced in a different order
  inside messages of different sizes. Each element's start rank follows the
  chunk boundaries, and those boundaries move with the message size and offset.
  Row-parallel TP 4 and 8 failed L2 from M = 2.
- **An algorithm with a single order works on one node.** On that node,
  `NCCL_ALGO=Tree` reduced every element in the same order at every size and
  passed L2. It cost 1.0–1.3x all-reduce, and 5.3x at 1 MB.
- **It also held across two nodes.** With 16 ranks over 25 GbE, pinned Tree
  still passed. Each element is summed along each node's chain, and then the
  two node sums are added once; a two-operand add commutes. Three or more nodes
  will likely break this (not tested), so verify on your topology.
- **Reduce-scatter has one order per output chunk, and that is not enough.**
  The order depends on which rank owns the row. Under sequence parallelism,
  ownership depends on position and batch, so TP with SP failed under every
  setting.
- **Fallback.** Exchange the partial sums (with all-to-all or all-gather) and
  add them locally in rank order in fp32. This is invariant by construction. The
  all-to-all form cost 1.04–1.20x all-reduce from 1 MB on. It changes the
  arithmetic, so for L4 both engines must use it.
- **Across nodes, use two levels.** There the flat fallbacks cost 2.1–8.8x
  all-reduce. A two-level fixed order was just as invariant at 1.06–1.8x: sum
  inside each node in rank order, then do one fp32 exchange between nodes.
- **Two ranks are order-free**, because a sum of two operands commutes. Test
  with 3 or more.
- **A different parallel degree means a different tree.** At TP 2–8,
  row-parallel results differed from unsharded ones in 37.5–59% of elements. A
  logical TP size reaches L5:
  - split K into a fixed number of chunks (8 here);
  - compute each chunk's partial sum with the same kernel;
  - gather the partial sums and add them in chunk order.

  TP 1, 2, 4 and 8 then agreed bit for bit, at 1.6–1.9x the cost of
  row-parallel at TP 8 in tree form. With 16 chunks, TP 1, 2, 4, 8 and 16
  agreed across two nodes.
- **Use the same chunk count everywhere.** Logical TP 8 and logical TP 16 give
  different results. Logical TP 8 could run at TP 16 only by splitting each
  chunk's output columns; splitting K instead changed the result.
- **Whole model** (case whole-model-tp): logical TP on every row-parallel layer
  (C = 16), plus a vocab-parallel log_softmax with fixed chunk sums, gave one
  result at TP 1, 2, 4, 8 and 16, and also with prefill at TP a and decode at
  TP b. In the backward, the input gradients of column-parallel layers are also
  a sum across ranks, so reduce them the same way (NCCL all-reduce broke L5 at
  all 9 points checked). The cost was 1.4–1.6x for prefill and 1.6–1.9x for
  decode at TP 2–8 on 8 GPUs over PCIe. Logical TP is different arithmetic from
  a single GPU (84.7% of logits differ at TP 1), so the trainer and the engine
  must both use it.
- **Data-parallel gradients** (case training-dp-fsdp, two nodes of 8 GPUs):
  under Ring, the bucket layout changes the bits, and torch DDP reduces its
  first iteration (and every restart) in a single bucket. Tree or an ordered
  all-reduce fixes the packing problem (1.05–1.20x). Reduce-scatter's order
  follows the owning rank. For L5 across DP degrees, split the batch into a
  fixed number of logical shards (16 here), compute each shard's gradient on
  its own, and sum them in a fixed tree that includes the cross-rank sum.
  Handing that cross-rank sum to NCCL failed from 3 ranks, even under Tree.

## 11. Closed or precompiled kernels

- Map the region where the kernel is invariant by running `check_batch` over
  many sizes. Padding to a fixed M helps only if every call really runs at that
  M. CUDA-graph buckets break this, because each bucket runs at its own M.
- If no knob works and no invariant region covers your sizes, replace the
  kernel. If you cannot replace it, keep a row-by-row wrapper as a debugging
  oracle, and exclude the op from parity claims.

## 12. Deployment wrappers: `torch.compile` and CUDA graphs

Measured on one GPU type ([paged, varlen and compile case](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)):
- **Register invariant kernels as opaque custom ops.** Called directly, the
  Triton kernels made Dynamo recompile at every batch size: 18 graphs, even
  with `dynamic=True`. Registered as custom ops, they needed 2.
- **Compile can keep L2 and still change the bits.** Inductor fused RMSNorm,
  the residual add and SiLU·up, and the compiled block then differed from eager
  in 65% of elements. Making every op that rounds opaque gave compiled = eager.
  An engine that must match an eager trainer needs this.
- **CUDA graphs keep invariance only with invariant kernels.** With padding to
  buckets, cuBLAS gave the same sequence 6 different results across 7 buckets.
  Graph replay itself reproduced eager's bits at the same shape.
- **Custom ops cost host time:** about 0.65 ms per call at small batch here,
  unless they run inside a graph.
- **Fingerprints inside compiled code** ([compile-graph-capture case](../../../handbook/1-pretraining/capture-compiled-code.md)):
  `tools/graph_probe.py` records OpGuard-format fingerprints from compiled code
  and graph replays. Give the probe an exact pointwise copy (`isolate=True`),
  because a user that is not pointwise can switch Inductor to `addmm` and
  change the bits. Build lazy caches (RoPE tables) in eager before compiling.

## 13. Sampling

Generate the noise with a counter-based generator keyed by (seed, absolute
position, purpose, sample index), never from a generator's running state. Make
the distribution exact with integer weights floor(exp(z−max)·2^S), so that sums
and cumsums do not depend on order. Break ties by (value desc, index asc) with a
stable sort, so that greedy decoding picks the lowest index. If tokens must
match decoding without a draft model, verify speculative decoding by checking
that its tokens equal the target model's own samples. This was measured on one
GPU type ([sampling case](../../../handbook/2-rl/sampling.md); the reference
sampler is [tools/bi_sampling.py](../../../tools/bi_sampling.py)).

## 14. LoRA and multi-adapter batches

Inside the kernel, select each row's adapter for that row; never regroup rows
by adapter for a reduction. Run the base GEMM with one configuration. Do the
shrink and expand steps in a fixed rank and K order, keep the LoRA update in
fp32, and add it to the base result in one fp32 FMA, followed by a single cast.

The trainer (one adapter) and the engine (a mix of adapters) must use the same
kernel. Pin the point where the scale is applied. Pad smaller ranks with zeros
inside the fp32 accumulation. Compute the trainer's log-probs with dropout off.
Do not merge the adapter into the weights in bf16 when the engine must match a
trainer that keeps it unmerged.

In the backward, compute dX with the same kernel, and compute dA/dB per adapter
over the rows that have a nonzero gradient, in ascending order, without
atomics. Measured on one GPU type ([LoRA case](../../../handbook/2-rl/lora.md)).

## 15. Training step: backward scatters, optimizer, loss, resume

- Replace `index_add_`, `scatter_add_`, the embedding backward and weighted
  `bincount` with an order you write yourself: sort by index, then take an
  ascending fp32 segmented sum. Strict mode alone does not give a written
  order, because its scatters and the embedding backward change order with the
  row width or the index count.
- Pin the optimizer path (fused, foreach or single-tensor) and the loss
  reduction. A mean is not layout-invariant.
- Save and restore every piece of state that a step reads; the case lists 7.
- Measured on one GPU type ([training step case](../../../handbook/1-pretraining/training-step.md));
  the multi-GPU gradients are covered in recipe 10.

## 16. Cross-engine rounding (L4)

Even when each side is L2 and L3, the two sides need not agree, so L4 needs its
own work. Write down both rounding sequences (packing, bias placement, norm
form, RoPE table, and the attention, lm_head and log_softmax kernels). Check the
engine's own L1 and L2 first, then align one rounding point at a time. The
[engine-parity skill](../../engine-parity/SKILL.md) describes the workflow. In
the [trainer vs vLLM case](../../../handbook/2-rl/engine-parity-vllm.md),
measured with vLLM 0.19.1 on one GPU type, equality was reached only when the
trainer called vLLM's own kernels.

## Sources for reported numbers

- Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025). It
  shows that missing batch invariance is the root cause of nondeterminism in
  inference, and gives invariant RMSNorm, matmul and attention.
- Yichuan Wang, "Closing the Training–Inference Mismatch for Linear Attention in
  RL". It uses the recurrent GDN forward and a batch-invariant `bmm` for the MoE
  router, at 2–3x trainer cost.
- verl-omni issue #637. It matches vLLM to HF numerics one op at a time (norm
  cast order, residual-add precision, RoPE formula, router weight cast); the
  reported rollout throughput was about 37%.
- IsoExec (SkyRL, vLLM blog). It describes an execution contract shared across
  frameworks, and parallel-invariant kernels.

The handbook's [references](../../../handbook/references.md) page has the
links. The observations on this page come from `tools/bi_check.py --demo`,
whose results are in a demo JSON under `tools/`, and from the case studies,
each of which keeps its script and results JSON under `examples/`.
