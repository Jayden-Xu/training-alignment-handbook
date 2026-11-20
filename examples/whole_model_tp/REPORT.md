# Whole model under tensor parallelism and disaggregation (L5): lab report

This is the full record behind the handbook page
[The model under different parallel degrees](../../handbook/2-rl/parallelism.md). The page teaches the step;
this report keeps the setup, every measurement and how the case was built.

## Summary

- A logical TP size on all 16 row-parallel layers plus a fixed-order
  vocab-parallel log_softmax gives bitwise-equal logits and realized log-probs
  at TP 1, 2, 4, 8 and 16 (two 8x A16 nodes).
- L5 composes with L2 and L3: decode in a changing batch and chunked prefill at
  TP 2 and 4 matched the TP 1 trainer forward. Disaggregated prefill/decode at
  different TP adds nothing: the KV hand-over is a copy.
- Stock TP moves every realized log-prob (KL 2–3e-3 mean). The backward needs
  the same fix: an NCCL all-reduce of column-parallel input gradients broke L5.
- **Cost:** 1.4–1.6x for prefill on one node; 1.6–1.9x for decode at TP 2–8.

## How

The [collectives case](../../handbook/3-kernels/collectives.md) showed L5 for one GEMM: a logical TP
size (a fixed number of K chunks, fp32 partials, a fixed summation order)
gives the same bits at every TP degree. The [whole-model case](../../handbook/2-rl/whole-model.md)
checked L1–L3 on one GPU. This case puts the two together. The whole model
runs sharded at TP 1, 2, 4, 8 and 16. Prefill runs at one TP degree and
decode at another, with the KV cache handed over between them. An RL trainer
and its engine often use different TP degrees, and disaggregated serving
does the same between its prefill and decode pools.
### The model and the two variants

`model.py`'s dense `Config()`: hidden 1024, 8 layers, 16 query / 4 KV heads
(head dim 64), MLP 3584, vocab 32000, bf16, 175M parameters, random weights
(seed 0). Every rank builds the same `WholeModel`; `TPModel` slices its
weights Megatron-style, so TP = 1 runs `model.py`'s weights.

| Part | Sharding |
|---|---|
| q/k/v projections | column-parallel, whole heads per rank |
| KV heads at TP 8 and 16 (4 KV heads) | replicated: each rank keeps the KV head its query heads use |
| o_proj, down_proj | row-parallel: one sum over ranks each, 16 per forward |
| gate/up projections | column-parallel |
| lm_head | vocab-parallel (column) |
| log_softmax | vocab-parallel: a max and a sum across ranks |
| embedding, RMSNorm, RoPE | replicated |

| Variant | Row-parallel sum | log_softmax over vocab shards | Other ops |
|---|---|---|---|
| `stock` | each rank's cuBLAS partial rounded to bf16, NCCL all-reduce (bf16, default algorithm) | all-reduce MAX; local `torch.sum` of exp; all-reduce SUM (fp32), as Megatron's vocab-parallel cross entropy | `model.py` stock op set: cuBLAS, SDPA, torch reductions |
| `inv` | logical TP, C = 16 (below) | max by all-reduce MAX (exact in any order); per fixed chunk of 2000 vocab entries a fixed-tree row sum; the 16 chunk sums all-gathered and added with a fixed pairwise tree | `model.py` bi op set: one-config GEMM, fixed-block attention, fixed-tree RMSNorm |

**Logical TP, C = 16.** Every row-parallel GEMM cuts K into 16 fixed chunks
(64 for o_proj, 224 for down_proj), whatever the degree.
1. A rank owns 16/TP chunks. One Triton launch computes their fp32 partials
   (`_mm_chunks_kernel`: grid z = chunk, the K loop of `mm_one_config`).
2. The rank adds its chunks with a fixed pairwise tree: (c0+c1), (c2+c3), ...
   Its chunks are one aligned subtree of the tree over all 16.
3. The ranks exchange fp32 subtrees by column block (all-to-all).
4. Each rank finishes the same tree on its N/TP columns, rounds to bf16 once,
   and all-gathers.

The tree is the same at TP 1 (16 chunks on one GPU) and TP 16 (one per rank).
Column-parallel layers need nothing: each output column is computed whole on
one rank by the same kernel.

**Backward (`inv`).** The input gradient of a column-parallel layer is a sum
over its output columns, spread over the ranks. It uses the same logical-TP
reduction. The chunks are heads for q (16) and k/v (4), 16 for gate/up and 16
for lm_head. The row-parallel input gradient needs no sum. The vocab-parallel
log_softmax backward needs one row sum of the incoming gradient, done like
the forward sum. The negative control sums the column-parallel input
gradients Megatron's way: a bf16 partial per rank, then NCCL all-reduce.

### Where the variance comes from

- **Row-parallel layers.** With stock TP, each rank rounds its partial
  product to bf16 and NCCL adds the partials. Another degree gives other
  partials, another number of bf16 roundings and another order. Two ranks are
  enough: x0 + x1 commutes, but the partials themselves differ from the
  unsharded sum.
- **cuBLAS on narrower shapes.** A column-parallel shard is a GEMM with a
  smaller N, for which cuBLAS may pick another kernel. With `stock` this
  changes bits even before any collective (not separated here).
- **The vocab-parallel log_softmax.** The sum of exp is split by shard and
  added by NCCL. The order follows the degree.
- **The engine's KV cache.** K and V come from column-parallel projections.
  They inherit any variance of the layers below, so a decode instance at
  another TP degree sees another prompt.
- **The backward.** Column-parallel input gradients are sums over ranks. A
  forward fix alone leaves them degree-dependent.

### What was run

| Section | What is compared | Sizes |
|---|---|---|
| tp | trainer forward (teacher-forced) at TP t against TP 1: every logit, the full log-softmax and the realized log-probs; the vocab-parallel log_softmax against its one-GPU form on the gathered logits; 3 repeats at the largest TP | B = 4, T = 160, random token ids; TP 1, 2, 4, 8 (one node), 1, 8, 16 (two nodes) |
| modes | `whole_model.py`'s engine (slot KV cache, `forward_cached`) at TP 2 and 4 against the TP 1 trainer forward | T = 160: decode one token per step among filler requests (1–33 rows per step); chunked prefill, chunk 64, among fillers; chunk 7 alone |
| disagg | prefill at TP a, KV hand-over, decode at TP b, against one-GPU prefill + decode and against the TP 1 trainer forward | 3 requests, prompts 37/64/100 prefilled in one call, then 40/25/33 decode steps in one batch that shrinks; (a, b) = (1, 2), (2, 4), (4, 1), (8, 2) |
| backward | `inv` only: gradient of a fixed random-weighted sum of realized log-probs with respect to the residual stream at 9 points (each layer's input, the final norm's input), TP 2 and 4 against TP 1; then with the NCCL backward | B = 2, T = 128 |
| cost | median of 5 (prefill) or 20 (decode) after warm-up, synchronized | prefill B = 4, T = 512, forward + vocab-parallel log_softmax; decode one step of B = 16 at context 512, logits gathered |

**KV hand-over.** Every KV head goes from the first prefill rank that holds it
to every decode rank that needs it, as one `[2, layers, slots, capacity, 64]`
bf16 buffer (`dist.batch_isend_irecv`, NCCL send/recv; a local copy when both
ends are the same process). Each instance's cache is then gathered head by
head, and the two are compared bit for bit, unwritten (NaN-poisoned) slots
included. With (8, 2) the two instances share GPUs 6 and 7; the other pairs
use disjoint GPUs.

## Evidence

### Setup, code and results

- Code: [examples/whole_model_tp/tp_model.py](tp_model.py)
  (the sharded model, two variants),
  [whole_model_tp.py](whole_model_tp.py) (the
  checks), [run_all.sh](run_all.sh) (one node),
  [run_2node.sh](run_2node.sh) (TP 16).
- Results: [results-a16.json](results-a16.json).
- Setup: two hosts with 8x NVIDIA A16 each (sm86, 15 GB), driver 535,
  torch 2.5.1+cu121, Triton 3.1.0, NCCL 2.21.5 (bundled with torch), default
  NCCL settings. Topology from `nvidia-smi topo -m` on the first host: GPUs 0–3
  share one PCIe switch (PIX), GPUs 4–7 another, and the two groups connect
  through the host bridge (NODE). No NVLink. The second host joins only for
  TP 16, over 25 GbE TCP sockets (no RDMA). TP group t is ranks 0..t-1;
  a decode instance at TP b is the last b ranks. Each part ran under the shared
  multi-GPU lock and the per-GPU locks of GPUs 4–7; no other process was on
  the first host's GPUs during the one-node runs (recorded in the JSON).

### Results

Full JSON: 68 checks, 55 ok, 0 regressions, 13 info. Every `inv` claim
passed and every `stock` comparison across degrees failed, as expected.

#### TP degree, trainer forward (B = 4, T = 160)

| Degree vs TP 1 | `inv` | `stock`: logits differing | tokens whose realized log-prob differs | max abs delta log-prob | KL mean / max |
|---|---|---|---|---|---|
| TP 2 | equal (logits, full log-softmax, realized log-probs) | 87.5% | 100% | 0.221 | 2.3e-3 / 7.8e-3 |
| TP 4 | equal | 87.9% | 100% | 0.250 | 2.4e-3 / 9.2e-3 |
| TP 8 | equal | 88.5% | 100% | 0.313 | 2.7e-3 / 1.6e-2 |
| TP 16 (two nodes) | equal | 89.3% | 100% | 0.343 | 3.0e-3 / 1.2e-2 |

- The `inv` vocab-parallel log_softmax was bitwise its one-GPU form at TP 2,
  4, 8 and 16. The `stock` one was not (3.2–5.8% of log-probs differ), on top of
  its logits differing.
- L1: `inv` at TP 8 and at TP 16, 3 runs, identical. `stock` was also
  identical run to run at TP 8 and 16 (info): its variance here is across
  degrees, not across runs.
- Sanity: `stock` at TP 1 (the sharding code on one rank) equals `model.py`'s
  stock forward bit for bit. `inv` at TP 1 differs from `model.py`'s bi
  forward in 84.7% of logits: a logical TP size is a different arithmetic,
  so the trainer and the engine must both use it.

#### Engine modes under TP (T = 160, against the TP 1 trainer forward)

| Mode | `inv` TP 2 | `inv` TP 4 | `stock` TP 2 / TP 4 |
|---|---|---|---|
| decode, one token per step, among fillers (1–33 rows per step) | equal | equal | 100% of realized log-probs differ; max abs delta 0.186 / 0.251 |
| chunked prefill, chunk 64, among fillers (1–16 rows) | equal | equal | not run |
| chunked prefill, chunk 7, alone | equal | equal | not run |

So L2, L3 and L5 hold together: a sequence decoded at TP 4 in a changing
batch gives the bits of a teacher-forced pass at TP 1.

#### Disaggregated prefill and decode

| (prefill TP, decode TP) | KV received = KV sent | `inv`: prefill KV = one-GPU KV | `inv` vs one-GPU prefill + decode | `inv` vs TP 1 trainer forward | `stock` vs one-GPU prefill + decode: realized log-probs differing, max abs delta |
|---|---|---|---|---|---|
| (1, 2) | yes, both variants | yes | equal | equal | 32%, 0.162 (prefill part equal: same degree) |
| (2, 4) | yes | yes | equal | equal | 100%, 0.238 |
| (4, 1) | yes | yes | equal | equal | 100%, 0.257 |
| (8, 2) | yes | yes | equal | equal | 100%, 0.335 |

- The hand-over is a copy: the gathered caches on both sides have the same
  SHA-1 of their bits (in the JSON), and replicated heads (TP 8) hold the same
  bits on every rank that keeps them.
- Each hand-over moved 3.47 MB (the whole cache once). The second, timed
  hand-over took 1.2–1.9 ms (`inv` runs) and 0.7–1.1 ms (`stock` runs); the
  first, with NCCL point-to-point setup, took 42–97 ms.
- With `stock`, the prefill KV at TP 2, 4 and 8 already differs from one-GPU
  prefill (info), so the decode instance continues from another prompt state.

#### Backward (`inv`, B = 2, T = 128)

| Degree vs TP 1 | logical-TP input gradients | column-parallel input gradients by NCCL all-reduce (Megatron) |
|---|---|---|
| TP 2 | equal at all 9 points | all 9 tensors differ |
| TP 4 | equal at all 9 points | all 9 tensors differ |

The forward fix alone does not carry over to the backward. Parameter
gradients and the gradient through the KV-cache path were not run.

#### Cost (ms per call, median; one node unless marked)

| TP | prefill `stock` | prefill `inv` | `inv` / `stock` | decode step `stock` | decode step `inv` | `inv` / `stock` |
|---|---|---|---|---|---|---|
| 1 | 83.1 (24.7k tok/s) | 132.6 (15.4k tok/s) | 1.60x | 15.08 (1.06k tok/s) | 10.49 (1.52k tok/s) | 0.70x |
| 2 | 59.8 (34.2k) | 91.4 (22.4k) | 1.53x | 10.62 (1.51k) | 18.75 (0.85k) | 1.77x |
| 4 | 48.9 (41.9k) | 75.3 (27.2k) | 1.54x | 9.80 (1.63k) | 18.92 (0.85k) | 1.93x |
| 8 | 49.7 (41.2k) | 71.3 (28.7k) | 1.43x | 12.19 (1.31k) | 18.89 (0.85k) | 1.55x |
| 16, two nodes | 77.2 (26.5k) | 264.4 (7.7k) | 3.43x | 66.08 (0.24k) | 52.63 (0.30k) | 0.80x |

- Prefill is 2048 tokens per call; decode is 16 tokens per step.
- `inv` decode at TP 2–8 stays at 18.9 ms whatever the degree. Each step makes
  32 collectives for the row-parallel sums (all-to-all and all-gather, 16
  times) where `stock` makes 16 all-reduces. This points to collective latency
  (not profiled).
- At TP 1, `inv` decode is faster than `stock`. `model.py`'s stock engine
  path pads ragged rows into one masked SDPA call (not separated further).
- `inv` prefill costs 1.4–1.6x on one node, about what the fp32 partials and
  the untuned one-config GEMM add (not separated).
- TP 16: the logical-TP exchange crosses the 25 GbE link with fp32 partials
  (2x the bytes of a bf16 all-reduce) in a flat all-to-all. The collectives
  case measured a two-level route at about 2x row-parallel; it was not used
  here. The two-node run had one other process on the first host, and its TP 8
  numbers (decode 27.9 / 28.4 ms) are slower than the one-node run; treat the
  TP 16 row as rough.
- No configuration here is faster with TP than at TP 1 by more than 1.7x:
  the model is small and the A16s talk over PCIe.

### What this shows

1. **The single-GEMM L5 result carries over to the whole model.** With a
   logical TP size on all 16 row-parallel layers and a fixed-order
   vocab-parallel log_softmax, logits and realized log-probs were bitwise
   equal at TP 1, 2, 4, 8 and 16 (two nodes). Nothing beyond the collectives
   case's scheme was needed in the forward.
2. **L5 composes with L2 and L3.** Decode in a changing batch and chunked
   prefill at TP 2 and 4 matched the TP 1 trainer forward. So a trainer at one
   TP degree and an engine at another can agree bit for bit.
3. **Disaggregation adds nothing once the parts are invariant.** The KV cache
   at prefill TP a equals the one-GPU cache, so re-sharding by head is a plain
   copy, and decode at TP b continues exactly. With stock TP the cache itself
   already differs.
4. **Stock TP moves every realized log-prob** (KL 2–3e-3 mean), at TP 2 as
   much as at TP 8. A two-rank sum commutes, but its bf16 partials still
   differ from the unsharded sum.
5. **The backward needs its own fix.** Column-parallel input gradients are a
   cross-rank sum; done with NCCL all-reduce they broke L5 at every
   checked point, and with the logical-TP reduction they held.
6. **Cost:** 1.4–1.6x for prefill on one node; 1.6–1.9x for decode at TP 2–8,
   where the doubled collective count dominates. A fused reduce (one
   collective) or a two-level route would be the next step.

## Limits

- One model shape (175M dense, 4 KV heads), random weights, bf16, eager. No
  MoE, no biases, no tied embeddings (`TPModel` asserts this), no
  `torch.compile`, no CUDA graphs.
- The embedding is replicated, not vocab-parallel. A vocab-parallel embedding
  sums one non-zero row with zeros, which is exact except that -0.0 + 0.0 is
  +0.0 (untested).
- Sequence parallelism was not used. The collectives case showed that
  reduce-scatter breaks L2; the all-to-all fallback was not tried inside the
  model.
- Degrees that do not divide C = 16 (3, 6, 12) and degrees above C are not
  covered. KV-head replication was tested at TP 8 and 16 in the forward only;
  the backward needs TP <= 4 here (whole KV heads per rank).
- The engine's log-probs come from gathered logits with the one-GPU formula;
  the tp section shows this equals the vocab-parallel result for `inv` at
  every degree tested.
- The KV hand-over ran inside one process group on one node; a real
  disaggregated system crosses hosts or uses another transport (NIXL, RDMA).
  Any byte copy should keep the bits, but only NCCL send/recv was run.
- Backward: activation gradients only. Parameter gradients sum over the batch
  and over ranks for the replicated norms; not run. No backward through the
  KV-cache path.
- `stock` here is model.py's stock op set with NCCL defaults. Megatron or vLLM
  use their own kernels and custom all-reduce; they were not run.
- Cost: an untuned Triton GEMM, A16 over PCIe, a two-node run with one
  foreign process on the first host. Collective counts are not fused.

## How it was built

An agent wrote `tp_model.py`, the checks, the drivers and this page; no human
edits. What took more than one try:
- The first backward run failed with "Triton Error [CUDA]: invalid device
  context". Autograd runs the CUDA backward on its own thread, and the first
  launch there was a Triton kernel; one `torch.cuda.synchronize()` per thread
  before it fixed it.
- The first full run hit out-of-memory on GPU 6: other jobs were using GPUs
  4–7 under per-GPU locks that `run_all.sh` did not take. The driver now takes
  them after the multi-GPU lock, and each section records foreign GPU
  processes. Its cost numbers were discarded.
- The first KV hand-over timings included NCCL point-to-point setup (25–97
  ms); the hand-over is now run twice and the second is reported.
- The forward checks passed on their first run; nothing in the logical-TP
  scheme or the model was changed to make them pass.
- Quick mode (`torchrun --nproc_per_node=4 whole_model_tp.py --quick`, TP
  1/2/4, 36 ok, 7 info) took 25 s wall on 4 A16s.
