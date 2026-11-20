# Gradients under data parallelism and FSDP-style sharding (emulated): lab report

This is the full record behind the handbook page [Data parallel](../../handbook/1-pretraining/data-parallel.md). The page teaches the idea and the step; this report keeps the setup, every configuration, every table, the limits and how the case was built.

## Summary

- Under Ring, DDP gradients depend on bucketing and on DDP's first-iteration
  bucket: up to 48% of gradient elements changed. `NCCL_ALGO=Tree` fixes this
  on one and two nodes, at 1.09–1.37x from 12 MB on and 4.9x at 1 MB (d = 8).
- Reduce-scatter (FSDP-style, emulated) is invariant to everything but the
  owner, which breaks on any resharding. An ordered reduce-scatter fixes it at
  1.11–1.30x on one node.
- No collective setting gives L5: DDP at another degree changed 96% of the
  gradient. **Fix:** logical shards (a fixed number of micro-batches, one
  backward each, one fixed tree that includes the cross-rank sum) give
  identical gradients, weights and Adam state at d = 1–16, sharded or not, and
  across elastic resume, on 2x8 A16.
- **Cost** of logical shards: lost batching at low d (1.1x with 4 sequences per
  shard, 18x with one tiny sequence) and the exchange at high d (1.2–1.4x
  across two nodes with the 2level route).

## How

The [collectives case](../../handbook/3-kernels/collectives.md) measured all-reduce and reduce-scatter
inside tensor-parallel layers. Training uses the same collectives on the
gradient: DDP all-reduces it in buckets, FSDP and ZeRO reduce-scatter a flat
copy of it, and both reduce a global grad-norm and a loss mean across ranks.
This case asks, for fp32 gradients:
- Does a parameter's averaged gradient depend on how gradients are packed into
  messages, or on which rank owns its slice (L2)?
- Does it depend on the data-parallel degree d or on the placement (L5)?
- Can a run change d in the middle (elastic resume) and still match bit for bit?
### Where the variance comes from

- **The gradient is a sum over ranks, ordered by NCCL.** With Ring, each
  element's sum starts at a rank set by its chunk. Chunk boundaries follow the
  message, so the bits of one parameter depend on what else is in its message
  and at which offset (measured in the collectives case for activations; here
  for gradients).
- **DDP changes its own packing.** On the first iteration after construction,
  DDP puts every gradient in one bucket. It then rebuilds buckets in backward
  order: here 1 MB and 12 MB from the second iteration on. `bucket_cap_mb`,
  parameter order and model edits move the boundaries as well.
- **Reduce-scatter's order follows the owner.** The slice kept by rank j is
  summed starting at rank j+1. In FSDP the owner of an element follows the flat
  layout: padding, parameter order, wrapping and resharding all move it.
- **The degree changes the arithmetic, not only the order.** At degree d, each
  rank runs one batched backward over 16/d sequences and the ranks add d
  partial gradients. Another d is another set of partials. No collective
  setting can fix that.
- **Global scalars.** A sharded grad-norm adds per-rank sums of squares; the
  loss mean adds per-rank means. Both are degree-dependent sums.

### The fix

1. **Ordered collectives.** No reduction in the network.
   - *Ordered reduce-scatter:* one all-to-all in which rank r receives only
     slice r of every rank's message; r adds the d slices in rank order in
     fp32. It moves the bytes of a reduce-scatter.
   - *Ordered all-reduce:* the same, then an all-gather. In DDP it is a comm
     hook (`Worker.ordered_hook`). Each element's order is fixed whatever the
     bucket or offset.
   - Two nodes: *2level* routes the same tree node first (all-to-all inside the
     node), then one fp32 exchange with the rank of the same local index.
2. **Logical shards for L5.** Fix S = 16 logical micro-batches. At degree d,
   rank r owns shards r·16/d … (r+1)·16/d − 1. It runs a separate backward per
   shard and adds its shards in a fixed pairwise tree over the shard index,
   ((s0+s1)+(s2+s3))+…, as far as they form whole subtrees. The ordered
   all-to-all moves those subtree sums, and the owner of each slice finishes
   the same tree. The tree never depends on d.
   - Degrees that do not divide 16 (3, 5, 6, 7, 12) still work: a rank sends
     each of its maximal aligned subtrees separately.
3. **Global scalars by the same rule.**
   - Grad-norm: cut the flat gradient into fixed blocks of 4096 elements. A
     block's sum of squares is a pairwise tree of elementwise fp64 adds.
     Slices are cut on block boundaries, the block sums are all-gathered and
     added in one pairwise tree. The norm is then a function of the vector
     only.
   - Loss mean: all-gather the 16 shard losses, add them in the pairwise tree.
4. **Optimizer.** `torch.optim.AdamW(foreach=False, fused=False)`, lr 1e-3,
   weight decay 0.01, clipping at a norm of 0.5 by the logical norm. The
   update is elementwise, so replicated (dp) and sharded (fsdp: each rank holds
   a block-aligned slice of weights and Adam state) give the same bits.

The canonical flat layout is `named_parameters()` order, zero-padded to a whole
number of blocks that is a multiple of 16.

### What was done

| Measure | Tested |
|---|---|
| Copy collectives | all-gather, all-to-all, broadcast of random bit patterns (NaN payloads, −0.0, subnormals), 240 B–16 MB, groups of 3, 4, 8 and 16 |
| 1. DDP packing | the same per-rank gradients (×1/d) reduced under 19 packings: one message per parameter (reference); one flat message in forward, reverse, size and shuffled order; padding of 1, 1000, 4096 and 100003 elements in front; DDP buckets of 64 KB–25 MB in reverse, forward, shuffled and size order; per layer. Also `torch.nn.parallel.DistributedDataParallel` itself: iteration 1 vs 2, `bucket_cap_mb` 0.25/1/4, the ordered hook. d = 4, 8 and 16 |
| 2. Degree | the same 16 sequences at d = 1, 2, 4, 8 (one node) and 16 (two nodes), stock and with the ordered all-reduce |
| 3. Logical shards | 16 and 64 sequences (small), 16 (medium). One node: d = 1–8, plus d = 1 on GPU 5, d = 2 on GPUs 0 and 4, d = 4 on 0, 1, 4, 5 (across PCIe switches). Two nodes: d = 1 on node 1, 2 (one rank per node), 4, 8 per node, 8 split 4+4, 12 split 6+6, 16, flat and 2level routes. Three repeats |
| Precondition | 4 shards × 2 batch sizes × 2 models: 3–5 repeats per rank, then bits compared across all 8 or 16 ranks |
| 4. FSDP | 8 flat layouts at d = 4, 8, 16: padding of 1, 1000 and half a slice in front, reversed and shuffled order, owners rotated by one, per layer, per parameter. Across d: the flat gradient reduce-scattered at d = 2–16. All-gather of the parameter slices |
| 5. Scalars | grad-norm of three vectors (two gradients, the weights) at every placement; padding 0/1/1000/4096 at d = 8 or 16; loss mean at every d |
| 6. End to end | 10 AdamW steps in dp and fsdp mode at every placement above; elastic resume through a checkpoint file: 5 steps at d = 4 then 5 at d = 8 (one node), 5 at d = 8 then 5 at d = 16 (two nodes), for dp→dp, dp→fsdp, fsdp→dp, fsdp→fsdp. The stock loop: DDP + `clip_grad_norm_` + the same AdamW |

Every result is compared bitwise with `tools/bi_check.py`, on every rank.

## Evidence

### Setup, code and results

Taken through the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md),
recipe 10.
- Code: [examples/training_dp/training_dp.py](training_dp.py).
  One file: the driver launches one `torch.distributed.run` per NCCL setting
  and merges them into one bi-results-v1 file.
- Results: [results-a16.json](results-a16.json)
  (one node, 8 ranks) and
  [results-a16-2node.json](results-a16-2node.json)
  (two nodes, 16 ranks).
- Setup: the hosts of the collectives case. 8x NVIDIA A16 (sm86) per node, PCIe
  inside a node, two nodes over 25 GbE sockets. torch 2.5.1+cu121, NCCL 2.21.5.
- NCCL settings: default, `NCCL_ALGO=Ring`, `NCCL_ALGO=Tree`, one launch each.
  On these hosts the default picked Ring for every gradient-sized message, so
  default and Ring gave the same counts throughout.

**Model.** A pre-LN decoder LM in fp32 with stock torch ops (SDPA, LayerNorm,
GELU, tied embeddings), run with
`torch.use_deterministic_algorithms(True, warn_only=False)`.
- small: width 256, 4 layers, 4 heads, context 128, vocab 256, 52 tensors,
  3,257,856 parameters (12.4 MB of gradient). Used for every check.
- medium: width 512, 6 layers, 8 heads, context 256, 19.2M parameters (73 MB).
  Used for the precondition, logical shards, packing and timing.
- Data: token chains from a fixed random successor table, 16 sequences per
  step (64 for some checks).

### Results

#### 1. DDP packing at a fixed degree (small model)

Elements of 3,257,856 that differ from per-parameter all-reduce, d = 8, one
node:

| Packing | messages | default / Ring | Tree | ordered all-reduce |
|---|---|---|---|---|
| one flat message, forward order | 1 | 1,571,627 | 0 | 0 |
| the same with 1 padding element in front | 1 | 1,571,628 | 0 | 0 |
| DDP default buckets (1 MB first, then 25 MB), reverse | 2 | 1,467,306 | 0 | 0 |
| 1 MB buckets, reverse | 13 | 641,061 | 0 | 0 |
| 1 MB buckets, forward | 13 | 764,745 | 0 | 0 |
| 128 KB buckets, reverse | 18 | 209,176 | 0 | 0 |
| 256 KB buckets, size order | 17 | 50,460 | 0 | 0 |
| per layer | 5 | 1,542,557 | 0 | 0 |
| all 19 packings | | 19 fail | pass | pass |

- **Ring confirmed: the bucket layout changes other parameters' bits.** The
  largest tensor (wte, tied with the head) got 14 different results over 20
  packings at d = 8, 15 at d = 4 and at d = 16 (two nodes). Max abs difference
  1.5e-8.
- Every element of every Ring result equalled a sequential ring order; none
  was unexplained.
- **Tree passed** at d = 4, 8 and across two nodes at d = 16: one result per
  parameter in all 19 packings, small and medium.
- **torch DDP, default and Ring.** The same input gave a different gradient on
  the first iteration (one 13 MB bucket) than on the second (1 MB + 12 MB):
  961,479, 1,435,340 and 1,804,979 elements at d = 4, 8 and 16. Iterations 2–4
  repeated. `bucket_cap_mb` 0.25, 1 and 4 changed 0.85–1.55M elements at d = 8.
  Under Tree both passed.
- The packing emulation reproduced DDP bit for bit when given DDP's bucket
  layout. torch's `allreduce_hook` (divide by d) gave the bits of DDP's default
  path (multiply by 1/d) at d = 3.
- **The ordered hook** gave one gradient on every iteration and every cap,
  equal to the ordered all-reduce.

#### 2. The same 16 sequences at d = 1, 2, 4, 8, 16

Elements differing from one batched backward at d = 1 (of 3,257,856):

| | d = 2 | d = 4 | d = 8 | d = 16 (2 nodes) |
|---|---|---|---|---|
| torch DDP, default | 3,131,036 | 3,146,452 | 3,145,595 | 3,144,297 |
| torch DDP, Tree | 3,131,036 | 3,146,763 | 3,145,813 | 3,137,075 |
| batched backward + ordered all-reduce | 3,131,036 | 3,146,070 | 3,145,435 | 3,144,360 |
| batched backward + NCCL reduce-scatter | 3,131,036 | 3,146,497 | 3,145,675 | 3,144,978 |
| loss mean (per-rank mean, all-reduce, ÷d) | differs | differs | differs | differs |

- 96% of elements, in every tensor, at every d, under every setting. The
  per-rank backward is the cause: a fixed order across ranks does not help.
- The loss was 5.5923338 at d = 1 and 5.5923333 or 5.5923324 at other d.

#### 3. Logical shards

| Check | one node | two nodes |
|---|---|---|
| precondition: a shard's gradient and loss repeat, and are the same bits on every rank (and node) | pass (8 ranks) | pass (16 ranks) |
| gradient and loss mean equal d = 1, small 16 and 64 sequences | pass: d = 1–8 and 3 cross-switch placements | pass: 14 placements, flat and 2level |
| the same, medium | pass: d = 1, 2, 3, 4, 8 | pass: 9 placements |
| repeats | pass | pass |
| the same bits under default, Ring and Tree | pass | pass |
| two-node bits equal one-node bits | — | pass (gradients, weights, Adam state, losses, norms) |

- **Without strict deterministic mode** the precondition fails: the medium
  model's shard gradient differed in 3 of 7 repeats, in up to 9.5M of 19.2M
  elements (SDPA backward).
- **Rejected designs** (small, elements differing from d = 1):

  | Design | d = 2 | d = 4 | d = 8 | d = 16 |
  |---|---|---|---|---|
  | logical shards, NCCL all-reduce across ranks (default) | pass | 966,948 | 1,662,483 | 2,087,935 |
  | the same under Tree | pass | 817,708 | 1,569,965 | 1,626,199 |
  | logical shards, NCCL reduce-scatter across ranks | pass | 966,160 | 1,658,972 | 2,084,220 |
  | strided shards (r, r+d, …) pre-summed per rank, then the ordered tree | 1,897,351 | 1,835,770 | 1,806,293 | — |

  Two ranks pass because a two-term sum commutes. Tree fails here: its chain
  order is fixed but is not the pairwise tree over shards. Strided shards pass
  when each leaf is sent unreduced.
- **Not the batched gradient.** Logical shards at d = 1 differ from one
  batched backward over the 16 sequences in 3,137,665 elements. It is another
  fixed arithmetic. Max abs error against an fp64 backward: 3.41e-8 (logical
  shards), 3.56e-8 (batched), 3.57e-8 (DDP, d = 8).

**Cost**, ms per global batch (forward, backward and reduction, all ranks),
median of 5–7 after 2 warm-up runs, default NCCL. In parentheses: the ratio to
torch DDP at the same d.

| Model, sequences | torch DDP d = 1 / 2 / 4 / 8 / 16 | logical shards d = 1 | d = 2 | d = 4 | d = 8 | d = 16 flat / 2level |
|---|---|---|---|---|---|---|
| small, 16 | 30.4 / 39.3 / 36.2 / 36.9 / 46.8 | 559 (18.4x) | 247 (6.3x) | 133 (3.7x) | 71.5 (1.94x) | 51.8 (1.11x) / 48.5 (1.04x) |
| medium, 16 | 239 / 134 / 82.1 / 61.0 / 86.5 | 636 (2.67x) | 377 (2.82x) | 197 (2.40x) | 116 (1.91x) | 173 (2.00x) / 101 (1.17x) |
| medium, 64 | 964 / 495 / 261 / 144 / 89.8 | 1057 (1.10x) | 548 (1.11x) | 289 (1.11x) | 166 (1.16x) | 203 (2.26x) / 129 (1.44x) |

- At low d the cost is lost batching: the backward passes alone took
  491–659 ms of the small model's d = 1 time. With one sequence per shard the
  small model's GPU is mostly idle; with 4 sequences per shard (medium, 64) the
  cost falls to 1.10x.
- At high d the cost is the exchange: medium, 64 sequences, d = 8 spent
  134 ms in backward passes and 166 ms in total; across two nodes the flat
  all-to-all doubled the time, the 2level route kept it at 1.17–1.44x.
- The ordered hook alone cost 0.78–1.18x DDP on one node, and 1.06–2.24x at
  d = 16 (flat).

#### 4. FSDP-style reduce-scatter

Small model, d = 8, one node. Elements that differ from the canonical layout,
elements whose owner changed, and differing elements whose owner did not:

| Layout | NCCL reduce-scatter | ordered reduce-scatter |
|---|---|---|
| 1000 padding elements in front | 1,838 / 3,500 / 0 | 0 |
| half a slice of padding in front | 370,490 / 712,656 / 0 | 0 |
| parameters reversed | 1,734,058 / 3,160,576 / 0 | 0 |
| owners rotated by one (a resharding) | 1,706,203 / 3,257,856 / 0 | 0 |
| all 8 layouts, d = 4, 8, 16 | fail at every d, under default and Tree | pass |

- **Only the owner matters.** Every differing element had changed owner; every
  element that kept its owner kept its bits.
- The slice kept by rank j was summed from rank j+1 around the ring, at d = 4,
  8 and 16. NCCL's reduce-scatter therefore equals a rank-order sum 0..d−1 only
  on the last rank's slice, and it differs from NCCL's all-reduce of the same
  inputs (107k–239k elements per owner at d = 8).
- `NCCL_ALGO=Tree` does not reach reduce-scatter (as in the collectives case).
- The ordered reduce-scatter, gathered, equals the ordered all-reduce bit for
  bit. With logical shards it gave the d = 1 gradient at every degree and
  placement, and under every shifted layout.
- **All-gather of parameter slices** returned the parameters bit for bit at
  every placement, as did every copy collective on every bit pattern.

**Cost**, fp32 messages, ms per call (ratio to the NCCL collective):

| Message | NCCL reduce-scatter d = 8 | ordered RS d = 8 | NCCL RS d = 16 | ordered RS d = 16 flat / 2level | NCCL all-reduce d = 8 | ordered AR d = 8 | Tree AR d = 8 | ordered AR d = 16 flat / 2level |
|---|---|---|---|---|---|---|---|---|
| 1 MB | 0.29 | 0.37 (1.30x) | 0.68 | 1.83 (2.69x) / 0.78 (1.15x) | 0.53 | 0.63 (1.20x) | 2.58 (4.9x) | 2.53 (2.11x) / 1.18 (0.99x) |
| 12 MB | 2.77 | 3.08 (1.11x) | 4.25 | 18.4 (4.3x) / 5.77 (1.36x) | 5.61 | 5.87 (1.05x) | 7.66 (1.37x) | 26.4 (3.1x) / 10.7 (1.26x) |
| 73 MB | 16.2 | 18.6 (1.15x) | 24.9 | 121 (4.9x) / 33.6 (1.35x) | 33.4 | 35.9 (1.07x) | 37.7 (1.13x) | 139 (2.7x) / 63.4 (1.25x) |
| 256 MB | 55.7 | 66.2 (1.19x) | — | — | 117 | 126 (1.07x) | 128 (1.09x) | — |

The Tree column comes from the Tree launch; its own NCCL reduce-scatter cost
the same as default's.

#### 5. Grad-norm and loss mean

| Check | Result |
|---|---|
| sharded norm (local sum of squares, NCCL all-reduce) equals the d = 1 norm | fails at some placements: one node, gradient d = 2, 2x, 3, 7; weights d = 2, 3, 8. Two nodes: weights also at d = 16, 8 per node, 4+4. Others passed |
| the same with padding in front of the flat vector, d = 8 | the norm changed for 2 of 3 vectors |
| logical norm at every placement, blocks of 4096 and of 1024 | pass, one and two nodes |
| loss mean: stock / pairwise tree over shards | differs at every d / pass |

- A scalar sum of a few dozen terms often rounds to the same value, so the
  stock norm passes by luck at many placements. Its result is still a
  function of the sharding.
- The logical norm cost 1.34 ms against 0.094 ms for `torch.linalg.vector_norm`
  on 3.3M elements (14x), and 7.5 against 0.46 ms on 19.2M (16x), one GPU.
  That is 2% (small) and 6% (medium) of a d = 8 step here. The kernel is a chain of torch ops, not
  tuned.

#### 6. End to end: 10 AdamW steps

Small model, 16 sequences per step, all 10 steps clipped (norms 1.0–6.8).

| Run | One node | Two nodes |
|---|---|---|
| logical shards + ordered reductions, dp mode: weights, Adam m and v, losses, norms equal d = 1 | pass at d = 1–8 and 3 cross-switch placements | pass at 14 placements, flat and 2level |
| the same, fsdp mode (sharded weights and Adam state) | pass | pass |
| elastic: 5 steps at d = 4 → checkpoint file → 5 steps at d = 8, all 4 mode pairings | pass | — |
| elastic: 5 steps at d = 8 → checkpoint → 5 steps at d = 16, all 4 pairings, both routes | — | pass |
| one node against two nodes | — | pass |
| stock loop (DDP, `clip_grad_norm_`, AdamW) against its d = 1 run | fails at d = 2, 4, 8: 3.06–3.13M of 3.28M elements | fails at d = 16 too |
| stock loop, d = 8 (16), restart from `state_dict`s after 5 steps | default: 734,771 (740,154) differ. Tree: pass | |
| stock loop, `bucket_cap_mb=1` against the default | default: 1,390,287 (1,381,125) differ. Tree: pass | |

The elastic run equals 10 steps at d = 1, and 10 steps at d = 4 equal d = 1, so
d = 8 then 16 matches d = 4 throughout.

- **The optimizer path is part of the arithmetic.** The foreach and fused
  AdamW paths are each degree-invariant (d = 8 equalled their own d = 1), but
  after 10 steps they differ from the single-tensor path in 1,717,418 and
  2,777,703 of 3,276,800 elements.
- **A restart alone breaks a stock DDP run.** Rebuilding DDP restarts with one
  bucket for one iteration, which changes the gradient under Ring.

### What this shows

1. **Ring makes DDP gradients depend on bucketing.** Changing the bucket cap,
   the parameter order or padding changed up to 48% of all gradient elements,
   in parameters whose own data did not change. torch DDP is not even
   self-consistent: its first iteration and every restart use one bucket.
   `NCCL_ALGO=Tree` fixed this on one and two nodes (strategy 1, configure),
   at 1.09–1.37x from 12 MB on and 4.9x at 1 MB (d = 8). The ordered all-reduce fixed it
   at 1.05–1.20x on one node.
2. **Reduce-scatter is invariant to everything but the owner.** That is enough
   at one fixed layout, and breaks on any resharding, padding or wrapping
   change. Tree does not reach it. The ordered reduce-scatter fixes it at
   1.11–1.30x on one node.
3. **No collective setting gives L5.** DDP at another d changed 96% of the
   gradient, also with an ordered collective. The per-rank batched backward is
   part of the arithmetic.
4. **Logical shards give L5, and with it elastic resume.** A fixed number of
   micro-batches, one backward each, one fixed tree: identical gradients,
   losses, norms, weights and Adam state at d = 1–16, on one node or two,
   replicated or sharded, and across a change of d mid-run.
   - The cost is lost batching at low d (1.1x with 4 sequences per shard,
     18x with one tiny sequence) and the exchange at high d (1.2–1.4x across
     two nodes with the 2level route).
   - Every sum across ranks must be part of the tree. Handing the cross-rank
     part to NCCL, Ring or Tree, failed from 3 ranks.
5. **Global scalars need the same treatment.** A block-aligned fp64 norm and a
   tree over shard losses are invariant; the stock sharded norm is not.
6. **Pin the optimizer path.** Single-tensor, foreach and fused AdamW are
   different arithmetic.

## Limits

- Two hosts, 8 A16 each over PCIe, 25 GbE sockets; NCCL 2.21.5. Not tested:
  three or more nodes, NVLink, InfiniBand/RDMA.
- fp32 gradients and fp32 weights only. Not tested: bf16 gradients, mixed
  precision, loss scaling, gradient accumulation across steps beyond the
  shards.
- Stock torch ops; invariance of the shard backward rests on strict
  deterministic mode and on every rank running the same shapes. Two small
  models; no real pretrained LM (none is available on these hosts).
- FSDP was emulated (flat gradient, reduce-scatter, sharded AdamW).
  `torch.distributed.fsdp` itself was not run; a probe was written but not
  launched. ZeRO-1/2/3 in DeepSpeed were not tested.
- Degrees above 16 and S other than 16 were not tested. A degree that does not
  divide S works (3, 5, 6, 7, 12 passed) but balances work unevenly.
- The logical norm is untuned torch ops. Accuracy of the two norms against an
  fp64 sum was recorded rounded to 4 decimals in the one-node file (0.0), so it
  is not reported.
- Timing: median of 5–7 runs, 2 warm-up runs, slowest rank. nvidia-smi on node 0
  showed no other compute process before or after each launch. No overlap of communication with the
  backward was attempted for logical shards.

## How it was built

An agent wrote the script, with no human edits.
- **Probes.** Single-GPU: without deterministic mode SDPA's backward was not
  run-to-run stable; summing per-sequence gradients differs from a batched
  backward in nearly every element. 8 ranks: torch DDP's iteration-1 single
  bucket, the bucket-cap effect, the ordered hook, and collective timings.
- **First debug launch deadlocked**: a collective sat inside a rank-0-only
  branch of the order diagnostic. Fixed; the next quick launch passed every
  check.
- **One full run** on one node (3 launches, 11 minutes) and one on two nodes
  (3 launches, 15 minutes), both with 0 regressions. The session was cut by a
  usage limit between them and resumed.
- After the runs, the measurement rounding (4 decimals) was found to hide the
  relative errors; it was changed to 5 significant digits for later runs.
  The two-node file was re-written in compact JSON to stay under 300 KB.
