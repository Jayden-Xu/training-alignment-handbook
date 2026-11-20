# Gradients under data parallelism and FSDP-style sharding

## Where we are

This page serves step 5 (classify) of the [pre-training workflow](README.md)
when the first divergence points to a gradient that ranks sum together, and
step 2 (control) for any run that uses data parallelism. The
[training step](training-step.md) is pinned on one rank; what is still open is
the sum across ranks. DDP all-reduces the gradient in buckets, FSDP and ZeRO
reduce-scatter a flat copy of it, and both reduce a gradient norm and a loss
mean across ranks. Each of those sums has an order, and the order can depend on
things that have nothing to do with the parameter.

## The idea: the order follows the position, and the degree changes the sum

**A tiny example: position.** Four ranks hold gradients g0, g1, g2, g3 of one
parameter. A ring all-reduce cuts the message into chunks, and each chunk's sum
starts at a different rank:

| Element in chunk | Order of the sum (one example) |
|---|---|
| 0 | ((g1 + g2) + g3) + g0 |
| 2 | ((g3 + g0) + g1) + g2 |

Put the same parameter at another offset in the message (another bucket size,
another parameter order, one padding element in front) and its elements land in
other chunks: another order, other bits. DDP even changes its own packing: its
first iteration puts every gradient in one bucket, later iterations rebuild
buckets, and every restart starts over. Reduce-scatter has the same property
through the owner: the slice a rank keeps is summed starting at the next rank,
so a resharding moves the order.

**A tiny example: degree.** With 16 sequences at degree d = 2, each rank runs a
backward over 8 sequences and two partial gradients are added. At d = 4, four
partials of 4 sequences. These are different sums, not the same sum in another
order. No collective setting can fix that.

<figure class="fx" data-fig="dp-tree" aria-label="A gradient summed at degree 2 and 4, per rank and with logical shards">
<div class="fx-stage">
<div class="fx-head"><span class="fx-label">16 sequences, data-parallel degree</span><span class="fx-swap"><span class="fx-chip" data-at="1" data-to="3">d = 2</span><span class="fx-chip is-key" data-at="3">d = 4</span></span></div>
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">per-rank backward</span><div class="fx-tree" style="--n:16">
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-band" data-c="0" style="--w:8">rank 0</span><span class="fx-band" data-c="1" style="--w:8">rank 1</span></div><div class="fx-level" data-at="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span><span class="fx-band" data-c="1" style="--w:4">rank 1</span><span class="fx-band" data-c="2" style="--w:4">rank 2</span><span class="fx-band" data-c="3" style="--w:4">rank 3</span></div></div>
<div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span><span class="fx-cell">10</span><span class="fx-cell">11</span><span class="fx-cell">12</span><span class="fx-cell">13</span><span class="fx-cell">14</span><span class="fx-cell">15</span></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-j" style="--w:8;--k:8"><span class="fx-node is-mono">Σ 0–7</span></span><span class="fx-j" style="--w:8;--k:8"><span class="fx-node is-mono">Σ 8–15</span></span></div><div class="fx-level" data-at="3"><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 0–3</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 4–7</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 8–11</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 12–15</span></span></div></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="2" data-to="3"><span class="fx-j" style="--w:16"><span class="fx-node is-mono is-key">x₂</span></span></div><div class="fx-level" data-at="4"><span class="fx-j" style="--w:16;--k:4"><span class="fx-node is-mono is-bad">x₄ ≠ x₂</span></span></div></div>
</div></div>
<div class="fx-row"><span class="fx-label">logical shards, any d</span><div class="fx-tree" style="--n:16">
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-band" data-c="0" style="--w:8">rank 0</span><span class="fx-band" data-c="1" style="--w:8">rank 1</span></div><div class="fx-level" data-at="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span><span class="fx-band" data-c="1" style="--w:4">rank 1</span><span class="fx-band" data-c="2" style="--w:4">rank 2</span><span class="fx-band" data-c="3" style="--w:4">rank 3</span></div></div>
<div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span><span class="fx-cell">10</span><span class="fx-cell">11</span><span class="fx-cell">12</span><span class="fx-cell">13</span><span class="fx-cell">14</span><span class="fx-cell">15</span></div>
<div class="fx-level" data-at="1" data-hot="3"><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span></div>
<div class="fx-level" data-at="1" data-hot="3"><span class="fx-j" style="--w:4"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:4"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:4"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:4"><span class="fx-node is-op">+</span></span></div>
<div class="fx-level" data-at="1" data-hot="4"><span class="fx-j" style="--w:8"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:8"><span class="fx-node is-op">+</span></span></div>
<div class="fx-level" data-at="2" data-hot="4"><span class="fx-j" style="--w:16"><span class="fx-node is-mono is-ok">x</span></span></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>16 sequences in one global batch.</li>
<li data-hold="2">d = 2: each rank owns 8 sequences. Per rank: one backward, one partial. Logical shards: each rank sums its subtree of one fixed tree.</li>
<li data-hold="1.5">The partials are added across ranks: x₂. The logical shards finish the same tree: x.</li>
<li data-hold="2">d = 4: each rank owns 4 sequences. Per rank: four partials, other sums. Logical shards: the same tree, smaller subtrees per rank.</li>
<li data-hold="2">Per rank: x₄ ≠ x₂, the result follows the degree. Logical shards: the same tree, the same bits x.</li>
</ol>
<figcaption>Figure: at d = 2 and d = 4 the per-rank partial sums depend on the degree. With logical shards the same tree is summed at every degree.</figcaption>
</figure>

**Rule.** The order of a cross-rank sum must be a function of the element and of
a fixed logical split of the batch, never of message position, owner or degree.
This is part 0's principle at L5: the parallel degree may change who computes a
partial sum, not which partial sums exist.

## The step

### Pin the cross-rank sums

**Goal:** a parameter's gradient, the norm, the loss and the updated weights do
not depend on bucketing, sharding layout or degree.

**How:** pick the options your run needs. Code:
[training_dp.py](../../examples/training_dp/training_dp.py) (ordered
collectives, the DDP comm hook `Worker.ordered_hook`, logical shards, the
logical norm); every comparison is bitwise with
[bi_check.py](../../tools/bi_check.py).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| `NCCL_ALGO=Tree` for all-reduce | One gradient per parameter whatever the bucketing, at a fixed degree | small to moderate for large messages, large for small ones | DDP at a fixed degree; quickest fix |
| Ordered all-reduce (an all-to-all, a sum in rank order in fp32, an all-gather; a DDP comm hook) | The same, independent of NCCL's algorithm | small to moderate on one node | You need it under any NCCL setting |
| Ordered reduce-scatter (rank r receives slice r of every message and adds the d slices in rank order) | FSDP-style gradients independent of padding, wrapping and resharding | moderate (under 2x) on one node; across nodes with the two-level route | Sharded training; any change of layout |
| Logical shards: S fixed micro-batches, one backward each, one pairwise tree over the shard index that includes the cross-rank sum | Identical gradients, weights and Adam state at every degree; elastic resume | moderate with several sequences per shard; large when shards are tiny | The degree changes, or runs at two degrees must agree |
| Logical norm and loss: fixed blocks, fp64 sums of squares, a pairwise tree; the loss as a tree over shard losses | A clipping norm and a loss that are functions of the values only | moderate for the norm itself; small in a step | Always with logical shards |
| Single-tensor AdamW (`foreach=False, fused=False`) | An elementwise update, so replicated and sharded weights give the same bits | small | Always; pin one path |

Logical shards need the [training step](training-step.md) pinned: each shard's
backward must repeat on every rank, which with stock ops needs strict
deterministic mode. Across two nodes, route the exchange inside the node first
(the two-level route); a flat all-to-all over the slower link costs much more.

**Success:** gradients, losses, norms, weights and Adam state equal the d = 1
run bitwise, on every rank, and an elastic resume at another degree matches a
straight run.

Shown with fp32 gradients up to 16 GPUs on two nodes, replicated and with
FSDP emulated (flat gradient, reduce-scatter, sharded AdamW; the framework's
own FSDP was not run): Tree and the ordered collectives removed the bucketing
effect, the ordered reduce-scatter removed the owner effect, and logical shards
gave identical results at every degree tested and across elastic resume
([report](../../examples/training_dp/REPORT.md#what-this-shows)).

### Classify a first divergence in the gradient sum

| Observation | Cause |
|---|---|
| Gradients change after a bucket-size, parameter-order or model edit, at the same degree | Ring all-reduce order follows the message offset |
| The first iteration, or the first after a restart, differs from the rest | DDP's one-bucket first iteration |
| Every element that differs changed its owning rank | reduce-scatter order follows the owner |
| Almost every element differs at another degree, under any collective | the per-rank backward is part of the arithmetic: use logical shards |
| The clipping norm differs only at some placements | a sharded sum of squares added by NCCL |

## What can go wrong

!!! warning "DDP is not consistent with itself"
    DDP's first iteration after construction uses one bucket, and every restart
    builds DDP again. A restart alone changes the gradient under Ring. Use Tree
    or the ordered hook, not a bucket-size setting.

!!! warning "Tree does not reach reduce-scatter"
    `NCCL_ALGO=Tree` changes all-reduce only. Sharded training still follows
    the owner. Use the ordered reduce-scatter for FSDP-style sharding.

!!! warning "The cross-rank part handed back to NCCL"
    Logical shards fail if the last step of the tree is an NCCL all-reduce or
    reduce-scatter, Ring or Tree, from three ranks on. The tree is then not the
    same tree. Every sum across ranks must be part of the fixed tree.

!!! warning "A scalar that passes by luck"
    A sum of a few dozen terms often rounds to the same value, so a stock
    sharded norm can agree at many placements. It is still a function of the
    sharding and fails at others. Use the block-aligned logical norm.

!!! warning "Logical shards are not the batched gradient"
    Logical shards at d = 1 differ from one batched backward over the same
    sequences. Both are correct roundings; they are different arithmetic. Pick
    one as the reference and use it everywhere.

## Summary

- Under Ring, a gradient's bits depend on its position in the message; under
  reduce-scatter, on its owner. Tree or ordered collectives fix the first, an
  ordered reduce-scatter the second.
- Another degree is another set of partial sums: only logical shards with one
  fixed tree give L5 and elastic resume.
- Norms and loss means are cross-rank sums too: give them the same tree.
- Pin the optimizer path.

## Go deeper

- [Report: data parallelism and FSDP-style sharding](../../examples/training_dp/REPORT.md):
  every packing, layout and placement, timings, the rejected designs, limits.
- Code: [training_dp.py](../../examples/training_dp/training_dp.py).
- [Collectives](../3-kernels/collectives.md): the same collectives inside
  tensor-parallel layers.
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md),
  recipe 10.
