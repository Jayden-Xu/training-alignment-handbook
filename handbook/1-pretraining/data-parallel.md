# Keeping gradients identical under data parallelism and FSDP-style sharding

## Where we are

This page serves step 5 (classify) of the [pre-training workflow](README.md)
when the first divergence points to a gradient that several ranks sum
together. It also serves step 2 (control) for any run that uses data
parallelism. The [training step](training-step.md) is already pinned on one
rank; what remains is the sum across ranks. DDP all-reduces the gradient in
buckets, FSDP and ZeRO reduce-scatter a flattened copy of it, and both reduce
a gradient norm and a mean loss across ranks. Each of those sums has an
order, and that order can depend on things that have nothing to do with the
parameter itself.

## The idea: the position in the message decides the order, and the degree decides which sums exist

**A small example about position.** Four ranks hold the gradients g0, g1, g2
and g3 of one parameter. A ring all-reduce cuts the message into chunks, and
the sum for each chunk starts at a different rank:

| Element in the chunk | Order of the sum (one example) |
|---|---|
| 0 | ((g1 + g2) + g3) + g0 |
| 2 | ((g3 + g0) + g1) + g2 |

If the same parameter sits at another offset in the message (because of
another bucket size, another parameter order, or one padding element in front
of it), its elements land in other chunks, so they are summed in another order
and get other bits. DDP even changes its own packing: its first iteration
puts every gradient in one bucket, later iterations rebuild the buckets, and
every restart starts over. Reduce-scatter has the same property through the
rank that owns a slice: the slice a rank keeps is summed starting at the next
rank, so resharding moves the order.

**A small example about degree.** Take 16 sequences. At data-parallel degree
d = 2, each rank runs a backward pass over 8 sequences, and the two partial
gradients are added. At d = 4, four partial gradients of 4 sequences each are
added. These are different sums, not the same sum in another order, and no
collective setting can make them equal.

<figure class="fx" data-fig="dp-tree" aria-label="A gradient summed at degree 2 and 4, per rank and with logical shards">
<div class="fx-stage">
<div class="fx-head"><span class="fx-label">16 sequences at data-parallel degree</span><span class="fx-swap"><span class="fx-chip" data-at="1" data-to="3">d = 2</span><span class="fx-chip is-key" data-at="3">d = 4</span></span></div>
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">one backward per rank</span><div class="fx-tree" style="--n:16">
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-band" data-c="0" style="--w:8">rank 0</span><span class="fx-band" data-c="1" style="--w:8">rank 1</span></div><div class="fx-level" data-at="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span><span class="fx-band" data-c="1" style="--w:4">rank 1</span><span class="fx-band" data-c="2" style="--w:4">rank 2</span><span class="fx-band" data-c="3" style="--w:4">rank 3</span></div></div>
<div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span><span class="fx-cell">10</span><span class="fx-cell">11</span><span class="fx-cell">12</span><span class="fx-cell">13</span><span class="fx-cell">14</span><span class="fx-cell">15</span></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-j" style="--w:8;--k:8"><span class="fx-node is-mono">Σ 0–7</span></span><span class="fx-j" style="--w:8;--k:8"><span class="fx-node is-mono">Σ 8–15</span></span></div><div class="fx-level" data-at="3"><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 0–3</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 4–7</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 8–11</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 12–15</span></span></div></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="2" data-to="3"><span class="fx-j" style="--w:16"><span class="fx-node is-mono is-key">x₂</span></span></div><div class="fx-level" data-at="4"><span class="fx-j" style="--w:16;--k:4"><span class="fx-node is-mono is-bad">x₄ ≠ x₂</span></span></div></div>
</div></div>
<div class="fx-row"><span class="fx-label">logical shards, at any d</span><div class="fx-tree" style="--n:16">
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
<li>One global batch holds 16 sequences.</li>
<li data-hold="2">At d = 2, each rank owns 8 sequences. With one backward per rank, each rank produces one partial sum. With logical shards, each rank sums its own subtree of one fixed tree.</li>
<li data-hold="1.5">The ranks then add their partial sums, which gives x₂. The logical shards finish the same fixed tree, which gives x.</li>
<li data-hold="2">At d = 4, each rank owns 4 sequences. With one backward per rank there are now four partial sums, so the sums themselves are different. With logical shards it is the same tree, and each rank simply holds a smaller subtree.</li>
<li data-hold="2">With one backward per rank, x₄ ≠ x₂: the result follows the degree. With logical shards, the tree is the same, and so are the bits of x.</li>
</ol>
<figcaption>Figure: with one backward per rank, the partial sums at d = 2 and d = 4 depend on the degree. With logical shards, the same tree is summed at every degree.</figcaption>
</figure>

**Rule.** The order of a sum across ranks must depend only on the element and
on a fixed logical split of the batch, never on the element's position in the
message, on which rank owns it, or on the degree. This is part 0's principle
at L5: the parallel degree may change who computes a partial sum, but not
which partial sums exist.

## The step

### Fix the order of every sum across ranks

**Goal:** a parameter's gradient, the gradient norm, the loss and the updated
weights do not depend on the bucketing, on the sharding layout or on the
degree.

**How:** pick the options your run needs. The code is in
[training_dp.py](../../examples/training_dp/training_dp.py) (the ordered
collectives, the DDP comm hook `Worker.ordered_hook`, logical shards and the
logical norm), and every comparison is made bit for bit with
[bi_check.py](../../tools/bi_check.py).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| `NCCL_ALGO=Tree` for all-reduce | One gradient per parameter whatever the bucketing, at a fixed degree | small to moderate for large messages, large for small ones | DDP at a fixed degree; the quickest fix |
| An ordered all-reduce (an all-to-all, then a sum in rank order in fp32, then an all-gather; installed as a DDP comm hook) | The same, and independent of NCCL's algorithm | small to moderate on one node | You need it to hold under any NCCL setting |
| An ordered reduce-scatter (rank r receives slice r of every message and adds the d slices in rank order) | FSDP-style gradients that do not depend on padding, wrapping or resharding | moderate (under 2x) on one node; across nodes, use the two-level route | Sharded training; any change of layout |
| Logical shards: S fixed micro-batches, one backward pass for each, and one pairwise tree over the shard index that also covers the sum across ranks | Identical gradients, weights and Adam state at every degree; elastic resume | moderate with several sequences per shard; large when the shards are tiny | The degree changes, or runs at two different degrees must agree |
| A logical norm and loss: fixed blocks, sums of squares in fp64, and a pairwise tree; the loss as a tree over the shard losses | A clipping norm and a loss that depend only on the values | moderate for the norm itself; small within a whole step | Always, when you use logical shards |
| Single-tensor AdamW (`foreach=False, fused=False`) | An elementwise update, so replicated and sharded weights get the same bits | small | Always; pin one code path |

Logical shards need the [training step](training-step.md) to be pinned
first: each shard's backward pass must repeat itself on every rank, which with
stock operators needs strict deterministic mode. Across two nodes, route the
exchange within each node first (the two-level route); a flat all-to-all over
the slower link between nodes costs much more.

**Success:** on every rank, the gradients, losses, norms, weights and Adam
state equal those of the d = 1 run bit for bit, and an elastic resume at
another degree matches a run that never stopped.

This was shown with fp32 gradients, on up to 16 GPUs on two nodes, both with
replicated weights and with FSDP emulated (a flat gradient, reduce-scatter and
a sharded AdamW; the framework's own FSDP was not run). Tree and the ordered
collectives removed the effect of bucketing, the ordered reduce-scatter
removed the effect of which rank owns a slice, and logical shards gave
identical results at every degree tested and across an elastic resume
([report](../../examples/training_dp/REPORT.md#what-this-shows)).

### Find the cause of a first divergence in a gradient sum

| What you observe | Cause |
|---|---|
| The gradients change after you change the bucket size, the parameter order or the model, at the same degree | the order of a ring all-reduce follows the parameter's offset in the message |
| The first iteration, or the first one after a restart, differs from the rest | DDP's first iteration puts every gradient in one bucket |
| Every element that differs has moved to another owning rank | the order of a reduce-scatter follows the owning rank |
| Almost every element differs at another degree, under any collective | the backward pass on each rank is part of the arithmetic, so use logical shards |
| The clipping norm differs only for some placements | NCCL adds up the sums of squares of the shards |

## What can go wrong

!!! warning "DDP is not even consistent with itself"
    DDP's first iteration after it is constructed uses a single bucket, and
    every restart constructs DDP again. So under Ring, a restart alone changes
    the gradient. Use Tree or the ordered hook, not a bucket-size setting.

!!! warning "Tree does not apply to reduce-scatter"
    `NCCL_ALGO=Tree` changes only all-reduce, so sharded training still
    follows the owning rank. Use the ordered reduce-scatter for FSDP-style
    sharding.

!!! warning "Handing the sum across ranks back to NCCL"
    Logical shards fail from three ranks on if the last step of the tree is
    an NCCL all-reduce or reduce-scatter, whether Ring or Tree, because the
    tree is then no longer the same tree. Every sum across ranks must be part
    of the fixed tree.

!!! warning "A scalar that agrees by luck"
    A sum of a few dozen terms often rounds to the same value, so a stock
    sharded norm can agree for many placements. It still depends on the
    sharding, and it fails for others. Use the logical norm with its aligned
    blocks.

!!! warning "Logical shards do not give the batched gradient"
    Logical shards at d = 1 give a different result from one batched backward
    pass over the same sequences. Both are correct roundings of different
    arithmetic. Pick one of them as the reference, and use it everywhere.

## Summary

- Under Ring, a gradient's bits depend on its position in the message; under
  reduce-scatter, they depend on the rank that owns it. Tree or the ordered
  collectives fix the first, and an ordered reduce-scatter fixes the second.
- Another degree means another set of partial sums, so only logical shards
  with one fixed tree give L5 and an elastic resume.
- Norms and mean losses are sums across ranks too, so give them the same
  fixed tree.
- Pin the optimizer's code path.

## Go deeper

- [Report: data parallelism and FSDP-style sharding](../../examples/training_dp/REPORT.md)
  covers every packing, layout and placement, the timings, the designs that
  were rejected, and the limits.
- The code is in [training_dp.py](../../examples/training_dp/training_dp.py).
- [Collectives](../3-kernels/collectives.md) covers the same collectives
  inside tensor-parallel layers.
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md),
  recipe 10.
