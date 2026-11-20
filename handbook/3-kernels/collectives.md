# Collectives and tensor-parallel layers

## Where we are

On the [MoE page](moe.md), the traffic between GPUs only moved rows, and we
wrote the sum ourselves. A tensor-parallel (TP) layer is different. Each GPU
computes a partial product, and a collective adds the partial products inside
the communication library. The library, not our code, now chooses the order
of that sum, and the parallel degree itself changes which partial sums exist.
This makes collectives the hardest class, because both the batch (L2) and the
parallel degree (L5) reach the reduction.

## The idea: the communication library picks the order of the sum from the message

A sum all-reduce adds up each element across the GPUs one add at a time and
rounds after every add. With the ring algorithm, the library cuts the message
into chunks, and the sum for each chunk starts at a different GPU and walks
around the ring:

<figure class="fx" data-fig="ring" aria-label="Ring all-reduce on 4 GPUs with two chunks">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">element in chunk 0</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node" data-hot="1">GPU 1</span><span class="fx-note fx-mono" data-at="1">g1</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="2">+ GPU 2</span><span class="fx-note fx-mono" data-at="2">g1 + g2</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="3">+ GPU 3</span><span class="fx-note fx-mono" data-at="3">(g1 + g2) + g3</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node to-key" data-hot="4 5">+ GPU 0</span><span class="fx-note fx-mono" data-at="4">(…) + g0</span></span></div></div>
<div class="fx-row"><span class="fx-label">element in chunk 1</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node" data-hot="1">GPU 2</span><span class="fx-note fx-mono" data-at="1">g2</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="2">+ GPU 3</span><span class="fx-note fx-mono" data-at="2">g2 + g3</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="3">+ GPU 0</span><span class="fx-note fx-mono" data-at="3">(g2 + g3) + g0</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node to-key" data-hot="4 5">+ GPU 1</span><span class="fx-note fx-mono" data-at="4">(…) + g1</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>Each of 4 GPUs holds a value g for every element, and the ring cuts the message into chunks.</li>
<li>The sum for each chunk starts at a different GPU: chunk 0 starts at GPU 1, and chunk 1 at GPU 2.</li>
<li>At each step the next GPU adds its value and rounds,</li>
<li>and passes the partial sum on along the ring.</li>
<li data-hold="1.5">After 3 adds, each chunk has its sum, each computed in its own order.</li>
<li data-hold="2">Which chunk an element falls in depends on its offset and on the message size, so a bigger batch can move it to another chunk and change its bits.</li>
</ol>
<figcaption>Figure: ring all-reduce on 4 GPUs. The sum for each chunk starts at a different GPU.</figcaption>
</figure>

Which chunk an element falls in depends on its offset and on the message
size. In a TP layer the message is the output of the whole batch, so when the
batch grows or the row moves, the row's sum starts at a different GPU, and L2
fails. Other algorithms behave differently:

| Algorithm | Order of an element's sum | Does it depend on the batch? |
|---|---|---|
| Ring all-reduce | it starts at a GPU chosen by the element's chunk | yes |
| Tree all-reduce on one node | one chain, the same for every element | no |
| Ring reduce-scatter | fixed for each output chunk, and set by the GPU that owns the chunk | yes, when the GPU that owns a row changes with the batch |
| Any algorithm on two GPUs | x0 + x1 = x1 + x0 exactly | no |

**The parallel degree is a second dependence.** A row-parallel layer rounds
the partial product on each GPU and then sums the partial products. On 2 GPUs
the partial products cover two halves of K; on 4 GPUs they cover four
quarters. Different partial sums give a different total, so no library
setting can give L5.

**The rule for collectives.** Fix the order of the cross-GPU sum so that it
does not depend on the message, either by pinning a tree or by exchanging the
partial sums and adding them yourself in a fixed order. For L5, also fix the
partial sums themselves: a logical TP size C cuts K into C fixed chunks, and
those chunks are summed in one fixed tree, however many GPUs hold them.

## The steps

### 1. Find which collectives and settings change a row's result

- **Goal:** know which collectives and which settings change a row's bits.
- **How:** reduce a fixed slice on its own, and then inside messages of many
  sizes and at several positions, and record from the library's tuning log
  what it chose for each call.
  [examples/collectives_tp/collectives_tp.py](../../examples/collectives_tp/collectives_tp.py)
  does this, and includes an order diagnostic that explains how each
  element's sum was formed.
- **Success:** every failing element is explained by a known order
  ([report](../../examples/collectives_tp/REPORT.md#what-this-shows)).

### 2. Fix the order of the cross-GPU sum (L2)

- **Goal:** a row's cross-GPU sum is the same in any batch.
- **How:** follow [recipe 10](../../skills/batch-invariant-kernel/references/operator-recipes.md#10-collectives-tp--ep--sp).
- **Success:** the TP layer passes `check_batch`. This was shown on one node
  of 8 GPUs and on two nodes of 8 GPUs
  ([report](../../examples/collectives_tp/REPORT.md#what-the-second-node-changes)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| The library's default | L2 fails | free | never, if you need parity |
| Pin the ring algorithm, with any protocol | L2 fails | free | never: pinning the algorithm does not fix the chunking |
| Pin the tree algorithm (`NCCL_ALGO=Tree`) | L2 for all-reduce, on one node and on two | small to moderate, and large at some message sizes | you can only change settings and run on at most two nodes; it does not cover reduce-scatter |
| All-gather, then add the values locally in fp32 in GPU order | L2 | large | small messages, or as a reference |
| All-to-all, add locally in fp32 in GPU order, then all-gather | L2; its first half can replace reduce-scatter | small to moderate on one node | the default fallback on one node |
| A fixed order on two levels: first inside each node, then one exchange across nodes | L2 | moderate | across nodes |

The fallbacks change the arithmetic (they accumulate in fp32 and round once),
so for L4 both sides must use the same fallback.

### 3. Fix the partial sums themselves (L5)

- **Goal:** the same rows give the same bits at TP 1, 2, 4 and higher.
- **How:** choose a logical TP size C. K is cut into C fixed chunks, each GPU
  computes whole chunks with an fp32 output, and the partial sums are added in
  one fixed pairwise tree. A parallel degree that divides C gives each GPU
  whole chunks. Above C, split each chunk's output columns across GPUs, never
  its range of K.
- **Success:** every parallel degree gives the same result. This was shown on
  up to 16 GPUs across two nodes
  ([report](../../examples/collectives_tp/REPORT.md#what-the-second-node-changes)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Row-parallel with any fixed-order sum | L2, but the result changes with the degree | small to moderate | the degree never changes between runs |
| Column-parallel (all-gather only) | L5, because nothing is summed across GPUs | free | layers that can be split by columns |
| A logical TP size with a fixed tree | L5 at every degree that divides C, and above C by splitting columns | moderate to large | the trainer and the engine run at different degrees |

### 4. Replace reduce-scatter under sequence parallelism

Reduce-scatter sums each row in the order set by the GPU that owns it, and
under sequence parallelism the owner depends on the row's position. Replace it
with the all-to-all half of the fallback above; pinning the tree algorithm has
no effect on reduce-scatter.

### 5. Verify with three or more GPUs

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
along its parallel axes: three or more GPUs, a fixed slice inside messages of
many sizes and at many positions, every degree you deploy, and the library
settings you might meet in production.

## What can go wrong

!!! warning "Pinning the ring does not give invariance"
    Under a pinned ring, every protocol still started each element's sum at a
    GPU chosen by the element's chunk. Check a fixed slice inside many message
    sizes.

!!! warning "Pinning the tree has no effect on reduce-scatter"
    The library kept using the ring for reduce-scatter, so a layer with
    sequence parallelism still fails. Use the all-to-all fallback.

!!! warning "Two GPUs hide every order bug"
    A sum of two terms is the same in either order, so a test at TP 2 passes
    whatever the order. Test with three or more GPUs.

!!! warning "The pinned tree has been shown on at most two nodes"
    With two nodes the top level adds only two operands, and that gives the
    same result in either order. With three or more nodes the trees can differ
    from channel to channel; this is not yet shown.

!!! warning "Two logical sizes are two different arithmetics"
    Logical TP 8 and logical TP 16 give results that differ from each other
    and from the plain GEMM. Fix one C for the whole deployment.

## Summary

- The communication library chooses the order of a cross-GPU sum from the
  message; in a ring, the order follows the element's chunk.
- For L2, pin a tree (up to two nodes, all-reduce only), or exchange the
  partial sums and add them yourself in a fixed order.
- For L5, fix the partial sums with a logical TP size and one fixed tree.
- Test with three or more GPUs and with many message sizes.

## Go deeper

- Report: [collectives and TP linear layers](../../examples/collectives_tp/REPORT.md).
- Code: [examples/collectives_tp/collectives_tp.py](../../examples/collectives_tp/collectives_tp.py).
- Recipe: [10, collectives](../../skills/batch-invariant-kernel/references/operator-recipes.md#10-collectives-tp--ep--sp).
- Related: [data-parallel gradients](../1-pretraining/data-parallel.md), and a
  whole model at several TP degrees ([parallelism](../2-rl/parallelism.md)).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
