# Collectives and tensor-parallel layers

## Where we are

In [MoE](moe.md), the cross-GPU traffic only moved rows, and we wrote the sum
ourselves. A tensor-parallel (TP) layer is different: each GPU computes a
partial product, and a collective adds the partials inside the communication
library. The order of that sum is now chosen by the library, not by our code,
and the parallel degree itself changes which partials exist. That makes
collectives the hardest class: the batch (L2) and the degree (L5) both reach
the reduction.

## The idea: the library chooses the order from the message

A sum all-reduce adds each element over the GPUs one add at a time, rounding
after every add. With the ring algorithm the message is cut into chunks, and
each chunk's sum starts at a different GPU and walks the ring:

<figure class="fig">
<div class="fig-rows">
<div class="fig-group"><span class="fig-label">element in chunk 0</span><div class="fig-chain"><span class="fig-node">GPU 1</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">+ GPU 2</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">+ GPU 3</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">+ GPU 0</span></div></div>
<div class="fig-group"><span class="fig-label">element in chunk 1</span><div class="fig-chain"><span class="fig-node">GPU 2</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">+ GPU 3</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">+ GPU 0</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">+ GPU 1</span></div></div>
</div>
<figcaption>Ring all-reduce on 4 GPUs: each chunk's sum starts at a different GPU.</figcaption>
</figure>

Which chunk an element falls in depends on its offset and on the message size.
In a TP layer the message is the whole batch's output, so when the batch grows
or the row moves, the row's sum starts at another GPU (L2 fails). Other
algorithms behave differently:

| Algorithm | Order of an element's sum | Depends on the batch? |
|---|---|---|
| Ring all-reduce | starts at a GPU set by the element's chunk | yes |
| Tree all-reduce on one node | one chain, the same for every element | no |
| Ring reduce-scatter | fixed for each output chunk, set by which GPU owns it | yes, when the owner of a row changes with the batch |
| Any, with two GPUs | x0 + x1 = x1 + x0 exactly | no |

**The parallel degree is a second dependence.** A row-parallel layer rounds
each GPU's partial product, then sums the partials. On 2 GPUs the partials are
two halves of K; on 4 GPUs they are four quarters. Different partials give a
different sum, so no library setting gives L5.

**The rule for collectives.** Fix the order of the cross-GPU sum so it does not
depend on the message (a pinned tree, or exchange the partials and sum them in
a fixed order yourself), and for L5 fix the partials themselves: a logical TP
size C cuts K into C fixed chunks, and those chunks are summed in one fixed
tree however many GPUs hold them.

## The steps

### 1. Locate the variance

- **Goal:** know which collectives and settings change a row's bits.
- **How:** reduce a fixed slice alone and inside messages of many sizes, at
  several positions, and record the library's per-call choice from its tuning
  log; [examples/collectives_tp/collectives_tp.py](../../examples/collectives_tp/collectives_tp.py)
  does this, with an order diagnostic that explains each element's sum.
- **Success:** every failing element explained by a known order
  ([report](../../examples/collectives_tp/REPORT.md#what-this-shows)).

### 2. Fix the order of the sum (L2)

- **Goal:** a row's cross-GPU sum is the same in any batch.
- **How:** [recipe 10](../../skills/batch-invariant-kernel/references/operator-recipes.md#10-collectives-tp--ep--sp).
- **Success:** the TP layer passes `check_batch`. Shown on one node of 8 GPUs
  and on two nodes of 8 GPUs ([report](../../examples/collectives_tp/REPORT.md#what-the-second-node-changes)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Library default | fails L2 | free | never for parity |
| Ring pinned, any protocol | fails L2 | free | never: pinning the algorithm does not fix the chunking |
| Tree pinned (`NCCL_ALGO=Tree`) | L2 for all-reduce, on one and two nodes | small to moderate; large at some message sizes | configure-only, up to two nodes; does not reach reduce-scatter |
| All-gather, then a local fp32 sum in GPU order | L2 | large | small messages, or as a reference |
| All-to-all, local fp32 sum in GPU order, all-gather | L2; its first half replaces reduce-scatter | small to moderate on one node | the default fallback on one node |
| Two-level fixed order: inside each node, then one exchange across nodes | L2 | moderate | across nodes |

The fallbacks change the arithmetic (fp32 accumulation, one rounding), so
both sides must use the same one for L4.

### 3. Fix the partials (L5)

- **Goal:** the same rows give the same bits at TP 1, 2, 4 and more.
- **How:** a logical TP size C: K is cut into C fixed chunks, each GPU computes
  whole chunks with an fp32 output, and the partials are summed in one fixed
  pairwise tree. Degrees that divide C give each GPU whole chunks; above C,
  split each chunk's output columns, never its K range.
- **Success:** one result at every degree. Shown up to 16 GPUs on two nodes
  ([report](../../examples/collectives_tp/REPORT.md#what-the-second-node-changes)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Row-parallel with any fixed-order sum | L2; changes with the degree | small to moderate | the degree never changes between runs |
| Column-parallel (all-gather only) | L5: no sum across GPUs | free | layers that can be column-split |
| Logical TP size, fixed tree | L5 at every degree that divides C, and above C by column split | moderate to large | trainer and engine run different degrees |

### 4. Sequence parallelism

Reduce-scatter gives each row the order of whichever GPU owns it, and under
sequence parallelism the owner follows the row's position. Replace it with the
all-to-all half of the fallback; the pinned tree does not reach it.

### 5. Verify with three or more GPUs

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
parallel axes: three or more GPUs, a fixed slice in messages of many sizes and
positions, every degree you deploy, and the library settings you might meet.

## What can go wrong

!!! warning "Pinning the ring is not invariance"
    Every protocol under a pinned ring still started each element's sum at a
    GPU set by its chunk. Check a fixed slice inside many message sizes.

!!! warning "The pinned tree does not reach reduce-scatter"
    The library kept using the ring for reduce-scatter. A layer with sequence
    parallelism still fails; use the all-to-all fallback.

!!! warning "Two GPUs hide every order bug"
    A two-term sum commutes. A TP-2 test passes any order; test with three or
    more GPUs.

!!! warning "The pinned tree is shown up to two nodes"
    With two nodes the top level adds two operands, which commutes. With three
    or more, the trees can differ per channel; not yet shown.

!!! warning "Two logical sizes are two arithmetics"
    Logical TP 8 and 16 differ from each other and from the plain GEMM. Fix one
    C for the whole deployment.

## Summary

- The communication library chooses the order of a cross-GPU sum from the
  message; a ring's order follows the element's chunk.
- For L2: pin a tree (up to two nodes, all-reduce only) or exchange partials
  and sum them in a fixed order yourself.
- For L5: fix the partials with a logical TP size and one fixed tree.
- Test with three or more GPUs and many message sizes.

## Go deeper

- Report: [collectives and TP linear layers](../../examples/collectives_tp/REPORT.md).
- Code: [examples/collectives_tp/collectives_tp.py](../../examples/collectives_tp/collectives_tp.py).
- Recipe: [10, collectives](../../skills/batch-invariant-kernel/references/operator-recipes.md#10-collectives-tp--ep--sp).
- Related: [data-parallel gradients](../1-pretraining/data-parallel.md) and a
  whole model at several TP degrees ([parallelism](../2-rl/parallelism.md)).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
