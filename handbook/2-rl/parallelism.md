# Parallel degrees: the same bits at every TP size (L5)

## Where we are

After the [whole-model step](whole-model.md) one side gives a sequence the same
bits in every batch and serving mode, on one GPU. In RL the trainer and the
engine rarely run at the same tensor-parallel (TP) degree, and disaggregated
serving runs prefill at one degree and decode at another. This step makes the
result independent of the degree (L5).

## The idea

A row-parallel layer splits its reduction dimension K across ranks: each rank
computes a partial sum, and a collective adds the partials. Take four
products a, b, c, d:

| Degree | What is computed |
|---|---|
| TP 1 | ((a + b) + c) + d, one rounding at the end |
| TP 2 | round(a + b) + round(c + d): two partials, each rounded to bf16, then added |
| TP 4 | a + b + c + d in whatever order the all-reduce picks |

Each degree gives a different tree and a different number of roundings, so a
different number. Two ranks are already enough: x0 + x1 commutes, but the
partials themselves are not the unsharded sum.

The fix is a **logical TP size**: cut K into a fixed number C of chunks,
whatever the degree, and add them with one fixed tree. A rank owns an aligned
subtree of chunks, keeps its partials in fp32, the ranks exchange subtrees,
and every rank finishes the same tree and rounds once. The tree at TP 1 (C
chunks on one GPU) is the tree at TP C (one chunk per rank). The rule: the
reduction tree is a property of the model, not of the cluster. It is the L5
form of the batch-invariance principle, shown for one GEMM in
[collectives](../3-kernels/collectives.md).

<figure class="fx" data-fig="tp-tree" aria-label="One fixed reduction tree at TP 1, 2 and 4">
<div class="fx-stage">
<div class="fx-head"><span class="fx-label">K in C = 4 chunks; tensor-parallel degree</span><span class="fx-swap"><span class="fx-chip" data-to="3">TP 1</span><span class="fx-chip is-key" data-at="3" data-to="5">TP 2</span><span class="fx-chip is-key" data-at="5">TP 4</span></span></div>
<div class="fx-tree" style="--n:4">
<div class="fx-swap is-block"><div class="fx-level" data-to="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span></div><div class="fx-level" data-at="3" data-to="5"><span class="fx-band" data-c="0" style="--w:2">rank 0</span><span class="fx-band" data-c="1" style="--w:2">rank 1</span></div><div class="fx-level" data-at="5"><span class="fx-band" data-c="0" style="--w:1">rank 0</span><span class="fx-band" data-c="1" style="--w:1">rank 1</span><span class="fx-band" data-c="2" style="--w:1">rank 2</span><span class="fx-band" data-c="3" style="--w:1">rank 3</span></div></div>
<div class="fx-level"><span class="fx-cell">c0</span><span class="fx-cell">c1</span><span class="fx-cell">c2</span><span class="fx-cell">c3</span></div>
<div class="fx-level" data-at="1" data-hot="3 5"><span class="fx-j" style="--w:2"><span class="fx-node is-mono">c0 + c1</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-mono">c2 + c3</span></span></div>
<div class="fx-level" data-at="2" data-hot="4 6"><span class="fx-j" style="--w:4"><span class="fx-node is-mono">same tree, fp32</span></span></div>
<div class="fx-level" data-at="2" data-hot="4 6"><span class="fx-j" style="--w:4;--k:1"><span class="fx-node is-key">one cast to bf16: y</span></span></div>
</div>
<div class="fx-chain is-tight"><span class="fx-label">result</span><span class="fx-chip is-ok" data-at="2">TP 1: y</span><span class="fx-chip is-ok" data-at="4">TP 2: y</span><span class="fx-chip is-ok" data-at="6">TP 4: y</span></div>
</div>
<ol class="fx-say">
<li>K is cut into C = 4 chunks, whatever the degree. TP 1: one GPU holds all four.</li>
<li>The fixed tree adds pairs of chunks, in fp32,</li>
<li>then the two halves, and rounds once to bf16: y.</li>
<li data-hold="1.5">TP 2: each rank sums its aligned subtree, in fp32.</li>
<li data-hold="1.5">The ranks exchange subtrees, finish the same tree and round once: y again.</li>
<li data-hold="1.5">TP 4: one chunk per rank. The ranks exchange chunks and add them in the same tree.</li>
<li data-hold="2">Same tree, one rounding: TP 1, TP 2 and TP 4 give the same bits.</li>
</ol>
<figcaption>Figure: each rank sums an aligned subtree; the ranks finish the same tree and round once, so TP 1 and TP 2 give the same bits.</figcaption>
</figure>

Three more places sum across ranks and need the same treatment:
- the vocab-parallel log_softmax, whose sum of exponentials is split by
  vocabulary shard;
- the backward of a column-parallel layer, whose input gradient sums over
  output columns held by different ranks;
- the KV cache, which inherits any variance of the projections that wrote it.

Column-parallel layers need nothing: each output column is computed whole on
one rank by the same kernel. Once every part is invariant, a disaggregated
KV hand-over is a plain byte copy.

## The step

1. **Make every row-parallel sum logical-TP.**
   - Goal: the same bits at every degree for o_proj, down_proj and any other
     row-parallel layer.
   - How: the sharded model in
     [tp_model.py](../../examples/whole_model_tp/tp_model.py) (variant `inv`);
     the single-GEMM scheme in [collectives](../3-kernels/collectives.md).
   - Success: logits at TP t equal TP 1 bit for bit.
2. **Make the vocab-parallel log_softmax fixed-order.**
   - Goal: realized logprobs equal at every degree.
   - How: the max by an all-reduce (exact in any order); the sum of
     exponentials in fixed vocabulary chunks, added with a fixed tree.
   - Success: equal to the one-GPU log_softmax of the gathered logits.
3. **Fix the backward the same way.**
   - Goal: activation gradients equal at every degree.
   - How: the logical-TP reduction for column-parallel input gradients.
   - Success: input gradients equal TP 1 at every layer boundary.
4. **Check the engine's modes under TP, and the hand-over.**
   - Goal: L2, L3 and L5 together.
   - How: [whole_model_tp.py](../../examples/whole_model_tp/whole_model_tp.py)
     runs decode in a changing batch and chunked prefill at TP 2 and 4, and
     prefill at one degree with decode at another.
   - Success: every mode equals the TP 1 trainer forward; the received KV
     cache equals the sent one.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Stock TP (bf16 partials, library all-reduce) | Nothing across degrees; every realized logprob moves | free | Trainer and engine at the same degree, and no audit needed |
| Logical TP in the forward only | L5 for logprobs; the backward still varies | moderate (under 2x) for prefill; decode can approach 2x from the extra collectives | Rollout scoring only |
| Logical TP in forward and backward | L5 for logprobs and activation gradients | as above, plus the backward reductions | The trainer recomputes and trains at another degree than the engine |

Shown: bitwise-equal logits and realized logprobs across TP degrees up to two
nodes, composed with L2 and L3, through a disaggregated hand-over, and equal
activation gradients up to four GPUs, on one GPU type over PCIe
([report](../../examples/whole_model_tp/REPORT.md#results)). Not yet shown:
MoE, sequence parallelism, degrees that do not divide C, parameter gradients,
a production engine's own all-reduce.

## What can go wrong

!!! warning "Logical TP is a different arithmetic"
    A logical-TP model at TP 1 does not equal the plain one-GPU model. If only
    the engine adopts it, the trainer is now further away. Both sides must use
    the same C and the same tree.

!!! warning "The forward fix does not carry over to the backward"
    Column-parallel input gradients are their own cross-rank sum. With a
    library all-reduce they break L5 even when the forward is equal. Fix the
    backward reductions explicitly and test them.

!!! warning "A shard is a GEMM with another shape"
    A column-parallel shard has a smaller N, and a library GEMM may pick
    another kernel for it before any collective runs. Use an invariant GEMM on
    every shard, not only an invariant sum.

!!! warning "Doubling the collectives costs latency"
    An exchange of fp32 subtrees plus a gather replaces one all-reduce, so
    decode, which is latency-bound, slows down more than prefill. Fuse the
    exchange or route it in two levels across nodes before judging the cost.

## Summary

- A TP degree changes the reduction tree of every row-parallel sum.
- Fix the tree, not the degree: a logical TP size with fp32 partials and one
  cast gives the same bits at every degree.
- The vocab-parallel log_softmax and the backward need the same fix.
- With invariant parts, disaggregated prefill/decode is just a copy of the KV
  cache.

## Go deeper

- [Lab report](../../examples/whole_model_tp/REPORT.md): sharding table, the
  two variants, every result, cost by degree, limits.
- Code: [tp_model.py](../../examples/whole_model_tp/tp_model.py),
  [whole_model_tp.py](../../examples/whole_model_tp/whole_model_tp.py).
- The single-GEMM case: [collectives](../3-kernels/collectives.md).
- Next step: [sampling](sampling.md).
