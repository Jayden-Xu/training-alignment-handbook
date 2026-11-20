# Any tensor-parallel degree (L5)

## Where we are

After the [whole-model step](whole-model.md), one side gives a sequence the
same bits in every batch and every serving mode, as long as it runs on one GPU.
In RL, however, the trainer and the engine rarely run at the same
tensor-parallel (TP) degree, and disaggregated serving even runs prefill at one
degree and decode at another. This step makes the result independent of the
degree (L5).

## The idea

A row-parallel layer splits its reduction dimension K across ranks. Each rank
computes a partial sum, and a collective then adds the partial sums together.
Take four products a, b, c and d:

| Degree | What is computed |
|---|---|
| TP 1 | ((a + b) + c) + d, rounded once at the end |
| TP 2 | round(a + b) + round(c + d): each rank's partial sum is rounded to bf16, and then the two are added |
| TP 4 | a + b + c + d, in whatever order the all-reduce chooses |

Each degree adds the products in a different tree and rounds a different
number of times, so each gives a different number. Two ranks are already
enough to cause this: x0 + x1 is the same as x1 + x0, but the partial sums
themselves already differ from the sum computed on one GPU.

The fix is a **logical TP size**. Whatever the degree, cut K into the same
fixed number C of chunks, and add the chunks with one fixed tree. Each rank
owns an aligned subtree of chunks and keeps its partial sums in fp32. The
ranks then exchange their subtrees, every rank finishes the same tree, and the
result is rounded once. The tree at TP 1, with all C chunks on one GPU, is
then the same tree as at TP C, with one chunk per rank. The rule is that the
reduction tree belongs to the model, not to the cluster. This is the L5 form
of the batch-invariance principle; [collectives](../3-kernels/collectives.md)
shows it for a single GEMM.

<figure class="fx" data-fig="tp-tree" aria-label="One fixed reduction tree at TP 1, 2 and 4">
<div class="fx-stage">
<div class="fx-tree" style="--n:4">
<div class="fx-lrow"><span class="fx-label">TP degree</span><div class="fx-chain is-tight"><span class="fx-chip" data-hot="0-2 6">TP 1<span class="fx-pass" data-at="2">&nbsp;→ y</span></span><span class="fx-chip" data-hot="3 4 6">TP 2<span class="fx-pass" data-at="4">&nbsp;→ y</span></span><span class="fx-chip" data-hot="5 6">TP 4<span class="fx-pass" data-at="6">&nbsp;→ y</span></span></div></div>
<div class="fx-lrow"><span class="fx-label">ranks</span><div class="fx-swap is-block"><div class="fx-level" data-to="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span></div><div class="fx-level" data-at="3" data-to="5"><span class="fx-band" data-c="0" style="--w:2">rank 0</span><span class="fx-band" data-c="1" style="--w:2">rank 1</span></div><div class="fx-level" data-at="5"><span class="fx-band" data-c="0">rank 0</span><span class="fx-band" data-c="1">rank 1</span><span class="fx-band" data-c="2">rank 2</span><span class="fx-band" data-c="3">rank 3</span></div></div></div>
<div class="fx-lrow"><span class="fx-label">K in 4 chunks</span><div class="fx-level fx-sweep" data-hot="0 5"><span class="fx-cell">c0</span><span class="fx-cell">c1</span><span class="fx-cell">c2</span><span class="fx-cell">c3</span></div></div>
<div class="fx-lrow"><span class="fx-label">fixed tree in fp32</span><div class="fx-level" data-at="1"><span class="fx-j" style="--w:2" data-hot="1 3 5"><span class="fx-node is-mono">c0 + c1</span></span><span class="fx-j" style="--w:2" data-hot="1 3 5"><span class="fx-node is-mono">c2 + c3</span></span></div></div>
<div class="fx-lrow"><span></span><div class="fx-level" data-at="2"><span class="fx-j" style="--w:4" data-hot="2 4 5"><span class="fx-node is-mono">(c0 + c1) + (c2 + c3)</span></span></div></div>
<div class="fx-lrow"><span class="fx-label">one rounding</span><div class="fx-level" data-at="2"><span class="fx-j" style="--w:4;--k:1" data-hot="2 4 6"><span class="fx-node is-key">bf16: y</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>K is cut into C = 4 chunks at every degree. At TP 1, one GPU holds all four.</li>
<li>The fixed tree first adds the chunks in pairs, in fp32,</li>
<li>then adds the two halves and rounds once to bf16, which gives y.</li>
<li data-hold="1.5">At TP 2, each rank sums its own aligned subtree in fp32.</li>
<li data-hold="1.5">The ranks exchange their subtrees, finish the same tree and round once, which gives y again.</li>
<li data-hold="1.5">At TP 4, each rank holds one chunk. The ranks exchange chunks and add them in the same tree.</li>
<li data-hold="2">The chunks, the tree and the single rounding are the same everywhere, so TP 1, TP 2 and TP 4 give the same bits.</li>
</ol>
<figcaption>Figure: the chunks and the tree are the same at every degree; only the way they are split across ranks changes. Each rank sums an aligned subtree and the result is rounded once, so TP 1, TP 2 and TP 4 give the same bits.</figcaption>
</figure>

Three more places sum across ranks and need the same treatment:

- the vocab-parallel log_softmax, whose sum of exponentials is split across
  vocabulary shards;
- the backward of a column-parallel layer, whose input gradient sums over
  output columns that live on different ranks;
- the KV cache, which inherits any variation of the projections that wrote
  it.

Column-parallel layers in the forward need nothing, because each output
column is computed in full on one rank by the same kernel. Once every part is
invariant, handing the KV cache from a prefill machine to a decode machine is
a plain byte copy.

## The step

1. **Make every row-parallel sum use the logical TP size.**
   - Goal: o_proj, down_proj and every other row-parallel layer give the same
     bits at every degree.
   - How: use the sharded model in
     [tp_model.py](../../examples/whole_model_tp/tp_model.py) (variant `inv`);
     [collectives](../3-kernels/collectives.md) describes the scheme for a
     single GEMM.
   - Success: the logits at TP t equal those at TP 1 bit for bit.
2. **Give the vocab-parallel log_softmax a fixed order.**
   - Goal: the logprobs of the sampled tokens are equal at every degree.
   - How: take the maximum with an all-reduce, which is exact in any order;
     compute the sum of exponentials over fixed vocabulary chunks and add the
     chunks with a fixed tree.
   - Success: the result equals a one-GPU log_softmax of the gathered logits.
3. **Fix the backward in the same way.**
   - Goal: the activation gradients are equal at every degree.
   - How: use the logical-TP reduction for the input gradients of
     column-parallel layers.
   - Success: at every layer boundary, the input gradients equal those at
     TP 1.
4. **Check the engine's serving modes under TP, and the KV hand-over.**
   - Goal: L2, L3 and L5 hold together.
   - How: [whole_model_tp.py](../../examples/whole_model_tp/whole_model_tp.py)
     runs decode in a changing batch and chunked prefill at TP 2 and 4, and
     also runs prefill at one degree followed by decode at another.
   - Success: every mode equals the trainer's forward at TP 1, and the KV
     cache that arrives equals the one that was sent.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Stock TP (bf16 partial sums, the library's all-reduce) | No agreement across degrees; the logprob of every sampled token moves | free | The trainer and the engine run at the same degree, and no audit is needed |
| Logical TP in the forward only | L5 for logprobs, but the backward still varies | moderate (under 2x) for prefill; decode can approach 2x because of the extra collectives | Only scoring rollouts |
| Logical TP in the forward and the backward | L5 for logprobs and activation gradients | as above, plus the backward reductions | The trainer recomputes and trains at a different degree from the engine |

What has been shown: the logits and the logprobs of the sampled tokens were
bitwise equal across TP degrees up to two nodes, together with L2 and L3 and
through a disaggregated hand-over, and the activation gradients were equal up
to four GPUs. This was on one GPU type connected over PCIe
([report](../../examples/whole_model_tp/REPORT.md#results)). Not yet shown:
MoE, sequence parallelism, degrees that do not divide C, parameter gradients,
and a production engine's own all-reduce.

## What can go wrong

!!! warning "Logical TP computes a different arithmetic"
    A model using logical TP at TP 1 does not equal the plain one-GPU model.
    If only the engine adopts it, the trainer ends up further away than
    before. Both sides must use the same C and the same tree.

!!! warning "Fixing the forward does not fix the backward"
    The input gradients of column-parallel layers are a separate sum across
    ranks. With a library all-reduce they break L5 even when the forward is
    equal. Fix the backward reductions explicitly, and test them.

!!! warning "A shard is a GEMM of a different shape"
    A column-parallel shard has a smaller N, and a library GEMM may choose a
    different kernel for it before any collective runs. Use an invariant GEMM
    on every shard, not only an invariant sum across ranks.

!!! warning "Doubling the collectives costs latency"
    An exchange of fp32 subtrees followed by a gather replaces a single
    all-reduce. Decode is bound by latency, so it slows down more than
    prefill. Fuse the exchange, or route it in two levels across nodes, before
    you judge the cost.

## Summary

- The TP degree changes the reduction tree of every row-parallel sum.
- Fix the tree, not the degree: a logical TP size with fp32 partial sums and a
  single cast gives the same bits at every degree.
- The vocab-parallel log_softmax and the backward need the same fix.
- Once every part is invariant, disaggregated prefill and decode only need a
  copy of the KV cache.

## Go deeper

- [Lab report](../../examples/whole_model_tp/REPORT.md): the sharding table,
  the two variants, every result, the cost at each degree, and the limits.
- Code: [tp_model.py](../../examples/whole_model_tp/tp_model.py),
  [whole_model_tp.py](../../examples/whole_model_tp/whole_model_tp.py).
- The single-GEMM case: [collectives](../3-kernels/collectives.md).
- Next step: [sampling](sampling.md).
