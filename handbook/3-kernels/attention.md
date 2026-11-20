# Softmax attention: forward, backward and serving layers

## Where we are

The [GEMM page](gemm-and-quantized.md) made one dot product per element
invariant by fixing how K is walked. Attention keeps those GEMMs and adds a
second reduction: each query sums over all earlier keys, with a running
softmax. That reduction is walked differently in each serving mode (prefill,
decode, chunked prefill), over a paged cache, in packed batches and under a
compiler. This page fixes the walk over keys, then the backward, then the
serving layers around the kernel.

## The idea: walk the keys in the same blocks in every mode

A query's output is a weighted sum over keys. Kernels visit keys in blocks and
keep a running maximum and sum (online softmax), so the block boundaries are
the grouping of the reduction. If the boundaries move, the bits move.

<figure class="fx" data-fig="kv-split" aria-label="One query's keys walked in fixed blocks and in split-KV decode">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">fixed blocks from position 0</span><div class="fx-tree" style="--n:10">
<div class="fx-level"><span class="fx-cell" style="--w:4" data-hot="1">keys 0–63</span><span class="fx-cell" style="--w:4" data-hot="2">keys 64–127</span><span class="fx-cell" style="--w:2" data-hot="3">128–t</span></div>
<div class="fx-level" data-at="1"><span class="fx-walk" style="--w:10">one walk, a running max and sum</span></div>
<div class="fx-level" data-at="3"><span class="fx-j" style="--w:10;--k:1"><span class="fx-node is-ok">the same sums in every mode</span></span></div>
</div></div>
<div class="fx-row"><span class="fx-label">split-KV decode: splits chosen from the batch</span><div class="fx-tree" style="--n:10">
<div class="fx-swap is-block"><div class="fx-level" data-to="6"><span class="fx-cell" style="--w:5" data-hot="4">keys 0–t/2</span><span class="fx-cell" style="--w:5" data-hot="4">keys t/2–t</span></div><div class="fx-level" data-at="6"><span class="fx-cell" style="--w:3">0–t/3</span><span class="fx-cell" style="--w:3">t/3–2t/3</span><span class="fx-cell" style="--w:4">2t/3–t</span></div></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="5" data-to="6"><span class="fx-j" style="--w:10"><span class="fx-node is-bad">merge two partials</span></span></div><div class="fx-level" data-at="6"><span class="fx-j" style="--w:10;--k:3"><span class="fx-node is-bad">merge three partials</span></span></div></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>One query at position t sums over keys 0 to t, with a running max and sum.</li>
<li>Fixed blocks counted from position 0: keys 0–63 first,</li>
<li>then keys 64–127,</li>
<li data-hold="1.5">then 128–t. Prefill, decode and chunked prefill all walk these blocks.</li>
<li data-hold="1.5">Split-KV decode: the batch size and core count set the number of splits; here two, summed at once.</li>
<li data-hold="1.5">The two partials are merged: another grouping of the same sum.</li>
<li data-hold="2">Another batch, three splits: the boundaries move, and so do the bits.</li>
</ol>
<figcaption>Figure: the block boundaries are the grouping of the reduction. If the boundaries move, the bits move.</figcaption>
</figure>

Three things move the boundaries:

- **The mode.** Prefill, decode and chunked prefill often run different
  kernels, and a decode kernel may split the keys into parts whose number is
  chosen from the batch size and GPU core count (split-KV).
- **The layout.** A paged kernel that makes each block one page ties the
  grouping to the page size; a packed kernel that lays blocks over the whole
  pack ties it to the neighbours' lengths.
- **The backend.** A library call can pick a backend from its arguments, so
  passing a mask instead of a causal flag changes the kernel.

**The rule for attention.** Every query walks the keys in fixed-size blocks
counted from absolute position 0 of its own sequence, with no split-KV (or
splits of a fixed size), and every mode runs the same kernel with the query's
absolute offset. A block past the query's position is fully masked and is an
exact no-op: the rescale factor is exactly 1 and the added weight exactly 0. So
a short decode call and a long prefill compute the same sums.

**The backward reduces the other way.** In the forward, one program owns each
output row. In the backward, a key's gradient sums over every later query, and
a query's gradient sums over every earlier key. Common kernels add those shares
with atomics, so the order follows timing and even L1 breaks. The fix is two
passes in which each program owns one output and walks its sum in a fixed
order.

## The steps

### 1. Forward: fixed key blocks in every mode

- **Goal:** prefill, decode and chunked prefill give the same bits per
  position, and a sequence gives the same bits in any batch (L2, L3).
- **How:** the fixed-block kernel `attention_fixed_blocks` in
  [tools/bi_kernels.py](../../tools/bi_kernels.py), following
  [recipe 4](../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv).
  Write the new keys and values into the cache before the kernel runs, so
  decode reads them like prefill does.
- **Success:** `check_modes` and `check_batch` from
  [tools/bi_check.py](../../tools/bi_check.py) pass. Shown on one GPU type,
  where the stock backend failed both
  ([report](../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Stock backend | L1; fails L2 and L3 | free | never for parity |
| Fixed-block kernel, one kernel for every mode | L2, L3 | moderate in prefill (untuned); no slower in decode | the default fix |
| Fixed-size split-KV for long decode | L2, L3 if the split size is fixed in tokens | small | long contexts; not yet shown here |

### 2. Backward: two passes, no atomics

- **Goal:** the same sequence gets the same gradients alone, in a larger batch,
  and right-padded (L1, L2).
- **How:** the atomic-free backward `attention_fixed_blocks_train` in
  [tools/bi_kernels.py](../../tools/bi_kernels.py). One pass gives each program
  a block of keys; it sums over the query heads of its group and over later
  queries in a fixed order and writes dK and dV once. A second pass gives each
  program a block of queries; it walks the keys from position 0, as the
  forward does, and writes dQ once.
- **Success:** L1 over repeats and L2 alone, batched and padded, with zero
  output gradient on the padding. Shown on one GPU type
  ([report](../../examples/attention_backward/REPORT.md#results)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Library backward, default | fails L1 (atomics) | free | never for parity |
| Library backward, strict deterministic mode | L1 only: its forward already varies with the batch | moderate to large | when L1 is all you need |
| Memory-efficient library backend, strict deterministic | L1 and L2 at the shapes tested | large | configure-only; inference engines do not run it, so no L4 |
| Atomic-free two-pass kernel | L1, L2, right padding included | moderate | the default fix |

Right padding is exact here: causality keeps padding out of the real tokens'
forward, and padding rows carry zero output gradient, so their contributions
are exact zeros. Left padding shifts absolute positions, so it is not covered.

### 3. Serving layers: pages, packs, compiler, graphs

- **Goal:** the layers an engine adds around the kernel keep the bits.
- **How:** the paged and varlen kernels and their checks in
  [examples/attention_serving](../../examples/attention_serving/serving_kernels.py);
  [recipe 12](../../skills/batch-invariant-kernel/references/operator-recipes.md#12-deployment-wrappers-torchcompile-and-cuda-graphs)
  for the compiler and graphs.
- **Success:** every page size, pack offset, compiled and graphed variant gives
  the contiguous kernel's bits. Shown on one GPU type
  ([report](../../examples/attention_serving/REPORT.md#what-the-serving-layer-results-show)).

| Layer | What to do | Cost class |
|---|---|---|
| Paged KV | keep the compute block on absolute positions; gather pages into it, whatever the page size | small to moderate, depending on page size and pipelining |
| Packed variable-length batch | start each sequence's blocks at its own first token, not on the pack's grid | small |
| `torch.compile` | register every op that rounds (norms, activations, not only the kernels) as an opaque custom op, so the compiler cannot move rounding points | small under CUDA graphs; large host overhead in eager at small batches |
| CUDA graphs | use invariant kernels; a padded batch then runs at the bucket's size with the same bits | free |

### 4. Verify

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
with the attention axes: modes at chunk sizes that do not divide the block,
several page sizes with random page placement, pack offsets that are not
multiples of the block, compiled against eager, and the backward with padding.

## What can go wrong

!!! warning "Warn-only deterministic mode is not deterministic"
    With `warn_only=True` the library warns and then runs its nondeterministic
    backward. The gradients still change run to run; use `warn_only=False`,
    which raises on any op without a deterministic version.

!!! warning "A backward cannot be more invariant than its forward"
    A deterministic backward on a batch-variant forward is still batch-variant.
    Fix the forward first.

!!! warning "One page size agrees with itself"
    A kernel whose block equals the page is consistent across modes at any one
    page size, and every page size gives a different answer. Test several.

!!! warning "Pack offsets that are block multiples pass by luck"
    A kernel on the pack's grid is right exactly when a sequence starts on a
    block boundary. Test offsets that are not multiples of the block.

!!! warning "The compiler moves rounding points"
    Compiled norms and activations stayed batch-invariant but differed from
    eager. Make every rounding op opaque if an engine must match an eager
    trainer.

## Summary

- Attention reduces over keys; the block boundaries are the grouping.
- Fixed blocks from absolute position 0, no split-KV, one kernel in every mode.
- The backward needs two atomic-free passes; strict deterministic mode alone
  gives L1, not L2.
- Pages, packs, compilers and graphs keep the bits only if the compute blocks
  ignore them.

## Go deeper

- Reports: [kernel cases](../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)
  (forward), [attention backward](../../examples/attention_backward/REPORT.md),
  [serving layers](../../examples/attention_serving/REPORT.md).
- Code: [tools/bi_kernels.py](../../tools/bi_kernels.py),
  [examples/attention_backward/attn_bwd.py](../../examples/attention_backward/attn_bwd.py),
  [examples/attention_serving/serving_cases.py](../../examples/attention_serving/serving_cases.py).
- Recipes: [4, softmax attention](../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv),
  [12, deployment wrappers](../../skills/batch-invariant-kernel/references/operator-recipes.md#12-deployment-wrappers-torchcompile-and-cuda-graphs).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
