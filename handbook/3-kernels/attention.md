# Softmax attention

## Where we are

The [GEMM page](gemm-and-quantized.md) made each element's dot product
invariant by fixing how K is walked. Attention keeps those GEMMs and adds a
second reduction: each query sums over all the keys before it, with a running
softmax. Engines walk that reduction differently in each serving mode
(prefill, decode, chunked prefill), over a paged cache, in packed batches and
under a compiler. This page first fixes the walk over the keys, then the
backward pass, and then the serving layers that sit around the kernel.

## The idea: visit the keys in the same blocks in every mode

A query's output is a weighted sum over keys. Kernels visit the keys in blocks
and keep a running maximum and a running sum (online softmax), so the block
boundaries decide how the reduction is grouped. If the boundaries move, the
bits of the result move with them.

<figure class="fx" data-fig="kv-split" aria-label="One query's keys walked in fixed blocks and in split-KV decode">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">fixed blocks counted from position 0</span><div class="fx-tree" style="--n:10">
<div class="fx-level"><span class="fx-cell" style="--w:4" data-hot="1">keys 0–63</span><span class="fx-cell" style="--w:4" data-hot="2">keys 64–127</span><span class="fx-cell" style="--w:2" data-hot="3">128–t</span></div>
<div class="fx-level" data-at="1"><span class="fx-walk" style="--w:10">one pass, keeping a running max and sum</span></div>
<div class="fx-level" data-at="3"><span class="fx-j" style="--w:10;--k:1"><span class="fx-node is-ok">the same sums in every mode</span></span></div>
</div></div>
<div class="fx-row"><span class="fx-label">split-KV decode, where the batch decides the splits</span><div class="fx-tree" style="--n:10">
<div class="fx-swap is-block"><div class="fx-level" data-to="6"><span class="fx-cell" style="--w:5" data-hot="4">keys 0–t/2</span><span class="fx-cell" style="--w:5" data-hot="4">keys t/2–t</span></div><div class="fx-level" data-at="6"><span class="fx-cell" style="--w:3">0–t/3</span><span class="fx-cell" style="--w:3">t/3–2t/3</span><span class="fx-cell" style="--w:4">2t/3–t</span></div></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="5" data-to="6"><span class="fx-j" style="--w:10"><span class="fx-node is-bad">merge two partial results</span></span></div><div class="fx-level" data-at="6"><span class="fx-j" style="--w:10;--k:3"><span class="fx-node is-bad">merge three partial results</span></span></div></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>One query at position t sums over keys 0 to t, keeping a running max and sum.</li>
<li>With fixed blocks counted from position 0, it first visits keys 0–63,</li>
<li>then keys 64–127,</li>
<li data-hold="1.5">then keys 128–t. Prefill, decode and chunked prefill all walk these same blocks.</li>
<li data-hold="1.5">In split-KV decode, the batch size and the number of cores set the number of splits. Here there are two, computed at the same time.</li>
<li data-hold="1.5">The two partial results are merged, which is another grouping of the same sum.</li>
<li data-hold="2">In another batch there are three splits. The boundaries move, and the bits move with them.</li>
</ol>
<figcaption>Figure: the block boundaries decide how the reduction is grouped. If the boundaries move, the bits move.</figcaption>
</figure>

Three things can move the boundaries:

- **The mode.** Prefill, decode and chunked prefill often run different
  kernels. A decode kernel may also split the keys into parts and choose the
  number of parts from the batch size and the number of GPU cores (split-KV).
- **The layout.** A paged kernel that makes each block exactly one page ties
  the grouping to the page size. A kernel for packed batches that lays its
  blocks over the whole pack ties the grouping to the lengths of the
  neighbouring sequences.
- **The backend.** A library call can pick a backend from its arguments, so
  passing a mask instead of a causal flag can change the kernel that runs.

**The rule for attention.** Every query walks the keys in fixed-size blocks
counted from absolute position 0 of its own sequence. There is no split-KV,
or only splits of a fixed size, and every mode runs the same kernel, given
the query's absolute offset. A block that lies past the query's position is
fully masked and changes nothing: its rescale factor is exactly 1 and the
weight it adds is exactly 0. So a short decode call and a long prefill compute
exactly the same sums.

**The backward pass reduces in the other direction.** In the forward pass,
one program owns each output row. In the backward pass, a key's gradient is a
sum over every later query, and a query's gradient is a sum over every
earlier key. Common kernels add these contributions with atomics, so the
order follows timing and even L1 breaks. The fix is to use two passes, in
each of which every program owns one output and walks its sum in a fixed
order.

## The steps

### 1. In the forward pass, use the same key blocks in every mode

- **Goal:** prefill, decode and chunked prefill give the same bits at every
  position, and a sequence gives the same bits in any batch (L2, L3).
- **How:** use the fixed-block kernel `attention_fixed_blocks` in
  [tools/bi_kernels.py](../../tools/bi_kernels.py), which follows
  [recipe 4](../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv).
  Write the new keys and values into the cache before the kernel runs, so
  that decode reads them in the same way as prefill does.
- **Success:** `check_modes` and `check_batch` from
  [tools/bi_check.py](../../tools/bi_check.py) pass. This was shown on one
  GPU type, where the stock backend failed both checks
  ([report](../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Stock backend | L1, but fails L2 and L3 | free | never, if you need parity |
| Fixed-block kernel, the same kernel in every mode | L2, L3 | moderate in prefill (untuned); no slower in decode | the default fix |
| Split-KV with a fixed split size, for long decode | L2 and L3, if the split size is fixed in tokens | small | long contexts; not yet shown here |

### 2. Write the backward pass as two passes without atomics {#backward-dq-dk-dv}

- **Goal:** the same sequence gets the same gradients when it runs alone, in
  a larger batch and with right padding (L1, L2).
- **How:** use the atomic-free backward `attention_fixed_blocks_train` in
  [tools/bi_kernels.py](../../tools/bi_kernels.py). The first pass gives each
  program a block of keys. The program sums over the query heads of its group
  and over the later queries in a fixed order, and writes dK and dV once. The
  second pass gives each program a block of queries. The program walks the
  keys from position 0, as the forward pass does, and writes dQ once.
- **Success:** L1 holds over repeated runs, and L2 holds alone, batched and
  padded, with a zero output gradient on the padding. This was shown on one
  GPU type ([report](../../examples/attention_backward/REPORT.md#results)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Library backward with default settings | fails L1 because of atomics | free | never, if you need parity |
| Library backward in strict deterministic mode | L1 only, because its forward pass already varies with the batch | moderate to large | when L1 is all you need |
| Memory-efficient library backend in strict deterministic mode | L1 and L2 at the shapes tested | large | when you can only change settings; inference engines do not run this backend, so it gives no L4 |
| Atomic-free two-pass kernel | L1 and L2, including right padding | moderate | the default fix |

Right padding is handled exactly here. Causality keeps the padding out of
the real tokens' forward pass, and the padding rows carry a zero output
gradient, so their contributions are exact zeros. Left padding shifts the
absolute positions, so it is not covered.

### 3. Keep the bits unchanged under paging, packing, compilation and CUDA graphs {#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs}

- **Goal:** the layers that an engine adds around the kernel do not change
  the bits.
- **How:** use the paged and variable-length kernels and their checks in
  [examples/attention_serving](../../examples/attention_serving/serving_kernels.py),
  and [recipe 12](../../skills/batch-invariant-kernel/references/operator-recipes.md#12-deployment-wrappers-torchcompile-and-cuda-graphs)
  for the compiler and CUDA graphs.
- **Success:** every page size, every pack offset, and every compiled or
  graphed variant gives the same bits as the kernel on a contiguous cache.
  This was shown on one GPU type
  ([report](../../examples/attention_serving/REPORT.md#what-the-serving-layer-results-show)).

| Layer | What to do | Cost class |
|---|---|---|
| Paged KV | keep the compute blocks on absolute positions and gather the pages into them, whatever the page size | small to moderate, depending on the page size and on pipelining |
| Packed batch of variable-length sequences | start each sequence's blocks at its own first token, not on the grid of the pack | small |
| `torch.compile` | register every op that rounds (norms and activations, not only the kernels) as an opaque custom op, so that the compiler cannot move rounding points | small under CUDA graphs; a large host overhead in eager mode at small batches |
| CUDA graphs | use invariant kernels; a padded batch then runs at the bucket's size and still gives the same bits | free |

### 4. Verify along the axes that matter for attention

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
with the axes that matter for attention: every mode at chunk sizes that do
not divide the block size, several page sizes with pages placed at random,
pack offsets that are not multiples of the block size, compiled against eager
execution, and the backward pass with padding.

## What can go wrong

!!! warning "Warn-only deterministic mode is not deterministic"
    With `warn_only=True` the library prints a warning and then runs its
    nondeterministic backward anyway, so the gradients still change from run
    to run. Use `warn_only=False`, which raises an error on any op that has
    no deterministic version.

!!! warning "A backward pass cannot be more invariant than its forward pass"
    A deterministic backward on top of a forward that varies with the batch
    still varies with the batch. Fix the forward pass first.

!!! warning "A test at a single page size cannot catch a page-dependent kernel"
    If a kernel's block is exactly one page, it is consistent across modes at
    any one page size, but every page size gives a different answer. Test
    several page sizes.

!!! warning "Pack offsets that are multiples of the block pass by luck"
    A kernel that follows the pack's grid is right exactly when a sequence
    happens to start on a block boundary. Test offsets that are not multiples
    of the block size.

!!! warning "The compiler moves rounding points"
    Compiled norms and activations stayed batch-invariant but no longer
    matched eager execution. If an engine must match an eager trainer, make
    every op that rounds opaque to the compiler.

## Summary

- Attention reduces over the keys, and the block boundaries decide the
  grouping.
- Use fixed blocks counted from absolute position 0, no split-KV, and the
  same kernel in every mode.
- The backward pass needs two passes without atomics; strict deterministic
  mode on its own gives L1, not L2.
- Pages, packs, compilers and CUDA graphs leave the bits unchanged only if
  the compute blocks ignore them.

## Go deeper

- Reports: [kernel cases](../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)
  (the forward pass), [attention backward](../../examples/attention_backward/REPORT.md)
  and [serving layers](../../examples/attention_serving/REPORT.md).
- Code: [tools/bi_kernels.py](../../tools/bi_kernels.py),
  [examples/attention_backward/attn_bwd.py](../../examples/attention_backward/attn_bwd.py),
  [examples/attention_serving/serving_cases.py](../../examples/attention_serving/serving_cases.py).
- Recipes: [4, softmax attention](../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv),
  [12, deployment wrappers](../../skills/batch-invariant-kernel/references/operator-recipes.md#12-deployment-wrappers-torchcompile-and-cuda-graphs).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
