# Linear attention (gated delta rule)

## Where we are

The [MLA page](mla-and-sparse.md) showed that when the same math has two
equal forms, every mode has to make the same choice between them. Linear
attention faces that choice in a sharper form. Training and prefill usually
run a chunked algorithm, while decode runs a recurrent step. Here the result
is also carried from token to token in a state, so what matters is where the
chunk boundaries fall.

## The idea: the chunk boundaries decide how the state's sum is grouped

A linear attention layer (here Gated DeltaNet) keeps a fixed-size state S for
each sequence and head, and updates it with every token. The recurrent form
applies one update per token. The chunked form takes C tokens at a time,
combines them with matrix products, and carries the state from one chunk to
the next.

Take ten tokens and chunks of 4:

<figure class="fx" data-fig="chunks" aria-label="Where the chunk boundaries fall for three ways of calling the layer and for the recurrent form">
<div class="fx-stage">
<div class="fx-rows" style="--n:10">
<div class="fx-row"><span class="fx-label">ten tokens, C = 4</span><div class="fx-track"><div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span></div><span class="fx-verdict"></span></div></div>
<div class="fx-row"><span class="fx-label">one prefill of tokens 0–9</span><div class="fx-track"><div class="fx-level fx-sweep" data-at="1" data-hot="1"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span" style="--w:2">{8–9}</span></div><span class="fx-verdict"></span></div></div>
<div class="fx-row"><span class="fx-label">prefill of 0–7, then one decode call per token</span><div class="fx-track"><div class="fx-level" data-at="2" data-hot="3"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span to-bad" style="--w:1">{8}</span><span class="fx-span to-bad" style="--w:1">{9}</span></div><span class="fx-verdict"><span class="fx-node is-bad" data-at="3">the bits depend on the call sizes</span></span></div></div>
<div class="fx-row"><span class="fx-label">aligned: keep the state at token 8 and recompute the unfinished chunk</span><div class="fx-track"><div class="fx-swap is-block"><div class="fx-level" data-at="4" data-to="5"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span" style="--w:1">{8}</span></div><div class="fx-level" data-at="5" data-hot="5"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span to-ok" style="--w:2">{8–9}</span></div></div><span class="fx-verdict"><span class="fx-node is-ok" data-at="5">same as prefill</span></span></div></div>
<div class="fx-row"><span class="fx-label">recurrent: one token per step</span><div class="fx-track"><div class="fx-level fx-sweep is-unit" data-at="6" data-hot="6"><span class="fx-span" style="--w:1">0</span><span class="fx-span" style="--w:1">1</span><span class="fx-span" style="--w:1">2</span><span class="fx-span" style="--w:1">3</span><span class="fx-span" style="--w:1">4</span><span class="fx-span" style="--w:1">5</span><span class="fx-span" style="--w:1">6</span><span class="fx-span" style="--w:1">7</span><span class="fx-span" style="--w:1">8</span><span class="fx-span" style="--w:1">9</span></div><span class="fx-verdict"><span class="fx-node is-ok" data-at="6">same bits in every mode</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>Ten tokens, in chunks of C = 4. The chunks decide how the state's sum is grouped.</li>
<li>One prefill of tokens 0–9 uses the chunks {0–3}, {4–7} and {8–9}.</li>
<li data-hold="1.5">Now prefill tokens 0–7 and then call decode once per token. Each call starts its own chunk, first {8} and then {9}.</li>
<li data-hold="1.5">Token 9 is summed in a different grouping, so its output and the state differ in their last bits.</li>
<li data-hold="1.5">The aligned version keeps the state at token 8, which is a chunk boundary, and recomputes the unfinished chunk when token 9 arrives.</li>
<li data-hold="1.5">That gives {8–9} again, as in prefill, so the bits are the same.</li>
<li data-hold="2">The recurrent form makes one update per token in every mode, so it gives the same bits at no extra cost.</li>
</ol>
<figcaption>Figure: a chunked form gives the same result in every mode only when every chunk starts at an absolute multiple of C. The recurrent form always does.</figcaption>
</figure>

In the second pattern the same terms are summed in a different grouping, so
token 9's output and the state differ in their last bits.

Three facts follow from this:

- **The batch is not the problem.** Each program owns one sequence and one
  head, and no reduction crosses sequences, so both forms are batch-invariant
  by construction (L2).
- **Mode invariance depends on the chunk boundaries.** The recurrent form is
  mode-invariant at no cost, because every mode runs the same loop body. The
  chunked form is mode-invariant only when every chunk starts at an absolute
  multiple of C.
- **The chunked and recurrent forms give different results, and the
  precision decides how different.** They are equally accurate when the
  chunked matrix products run in fp32. With bf16 operands, which common
  kernels use, the chunked form carries a much larger error, and that error is
  most of the gap seen in practice.

## The steps

### 1. Confirm that the layer is batch-invariant

- **Goal:** rule out the cheap failure first.
- **How:** run `bi_check.check_batch` from
  [tools/bi_check.py](../../tools/bi_check.py) on a sequence whose last chunk
  is only partly filled and whose initial state is not zero.
- **Success:** both forms pass. This was shown on one GPU type
  ([report](../../examples/linear_attention/REPORT.md#results)).

### 2. Choose one form that the trainer and the engine both use

- **Goal:** the forward pass uses the same arithmetic on both sides.
- **How:** use the recurrent and chunked kernels in
  [gdn_kernels.py](../../examples/linear_attention/gdn_kernels.py), and the
  aligned chunk wrapper `AlignedChunked` in
  [gdn_bi.py](../../examples/linear_attention/gdn_bi.py); see
  [recipe 7](../../skills/batch-invariant-kernel/references/operator-recipes.md#7-linear-attention-and-ssms-gdn-mamba-style).
- **Success:** decode and every chunk size give an output and a state that
  are bitwise equal to those of a full prefill. This was shown on one GPU type
  ([report](../../examples/linear_attention/REPORT.md#recommendation-with-the-measured-cost-of-each-option)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| (a) Run the recurrent forward everywhere | L1–L3, with the same kernel on both sides | trainer forward: moderate for short and medium sequences, large for long ones; decode is the cheapest | the default when you need parity |
| (b) Run the chunked form everywhere, and decode in aligned chunk steps | L1–L3, with the trainer unchanged | trainer: free; decode: large | long sequences, where (a) costs the most |
| (c) Accept the difference | not bitwise equal | free | parity is not required; then prefer fp32 for the chunked products |

Which of (a) and (b) is cheaper overall depends on how a workload divides its
time between the trainer's forward pass and rollout decode.

### 3. Use the same kernel configuration on both sides

- **Goal:** the kernel uses the same internal reduction tree in the trainer
  and in the engine.
- **How:** for the recurrent kernel, use the same tile width and warp count
  on both sides; for the chunked kernel, use the same chunk size and the same
  precision for the products.
- **Success:** the kernels on the two sides give the same bits on the same
  input. Different recurrent configurations each pass L1–L3 but differ from
  each other ([report](../../examples/linear_attention/REPORT.md#what-this-shows)).

### 4. Verify each mode as it continues from a saved state

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
with each mode continuing from the saved fp32 state. Compare the fp32 outputs
before the cast, and the final state, not only the bf16 output.

## What can go wrong

!!! warning "A chunk of one token is not a recurrent step"
    If you decode with the chunked kernel one new token at a time, every call
    starts a new chunk, and the result is not bitwise equal to prefill. Keep
    the state at the last absolute chunk boundary and recompute the
    unfinished chunk instead.

!!! warning "Two configurations are two reduction trees"
    The tile width and warp count of the recurrent kernel change how the sum
    inside the state update is grouped. To reach L4, pin them to the same
    values on both sides.

!!! warning "A bf16 output hides differences in the state"
    Most differences disappear in the final cast to bf16 and reappear in later
    tokens. Compare the fp32 outputs and the state.

**Not yet shown:** a backward pass for the recurrent forward; decode batches
for the aligned wrapper in which the sequences sit at different positions;
and agreement with a production kernel library.

## Summary

- Both forms are batch-invariant by construction; whether they are
  mode-invariant depends on where the chunk boundaries fall.
- The recurrent form is mode-invariant at no cost; the chunked form is
  mode-invariant when its boundaries sit at absolute multiples of C.
- Most of the practical gap between the chunked and the recurrent form comes
  from the precision of the chunked matrix products.
- Choose (a) recurrent or (b) aligned chunked according to where the cost
  falls, and pin the kernel configuration on both sides.

## Go deeper

- Reports: [linear attention (GDN)](../../examples/linear_attention/REPORT.md)
  and [kernel cases](../../examples/kernel_cases/REPORT.md#plain-linear-attention-chunked-vs-recurrent-an-algorithmic-difference)
  (plain linear attention).
- Code: [examples/linear_attention/gdn_bi.py](../../examples/linear_attention/gdn_bi.py),
  [gdn_kernels.py](../../examples/linear_attention/gdn_kernels.py).
- Recipe: [7, linear attention](../../skills/batch-invariant-kernel/references/operator-recipes.md#7-linear-attention-and-ssms-gdn-mamba-style);
  the cost that others have reported for training with the recurrent form
  only is listed in the [references](../references.md#rl-traininference-alignment).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
