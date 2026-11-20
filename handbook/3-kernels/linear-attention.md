# Linear attention (gated delta rule)

## Where we are

[MLA](mla-and-sparse.md) showed that two equal forms of the same math are a
choice that every mode must make the same way. Linear attention has the same
choice in a sharper form. Training and prefill usually run a chunked
algorithm; decode runs a recurrent step. Here the result also carries over
from token to token in a state, so what matters is where the chunk boundaries
fall.

## The idea: the state is summed in groups set by the chunks

A linear attention layer (here Gated DeltaNet) keeps a fixed-size state S per
sequence and head and updates it with every token. The recurrent form applies
one update per token. The chunked form groups C tokens, combines them with
matrix products, and carries the state from chunk to chunk.

Take ten tokens and chunks of 4:

<figure class="fx" data-fig="chunks" aria-label="Where the chunk boundaries fall for three call patterns and the recurrent form">
<div class="fx-stage">
<div class="fx-rows" style="--n:10">
<div class="fx-row"><span class="fx-label">ten tokens, C = 4</span><div class="fx-track"><div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span></div><span class="fx-verdict"></span></div></div>
<div class="fx-row"><span class="fx-label">one prefill of tokens 0–9</span><div class="fx-track"><div class="fx-level fx-sweep" data-at="1" data-hot="1"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span" style="--w:2">{8–9}</span></div><span class="fx-verdict"></span></div></div>
<div class="fx-row"><span class="fx-label">prefill of 0–7, then one decode call per token</span><div class="fx-track"><div class="fx-level" data-at="2" data-hot="3"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span to-bad" style="--w:1">{8}</span><span class="fx-span to-bad" style="--w:1">{9}</span></div><span class="fx-verdict"><span class="fx-node is-bad" data-at="3">bits follow the call sizes</span></span></div></div>
<div class="fx-row"><span class="fx-label">aligned: keep the state at token 8, recompute the open chunk</span><div class="fx-track"><div class="fx-swap is-block"><div class="fx-level" data-at="4" data-to="5"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span" style="--w:1">{8}</span></div><div class="fx-level" data-at="5" data-hot="5"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span to-ok" style="--w:2">{8–9}</span></div></div><span class="fx-verdict"><span class="fx-node is-ok" data-at="5">same as prefill</span></span></div></div>
<div class="fx-row"><span class="fx-label">recurrent: one token per step</span><div class="fx-track"><div class="fx-level fx-sweep is-unit" data-at="6" data-hot="6"><span class="fx-span" style="--w:1">0</span><span class="fx-span" style="--w:1">1</span><span class="fx-span" style="--w:1">2</span><span class="fx-span" style="--w:1">3</span><span class="fx-span" style="--w:1">4</span><span class="fx-span" style="--w:1">5</span><span class="fx-span" style="--w:1">6</span><span class="fx-span" style="--w:1">7</span><span class="fx-span" style="--w:1">8</span><span class="fx-span" style="--w:1">9</span></div><span class="fx-verdict"><span class="fx-node is-ok" data-at="6">same bits in every mode</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>Ten tokens, chunks of C = 4. The chunks set how the state's sum is grouped.</li>
<li>One prefill of tokens 0–9: chunks {0–3}, {4–7}, {8–9}.</li>
<li data-hold="1.5">Prefill 0–7, then one decode call per token: each call starts its own chunk, {8} and then {9}.</li>
<li data-hold="1.5">Token 9 is summed in another grouping: its output and the state differ in their last bits.</li>
<li data-hold="1.5">Aligned: keep the state at token 8, a chunk boundary, and recompute the open chunk when token 9 arrives:</li>
<li data-hold="1.5">{8–9} again, as in prefill, so the same bits.</li>
<li data-hold="2">Recurrent: one update per token in every mode, so the same bits for free.</li>
</ol>
<figcaption>Figure: chunked forms are mode-invariant only when every chunk starts at an absolute multiple of C; the recurrent form always is.</figcaption>
</figure>

The second pattern sums the same terms in a different grouping, so token 9's
output and the state differ in their last bits.

Three facts follow:

- **Batch is not the problem.** Each program owns one sequence and head; no
  reduction crosses sequences, so both forms are batch-invariant by
  construction (L2).
- **Mode invariance is about the boundaries.** The recurrent form is
  mode-invariant for free: every mode runs the same loop body. The chunked
  form is mode-invariant only when every chunk starts at an absolute multiple
  of C.
- **Chunked and recurrent differ, and precision decides by how much.** They are
  equally accurate when the chunked matrix products run in fp32; with bf16
  operands, as common kernels use, the chunked form carries much larger error
  and most of the practical gap.

## The steps

### 1. Confirm batch invariance

- **Goal:** rule out the cheap failure first.
- **How:** `bi_check.check_batch` from [tools/bi_check.py](../../tools/bi_check.py)
  on a sequence with a partial last chunk and a nonzero initial state.
- **Success:** a pass for both forms. Shown on one GPU type
  ([report](../../examples/linear_attention/REPORT.md#results)).

### 2. Choose how trainer and engine agree

- **Goal:** one arithmetic for the forward on both sides.
- **How:** the recurrent and chunked kernels in
  [gdn_kernels.py](../../examples/linear_attention/gdn_kernels.py), and the
  aligned chunk wrapper `AlignedChunked` in
  [gdn_bi.py](../../examples/linear_attention/gdn_bi.py);
  [recipe 7](../../skills/batch-invariant-kernel/references/operator-recipes.md#7-linear-attention-and-ssms-gdn-mamba-style).
- **Success:** decode and every chunk size bitwise equal to a full prefill,
  output and state. Shown on one GPU type
  ([report](../../examples/linear_attention/REPORT.md#recommendation-with-the-measured-cost-of-each-option)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| (a) Recurrent forward everywhere | L1–L3; same kernel on both sides | trainer forward moderate at short and medium sequences, large at long ones; decode cheapest | the parity default |
| (b) Chunked everywhere, decode as aligned chunk steps | L1–L3; trainer unchanged | trainer free; decode large | long sequences, where (a) costs most |
| (c) Accept the difference | not bitwise | free | parity not required; then prefer fp32 chunked products |

Which of (a) and (b) is cheaper overall depends on how a workload splits time
between trainer forward and rollout decode.

### 3. Fix the configuration on both sides

- **Goal:** the same reduction tree inside the kernel on trainer and engine.
- **How:** use the same tile width and warp count for the recurrent kernel, or
  the same chunk size and product precision for the chunked one.
- **Success:** the two sides' kernels give the same bits on the same input.
  Different recurrent configurations each pass L1–L3 and differ from each
  other ([report](../../examples/linear_attention/REPORT.md#what-this-shows)).

### 4. Verify modes from a saved state

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
with each mode continuing from the saved fp32 state, and compare the outputs
in fp32 before the cast and the final state, not only the bf16 output.

## What can go wrong

!!! warning "A one-token chunk is not a recurrent step"
    Decoding with the chunked kernel on each new token starts a new chunk at
    every call. It is not bitwise equal to prefill; keep the state at the last
    absolute boundary and recompute the open chunk.

!!! warning "Two configurations are two reduction trees"
    The recurrent kernel's tile width and warp count change the sum inside the
    state update. Pin them on both sides for L4.

!!! warning "A bf16 output hides state differences"
    Most differences vanish in the final cast and come back in later tokens.
    Compare fp32 outputs and the state.

**Not yet shown:** a backward for the recurrent forward, decode batches whose
sequences sit at different positions for the aligned wrapper, and agreement
with a production kernel library.

## Summary

- Both forms are batch-invariant by construction; mode invariance is about
  where chunk boundaries fall.
- Recurrent is mode-invariant for free; chunked is mode-invariant with
  boundaries at absolute multiples of C.
- The practical chunked–recurrent gap is mostly the precision of the chunked
  matrix products.
- Pick (a) recurrent or (b) aligned chunked by where the cost lands, and pin
  the kernel configuration on both sides.

## Go deeper

- Reports: [linear attention (GDN)](../../examples/linear_attention/REPORT.md),
  [kernel cases](../../examples/kernel_cases/REPORT.md#plain-linear-attention-chunked-vs-recurrent-an-algorithmic-difference)
  (plain linear attention).
- Code: [examples/linear_attention/gdn_bi.py](../../examples/linear_attention/gdn_bi.py),
  [gdn_kernels.py](../../examples/linear_attention/gdn_kernels.py).
- Recipe: [7, linear attention](../../skills/batch-invariant-kernel/references/operator-recipes.md#7-linear-attention-and-ssms-gdn-mamba-style);
  the reported cost of recurrent-only training is in the
  [references](../references.md#rl-traininference-alignment).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
