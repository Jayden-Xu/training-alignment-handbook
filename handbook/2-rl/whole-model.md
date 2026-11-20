# Make one side invariant: the whole model (L1–L3)

## Where we are

This is the first step of the [RL workflow](README.md). You have a model that
the trainer and the engine both run, and logprobs that disagree even with
identical weights. Here you audit its operators and make one side give a
sequence the same bits however it is batched and served: run to run (L1), in
any batch (L2) and in any mode (L3). Making the two sides agree comes later, in
[engine parity](engine-parity-vllm.md).

## The idea

The trainer and the engine compute the same tokens in different shapes. The
trainer runs the whole sequence at once (prefill) inside a batch of training
sequences. The engine runs one token per step (decode) from a KV cache, inside
a batch whose members join and leave at every step; it may also chunk the
prompt, reuse a cached prefix or verify speculative tokens.

<figure class="fx" data-fig="prefill-decode" aria-label="One sequence computed by prefill in the trainer and by decode in the engine">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label"><b>trainer</b> prefill: all tokens at once, in a batch of B</span><div class="fx-col is-wide"><div class="fx-batch"><span class="fx-rowbar"></span><span class="fx-rowbar"></span><div class="fx-cells" style="--n:6"><span class="fx-cell" data-hot="1">p</span><span class="fx-cell" data-hot="1">p</span><span class="fx-cell is-mine" data-hot="1">t1</span><span class="fx-cell is-mine" data-hot="1">t2</span><span class="fx-cell is-mine" data-hot="1">t3</span><span class="fx-cell is-mine" data-hot="1">t4</span></div><span class="fx-rowbar"></span></div><div class="fx-cells fx-lp" style="--n:6"><span></span><span></span><span class="fx-chip" data-at="2">ℓ1</span><span class="fx-chip" data-at="2">ℓ2</span><span class="fx-chip" data-at="2">ℓ3</span><span class="fx-chip" data-at="2">ℓ4</span></div></div></div>
<div class="fx-row"><span class="fx-label"><b>engine</b> decode: one token per step, the batch changes every step</span><div class="fx-col is-wide"><div class="fx-batch"><span class="fx-rowbar" data-at="1" data-to="3"></span><span class="fx-rowbar" data-at="1" data-to="2"></span><div class="fx-cells" style="--n:6"><span class="fx-cell">p</span><span class="fx-cell">p</span><span class="fx-cell is-mine" data-hot="1">t1</span><span class="fx-cell is-mine" data-hot="2">t2</span><span class="fx-cell is-mine" data-hot="3">t3</span><span class="fx-cell is-mine" data-hot="4">t4</span></div><span class="fx-rowbar" data-at="2" data-to="3"></span><span class="fx-rowbar" data-at="4"></span><span class="fx-rowbar" data-at="4"></span></div><div class="fx-cells fx-lp" style="--n:6"><span></span><span></span><span class="fx-chip" data-at="1">ℓ1</span><span class="fx-chip" data-at="2">ℓ2</span><span class="fx-chip" data-at="3">ℓ3</span><span class="fx-chip" data-at="4">ℓ4</span></div></div></div>
</div>
<div class="fx-chain is-tight"><span class="fx-node is-decision" data-at="5">logprobs ℓ1–ℓ4: bitwise equal?</span></div>
</div>
<ol class="fx-say">
<li>One sequence: a prompt (p) and four sampled tokens, t1 to t4. Grey bars are other sequences in the batch.</li>
<li data-hold="1.5">Trainer: one prefill computes every token at once, in a batch of B. Engine: decode computes t1, with two other requests.</li>
<li data-hold="1.5">Trainer: a logprob for every token. Engine: t2; one request left, another joined.</li>
<li>Engine: t3, now alone in the batch.</li>
<li>Engine: t4, with two new requests.</li>
<li data-hold="2">Same tokens, other shapes and other batches. The step makes the logprobs bitwise equal.</li>
</ol>
<figcaption>Figure: the trainer and the engine compute the same tokens in different shapes: one prefill in a fixed batch against one decode step at a time in a changing batch.</figcaption>
</figure>

A floating-point sum depends on its order. Add 1e8, -1e8 and 1 in fp32: left
to right gives 1, but adding 1 to -1e8 first rounds it away and gives 0. A
kernel that picks its reduction order from the number of rows therefore gives
one row different bits in a batch of 1 and a batch of 64. Three things change
the order between the two sides:

| Source | What happens | Level it breaks |
|---|---|---|
| Batch-variant kernels | A GEMM library or a reduction picks another kernel or split when the number of rows changes | L2, batch |
| Mode-dependent kernels | Prefill, chunked prefill and decode take different attention code paths | L3, mode |
| Rounding points | The two sides cast to bf16 at different places, for example before or after the norm weight multiplies | L4, cross-engine |

The rule: every operator on the path from token to logprob must reduce in an
order fixed by the row's own data, never by the batch or the mode. This is the
batch-invariance principle of [Part 3](../3-kernels/README.md) applied to a
whole model. The last row of the table is why invariant kernels alone are not
enough: an engine that is L2 and L3 but rounds once differently is as far from
the trainer as a stock engine. That row is settled in a later step.

## The step

1. **Audit the operators.**
   - Goal: the list of modules whose output depends on the batch.
   - How: [tools/bi_audit.py](../../tools/bi_audit.py) replays every module on
     one row alone and in a batch, and names the first module whose output
     differs, including fp32 reductions hidden by a bf16 output.
   - Success: every module is reported invariant, or you have the list to
     replace.
2. **Replace the variant operators.**
   - Goal: an op set with a fixed reduction order.
   - How: the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)
     and its [operator recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md).
   - Success: the audit passes.
3. **Check every serving mode the engine uses** against the trainer forward.
   - Goal: L1–L3 on the whole model, not just per operator.
   - How: decode in a changing batch, chunked prefill, prefix hit,
     speculative verification, preemption, padding, varlen;
     [whole_model.py](../../examples/whole_model/whole_model.py) is this check
     for the reference model.
   - Success: bitwise-equal logits in every mode, and equal activation
     gradients.
4. **Write down the rounding sequence.**
   - Goal: the record the engine will be aligned to later.
   - How: every cast, which ops run in fp32, the RoPE formula, where the norm
     weight multiplies; checklist in
     [rounding points](../../skills/engine-parity/references/rounding-points.md),
     worked example in [model.py](../../examples/whole_model/model.py).
   - Success: one line per rounding point, in order.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Keep stock ops | Nothing bitwise; most tokens differ between batches and modes | free | Only to measure the gap |
| Batch-invariant op set on both sides | L1–L3 per side, forward and activation gradients | moderate (under 2x), up to large at big prefills | You own the model code of both sides |
| Batch-invariant op set plus one rounding sequence | The precondition for L4 | as above, plus aligning the engine's casts | Trainer and engine are different code ([engine parity](engine-parity-vllm.md)) |

Shown: a model built from batch-invariant kernels gave bitwise-equal logits
and activation gradients in every serving mode tested, for four architectures,
on one GPU type in eager mode; with stock ops nearly every logprob differed
([report: invariance](../../examples/whole_model/REPORT.md#invariance),
[what RL would see](../../examples/whole_model/REPORT.md#what-rl-would-see-with-stock-ops),
[cost](../../examples/whole_model/REPORT.md#cost)). Not yet shown:
`torch.compile`, CUDA graphs, tensor parallelism (next step), a real engine.

## What can go wrong

!!! warning "A bf16 output can hide a batch-variant fp32 reduction"
    An fp32 reduction inside a module (the RMSNorm variance) can change with
    the row count while the module's bf16 output does not. A check on outputs
    passes at one shape and fails at another. Replace it anyway; the audit
    reports it as "variant inside" ([report](../../examples/whole_model/REPORT.md#where-the-variance-comes-from)).

!!! warning "A RoPE table built under a compiler is not the eager table"
    A table built inside a compiled region uses the compiler's math and differs
    from the eager one in a few entries, and it is kept for later eager runs.
    Build it in eager before compiling.

!!! warning "Parameter gradients can only be L1"
    A parameter gradient sums over the whole batch, so it cannot be compared
    row by row. Test it run to run, and test activation gradients for L2.

!!! warning "Two Hugging Face references are not the same model"
    A model cast to bf16 after loading rounds its RoPE frequencies to bf16,
    unlike one loaded in bf16, and the two differ visibly in the logits. Say
    which one an engine is compared against.

!!! warning "Test the sequence lengths you run"
    An attention kernel can be invariant at short rows and change with the
    batch at long ones. A pass at short lengths says nothing about long ones.

## Summary

- The trainer and the engine run the same tokens in different batches and
  modes; stock kernels change their reduction order with both.
- Audit the model, replace every variant operator, then check every serving
  mode against the trainer forward.
- Batch invariance per side is necessary, not sufficient: the two sides must
  also round at the same places.
- Write the rounding sequence down now; the engine is aligned to it later.

## Go deeper

- [Lab report](../../examples/whole_model/REPORT.md): model variants, op sets,
  execution modes, every result table, the audit tool, cost and limits.
- Code: [model.py](../../examples/whole_model/model.py),
  [whole_model.py](../../examples/whole_model/whole_model.py),
  [tools/bi_audit.py](../../tools/bi_audit.py).
- Skills: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md),
  [engine-parity](../../skills/engine-parity/SKILL.md).
- Next step: [parallelism](parallelism.md).
