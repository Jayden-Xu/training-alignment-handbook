# Whole-model invariance (L1–L3)

## Where we are

This is the first step of the [RL workflow](README.md). You have a model that
both the trainer and the engine run, and their logprobs disagree even with
identical weights. In this step you audit the model's operators and make one
side give a sequence the same bits however that sequence is batched and
served: from run to run (L1), in any batch (L2) and in any serving mode (L3).
Making the two sides agree with each other comes later, in
[engine parity](engine-parity-vllm.md).

## The idea

The trainer and the engine compute the same tokens in different shapes. The
trainer processes the whole sequence in one pass (prefill), inside a batch of
training sequences. The engine generates one token per step (decode) from a KV
cache, inside a batch whose members join and leave at every step. The engine
may also split the prompt into chunks, reuse a cached prefix, or verify
speculative tokens.

<figure class="fx" data-fig="prefill-decode" aria-label="One sequence computed by prefill in the trainer and by decode in the engine">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">trainer<span class="fx-note">prefill: every token in one pass</span></span><div class="fx-lane" style="--n:6"><div class="fx-cells"><span class="fx-cell" data-hot="1">p</span><span class="fx-cell" data-hot="1">p</span><span class="fx-cell is-mine" data-hot="1">t1</span><span class="fx-cell is-mine" data-hot="1">t2</span><span class="fx-cell is-mine" data-hot="1">t3</span><span class="fx-cell is-mine" data-hot="1">t4</span></div><div class="fx-cells"><span></span><span></span><span class="fx-span" style="--w:4" data-at="1">one pass in a batch of B</span></div><div class="fx-cells fx-lp"><span></span><span></span><span class="fx-chip" data-at="2">ℓ1</span><span class="fx-chip" data-at="2">ℓ2</span><span class="fx-chip" data-at="2">ℓ3</span><span class="fx-chip" data-at="2">ℓ4</span></div></div></div>
<div class="fx-row"><span class="fx-label">engine<span class="fx-note">decode: one token per step</span></span><div class="fx-lane" style="--n:6"><div class="fx-cells"><span class="fx-cell">p</span><span class="fx-cell">p</span><span class="fx-cell is-mine" data-hot="1">t1</span><span class="fx-cell is-mine" data-hot="2">t2</span><span class="fx-cell is-mine" data-hot="3">t3</span><span class="fx-cell is-mine" data-hot="4">t4</span></div><div class="fx-cells fx-who"><span></span><span></span><span class="fx-note" data-at="1">+ A, B</span><span class="fx-note" data-at="2">+ B, C</span><span class="fx-note" data-at="3">alone</span><span class="fx-note" data-at="4">+ D, E</span></div><div class="fx-cells fx-lp"><span></span><span></span><span class="fx-chip" data-at="1">ℓ1</span><span class="fx-chip" data-at="2">ℓ2</span><span class="fx-chip" data-at="3">ℓ3</span><span class="fx-chip" data-at="4">ℓ4</span></div></div></div>
</div>
<div class="fx-chain is-tight"><span class="fx-node is-decision" data-at="5">are logprobs ℓ1–ℓ4 bitwise equal?</span></div>
</div>
<ol class="fx-say">
<li>Take one sequence: a prompt (p) followed by four sampled tokens, t1 to t4.</li>
<li data-hold="1.5">The trainer computes every token in a single prefill, inside a batch of B sequences. The engine decodes t1 in a batch with requests A and B.</li>
<li data-hold="1.5">The trainer now has a logprob for every token. The engine decodes t2; A has left the batch and C has joined.</li>
<li>The engine decodes t3, now alone in the batch.</li>
<li>The engine decodes t4 together with two new requests, D and E.</li>
<li data-hold="2">The tokens are the same, but the shapes and the batches differ. This step makes the logprobs bitwise equal all the same.</li>
</ol>
<figcaption>Figure: the trainer and the engine compute the same tokens in different shapes. The trainer runs one prefill in a fixed batch; the engine runs one decode step at a time in a batch that keeps changing.</figcaption>
</figure>

A floating-point sum depends on the order of its additions. Add 1e8, -1e8 and
1 in fp32: going left to right gives 1, but if 1 is first added to -1e8, it is
rounded away and the result is 0. A kernel that chooses its reduction order
from the number of rows will therefore give the same row different bits in a
batch of 1 and in a batch of 64. Three things change the order between the
two sides:

| Source | What happens | Level it breaks |
|---|---|---|
| Kernels that vary with the batch | When the number of rows changes, a GEMM library or a reduction picks a different kernel or a different split | L2, batch |
| Kernels that depend on the mode | Prefill, chunked prefill and decode go through different attention code | L3, mode |
| Rounding points | The two sides cast to bf16 at different places, for example before or after multiplying by the norm weight | L4, cross-engine |

So the rule is that every operator on the path from a token to its logprob
must add things up in an order fixed by that row's own data, never by the
batch or the mode. This is the batch-invariance principle of
[Part 3](../3-kernels/README.md), applied to a whole model. The last row of
the table shows why invariant kernels alone are not enough: an engine that
reaches L2 and L3 but rounds differently at one place is as far from the
trainer as an unmodified engine. A later step settles that row.

## The step

1. **Audit the operators.**
   - Goal: a list of the modules whose output depends on the batch.
   - How: [tools/bi_audit.py](../../tools/bi_audit.py) replays every module
     on one row alone and on the same row inside a batch, and names the first
     module whose output differs. It also catches fp32 reductions whose
     variation is hidden by a bf16 output.
   - Success: every module is reported invariant, or you have the list of
     modules to replace.
2. **Replace the operators that vary.**
   - Goal: a set of operators that all reduce in a fixed order.
   - How: follow the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)
     and its [operator recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md).
   - Success: the audit passes.
3. **Check every serving mode the engine uses** against the trainer's forward.
   - Goal: L1–L3 for the whole model, not only for each operator.
   - How: test decode in a changing batch, chunked prefill, a prefix-cache
     hit, speculative verification, preemption, padding and variable-length
     batches; [whole_model.py](../../examples/whole_model/whole_model.py)
     runs this check for the reference model.
   - Success: the logits are bitwise equal in every mode, and so are the
     activation gradients.
4. **Write down the rounding sequence.**
   - Goal: a record that the engine will be aligned to in a later step.
   - How: note every cast, which operations run in fp32, the RoPE formula,
     and where the norm weight is multiplied in. The
     [rounding points](../../skills/engine-parity/references/rounding-points.md)
     checklist lists what to record, and
     [model.py](../../examples/whole_model/model.py) is a worked example.
   - Success: one line per rounding point, in the order they happen.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Keep the stock operators | No bitwise guarantee; most tokens differ between batches and between modes | free | Only to measure the gap |
| A batch-invariant operator set on both sides | L1–L3 on each side, for the forward and for activation gradients | moderate (under 2x), up to large for big prefills | You own the model code on both sides |
| A batch-invariant operator set plus one shared rounding sequence | The precondition for L4 | as above, plus the work of aligning the engine's casts | The trainer and the engine are different code ([engine parity](engine-parity-vllm.md)) |

What has been shown: a model built from batch-invariant kernels gave
bitwise-equal logits and activation gradients in every serving mode tested,
for four architectures, on one GPU type in eager mode. With stock operators,
nearly every logprob differed
([report: invariance](../../examples/whole_model/REPORT.md#invariance),
[what RL would see](../../examples/whole_model/REPORT.md#what-rl-would-see-with-stock-ops),
[cost](../../examples/whole_model/REPORT.md#cost)). Not yet shown:
`torch.compile`, CUDA graphs, tensor parallelism (the next step) and a real
engine.

## What can go wrong

!!! warning "A bf16 output can hide an fp32 reduction that varies with the batch"
    An fp32 reduction inside a module, such as the variance in RMSNorm, can
    change with the number of rows while the module's bf16 output stays the
    same. A check that only looks at outputs then passes at one shape and
    fails at another. Replace such a reduction anyway; the audit reports it
    as "variant inside" ([report](../../examples/whole_model/REPORT.md#where-the-variance-comes-from)).

!!! warning "A RoPE table built under the compiler differs from the eager one"
    A table built inside a compiled region uses the compiler's math, so a few
    of its entries differ from the eager table, and later eager runs keep
    using it. Build the table in eager mode before compiling.

!!! warning "Parameter gradients can only reach L1"
    A parameter gradient is a sum over the whole batch, so it cannot be
    compared row by row. Test it for repeatability from run to run, and test
    the activation gradients for L2.

!!! warning "Two Hugging Face references are not the same model"
    A model that is cast to bf16 after loading has its RoPE frequencies
    rounded to bf16, while a model loaded directly in bf16 does not, and the
    two give visibly different logits. State which of them an engine is
    compared against.

!!! warning "Test the sequence lengths you actually run"
    An attention kernel can be invariant on short rows and still change with
    the batch on long ones. A pass at short lengths tells you nothing about
    long ones.

## Summary

- The trainer and the engine run the same tokens in different batches and
  modes, and stock kernels change their reduction order with both.
- Audit the model, replace every operator that varies, and then check every
  serving mode against the trainer's forward.
- Batch invariance on each side is necessary but not sufficient: the two
  sides must also round at the same places.
- Write the rounding sequence down now, because the engine is aligned to it
  later.

## Go deeper

- [Lab report](../../examples/whole_model/REPORT.md): the model variants,
  operator sets and execution modes, every result table, the audit tool, the
  cost and the limits.
- Code: [model.py](../../examples/whole_model/model.py),
  [whole_model.py](../../examples/whole_model/whole_model.py),
  [tools/bi_audit.py](../../tools/bi_audit.py).
- Skills: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md),
  [engine-parity](../../skills/engine-parity/SKILL.md).
- Next step: [parallelism](parallelism.md).
