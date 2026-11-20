---
hide:
  - toc
---

# Training Alignment Handbook

How to make two training or inference runs agree bit for bit, and find the
first place they stop agreeing.

## Three workflows

The handbook teaches three workflows. Two start from a symptom: two
pre-training runs that should agree do not (part 1), or the rollout engine and
the trainer give different log-probabilities for the same tokens (part 2).
Both end the same way when the cause is an operator whose result depends on
the batch around it: they hand that operator to the third workflow, which
makes it invariant (part 3). [Part 0](0-foundations/README.md) teaches the
ideas all three rest on.

<figure class="flows" aria-label="The three workflows" markdown>
<p class="flows-base" markdown>Built on [Part 0: Foundations](0-foundations/README.md): levels L1–L5, and where and how to compare.</p>
<div class="flow-lane" data-lane="pre" data-side="above" markdown>
<p class="flow-head" markdown><span class="flow-part">Part 1</span> [Pre-training](1-pretraining/README.md) <span class="flow-goal">Find where two runs part</span></p>
<ol class="flow-steps" style="--steps: 7" markdown>
<li markdown="span">[Preflight](1-pretraining/determinism-and-preflight.md)<span class="flow-desc">Turn on the settings that make a run repeatable, and check both runs really use them.</span></li>
<li markdown="span">[Data alignment](1-pretraining/data-alignment.md)<span class="flow-desc">Check that every rank of both runs gets the same batches at every step.</span></li>
<li class="flow-handoff" markdown="span">[Localize with OpGuard](1-pretraining/localize-with-opguard.md)<span class="flow-desc">Capture both runs, find the first operator where they differ, walk from it to the cause, fix and confirm.</span></li>
<li markdown="span">[Training step](1-pretraining/training-step.md)<span class="flow-desc">When the cause is in backward, the optimizer, the loss or resume: make them repeat bit for bit.</span></li>
<li markdown="span">[Data parallel](1-pretraining/data-parallel.md)<span class="flow-desc">When the cause is a gradient summed across ranks or sharded: keep it equal.</span></li>
<li markdown="span">[Compiled code](1-pretraining/capture-compiled-code.md)<span class="flow-desc">When the job runs `torch.compile` or CUDA graphs: take fingerprints inside them, where an eager capture cannot see.</span></li>
<li markdown="span">[SDC](1-pretraining/sdc.md)<span class="flow-desc">When the software is ruled out: find hardware that silently returns wrong values.</span></li>
</ol>
</div>
<div class="flow-lane" data-lane="kernel" data-delay="1" data-target markdown>
<p class="flow-head" markdown><span class="flow-part">Part 3</span> [Batch-invariant kernels](3-kernels/README.md) <span class="flow-goal">Make one operator invariant</span></p>
<ol class="flow-steps" markdown>
<li markdown="span">[GEMM and quantized GEMM](3-kernels/gemm-and-quantized.md)<span class="flow-desc">Fix one tile and reduction order for every batch size.</span></li>
<li markdown="span">[Attention](3-kernels/attention.md)<span class="flow-desc">One split of the keys for prefill, decode and backward.</span></li>
<li markdown="span">[MLA and sparse attention](3-kernels/mla-and-sparse.md)<span class="flow-desc">The same for compressed keys and top-k selection.</span></li>
<li markdown="span">[Linear attention](3-kernels/linear-attention.md)<span class="flow-desc">One chunk order for the recurrent state.</span></li>
<li markdown="span">[MoE](3-kernels/moe.md)<span class="flow-desc">Routing, expert GEMMs and the combine, independent of which tokens share a batch.</span></li>
<li markdown="span">[Collectives](3-kernels/collectives.md)<span class="flow-desc">A pinned reduction order across GPUs.</span></li>
</ol>
</div>
<div class="flow-lane" data-lane="rl" data-side="below" markdown>
<p class="flow-head" markdown><span class="flow-part">Part 2</span> [RL train–inference](2-rl/README.md) <span class="flow-goal">Make rollout and trainer agree</span></p>
<ol class="flow-steps" markdown>
<li class="flow-handoff" markdown="span">[Whole model](2-rl/whole-model.md)<span class="flow-desc">Make one side give a sequence the same logits alone, in a batch, and in every serving mode.</span></li>
<li markdown="span">[Parallelism](2-rl/parallelism.md)<span class="flow-desc">Keep that true under tensor parallelism and disaggregated prefill and decode.</span></li>
<li markdown="span">[Sampling](2-rl/sampling.md)<span class="flow-desc">Make the chosen tokens replayable from the logits and a seed.</span></li>
<li markdown="span">[LoRA](2-rl/lora.md)<span class="flow-desc">Keep adapters invariant when one batch mixes several.</span></li>
<li class="flow-handoff" markdown="span">[Engine parity](2-rl/engine-parity-vllm.md)<span class="flow-desc">Make a real engine follow the trainer's rounding steps, so both sides agree.</span></li>
<li markdown="span">[RL loop](2-rl/rl-loop.md)<span class="flow-desc">Record each rollout, separate staleness from mismatch, and localize a gap offline.</span></li>
</ol>
</div>
<p class="flows-next" markdown>Afterwards, [Part 4: Operations](4-operations/README.md) re-checks the results when the stack or the hardware changes.</p>
<figcaption><span class="flow-key"></span>A dashed step hands the operator it found to part 3. Hover over or focus a step to read what it does.</figcaption>
</figure>

Not sure where to start? The [alignment-pipeline skill](../skills/alignment-pipeline/SKILL.md)
asks the questions and routes you. The words used throughout are in the
[glossary](glossary.md).

## Part 1: two pre-training runs part

[Overview](1-pretraining/README.md) · agent skill:
[pretraining-alignment](../skills/pretraining-alignment/SKILL.md)

1. [Determinism and preflight](1-pretraining/determinism-and-preflight.md):
   turn on the settings that make a run repeatable, and check both runs really use them.
2. [Data alignment](1-pretraining/data-alignment.md): check that every rank of
   both runs gets the same batches at every step.
3. [Localize with OpGuard](1-pretraining/localize-with-opguard.md): capture both
   runs, find the first operator where they differ, walk from it to the cause,
   fix and confirm.

Then the page for where the first divergence points:

- [Training step](1-pretraining/training-step.md): backward, the optimizer, the
  loss and resume from a checkpoint, made to repeat bit for bit.
- [Data parallel](1-pretraining/data-parallel.md): gradients summed across
  ranks or sharded, kept equal.
- [When the job runs compiled code](1-pretraining/capture-compiled-code.md):
  fingerprints inside `torch.compile` and CUDA graphs, where an eager capture
  cannot see.
- [SDC](1-pretraining/sdc.md): hardware that silently returns wrong values.

## Part 2: rollout and trainer disagree

[Overview](2-rl/README.md) · agent skill:
[engine-parity](../skills/engine-parity/SKILL.md); OpGuard's `rl-alignment`

1. [Whole model](2-rl/whole-model.md): make one side give a sequence the same
   logits alone, in a batch, and in every serving mode.
2. [Parallelism](2-rl/parallelism.md): keep that true under tensor parallelism
   and disaggregated prefill and decode.
3. [Sampling](2-rl/sampling.md): make the chosen tokens replayable from the
   logits and a seed.
4. [LoRA](2-rl/lora.md): keep adapters invariant when one batch mixes several.
5. [Engine parity](2-rl/engine-parity-vllm.md): make a real engine follow the
   trainer's rounding steps, so both sides agree.
6. [RL loop](2-rl/rl-loop.md): record each rollout, separate staleness from
   mismatch, and localize a gap offline.

## Part 3: make one operator invariant

[The standard](3-kernels/README.md) · agent skill:
[batch-invariant-kernel](../skills/batch-invariant-kernel/SKILL.md)

1. [GEMM and quantized GEMM](3-kernels/gemm-and-quantized.md): fix one tile
   and reduction order for every batch size.
2. [Attention](3-kernels/attention.md): one split of the keys for prefill,
   decode and backward.
3. [MLA and sparse attention](3-kernels/mla-and-sparse.md): the same for
   compressed keys and top-k selection.
4. [Linear attention](3-kernels/linear-attention.md): one chunk order for the
   recurrent state.
5. [MoE](3-kernels/moe.md): routing, expert GEMMs and the combine, independent
   of which tokens share a batch.
6. [Collectives](3-kernels/collectives.md): a pinned reduction order across GPUs.

## Part 5: other scenarios

[Overview](5-scenarios/README.md) · [Framework migration](5-scenarios/framework-migration.md) · [Checkpoint conversion](5-scenarios/checkpoint-conversion.md) · [Offline post-training](5-scenarios/offline-post-training.md): the same method for other pairs.

## Part 4 and reference

- [Part 4: Operations](4-operations/README.md): re-check the results after a
  [stack change](4-operations/regression.md) and on
  [new hardware](4-operations/hardware.md).
- Skills: [alignment-pipeline](../skills/alignment-pipeline/SKILL.md),
  [pretraining-alignment](../skills/pretraining-alignment/SKILL.md),
  [engine-parity](../skills/engine-parity/SKILL.md),
  [batch-invariant-kernel](../skills/batch-invariant-kernel/SKILL.md),
  [handbook-writing](../skills/handbook-writing/SKILL.md).
- [Open items](open-items.md) · [References](references.md) ·
  [Glossary](glossary.md) · [Audit](../AUDIT.md) · [Tools](../tools/)

Every claim on a page links to the report with its measurements and the stack
it was measured on.
