# Training Alignment Handbook

How to make two training or inference runs agree bit for bit, and find the
first place they stop agreeing.

## Three workflows

The handbook teaches three workflows. Two start from a symptom: two
pre-training runs that should agree do not (part 1), or the rollout engine and
the trainer give different log-probabilities for the same tokens (part 2).
Both end the same way when the cause is an operator whose result depends on
the batch around it: they hand that operator to the third workflow, which
makes it invariant (part 3). [Part 0](handbook/0-foundations/README.md) teaches the
ideas all three rest on.

- **Part 1, [pre-training](handbook/1-pretraining/README.md): find where two runs part.**
  Determinism and preflight → Training step → Data parallel → Capture compiled
  code → *Localize with OpGuard* → SDC.
- **Part 2, [RL train–inference](handbook/2-rl/README.md): make rollout and trainer agree.**
  *Whole model* → Parallelism → Sampling → LoRA → *Engine parity* → RL loop.
- **Part 3, [batch-invariant kernels](handbook/3-kernels/README.md): make one operator invariant.**
  GEMM → Attention → MLA and sparse → Linear attention → MoE → Collectives.

The steps in *italics* hand the operator they find to part 3. Part 0 comes
before all three; [part 4](handbook/4-operations/README.md) re-checks the
results after a stack change or on new hardware.

Not sure where to start? The [alignment-pipeline skill](skills/alignment-pipeline/SKILL.md)
asks the questions and routes you. The words used throughout are in the
[glossary](handbook/glossary.md).

## Part 1: two pre-training runs part

[Overview](handbook/1-pretraining/README.md) · agent skill:
[pretraining-alignment](skills/pretraining-alignment/SKILL.md)

1. [Determinism and preflight](handbook/1-pretraining/determinism-and-preflight.md):
   turn on the settings that make a run repeatable, and check both runs really use them.
2. [Training step](handbook/1-pretraining/training-step.md): make backward, the optimizer,
   the loss and resume from a checkpoint repeat bit for bit.
3. [Data parallel](handbook/1-pretraining/data-parallel.md): keep gradients equal when
   they are summed across ranks or sharded.
4. [Capture compiled code](handbook/1-pretraining/capture-compiled-code.md): take
   fingerprints inside `torch.compile` and CUDA graphs, where hooks cannot see.
5. [Localize with OpGuard](handbook/1-pretraining/localize-with-opguard.md): find the
   first operator where the two runs differ; that is what to fix.
6. [SDC](handbook/1-pretraining/sdc.md): rule out hardware that silently returns wrong values.

## Part 2: rollout and trainer disagree

[Overview](handbook/2-rl/README.md) · agent skill:
[engine-parity](skills/engine-parity/SKILL.md); OpGuard's `rl-alignment`

1. [Whole model](handbook/2-rl/whole-model.md): make one side give a sequence the same
   logits alone, in a batch, and in every serving mode.
2. [Parallelism](handbook/2-rl/parallelism.md): keep that true under tensor parallelism
   and disaggregated prefill and decode.
3. [Sampling](handbook/2-rl/sampling.md): make the chosen tokens replayable from the
   logits and a seed.
4. [LoRA](handbook/2-rl/lora.md): keep adapters invariant when one batch mixes several.
5. [Engine parity](handbook/2-rl/engine-parity-vllm.md): make a real engine follow the
   trainer's rounding steps, so both sides agree.
6. [RL loop](handbook/2-rl/rl-loop.md): record each rollout, separate staleness from
   mismatch, and localize a gap offline.

## Part 3: make one operator invariant

[The standard](handbook/3-kernels/README.md) · agent skill:
[batch-invariant-kernel](skills/batch-invariant-kernel/SKILL.md)

1. [GEMM and quantized GEMM](handbook/3-kernels/gemm-and-quantized.md): fix one tile
   and reduction order for every batch size.
2. [Attention](handbook/3-kernels/attention.md): one split of the keys for prefill,
   decode and backward.
3. [MLA and sparse attention](handbook/3-kernels/mla-and-sparse.md): the same for
   compressed keys and top-k selection.
4. [Linear attention](handbook/3-kernels/linear-attention.md): one chunk order for the
   recurrent state.
5. [MoE](handbook/3-kernels/moe.md): routing, expert GEMMs and the combine, independent
   of which tokens share a batch.
6. [Collectives](handbook/3-kernels/collectives.md): a pinned reduction order across GPUs.

## Skills, tools and operations

- **Skills** ([skills/](skills/)): [alignment-pipeline](skills/alignment-pipeline/SKILL.md),
  [pretraining-alignment](skills/pretraining-alignment/SKILL.md),
  [engine-parity](skills/engine-parity/SKILL.md),
  [batch-invariant-kernel](skills/batch-invariant-kernel/SKILL.md),
  [handbook-writing](skills/handbook-writing/SKILL.md), and
  `rl-alignment` in OpGuard's `harness/skills/`.
- **Tools** ([tools/](tools/)):
  - [bi_check.py](tools/bi_check.py): bitwise invariance checks (`--demo` on stock ops; [a recorded run](tools/demo-a40.json));
  - [bi_kernels.py](tools/bi_kernels.py): reference invariant Triton kernels (GEMM, attention);
  - [bi_sampling.py](tools/bi_sampling.py): a replayable reference sampler;
  - [bi_audit.py](tools/bi_audit.py): operator inventory and per-module L2 verdicts;
  - [bi_report.py](tools/bi_report.py) and [bi_suite.py](tools/bi_suite.py): `bi-results-v1` files and the regression suite ([suite/](suite/));
  - [determinism_preflight.py](tools/determinism_preflight.py): read back and diff determinism settings;
  - [opguard_bridge.py](tools/opguard_bridge.py): OpGuard's formats, protocol and harness, and `suggest()`;
  - [graph_probe.py](tools/graph_probe.py): fingerprints inside compiled code and CUDA graphs;
  - [data_fingerprint.py](tools/data_fingerprint.py): per-batch fingerprints at the data loader's exit, and a compare that classifies the first difference;
  - [ckpt_compare.py](tools/ckpt_compare.py): compare two checkpoints of different layouts row by row;
  - [sdc_scan.py](tools/sdc_scan.py): silent data corruption scan.
- **Other scenarios** ([part 5](handbook/5-scenarios/README.md)): framework migration, checkpoint conversion, offline post-training, serving and mixed hardware.
- **Operations** ([part 4](handbook/4-operations/README.md)):
  [regression](handbook/4-operations/regression.md) after an upgrade or on new
  hardware, and [hardware coverage](handbook/4-operations/hardware.md).
- **Evidence**: each example under [examples/](examples/) has a `REPORT.md` with
  its setup and measurements; the suite's baselines are in [suite/](suite/).
- [Open items](handbook/open-items.md) · [References](handbook/references.md) ·
  [Glossary](handbook/glossary.md) · [Audit](AUDIT.md)

The handbook is also a bilingual website built from `handbook/`
([how to build it](handbook/4-operations/README.md#building-the-site)).
