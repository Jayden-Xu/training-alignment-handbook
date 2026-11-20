# Training Alignment Handbook

How to make two training or inference runs agree bit for bit, and how to find
the first place where they stop agreeing.

## The three workflows

The handbook teaches three workflows. Two of them start from a symptom. In
the first, two pre-training runs that should agree do not (part 1). In the
second, the rollout engine and the trainer give different log-probabilities
for the same tokens (part 2). Both can end at an operator whose result depends
on the other rows in its batch. When that happens, they hand the operator to
the third workflow, which makes it batch-invariant (part 3).
[Part 0](handbook/0-foundations/README.md) teaches the ideas that all three
rest on.

- **Part 1, [pre-training](handbook/1-pretraining/README.md), finds where two runs stop agreeing.**
  Determinism and preflight → Data alignment → *Localize with OpGuard* →
  Training step → Data parallel → Compiled code → SDC.
- **Part 2, [RL train–inference](handbook/2-rl/README.md), makes the rollout engine and the trainer agree.**
  *Whole model* → Parallelism → Sampling → LoRA → *Engine parity* → RL loop.
- **Part 3, [batch-invariant kernels](handbook/3-kernels/README.md), makes one operator batch-invariant.**
  GEMM → Attention → MLA and sparse attention → Linear attention → MoE →
  Collectives.

The steps in *italics* hand the operator they find to part 3. Part 0 comes
before all three, and [part 4](handbook/4-operations/README.md) checks the
results again after a stack change or on new hardware.

Not sure where to start? The [alignment-pipeline skill](skills/alignment-pipeline/SKILL.md)
asks you a few questions and sends you to the right workflow. The terms used
throughout the handbook are defined in the [glossary](handbook/glossary.md).

## Part 1: when two pre-training runs stop agreeing

Start with the [overview](handbook/1-pretraining/README.md). The agent skill
for this part is [pretraining-alignment](skills/pretraining-alignment/SKILL.md).

1. [Determinism and preflight](handbook/1-pretraining/determinism-and-preflight.md):
   turn on the settings that make a run repeatable, and check that both runs
   really use them.
2. [Data alignment](handbook/1-pretraining/data-alignment.md): check that every
   rank of both runs gets the same batches at every step.
3. [Localize with OpGuard](handbook/1-pretraining/localize-with-opguard.md):
   capture both runs, find the first operator where they differ, and trace
   back from it to the cause. Then fix the cause and confirm the fix.

After that, read the page for the place the first divergence points to:

- [Training step](handbook/1-pretraining/training-step.md) shows how to make
  the backward pass, the optimizer, the loss and a resume from a checkpoint
  repeat bit for bit.
- [Data parallel](handbook/1-pretraining/data-parallel.md) shows how to keep
  gradients equal when ranks sum them together or shard them.
- [Compiled code and CUDA graphs](handbook/1-pretraining/capture-compiled-code.md)
  shows how to take fingerprints inside `torch.compile` and CUDA graphs, where
  an eager capture cannot see.
- [SDC](handbook/1-pretraining/sdc.md) shows how to find hardware that silently
  returns wrong values.

## Part 2: when the rollout engine and the trainer disagree

Start with the [overview](handbook/2-rl/README.md). The agent skills for this
part are [engine-parity](skills/engine-parity/SKILL.md) and OpGuard's
`rl-alignment`.

1. [Whole model](handbook/2-rl/whole-model.md): make one side give a sequence
   the same logits whether it runs alone, in a batch or in any serving mode.
2. [Parallelism](handbook/2-rl/parallelism.md): keep that true with tensor
   parallelism, and when prefill and decode run on separate servers.
3. [Sampling](handbook/2-rl/sampling.md): make sure the sampled tokens can be
   reproduced from the logits and a seed.
4. [LoRA](handbook/2-rl/lora.md): keep each adapter's result unchanged when one
   batch mixes several adapters.
5. [Engine parity](handbook/2-rl/engine-parity-vllm.md): make a real inference
   engine round at the same points as the trainer, so that the two sides agree.
6. [RL loop](handbook/2-rl/rl-loop.md): record each rollout, tell staleness
   apart from a real mismatch, and localize any gap offline.

## Part 3: making one operator batch-invariant

Start with [the standard](handbook/3-kernels/README.md) that every invariant
operator must meet. The agent skill for this part is
[batch-invariant-kernel](skills/batch-invariant-kernel/SKILL.md).

1. [GEMM and quantized GEMM](handbook/3-kernels/gemm-and-quantized.md): use the
   same tile and the same reduction order at every batch size.
2. [Attention](handbook/3-kernels/attention.md): split the keys the same way in
   prefill, in decode and in the backward pass.
3. [MLA and sparse attention](handbook/3-kernels/mla-and-sparse.md): do the
   same when the keys are compressed or picked by top-k selection.
4. [Linear attention](handbook/3-kernels/linear-attention.md): update the
   recurrent state in the same chunk order every time.
5. [MoE](handbook/3-kernels/moe.md): make routing, the expert GEMMs and the
   combine independent of which tokens share a batch.
6. [Collectives](handbook/3-kernels/collectives.md): add up the partial results
   of all GPUs in one fixed order.

## Using the skills with an agent

Each skill is a plain folder that holds a `SKILL.md` and a `references/`
folder. Two sets of skills work together:

| Set | Where | Skills |
|---|---|---|
| This handbook | [skills/](skills/) | alignment-pipeline, pretraining-alignment, engine-parity, batch-invariant-kernel, handbook-writing |
| OpGuard | `harness/skills/` in [OpGuard](https://github.com/ziming-zh/OpGuard) | investigate-first-diff, trace-upstream, validate-hypothesis, rl-alignment |

To use them with Claude Code, copy or link both sets into `~/.claude/skills/`
(for all your projects) or into `.claude/skills/` (for one project), with each
skill in its own folder:

```bash
git clone https://github.com/training-alignment/training-alignment-handbook
git clone https://github.com/ziming-zh/OpGuard
mkdir -p ~/.claude/skills
for d in training-alignment-handbook/skills/*/ OpGuard/harness/skills/*/; do ln -s "$(realpath "$d")" ~/.claude/skills/; done
```

Keep the repository checked out, because the skills link to its tools,
examples and pages. Start from `alignment-pipeline`, which sends each question
to the right skill. The RL steps need OpGuard's harness actions (`rl.*`,
`vectors.*`). They live in [this OpGuard repository](https://github.com/ziming-zh/OpGuard);
the upstream OpGuard does not have them yet.

## Skills, tools and operations

- **Skills** live in [skills/](skills/): [alignment-pipeline](skills/alignment-pipeline/SKILL.md),
  [pretraining-alignment](skills/pretraining-alignment/SKILL.md),
  [engine-parity](skills/engine-parity/SKILL.md),
  [batch-invariant-kernel](skills/batch-invariant-kernel/SKILL.md) and
  [handbook-writing](skills/handbook-writing/SKILL.md). OpGuard adds
  `rl-alignment` in its `harness/skills/` folder.
- **Tools** live in [tools/](tools/):
  - [bi_check.py](tools/bi_check.py) checks bitwise invariance. `--demo` runs the checks on stock PyTorch ops, and [a recorded run](tools/demo-a40.json) shows the output;
  - [bi_kernels.py](tools/bi_kernels.py) holds batch-invariant reference kernels for GEMM and attention, written in Triton;
  - [bi_sampling.py](tools/bi_sampling.py) is a reference sampler whose choices can be replayed;
  - [bi_audit.py](tools/bi_audit.py) lists a model's operators and gives each module an L2 verdict;
  - [bi_report.py](tools/bi_report.py) and [bi_suite.py](tools/bi_suite.py) write `bi-results-v1` files and run the regression suite in [suite/](suite/);
  - [determinism_preflight.py](tools/determinism_preflight.py) reads back the determinism settings from inside a running job and diffs the reports of two runs;
  - [opguard_bridge.py](tools/opguard_bridge.py) speaks OpGuard's formats, protocol and harness, and offers leads through `suggest()`;
  - [graph_probe.py](tools/graph_probe.py) takes fingerprints inside compiled code and CUDA graphs;
  - [data_fingerprint.py](tools/data_fingerprint.py) fingerprints each batch where it leaves the data loader, and its compare command classifies the first difference;
  - [ckpt_compare.py](tools/ckpt_compare.py) compares two checkpoints with different layouts, row by row;
  - [sdc_scan.py](tools/sdc_scan.py) scans GPUs for silent data corruption.
- **Other scenarios** are in [part 5](handbook/5-scenarios/README.md): framework migration, checkpoint conversion, offline post-training, serving and mixed hardware.
- **Operations** are in [part 4](handbook/4-operations/README.md):
  [regression checks](handbook/4-operations/regression.md) after an upgrade or
  on new hardware, and [hardware coverage](handbook/4-operations/hardware.md).
- **Evidence**: each example under [examples/](examples/) has a `REPORT.md`
  with its setup and measurements, and the suite's baselines are in
  [suite/](suite/).
- Reference pages: [Open items](handbook/open-items.md) · [References](handbook/references.md) ·
  [Glossary](handbook/glossary.md) · [Audit](AUDIT.md)

The handbook is also a bilingual website built from `handbook/`
([how to build it](handbook/4-operations/README.md#building-the-site)).
