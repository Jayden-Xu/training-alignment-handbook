# verl × FSDP2 × vLLM on gpt-oss-20b (L4 in a full RL loop): lab report

This is the record behind the handbook page
[Engine parity](../../handbook/2-rl/engine-parity-vllm.md). The page teaches the
step; this report keeps the setup, the measurements and how the case was built.
Compared with the [cross-engine case](../cross_engine/REPORT.md), it adds a full
RL loop in a real framework (verl), real FSDP2 sharding, a real MoE checkpoint
with attention sinks and sliding-window attention, a rollout that runs with CUDA
graphs, and a training backward with a gradient check.

## Summary

- gpt-oss-20b, GRPO on GSM8K, verl with FSDP2 over 16 L20 GPUs and a colocated
  vLLM rollout with CUDA graphs. In the baseline (verl's native Hugging Face
  path, batch invariance off), 263,018 of 263,055 logprobs differ between
  rollout and training over 10 steps, with a mean k3 KL of 1.5e-3.
- **Fix:** vLLM runs unchanged apart from its batch-invariant mode, and the
  training forward calls vLLM's own linear, attention, MoE, RMSNorm, rotary and
  logprob kernels. Six divergences were found on a 4-layer proxy, and all six
  were fixed on the training side.
- **Result:** 0 of 257,479 logprobs differ over 10 steps (mean k3 KL 0). At
  each of the 11 weight syncs, every parameter tensor in all 16 vLLM replicas
  matched the trainer's by SHA.
- Gradients are not required to be bitwise. Every one of the 71 trainable
  tensors of the 4-layer proxy passes relative L2 ≤ 1e-3 against an
  independent FP32 reference; the largest error is 1.9e-7.
- **Cost:** turning Inductor off in the rollout cost 0.12% decode throughput.
  No other cost was measured.

## How

The [cross-engine case](../cross_engine/REPORT.md) matched a forward-only
trainer to eager vLLM on a small dense model. This case does the same inside a
running RL job, where the trainer also has to take a backward step and the
rollout uses CUDA graphs. Two rules decide where each side's values come from.

**Rule 1: the inference kernels define the values.** vLLM runs unchanged apart
from its batch-invariant mode, and the training forward calls the same kernels.
verl's `batch_invariant` switch covers softmax, log_softmax, mean and the
batched GEMMs. Beyond that switch, the training forward calls from the start
vLLM's batch-invariant linear, its attention (`kernel_unified_attention`, with
the sinks and the 128-token window) and its MoE (`fused_topk`,
`TritonExperts`).

**Rule 2: logprobs must match bit for bit; gradients only need to be correct.**
The backward may compute differently from the forward kernels, as long as every
gradient passes relative L2 ≤ 1e-3 against an independent FP32 reference. Small
operators get hand-written FP32 derivatives. Attention and MoE are recomputed in
PyTorch on the saved inputs to get their gradients, while the forward keeps the
kernels' values.

**The rollout uses CUDA graphs without Inductor.** Inductor's fusions skip
intermediate BF16 roundings, and training would then have to reproduce those
fused programs. The rollout recipe therefore sets vLLM's compilation mode 0, a
FULL CUDA graph and a selection of custom ops. Turning Inductor off cost 0.12%
decode throughput.

### The fix

Six divergences were found on the 4-layer proxy (two sliding-window and two
full-attention layers), before the full model was run. All fixes are on the
training side. The counts are the elements or logprobs that differed for each
divergence.

| Module | What differed | Fix |
|---|---|---|
| YaRN cache | FSDP2 casts the FP32 cos/sin table to BF16 when it crosses a sharded-module boundary: all 2,048 logprobs differ | Pass positions across the boundary and build the table inside the layer |
| YaRN rotation | Cache construction and rotation differ from vLLM's CUDA op: 968,505 of 2,703,360 Q elements and 119,704 of 337,920 K elements | Build the FP32 table on CUDA, round it to BF16, and call vLLM's `rotary_embedding` |
| Input RMSNorm | The reduction differs from vLLM's batch-invariant RMSNorm: 25 of 1,900,800 elements | Call vLLM's batch-invariant RMSNorm |
| Post-attention, cross-layer and final add+RMSNorm | vLLM's fused kernel rounds the sum to BF16, then computes the FP32 variance on the rounded value; training's earlier path did not round: 431,501 of 1,900,800 elements | Pass both BF16 operands to vLLM's fused add+RMSNorm and carry its BF16 residual |
| Token logprob | The vocabulary reduction order differs from vLLM's: 18 of 283 logprobs | Call vLLM's `compute_token_logprobs`, with an FP32 VJP |
| Vocabulary mask | The LM head has 201,088 columns; vLLM masks the 1,069 beyond the tokenizer's 200,019 tokens, while training normalized over all of them: 64,065 of 65,536 logprobs | Keep the full width, fill the tail with −inf, and use the same vLLM kernel |

## Evidence

### Setup, code and results

- verl: upstream [verl-project/verl](https://github.com/verl-project/verl) at
  `453954b`, which already contains the `batch_invariant` switch
  ([PR #8026](https://github.com/verl-project/verl/pull/8026), commit
  `2b4ed682`, merged 2026-11-10). On top of it: a cherry-pick of
  [PR #8736](https://github.com/verl-project/verl/pull/8736) (commit `644347f`,
  a fix for the `batch_invariant` backward on SM8x, not yet merged upstream),
  and this case's patch. The case patch adds a gpt-oss branch in the worker, a
  `gpt_oss_case2` model patch, vocabulary masking, and the rollout recipe (vLLM
  compilation mode 0, FULL CUDA graph, the custom-ops selection).
- Other software: vLLM 0.32.0 (commit `5df63a7`), torch 2.13.0+cu130 (CUDA
  13.0), Triton 3.7.1, transformers 5.16.1, Python 3.12.11.
- Hardware: one node with 16 NVIDIA L20 GPUs (SM89, 46,068 MiB each), driver
  535.261.03.
- Layout: training and rollout are colocated (verl's hybrid engine). FSDP2
  shards across all 16 GPUs, and the rollout is 16 vLLM replicas at TP1, one
  per GPU. Parameters and optimizer states are offloaded to the CPU so that the
  colocated rollout fits. Weight sync uses 128 MiB IPC buckets.
- Model: gpt-oss-20b, an MoE with 24 layers, attention sinks and sliding-window
  attention (128-token window). BF16 weights dequantized from the official
  MXFP4 checkpoint. A 4-layer proxy (two sliding-window and two full-attention
  layers) was aligned before the full model.
- Precision: master weights and AdamW states in BF16; gradients reduced in
  FP32.
- Workload: GRPO on GSM8K, 16 prompts × 8 samples per step, at most 512
  response tokens, 10 steps per run, gpt-oss reasoning effort "low".
- Data: produced by verl's `examples/data_preprocess/gsm8k.py` at
  `d62da4950573d7a4b7ef2362337952e7ab59e78d`, from Hugging Face
  [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (config `main`,
  snapshot `740312add88f781978c0658806c59bc2815b9866`); 7,473 train and 1,319
  test rows.
- Environment: the env file that the scripts source (`CASE2_ENV_SH`) must be
  prepared by the user; the header of `tic/scripts/train_tic.sh` lists what it
  has to provide.
- Workflow: vLLM's official
  [train-inference-consistency skill](https://github.com/aoshen02/vllm/tree/939215b66ce8fef6ebc2009de3d6b254bdd371ca/.agents/skills/train-inference-consistency),
  followed gate by gate. This handbook's
  [engine-parity skill](../../skills/engine-parity/SKILL.md) was not used.
  [OpGuard](https://github.com/OrderLab/OpGuard) localized the divergences.
- Files in this folder. The layout mirrors the cluster's: `CASE2_TIC_ROOT` is
  `tic/`, the patched verl checkout goes at `tic/verl`, and `$case_root/audit`
  is `audit/`. The launch chain is `c9_suite.sh` → `train_tic_job.sh` →
  `train_tic.sh` → `verl.trainer.main_ppo`, after which `c9_results.py`
  verifies and summarizes each arm.
  - [verl.patch](verl.patch): the full diff against `453954b` (the PR #8736
    cherry-pick, the case patch and three new files).
  - [aligned.yaml](aligned.yaml) and [baseline.yaml](baseline.yaml): the
    resolved configurations of the aligned and the baseline arm, as written
    by `train_tic_job.sh`.
  - [tic/scripts/c9_suite.sh](tic/scripts/c9_suite.sh): the entry point; it
    runs the aligned arm and then the baseline arm (or only the baseline,
    against an archived aligned arm, when `CASE2_C9_ALIGNED_SUITE` is set),
    picks a free rendezvous port for each arm, and calls `c9_results.py`.
  - [tic/scripts/train_tic_job.sh](tic/scripts/train_tic_job.sh): creates the
    arm's run folder and status file, writes the resolved configuration with
    `--cfg job --resolve`, then runs `train_tic.sh` at TP 1 with all output in
    `run.log`.
  - [tic/scripts/train_tic.sh](tic/scripts/train_tic.sh): starts
    `verl.trainer.main_ppo` for one arm with the case's Hydra overrides; in
    the aligned arm it turns on verl's `batch_invariant` switch, exports
    `VLLM_BATCH_INVARIANT=1` and passes the compilation config below. Its
    header lists the environment variables.
  - [tic/scripts/c9_results.py](tic/scripts/c9_results.py): verifies each arm
    (`verify_arm`: every rank's training path, the ten per-step scores, which
    must all be zero in the aligned arm, and the SHA check of all 11 syncs
    through `check_sync`), re-checks an archived aligned arm, and writes
    `results.csv`. `c9_suite.sh` and `c9_results.py` also update the coding
    agent's progress notes (`PROGRESS.md` and `report_tic.md` at the top of
    this folder, and `STATE.md` one level above), which are not included; to
    run the suite from this folder, create those files or remove those calls.
  - [tic/scripts/audit/sitecustomize.py](tic/scripts/audit/sitecustomize.py):
    loaded from `PYTHONPATH` in every process of a run; when a run folder is
    set, it hooks five verl modules on import to install the observers in
    `tic_audit.py`.
  - [tic/scripts/audit/tic_audit.py](tic/scripts/audit/tic_audit.py): the
    observers. They record and check each rank's training path, compare old
    and rollout logprobs at every step and stop the aligned arm before the
    update on any difference, pass the rendezvous port to verl's worker group,
    and install the SHA hooks from `gate0_audit.py`.
  - [tic/steps/C3_graph_only_config.json](tic/steps/C3_graph_only_config.json):
    the aligned arm's vLLM compilation config (mode 0, FULL CUDA graph, custom
    ops `none`, `+rms_norm`, `+rotary_embedding`).
  - [audit/gate0_audit.py](audit/gate0_audit.py): hashes every tensor with
    SHA-256 on the sender, on each IPC receiver and in each vLLM replica after
    loading, and compares them for one sync (`check_sync`).
- No results JSON exists for this case. Per-step mismatch counts and k3 KL are
  computed by `tic_audit.py` (`observe_trainer.compare`:
  Δ = old_log_prob − rollout_log_prob, k3 = mean(expm1(Δ) − Δ) over valid
  response tokens, in FP64), the weight-sync SHA check by `tic_audit.py`'s
  hook and `gate0_audit.py`, and `c9_results.py` summarizes and verifies each
  arm (`verify_arm`, `check_sync`); the run logs are not published.
- This case is not in the
  [regression suite](../../handbook/4-operations/regression.md), because it
  needs 16 GPUs.

### Metrics

- **Mismatched logprobs:** FP32 logprobs whose bit patterns differ between the
  rollout and the training side's old-logprob recomputation before each
  update, counted per value. Only the old-logprob path is tracked, not the
  actor path.
- **k3 KL:** the mean of exp(δ) − 1 − δ over response tokens, where
  δ = old logprob − rollout logprob, computed in FP64. It estimates
  KL(π_rollout ‖ π_old).
- **Weight-sync check:** after each sync, the SHA of every physical parameter
  tensor in each vLLM replica is compared with the trainer's.

### Results

| Run | Mismatched logprobs (10 steps) | Per step | Mean k3 KL |
|---|---|---|---|
| Baseline | 263,018 / 263,055 | 99.98–99.99% | 1.5e-3 |
| Aligned | **0 / 257,479** | 0% | 0 |

At each of the 11 weight syncs of the aligned run, every parameter tensor in
all 16 vLLM replicas matched the trainer's by SHA.

The baseline is verl's native Hugging Face path with batch invariance off.
In [baseline.yaml](baseline.yaml), verl's `batch_invariant` switch is false,
and the rollout runs vLLM with the same compilation mode 0 and FULL CUDA graph
as the aligned run (`{"mode":0,"cudagraph_mode":"FULL"}`), but without the
aligned run's custom-ops selection (`["none", "+rms_norm",
"+rotary_embedding"]`); [train_tic.sh](tic/scripts/train_tic.sh) exports
`VLLM_BATCH_INVARIANT=1` only in the aligned arm, so the baseline rollout runs
without vLLM's batch-invariant mode. In the baseline arm only,
[sitecustomize.py](tic/scripts/audit/sitecustomize.py) hooks
`verl.trainer.constants_ppo`, and `tic_audit.py`'s `observe_ppo_runtime_env`
then wraps verl's `get_ppo_ray_runtime_env`: it fails if
`VLLM_BATCH_INVARIANT` is set in the driver or forwarded with a value other
than `"0"`, and it removes the `"0"` that verl forwards for an absent variable,
so the baseline's Ray workers see `VLLM_BATCH_INVARIANT` unset. This hook is
outside `verl.patch`. The totals differ because each run samples its own
responses.

### Gradient check

On the 4-layer proxy, all 71 trainable tensors (4.45B parameters), including
the attention sinks and the expert biases, pass relative L2 ≤ 1e-3 against an
independent FP32 reference. The largest relative L2 is 1.9e-7, on layer 0's
attention sinks.

### What this shows

In this setup, a training forward that calls vLLM's own kernels gives FP32
logprobs that equal the CUDA-graph rollout's bit for bit over 10 GRPO steps on
gpt-oss-20b, with FSDP2 sharding across 16 GPUs, while verl's native path
differs on nearly every logprob. vLLM needed only its batch-invariant mode and
the rollout recipe; the six divergences were all fixed on the training side.
Bitwise logprobs did not require a bitwise backward: gradients recomputed in
PyTorch passed a relative L2 check against an FP32 reference on the proxy.

## Limits

- One GPU type (L20, SM89). Kernels can change on other generations.
- BF16 only: the rollout runs dequantized BF16 weights. Nothing is claimed
  about MXFP4 or FP8 alignment.
- Rollout at TP1 only. The run is synchronous and on-policy (staleness 0).
- No throughput, latency or training-quality measurement beyond the 0.12%
  Inductor cost. Ten steps show that logprobs stay bitwise equal, not that
  training converges.
- The gradient check ran on the 4-layer proxy, not on the full 24-layer model.
- OpGuard records host-side Python call sites and does not yet record inside
  CUDA graph replay. Localization therefore covered the eager and preflight
  runs, not the CUDA graph path the rollout uses.
- The case relies on batch-invariant kernels that vLLM already ships.
  Operators without them need new kernels first.

## How it was built

A coding agent did the alignment end to end, following vLLM's
[train-inference-consistency skill](https://github.com/aoshen02/vllm/tree/939215b66ce8fef6ebc2009de3d6b254bdd371ca/.agents/skills/train-inference-consistency)
gate by gate, and used [OpGuard](https://github.com/OrderLab/OpGuard) to
localize divergences. The agent found and fixed every divergence in the table
above, first on the 4-layer proxy. People set the scope, approved the
full-model recipe at the skill's gate C9, and reviewed the results.
