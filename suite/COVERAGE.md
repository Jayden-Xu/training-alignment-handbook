# Measured coverage: stacks, baselines and cross-GPU results

This is the record behind the operations pages
[Regression](../handbook/4-operations/regression.md) and
[Hardware coverage](../handbook/4-operations/hardware.md). The pages say what to
run and how to read it; this report keeps the baselines recorded here, the
machines and their locks, and the cross-GPU comparison, with every number.
The handbook pages name no GPU model; this file is where the concrete
hardware is listed.

## What was measured

| Machine | GPU | Driver | Software | Interconnect | Used for |
|---|---|---|---|---|---|
| gpu-host | 1x NVIDIA A40, sm86, 84 SMs, 48 GB | 550.144.03 | torch 2.5.1+cu121, Triton 3.1.0, NCCL 2.21.5, cuBLAS 12.1.3.1, cuDNN 9.1.0 | — | every single-GPU kernel case, sampling, LoRA, training step, whole model, graph probes, SDC scan |
| node0 | 8x NVIDIA A16, sm86, 10 SMs, 15 GB each | 535.288.01 | as gpu-host | PCIe inside the node | collectives, MoE under EP, DP and FSDP-style (emulated: flat gradient, reduce-scatter, sharded AdamW), whole model under TP, SDC fleet, suite runs |
| node1 | 8x NVIDIA A16, as node0 | 535.309.01 | as gpu-host | PCIe inside the node | as node0; pre-training and RL-loop cases |
| node0 + node1 | 16x A16 | both | as gpu-host | 25 GbE sockets between the nodes, no RDMA | two-node collectives, EP 16, DP 16, TP 16 |
| node0, engine env | one A16 | 535.288.01 | vLLM 0.19.1, torch 2.10.0+cu128, Triton 3.6.0, NCCL 2.27.5 | — | trainer vs vLLM (L4) |
| node0, old engine env | one A16 | 535.288.01 | vLLM 0.6.3.post1, SGLang 0.4.0, torch 2.4.0 | — | the cross-engine case's older-vLLM rows |

Per result file: [references](../handbook/references.md#measurements-in-this-repository).
Suite baselines per stack: [below](#baselines-recorded-here-quick-mode).

Both GPU models are Ampere sm86. No other architecture was measured.

## Result files

Every result file sits next to its code. Unless a row says otherwise: torch
2.5.1+cu121, Triton 3.1.0, NCCL 2.21.5. A40: driver 550.144.03. A16 (sm86):
node0 driver 535.288.01, node1 driver 535.309.01. Older files (no `env`
field) state their stack on their topic page. All hardware is summarized in
[hardware coverage](../handbook/4-operations/hardware.md); suite baselines are in `suite/baselines/`.

| File | Hardware | Notes |
|---|---|---|
| `tools/demo-a40.json` | A40 | `bi_check.py --demo` |
| `examples/kernel_cases/results-a40.json` | A40 | |
| `examples/moe_combine/results-a40.json` | A40 | |
| `examples/attention_backward/results-a40.json` | A40 | |
| `examples/attention_serving/results-a40.json` | A40 | paged, varlen, compile, graphs |
| `examples/mla_sparse/results-a40.json` | A40 | |
| `examples/linear_attention/results-a40.json` | A40 | |
| `examples/lora/results-a40.json` | A40 | |
| `examples/sampling/results-a40.json` | A40 | |
| `examples/training_step/results-a40.json` | A40 | |
| `examples/whole_model/results-a40.json`, `results-hf-a40.json` | A40 | HF check with transformers from the `maia` env |
| `examples/graph_probe/results-a40.json` | A40 | run with an OpGuard checkout |
| `examples/sdc_scan/results-a40.json`, `results-guard-a40.json` | A40 | |
| `examples/sdc_scan/results-a16-fleet.json`, `results-ranks-a16.json` | 2x8 A16 | both drivers |
| `examples/pretraining_e2e/results-a16.json` | one A16 (node1) | OpGuard 39601f8 |
| `examples/rl_loop/results-a16.json`, `transcript-a16.json` | two A16 (node1) | OpGuard working tree with the RL harness |
| `examples/moe_ep/results-a16.json`, `results-a16-2node.json` | 8x A16, 2x8 A16 | 25 GbE between nodes |
| `examples/collectives_tp/results-a16.json`, `results-a16-2node.json` | 8x A16, 2x8 A16 | |
| `examples/training_dp/results-a16.json`, `results-a16-2node.json` | 8x A16, 2x8 A16 | |
| `examples/whole_model_tp/results-a16.json` | 8x A16 (TP 16 over two nodes) | |
| `examples/cross_engine/results-A16.json` | one A16 | vLLM 0.19.1, torch 2.10.0+cu128, Triton 3.6.0, NCCL 2.27.5 (and vLLM 0.6.3.post1, torch 2.4.0) |
| `examples/cross_engine/results-A16-torch251.json` | one A16 | `model.py` under torch 2.5.1+cu121 |

## Baselines recorded here (quick mode)

Quick mode, 2026-11-19, after the handbook's last integration. torch
2.5.1+cu121, Triton 3.1.0, NCCL 2.21.5, CUDA build 12.1 on all (the
`cross_engine` entry runs in its own vLLM environment: torch 2.10.0+cu128,
Triton 3.6.0; its baseline is filed under the runner's stack).

| Hardware | Driver | Entries | Checks (required) | Regressions | Changed | Run time |
|---|---|---|---|---|---|---|
| 1x A40 (48 GB) | 550.144.03 | 14 single-GPU (all but `cross_engine`, which needs the vLLM env) | 809 (443) | 0 | 0 | 6.0 min |
| 8x A16 (sm86, PCIe) | 535.288.01 | 15 single-GPU, `whole_model_tp` (4 GPUs), `training_dp`, `moe_ep`, `collectives_tp` | 1,366 (754) | 0 | 11 | 17.4 min |
| 2 nodes x 8x A16, 25 GbE | 535.288.01 + 535.309.01 | `moe_ep_2node`, `collectives_tp_2node`, `training_dp_2node` | 413 (223) | 0 | 0 | 6.2 min |

- Run time is the sum of the entries' times, without lock waits.
- Every required check passed on every stack. No regression against the
  previous baselines; the entries added in this round appear as NEW.
- The A16 "changed" count (11) is from the run that recorded the baseline:
  stock ops whose failure the topic pages measured on the A40 and which passed
  on the A16 at quick sizes: SDPA L2 (kernel cases), cuBLAS 1024→512 (MoE), the
  fp32 indexer matmul and one SDPA decode L2 (MLA), and 7 cuBLAS LoRA checks.
  The A16 baseline records them passing, so later A16 runs report them as
  `11 as baseline`, not CHANGED (checked by replaying the A16 baseline through
  `--from-report`: 0 regressions, 0 changed, 11 as baseline, exit 0). Each case
  page says the failure is A40-only at those sizes. On the A16 the
  stock CUDA-graph bucket 16 vs 1 comparison of `graph_probe` (an `info` check)
  also gave equal bits, unlike the A40.
- Two-node PATTERN lines: the atomic MoE combine (`arrival_atomic`, documented
  as not L1) differed from the one-node dump in another shape than at baseline
  time. Expected for an atomic: information only.
- TIMING lines (information): small collectives (32 KB) on 8 A16s ran 0.5–2.5x
  their baseline time; two-node timings moved by more than 1.5x in 28
  measurements. Timings on shared PCIe and 25 GbE vary run to run.
- Quick and full mode share check names. Quick verdicts differ from the
  committed full results only in stock pieces: 4 MLA/sparse checks (all
  `expect="info"`: at the quick length the cuBLAS projection's M is no longer
  2048·B, as the topic page's limits section predicts), 6–7 `sampling` top-p
  checks (stock, passing at quick sizes), and on the A16 the A40-only stock
  failures above. No required check differed.

## Stock results that are stack-specific

A stock op that fails a check is documented variance, not a claim about all
versions. When it starts to pass, or fails in another place, the stack changed
under it. Examples:

- The batch size at which `torch.mm` starts to vary (31 rows at 4096 × 4096 bf16
  on the A40 with torch 2.5.1) is a property of one cuBLAS version and one GPU.
  Another cuBLAS can move it, or remove it at the shapes tested.
- SDPA's backend choice and work split change between torch releases, so its
  L2 and L3 verdicts can change.
- NCCL's algorithm and protocol choice per message size changes between NCCL
  versions; Ring vs Tree decides the collectives verdicts.

What to do with a change:
1. Read the entry's log and result file; confirm the change is in the stock op,
   not in the harness.
2. Update the topic page's table, with the new stack named. The old row stays
   true for the old stack.
3. If a stock op now passes a check that motivated one of our kernels, the
   kernel is still needed on older stacks. Say which.
4. Record a new baseline.

A PATTERN line needs no action unless a topic page quotes the old shape (for
example "fails from M = 31"). Then update the page.

## Machines and locks

One rule: **one lock per node, and every GPU job on that node takes it.** A
job that spans two nodes holds the lock of both.

| Machine | Lock | Who takes it |
|---|---|---|
| gpu-host (1x A40) | `/tmp/bi-check-a40.lock` | every GPU command (`flock`, or the suite's `--lock-file`) |
| node1, node0 (8x A16 each) | `/tmp/bi-check-16gpu.lock` on that node | every GPU job on the node, one GPU or eight; the suite's `--lock-file` |
| node0 + node1 | both nodes' `/tmp/bi-check-16gpu.lock` | `~/bi-check-xnode/launch2.sh`: node0's lock, then node 1 runs under node1's |

Why one lock: per-GPU locks (`/tmp/bi-check-gpuN.lock`) for single-GPU jobs
on GPUs 4–7 existed next to the node lock taken by 8-GPU jobs. A job holding
only one of them ran on GPUs another job was using: one 8-GPU run failed with
an NCCL "unhandled cuda error" and another ran out of memory. Single-GPU jobs
now wait for the node; that costs throughput on a busy node, and is the price
of never sharing a GPU between two measurements.

- Never skip a lock that is held: wait.
- Entries with `own_lock` (multi-launch drivers, two-node entries) take the
  lock per launch; the runner only waits its turn before starting them.
- A command started inside an entry must not take the same lock again: the
  runner already holds it, and `flock` on a second descriptor waits forever.
  (`training_step`'s `BI_LOCK` is for running it by hand, not under the suite.)
- Leave no process running and no file over 100 MB in `/tmp` after a job.

## Limits of the suite as run here

- The CI workflow is **untested**. This repository has no remote, so it has
  never run. It targets a self-hosted GPU runner; its commands were run by
  hand, the workflow around them was not.
- Full mode was not run for this page; only quick mode has baselines. Full mode
  holds a shared GPU's lock for longer than the 5 minutes agreed on the A40.
- The full commands of the entries added last (`training_step`,
  `whole_model`, `sdc_scan`, `graph_probe`, `rl_loop`, `cross_engine`,
  `whole_model_tp`, `training_dp`, `training_dp_2node`, full
  `pretraining_e2e`) were written from each example's own usage and were not
  run through the suite.
- On a shared machine, a full-mode entry can hold the lock for many minutes.
- The comparison with committed results matches checks by name. A check whose
  quick size differs from the full one compares the same claim at other sizes.

## Same bits on two GPUs of one generation

The A40 and the A16 share the architecture (sm86) and every library version,
and differ in SM count (84 against 10), memory and driver. The SDC scan's golden
workload ran on both with identical inputs
([examples/sdc_scan/golden-a40.json](../examples/sdc_scan/golden-a40.json),
[golden-a16.json](../examples/sdc_scan/golden-a16.json)):

| Op family | Same bits on A40 and A16 |
|---|---|
| The handbook's Triton kernels (`mm_one_config`, fixed-block attention forward and backward) | 8 of 8 |
| Elementwise, casts, IEEE edge cases, integer ops, sorts, memory patterns, norms, softmax, cross-entropy, AdamW, deterministic SDPA backward | all |
| cuBLAS GEMMs, `bmm` and `linear` (bf16, fp16, fp32) | 4 of 13 |
| fp64 GEMM, fp32 GEMV | 2 of 2 |
| Large reductions (sum, dot, norm over 4–32 M elements) | 18 of 26 |
| Philox RNG (`rand`, `randn`, `randint`, `bernoulli`) | 0 of 4 |
| SDPA flash, fp16 | 0 of 1 |
| A bf16 training step (loss, gradients, parameters) | 0 of 3 |

In all, 273 of 298 fingerprints were equal. Two node0/node1 groups with
different drivers (535.288.01 and 535.309.01) gave 298 of 298 equal
([SDC scan case](../handbook/1-pretraining/sdc.md)).

The suite agrees: several stock failures the topic pages measured on the A40 did
not reproduce on the A16 at quick sizes (SDPA L2, a cuBLAS shape, the fp32
indexer, 7 cuBLAS LoRA checks); each topic page says so.

## Across GPU generations: bitwise agreement is not expected

Bitwise agreement between two GPU generations is generally not expected, even
with the same code, because:
- libraries choose kernels, tiles and split-K from the SM count and the
  architecture (measured above: cuBLAS differed on the same architecture with
  another SM count);
- reduction and RNG kernels size their grids from the SM count (measured above:
  large reductions and every Philox stream differed);
- each generation has its own matrix instructions (Ampere `mma.sync`, Hopper
  `wgmma`, Blackwell `tcgen05`), and nothing guarantees the same accumulation
  order or intermediate precision across them. Reported: the DeepSeek-V3
  technical report (arXiv 2412.19437) states that FP8 GEMM accumulation on
  H800 tensor cores keeps about 14 bits, which is why it promotes partial sums
  to CUDA cores at a fixed interval;
- libraries ship different kernels per architecture: FlashAttention 3 is
  Hopper-only, FP8 tensor cores start at sm89, NVFP4 at sm100;
- Triton compiles per architecture, so the same Triton source becomes different
  machine code.

What can carry over is a kernel whose reduction order is written in its source
and does not depend on the device: the handbook's Triton kernels gave the same
bits on 84 and 10 SMs. Whether they give the same bits on another architecture
is **untested**.

So: verify each invariance level (L1–L5) on each stack. Claim cross-generation
equality only after measuring it.

## Untested

This table is about this handbook's own measurements (its kernels and fixes,
the fixed collective trees, the engine-parity setup). OpGuard itself is used
in production on these platforms, as its authors report
([OpGuard paper](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)).

| Area | Status |
|---|---|
| Hopper (sm90), Blackwell (sm100/sm120), Ada (sm89) | untested; no GPU available |
| FP8 tensor-core GEMM, FP8 promotion interval | untested (FP8 scales were emulated on sm86) |
| NVFP4 | untested |
| FlashAttention 3, FlashMLA, FlashInfer on newer GPUs | untested (vLLM's FlashInfer backend was run on an A16 only) |
| NVLink, NVSwitch, NVLS | untested (PCIe only) |
| RDMA / InfiniBand / RoCE | untested (25 GbE sockets only) |
| More than two nodes | untested; a pinned Tree that held on two nodes may not hold on three |
| Other torch, Triton or NCCL versions | only vLLM's own stack (torch 2.10.0, Triton 3.6.0, NCCL 2.27.5) for the cross-engine case |
| CUDA builds other than 12.1 (12.8 in the vLLM env) | untested for the handbook's kernels |
| AMD, other accelerators | untested |
| SGLang at L4 | untested: the node0 environments lack libcudnn 9 and libnvJitLink |
