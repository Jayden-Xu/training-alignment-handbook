# Open items

Each page of the three workflows says, in a short "not yet shown" line, what
changes what a reader would do. This page keeps the full list: everything the
handbook has not run, could not run, or has not designed yet. Sources: the "Limits" sections of the topic pages in parts 1–3,
[regression](4-operations/regression.md#limits), [hardware coverage](4-operations/hardware.md#untested) and
the [audit](../AUDIT.md). The topic pages keep the details; this page keeps the
list. When an item is done, remove it here and update its topic page.

## Needs hardware (or software that cannot be installed here)

OpGuard itself (capture, fingerprints, first-divergence comparison) is used in
production LLM training on the platforms in the first four items, as its
authors report ([OpGuard paper](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)). What is open is this handbook's own
measurements there: its batch-invariant kernels and fixes, the fixed
collective trees, and the engine-parity setup.

- Other GPU architectures and other accelerators: none available here. The
  measured stacks and this split are in
  [hardware coverage](4-operations/hardware.md#untested).
- FP8 tensor-core GEMM and its promotion interval, NVFP4; FlashAttention 3,
  FlashMLA and FlashInfer on other GPU architectures (FP8 scales were emulated).
- NVLink, NVSwitch, NVLS; RDMA, InfiniBand, RoCE; more than two nodes. All of
  the handbook's multi-GPU results are over PCIe inside a node and plain
  Ethernet sockets between two nodes.
- Other torch, Triton, NCCL and CUDA versions for the handbook's kernels (one
  stack, plus vLLM's own stack for the cross-engine case).
- SGLang at L4: the engine environment here lacks CUDA libraries SGLang needs.
- Real pretrained or MoE checkpoints in an engine (L4 with TP/EP, prefix
  caching, speculative decoding): needs weight downloads. A real architecture
  was checked only with random weights and a few of its layers
  ([whole-model report](../examples/whole_model/REPORT.md)).
- `fla` (flash-linear-attention) as a production baseline for the GDN kernels:
  not installed.

## Needs GPU time on the existing machines

- Real `torch.distributed.fsdp`: the probe is written, never launched. The
  [DP case](1-pretraining/data-parallel.md) is FSDP-style (emulated: flat gradient,
  reduce-scatter, sharded AdamW). Also untested there: bf16 gradients, mixed
  precision, loss scaling.
- Suite full mode: never run through `tools/bi_suite.py`; the full commands of
  the entries added last are unverified, and only quick baselines exist.
- The CI workflow (`.github/workflows/bi-suite.yml`): never run; needs one run
  on a self-hosted GPU runner.
- `perfetto/token_match.py` (OpGuard) on the [rl-loop](2-rl/rl-loop.md)
  exports: never run.
- RL replay: batch-invariant against original-composition replays not compared;
  mixed-version batches not recaptured.
- OpGuard `tests/test_xor_vectors.py`: the GPU-only paths (native kernel against
  the fallback, v1 capture for every dtype, the workload through the tracer)
  were not re-run after the last XOR-vector change.
- The audit's GPU spot re-run: checksums against `~/bi-check-suite`, and
  `lora_bi.py --quick` and `sampling_bi.py --quick` on the second GPU type
  against its baseline ([AUDIT.md section 5](../AUDIT.md#5-gpu-re-run)).
- Full sizes on the second GPU type for the stock failures that pass there at
  quick sizes (kernel cases, LoRA, MLA, MoE): those failures are claimed for
  the first GPU type only ([measured coverage](../suite/COVERAGE.md)).
- Pre-training through OpGuard: multi-rank captures, CUPTI and CUDA-API
  call-site backends, fused AdamW against foreach.
- `torch.compile` and CUDA-graph bits for kernels measured eager only:
  attention backward, paged and varlen kernels, linear attention, LoRA, MLA and
  sparse attention, sampling, the whole model, TP.
- Backward paths not run: MoE under EP, TP parameter gradients, the KV-cache
  path.

## Needs an engine or framework integration

- Record hooks (`RecordWriter.forward`, `rollout_batch`, `rows`) for vLLM,
  SGLang, verl and OpenRLHF. Records were only ever written from the toy engine
  in `examples/rl_loop/rl_core.py`.
- A vLLM-matched trainer backward: vLLM's Triton GEMM, RMSNorm, `log_softmax`
  and its FlashAttention build have no backward here, so the matched trainer
  is forward only.
- Compiled vLLM parity: compiled and CUDA-graph vLLM (its default) was not
  matched; localizing it needs OpGuard per-vector fingerprints inside compiled
  graphs, not yet shown on any engine.
- vLLM at L4 beyond the tested case: prefix caching, speculative decoding,
  temperature sampling, CUDA graphs in batch-invariant mode, TP, MoE, LoRA.
- L4 against a trainer for the kernel cases: MoE (combine and EP), MLA and
  sparse attention, linear attention, attention kernels.
- DeepSpeed ZeRO-1/2/3, Megatron's kernels and custom all-reduce.

## Needs design work

- A blind trial: an agent that is not told the planted cause follows
  `alignment-pipeline` (pre-training and RL paths), with launches and harness
  calls recorded. Every end-to-end case so far is known-answer.
- Fused AdamW and fused SGD: their exact summation order is not established;
  the order models are close, not exact ([training step](1-pretraining/training-step.md)).
- OpGuard coverage: GEMMs inside `torch.nn.Linear` are not captured (launched
  from torch's own file), and one-letter variable names in model code block
  wrapping ([pre-training case](1-pretraining/localize-with-opguard.md)).
- Kernels not built: a backward for the recurrent GDN forward, a paged
  attention backward, fixed-size split-KV for long decode.
- MoE routing rules: capacity-based dropping applied on the expert rank, and a
  (score, expert id) tie-break for exact ties.
