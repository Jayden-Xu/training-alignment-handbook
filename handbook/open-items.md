# Open items

Each page of the three workflows has a short "not yet shown" line for the gaps
that would change what a reader does. This page keeps the full list:
everything the handbook has not run, could not run, or has not designed yet.
The items come from the "Limits" sections of the topic pages in parts 1–3,
from [regression](4-operations/regression.md#limits), from
[hardware coverage](4-operations/hardware.md#untested) and from the
[audit](../AUDIT.md). The topic pages keep the details, and this page keeps
the list. When an item is done, remove it here and update its topic page.

## Items that need hardware, or software that cannot be installed here

OpGuard itself (capture, fingerprints and the comparison that finds the first
divergence) is used in production LLM training on the platforms in the first
four items, as its authors report
([OpGuard paper](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)).
What remains open is this handbook's own measurements on those platforms: its
batch-invariant kernels and fixes, the fixed collective trees and the
engine-parity setup.

- Other GPU architectures and other accelerators: none is available here. The
  measured stacks, and the split between OpGuard and this handbook, are
  described in [hardware coverage](4-operations/hardware.md#untested).
- FP8 tensor-core GEMM with its promotion interval, and NVFP4; FlashAttention
  3, FlashMLA and FlashInfer on other GPU architectures. The FP8 scales so far
  were emulated.
- NVLink, NVSwitch and NVLS; RDMA, InfiniBand and RoCE; more than two nodes.
  All of the handbook's multi-GPU results ran over PCIe inside a node and over
  plain Ethernet sockets between two nodes.
- Other torch, Triton, NCCL and CUDA versions for the handbook's kernels. So
  far there is one stack, plus vLLM's own stack for the cross-engine case and
  the stack of the verl × FSDP2 × vLLM case
  ([report](../examples/verl_fsdp2_vllm/REPORT.md)).
- SGLang at L4: the engine environment here lacks CUDA libraries that SGLang
  needs.
- Real pretrained or MoE checkpoints in an engine: a real MoE checkpoint has
  reached L4 at TP 1 inside a full RL loop
  ([report](../examples/verl_fsdp2_vllm/REPORT.md)). L4 with TP or EP, prefix
  caching and speculative decoding remain open: this needs weight downloads.
  In the [whole-model case](../examples/whole_model/REPORT.md), one real
  architecture was checked only with random weights, and only for a few of its
  layers.
- `fla` (flash-linear-attention) as a production baseline for the GDN
  kernels: it is not installed.

## Items that need GPU time on the existing machines

- Real `torch.distributed.fsdp`: the probe is written but has never been
  launched. The [DP case](1-pretraining/data-parallel.md) is FSDP-style and
  emulated (a flat gradient, reduce-scatter, sharded AdamW). It also leaves
  bf16 gradients, mixed precision and loss scaling untested.
- Full mode of the suite: it has never been run through `tools/bi_suite.py`.
  The full commands of the most recently added entries are unverified, and
  only quick-mode baselines exist.
- The CI workflow (`.github/workflows/bi-suite.yml`) has never run; it needs
  one run on a self-hosted GPU runner.
- `perfetto/token_match.py` (OpGuard) has never been run on the
  [rl-loop](2-rl/rl-loop.md) exports.
- RL replay: batch-invariant replays have not been compared with replays in
  the original batch composition, and batches that mix policy versions have
  not been captured again.
- OpGuard `tests/test_xor_vectors.py`: the GPU-only paths (the native kernel
  against the fallback, v1 capture for every dtype, the workload through the
  tracer) were not re-run after the last change to XOR vectors.
- The audit's spot re-run on GPUs: the checksums against `~/bi-check-suite`,
  and `lora_bi.py --quick` and `sampling_bi.py --quick` on the second GPU type
  against its baseline ([AUDIT.md section 5](../AUDIT.md#5-gpu-re-run)).
- Full sizes on the second GPU type for the stock failures that pass there at
  quick sizes (kernel cases, LoRA, MLA, MoE). Until then, those failures are
  claimed for the first GPU type only
  ([measured coverage](../suite/COVERAGE.md)).
- Pre-training through OpGuard: captures across several ranks, the CUPTI and
  CUDA-API call-site backends, and fused AdamW compared with the foreach
  version.
- `torch.compile` and CUDA-graph results for the kernels that were measured
  only in eager mode: attention backward, paged and varlen kernels, linear
  attention, LoRA, MLA and sparse attention, sampling, the whole model, and
  TP.
- Backward paths that have not been run: MoE under EP, TP parameter
  gradients, and the KV-cache path.

## Items that need an engine or framework integration

- Record hooks (`RecordWriter.forward`, `rollout_batch`, `rows`) for vLLM,
  SGLang, verl and OpenRLHF. So far, records have only been written from the
  toy engine in `examples/rl_loop/rl_core.py`.
- A trainer backward that matches vLLM: vLLM's Triton GEMM, RMSNorm,
  `log_softmax` and its FlashAttention build have no backward here, so the
  matched trainer runs the forward pass only.
- Parity with compiled vLLM: vLLM with torch.compile and CUDA graphs (its
  default) was not matched. Localizing a difference there needs OpGuard
  per-vector fingerprints inside compiled graphs, which has not yet been shown
  on any engine.
- vLLM at L4 beyond the tested case: prefix caching, speculative decoding,
  temperature sampling, CUDA graphs in batch-invariant mode, TP, MoE and LoRA.
- L4 against a trainer for the kernel cases: MoE (combine and EP), MLA and
  sparse attention, linear attention, and the attention kernels.
- DeepSpeed ZeRO-1/2/3, Megatron's kernels and its custom all-reduce.

## Items that need design work

- A blind trial, in which an agent that is not told the planted cause follows
  `alignment-pipeline` (both the pre-training and the RL path) while its
  launches and harness calls are recorded. Every end-to-end case so far is a
  known-answer demonstration.
- Fused AdamW and fused SGD: their exact summation order has not been
  established. Our models of their order come close but are not exact
  ([training step](1-pretraining/training-step.md)).
- OpGuard coverage: GEMMs inside `torch.nn.Linear` are not captured, because
  they are launched from torch's own file, and one-letter variable names in
  model code prevent wrapping
  ([pre-training case](1-pretraining/localize-with-opguard.md)).
- Kernels not yet built: a backward for the recurrent GDN forward, a backward
  for paged attention, and fixed-size split-KV for long decodes.
- MoE routing rules: capacity-based dropping applied on the expert's rank, and
  a (score, expert id) tie-break for exact ties.
