# Hardware coverage: what changes bits, and qualifying new hardware

Every bitwise claim in this handbook holds on the stack it was shown on. When
you move to other GPUs, another driver or other libraries, the
[regression suite](regression.md) tells you what still holds. This page explains
which properties of a stack can change the bits, why two GPU generations are
not expected to agree, and how to qualify new hardware.

## The idea

Add 4,096 numbers on a GPU with many SMs and the library may split them into
many partial sums; on a GPU with fewer SMs it may split them into fewer. Each
split rounds at different places, so the total can differ in the last bit,
though the source code and the inputs are the same. The rule: **anything that
decides how a reduction is split, ordered or rounded is part of the stack, and
a bitwise result belongs to that stack.** It is the same principle as batch
invariance, one level up: there the batch chose the kernel, here the machine
does.

## What changes bits

| Property | How it changes bits |
|---|---|
| Architecture | Each generation has its own matrix instructions, and nothing guarantees the same accumulation order or intermediate precision across them. Libraries ship different kernels per architecture, and some kernels or number formats exist on newer generations only. Triton compiles per architecture, so the same source becomes different machine code. |
| SM count | Libraries choose kernels, tiles and split-K from the SM count; reduction and random-number kernels size their grids from it. Two GPUs of one architecture with different SM counts can differ. |
| Driver | Two driver versions with the same libraries gave the same bits in the measured scan; the driver is still part of the baseline key, since it ships parts of CUDA. |
| CUDA build and libraries | cuBLAS, cuDNN and NCCL choose kernels, algorithms and protocols per version; torch changes which backend an operator uses (for example SDPA's). |
| Interconnect | NCCL chooses its algorithm (Ring or Tree) and protocol from the topology and message size, and the algorithm decides the summation order of a collective. |

What can carry over is a kernel whose reduction order is written in its source
and does not depend on the device: the handbook's Triton kernels gave the same
bits on two GPUs of one generation with very different SM counts, while stock
GEMMs, large reductions and random-number streams did not. Whether those
kernels give the same bits on another architecture is **untested**. Measured:
see [measured coverage](../../suite/COVERAGE.md#same-bits-on-two-gpus-of-one-generation).

Across generations, bitwise agreement is not expected. A reported example: the
DeepSeek-V3 technical report (arXiv 2412.19437) states that FP8 GEMM
accumulation on newer tensor cores keeps limited precision, which is why it
promotes partial sums to CUDA cores at a fixed interval. So verify each
invariance level (L1–L5) on each stack, and claim equality across generations
only after measuring it.

## Qualifying new hardware

**Goal:** know which claims hold on the new stack, and which stock behaviour differs.
**Success:** a reviewed baseline for the new stack is committed, and every
difference is either a recorded stack-specific stock result or a reported finding.

1. Run the regression suite in full mode with `--vs-committed`, as
   [regression, New hardware](regression.md#new-hardware) describes
   ([tools/bi_suite.py](../../tools/bi_suite.py)).
2. Run the SDC scan with `--cross-group-info` against the committed golden
   files ([tools/sdc_scan.py](../../tools/sdc_scan.py)): it shows, as
   information, which operator families give the same bits as the stacks
   already measured.
3. Every required check that fails is a finding: report it. Every documented
   stock failure that now passes is a change to record on its topic page, with
   the new stack named.
4. Record the baseline, and add the stack to
   [measured coverage](../../suite/COVERAGE.md#what-was-measured).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| suite, quick mode | every claim at reduced sizes | small | a first look, or a minor library update |
| suite, full mode with `--vs-committed` | the sizes the pages were written from, and what moved | large | new GPUs, a new architecture, before a release |
| SDC scan, cross-group | which operator families match the measured stacks bit for bit | small | new GPUs, to see what can be compared across machines |

## Untested

Two things must be kept apart here. **OpGuard itself** (capture, fingerprints,
first-divergence comparison) is used in production LLM training on the
platforms below, as its authors report
([OpGuard paper](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)). **This handbook's own measurements** (the
batch-invariant kernels and fixes, the fixed collective trees, the
engine-parity setup) come from one GPU generation, over PCIe inside a node and
plain Ethernet between two nodes, and are not yet shown on them:

| Area | OpGuard | This handbook's kernels, fixes and setups |
|---|---|---|
| Other GPU architectures, other accelerators | used in production (reported) | untested |
| FP8 tensor-core GEMM and its promotion interval, NVFP4 | used in production (reported) | untested (FP8 scales were emulated) |
| Newer attention kernels (FlashAttention 3, FlashMLA, FlashInfer) on other GPU architectures | used in production (reported) | untested |
| NVLink, NVSwitch, NVLS; RDMA, InfiniBand, RoCE | used in production (reported) | untested; the fixed collective trees were shown over PCIe and Ethernet only |
| More than two nodes | used in production (reported) | untested; a pinned Tree that held on two nodes may not hold on three |
| Other torch, Triton, NCCL and CUDA versions | used in production (reported) | untested, except vLLM's own stack in the cross-engine case |

So on such a platform, localize with OpGuard as usual; re-verify every
invariance claim of this handbook with the suite before relying on it.
The concrete stacks, per result file, are in
[measured coverage](../../suite/COVERAGE.md); the full list of open work is in
[open items](../open-items.md).

## What can go wrong

!!! warning "Assuming bits carry over"
    A new GPU looks like a faster old one, but its library picks other kernels.
    A claim carried over unmeasured may be false on it. Run the suite before
    relying on any bitwise claim.

!!! warning "Comparing across machines with stock ops"
    Two machines of different types running stock GEMMs or random-number
    kernels can disagree with no bug anywhere. Compare across machines only on
    operators whose order is fixed in their source, or compare each machine
    with itself.

!!! warning "Tuning a failure away on new hardware"
    A required check that fails on new hardware is a finding about that
    hardware or its libraries. Hiding it with a tolerance makes every later
    claim on that stack untrustworthy. Report it.

## Summary

- A bitwise result belongs to its stack: architecture, SM count, driver, CUDA build, libraries, interconnect.
- Bitwise agreement across GPU generations is not expected; measure before claiming it.
- Kernels whose reduction order is fixed in source are the ones that can carry over.
- Qualify new hardware with the suite in full mode and the SDC scan, then record a baseline.

## Go deeper

- [Measured coverage](../../suite/COVERAGE.md): the stacks measured, result files, baselines, the cross-GPU comparison.
- [Regression](regression.md): running and reading the suite.
- [SDC scan](../1-pretraining/sdc.md): the cross-machine fingerprint scan.
