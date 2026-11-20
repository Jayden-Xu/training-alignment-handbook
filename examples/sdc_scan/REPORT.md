# Scanning GPUs for silent data corruption: lab report

This is the full record behind the handbook page [SDC](../../handbook/1-pretraining/sdc.md). The page teaches the idea and the step; this report keeps the setup, every configuration, every table, the limits and how the case was built.

## Summary

- [tools/sdc_scan.py](../../tools/sdc_scan.py) runs a fixed, seeded golden
  workload and compares on-device fingerprints: self-replay, fleet within a
  group, golden file, stress and NCCL pairs. It names the device, op and repeat.
- A fault-injection self-test checks that the scan names every injected fault
  and nothing else (9 checks on one device, 10 with 3 or more).
- 17 GPUs measured clean (one A40, 16 A16s). `--quick` takes 26 s on the A40; a
  full scan about 2 min per 8-GPU node.
- In a job, `sdc_guard` costs +37% on a guarded step (about 0.4% when guarding
  1 step in 100); `cross_rank_check` cost 9 ms here.

## How

A device that gives different bits for identical inputs and an identical kernel
is faulty (see [part 1, SDC](../pretraining_e2e/REPORT.md#overview-sdc)). This case turns that
definition into a tool and a schedule:
- a fixed, seeded golden workload;
- fingerprints of every output, compared on the device, never tensors;
- four comparisons: self-replay, fleet, golden file, stress;
- in-job checks for sampled training steps.
### What the scan does

```
python tools/sdc_scan.py --quick --json out.json        # 3 repeats + fault-injection self-test
python tools/sdc_scan.py --json out.json                # 20 repeats, NCCL pairs, self-test
python tools/sdc_scan.py --phases stress --stress 180 [--offsets] --part s.json
python tools/sdc_scan.py --merge a.json b.json --json fleet.json --record-golden g.json
python tools/sdc_scan.py --golden g.json ...            # compare with a recorded run
python tools/sdc_scan.py --list-ops
```

One worker process runs per GPU, in parallel. The exit code is nonzero when a
required check fails. Results use `bi_report` (bi-results-v1), with one verdict
per device in `extra.devices`.

**Inputs.**
- They come from a seeded CPU generator. Floats are built from integers with
  exact steps only, so every host makes the same bytes.
- Every worker checks this: the SHA-256 of its inputs matched on all 17 devices
  of the three hosts (Python 3.10 with numpy 2.0 on gpu-host, Python 3.11 with
  numpy 1.26 on node1 and node0).
- After every pass, the inputs on the device are fingerprinted again and
  compared with the CPU's copy. Corruption of inputs is thus told apart from
  corruption of results.

**Fingerprints.** Each output becomes three numbers over its 32-bit words, all
computed on the device:

| Fingerprint | Catches | Misses |
|---|---|---|
| `xor32` (as in OpGuard) | any odd number of flips of a bit position | two flips of the same bit |
| `sum64`, words summed mod 2^64 | the pair above when both flips go the same way | flips in opposite directions; values moved to other positions |
| `wsum64`, sum of (2i+1)·w_i mod 2^64 | both of those | (not observed) |

The self-test checks each line of this table (below). On every run, the torch
code is also checked against a numpy reference on the CPU and the GPU: 61 cases
across 10 dtypes, odd lengths, and a 64 MB chunk boundary.

**The golden workload.**
- 76 ops: 70 judged and 6 probes. That is 298 fingerprints in full mode and 294
  in quick mode.
- Every op runs under `torch.use_deterministic_algorithms(True, warn_only=False)`,
  with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and TF32 off.

| Class | Ops |
|---|---|
| GEMM | cuBLAS fp32 (1024², odd 1000×777×1333, 4096² full only), bf16, fp16, fp64, bmm, GEMV, transposed (weight-gradient shape), `F.linear` with bias, M = 512/31/1 rows; `mm_one_config` (Triton) |
| attention | SDPA flash bf16 and fp16, memory-efficient fp32, math with a mask; SDPA backward (flash and efficient) in strict deterministic mode; `attention_fixed_blocks` prefill and decode, `attention_fixed_blocks_train` forward and backward |
| norms | LayerNorm fp32 and bf16 with its backward; RMSNorm |
| softmax | softmax, log_softmax, logsumexp (fp32, bf16, 32000 classes); cross-entropy with its gradient |
| exp / log | exp, log, expm1, log1p, tanh, sigmoid, erf, GELU, SiLU, sqrt, rsqrt, sin, cos, pow in 4 dtypes; wide-exponent fp32 |
| exact float | +, −, ×, ÷, min, max, neg; casts between f32, bf16, f16, f64, i32 and i64 |
| gather / scatter | embedding, gather, index_select, advanced indexing; scatter, index_copy and index_put with unique indices; integer scatter_add, index_add, bincount; masked ops; float index_add in deterministic mode; embedding backward |
| sort / top-k | sort over 4M values, row sort, top-k, argmax, stable integer sort, unique, searchsorted |
| cumsum | int64 and int32 cumsum, cummax |
| integer | int64, int32, int16, int8 and uint8 arithmetic, shifts, logic, division with all rounding modes, reductions |
| memory | write a pattern, clone it, copy it to a 40-byte offset, flip, roll, transpose-copy, complement, read back to the CPU; 4 KB, 1 MB, 64 MB and 256 MB (full only); fixed patterns: checkerboards, walking ones and zeros |
| long reductions | sum, mean, norm, variance, dot, logsumexp over 32M fp32 and 4M fp64 values; bf16 and fp16 over 16M; min and max |
| other | Philox random numbers (rand, randn, randint, bernoulli); AdamW fused and foreach; two training steps of a 2-layer LM (bf16 autocast, fused AdamW) |
| NCCL (≥ 2 GPUs) | all-reduce (i32, f32, bf16), all-gather (bf16, i64), broadcast and reduce-scatter (f32); 256, 64K and 16M elements; every pair of GPUs |

**Known answers.** 147 fingerprints come from ops that do not round
(integer ops, gathers, scatters with unique indices, sorts, copies, single IEEE
operations, min and max). Those are also computed on the CPU and must match.
That says which side is right when devices disagree. NCCL results are checked
the same way, against the sum or concatenation computed on the CPU.

**Excluded ops.** The scan must give zero false alarms. Ops that are not
deterministic run as probes: their results are recorded, never judged.

| Probe | Why excluded | Distinct results over the repeats |
|---|---|---|
| `cumsum_fp32`, `cumsum_bf16` (default mode) | torch documents float cumsum on CUDA as nondeterministic; strict mode raises an error | A40: 116 of 116; A16: 20 of 20 |
| `index_add_fp32_default` (index_add and scatter_add with duplicate indices) | float atomics | A40: 116 of 116; A16: 20 of 20 |
| `sdpa_bwd_flash_default` | dQ added with atomics (measured in the [attention backward case](../../handbook/3-kernels/attention.md#backward-dq-dk-dv)) | A40: 90 of 116; A16: up to 4 of 20 |
| `sdpa_bwd_efficient_default` | the same | A40: 1; A16: up to 3 of 20 |
| `embedding_bwd_fp32_default` | named in part 1 as a racy kernel in production | 1 on both (it stays a probe: one run proves nothing) |

The deterministic variants of these ops are in the judged set. "Distinct
results" is the most distinct fingerprints that any one output of the op gave
on any one device.

Two more candidates were removed after measurement:
- **sqrt as a known answer.** torch's CPU `sqrt` differed from the GPU's in
  26,540 of 4.2M fp32 elements and 29,772 of 4.2M fp64 elements on the A40
  host (measured during development; not in the saved results). In the fp32 examples examined, the GPU's result was the one nearer the
  exact root. sqrt stays in the self-replay ops; it is no longer compared with
  the CPU.
- **Memory offsets that are not 256-byte multiples.** See the stress results.

### Scan modes

| Mode | Compares | Catches | Cannot catch |
|---|---|---|---|
| self-replay | repeat i with repeat 1 on one device; also separate processes | transient faults, races | a fault present in every repeat |
| known answer | exact ops with the CPU | persistent faults in non-rounding ops | rounding ops |
| fleet | each device with its group's majority (needs ≥ 3 devices to name one) | persistent faults in any op | a fault shared by the whole group |
| golden file | each device with fingerprints recorded earlier for its group | persistent faults, also on a single device | faults present when the file was recorded |
| stress | the same, for a set time under load, at changing addresses, plus a memory sweep | faults that need heat, power or particular addresses | faults under the user's own kernels and load |
| NCCL pairs | every pair, both ranks, repeats, CPU answer | link and copy-engine faults on the paths tested | other topologies and protocols |

**Groups.** A group is the same GPU model, SM count, memory size, driver,
torch, CUDA build, cuDNN, cuBLAS, Triton and NCCL. Fleet and golden-file
verdicts never cross groups. With `--cross-group-info`, the scan also records,
as information only, whether two groups gave the same fingerprints.

**When devices in a group disagree.** The scan reports the fact and the
procedure, not "SDC":
1. Rule out software: compare group keys and host stacks.
2. Repeat on that device. Changing fingerprints point at hardware; stable ones
   can still be software or a hard fault.
3. Settle it: an exact op's CPU answer says which side is right. Otherwise move
   the card, rerun, and capture the op's output on both devices to see how the
   values differ.

### Recommended schedule

| When | What | Cost measured here |
|---|---|---|
| before a long run, on every node | full scan (self-replay + NCCL pairs + self-test), compared with the group's golden file | ~2 min per 8-GPU node |
| after a hardware change (new card, reseat, driver or library upgrade) | full scan plus one 180 s stress piece; record a new golden file after a software change, because the group changes | ~6 min |
| periodically (for example weekly, or when a node returns from repair) | `--quick` with `--golden` | ~1 min |
| during training | `sdc_guard` on 1 in N steps, and `cross_rank_check` at the same rate | +37% on a guarded step (0.4% at N = 100); 9 ms per cross-rank check here |

When a check fires, follow the procedure above before draining the node.

### How to add an op

```python
@op("my_op_bf16", "my_class", exact=False, full=False,
    x=N(4096, 1024, dtype=BF16), idx=I(4096, hi=1024))
def _my_op(x, idx):
    return {"out": my_kernel(x, idx), "aux": ...}   # tensor or dict of tensors
```

- Inputs are specs:
  - `N` bell-shaped floats, `NZ` non-zero floats, `U` distinct floats;
  - `I` integers in a range, `R` random 64-bit words, `P` a permutation;
  - `K` a mask, `L` wide exponents, `W` memory test patterns.
- An input's bytes depend on the seed and on its own name only.
- `exact=True` only when no output rounds (integers, copies, single IEEE
  operations); the CPU then gives the known answer.
- Run `--ops 'my_op*' --repeats 50` and a stress piece with `--offsets` on a
  healthy device. If any repeat differs, pin a deterministic variant, or
  register it with `probe=True, det=False, why=...`.
- Changing an op's inputs or outputs changes its fingerprints: record new
  golden files.

## Evidence

### Setup, code and results

- Tool: [tools/sdc_scan.py](../../tools/sdc_scan.py).
- Example and drivers: [examples/sdc_scan/](./):
  `guard_demo.py` (in-job checks) and `run_a16_fleet.sh` (two-host fleet).
- Results:
  - [results-a40.json](results-a40.json);
  - [results-a16-fleet.json](results-a16-fleet.json);
  - [results-guard-a40.json](results-guard-a40.json);
  - [results-ranks-a16.json](results-ranks-a16.json).
- Golden files: [golden-a40.json](golden-a40.json) and
  [golden-a16.json](golden-a16.json).
- Setup:
  - gpu-host: one NVIDIA A40 (84 SMs), driver 550.144.03.
  - node1 and node0: 8x NVIDIA A16 each (10 SMs, 15 GB, PCIe). Driver
    535.309.01 on node1 and 535.288.01 on node0.
  - Every host: torch 2.5.1+cu121, CUDA build 12.1, cuDNN 9.1.0, cuBLAS
    12.1.3.1, Triton 3.1.0, NCCL 2.21.5.

### Fault-injection self-test

Faults are injected in software into the outputs or inputs on the device. The
scan must name each one, and nothing else. Measured on the A40 and on all 16
A16s: 9 checks on one device, 10 with 3 or more devices (the fleet check).

| Injected | Reported |
|---|---|
| bit 7 of word 123457 of `gemm_fp32_1024`, repeat 2 | device, op, repeat 2; evidence: 1 differing word, word 123457, XOR `00000080`; recompute equals repeat 1 |
| a flip in repeat 0 of `gemm_bf16_2048` | repeat 0, because every later repeat agrees with the others |
| two flips of bit 12, same direction | `xor32` unchanged; `sum64` and `wsum64` changed |
| two flips of bit 12, opposite directions | only `wsum64` changed |
| two values swapped | only `wsum64` changed |
| a flip in input `gemm_fp16_2048.a`, repeat 2 | the input, the repeat, and the op that read it |
| a flip in every repeat of `layernorm_fp32` on one device | self-replay passes; the golden file names it; with 8 devices (each A16 host), the fleet majority also names it |
| nothing else | 0 unexpected reports |

### Results

**A40 (gpu-host).** Every GPU command ran under `flock /tmp/bi-check-a40.lock`,
in pieces of under 5 minutes.

| Run | Result |
|---|---|
| self-replay, 20 repeats × 298 fingerprints | pass: every repeat equal |
| known answers, inputs, fingerprint code | pass: 147 of 147 exact fingerprints; 3,120 input checks; 61 of 61 cases |
| stress 1, 180 s: load stream, padding, memory sweep | pass: 93 passes; 34.25 GiB swept, held 183 s, verified, complemented, verified; 87% mean utilization, 28–58 °C, up to 295 W |
| stress 2, 180 s, with inputs placed 256–3,328 bytes into their blocks | pass: 116 passes; up to 97 distinct addresses per input; 36.25 GiB swept |
| the 3 runs in separate processes | the same 298 fingerprints |
| self-test | 9 of 9 |
| ECC counters before and after | 0 corrected, 0 uncorrected |
| **verdict** | **pass** |

Stress 1 ran an earlier workload variant in which the large memory op was
512 MiB instead of 256 MiB. Its other fingerprints are the same set.

**16 A16s (node1 and node0).** Driven from node0 under
`/tmp/bi-check-16gpu.lock`. Both hosts ran at the same time.

| Device set | Self-replay, 20 × 298 | Known answers | Fleet (in group) | NCCL pairs | Self-test |
|---|---|---|---|---|---|
| node0 gpu0–7 (group `67cf7a2b`, driver 535.288.01) | pass, all 8 | pass | pass: 8 of 8 equal | pass: 28 pairs × 21 collectives × 5 repeats, both ranks, CPU answer | 10 of 10 |
| node1 gpu0–7 (group `fe3d582d`, driver 535.309.01) | pass, all 8 | pass | pass: 8 of 8 equal | pass (same) | 10 of 10 |

- **Per device:** all 16 A16s pass. Their UUIDs are in the JSON. Volatile and
  aggregate ECC counters read 0 corrected and 0 uncorrected, before and after.
- **Groups:** the two hosts differ in driver, so they form two groups of 8, and
  no verdict compares them.
- **Across the two groups (information only):** all 298 fingerprints were
  equal, so on this workload the two driver versions computed the same bits.

**Runtime.**

| Run | Wall time |
|---|---|
| `--quick` on the A40 (3 repeats, self-test), from the A40 quick suite baseline | 26 s with process start; self-replay 14.7 s + self-test 8.6 s |
| full self-replay, A40, 20 repeats | 37 s (median pass 0.79 s) |
| full, 8 A16s per host in parallel: self-replay + NCCL pairs + self-test | 118–121 s per host (75–79 s + 26 s + 15–16 s); 125 s for both hosts |
| stress piece | 180 s requested, 205 s wall |

Startup of about 10 s per worker is spent generating 2.1 GiB of inputs and
computing the CPU's known answers.

### In-job checks

`sdc_guard(fn, *inputs)` returns `fn(*inputs)`. It then recomputes `fn` without
autograd, with the random-number state put back, and compares fingerprints. On
a mismatch it raises `SDCMismatch`, which names the output and the differing
words. `fn` must be deterministic and must not change state.

Measured on the A40 (`guard_demo.py --part guard`): an LM with d=1024, 8
layers, 134M parameters, bf16 autocast, batch 8 × 512 and 32000 classes.

| | |
|---|---|
| training step (forward, backward, fused AdamW) | 82.9 ms |
| the same step with `sdc_guard` on its forward | 113.7 ms (+37%) |
| forward alone, no grad | 18.9 ms |
| one fingerprint of the logits (250 MiB bf16) | 5.95 ms |
| false alarms over 20 guarded steps | 0 |
| a bit flipped in the recomputed logits | `SDCMismatch`, 1 differing word, the right element |

The cost is one forward plus two fingerprints of the outputs. Guarding 1 step
in 100 costs about 0.4% at this size.

`cross_rank_check(fn, *inputs)` is for data-parallel jobs. Rank 0's micro-batch
is broadcast, every rank computes `fn` on it, and the fingerprints of the
received inputs and of the outputs are all-gathered. Ranks are compared only
within one device group. Two ranks can only say that they differ; three or
more name the outlier. `replica_check(params)` compares the replicas'
parameters the same way.

Measured on 4 A16s (node0, `guard_demo.py --part ranks`):
- healthy ranks agree;
- a bit flipped on rank 3's output names rank 3 and no input;
- the 4 replicas' parameters agree;
- one check costs 8.96 ms for a 4-layer, d=512 LM forward on a 4 × 256
  micro-batch.

## Limits

### What the scan covers and what it does not

- It tests the ops in the golden workload, at its shapes and dtypes, on idle
  devices or under its own load. It does not test the user's kernels, the
  user's shapes, or the user's load and power profile.
- Use the in-job guard, or OpGuard's fingerprints on the job itself, for that.
- A pass is evidence about these paths at the time of the scan, not a
  certificate.
- Not tested:
  - real faulty hardware (none was available; faults were injected in software);
  - inter-node NCCL (the pairs run inside one host);
  - FP8 GEMM;
  - `torch.compile` and CUDA graphs;
  - other GPU models;
  - multi-hour stress.
- The memory sweep covers the free memory left after the workload (34–36 GiB of
  45 GiB on the A40). The A16 runs had no stress phase.

## How it was built

An agent wrote the tool, the example and this page. Several steps took more
than one try:
- **sqrt as a known answer.** The first quick run on the A40 failed its
  known-answer check on `sqrt` alone: the CPU and the GPU disagree, see above.
- **The large memory op.** It was 512 MiB. It was cut to 256 MiB before the
  A16 runs, because 7 live copies would not fit in 15 GB.
- **Stress offsets.** The first version placed inputs 1–13 elements into their
  blocks. The worker aborted (exit −6), most likely from misaligned vector
  loads. 16-byte steps then gave a false alarm: the fp64 column sum
  `x.view(4096, -1).sum(0)` gives one stable result for 32-byte-aligned inputs
  and another for inputs aligned to 16 bytes only. This was measured at 14
  offsets × 3 runs each; fp32 was unaffected. The offsets are now multiples of
  256 bytes. A fingerprint that changes only with placement is a software
  property, not SDC: the scan must hold alignment fixed.
- **The guard demo.** Its first run failed because the model's 134M parameters
  exceeded the generator's 2^25-value source; the weights are now generated in
  chunks.
- **The cross-rank run.** The first launch was killed by its own timeout while
  queued behind another agent's job holding the 16-GPU lock. It then ran
  unchanged.
