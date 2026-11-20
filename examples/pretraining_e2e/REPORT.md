# Localizing pre-training divergences with OpGuard, and the determinism preflight: lab report

This is the full record behind the handbook pages [Localize with OpGuard](../../handbook/1-pretraining/localize-with-opguard.md) and [Determinism and preflight](../../handbook/1-pretraining/determinism-and-preflight.md). The page teaches the idea and the step; this report keeps the setup, every configuration, every table, the limits and how the case was built.

## Summary

- Three planted divergences (a moved norm rounding point, a `warn_only` SDPA
  backward, a resume without AdamW's step counter) were each localized by
  OpGuard with no unaligned event before the first difference, and fixed:
  reruns showed 513 of 513 and 257 of 257 events equal and identical final
  parameters (one A16, a 4-layer LM, 4 steps).
- The first difference was not the cause in two of three. Call-site statuses
  and `suggest` leads pointed at attention's dQ and at the resumed weights.
- Coverage decides what can be localized: with `torch.nn.Linear` no GEMM was
  captured. Check `coverage` first.
- **Cost:** 4 workload launches and 45–49 s per pair. The
  [preflight](../../handbook/1-pretraining/determinism-and-preflight.md) caught two of the three before any
  capture.

## How

Part 1 describes OpGuard's method in prose. This case runs it: a small LM
trains for a few steps, three planted divergences from part 1's root-cause
table are each run as a reference/suspect pair, and each goes through the
handbook's tools and OpGuard unchanged:

1. preflight report of each run and their diff;
2. OpGuard capture of both runs and its comparison;
3. the facts OpGuard reports about the first divergence;
4. the handbook's leads for that operator (`suggest`);
5. the fix, and a rerun that OpGuard reports as equal.
### The three divergences

| Pair | Planted change | Part 1 pattern |
|---|---|---|
| norm | the suspect's RMSNorm multiplies by its weight in fp32 and rounds once; the reference rounds to bf16 before the weight | rounding point (cross-stack, upgrade) |
| backward | the same job launched twice with `use_deterministic_algorithms(True, warn_only=True)` | racy or nondeterministic kernel; missing flag |
| resume | train 2 steps, save, resume to 4; the suspect restores everything but AdamW's step counter | state bookkeeping |

Each pair differs only in an argument: both sides run the same source lines.

## Evidence

### Setup, code and results

Taken through the [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md).
- Code: [examples/pretraining_e2e/pretraining_e2e.py](pretraining_e2e.py),
  the workload in [examples/pretraining_e2e/workload/](workload/),
  and the tools [opguard_bridge.py](../../tools/opguard_bridge.py) and
  [determinism_preflight.py](../../tools/determinism_preflight.py).
- Results: [results-a16.json](results-a16.json).
- Setup: one NVIDIA A16 (sm86, 10 SMs), driver 535.309.01, torch 2.5.1+cu121,
  cuDNN 9.1.0, NCCL 2.21.5. OpGuard at commit 39601f8 from a source checkout
  with its native XOR and call-site extensions built, run through its
  automation protocol v1 and harness 0.1.0. Nothing in OpGuard was changed.
- Workload: 4-layer decoder LM, width 384, 6 heads, sequence 1024, batch 8,
  vocab 4096, bf16 autocast, SDPA (flash backend), AdamW (foreach), dropout
  0.1, 4 steps. One GPU per run.

### What OpGuard reported

`first_divergence` calls the harness's `trace_summary`, `first_diff`,
`inspect_event`, `compare_events` and `trace_neighbors`. Trimmed:

| Pair | Status counts (reference run) | First observed difference | Recorded at that event |
|---|---|---|---|
| norm | 25 ok, 488 diff; 36 suspect events unaligned, all at model.py:61 (`scaled.to(hidden.dtype)`), where the reference recorded no event (`no_corresponding_trace_file`) | `<None.to>` model.py:59, call 9, in `RMSNorm.forward`: `normed = (hidden32 * inv_rms).to(mul_dtype)` | output `[8, 1024, 384]` bfloat16 vs float32; previous ok: `torch.rsqrt` at line 58 |
| backward | 144 ok, 369 diff | `torch._C._linalg.linalg_vector_norm` train_lm.py:134, call 90, in `grad_norms`: `torch.linalg.vector_norm(param.grad)` | input differs in all 4 regions; previous ok: `loss.backward()`. Statuses of the 36 calls of step 1 (one per parameter): `dddddddddddddddddddddddoddddoooooooo` |
| resume | 135 ok, 122 diff | `torch.nn.modules.sparse` (the token embedding) model.py:139, call 131, in `TinyLM.forward` | input (token ids) equal, output differs in all 4 regions; site statuses `od`: equal at step 3, different at step 4 |

In all three pairs no event before the first difference was unaligned
(`unaligned_before` 0).

### From the facts to the cause

| Pair | Preflight diff | `suggest` leads (after preflight) | What it took to confirm |
|---|---|---|---|
| norm | blocking: none (the argument is declared) | rounding point moved (dtype differs at the same call), class row-wise reductions (recipe 1) | the source line names the cast; making the suspect round before the weight gives the reference's bits (fix row below) |
| backward | blocking: none; `at_risk` on both sides: `torch.deterministic_algorithms.warn_only = True` | nondeterministic backward upstream: a gradient consumer whose input differs while the forward events agree | mapping the 36 statuses to the parameters in order: in the last block only `q_proj`'s gradient differs, while its `k_proj`, `v_proj`, `o_proj`, MLP, the final norm and the head agree. That is dQ of attention's backward. A first full run gave a different string (`dddddddddddddddoddodooooooodoooooooo`, the race differs run to run) with the same last-block pattern. A repeat check of SDPA's backward on the same GPU (probe below) showed the same: only dQ varies |
| resume | blocking: `optimizer.state.step` min/max 2.0 vs 0.0 | a lookup with equal indices and a different result: the weights changed before this call | the preflight already named the step counter before any capture; OpGuard placed the first difference at the first forward after the first resumed update |

Probe on the same A16, SDPA causal bf16 forward+backward repeated 6 times on
fixed inputs: with `warn_only=True` dQ differed in 5 or 6 of 6 runs at T ≥ 512
(6 to 807 elements); K and V gradients and the output did not differ; with
`warn_only=False` none differed. At T ≤ 256 no run differed. This was a
separate script, not part of the results file.

### Fixes and reruns

| Fix | OpGuard rerun | Final parameters | Cost (measured here) |
|---|---|---|---|
| norm rounds before the weight, and strict deterministic algorithms (one configuration for both pairs) | 513 of 513 events ok, none unaligned | identical, all 36 tensors | train step 242 ms with strict mode (345 ms in a first full run, not in the saved results), 98 ms with `warn_only=True`, averaged over 4 steps including the first |
| full restore (step counter included) | 257 of 257 events ok, none unaligned | identical | none |

Before any fix, the final parameters of every pair differed in all 36 tensors
(compared with `bi_check.bit_equal`). The captured reference runs ended with
the same parameters as uninstrumented runs of the same configuration, so
capture did not change the numerics here.

Other restore pieces, each left out alone (in process, no OpGuard), and what the
preflight diff showed for each, are in
[determinism and preflight](#restore-pieces-each-left-out-alone).

### Cost of each investigation

| Pair | Workload launches | Wall time of the OpGuard workflow | Facts queries |
|---|---|---|---|
| norm | 4 | 47.2 s | 10.5 s |
| backward | 4 | 45.7 s | 5.5 s |
| resume | 4 | 48.4 s | 2.5 s |
| each fix rerun | 4 | 45.5–48.9 s | 0.3 s |

Discovery and capture launch each side once: 4 launches per pair. Prepare and
compare take 3–4 s each.

### Coverage: what OpGuard did not capture

`coverage()` reads OpGuard's own discovery and plan files.

- **Linear layers through `torch.nn.Linear`.** The GEMM is launched from
  torch's `linear.py`, outside the checkout. The lines that call the
  projections (`query = self.heads_first(self.q_proj(hidden))`) were
  discovered but not planned, so no GEMM was captured (36 lines discovered,
  23 planned). With a Linear that calls `F.linear` in model code, line 36 was
  planned and every GEMM was captured.
- **Lines with one-letter names.** A first version of the model used names
  like `x`, `q`, `h`. OpGuard's instrumenter matches call sites by the names
  around them and ignores one-letter names, so SDPA, the projections and GELU
  were planned but not wrapped. The model was rewritten with longer names.
  This is OpGuard's instrumenter, not the method; the topic page of the next
  workload should check `coverage` first.
- **Optimizer, gradient clipping and the position embedding** (train_lm.py
  lines 209–210, model.py:138) were discovered and not planned.
- **Compiled code and CUDA graphs.** OpGuard captures eager execution. The
  plain-torch fingerprints of the bridge gave eager's bits inside
  `torch.compile` and inside a replayed CUDA graph (`info` checks); the tool
  that captures there is separate.

### Fingerprints and the trace writer

- `xor_signature` and `xor_vectors` in plain torch equal OpGuard's own
  functions bit for bit, on CPU and GPU: bf16, fp16, fp32, fp64, int64, bool,
  uint8, byte counts 462, 1001, 91 and 6, empty, 0-dim, transposed, strided
  and offset views, 1, 4, 7 and 64 regions; and two 1 GiB tensors (fp32
  `[65536, 4096]`, and bf16 of 2^29 + 1 elements).
- On 1 GiB on the A16: plain torch 19.3–19.5 ms, OpGuard's native kernel 6.3–7.3 ms
  (signature and vectors). On CPU, for the signature: plain torch 143–161 ms,
  OpGuard's NumPy path 372–517 ms (bf16 and fp32 respectively); for the fp32
  vectors, 135 and 239 ms.
- `TraceWriter` wrote five boundary records per run, one with one flipped bit.
  OpGuard's backend `compare` exported the pair; the harness tools reported
  4 ok and 1 diff, the planted record (`<layer1.op>`, call 4), output region
  2 only, no unaligned event. The writer needs nothing from OpGuard's
  instrumenter.

### What this shows

1. **OpGuard localized all three.** Each first difference was at the planted
   change (norm), at the first consumer of the affected gradients (backward),
   or at the first use of the affected weights (resume), with no unaligned
   event before it.
2. **The preflight caught two of three before any capture.** `warn_only` and
   the missing step counter were visible in the reports. The rounding point
   was not: it is a code difference, not a setting.
3. **The first difference is not the cause in two of three.** For backward,
   the pivot is a gradient norm; the cause is attention's backward, which has
   no boundary record. The site status string, not the pivot alone, pointed at
   it. For resume, the pivot is an embedding lookup with equal inputs.
4. **Coverage decides what can be localized.** With `torch.nn.Linear` no GEMM
   is a boundary. Check `coverage` before reading a first difference.

## Limits

- One GPU type, one rank, 4 steps, one model size. No multi-rank run, no
  collective, no SDC.
- The planted causes are known: this is a known-answer walkthrough, not a
  blind evaluation. `suggest` was written knowing the three cases.
- The scheduler, RNG and data-position faults were run in process only, not
  through OpGuard.
- The preflight sees the loader's generator state, not how many batches were
  consumed after it: a fault that restores the generator and skips the wrong
  number of batches would not show in it.
- Untested: fused AdamW against foreach, multi-rank captures, CUPTI and
  CUDA-API call-site backends (off in all runs).

## How it was built

An agent wrote the tools, the workload and the script. What took more than
one try:
- The first workload used one-letter names; OpGuard wrapped 15 of 20 planned
  call sites and none of attention, projections or GELU. Renaming fixed it.
- `torch.nn.Linear` hid every GEMM from the capture; a Linear calling
  `F.linear` in model code is the default here.
- The first full run flagged the numpy RNG as a blocking preflight difference
  in every pair: the workload never seeded numpy. It now does.
- `first_divergence` first missed the suspect side's unaligned call sites; it
  now walks both sides.


## The determinism preflight

Measured in the same workload and pairs as the OpGuard run above.

### Preflight: Summary

- `determinism_preflight.report(...)` reads every determinism setting and
  piece of resume state back from inside the job, per rank, as JSON.
  `diff(a, b)` lists what differs between two runs and which differences make a
  bitwise comparison meaningless.
- In the [OpGuard run](../../handbook/1-pretraining/localize-with-opguard.md) it caught two of three planted
  divergences before any capture (`warn_only=True`, a missing optimizer step
  counter). A moved rounding point did not show: it is code, not a setting.
- Each resume piece left out alone (step counter, scheduler, CUDA RNG, data
  position) showed in its own field of the diff.
- **Cost of the fix it most often points at:** strict deterministic mode,
  242 ms against 98 ms per step with `warn_only=True` in that run.

### Preflight: How

Set the five control surfaces of [step 2](#overview-step-2-control-the-avoidable-nondeterminism)
(randomness and data order, library kernels, collectives, checkpoint resume,
TP degree), then check them:

**Check, then diff.** `determinism_preflight.report(...)` reads these surfaces
back from inside the job, per rank, as JSON: the deterministic flag and its
`warn_only`, cuDNN, TF32 and matmul precision, `CUBLAS_WORKSPACE_CONFIG`, SDPA
backends, `NCCL_*` and the NCCL version, RNG states, the dataloader's workers
and generator, the optimizer's path (fused, foreach) and state, scheduler
counters, AMP, compile and CUDA-graph use, GPU, driver and library versions.
Take it after the restore, before the first step. `findings` lists what one run
leaves uncontrolled; `diff(a, b)` lists what differs and which differences make
a bitwise comparison meaningless. When OpGuard is installed its `doctor` output
is embedded, not repeated.

Which settings and pieces of state matter for a whole training step, and what
fixing each costs, is measured on the [training step](../../handbook/1-pretraining/training-step.md) page
(its checklist: [what to pin](../training_step/REPORT.md#what-to-pin-so-two-trainers-agree));
for gradient collectives, on the [data parallel](../../handbook/1-pretraining/data-parallel.md) page.

### Preflight: Evidence

Code: [tools/determinism_preflight.py](../../tools/determinism_preflight.py);
the workload and pairs are those of [localize with OpGuard](../../handbook/1-pretraining/localize-with-opguard.md)
([examples/pretraining_e2e/](./), results
[results-a16.json](results-a16.json)).

#### What the preflight caught

Measured in the [OpGuard run](../../handbook/1-pretraining/localize-with-opguard.md) (one A16, a 4-layer LM, 4 steps):
- `warn_only=True` was flagged before any capture, and it left SDPA's backward
  nondeterministic (dQ only), as in the [attention backward case](../../handbook/3-kernels/attention.md#backward-dq-dk-dv).
  Strict mode cost 242 ms against 98 ms per step there (the saved results; a
  first full run gave 345 ms, which is not in the saved results).
- A resume without AdamW's step counter showed as `optimizer.state.step`
  2.0 against 0.0; without the scheduler, CUDA RNG or loader state, each showed
  in its own field.
- A moved rounding point did not show: it is code, not a setting.

#### Restore pieces, each left out alone

Each piece left out alone on resume, in process (no OpGuard), in the same workload:

| Left out | Final parameters | Preflight diff after restore |
|---|---|---|
| optimizer step counter | differ | `optimizer.state.step` |
| scheduler state | differ | `scheduler.*` |
| CUDA RNG state | differ (dropout) | `seeds.cuda_rng_state_sha256` |
| data position | differ | `dataloader.generator.state_sha256` (the epoch's shuffle state) |
| CPU RNG state | identical | none: this workload draws nothing from the CPU RNG during training, so the state at the restore point equals the saved one |

The 20-step version of this test, with every piece of state and the step at
which leaving it out shows, is the [resume section of the training step page](../training_step/REPORT.md#9-resume).

### Preflight: Limits

- Settings only: a difference in code (a moved rounding point, a different
  kernel) is not visible to the preflight.
- The preflight sees the loader's generator state, not how many batches were
  consumed after it (see [limits](#limits)).
- Measured on one A16, one rank, 4 steps.

### Preflight: How it was built

The preflight was written with the [OpGuard run](#how-it-was-built).
Its first full run flagged the numpy RNG as a blocking difference in every pair,
because the workload never seeded numpy; the workload now does.

## Overview: the earlier part 1 page

The earlier version of [part 1's overview](../../handbook/1-pretraining/README.md) quoted measurements and the OpGuard paper's production numbers on the page. It is kept here in full, so that no statement is lost.

Typical triggers:
- a loss spike or slow drift after an upgrade, migration or config change;
- a run that does not resume to the same trajectory;
- a machine suspected of silent data corruption (SDC).

The method: capture one suspect run and one reference run, control only the
avoidable nondeterminism, fingerprint model-level operator boundaries, take the
first bitwise difference, and walk outward from it to the cause. It is the one
OpGuard takes; its production numbers come from the OSDI '26 paper.

The agent workflow that applies this part is the
[pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md). It
drives OpGuard unchanged through [tools/opguard_bridge.py](../../tools/opguard_bridge.py)
and checks settings with [tools/determinism_preflight.py](../../tools/determinism_preflight.py).

### Overview: Stages

In the order a run is set up and debugged:

| Stage | Page | What it settles |
|---|---|---|
| 1. Control and check the settings (steps 1–2) | [Determinism and preflight](../../handbook/1-pretraining/determinism-and-preflight.md) | Which settings and state one run leaves uncontrolled, and what differs between two runs, read back from inside the job |
| 2. Pin the training step (step 2) | [Training step](../../handbook/1-pretraining/training-step.md) | Which backward ops, optimizer paths, loss reductions, AMP, accumulation and resume state change bits; fixes at 1.03–1.18x |
| 3. Pin data parallelism (step 2) | [Data parallel](../../handbook/1-pretraining/data-parallel.md) | DDP buckets, FSDP-style (emulated) reduce-scatter, logical shards that give the same weights at DP 1–16 |
| 4. Capture compiled code (step 3) | [Capture compiled code](../../handbook/1-pretraining/capture-compiled-code.md) | Fingerprints inside `torch.compile` and CUDA graphs, in OpGuard's format |
| 5. Localize the first divergence (steps 3–5) | [Localize with OpGuard](../../handbook/1-pretraining/localize-with-opguard.md) | Three planted divergences taken from capture to fix (one A16, a 4-layer LM, 4 steps) |
| 6. Rule out the hardware | [SDC](../../handbook/1-pretraining/sdc.md) | A golden-workload scan per device, and in-job checks |

When the first difference lands in an operator that has no batch-invariant or
deterministic implementation, go to [part 3](../../handbook/3-kernels/README.md).

### Overview: Step 1: pick a reference run

One captured suspect run and one reference run are enough. The suspect need not
be deterministic. Choose the reference that isolates your question:

| Reference | Answers |
|---|---|
| Self-replay (same everything) | Is there residual nondeterminism, a race or SDC? |
| Last stable commit | Which change introduced the drift? |
| Config tweak (one flag off: overlap, offload, recompute, fusion) | Does this feature change the computation? |
| Library or driver version | Did the upgrade change the numerics? |
| Cross-stack (other framework, same model) | Where do two implementations disagree? |
| Single GPU, or TP-simulator at logical TP > 1 | Is the parallel implementation wrong? |

Both runs use the same data, seeds and checkpoint.

Write down the one difference under test. The preflight diff (step 2) takes it
as `declared`, so it is listed without blocking the comparison.

### Overview: Step 2: control the avoidable nondeterminism

Five surfaces, all lightweight; none disables core optimizations:

1. **Randomness and data order.** Fix CPU/CUDA RNG seeds, dropout streams,
   dataloader workers, sampler state and distributed initialization.
2. **Library kernels.** Set `torch.use_deterministic_algorithms(True)`, the
   deterministic cuDNN mode, no autotuning, TF32 off and
   `CUBLAS_WORKSPACE_CONFIG=:16:8`. Use the same toolkit, GPU architecture and SM
   count, and select deterministic variants of fused attention and MoE kernels.
3. **Collectives.** Pin the algorithm, protocol, topology and bucketization, so
   partial sums follow the same tree. A different tree is a different
   configuration, not a fault.
4. **Checkpoint resume.** Restore RNG streams, optimizer state, the dataloader
   position and scheduler counters on every rank.
5. **Across TP degrees.** Use a TP-simulator mode: logical TP stays above 1 while
   every rank runs the unsharded arithmetic, so TP=N matches TP=1 bitwise.

Do **not** fix scheduling or user and third-party kernels. If they change the
bits, that is the bug signal.

**Check, then diff.** `determinism_preflight.report(...)` reads these surfaces
back from inside the job, per rank, as JSON, and `diff(a, b)` lists what
differs between two runs and which differences make a bitwise comparison
meaningless. What it reads, and what it caught and missed in a measured run, is
in [determinism and preflight](../../handbook/1-pretraining/determinism-and-preflight.md).

Which settings and pieces of state matter for a full training step, and what
each fix costs, is measured on two pages:
- [training step](../../handbook/1-pretraining/training-step.md) (one A40): backward ops, optimizer
  path, loss reduction, AMP, accumulation, and the 7 pieces of state a bitwise
  resume needs; strict mode costs 1.44x, targeted fixes 1.03–1.18x;
- [DP and FSDP-style (emulated: flat gradient, reduce-scatter, sharded AdamW) gradients](../../handbook/1-pretraining/data-parallel.md) (2x8 A16): bucket layout
  under Ring, DDP's first-iteration bucket, reduce-scatter order, and logical
  shards that give the same weights at DP 1–16.

### Overview: Step 3: observe model-level boundaries

Find, with a short preflight run on each side, the calls in model code that
launch device work, and fingerprint their inputs and outputs. Compute
fingerprints on the stream the op ran on, with no added synchronization, and
write them out in the background.

`opguard_bridge.run_pair(reference, suspect, ...)` runs both sides through
OpGuard's harness: discovery, instrumentation plan, capture, comparison. That is
four workload launches; 45–49 s per pair in the [measured run](../../handbook/1-pretraining/localize-with-opguard.md). Its fingerprint formats
(`flat-u32-le-v1` regions, `lastdim-u32le-v1` vectors) are also in the bridge in
plain torch, bit-identical to OpGuard's on CPU and GPU, for tools that run
where OpGuard is not installed.

Eager OpGuard captures do not show compiled bits. For a run under
`torch.compile` or CUDA graphs, [tools/graph_probe.py](../../tools/graph_probe.py)
records fingerprints in OpGuard's formats inside compiled code and graph
replays; OpGuard's harness then compares them as usual. In the
[compile and CUDA-graph page](../../handbook/1-pretraining/capture-compiled-code.md) it found the same
first divergence as a direct comparison in 19 of 19 pairs. Two cautions from
that page: give the probe an exact pointwise copy (`isolate=True`), because a
naive probe changed the compiled bits; and build lazy caches such as RoPE
tables in eager before compiling.

Coverage matters. Cutting instrumentation down to 40 hand-placed probes left
only 2 of 20 production bugs exactly localized. Check what a capture covers
with `opguard_bridge.coverage(run)` before reading a first difference.
Measured: with `torch.nn.Linear` the GEMMs are launched from torch's own file
and no projection was captured; with a Linear that calls `F.linear` in model
code every GEMM was. One-letter variable names in model code also blocked
wrapping.

Choose the operator, not a trailing dtype cast. On a line like
`q = q_norm(x).float()` the boundary is `q_norm`.

### Overview: Step 4: align and take the first divergence

The two traces never line up position by position:
- stream overlap, fusion and graph rewrites move events locally;
- one side has extra events.

Pair them with order-preserving matching anchored on unambiguous boundaries.
Leave uncertain events as gaps rather than forcing a pairing. Then walk the
pairs in order: the first pair whose fingerprints differ is the pivot. An
unpaired event is not a difference.

`opguard_bridge.first_divergence(export, ...)` collects what OpGuard's harness
reports about the pivot: operator, line, call number, which fields differ,
inputs and outputs on both sides, differing regions, unaligned events before
it, and the status of every call at its call site. It judges nothing.

### Overview: Step 5: walk outward from the pivot

The first difference is where debugging starts, not the cause:
- read the operator's source and its producers;
- walk downstream to connect it to the symptom;
- test the hypothesis with the smallest intervention.

OpGuard's harness skills do this: `investigate-first-diff` owns the boundary and
the evidence record, `trace-upstream` follows producers when inputs already
differ, `validate-hypothesis` tests one cause. `opguard_bridge.suggest(facts)`
maps the pivot to this handbook's operator classes and
[recipes](../../skills/batch-invariant-kernel/references/operator-recipes.md),
as leads with reasons; a kernel fix goes through the
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md).
Confirm the fix by capturing the pair again: no event differs, none is
unaligned, final parameters are bit-identical.

In the [measured run](../../handbook/1-pretraining/localize-with-opguard.md), OpGuard placed all three first differences with no unaligned event
before them, and two of the three were not the cause. The rounding point was the
pivot itself (`.to(mul_dtype)` in the norm: bf16 against fp32). The
nondeterministic backward appeared first at a gradient norm; the call-site
statuses showed only `q_proj`'s gradient differing in the last block, which is
attention's dQ. The resume fault appeared first at an embedding lookup with equal
token ids, at the first forward after the first resumed update.

Root causes OpGuard pinpointed in production:

| Pattern | Examples |
|---|---|
| Racy or nondeterministic kernel | `torch.embedding` backward, fused embedding backward, a Triton kernel missing a barrier |
| Stream misuse | an FP32 RMSNorm stream race, activation offload |
| Missing flag | deterministic mode unset for one backward kernel |
| Topology-dependent reduction | reduce-scatter under a different NCCL topology |
| Layout assumption | QKV projections relying on implicit contiguity |
| State bookkeeping | an optimizer step variable overwritten on resume |
| MoE routing | a histogram bin mismatch causing routing drift |
| Hardware SDC | a boolean scatter misrouting one token, found as a shape mismatch 30 ops later |

Engineers estimated 2–14 days of manual debugging for these cases, and 5–30
minutes with OpGuard.

### Overview: SDC

The same fingerprints catch faulty hardware. A device that produces different
bits from identical inputs and kernel is SDC. OpGuard flagged 21 machines that
had all passed vendor pre-flight diagnostics. The first-difference operator names
the computation or communication that exposes the fault, which vendor tools do
not.

To check a device before or between runs, run
[tools/sdc_scan.py](../../tools/sdc_scan.py). It runs a fixed, seeded workload,
compares fingerprints across repeats, across devices of the same group (GPU,
driver and library versions) and with a recorded golden file, and names the
device, op and repeat. Inside a job, `sdc_guard` recomputes sampled steps and
`cross_rank_check` makes data-parallel ranks compute the same micro-batch. A
disagreement between devices is a fact to explain, starting with software
differences, before it is called SDC. See the [SDC page](../../handbook/1-pretraining/sdc.md).

### Overview: Checklist

- [ ] The reference run is chosen for the question; data, seeds and checkpoint
      are identical.
- [ ] The five control surfaces are set and recorded (preflight reports of
      both runs, and their diff with only the declared difference).
- [ ] The capture covers the region in question (`coverage`).
- [ ] Boundaries come from preflight, on the operator rather than a trailing cast.
- [ ] Comparison is bitwise; unpaired events stay gaps.
- [ ] The first difference is followed upstream in source and downstream to the
      symptom, then tested.

## Overview: tolerances against bitwise comparison

From the earlier version of [part 0](../../handbook/0-foundations/README.md):

Tolerances do not answer it. A threshold that hides harmless reduction-order
noise also hides a few corrupted elements. In OpGuard's evaluation on 20
production bugs, switching from controlled bitwise comparison to tolerances of
1e-5 and 1e-3 dropped the bugs localized exactly from 20 to 10 and to 7.
