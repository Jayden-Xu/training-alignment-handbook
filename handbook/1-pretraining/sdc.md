# Scanning GPUs for silent data corruption

## Where we are

This page serves step 5 (classify) of the [pre-training workflow](README.md),
and the time before a long run. The pages before it covered causes in software:
the [training step](training-step.md), the [collectives](data-parallel.md) and
[compiled code](capture-compiled-code.md). You reach this one when the software
is ruled out: a self-replay differs with no setting or code difference, one
rank disagrees with the others, or a node returns from repair. What is still open: is the device itself returning
wrong values without an error? That is silent data corruption (SDC).

## The idea: same inputs, same kernel, same stack, different bits

**Definition.** A device that gives different bits for identical inputs and an
identical deterministic kernel, on the same software stack, is faulty. Every
part of that sentence matters: an op that is not deterministic, or two devices
with different drivers, give different bits with no fault at all.

So a scan has two jobs: run only work whose bits are fixed, and summarize the
outputs so that any corruption shows. Part 0's XOR fingerprint catches any
single flip; a scan wants more, because a fault can flip the same bit twice.
A tiny example over a tensor's 32-bit words:

| Fault | `xor32` | `sum64` (words summed mod 2^64) | `wsum64` (sum of (2i+1)·w_i) |
|---|---|---|---|
| one bit flipped | changes | changes | changes |
| the same bit flipped in two words, both 0→1 | unchanged | changes | changes |
| the same bit flipped in two words, one 0→1, one 1→0 | unchanged | unchanged | changes |
| two words swapped | unchanged | unchanged | changes |

**Rule.** Judge only deterministic work, compare fingerprints within a group of
identical stacks, and use several fingerprints so that no simple fault pattern
cancels. When devices disagree, suspect software first: it is a fact to
explain, not yet a verdict.

## The step

### 1. Scan the devices

**Goal:** each device named pass or fail, with the op and the repeat.

**How:** [sdc_scan.py](../../tools/sdc_scan.py) runs a fixed, seeded golden
workload (GEMMs, attention, norms, softmax, elementwise math, gathers and
scatters, sorts, integer ops, memory patterns, long reductions, an optimizer
step, a small training step, and NCCL collectives between every pair of GPUs)
in one worker per GPU, and compares fingerprints on the device.

```
python tools/sdc_scan.py --quick --json out.json        # few repeats + fault-injection self-test
python tools/sdc_scan.py --json out.json                # more repeats, NCCL pairs, self-test
python tools/sdc_scan.py --phases stress --stress SECONDS [--offsets] --part s.json
python tools/sdc_scan.py --merge a.json b.json --json fleet.json --record-golden g.json
python tools/sdc_scan.py --golden g.json ...            # compare with a recorded run
python tools/sdc_scan.py --list-ops
```

| Mode | Compares | Catches | Cannot catch | Cost class |
|---|---|---|---|---|
| self-replay | each repeat with the first, on one device | transient faults, races | a fault present in every repeat | small |
| known answer | exact ops (no rounding) with the CPU | persistent faults in non-rounding ops; says which side is right | rounding ops | included |
| fleet | each device with its group's majority (needs three or more) | persistent faults in any op | a fault shared by the whole group | included |
| golden file | each device with fingerprints recorded earlier for its group | persistent faults, also on a single device | faults present when the file was recorded | included |
| stress | the same, for a set time under load, at changing addresses, plus a memory sweep | faults that need heat, power or particular addresses | faults under the user's own kernels and load | moderate |
| NCCL pairs | every pair of GPUs, both ranks, against the CPU's answer | link and copy-engine faults on the paths tested | other topologies and protocols | small |

A **group** is the same GPU model, SM count, memory size, driver, torch, CUDA
build, cuDNN, cuBLAS, Triton and NCCL. Fleet and golden-file verdicts never
cross groups.

**Success:** every device passes, the fault-injection self-test names every
injected fault and nothing else, and the exit code is zero.

### 2. Schedule it

| When | What | Cost class |
|---|---|---|
| Before a long run, on every node | full scan, compared with the group's golden file | small |
| After a hardware change (new card, reseat) or a driver or library upgrade | full scan plus one stress piece; after a software change record a new golden file, because the group changed | moderate |
| Periodically, or when a node returns from repair | `--quick` with `--golden` | small |
| During training | `sdc_guard` on one step in N, and `cross_rank_check` at the same rate | small when N is large; one extra forward per guarded step |

### 3. Check inside the job

**Goal:** catch faults under the job's own kernels, shapes and load, which a
scan does not test.

**How:** `sdc_guard(fn, *inputs)` returns `fn(*inputs)`, then recomputes `fn`
without autograd with the random state put back, compares fingerprints, and
raises `SDCMismatch` naming the output and the differing words; `fn` must be
deterministic and must not change state. `cross_rank_check(fn, *inputs)`
broadcasts rank 0's micro-batch, has every rank compute `fn` on it, and
all-gathers the fingerprints of inputs and outputs; two ranks can only say they
differ, three or more name the outlier. `replica_check(params)` compares the
replicas' parameters. Example: [guard_demo.py](../../examples/sdc_scan/guard_demo.py).

**Success:** no false alarms over healthy steps, and a planted bit flip is
named at the right element and rank.

### 4. When devices in a group disagree

1. Rule out software: compare group keys and host stacks.
2. Repeat on that device. Changing fingerprints point at hardware; stable ones
   can still be software or a hard fault.
3. Settle it: an exact op's CPU answer says which side is right. Otherwise move
   the card, rerun, and capture the op's output on both devices to see how the
   values differ.

### Adding an op to the golden workload

```python
@op("my_op_bf16", "my_class", exact=False, full=False,
    x=N(ROWS, COLS, dtype=BF16), idx=I(ROWS, hi=COLS))
def _my_op(x, idx):
    return {"out": my_kernel(x, idx), "aux": ...}   # tensor or dict of tensors
```

- Inputs are specs: `N` bell-shaped floats, `NZ` non-zero floats, `U` distinct
  floats, `I` integers in a range, `R` random 64-bit words, `P` a permutation,
  `K` a mask, `L` wide exponents, `W` memory test patterns. An input's bytes
  depend only on the seed and its own name.
- `exact=True` only when no output rounds (integers, copies, single IEEE
  operations); the CPU then gives the known answer.
- Run the op with many repeats and a stress piece with `--offsets` on a healthy
  device. If any repeat differs, pin a deterministic variant, or register it
  with `probe=True, det=False, why=...`: recorded, never judged.
- Changing an op's inputs or outputs changes its fingerprints: record new
  golden files.

Shown with software fault injection on two GPU types: the self-test named every
injected fault and nothing else, and every healthy device passed with no false
alarm. No real faulty hardware was available
([report](../../examples/sdc_scan/REPORT.md#fault-injection-self-test)).

## What can go wrong

!!! warning "A nondeterministic op in the judged set"
    Float atomics, a default-mode `cumsum` or attention backward vary on a
    healthy device. The scan raises false alarms and loses trust. Judge only
    deterministic ops; record the rest as probes.

!!! warning "Placement that changes the result"
    Some reductions give one result for inputs aligned to one boundary and
    another for inputs aligned to a smaller one. A fingerprint that changes
    with placement looks like a fault. Hold alignment fixed: offsets in
    multiples of 256 bytes.

!!! warning "The CPU taken as the known answer for a rounding op"
    CPU and GPU math libraries can round functions like `sqrt` differently,
    and neither is broken. The known-answer check then fails a healthy device.
    Use the CPU only for ops that do not round.

!!! warning "Comparing across groups"
    Two drivers or library versions may compute different bits on the same
    work. A cross-group verdict would call that a fault. Compare within a
    group; cross-group equality is information only.

!!! warning "A pass taken as a certificate"
    The scan tests its own ops, shapes and load, not the job's. A fault in the
    job's kernels can pass. Use the in-job checks, or OpGuard's fingerprints on
    the job itself.

## Summary

- SDC is different bits from identical inputs, an identical deterministic
  kernel and an identical stack.
- Judge only deterministic work; use several fingerprints (`xor32`, `sum64`,
  `wsum64`) so faults cannot cancel.
- Scan before long runs and after changes; guard sampled steps inside the job.
- A disagreement is a fact to explain, starting with software, before it is
  called SDC.

## Go deeper

- [Report: scanning GPUs for SDC](../../examples/sdc_scan/REPORT.md): the full
  op list, the excluded ops, the self-test, results per device, runtime and the
  in-job checks' cost.
- Tool: [sdc_scan.py](../../tools/sdc_scan.py); example and drivers:
  [examples/sdc_scan/](../../examples/sdc_scan/).
- [Localize with OpGuard](localize-with-opguard.md): a pivot that ends in a
  hardware fault.
