# Silent data corruption

## Where we are

This page serves step 5 (classify) of the [pre-training workflow](README.md),
and also the time before a long run starts. The pages before it covered causes
in software: the [training step](training-step.md), the
[collectives](data-parallel.md) and [compiled code](capture-compiled-code.md).
You reach this page once the software is ruled out: a self-replay differs
although no setting and no code differs, one rank disagrees with the others,
or a node comes back from repair. The open question is whether the device
itself returns wrong values without reporting an error. That is silent data
corruption (SDC).

## The idea: a faulty device gives different bits for the same inputs, the same kernel and the same stack

**Definition.** A device is faulty if it gives different bits for identical
inputs and an identical deterministic kernel, on the same software stack.
Every part of that sentence matters: an operator that is not deterministic,
or two devices with different drivers, give different bits without any fault
at all.

So a scan has two jobs. It must run only work whose bits are fixed, and it
must summarize the outputs in a way that lets any corruption show. Part 0's
XOR fingerprint catches any single flipped bit, but a scan needs more,
because a fault can flip the same bit twice. Here is a small example over the
32-bit words of a tensor:

| Fault | `xor32` | `sum64` (the words summed mod 2^64) | `wsum64` (the sum of (2i+1)·w_i) |
|---|---|---|---|
| one bit flipped | changes | changes | changes |
| the same bit flipped in two words, both from 0 to 1 | unchanged | changes | changes |
| the same bit flipped in two words, one from 0 to 1 and one from 1 to 0 | unchanged | unchanged | changes |
| two words swapped | unchanged | unchanged | changes |

**Rule.** Judge only deterministic work, compare fingerprints only within a
group of identical stacks, and use several fingerprints, so that no simple
pattern of faults can cancel out. When devices disagree, suspect the software
first: a disagreement is a fact to explain, not yet a verdict.

## The step

### 1. Scan the devices

[sdc_scan.py](../../tools/sdc_scan.py) gives a pass or a fail for each
device, naming the operator and the repeat that failed. It runs a fixed, seeded golden
workload in one worker per GPU and compares the fingerprints on the device.
The workload covers GEMMs, attention, norms, softmax, elementwise math,
gathers and scatters, sorts, integer operations, memory patterns, long
reductions, an optimizer step, a small training step, and NCCL collectives
between every pair of GPUs.

```
python tools/sdc_scan.py --quick --json out.json        # few repeats + fault-injection self-test
python tools/sdc_scan.py --json out.json                # more repeats, NCCL pairs, self-test
python tools/sdc_scan.py --phases stress --stress SECONDS [--offsets] --part s.json
python tools/sdc_scan.py --merge a.json b.json --json fleet.json --record-golden g.json
python tools/sdc_scan.py --golden g.json ...            # compare with a recorded run
python tools/sdc_scan.py --list-ops
```

| Mode | What it compares | What it catches | What it cannot catch | Cost class |
|---|---|---|---|---|
| self-replay | each repeat with the first one, on one device | transient faults and races | a fault that is present in every repeat | small |
| known answer | exact operators (no rounding) against the CPU | persistent faults in operators that do not round; it also says which side is right | operators that round | included |
| fleet | each device against the majority of its group (which needs three devices or more) | persistent faults in any operator | a fault that the whole group shares | included |
| golden file | each device against fingerprints recorded earlier for its group | persistent faults, even on a single device | faults that were already present when the file was recorded | included |
| stress | the same comparisons, for a set time under load and at changing addresses, plus a sweep of the memory | faults that need heat, power draw or particular addresses to appear | faults that appear only under the user's own kernels and load | moderate |
| NCCL pairs | every pair of GPUs, on both ranks, against the CPU's answer | faults in the links and copy engines on the paths tested | other topologies and protocols | small |

A **group** is a set of devices with the same GPU model, SM count, memory
size, driver, torch, CUDA build, cuDNN, cuBLAS, Triton and NCCL. The fleet and
golden-file verdicts never compare across groups.

A clean scan means every device passes, the fault-injection self-test names
every injected fault and nothing else, and the exit code is zero.

### 2. Decide when to scan

| When | What to run | Cost class |
|---|---|---|
| Before a long run, on every node | a full scan, compared with the group's golden file | small |
| After a hardware change (a new card, a reseated card) or a driver or library upgrade | a full scan plus one stress piece; after a software change, also record a new golden file, because the group has changed | moderate |
| Periodically, or when a node comes back from repair | `--quick` with `--golden` | small |
| During training | `sdc_guard` on one step in every N, and `cross_rank_check` at the same rate | small when N is large; each guarded step costs one extra forward pass |

### 3. Check for corruption inside the training job

A scan does not test the job's own kernels, shapes and load, so some faults
appear only inside the job. `sdc_guard(fn, *inputs)` returns `fn(*inputs)`, then computes `fn`
again without autograd and with the random state put back, compares the
fingerprints, and raises `SDCMismatch`, naming the output and the words that
differ. `fn` must be deterministic and must not change any state.
`cross_rank_check(fn, *inputs)` broadcasts rank 0's micro-batch, has every
rank compute `fn` on it, and all-gathers the fingerprints of the inputs and
outputs. Two ranks can only tell you that they differ; three or more can name
the one that is wrong. `replica_check(params)` compares the parameters of the
replicas. For an example, see
[guard_demo.py](../../examples/sdc_scan/guard_demo.py). The checks work when
there are no false alarms over healthy steps and a bit flip planted on purpose
is reported at the right element and rank.

### 4. What to do when devices in a group disagree

1. Rule out software: compare the group keys and the software stacks of the
   hosts.
2. Repeat the work on that device. Fingerprints that keep changing point to
   the hardware; fingerprints that stay the same can still mean a software
   difference or a permanent hardware fault.
3. Settle the question. For an exact operator, the CPU's answer says which
   side is right. Otherwise, move the card to another slot, run again, and
   capture the operator's output on both devices to see how the values
   differ.

### Adding an operator to the golden workload

```python
@op("my_op_bf16", "my_class", exact=False, full=False,
    x=N(ROWS, COLS, dtype=BF16), idx=I(ROWS, hi=COLS))
def _my_op(x, idx):
    return {"out": my_kernel(x, idx), "aux": ...}   # tensor or dict of tensors
```

- Inputs are given as specs: `N` for bell-shaped floats, `NZ` for non-zero
  floats, `U` for distinct floats, `I` for integers in a range, `R` for random
  64-bit words, `P` for a permutation, `K` for a mask, `L` for floats with
  widely spread exponents, and `W` for memory test patterns. The bytes of an
  input depend only on the seed and on the input's own name.
- Set `exact=True` only when no output is rounded (integers, copies, single
  IEEE operations); the CPU then provides the known answer.
- On a healthy device, run the operator with many repeats, plus a stress
  piece with `--offsets`. If any repeat differs, pin a deterministic variant,
  or register the operator with `probe=True, det=False, why=...`, which means
  it is recorded but never judged.
- Changing an operator's inputs or outputs changes its fingerprints, so
  record new golden files afterwards.

This was shown with faults injected in software, on two GPU types. The
self-test named every injected fault and nothing else, and every healthy
device passed with no false alarm. No hardware with a real fault was available
([report](../../examples/sdc_scan/REPORT.md#fault-injection-self-test)).

## What can go wrong

!!! warning "A nondeterministic operator among the judged ones"
    Float atomics, a `cumsum` in default mode or an attention backward vary
    even on a healthy device. The scan then raises false alarms, and people
    stop trusting it. Judge only deterministic operators, and record the rest
    as probes.

!!! warning "A memory placement that changes the result"
    Some reductions give one result for inputs aligned to one boundary and
    another result for inputs aligned to a smaller boundary. A fingerprint
    that changes with placement then looks like a fault. Keep the alignment
    fixed, with offsets in multiples of 256 bytes.

!!! warning "Using the CPU as the known answer for an operator that rounds"
    The math libraries of the CPU and the GPU may round functions such as
    `sqrt` differently, and neither of them is broken. The known-answer check
    then fails a healthy device. Use the CPU only for operators that do not
    round.

!!! warning "Comparing across groups"
    Two drivers or library versions may compute different bits for the same
    work. A verdict across groups would call that a fault. Compare only
    within a group; equality across groups is for information only.

!!! warning "Treating a pass as a certificate"
    The scan tests its own operators, shapes and load, not the job's. A fault
    in the job's own kernels can pass the scan. Use the checks inside the job,
    or OpGuard's fingerprints on the job itself.

## Summary

- SDC means different bits from identical inputs, an identical deterministic
  kernel and an identical stack.
- Judge only deterministic work, and use several fingerprints (`xor32`,
  `sum64`, `wsum64`) so that faults cannot cancel out.
- Scan before long runs and after changes, and guard a sample of the steps
  inside the job.
- A disagreement is a fact to explain, starting with the software, before you
  call it SDC.

## Go deeper

- [Report: scanning GPUs for SDC](../../examples/sdc_scan/REPORT.md) lists
  every operator and the excluded ones, and covers the self-test, the results
  per device, the runtime and the cost of the checks inside the job.
- The tool is [sdc_scan.py](../../tools/sdc_scan.py); the example and the
  driver scripts are in [examples/sdc_scan/](../../examples/sdc_scan/).
- [Localize with OpGuard](localize-with-opguard.md) covers a pivot that turns
  out to be a hardware fault.
