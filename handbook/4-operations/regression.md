# Regression checks

You have a stack on which the handbook's checks hold, and something in it is
about to change: torch, Triton, NCCL, the driver, the CUDA build or the GPUs.
This page shows how to re-run every verified check on the new stack. The
report tells you which claims still hold, which broke, and which stock
behaviour merely moved.

## The idea

Take a stock matrix multiply that starts to round a row differently once the
batch grows past some size. That size is a property of one library version on
one GPU. The library picks its kernel from the shape, and another version may
switch kernels at another size, or use the same kernel everywhere. So after
an upgrade, two kinds of change look alike but mean opposite things:

| What changed | Example | What it means |
|---|---|---|
| A check that the handbook claims now fails | the handbook's invariant GEMM gives a row different bits in a larger batch | a finding: the claim no longer holds on this stack |
| A documented stock failure now passes, or fails somewhere else | the stock GEMM starts to vary at another batch size, or not at all | the stack changed underneath it; update the topic page |

The suite keeps the two apart in two ways. It marks each check either as a
claim (`expect="pass"`) or as a documented stock failure (`expect="fail"`),
and it compares the verdicts with a baseline recorded for this exact stack.
Inside each check, equality is bitwise; the suite never judges a number by
how close it is.

## The step

- Runner: [tools/bi_suite.py](../../tools/bi_suite.py)
- Entries, the list of checks to run: [suite/manifest.json](../../suite/manifest.json)
- Baselines: [suite/baselines/](../../suite/baselines/)
- Adapters for the result formats of the first examples: [suite/adapters/](../../suite/adapters/)
- CI: [.github/workflows/bi-suite.yml](../../.github/workflows/bi-suite.yml), **untested** (see [limits](#limits))

### When to run it

- Before and after you upgrade torch, Triton, NCCL, the driver or the CUDA
  build.
- On new hardware ([see below](#new-hardware)).
- Before you claim parity in a release.
- After you edit an example, a kernel in `tools/` or an adapter.

### How to run it

Run every check that fits this machine and compare the results. The run is
clean when the runner exits with 0 and the report has no REGRESSIONS or NO
RESULT lines.

```bash
python tools/bi_suite.py --list                       # what runs here, what is skipped and why
python tools/bi_suite.py                              # quick mode
python tools/bi_suite.py --full                       # the sizes behind the topic pages
python tools/bi_suite.py --vs-committed               # also compare with the committed result files
python tools/bi_suite.py --lock-file /tmp/gpu.lock    # shared GPUs: one flock per entry
python tools/bi_suite.py --peer node1 --launcher2 ~/launch2.sh   # also the two-node entries
python tools/bi_suite.py --opguard-root ~/OpGuard     # also the entries that need an OpGuard checkout
```

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| quick mode | every claim and the main stock failures, at reduced sizes | small | every upgrade, every edit |
| `--full` | the sizes the topic pages were written from | large | before a release, on new hardware |
| `--vs-committed` | also a comparison with the result files behind the topic pages | free | when you want to see what has moved since the pages were written |

- The runner detects the GPUs and runs every entry that fits the machine (one
  GPU, N GPUs on one node, or two nodes). It skips the others and says why.
- Each entry writes one `bi-results-v1` file
  ([tools/bi_report.py](../../tools/bi_report.py)), and the runner writes one
  combined `report.json` under `suite/runs/`, which is not committed.
- An entry can name its own interpreter (`"python"` in the manifest), for
  example to run inside an engine's environment; where that interpreter does
  not exist, the entry is skipped. Entries whose command uses `{opguard}` need
  `--opguard-root` or `$OPGUARD_ROOT`.
- `moe_ep_2node` compares with the dumps that `moe_ep` wrote under the same
  `--out`, so run the two-node pass with the output directory of the one-node
  run.

### How to read the report

| Section | Meaning | Exit code |
|---|---|---|
| REGRESSIONS | a check with `expect="pass"` failed, so a claim of this handbook no longer holds here | 1 |
| NO RESULT | an entry crashed or timed out | 1 |
| CHANGED | a documented stock failure (`expect="fail"`) now passes, or a verdict differs from the baseline | 0 |
| AS BASELINE | a documented stock failure passes here and also passed in this stack's baseline, so the failure occurs only on some stacks. These lines are counted, and they are listed with `--verbose` and in `report.json` | 0 |
| PATTERN | the verdict matches the baseline, but the failure has a different shape (other failing batch sizes, other counts) | 0 |
| NEW / MISSING | checks that the baseline does not have, or baseline checks that this run did not produce | 0 |
| TIMING | a measurement that differs from the baseline by more than `--timing-ratio` (1.5x by default), in either direction | 0 |
| INCOMPLETE | a section inside an entry raised an exception; its required checks also appear as regressions | — |

- A baseline key consists of the GPU model and count (and those of the second
  node, for two-node entries), torch, Triton, NCCL, the driver and the CUDA
  build. Any change gives a new key.
- When no baseline matches the stack, the report says so. Review the results,
  then record one:
  `python tools/bi_suite.py --from-report <report.json> --record-baseline`.
- A stock failure that passes is reported as CHANGED until a reviewed
  baseline for that stack records it as passing; from then on it is reported
  as AS BASELINE.

### What to do when a stock result changes

1. Read the entry's log and result file, and confirm that the change is in
   the stock op and not in the harness.
2. Update the claim on the topic page and its report, and name the new stack.
   The old result stays true for the old stack.
3. If a stock op now passes a check that motivated one of our kernels, the
   kernel is still needed on the older stacks. Say which ones.
4. Record a new baseline.

A PATTERN line needs no action, unless a topic page quotes the old shape; in
that case, update the page. Examples of stock results that are specific to one
stack are in the
[measured coverage](../../suite/COVERAGE.md#stock-results-that-are-stack-specific).

### How to add a check or an example

Add an entry so that a new claim is re-checked on every stack from now on.
You are done when the entry runs in the suite and its baseline is committed.

An example accepts `--quick` and `--json <file>`, and writes its results in
the `bi-results-v1` format through `tools/bi_report.py`:

```python
res = Results("my_example", quick=args.quick)
res.check("a row alone vs inside batches", rep.passed, level="L2", impl="my_kernel")          # our claim
res.check("a row alone vs inside batches", rep2.passed, level="L2", impl="torch.mm", expect="fail")
res.write(args.json); sys.exit(res.exit_code())
```

- Name a check after its claim, not after its sizes, so that quick and full
  mode produce the same names. Put the sizes in `detail`, and put a short
  description of the failure's shape in `detail["signature"]` to get PATTERN
  lines.
- Use `expect="pass"` for what the handbook claims, `"fail"` for documented
  stock failures, and `"info"` for anything that is recorded but not judged.

Then add one entry to the manifest:

```json
  {"name": "my_example", "gpus": 1,
   "quick": "{python} examples/my_example/my_bi.py --quick --json {result}",
   "full": "{python} examples/my_example/my_bi.py --json {result}",
   "timeout": {"quick": 900, "full": 3600},
   "committed": "examples/my_example/results.json"},
```

- `gpus` and `nodes` (default 1) say what the entry needs. `quick` and `full`
  may each be a list of steps. `{result}` is where the result file lands, and
  `{run}` is the entry's own directory. Set `own_lock: true` when the command
  takes the lock itself.
- `committed` is optional. It names the result file behind the topic page,
  which `--vs-committed` compares with. For a file in an example's own
  format, add `"adapter": "<name>"` (see `suite/adapters/`).
- Run the entry, review the report, record the baseline and commit the
  baseline file.

### Running the suite on new hardware {#new-hardware}

1. Record the stack: `python tools/bi_suite.py --env`.
2. Run full mode: `python tools/bi_suite.py --full --vs-committed`.
3. Review every REGRESSION and CHANGED line, and every difference from the
   committed results.
4. Record a baseline with `--from-report <report.json> --record-baseline`, and
   commit the file under `suite/baselines/`.
5. Add the stack to [measured coverage](../../suite/COVERAGE.md#what-was-measured).

## What can go wrong

!!! warning "Hiding a failure with a tolerance"
    When a required check fails, it is tempting to fix it with a tolerance or
    a skip. That turns a finding into a false claim about this stack. Report
    the failure, and never round outputs or add a tolerance inside a check.

!!! warning "Recording a baseline on top of a regression"
    A baseline records what a stack does, including a required check that
    already fails. Recording it never hides the failure: that check stays a
    regression on every run. So review the results before you record.

!!! warning "Two jobs on one GPU"
    Two measurements that share a GPU can crash or run out of memory, and
    their results cannot be trusted. Every GPU job takes the one lock of its
    node ([machines and locks](#machines-and-locks)).

!!! warning "Taking the lock twice"
    If a command started inside a suite entry takes the same lock again, it
    waits forever, because the runner already holds the lock. Take the lock
    either in the runner or in the command, not in both.

## The baselines recorded so far {#baselines-recorded-here}

Quick-mode baselines have been recorded for three stacks: one GPU, eight GPUs
in one node, and two such nodes. On each stack every required check passed,
and no regression was found. The files are in
[suite/baselines/](../../suite/baselines/), one per stack, named by the
stack's key. The counts, the run times, and the stock failures that occur on
one GPU type but not on the other are in
[measured coverage](../../suite/COVERAGE.md#baselines-recorded-here-quick-mode).

## Sharing machines: one lock per node {#machines-and-locks}

There is one rule: **one lock per node, and every GPU job on that node takes
it.** A job that spans two nodes holds the locks of both. Never skip a lock
that is held; wait for it. After a job, leave no process running and no large
file in `/tmp`. The machines, their lock files, and the reason per-GPU locks
were dropped are in the
[measured coverage](../../suite/COVERAGE.md#machines-and-locks).

## What the suite does not yet cover {#limits}

- The CI workflow is **untested**. This repository has no remote, so the
  workflow has never run. Its commands were run by hand, but the workflow
  around them was not.
- Only quick mode has baselines. Full mode has not been run through the
  suite, and the full-mode commands of the most recently added entries were
  written from each example's own usage without being run.
- The comparison with committed results matches checks by name. If a check's
  quick sizes differ from its full sizes, the comparison checks the same claim
  at different sizes.

## Summary

- A bitwise result belongs to one stack, so re-run the suite whenever the
  stack changes.
- REGRESSIONS means that a claim broke. It is a finding, never something to
  tune away.
- CHANGED means that stock behaviour moved. Update the topic page and record
  a baseline.
- Name checks after their claims, mark each one as a claim or a stock
  failure, and never add a tolerance.
- Every GPU job takes the one lock of its node.

## Go deeper

- [Measured coverage](../../suite/COVERAGE.md): baselines, machines and locks, and stock results that are specific to one stack.
- [tools/bi_suite.py](../../tools/bi_suite.py), [tools/bi_report.py](../../tools/bi_report.py), [suite/manifest.json](../../suite/manifest.json).
- [Hardware coverage](hardware.md): what changes the bits between stacks, and how to qualify new hardware.
