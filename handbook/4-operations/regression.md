# Regression: re-check the verified results

You have a stack where the handbook's checks hold, and something in it is about
to change: torch, Triton, NCCL, the driver, the CUDA build, or the GPUs. This
page re-runs every verified check on the new stack and tells you which claims
still hold, which broke, and which stock behaviour merely moved.

## The idea

Take a stock matrix multiply that rounds a row differently once the batch
passes some size. That size is a property of one library version on one GPU:
the library picks its kernel from the shape, and another version can pick
another kernel at another size, or the same kernel everywhere. So after an
upgrade two kinds of change look alike and mean opposite things:

| What changed | Example | What it means |
|---|---|---|
| A check the handbook claims now fails | the handbook's invariant GEMM gives a row other bits in a larger batch | a finding: the claim no longer holds on this stack |
| A documented stock failure now passes, or fails elsewhere | the stock GEMM varies from another batch size, or not at all | the stack changed under it; update the topic page |

The suite keeps them apart by marking each check as a claim (`expect="pass"`) or
a documented stock failure (`expect="fail"`), and by comparing verdicts with a
baseline recorded for this exact stack. Equality inside each check is bitwise;
the suite never judges a number by closeness.

## The step

- Runner: [tools/bi_suite.py](../../tools/bi_suite.py)
- Entries: [suite/manifest.json](../../suite/manifest.json)
- Baselines: [suite/baselines/](../../suite/baselines/)
- Adapters for the first examples' own result formats: [suite/adapters/](../../suite/adapters/)
- CI: [.github/workflows/bi-suite.yml](../../.github/workflows/bi-suite.yml), **untested** (see [limits](#limits))

### When to run it

- Before and after upgrading torch, Triton, NCCL, the driver or the CUDA build.
- On new hardware ([below](#new-hardware)).
- Before claiming parity in a release.
- After editing an example, a kernel in `tools/`, or an adapter.

### How to run it

**Goal:** every check that fits this machine, run and compared.
**Success:** the runner exits 0 and the report has no REGRESSIONS or NO RESULT lines.

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
| `--vs-committed` | also compares with the result files behind the topic pages | free | when you want to see what moved since the pages were written |

- The runner reads the GPUs and runs every entry that fits (one GPU, N GPUs on
  one node, two nodes); it skips the others and says why.
- Each entry writes one `bi-results-v1` file ([tools/bi_report.py](../../tools/bi_report.py));
  the runner writes one combined `report.json` under `suite/runs/` (not committed).
- An entry can name its own interpreter (`"python"` in the manifest), for an
  engine environment; it is skipped where that interpreter does not exist.
  Entries whose command uses `{opguard}` need `--opguard-root` or `$OPGUARD_ROOT`.
- `moe_ep_2node` compares with the dumps `moe_ep` wrote under the same `--out`:
  run the two-node pass with the one-node run's output directory.

### How to read the report

| Section | Meaning | Exit code |
|---|---|---|
| REGRESSIONS | a check with `expect="pass"` failed: a claim of this handbook no longer holds here | 1 |
| NO RESULT | an entry crashed or timed out | 1 |
| CHANGED | a documented stock failure (`expect="fail"`) now passes, or a verdict differs from the baseline | 0 |
| AS BASELINE | a documented stock failure passes here and passed in this stack's baseline too: the failure is stack-specific. Counted, listed with `--verbose` and in `report.json` | 0 |
| PATTERN | same verdict as the baseline, but the failure has another shape (other failing batch sizes, other counts) | 0 |
| NEW / MISSING | checks the baseline does not have, or baseline checks this run did not produce | 0 |
| TIMING | a measurement beyond `--timing-ratio` (default 1.5x) of the baseline, either way | 0 |
| INCOMPLETE | a section inside an entry raised; its required checks also appear as regressions | — |

- A baseline key is: GPU model and count (and the second node's, for two-node
  entries), torch, Triton, NCCL, driver and CUDA build. Any change gives a new key.
- When no baseline matches the stack, the report says so. Review the results,
  then record one: `python tools/bi_suite.py --from-report <report.json> --record-baseline`.
- A stock failure that passes is CHANGED until a reviewed baseline for that
  stack records it passing; from then on it is AS BASELINE.

### What to do with a changed stock result

1. Read the entry's log and result file; confirm the change is in the stock op,
   not in the harness.
2. Update the topic page's claim and its report, naming the new stack. The old
   result stays true for the old stack.
3. If a stock op now passes a check that motivated one of our kernels, the
   kernel is still needed on older stacks. Say which.
4. Record a new baseline.

A PATTERN line needs no action unless a topic page quotes the old shape; then
update the page. Examples of stack-specific stock results are in the
[measured coverage](../../suite/COVERAGE.md#stock-results-that-are-stack-specific).

### How to add a check or an example

**Goal:** a new claim is re-checked on every stack from now on.
**Success:** the entry runs in the suite, and its baseline is committed.

An example takes `--quick` and `--json <file>`, and writes `bi-results-v1`
through `tools/bi_report.py`:

```python
res = Results("my_example", quick=args.quick)
res.check("a row alone vs inside batches", rep.passed, level="L2", impl="my_kernel")          # our claim
res.check("a row alone vs inside batches", rep2.passed, level="L2", impl="torch.mm", expect="fail")
res.write(args.json); sys.exit(res.exit_code())
```

- Name a check after the claim, not the sizes, so quick and full mode give the
  same names. Put sizes in `detail`; put a short failure shape in
  `detail["signature"]` to get PATTERN lines.
- `expect="pass"` for what the handbook claims; `"fail"` for documented stock
  failures; `"info"` for anything recorded but not judged.

Then add one manifest entry:

```json
  {"name": "my_example", "gpus": 1,
   "quick": "{python} examples/my_example/my_bi.py --quick --json {result}",
   "full": "{python} examples/my_example/my_bi.py --json {result}",
   "timeout": {"quick": 900, "full": 3600},
   "committed": "examples/my_example/results.json"},
```

- `gpus` and `nodes` (default 1) say what it needs. `quick`/`full` may be a list
  of steps. `{result}` is where the result lands; `{run}` is the entry's own
  directory. `own_lock: true` when the command takes the lock itself.
- `committed` is optional: the result file behind the topic page, compared by
  `--vs-committed`. For a file in an example's own format, add
  `"adapter": "<name>"` (see `suite/adapters/`).
- Run it, review the report, record the baseline, commit the baseline file.

### New hardware

1. Record the stack: `python tools/bi_suite.py --env`.
2. Run full mode: `python tools/bi_suite.py --full --vs-committed`.
3. Review every REGRESSION and CHANGED line, and every difference from the
   committed results.
4. Record a baseline: `--from-report <report.json> --record-baseline`, and
   commit the file under `suite/baselines/`.
5. Add the stack to [measured coverage](../../suite/COVERAGE.md#what-was-measured).

## What can go wrong

!!! warning "Tuning a failure away"
    A failing required check is tempting to fix with a tolerance or a skip.
    That turns a finding into a false claim on this stack. Report it, and never
    round outputs or add a tolerance in a check.

!!! warning "Recording a baseline over a regression"
    A baseline records what a stack does, including a required check that
    already fails. Recording one never hides it: that check stays a regression
    on every run, so review before you record.

!!! warning "Two jobs on one GPU"
    Two measurements sharing a GPU can crash or run out of memory, and the
    results cannot be trusted. Take one lock per node for every GPU job
    ([machines and locks](#machines-and-locks)).

!!! warning "Taking the lock twice"
    A command started inside a suite entry that takes the same lock again
    waits forever, because the runner already holds it. Take the lock in the
    runner or in the command, not both.

## Baselines recorded here

Quick-mode baselines are recorded for three stacks: one GPU, eight GPUs in one
node, and two such nodes. On each, every required check passed and no
regression was found. The files are in
[suite/baselines/](../../suite/baselines/), one per stack, named by its key.
The counts, run times, and the stock failures that pass on one GPU type and
not the other are in
[measured coverage](../../suite/COVERAGE.md#baselines-recorded-here-quick-mode).

## Machines and locks

One rule: **one lock per node, and every GPU job on that node takes it.** A
job that spans two nodes holds the lock of both. Never skip a lock that is
held: wait. Leave no process running and no large file in `/tmp` after a job.
The machines, their lock files and why per-GPU locks were dropped are in the
[measured coverage](../../suite/COVERAGE.md#machines-and-locks).

## Limits

- The CI workflow is **untested**: this repository has no remote, so it has
  never run. Its commands were run by hand; the workflow around them was not.
- Only quick mode has baselines. Full mode was not run through the suite, and
  the full commands of the entries added last were written from each
  example's own usage, not run.
- The comparison with committed results matches checks by name; a check whose
  quick size differs from the full one compares the same claim at other sizes.

## Summary

- A bitwise result belongs to one stack; re-run the suite whenever the stack changes.
- REGRESSIONS means a claim broke: a finding, never something to tune away.
- CHANGED means stock behaviour moved: update the topic page and record a baseline.
- Name checks after claims, mark them claim or stock failure, and never add a tolerance.
- One lock per node for every GPU job.

## Go deeper

- [Measured coverage](../../suite/COVERAGE.md): baselines, machines and locks, stack-specific stock results.
- [tools/bi_suite.py](../../tools/bi_suite.py), [tools/bi_report.py](../../tools/bi_report.py), [suite/manifest.json](../../suite/manifest.json).
- [Hardware coverage](hardware.md): what changes bits between stacks, and how to qualify new hardware.
