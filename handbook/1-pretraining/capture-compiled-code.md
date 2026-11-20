# When the job runs compiled code

## Where we are

You can capture a pair of runs with OpGuard and walk from the first divergence
to a cause ([Localize with OpGuard](localize-with-opguard.md)), and the last
two pages covered causes in the [training step](training-step.md) and in the
[gradient collectives](data-parallel.md). This page is about a gap in the
capture itself. Many jobs do not run their model op by op: they run it through
`torch.compile`, or replay it as a CUDA graph. To capture, OpGuard runs the job
op by op (eager mode), and eager mode runs other kernels than the compiled job.
So the bits you capture may not be the bits that diverged in production.

You reach this page from two steps of the [workflow](README.md): at capture
(step 3), when the production job is compiled; and at classify (step 5), when an
eager replay does not show the symptom, or the pivot sits in code the compiler
generated. The page explains what the compiler and CUDA graphs do and why each
changes the bits, then how to record fingerprints from inside them in OpGuard's
format, so the rest of the workflow runs unchanged.

## What compiled code does

### torch.compile: many small ops become a few generated kernels

In eager mode PyTorch runs each op when Python calls it: one kernel per op.
Each kernel reads its inputs from GPU memory, computes, and writes its output
back, rounded to the output's dtype. A norm written as a few small ops is that
many kernels, each writing a rounded intermediate.

`torch.compile` reads the Python once (its front end, Dynamo, traces it),
records the ops as a graph, and hands the graph to a compiler (by default
Inductor) that writes new kernels for it. The compiler **fuses** runs of
elementwise ops, and the reductions among them, into one kernel that keeps the
intermediate values in registers (in fp32 when the tensors are bf16) and writes
only the result. It chooses for itself how to split each reduction across
threads. It also rewrites patterns: a matrix multiply followed by an add can
become one matrix multiply that also adds (`addmm`).

<figure class="fx" data-fig="fusion" aria-label="One norm run eagerly and as one compiled kernel">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">eager: one kernel per op</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node is-mono" data-hot="1">x²</span><span class="fx-chip" data-at="1">↓ bf16</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="2">mean</span><span class="fx-chip" data-at="2">↓ bf16</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="3">rsqrt</span><span class="fx-chip" data-at="3">↓ bf16</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="4">× x</span><span class="fx-chip" data-at="4">↓ bf16</span></span><span class="fx-arrow" data-hot="5" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="5">× w</span><span class="fx-chip" data-at="5">↓ bf16</span></span></div></div>
<div class="fx-row"><span class="fx-label">compiled: one generated kernel</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-box to-key" data-hot="6"><span class="fx-note">one kernel</span><span class="fx-chain is-tight fx-sweep" data-hot="6"><span class="fx-chip">x²</span><span class="fx-chip">mean</span><span class="fx-chip">rsqrt</span><span class="fx-chip">× x</span><span class="fx-chip">× w</span></span><span class="fx-note">intermediates in registers, fp32</span></span><span class="fx-chip" data-at="7">↓ bf16 result only</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>The same norm, written as five ops; w is its weight.</li>
<li>Eager: one kernel per op. x² is written to memory, rounded to bf16.</li>
<li>The mean reads it back, reduces, and writes a rounded result.</li>
<li>rsqrt: another kernel, another rounding.</li>
<li>× x: another kernel, another rounding.</li>
<li>× w: five kernels, five rounded intermediates in memory.</li>
<li data-hold="2">Compiled: one generated kernel runs all five ops in fp32 registers, and sums the mean in its own order.</li>
<li data-hold="2">It writes only the result: one rounding instead of five.</li>
</ol>
<figcaption>Figure: the same norm (w is its weight). Eager writes every intermediate to memory in its dtype; the fused kernel keeps them in registers, sums the mean in its own order, and writes only the result.</figcaption>
</figure>

### CUDA graphs: record the launches once, replay them

Launching kernels one by one from Python costs CPU time for every kernel. When
a step is small, that time can exceed the GPU work. A **CUDA graph** records a
sequence of launches once (which kernels, with which arguments, on which
buffers) and then replays the whole sequence with one call. The replay runs no
Python and makes no new choices: kernels, shapes and memory addresses are those
of the recording.

So a graph fits one shape. A job that sees inputs of many sizes records a few
graphs, one per size bucket, and pads each input up to the next recorded
size.

<figure class="fx" data-fig="graph-pad" aria-label="A batch of 5 rows replayed in the graph recorded for 8">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-box to-key" data-hot="2"><span class="fx-swap"><span class="fx-note" data-to="2">a batch of 5 rows</span><span class="fx-note" data-at="2">graph recorded for 8 rows</span></span><span class="fx-cells fx-sweep" style="--n:8;--cw:1.7rem" data-hot="2"><span class="fx-cell is-mine">0</span><span class="fx-cell is-mine">1</span><span class="fx-cell is-mine">2</span><span class="fx-cell is-mine">3</span><span class="fx-cell is-mine">4</span><span class="fx-cell is-pad" data-at="1">5</span><span class="fx-cell is-pad" data-at="1">6</span><span class="fx-cell is-pad" data-at="1">7</span></span></span><span class="fx-arrow" data-at="3">keep</span><span class="fx-box" data-at="3"><span class="fx-note">first 5 rows</span><span class="fx-cells" style="--n:5;--cw:1.7rem"><span class="fx-cell is-mine">0</span><span class="fx-cell is-mine">1</span><span class="fx-cell is-mine">2</span><span class="fx-cell is-mine">3</span><span class="fx-cell is-mine">4</span></span></span></div>
</div>
<ol class="fx-say">
<li>A batch of 5 rows arrives.</li>
<li>It is padded up to the next recorded size: 3 padding rows (5 to 7) make 8.</li>
<li data-hold="2">The graph recorded for 8 rows replays: every kernel in it sees 8 rows.</li>
<li data-hold="2">Only the first 5 rows are kept as the answer.</li>
</ol>
<figcaption>Figure: a batch of 5 runs in the graph recorded for 8. The answer uses only the real rows, but every kernel in the replay saw 8.</figcaption>
</figure>

## Why the bits change

**Fusion moves rounding points.** A tiny example in bf16, where the step
between neighbouring numbers near 1 is ε = 2⁻⁷. Let a = b = 1 + ε and
c = −(1 + 2ε), and compute a · b + c.

| | Eager: two kernels | Compiled: one fused kernel |
|---|---|---|
| a · b | 1 + 2ε + ε², rounded to bf16 when written: ε² is less than half a step, so 1 + 2ε | 1 + 2ε + ε², kept in fp32 in a register |
| + c | (1 + 2ε) − (1 + 2ε) = 0 | ε² |
| Result in bf16 | 0 | ε² |

The mathematics is the same; the number of roundings is not. The fused kernel's
own reduction order changes a sum the same way part 0's three numbers did
([part 0](../0-foundations/README.md#floating-point-addition-is-not-associative)).

**Replay at a padded size changes the shape a GEMM sees.** The padding rows are
not in the answer, but they are in the matrix: the GEMM in the replay multiplies
8 rows, not 5. The GEMM library chooses its kernel, and with it the reduction
order, from the shape
([part 0: a reduction's order depends on shape](../0-foundations/README.md#a-reductions-order-depends-on-shape)).
So the 5 real rows can round differently from an eager run at 5 rows. A
batch-invariant GEMM ([part 3](../3-kernels/gemm-and-quantized.md)) removes this
one.

**Rule.** Compiled code is different arithmetic from the eager code it came
from: other rounding points, other reduction orders, other shapes. A comparison
made in eager tells you about eager.

## Why the usual capture cannot see inside

OpGuard captures by wrapping calls in the model's Python code: when a wrapped
call runs, the wrapper fingerprints its inputs and outputs
([Localize with OpGuard](localize-with-opguard.md)). Compiled code does not
make those calls one by one. The compiler read the Python once and then runs
the kernels it generated; a graph replay runs no Python at all. Inside a
compiled graph or a replay there are no Python calls left to wrap.

So the usual capture runs the job in eager, which is the other arithmetic from
the section above. That is enough when the symptom reproduces in eager. When it
does not, the observer has to be placed inside the compiled code.

## The idea: an observer the compiler keeps and cannot see through

**An observer inside compiled code is an op the compiler must keep.** A custom
op is one: the compiler calls it as a black box at its place in the graph, and
a CUDA graph records it like any other kernel, so every replay runs it again
and it fingerprints the tensor as the compiled code computed it.

**A probe can change what it measures.** A tiny example: compiled code computes
`s = x + o_proj(h)` and passes `s` to the next norm. The add is pointwise, so
Inductor fuses it into the norm kernel, and the GEMM rounds its output first.
Add a probe that reads `s`, and `s` now has a non-pointwise user; Inductor then
folds the add into the GEMM (`addmm`), which adds before its one rounding.
Probing `s` changed `s`.

**Rule.** Observe compiled code from inside it, and make the observer
invisible to the compiler: the probe must read an exact copy that the
compiler cannot fuse through. Then prove it, by comparing a run with probes
against one without. This instances a general measurement principle: the
instrument must not change the quantity.

## The step

### Probe compiled code and compare with OpGuard

**Goal:** fingerprints of chosen tensors inside compiled code and graph
replays, readable by OpGuard's harness.

**How:** [graph_probe.py](../../tools/graph_probe.py).

1. Build lazy caches (RoPE tables and the like) in eager before compiling.
2. Create a `ProbeSet` and call `ps.probe(name, *tensors)` at the boundaries
   you want, and `ps.step()` once per step. Each probe is a custom op
   (`graph_probe::record`) that Dynamo keeps and Inductor calls as an opaque
   kernel; it writes `xor_signature` (and optionally `xor_vectors`) into a
   device ring buffer, so each graph replay writes its own row.
3. Keep `isolate=True` (the default): under compile the probe reads an exact
   pointwise copy (`neg(neg(x))`, integers `x ^ 0`) instead of the tensor.
4. Run once with probes and once without, and compare the outputs bitwise.
5. `ps.read()` and `ps.emit(writer)` write one `TraceWriter` record per step
   and probed tensor; `export_pair` and OpGuard's harness compare them as
   usual ([localize](localize-with-opguard.md)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Eager OpGuard capture of an eager replay | Every boundary, call sites, no new code | small per boundary | The symptom reproduces in eager |
| Graph probe inside a CUDA graph | The bits the graph replays produce | small per probe (its kernels only); many probes on a small step add up to large | Production replays CUDA graphs |
| Graph probe under `torch.compile` without graphs | The bits compiled code produces | large: the custom op's Python body runs per probe | Compile without graphs; probe few tensors |
| Every rounding op opaque (`bi_opaque_all`-style custom ops) | Compiled bits equal to eager bits | not measured here | You need compiled = eager, not just a capture |

**Success:** the probe's records equal `opguard_bridge.xor_signature` on the
same tensors; outputs with and without probes are bitwise equal; OpGuard's
pivot on the exported pair equals a direct comparison of the records.

Shown on one GPU type with a decode step of a small model: the probe's
fingerprints matched the bridge and OpGuard's own function, OpGuard found the
same first divergence as a direct comparison in every pair, and with isolation
the probes left the compiled bits unchanged in every compile mode
([report](../../examples/graph_probe/REPORT.md#what-this-shows)).

### Read what the probes show

| Observation | Meaning |
|---|---|
| Compiled differs from eager at the first norm | Inductor's fused norm has its own order: expected for stock ops |
| A CUDA graph equals eager at the same shape, and differs when padded into a bigger bucket | the GEMM chose a kernel by the bucket size: a batch-variant kernel (L2) |
| Compiled equals eager only when every rounding op is opaque | the fused code was the difference; batch-invariant kernels alone are not enough under compile |

After the first difference every later probe differs too; probes cannot say
whether later ops would agree on equal inputs. Compile each op alone for that
([attention: serving layers](../3-kernels/attention.md)).

## What can go wrong

!!! warning "A probe that is not isolated"
    A probe on a tensor gives it a non-pointwise user, and Inductor can change
    its plan (folding an add into `addmm`). You then measure a run that
    production never runs. Keep `isolate=True` and run the with/without check.

!!! warning "A lazy cache filled inside compiled code"
    A table built on first use, when that use is compiled, is computed by
    Inductor and then served to every later run, eager included. Eager runs
    change too. Build such caches in eager before compiling.

!!! warning "`clone()` as isolation"
    The compiler removes a clone as a no-op, and a dtype view inside a fused
    kernel may fail to compile. The probe is then not isolated. Use the
    double negation the tool uses.

!!! warning "Shapes that change"
    A probe slot keeps the shape of its first call, and dynamic shapes were
    not tested. A new shape under the same name is wrong. Give it another name.

## Summary

- `torch.compile` fuses ops and picks its own reduction order; a CUDA graph
  replays fixed launches at a padded shape. Both are different arithmetic from
  eager.
- OpGuard wraps Python calls, and compiled code and graph replays make none, so
  an eager capture shows eager bits: capture compiled code from inside.
- A probe is a custom op with a device ring buffer; graph replays write their
  own rows.
- Isolate every probe behind an exact pointwise copy, and prove it with a
  with/without comparison.
- Build lazy caches in eager before compiling.

## Go deeper

- [Report: fingerprints inside torch.compile and CUDA graphs](../../examples/graph_probe/REPORT.md):
  every mode, bucket and kit compared, the probe's effect, costs, the
  reproduction commands.
- Code: [graph_probe.py](../../tools/graph_probe.py) and
  [graph_probe_cases.py](../../examples/graph_probe/graph_probe_cases.py).
- [Attention](../3-kernels/attention.md): paged, varlen and compiled attention.
- [Localize with OpGuard](localize-with-opguard.md): what to do with the
  records.
