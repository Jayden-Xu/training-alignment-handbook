# Compiled code and CUDA graphs

## Where we are

You can capture a pair of runs with OpGuard and trace from the first
divergence to a cause ([Localize with OpGuard](localize-with-opguard.md)), and
the last two pages covered causes in the [training step](training-step.md) and
in the [gradient collectives](data-parallel.md). This page is about a gap in
the capture itself. Many jobs do not run their model one operator at a time.
They run it through `torch.compile`, or they replay it as a CUDA graph. To
capture, OpGuard runs the job one operator at a time (eager mode), and eager
mode runs different kernels from the compiled job. So the bits you capture may
not be the bits that diverged in production.

You reach this page from two steps of the [workflow](README.md). At step 3
(capture), you come here when the production job is compiled. At step 5
(classify), you come here when an eager replay does not show the symptom, or
when the pivot sits in code that the compiler generated. The page first
explains what the compiler and CUDA graphs do and why each of them changes the
bits. It then shows how to record fingerprints from inside them in OpGuard's
format, so that the rest of the workflow runs unchanged.

## What compiled code does

### torch.compile turns many small operators into a few generated kernels

In eager mode, PyTorch runs each operator as soon as Python calls it, with one
kernel per operator. Each kernel reads its inputs from GPU memory, computes,
and writes its output back to memory, rounded to the output's dtype. A norm
written as a few small operators is that many kernels, and each one writes out
a rounded intermediate result.

`torch.compile` reads the Python code once (its front end, Dynamo, traces it),
records the operators as a graph, and hands the graph to a compiler (Inductor
by default) that writes new kernels for it. The compiler **fuses** runs of
elementwise operators, and the reductions among them, into a single kernel.
That kernel keeps the intermediate values in registers (in fp32 when the
tensors are bf16) and writes only the final result. The compiler also decides
for itself how to split each reduction across threads. And it rewrites
patterns: a matrix multiply followed by an add can become a single matrix
multiply that also does the add (`addmm`).

<figure class="fx" data-fig="fusion" aria-label="One norm run eagerly and as one compiled kernel">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">eager: one kernel per operator</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node is-mono" data-hot="1">x²</span><span class="fx-chip" data-at="1">↓ bf16</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="2">mean</span><span class="fx-chip" data-at="2">↓ bf16</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="3">rsqrt</span><span class="fx-chip" data-at="3">↓ bf16</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="4">× x</span><span class="fx-chip" data-at="4">↓ bf16</span></span><span class="fx-arrow" data-hot="5" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="5">× w</span><span class="fx-chip" data-at="5">↓ bf16</span></span></div></div>
<div class="fx-row"><span class="fx-label">compiled: one generated kernel</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-box to-key" data-hot="6"><span class="fx-note">one kernel</span><span class="fx-chain is-tight fx-sweep" data-hot="6"><span class="fx-chip">x²</span><span class="fx-chip">mean</span><span class="fx-chip">rsqrt</span><span class="fx-chip">× x</span><span class="fx-chip">× w</span></span><span class="fx-note">intermediate values stay in fp32 registers</span></span><span class="fx-chip" data-at="7">↓ bf16, only the result</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>The same norm, written as five operators; w is its weight.</li>
<li>In eager mode, each operator is one kernel. x² is written to memory, rounded to bf16.</li>
<li>The mean reads it back, reduces it, and writes a rounded result.</li>
<li>rsqrt is another kernel, with another rounding.</li>
<li>× x is another kernel, with another rounding.</li>
<li>After × w there have been five kernels, and five rounded intermediate results in memory.</li>
<li data-hold="2">Compiled, one generated kernel runs all five operators in fp32 registers, and sums the mean in its own order.</li>
<li data-hold="2">It writes only the result: one rounding instead of five.</li>
</ol>
<figcaption>Figure: the same norm (w is its weight). Eager mode writes every intermediate result to memory in its dtype; the fused kernel keeps them in registers, sums the mean in its own order, and writes only the result.</figcaption>
</figure>

### A CUDA graph records the kernel launches once and replays them

Launching kernels one at a time from Python costs CPU time for every kernel.
When a step is small, that time can exceed the GPU work itself. A **CUDA
graph** records a sequence of launches once (which kernels, with which
arguments, on which buffers) and then replays the whole sequence with a single
call. The replay runs no Python and makes no new choices: the kernels, shapes
and memory addresses are those of the recording.

So a graph fits only one shape. A job that sees inputs of many sizes records a
few graphs, one for each size bucket, and pads each input up to the next
recorded size.

<figure class="fx" data-fig="graph-pad" aria-label="A batch of 5 rows replayed in the graph recorded for 8">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-box to-key" data-hot="2"><span class="fx-swap"><span class="fx-note" data-to="2">a batch of 5 rows</span><span class="fx-note" data-at="2">the graph recorded for 8 rows</span></span><span class="fx-cells fx-sweep" style="--n:8;--cw:1.7rem" data-hot="2"><span class="fx-cell is-mine">0</span><span class="fx-cell is-mine">1</span><span class="fx-cell is-mine">2</span><span class="fx-cell is-mine">3</span><span class="fx-cell is-mine">4</span><span class="fx-cell is-pad" data-at="1">5</span><span class="fx-cell is-pad" data-at="1">6</span><span class="fx-cell is-pad" data-at="1">7</span></span></span><span class="fx-arrow" data-at="3">keep</span><span class="fx-box" data-at="3"><span class="fx-note">the first 5 rows</span><span class="fx-cells" style="--n:5;--cw:1.7rem"><span class="fx-cell is-mine">0</span><span class="fx-cell is-mine">1</span><span class="fx-cell is-mine">2</span><span class="fx-cell is-mine">3</span><span class="fx-cell is-mine">4</span></span></span></div>
</div>
<ol class="fx-say">
<li>A batch of 5 rows arrives.</li>
<li>It is padded up to the next recorded size: 3 padding rows (rows 5 to 7) bring it to 8.</li>
<li data-hold="2">The graph recorded for 8 rows replays, and every kernel in it sees 8 rows.</li>
<li data-hold="2">Only the first 5 rows are kept as the answer.</li>
</ol>
<figcaption>Figure: a batch of 5 rows runs in the graph recorded for 8. The answer uses only the real rows, but every kernel in the replay saw 8.</figcaption>
</figure>

## Why compiled code gives different bits {#why-the-bits-change}

**Fusion moves the rounding points.** Here is a small example in bf16, where
the gap between neighbouring numbers near 1 is ε = 2⁻⁷. Let a = b = 1 + ε and
c = −(1 + 2ε), and compute a · b + c.

| | Eager: two kernels | Compiled: one fused kernel |
|---|---|---|
| a · b | 1 + 2ε + ε², rounded to bf16 when it is written out: ε² is less than half a gap, so the result is 1 + 2ε | 1 + 2ε + ε², kept in fp32 in a register |
| + c | (1 + 2ε) − (1 + 2ε) = 0 | ε² |
| Result in bf16 | 0 | ε² |

The mathematics is the same, but the number of roundings is not. The fused
kernel's own reduction order also changes a sum, in the same way that part 0's
three numbers did
([part 0](../0-foundations/README.md#floating-point-addition-is-not-associative)).

**A replay at a padded size changes the shape that a GEMM sees.** The padding
rows are not part of the answer, but they are part of the matrix: the GEMM in
the replay multiplies 8 rows, not 5. The GEMM library chooses its kernel, and
with it the reduction order, from the shape
([part 0: a kernel picks its reduction order from the shape](../0-foundations/README.md#a-reductions-order-depends-on-shape)).
So the 5 real rows can round differently from an eager run with 5 rows. A
batch-invariant GEMM ([part 3](../3-kernels/gemm-and-quantized.md)) removes
this particular effect.

**Rule.** Compiled code does different arithmetic from the eager code it was
compiled from: it has other rounding points, other reduction orders and other
shapes. A comparison made in eager mode tells you about eager mode.

## Why the usual capture cannot see inside compiled code {#why-the-usual-capture-cannot-see-inside}

OpGuard captures by wrapping calls in the model's Python code: when a wrapped
call runs, the wrapper fingerprints its inputs and outputs
([Localize with OpGuard](localize-with-opguard.md)). Compiled code does not
make those calls one by one. The compiler read the Python once and from then
on runs the kernels it generated, and a graph replay runs no Python at all.
Inside a compiled graph or a replay, there are no Python calls left to wrap.

So the usual capture runs the job in eager mode, which is the other
arithmetic described in the section above. That is enough when the symptom
reproduces in eager mode. When it does not, the observer has to be placed
inside the compiled code.

## The idea: an observer that the compiler keeps but cannot optimize through

**An observer inside compiled code must be an operator that the compiler
keeps.** A custom operator is one. The compiler calls it as a black box at
its place in the graph, and a CUDA graph records it like any other kernel, so
every replay runs it again, and it fingerprints the tensor exactly as the
compiled code computed it.

**A probe can change what it measures.** Here is a small example. Compiled
code computes `s = x + o_proj(h)` and passes `s` to the next norm. The add is
pointwise, so Inductor fuses it into the norm's kernel, and the GEMM rounds
its own output first. Now add a probe that reads `s`. `s` then has a user that
is not pointwise, so Inductor instead folds the add into the GEMM (`addmm`),
which adds before its single rounding. Probing `s` changed `s`.

**Rule.** Observe compiled code from inside it, and make the observer
invisible to the compiler: the probe must read an exact copy of the tensor
that the compiler cannot fuse through. Then prove that it worked, by comparing
a run with probes against a run without them. This applies a general
principle of measurement: the instrument must not change the quantity it
measures.

## The step

### Place probes in the compiled code, and compare the runs with OpGuard

Use [graph_probe.py](../../tools/graph_probe.py) to fingerprint chosen
tensors inside compiled code and inside graph replays, in a form that
OpGuard's harness can read.

1. Build lazy caches (RoPE tables and the like) in eager mode before you
   compile.
2. Create a `ProbeSet`, call `ps.probe(name, *tensors)` at the boundaries you
   want, and call `ps.step()` once per step. Each probe is a custom operator
   (`graph_probe::record`) that Dynamo keeps and Inductor calls as an opaque
   kernel. It writes `xor_signature` (and, optionally, `xor_vectors`) into a
   ring buffer on the device, so each graph replay writes its own row.
3. Keep `isolate=True` (the default). Under compile, the probe then reads an
   exact pointwise copy of the tensor (`neg(neg(x))`, or `x ^ 0` for
   integers) instead of the tensor itself.
4. Run once with probes and once without, and compare the outputs bit for bit.
5. `ps.read()` and `ps.emit(writer)` write one `TraceWriter` record for each
   step and probed tensor; `export_pair` and OpGuard's harness then compare
   them as usual ([localize](localize-with-opguard.md)).

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| An eager OpGuard capture of an eager replay | Every boundary and its call site, with no new code | small at each boundary | The symptom reproduces in eager mode |
| A graph probe inside a CUDA graph | The bits that the graph replays actually produce | small for each probe (only its own kernels); many probes on a small step add up to large | Production replays CUDA graphs |
| A graph probe under `torch.compile` without CUDA graphs | The bits that compiled code produces | large: the custom operator's Python body runs for every probe | Compile without graphs, and probe only a few tensors |
| Every rounding operator made opaque (custom operators in the style of `bi_opaque_all`) | Compiled bits equal to eager bits | not measured here | You need compiled code to equal eager, not just a capture |

The probes work when their records equal `opguard_bridge.xor_signature` on
the same tensors, the outputs with and without probes are equal bit for bit,
and OpGuard's pivot on the exported pair is the same as a direct comparison of
the records.

This was shown on one GPU type, with a decode step of a small model. The
probe's fingerprints matched both the bridge and OpGuard's own function,
OpGuard found the same first divergence as a direct comparison in every pair,
and with isolation, the probes left the compiled bits unchanged in every
compile mode ([report](../../examples/graph_probe/REPORT.md#what-this-shows)).

### Interpret what the probes show

| What you observe | What it means |
|---|---|
| Compiled code differs from eager at the first norm | Inductor's fused norm sums in its own order, which is expected for stock operators |
| A CUDA graph equals eager at the same shape, but differs when the input is padded into a bigger bucket | the GEMM chose its kernel by the bucket size, so it is a kernel that varies with the batch (it breaks L2) |
| Compiled code equals eager only when every rounding operator is opaque | the fused code caused the difference; under compile, batch-invariant kernels alone are not enough |

After the first difference, every later probe differs too, so the probes
cannot tell you whether the later operators would agree if their inputs were
equal. To find that out, compile each operator on its own
([attention: serving layers](../3-kernels/attention.md)).

## What can go wrong

!!! warning "A probe that is not isolated"
    A probe on a tensor gives that tensor a user that is not pointwise, and
    Inductor may then change its plan (for example, folding an add into
    `addmm`). You then measure a run that production never runs. Keep
    `isolate=True`, and run the comparison with and without probes.

!!! warning "A lazy cache filled inside compiled code"
    A table that is built on first use is computed by Inductor when that first
    use is compiled, and is then served to every later run, eager runs
    included. So the eager runs change too. Build such caches in eager mode
    before you compile.

!!! warning "`clone()` used for isolation"
    The compiler removes a clone as a no-op, and a dtype view inside a fused
    kernel may fail to compile. Either way, the probe is not isolated. Use the
    double negation that the tool uses.

!!! warning "Shapes that change"
    A probe slot keeps the shape of its first call, and dynamic shapes have
    not been tested. A new shape under the same name gives wrong records. Give
    it another name.

## Summary

- `torch.compile` fuses operators and picks its own reduction order, and a
  CUDA graph replays a fixed set of launches at a padded shape. Both do
  different arithmetic from eager mode.
- OpGuard wraps Python calls, and compiled code and graph replays make none,
  so an eager capture shows eager bits. Capture compiled code from inside it.
- A probe is a custom operator that writes into a ring buffer on the device;
  each graph replay writes its own row.
- Isolate every probe behind an exact pointwise copy, and prove the isolation
  by comparing runs with and without probes.
- Build lazy caches in eager mode before you compile.

## Go deeper

- [Report: fingerprints inside torch.compile and CUDA graphs](../../examples/graph_probe/REPORT.md)
  compares every compile mode, bucket and kernel set, and covers the probe's
  effect, the costs and the commands to reproduce the results.
- The code is in [graph_probe.py](../../tools/graph_probe.py) and
  [graph_probe_cases.py](../../examples/graph_probe/graph_probe_cases.py).
- [Attention](../3-kernels/attention.md) covers paged, varlen and compiled
  attention.
- [Localize with OpGuard](localize-with-opguard.md) explains what to do with
  the records.
