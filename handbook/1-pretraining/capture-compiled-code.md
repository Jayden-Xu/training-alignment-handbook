# Fingerprints inside torch.compile and CUDA graphs

## Where we are

This is step 3 (capture) of the [pre-training workflow](README.md), for a run
that uses `torch.compile` or CUDA graphs. The two runs are controlled
([step 2](determinism-and-preflight.md)). What is still open: OpGuard captures
eager execution, and compiled code runs other kernels, so an eager capture may
not show the bits production computes. This page records fingerprints from
inside compiled code and graph replays, in OpGuard's format, so the rest of the
workflow ([localize](localize-with-opguard.md)) works unchanged.

## The idea: compiled code is different arithmetic, and a probe can change it

**Compile changes the order.** In eager mode an RMSNorm is several kernels,
each rounding its output. Inductor fuses the ops between GEMMs (norms, RoPE,
SiLU·mul, residual adds) into its own Triton kernels, with its own reduction
order and math functions, and rewrites `mm` + add into one `addmm`. The same
norm is then a different sum.

**A CUDA graph changes the shape.** A graph replays the kernels chosen at
capture. A request padded into a bucket runs at the bucket's batch size, and
the GEMM library picks its kernel by that size: part 0's split-K example,
replayed.

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

- Compiled code and CUDA graphs are different arithmetic from eager: capture
  them from inside.
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
