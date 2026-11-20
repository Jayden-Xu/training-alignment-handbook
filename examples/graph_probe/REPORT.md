# Fingerprints inside torch.compile and CUDA graphs: lab report

This is the full record behind the handbook page [Capture compiled code](../../handbook/1-pretraining/capture-compiled-code.md). The page teaches the idea and the step; this report keeps the setup, every configuration, every table, the limits and how the case was built.

## Summary

- [tools/graph_probe.py](../../tools/graph_probe.py) records OpGuard-format
  fingerprints inside `torch.compile` code and CUDA-graph replays. OpGuard's
  harness then found the same first divergence as a direct comparison in 19 of
  19 pairs (A40).
- A stock compiled decode step differs from eager at the first RMSNorm; a CUDA
  graph equals eager at the same shape; padding into a bigger bucket changed
  stock bits from the o_proj GEMM on, and did not with bi kernels.
- Compiled = eager bits only with every rounding op opaque (`bi_opaque_all`),
  in all three compile modes.
- **Cautions:** a naive probe changed the compiled bits (`isolate=True` fixes
  it); build lazy caches such as RoPE tables in eager before compiling.

## How

OpGuard captures eager execution. A serving engine runs `torch.compile` output
and CUDA graphs, and those run other kernels. So an eager capture may not show
the bits that production computes. This case adds a probe that records
OpGuard-format fingerprints from inside compiled code and graph replays, and
uses it to locate where compiled and graph runs part from eager.
### Where the variance comes from

- **Compile.** Inductor fuses the torch ops between GEMMs (RMSNorm, RoPE,
  SiLU·mul, residual adds) into its own Triton kernels, with its own reduction
  order and math functions. It also rewrites `mm` + add into `addmm`.
- **CUDA graphs.** A graph replays the kernels chosen at capture. A request
  padded into a bucket runs at the bucket's batch size, and cuBLAS picks its
  kernel by that size.
- **Eager capture misses both.** OpGuard sees eager kernels only.

### The probe

`ProbeSet` holds static device buffers: a step counter, a ring of `depth` rows
of fingerprints, a per-slot step stamp, and an arena for per-vector values.

| Piece | What it does |
|---|---|
| `ps.step()` | custom op `graph_probe::step`: counter += 1 on the device |
| `ps.probe(name, *tensors)` | custom op `graph_probe::record` per tensor. Computes `xor_signature` (flat-u32-le-v1, 4 regions) and optionally `xor_vectors` (lastdim-u32le-v1) with opguard_bridge's torch path. Writes them with `index_copy_` at row `counter % depth`. |
| Registration | `torch.library.custom_op`, declared to mutate the counter. Dynamo keeps it (a side effect); Inductor calls it as an opaque kernel. |
| CUDA graphs | The row index is computed on the device, so each replay writes its own row. Slots are assigned in Python when the op runs (each eager or compiled call, once at capture). |
| `ps.read()`, `ps.emit(writer)` | One host read. One `TraceWriter` record per step and probed tensor, in call order. `export_pair` and the harness tools read them. |
| `isolate=True` (default) | Under compile, the op gets an exact copy made by a pointwise op (`neg(neg(x))`; integers `x ^ 0`). See "Does the probe change the numerics". |

**Fingerprints equal opguard_bridge's.** Every record was compared with
`opguard_bridge.xor_signature` / `xor_vectors` on the same tensor, with both the
bridge's torch path and OpGuard's own function.

| Run | Records | Mismatching |
|---|---|---|
| eager decode, default model, 3 steps | 99 | 0 (torch), 0 (OpGuard) |
| `torch.compile` default, same | 99 | 0, 0 |
| CUDA graph, 3rd replay, 5 tensors (bf16 3-D, fp32 7×33, fp16 5×3, fp32 0-dim, transposed fp32) | 5 | 0 |

### The decode step

`graph_probe_cases.py` builds a static-shape decode step from the model's
modules: B rows, one token each, KV cache `[L, B, cap, Hk, D]` preallocated,
24 teacher-forced steps from position 0. Probes per layer: `attn_norm`
(input RMSNorm output), `attn_res` (after the attention residual add),
`mlp_norm`, `out` (layer output); then `logits`. Default config: hidden 1024,
8 layers, 16/4 heads, vocab 32000; 33 probes per step, 792 records per run.

Op sets ("kits"):

| Kit | GEMM | Attention | RMSNorm, RoPE, SiLU·mul, add |
|---|---|---|---|
| `stock` | cuBLAS | SDPA over the full cache with a mask | torch ops |
| `bi` | `mm_one_config` (Triton, traced by Dynamo) | the model's paged serving kernel | torch ops (row sum: Triton) |
| `bi_opaque_kernels` | custom op | custom op | torch ops; row sum custom op |
| `bi_opaque_all` | custom op | custom op | each a custom op running the model's module |

## Evidence

### Setup, code and results

- Code: [tools/graph_probe.py](../../tools/graph_probe.py) (the probe) and
  [examples/graph_probe/graph_probe_cases.py](graph_probe_cases.py)
  (the checks). The model is [examples/whole_model/model.py](../whole_model/model.py), imported.
- Results: [results-a40.json](results-a40.json).
- Setup: one NVIDIA A40 (shared with other jobs), driver 550.144.03, torch
  2.5.1+cu121, Triton 3.1.0, bf16. OpGuard from a source checkout (commit
  39601f8 plus local changes) on `PYTHONPATH`; its native XOR extension was not
  built, so OpGuard's own fingerprint function ran its tensor path.

### Results

#### 1. Eager vs compile vs CUDA graph (stock, B = 1)

OpGuard's pivot is from `first_divergence` (trace_summary, first_diff,
inspect_event, compare_events) on the exported pair. It matched the host
comparison of the probe records in every case.

| Mode | Compile (s) | Logits = eager | First differing probe (OpGuard) | Records differing |
|---|---|---|---|---|
| CUDA graph (manual capture) | — | **yes** | none | 0 of 792 |
| compile default | 7.9 | no | `L0.attn_norm`, step 1, all 4 regions | 792 of 792 |
| compile max-autotune-no-cudagraphs | 57.4 | no | `L0.attn_norm`, step 1 | 792 of 792 |
| compile reduce-overhead | 6.6 | no | `L0.attn_norm`, step 1 | 792 of 792 |

- The first RMSNorm already differs, so every later probe differs too. Probes
  locate the first point; they cannot say whether later ops would agree on
  equal inputs (the [paged/compile case](../../handbook/3-kernels/attention.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)
  compiled each op alone for that).

#### 2. CUDA-graph buckets (one request in row 0; batch 1, 8, 16)

| Kit | Bucket 8 vs 1 | Bucket 16 vs 1 |
|---|---|---|
| `stock` | first diff `L0.attn_res` step 1 (o_proj GEMM); `L0.attn_norm` equal in all 24 steps; 768 of 792 records differ | same: `L0.attn_res`, 768 of 792 |
| `bi` | **equal**, all 792 | **equal** |
| `bi_opaque_all` | **equal** | **equal** |

#### 3. Compile = eager with opaque ops

| Kit | compile default | max-autotune-no-cudagraphs | reduce-overhead |
|---|---|---|---|
| `bi` (Triton traced) | no: from `L0.attn_norm`, 792 records | no, same | no, same |
| `bi_opaque_kernels` | no: from `L0.attn_norm` | no | no |
| `bi_opaque_all` | **yes**, 0 of 792 differ, logits equal | **yes** | **yes** |

This agrees with the paged/compile case: compiled equals eager only when every
rounding op is opaque.

#### 4. Does the probe change the numerics

Each compiled run was repeated without probes, with probes, with probes and
`isolate=False`, and without probes again. Logits over 24 steps, bitwise:

| Kit, mode | Probes (isolated) vs none | Probes, not isolated, vs none | No probes twice | Inductor kernels: none / isolated / not isolated |
|---|---|---|---|---|
| stock, compile default | **equal** | **differ** | equal | 75 / 76 / 75 |
| stock, max-autotune-no-cudagraphs | **equal** | **differ** | equal | 1069 / 1071 / 1085 (includes autotune candidates) |
| stock, reduce-overhead | **equal** | **differ** | equal | 75 / 76 / 75 |
| bi_opaque_all, compile default | **equal** | equal | equal | 27 / 60 / 27 |
| stock eager; stock CUDA graph | equal | — | equal | — |

Why the non-isolated probe changed the bits (read from Inductor's generated
code, tiny config): without probes, `s + o_proj(x)` is an `mm` and the add is
fused into the next RMSNorm kernel. A probe on `s` is a non-pointwise user of
the sum. Inductor's post-grad pass then folds the add into `addmm`, so the GEMM
adds the residual before its one rounding. Eager rounds the GEMM output first.
With the isolated copy, the probe consumes the copy, the sum's users stay
pointwise, and Inductor keeps the no-probe plan (one extra kernel in total for
stock). Under `bi_opaque_all` the producers are opaque, so each copy is its own
kernel (33 extra).

#### Cost

Median ms per decode step (steps 3–24 of one run; the GPU was shared, so
repeat runs varied by up to 1.7x, for example 3.36 vs 2.06 ms for the same
compiled step). 33 probes per step, per-vector fingerprints on.

| Kit, mode | No probes | Probes | Per probe |
|---|---|---|---|
| stock, CUDA graph | 1.33 | 3.16 | ~0.055 ms (GPU work only) |
| stock, reduce-overhead | 1.09 | 2.92 | ~0.055 ms |
| stock, eager | 6.17 | 32.5 | ~0.8 ms (Python dispatch) |
| stock, compile default | 2.06–3.36 | 48.7 | ~1.4 ms (Python dispatch) |
| stock, max-autotune-no-cudagraphs | 2.01 | 28.9 | ~0.8 ms |

- Inside a graph the probe costs only its kernels: about log2(numel) XOR-halving
  launches per region and per vector set, plus index writes.
- Outside a graph the custom op's Python body dominates.
- Memory: `depth × max_slots × (4 regions × 4 B + 8 B)` plus `depth × vector_words × 4 B`.
  The runs here used depth 25, 256 slots, 65,536 vector words: 6.7 MB.

#### A side finding: a lazy cache filled inside compiled code

`model.py` builds its RoPE cos/sin tables on first use. When the first use is
inside `torch.compile`, Inductor computes the tables, and the cached result
then serves every later run, eager included. The compiled cos table differed
from the eager one in 178 of 131,072 elements. The checks here build the tables
in eager before any compile. Any engine that fills a cache lazily under compile
has the same hazard.

### What this shows

- A probe in a custom op records fingerprints inside compiled code and graph
  replays. The records are OpGuard's format, and OpGuard's harness finds the
  same first divergence as a direct comparison (all 19 pairs).
- A stock compiled decode step differs from eager at the first RMSNorm. A CUDA
  graph equals eager at the same shape.
- Padding a request into a bigger bucket changes stock bits from the first
  attention residual (the o_proj GEMM). bi kernels give one result across buckets.
  On 8x A16 (sm86, driver 535.288.01, same torch) at quick sizes, the stock
  bucket 16 vs 1 comparison gave equal bits (regression suite, 2026-11-19): the
  A40's stock difference did not reproduce. The
  OpGuard-vs-direct checks passed there too.
- `bi_opaque_all` gives compiled = eager bits through every probe in all three
  compile modes.
- A probe can change what it measures under compile. Isolate it and confirm
  with a with/without comparison.

### Reproduce

    python examples/graph_probe/graph_probe_cases.py --quick --opguard-root ~/OpGuard --json /tmp/q.json   # ~50 s on an A40
    for p in probe modes buckets opaque cost; do
      python examples/graph_probe/graph_probe_cases.py --parts $p --opguard-root ~/OpGuard --work /tmp/gp --json part-$p.json
    done
    python examples/graph_probe/graph_probe_cases.py --merge part-*.json --json results-a40.json

Quick mode: tiny model (hidden 256, 2 layers), 6 steps; eager vs compile vs
graph, buckets for stock and `bi_opaque_all`, `bi_opaque_all` compile = eager,
and the with/without-probe check. Without `--opguard-root` the comparisons are
by host only.

## Limits

- The probe sees values, not kernels: after the first difference, it cannot
  tell which later ops also differ on equal inputs.
- Slot shapes are fixed after the first call; a new shape needs another name.
- Dynamic shapes (`dynamic=True`) were not tested.
- The isolated copy keeps Inductor's plan in these graphs. Other graphs may
  have rewrites that depend on users in other ways; run the with/without check.
- `bi` traced through Dynamo ran with `fullgraph=False`; graph breaks were not counted.
- Records carry outputs only. OpGuard compares by name, call site and order; a
  probe that runs on one side only shows as unaligned.
- Timing on a shared GPU, one run each.

## How it was built (what took more than one try)

- The first compiled check compared probe records with eager-computed values
  and failed: Inductor's `x * 1.5 + 1` uses an FMA. The check now fingerprints
  the tensor the compiled code produced.
- The first opaque compile failed: the model's parameters require grad, so AOT
  autograd asked for a backward of the custom op. The runs now use `no_grad`.
- The with/without-probe check failed at first. A control (no probes, twice)
  was equal, so the probe was the cause. Diffing Inductor's generated code
  showed the `addmm` rewrite.
- Isolation by `clone()` does not work (removed as a no-op), and a dtype view
  of bf16 inside a fused kernel failed to compile in Triton (fp32 vs int16
  operands). Double negation works.
- `torch.compile` filled the model's RoPE table cache from inside the graph,
  which changed later eager runs until the tables were built in eager first.
- Full mode runs one part per process (each under ~2.5 min on the A40 lock) and
  merges the JSON with `--merge`.
