"""graph_probe: OpGuard-format fingerprints recorded from inside compiled code and CUDA graphs.

OpGuard captures eager execution. A serving engine runs torch.compile output and
CUDA graphs, whose kernels differ from eager. This module records fingerprints
from inside those runs, without breaking the graph:

    ps = ProbeSet(depth=64, vectors=True)            # static device buffers
    def step(x):
        ps.step()                                    # once per step: advances the device counter
        h = layer(x)
        ps.probe("layer0.out", h)                    # fingerprint of h, written on the device
        return h
    fn = torch.compile(step)                         # or a torch.cuda.CUDAGraph capture of step
    ...
    with TraceWriter(root) as w:                     # tools/opguard_bridge.py
        ps.emit(w)                                   # one host read, OpGuard-format records

How it works
- ``probe`` calls the custom op ``graph_probe::record`` once per tensor. The op is
  registered with ``torch.library.custom_op`` and declares that it mutates the
  probe set's counter, so Dynamo keeps it in the graph (a side effect, never
  dead code) and Inductor treats it as an opaque call it cannot fuse into.
- The op computes the fingerprints with opguard_bridge's plain-torch path
  (``xor_signature``: flat-u32-le-v1; ``xor_vectors``: lastdim-u32le-v1) and writes
  them with ``index_copy_`` into preallocated buffers at row ``counter % depth``.
  The counter lives on the device, so a CUDA-graph replay writes the row of that
  replay. Each written slot also stores the counter value (its step number).
- Slots (one per probe name, tensor index and call within a step) are assigned
  in Python when the op runs: every eager or compiled call, and once at CUDA-graph
  capture. Their shapes must not change afterwards.
- ``read`` copies the buffers to the host once; ``emit`` writes one TraceWriter
  record per step and slot, in call order, so ``export_pair`` and the harness
  tools (first_diff, compare_events) work on them.

Cost: the op runs about log2(numel) XOR-halving launches per region and per
vector set, plus a few index launches; see examples/graph_probe for measured
numbers. Outside a CUDA graph each probe also costs Python dispatch time.

Does a probe change the numerics? It can. Inductor decides some rewrites by
the users of a value. Measured (torch 2.5.1): ``s + o_proj(x)`` stays an ``mm``
plus an add fused into the next RMSNorm when every user of the sum is
pointwise (read from the generated code). A probe on the sum is a
non-pointwise user, so Inductor's post-grad pass folds the add into ``addmm``,
the GEMM adds the residual before its single rounding, and the bits change. With ``isolate=True`` (default) the op receives an exact copy made by a
pointwise op (negated twice), so the probed value's users stay pointwise.
Check it for your graph: compile with and without probes and compare outputs.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opguard_bridge as ob  # noqa: E402

_SETS: Dict[int, "ProbeSet"] = {}
_CURRENT: List["ProbeSet"] = []


# ----------------------------------------------------------------------------- custom ops
@torch.library.custom_op("graph_probe::step", mutates_args=("counter",))
def _step_op(counter: torch.Tensor, pid: int) -> None:
    _SETS[pid]._occ.clear()
    counter.add_(1)


@_step_op.register_fake
def _(counter, pid):
    return None


@torch.library.custom_op("graph_probe::record", mutates_args=("counter",))
def _record_op(counter: torch.Tensor, x: torch.Tensor, pid: int, name: str, index: int) -> None:
    _SETS[pid]._write(counter, x, name, index)


@_record_op.register_fake
def _(counter, x, pid, name, index):
    return None


def _as_i32(v):
    """uint32 fingerprints (or int64 holding them, torch < 2.3) as int32 bits."""
    if v.dtype == torch.int32:
        return v
    if v.element_size() == 4:
        return v.view(torch.int32)
    return (v - (v >= 2 ** 31).to(v.dtype) * (1 << 32)).to(torch.int32)


def _bit_copy(t):
    """An exact copy of t made by a pointwise op that Inductor fuses into t's producer.
    clone, x * 1 and x + 0 are removed as no-ops before fusion, and a dtype view
    of bf16 inside a fused kernel fails to compile (torch 2.5.1), so floats are
    negated twice (exact, sign of NaN restored) and integers XORed with 0."""
    if t.dtype.is_floating_point:
        return torch.neg(torch.neg(t))
    if t.dtype == torch.bool or t.dtype.is_complex:
        return t
    return torch.bitwise_xor(t, 0)


# ----------------------------------------------------------------------------- probe set
class ProbeSet:
    """Static device buffers for probe fingerprints.

    depth         rows of the ring buffer: the last ``depth`` steps are kept
    max_slots     probed tensors per step (name x tensor index x call)
    regions       regions of the flat fingerprint (OpGuard's default is 4)
    vectors       also record per-vector fingerprints (one per last-dim row)
    vector_words  per-step room for vector fingerprints, in 32-bit words
    isolate       under torch.compile, hand the op an exact pointwise copy of each tensor
                  (negated twice; integers XOR 0) instead of the tensor itself, so every
                  user of the probed value stays pointwise and Inductor's fusion choices
                  do not change (see "Does a probe change the numerics" in the module doc)
    keep          also keep a clone of every probed tensor on the host side of the
                  call (eager and compiled runs only; for checking the probe)
    """

    def __init__(self, device="cuda", depth: int = 64, max_slots: int = 256, regions: int = 4,
                 vectors: bool = False, vector_words: int = 1 << 16, keep: bool = False, isolate: bool = True):
        self.device, self.depth, self.max_slots, self.regions = torch.device(device), depth, max_slots, regions
        self.vectors, self.vector_words, self.keep = vectors, vector_words if vectors else 0, keep
        self.isolate = isolate
        dev = self.device
        self.counter = torch.zeros(1, dtype=torch.int64, device=dev)
        self.sig = torch.zeros(depth * max_slots * regions, dtype=torch.int32, device=dev)
        self.stamp = torch.zeros(depth * max_slots, dtype=torch.int64, device=dev)
        self.arena = torch.zeros(max(1, depth * self.vector_words), dtype=torch.int32, device=dev)
        self.iota = torch.arange(max(regions, self.vector_words), dtype=torch.int64, device=dev)
        for t in (self.counter, self.sig, self.stamp, self.arena, self.iota):
            torch._dynamo.mark_static_address(t)
        self.slots: Dict[Tuple[str, int, int], dict] = {}
        self.order: List[Tuple[str, int, int]] = []
        self._occ: Dict[Tuple[str, int], int] = {}
        self._vused = 0
        self.kept: List[Tuple[int, Tuple[str, int, int], torch.Tensor]] = []
        self.pid = id(self)
        _SETS[self.pid] = self

    # -- inside the computation
    def step(self) -> None:
        """Start a step: the counter (on the device) advances by one."""
        torch.ops.graph_probe.step(self.counter, self.pid)

    def probe(self, name: str, *tensors):
        compiling = self.isolate and torch.compiler.is_compiling()
        for i, t in enumerate(tensors):
            torch.ops.graph_probe.record(self.counter, _bit_copy(t) if compiling else t, self.pid, name, i)
        return tensors[0] if len(tensors) == 1 else tensors

    def _slot(self, key, x) -> dict:
        info = self.slots.get(key)
        if info is None:
            if len(self.order) >= self.max_slots:
                raise RuntimeError(f"graph_probe: more than max_slots={self.max_slots} probes per step")
            layout = ob.vector_layout(x) if self.vectors else {"skipped": "not_requested"}
            info = {"slot": len(self.order), "shape": list(x.shape), "dtype": str(x.dtype),
                    "device": str(x.device), "nbytes": x.numel() * x.element_size(), "layout": layout}
            if "count" in layout:
                if self._vused + layout["count"] > self.vector_words:
                    raise RuntimeError("graph_probe: vector_words too small for the probed tensors")
                info["voff"], self._vused = self._vused, self._vused + layout["count"]
            self.slots[key] = info
            self.order.append(key)
        elif info["shape"] != list(x.shape) or info["dtype"] != str(x.dtype):
            raise RuntimeError(f"graph_probe: {key} changed shape or dtype "
                               f"({info['shape']} {info['dtype']} -> {list(x.shape)} {x.dtype}); use another name")
        return info

    def _write(self, counter, x, name, index) -> None:
        occ = self._occ.get((name, index), 0)
        self._occ[(name, index)] = occ + 1
        key = (name, index, occ)
        info = self._slot(key, x)
        R, S, s = self.regions, self.max_slots, info["slot"]
        row = counter.remainder(self.depth)                                   # [1], on the device
        sig = _as_i32(ob.xor_signature(x, R, impl="torch"))
        self.sig.index_copy_(0, row * (S * R) + s * R + self.iota[:R], sig)
        self.stamp.index_copy_(0, row * S + s, counter)
        if "voff" in info:
            n = info["layout"]["count"]
            vec = ob.xor_vectors(x, impl="torch")
            self.arena.index_copy_(0, row * self.vector_words + info["voff"] + self.iota[:n], vec)
        if self.keep and not torch.cuda.is_current_stream_capturing():
            self.kept.append((key, x.detach().clone()))

    # -- after the run
    def reset(self) -> None:
        """Forget the recorded rows (buffers zeroed, slots kept)."""
        for t in (self.counter, self.sig, self.stamp, self.arena):
            t.zero_()
        self.kept.clear()

    def read(self) -> dict:
        """One host read of every buffer. Returns {"steps": [step, ...], "rows": {step: [(key, sig,
        vectors or None), ...]}, "dropped": steps overwritten in the ring, "slots": ...}."""
        torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
        S, R = self.max_slots, self.regions
        sig = self.sig.view(self.depth, S, R).cpu()
        stamp = self.stamp.view(self.depth, S).cpu()
        arena = self.arena[: self.depth * self.vector_words].view(self.depth, -1).cpu() if self.vectors else None
        total = int(self.counter.item())
        rows: Dict[int, list] = {}
        for r in range(self.depth):
            for key in self.order:
                info = self.slots[key]
                n = int(stamp[r, info["slot"]])
                if n <= 0:
                    continue
                vec = None
                if arena is not None and "voff" in info:
                    vec = arena[r, info["voff"]: info["voff"] + info["layout"]["count"]]
                rows.setdefault(n, []).append((key, sig[r, info["slot"]], vec))
        for n in rows:
            rows[n].sort(key=lambda e: self.slots[e[0]]["slot"])
        return {"steps": sorted(rows), "rows": rows, "counter": total,
                "dropped": max(0, total - self.depth), "slots": dict(self.slots)}

    def records(self, data: Optional[dict] = None) -> List[dict]:
        """Flat list of {"step", "name", "index", "call", "shape", "dtype", "signature", "vectors"}."""
        data = data or self.read()
        out = []
        for n in data["steps"]:
            for key, sig, vec in data["rows"][n]:
                info = self.slots[key]
                out.append({"step": n, "name": key[0], "index": key[1], "call": key[2], "shape": info["shape"],
                            "dtype": info["dtype"], "signature": ob._unsigned_list(sig),
                            "vectors": None if vec is None else ob._unsigned_list(vec)})
        return out

    def emit(self, writer, data: Optional[dict] = None, file: str = "graph_probe") -> int:
        """Write the recorded fingerprints as OpGuard-format records through
        opguard_bridge.TraceWriter: one record per step and probed tensor, named by
        the probe, line = tensor index + 1, in call order. Returns the count."""
        count = 0
        for rec in self.records(data):
            info = self.slots[(rec["name"], rec["index"], rec["call"])]
            lay = info["layout"]
            desc = ob.describe(rec["shape"], rec["dtype"].replace("torch.", ""), info["device"], rec["signature"],
                               vectors=rec["vectors"], vector_bytes=lay.get("vector_bytes"), nbytes=info["nbytes"])
            if self.vectors and rec["vectors"] is None:
                desc["stats"]["xor_vectors_skipped"] = lay.get("skipped")
            writer.record(rec["name"], outputs=[desc], file=file, line=rec["index"] + 1)
            count += 1
        return count

    # -- as the default set
    def __enter__(self):
        _CURRENT.append(self)
        return self

    def __exit__(self, *exc):
        _CURRENT.remove(self)


def probe(name: str, *tensors, ps: Optional[ProbeSet] = None):
    """Record fingerprints of ``tensors`` under ``name`` in ``ps`` (default: the
    innermost ``with ProbeSet(...)``). Returns the tensors unchanged. Without a
    probe set it does nothing, so probes can stay in the code."""
    ps = ps or (_CURRENT[-1] if _CURRENT else None)
    if ps is None:
        return tensors[0] if len(tensors) == 1 else tensors
    return ps.probe(name, *tensors)


def step(ps: Optional[ProbeSet] = None) -> None:
    ps = ps or (_CURRENT[-1] if _CURRENT else None)
    if ps is not None:
        ps.step()


def first_difference(a: List[dict], b: List[dict]) -> Optional[dict]:
    """First record (in order) whose fingerprints differ between two ``records()``
    lists, with the count of differing records and vectors. Records pair by
    (step, name, index, call). A plain host comparison, to cross-check OpGuard."""
    kb = {(r["step"], r["name"], r["index"], r["call"]): r for r in b}
    first, nrec, nvec = None, 0, 0
    for r in a:
        o = kb.get((r["step"], r["name"], r["index"], r["call"]))
        if o is None:
            continue
        if r["signature"] != o["signature"] or r["vectors"] != o["vectors"]:
            nrec += 1
            if r["vectors"] is not None and o["vectors"] is not None:
                nvec += sum(x != y for x, y in zip(r["vectors"], o["vectors"]))
            if first is None:
                first = {"step": r["step"], "name": r["name"], "index": r["index"], "call": r["call"]}
    return None if first is None else {**first, "records_differing": nrec, "vectors_differing": nvec,
                                       "records_compared": len(a)}
