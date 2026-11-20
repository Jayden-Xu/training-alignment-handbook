"""opguard_bridge: the handbook's interface to OpGuard, by reference only.

Nothing here changes OpGuard or copies its logic. Four parts:

a. Fingerprints in OpGuard's formats, in plain torch, on the tensor's device,
   usable where OpGuard is not installed. When ``opguard`` is importable its own
   functions are called instead; the two were measured bit-identical
   (handbook/1-pretraining/localize-with-opguard.md).

       xor_signature(t, regions=4)   uint32[regions]   ``flat-u32-le-v1``
       xor_vectors(t)                int32[count]      ``lastdim-u32le-v1``
       encode_vectors / decode_vectors                 the JSON forms, v1 and legacy
       merge(values, group=...)                        vectors -> rows, regions, whole

b. Thin wrappers over OpGuard's automation protocol (``python -m opguard.tools``,
   docs/AgentTools.md) and the harness CLI (``opguard-harness call``,
   harness/doc/Usage.md): run a reference and a suspect workload through
   discovery, capture and comparison, then collect the facts OpGuard reports
   about the first divergence. Every fact comes from a harness tool.

       og = locate_opguard("~/OpGuard", python="/env/bin/python")
       run = run_pair(reference, suspect, root, og)     # 4 workload launches
       facts = first_divergence(run["reports"][0], root, og)

c. A trace writer: boundary records made outside OpGuard (name, order, shapes,
   dtypes, fingerprints) in the capture layout OpGuard's exporter reads, so that
   ``export_pair`` and the harness tools work on them.

d. ``suggest(facts)``: the operator OpGuard names, mapped to the handbook's
   operator classes and recipes. Leads with reasons, never a verdict.

Run ``python opguard_bridge.py selftest`` for a fingerprint self-check that needs
no OpGuard.
"""
from __future__ import annotations

import argparse
import array
import ast
import base64
import datetime
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

FLAT_SCHEME = "flat-u32-le-v1"
VECTOR_SCHEME = "lastdim-u32le-v1"
VECTOR_ENCODING = "base64-u32le"
MASK32 = 0xFFFFFFFF

# ============================================================================
# a. Fingerprints
# ============================================================================


def _torch():
    import torch
    return torch


@functools.lru_cache(maxsize=1)
def _opguard_xor():
    """OpGuard's own fingerprint module, or None when it cannot be imported."""
    try:
        import torch  # noqa: F401  (its native extension links against torch)
        from opguard.tracer import xor_signature as module
        return module
    except Exception:
        return None


def opguard_available() -> bool:
    return _opguard_xor() is not None


def _use_opguard(impl: str) -> bool:
    if impl not in ("auto", "torch", "opguard"):
        raise ValueError("impl must be 'auto', 'torch' or 'opguard'")
    if impl == "opguard" and _opguard_xor() is None:
        raise RuntimeError("impl='opguard' but opguard cannot be imported")
    return impl == "opguard" or (impl == "auto" and _opguard_xor() is not None)


def flat_word_layout(nbytes: int, regions: int = 4) -> dict:
    """The ``flat-u32-le-v1`` layout: half-open byte ranges of each region.

    The flattened bytes are packed into little-endian 32-bit words, with only the
    last partial word zero-padded. Words are split evenly into ``regions``
    contiguous regions; the first ``words % regions`` regions get one more word.
    """
    if type(regions) is not int or not 1 <= regions <= 64:
        raise ValueError("regions must be an integer between 1 and 64")
    base, rem = divmod((nbytes + 3) // 4, regions)
    ranges, start = [], 0
    for r in range(regions):
        end = start + base + (r < rem)
        ranges.append([min(4 * start, nbytes), min(4 * end, nbytes)])
        start = end
    return {"scheme": FLAT_SCHEME, "regions": regions, "logical_nbytes": nbytes, "byte_ranges": ranges}


def _raw_bytes(t):
    """The tensor's bytes in logical contiguous order, as a 1-D uint8 tensor."""
    torch = _torch()
    if sys.byteorder != "little":
        raise NotImplementedError("fingerprints are defined on little-endian hosts only")
    if t.device.type == "meta":
        raise ValueError("a meta tensor has no data to fingerprint")
    return t.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)


def _fold_xor(words):
    """XOR along dim 1 of an int32 [rows, width] tensor. XOR is exact and
    associative, so any order gives the same bits; this one halves contiguous
    blocks, about log2(width) launches and width/2 temporary words."""
    torch = _torch()
    rows = words.shape[0]
    extra = None
    while words.shape[1] > 1:
        width = words.shape[1]
        half = width // 2
        if width % 2:                                   # the middle column waits
            mid = words[:, half]
            extra = mid if extra is None else torch.bitwise_xor(extra, mid)
        words = torch.bitwise_xor(words[:, :half], words[:, width - half:])
    out = words[:, 0] if words.shape[1] else torch.zeros(rows, dtype=torch.int32, device=words.device)
    return out if extra is None else torch.bitwise_xor(out, extra)


def _as_words(raw):
    """(int32 words of the full 4-byte groups, the zero-padded last word or None)."""
    torch = _torch()
    nbytes = raw.numel()
    full = nbytes // 4
    body = raw[: 4 * full]
    if full and body.storage_offset() % 4:              # an int32 view needs an aligned start
        body = body.clone()
    words = body.view(torch.int32) if full else torch.zeros(0, dtype=torch.int32, device=raw.device)
    tail = None
    if nbytes % 4:
        tail = torch.zeros(4, dtype=torch.uint8, device=raw.device)
        tail[: nbytes % 4] = raw[4 * full:]
        tail = tail.view(torch.int32)
    return words, tail


def _to_u32(values):
    torch = _torch()
    u32 = getattr(torch, "uint32", None)
    if u32 is None:                                     # torch < 2.3: unsigned values in int64
        return values.to(torch.int64) & MASK32
    try:
        return values.view(u32)
    except RuntimeError:
        return values.to(u32)


def _signature_torch(t, regions: int):
    torch = _torch()
    raw = _raw_bytes(t)
    nbytes = raw.numel()
    layout = flat_word_layout(nbytes, regions)
    words, tail = _as_words(raw)
    full = words.numel()
    parts = []
    for lo, hi in layout["byte_ranges"]:
        first, last = lo // 4, (hi + 3) // 4            # word range of this region
        value = _fold_xor(words[first:min(last, full)].unsqueeze(0))
        if tail is not None and hi > lo and last == full + 1:
            value = torch.bitwise_xor(value, tail)
        parts.append(value)
    return _to_u32(torch.cat(parts))


def xor_signature(t, regions: int = 4, impl: str = "auto"):
    """Regional XOR fingerprint of a tensor in the ``flat-u32-le-v1`` layout.

    Returns uint32[regions] on the tensor's device (CPU tensors give CPU values)
    with no host read and no synchronization. ``flat_word_layout`` gives the byte
    range of each region. ``impl``: "auto" calls OpGuard's own function when
    ``opguard`` is importable, "torch" forces the plain-torch path, "opguard"
    requires OpGuard.
    """
    if _use_opguard(impl):
        return _opguard_xor().xor_signature(t, regional=regions, by="words")
    return _signature_torch(t, regions)


def vector_layout(t) -> dict:
    """``lastdim-u32le-v1`` layout of a tensor: one vector per last-dimension row.

    Returns {"scheme", "vector_bytes", "count"}; when the format defines no
    vectors for this tensor, {"scheme", "skipped": reason} instead (reasons as
    OpGuard records them in ``xor_vectors_skipped``). A 0-dim tensor is one
    vector of one element.
    """
    if t.device.type == "meta":
        return {"scheme": VECTOR_SCHEME, "skipped": "meta"}
    if t.numel() == 0:
        return {"scheme": VECTOR_SCHEME, "skipped": "empty"}
    width = t.shape[-1] if t.dim() else 1
    vector_bytes = width * t.element_size()
    if vector_bytes % 4:
        return {"scheme": VECTOR_SCHEME, "skipped": "vector_bytes_not_multiple_of_4"}
    return {"scheme": VECTOR_SCHEME, "vector_bytes": vector_bytes, "count": t.numel() // width}


def xor_vectors(t, impl: str = "auto"):
    """Per-vector XOR fingerprints in ``lastdim-u32le-v1``: one value per
    last-dimension vector, in flattened order.

    Returns int32[count] on the tensor's device whose bits are the unsigned
    32-bit fingerprints (mask with 0xFFFFFFFF on the host, or use
    ``encode_vectors``). Returns None when the format defines no vectors for
    this tensor; ``vector_layout(t)["skipped"]`` says why.
    """
    layout = vector_layout(t)
    if "skipped" in layout:
        return None
    if _use_opguard(impl):
        return _opguard_xor().xor_chunk_signature(t, layout["vector_bytes"])
    words, _ = _as_words(_raw_bytes(t))
    return _fold_xor(words.view(layout["count"], layout["vector_bytes"] // 4))


def _unsigned_list(values) -> List[int]:
    """Host list of unsigned 32-bit ints from a tensor, a sequence or a scalar."""
    if hasattr(values, "detach"):                       # a tensor
        values = values.detach().reshape(-1).cpu().tolist()
    elif isinstance(values, (int, float)):
        values = [values]
    out = []
    for v in values:
        if isinstance(v, float):
            if v != int(v):
                raise ValueError(f"fingerprint is not an integer: {v!r}")
            v = int(v)
        out.append(int(v) & MASK32)
    return out


def encode_vectors(values, vector_bytes: int) -> dict:
    """The ``xor_vectors`` JSON object of docs/RLAlignment.md for these
    fingerprints (a tensor from ``xor_vectors`` or a sequence of ints)."""
    if type(vector_bytes) is not int or vector_bytes < 4 or vector_bytes % 4:
        raise ValueError("vector_bytes must be a positive multiple of 4")
    packed = array.array("I", _unsigned_list(values))
    if packed.itemsize != 4:
        raise NotImplementedError("needs a 4-byte unsigned int array type")
    if sys.byteorder != "little":
        packed.byteswap()
    return {"scheme": VECTOR_SCHEME, "vector_bytes": vector_bytes, "count": len(packed),
            "encoding": VECTOR_ENCODING, "data": base64.b64encode(packed.tobytes()).decode("ascii")}


def decode_vectors(obj) -> List[int]:
    """Unsigned 32-bit vector fingerprints from any recorded form:

    - the v1 ``xor_vectors`` object ({"scheme": "lastdim-u32le-v1", ...});
    - legacy ``xor_rows``: a list of signed int32 decimals, or one scalar for a
      single-vector tensor (OpGuard writes that one as a float);
    - a tensor ``stats`` dict holding either of them.
    Every value is masked with 0xFFFFFFFF.
    """
    if isinstance(obj, dict):
        if "scheme" not in obj:
            for key in ("xor_vectors", "xor_rows"):
                if key in obj:
                    return decode_vectors(obj[key])
            raise ValueError("no xor_vectors or xor_rows in this object")
        if obj.get("scheme") != VECTOR_SCHEME or obj.get("encoding") != VECTOR_ENCODING:
            raise ValueError(f"unsupported vector fingerprint object: {obj.get('scheme')!r}/{obj.get('encoding')!r}")
        if not isinstance(obj.get("data"), str):
            raise ValueError("xor_vectors.data must be a base64 string")
        raw = base64.b64decode(obj["data"], validate=True)
        if len(raw) % 4 or len(raw) // 4 != obj.get("count"):
            raise ValueError("xor_vectors.data does not hold `count` 32-bit values")
        unpacked = array.array("I")
        unpacked.frombytes(raw)
        if sys.byteorder != "little":
            unpacked.byteswap()
        return list(unpacked)
    if isinstance(obj, (list, tuple, int, float)) and not isinstance(obj, bool):
        return _unsigned_list(obj)
    raise ValueError(f"cannot decode vector fingerprints from {type(obj).__name__}")


def merge(values, group: Optional[int] = None, *, regions: Optional[int] = None,
          vector_bytes: Optional[int] = None):
    """XOR consecutive vector fingerprints into coarser fingerprints.

    merge(v, group=k)                    rows of k consecutive vectors, e.g. the
                                         heads of one token -> that token's row
    merge(v, regions=R, vector_bytes=b)  the R ``flat-u32-le-v1`` regions of the
                                         whole tensor, comparable to
                                         ``xor_signature(t, R)``; raises when a
                                         region boundary cuts through a vector
    merge(v)                             the whole tensor (one value)

    A tensor comes back as an int32 tensor on its device, a sequence as a list
    of unsigned ints. XOR of a byte range's vectors equals the flat XOR of that
    range, so the result can be checked against ``xor_signature``.
    """
    is_tensor = hasattr(values, "detach")
    count = values.numel() if is_tensor else len(values)
    if group is not None and regions is not None:
        raise ValueError("give group or regions, not both")
    if regions is not None:
        if vector_bytes is None or vector_bytes % 4:
            raise ValueError("regions needs vector_bytes, a multiple of 4")
        layout = flat_word_layout(count * vector_bytes, regions)
        bounds = []
        for lo, hi in layout["byte_ranges"]:
            if lo % vector_bytes or hi % vector_bytes:
                raise ValueError(f"region [{lo}, {hi}) does not fall on vector boundaries "
                                 f"({vector_bytes} bytes each): merge with group= instead")
            bounds.append((lo // vector_bytes, hi // vector_bytes))
    else:
        size = count if group is None else group
        if size <= 0 or (count % size if count else 0):
            raise ValueError(f"{count} vectors do not split into groups of {size}")
        bounds = [(i, i + size) for i in range(0, count, size)] if count else ([(0, 0)] if group is None else [])
    if is_tensor:
        torch = _torch()
        flat = values.detach().reshape(-1)
        if flat.dtype != torch.int32:
            flat = flat.view(torch.int32) if flat.element_size() == 4 else (flat.to(torch.int64) & MASK32).to(torch.int32)
        if regions is None and count:
            return _fold_xor(flat.view(len(bounds), -1))
        parts = [_fold_xor(flat[lo:hi].unsqueeze(0)) for lo, hi in bounds]
        return torch.cat(parts) if parts else flat[:0]
    flat = _unsigned_list(values)
    out = []
    for lo, hi in bounds:
        acc = 0
        for v in flat[lo:hi]:
            acc ^= v
        out.append(acc)
    return out


def _signed32(v: int) -> int:
    v &= MASK32
    return v - (1 << 32) if v & 0x80000000 else v


def describe(shape: Sequence[int], dtype, device, signature, *, vectors=None,
             vector_bytes: Optional[int] = None, nbytes: Optional[int] = None) -> dict:
    """A tensor summary in the form OpGuard's captures hold, from fingerprints
    computed earlier (for example inside compiled code): ``signature`` is the
    result of ``xor_signature`` or its values, ``vectors`` that of ``xor_vectors``.
    Reading the fingerprints here is the only host transfer."""
    sig = _unsigned_list(signature)
    shape = [int(s) for s in shape]
    numel = 1
    for s in shape:
        numel *= s
    if nbytes is None:
        nbytes = numel * _element_size(dtype)
    stats = {"dtype": str(dtype), "device": str(device), "numel": numel, "xor_signature": sig,
             "xor_sig_mode": "bytes", "xor_sig_layout": flat_word_layout(nbytes, len(sig))}
    if vectors is not None:
        if vector_bytes is None:
            raise ValueError("vectors needs vector_bytes")
        values = _unsigned_list(vectors)
        # Today's OpGuard writes and reads the legacy pair; the v1 object is its draft format.
        stats["xor_rows"] = [_signed32(v) for v in values]
        stats["xor_row_bytes"] = vector_bytes
        stats["xor_vectors"] = encode_vectors(values, vector_bytes)
    return {"shape": shape, "stats": stats}


def _element_size(dtype) -> int:
    torch = _torch()
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.replace("torch.", ""))
    return torch.empty((), dtype=dtype).element_size()


def summarize(t, vectors: bool = False, regions: int = 4, impl: str = "auto") -> dict:
    """``describe`` for a live tensor: computes the fingerprints and reads them."""
    layout = vector_layout(t) if vectors else {"skipped": "not_requested"}
    vec = xor_vectors(t, impl) if "skipped" not in layout else None
    out = describe(t.shape, t.dtype, t.device, xor_signature(t, regions, impl), vectors=vec,
                   vector_bytes=layout.get("vector_bytes"), nbytes=t.numel() * t.element_size())
    if vectors and vec is None:
        out["stats"]["xor_vectors_skipped"] = layout["skipped"]
    return out


def reference_signature(data: bytes, regions: int = 4) -> List[int]:
    """The ``flat-u32-le-v1`` fingerprint of raw bytes in pure Python: the format
    as documented, with no torch code shared with the functions above."""
    layout = flat_word_layout(len(data), regions)
    padded = data + b"\0" * (-len(data) % 4)
    words = array.array("I")
    words.frombytes(padded)
    if sys.byteorder != "little":
        words.byteswap()
    out = []
    for lo, hi in layout["byte_ranges"]:
        acc = 0
        for w in words[lo // 4:(hi + 3) // 4] if hi > lo else ():
            acc ^= w
        out.append(acc)
    return out


def reference_vectors(data: bytes, vector_bytes: int) -> List[int]:
    """Per-vector fingerprints of raw bytes in pure Python (see ``reference_signature``)."""
    return [reference_signature(data[i:i + vector_bytes], 1)[0] for i in range(0, len(data), vector_bytes)]


# ============================================================================
# b. OpGuard's automation protocol and harness CLI
# ============================================================================


class OpGuardError(RuntimeError):
    pass


class OpGuardEnv(dict):
    """Where OpGuard lives: {"python", "root", "pythonpath", "backend", "harness"}."""


def locate_opguard(root: Optional[str] = None, python: Optional[str] = None) -> OpGuardEnv:
    """Find a runnable OpGuard and its harness without installing anything.

    ``root`` is a source checkout (default: $OPGUARD_ROOT); leave it out when
    ``opguard`` and ``opguard-harness`` are installed in ``python``'s
    environment. ``python`` is the training interpreter (default: this one).
    """
    python = python or os.environ.get("OPGUARD_PYTHON") or sys.executable
    root = root or os.environ.get("OPGUARD_ROOT")
    paths: List[str] = []
    if root:
        base = Path(root).expanduser().resolve()
        if not (base / "opguard" / "tools" / "__main__.py").is_file():
            raise OpGuardError(f"no OpGuard checkout at {base} (opguard/tools/__main__.py missing)")
        paths = [str(base), str(base / "harness")]
        root = str(base)
    env = _child_env(paths)
    probe = subprocess.run([python, "-m", "opguard.tools", "capabilities"], env=env,
                           capture_output=True, text=True)
    try:
        caps = json.loads(probe.stdout)["result"]
    except Exception:
        raise OpGuardError(f"`{python} -m opguard.tools capabilities` failed: "
                           f"{(probe.stderr or probe.stdout).strip()[-400:]}")
    harness = [python, "-m", "opguard_harness"]
    check = subprocess.run(harness + ["tools"], env=env, capture_output=True, text=True)
    if check.returncode:
        exe = shutil.which("opguard-harness")
        if not exe:
            raise OpGuardError("the OpGuard harness is neither importable nor on PATH")
        harness = [exe]
    return OpGuardEnv(python=python, root=root, pythonpath=paths, backend=[python, "-m", "opguard.tools"],
                      harness=harness, capabilities=caps)


def _child_env(pythonpath: Sequence[str]) -> dict:
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(list(pythonpath) + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _run_json(command: List[str], payload: Optional[dict], og: OpGuardEnv, timeout: Optional[float]):
    with tempfile.TemporaryDirectory(prefix="opg-bridge-") as scratch:
        if payload is not None:
            path = Path(scratch) / "arguments.json"
            path.write_text(json.dumps(payload))
            command = command + [str(path)]
        done = subprocess.run(command, env=_child_env(og["pythonpath"]), capture_output=True,
                              text=True, timeout=timeout)
    try:
        return json.loads(done.stdout), done
    except ValueError:
        raise OpGuardError(f"{' '.join(command[:6])} printed no JSON: {(done.stderr or done.stdout).strip()[-600:]}")


def backend(action: str, request: Optional[dict] = None, og: Optional[OpGuardEnv] = None,
            timeout: Optional[float] = None) -> dict:
    """One call of the automation protocol v1: ``python -m opguard.tools ACTION
    --request FILE``. Returns ``result``; raises OpGuardError on ``ok: false``."""
    og = og or locate_opguard()
    command = og["backend"] + [action]
    if request is not None:
        request = {"protocol_version": 1, **request}
        command += ["--request"]
    response, _ = _run_json(command, request, og, timeout)
    if response.get("protocol_version") != 1 or response.get("ok") is not True:
        raise OpGuardError(f"backend {action} failed: {response.get('error', response)}")
    return response["result"]


def harness(tool: str, arguments: dict, root: Union[str, Path], og: Optional[OpGuardEnv] = None,
            timeout: Optional[float] = None) -> dict:
    """One harness tool call: ``opguard-harness --root ROOT call TOOL --arguments
    FILE``. ``root`` is the harness's experiment and review store."""
    og = og or locate_opguard()
    command = og["harness"] + ["--root", str(root), "call", tool, "--arguments"]
    response, _ = _run_json(command, arguments, og, timeout)
    if response.get("ok") is not True:
        raise OpGuardError(f"harness {tool} failed: {response.get('error', response)}")
    return response["result"]


def workload(cwd: Union[str, Path], script: str, args: Sequence[str] = (), *, og: Optional[OpGuardEnv] = None,
             python: Optional[str] = None, launcher: Optional[Sequence[str]] = None,
             env: Optional[Dict[str, str]] = None, ranks: Sequence[int] = (0,),
             pythonpath: Sequence[str] = ()) -> dict:
    """A workload entry of a harness experiment spec. OpGuard's source checkout
    (if it is one) and ``pythonpath`` go on the workload's PYTHONPATH. Native
    call-site backends are off unless ``env`` turns them on."""
    paths = list((og or {}).get("pythonpath", [])[:1]) + [str(p) for p in pythonpath]
    merged = {"OPGUARD_CALLSITE_CUPTI": "0", "OPGUARD_CALLSITE_CUDAAPI": "0"}
    if paths:
        merged["PYTHONPATH"] = os.pathsep.join(paths)
    merged.update(env or {})
    return {"cwd": str(Path(cwd).resolve()), "script": script, "args": [str(a) for a in args],
            "launcher": list(launcher or [python or (og or {}).get("python") or sys.executable]),
            "env": merged, "ranks": sorted(int(r) for r in ranks)}


def stage_checkouts(source: Union[str, Path], dest: Union[str, Path],
                    names: Sequence[str] = ("reference", "suspect")) -> List[Path]:
    """Copy a workload's source into dedicated checkouts, one per side. OpGuard
    edits the source it captures and restores it afterwards; the harness wants a
    separate checkout for each side."""
    out = []
    for name in names:
        target = Path(dest) / name
        if target.exists():
            raise FileExistsError(f"{target} exists: use a fresh directory")
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", ".opguard", "*.pyc"))
        out.append(target)
    return out


def run_pair(reference: dict, suspect: dict, root: Union[str, Path], og: Optional[OpGuardEnv] = None, *,
             local_root: Optional[str] = None, timeout: float = 1800, label: Optional[str] = None,
             max_stage_attempts: int = 1, progress=None) -> dict:
    """Run both workloads through the harness's fixed workflow:
    baseline discover, prepare, trace; candidate the same; compare.

    Returns what OpGuard reports plus what it cost here: the per-rank comparison
    exports (``reports``), each stage with its wall time, the number of workload
    launches (discovery and capture launch the workload once each, per side) and
    the total wall time. ``verdict`` is OpGuard's: always "not_evaluated".
    A failed stage is returned with ``status: "failed"`` and its error, not raised.
    """
    og = og or locate_opguard()
    root = Path(root).resolve()
    store = root / "experiments"
    spec = {"baseline": reference, "candidate": suspect, "backend": og["backend"],
            "local_root": str(local_root or root / "local"), "timeout_seconds": timeout,
            "max_stage_attempts": max_stage_attempts}
    if label:
        spec["label"] = label
    t_all = time.time()
    status = harness("create_experiment", {"spec": spec}, store, og)
    eid, stages, errors = status["id"], [], []
    while status["next"] is not None and status["status"] != "failed":
        step, t0 = status["next"], time.time()
        status = harness("advance_experiment", {"experiment_id": eid}, store, og, timeout=timeout + 600)
        record = next(s for s in status["stages"] if (s["side"], s["action"]) == (step["side"], step["action"]))
        attempt = record["attempts"][-1]
        stages.append({"side": step["side"], "action": step["action"], "status": record["status"],
                       "seconds": round(time.time() - t0, 2), "directory": attempt["directory"]})
        if attempt.get("error"):
            errors.append({"side": step["side"], "action": step["action"], "error": attempt["error"],
                           "logs": attempt["directory"]})
        if progress:
            progress(stages[-1])
    out = {"experiment_id": eid, "root": str(root), "store": str(store), "status": status["status"],
           "stages": stages, "errors": errors, "verdict": "not_evaluated",
           "launches": sum(1 for s in stages if s["action"] in ("discover", "trace")),
           "wall_seconds": round(time.time() - t_all, 2), "reports": [], "ranks": reference["ranks"]}
    if status["status"] == "succeeded":
        results = harness("experiment_results", {"experiment_id": eid}, store, og)
        out["reports"] = results["comparison"]["reports"]
        out["verdict"] = results["verdict"]
        out["coverage_claim"] = results["comparison"].get("coverage")
    return out


def _plan_entries(text: str) -> List[dict]:
    """(file, line, function) of each planned call site in a capture.yaml."""
    try:
        import yaml
        return [{"file": e.get("file_path"), "line": e.get("line_num"), "function": e.get("function")}
                for e in (yaml.safe_load(text) or {}).get("instrumentations", [])]
    except ImportError:
        entries, cur = [], {}
        for line in text.splitlines():
            m = re.match(r"^(-| ) (file_path|line_num|function): (.*)$", line)
            if line.startswith("- ") and cur:
                entries.append(cur)
                cur = {}
            if m:
                key = {"file_path": "file", "line_num": "line", "function": "function"}[m.group(2)]
                cur[key] = int(m.group(3)) if key == "line" else m.group(3).strip().strip("'\"")
        return entries + ([cur] if cur else [])


def coverage(run: dict) -> dict:
    """What each side's capture covers, read from OpGuard's own artifacts of that
    run: call sites discovered in the workload's checkout, those planned for
    instrumentation, and those actually wrapped. A discovered line that is not
    planned or not wrapped is device work with no boundary record."""
    out = {}
    for side in ("baseline", "candidate"):
        dirs = {s["action"]: Path(s["directory"]) for s in run["stages"] if s["side"] == side}
        if "prepare" not in dirs:
            continue
        request = json.loads((dirs["prepare"] / "request.json").read_text())
        cwd = request["workload"]["cwd"].rstrip("/") + "/"
        found = {}
        for path in (dirs["prepare"] / "output").glob("*_combined.json"):
            for key, info in json.loads(path.read_text()).items():
                m = re.match(r"(.+):(\d+) in (.+)", key)
                if m and m.group(1).startswith(cwd):
                    found[(m.group(1)[len(cwd):], int(m.group(2)))] = sorted(info.get("ops", []))
        plan_file = dirs["prepare"] / "output" / "capture.yaml"
        planned = {(e["file"][len(cwd):] if str(e["file"]).startswith(cwd) else e["file"], e["line"])
                   for e in _plan_entries(plan_file.read_text())} if plan_file.is_file() else set()
        wrapped = None
        if "trace" in dirs:
            report = dirs["trace"] / "output-instrument" / "instrument_report.json"
            if report.is_file():
                wrapped = json.loads(report.read_text()).get("wrapped_count")
        not_planned = sorted(k for k in found if k not in planned)
        out[side] = {"checkout": cwd, "discovered_lines": len(found), "planned_lines": len(planned),
                     "wrapped_calls": wrapped,
                     "discovered_not_planned": [{"file": f, "line": n, "ops": found[(f, n)][:6]}
                                                for f, n in not_planned]}
    return out


_COMPACT = ("event_ref", "name", "fn", "file", "line", "counter", "status", "run", "rank")


def _compact(event: Optional[dict]) -> Optional[dict]:
    if not event:
        return None
    out = {k: event.get(k) for k in _COMPACT}
    corr = event.get("correspondence") or {}
    out["pairing"] = corr.get("basis") if corr.get("state") == "paired" else corr.get("reason")
    out["paired"] = corr.get("state") == "paired"
    return out


def _slots(io: Optional[dict]) -> dict:
    def one(t):
        return {"shape": t.get("shape"), "dtype": t.get("dtype"), "xor_signature": t.get("xor_signature")}
    io = io or {}
    return {"inputs": [one(t) for t in io.get("inputs", [])], "outputs": [one(t) for t in io.get("outputs", [])]}


def _source_line(path: Optional[str], line: Optional[int]) -> dict:
    """The original source line of a call site and its enclosing definition, read
    from the workload checkout (the harness restores it after capture)."""
    if not path or not line or not os.path.isfile(path):
        return {}
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()
    out = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
           "text": lines[line - 1].strip() if 0 < line <= len(lines) else None, "enclosing": None}
    try:
        best: Tuple[int, str] = (10 ** 9, "")

        def visit(node, names):
            nonlocal best
            named = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            names = names + [node.name] if named else names
            if named and node.lineno <= line <= (node.end_lineno or node.lineno):
                span = (node.end_lineno or node.lineno) - node.lineno
                if span <= best[0]:
                    best = (span, ".".join(names))
            for child in ast.iter_child_nodes(node):
                visit(child, names)
        visit(ast.parse(text), [])
        out["enclosing"] = best[1] or None
    except SyntaxError:
        pass
    return out


def _walk(artifact: str, start_ref: str, store, og, limit: int = 4000) -> List[dict]:
    """Every operation of one run and rank in export order, through repeated
    ``trace_neighbors`` queries (41 events each)."""
    seen: Dict[str, dict] = {}
    order: List[str] = []

    def fetch(ref):
        got = harness("trace_neighbors", {"artifact": artifact, "event_ref": ref, "radius": 20}, store, og)["events"]
        fresh = [e for e in got if e["event_ref"] not in seen]
        for e in got:
            seen.setdefault(e["event_ref"], e)
        return got, fresh
    got, _ = fetch(start_ref)
    order = [e["event_ref"] for e in got]
    while len(order) < limit:                           # extend to the right
        got, fresh = fetch(order[-1])
        tail = [e["event_ref"] for e in got]
        new = tail[tail.index(order[-1]) + 1:]
        if not new:
            break
        order += new
    while len(order) < limit:                           # extend to the left
        got, fresh = fetch(order[0])
        head = [e["event_ref"] for e in got]
        new = head[:head.index(order[0])]
        if not new:
            break
        order = new + order
    return [seen[r] for r in order]


def first_divergence(artifact: Union[str, Path], root: Union[str, Path], og: Optional[OpGuardEnv] = None, *,
                     rank: Optional[int] = None, walk: bool = True) -> dict:
    """The facts OpGuard's harness reports about the first divergence in one
    comparison export. Tools used: trace_summary, first_diff, inspect_event,
    compare_events and, with ``walk``, trace_neighbors.

    Nothing is judged here. ``diverged`` only says whether the exporter labelled
    any paired event as different; "no diff" is not proof of equivalence, and
    equal fingerprints are not proof of equal tensors. Fields:

    status_counts     events per run and status (ok, diff, unaligned, anomaly)
    pivot             the reference side's first observed diff: operator name,
                      file, line, counter, which recorded fields differ, input
                      and output summaries on both sides, per-region comparison,
                      the source line and its enclosing definition, callers
    previous_ok       the last event before it that the exporter labelled ok
    unaligned_before  events before the pivot with no partner on the other side
    unaligned_sites   call sites that exist on one side only
    site_statuses     the status of every call at the pivot's call site, in
                      order ("o" ok, "d" diff, "u" unaligned): which calls agree
    sites             the same string for every call site of the reference run
    """
    og = og or locate_opguard()
    artifact, store = str(Path(artifact).resolve()), Path(root).resolve() / "experiments"
    summary = harness("trace_summary", {"artifact": artifact}, store, og)
    counts: Dict[str, Dict[str, int]] = {}
    for item in summary["status_counts"]:
        if rank is None or item["rank"] == rank:
            run_counts = counts.setdefault(item["run"], {})
            run_counts[item["status"]] = run_counts.get(item["status"], 0) + item["count"]
    facts: Dict[str, Any] = {"schema": "opguard-first-divergence-v1", "artifact": summary["artifact"],
                             "verdict": "not_evaluated", "operation_count": summary["operation_count"],
                             "status_counts": counts, "alignment_policy": (summary.get("alignment_policy") or {}).get("method"),
                             "diverged": False, "pivot": None, "tools": ["trace_summary", "first_diff"]}
    bounds = [b for b in harness("first_diff", {"artifact": artifact}, store, og)["boundaries"]
              if rank is None or b["rank"] == rank]
    ref_side = sorted((b for b in bounds if b["run"] == "run1" and b["first_observed_diff"]),
                      key=lambda b: (b["rank"], b["first_observed_diff"].get("counter") or 0))
    facts["unaligned_first"] = {b["run"]: _compact(b["first_unaligned"]) for b in bounds if b.get("first_unaligned")}
    if not ref_side:
        facts["note"] = ("No paired event is labelled different. That is not proof of equivalence: "
                         "check coverage and the unaligned events.")
        return facts
    bound = ref_side[0]
    first = bound["first_observed_diff"]
    facts.update(diverged=True, rank=bound["rank"], unaligned_before=bound["unaligned_before"],
                 previous_ok=_compact(bound["previous_exporter_ok"]))
    seen = harness("inspect_event", {"artifact": artifact, "event_ref": first["event_ref"]}, store, og)
    facts["tools"] += ["inspect_event"]
    peer = (seen["alignment"]["peers"] or [None])[0]
    pivot = _compact(first)
    pivot.update(diff_fields=(seen.get("diff") or {}).get("fields", []),
                 diff_summary=(seen.get("diff") or {}).get("summary"),
                 reference=_slots(seen.get("io")), suspect=None, peer=_compact(peer), regional=None,
                 callers=[{"function": f.get("function"), "file": f.get("source"), "line": f.get("source_line")}
                          for f in seen.get("source_context", [])][-6:],
                 source=_source_line(first.get("file"), first.get("line")))
    if peer:
        both = harness("compare_events", {"artifact": artifact, "left_ref": first["event_ref"],
                                          "right_ref": peer["event_ref"]}, store, og)
        facts["tools"] += ["compare_events"]
        pivot["suspect"] = _slots(both.get("right_io"))
        pivot["regional"] = [{"slot": t["left_tensor"], "status": t["status"],
                              "differing_regions": t.get("differing_regions"),
                              "byte_ranges": [r["byte_range"] for r in t.get("regions", [])
                                              if not r["fingerprints_agree"]]}
                             for t in both["regional_comparison"]["tensors"]]

        def differs(kind):
            a, b = pivot["reference"][kind], pivot["suspect"][kind]
            return len(a) != len(b) or any(x != y for x, y in zip(a, b))
        pivot.update(inputs_recorded=len(pivot["reference"]["inputs"]),
                     inputs_differ=differs("inputs"), outputs_differ=differs("outputs"),
                     dtype_differs=any(x.get("dtype") != y.get("dtype")
                                       for k in ("inputs", "outputs")
                                       for x, y in zip(pivot["reference"][k], pivot["suspect"][k])),
                     shape_differs=any(x.get("shape") != y.get("shape")
                                       for k in ("inputs", "outputs")
                                       for x, y in zip(pivot["reference"][k], pivot["suspect"][k])))
    facts["pivot"] = pivot
    if walk:
        facts["tools"] += ["trace_neighbors"]
        site = (first.get("fn"), first.get("file"), first.get("line"))
        events = _walk(artifact, first["event_ref"], store, og)
        at_site = [e for e in events if (e.get("fn"), e.get("file"), e.get("line")) == site]
        index = [e["event_ref"] for e in events].index(first["event_ref"])
        facts["site_statuses"] = {"calls": len(at_site),
                                  "statuses": "".join({"ok": "o", "diff": "d", "unaligned": "u"}.get(e["status"], "?")
                                                      for e in at_site),
                                  "pivot_call": [e["event_ref"] for e in at_site].index(first["event_ref"])}
        facts["preceding"] = [_compact(e) for e in events[max(0, index - 6):index]]
        facts["events_in_run"] = len(events)
        table: Dict[Tuple, List[str]] = {}
        for e in events:                                # every call site, in order of first use
            key = (e.get("fn"), os.path.basename(str(e.get("file"))), e.get("line"))
            table.setdefault(key, []).append({"ok": "o", "diff": "d", "unaligned": "u"}.get(e["status"], "?"))
        facts["sites"] = [{"fn": k[0], "file": k[1], "line": k[2], "calls": len(v), "statuses": "".join(v)[:600]}
                          for k, v in table.items()]
        lone: Dict[Tuple, int] = {}
        for e in events:
            if e["status"] == "unaligned":
                key = (e.get("fn"), os.path.basename(str(e.get("file"))), e.get("line"),
                       (e.get("correspondence") or {}).get("reason"))
                lone[key] = lone.get(key, 0) + 1
        other = facts["unaligned_first"].get("run2") or (peer if counts.get("run2", {}).get("unaligned") else None)
        if other:                                       # the suspect's side, same walk
            for e in _walk(artifact, other["event_ref"], store, og):
                if e["status"] == "unaligned":
                    key = (e.get("fn"), os.path.basename(str(e.get("file"))), e.get("line"),
                           (e.get("correspondence") or {}).get("reason"), "run2")
                    lone[key] = lone.get(key, 0) + 1
        facts["unaligned_sites"] = [{"fn": k[0], "file": k[1], "line": k[2], "reason": k[3],
                                     "run": "run2" if len(k) > 4 else "run1", "events": n}
                                    for k, n in sorted(lone.items(), key=lambda kv: str(kv[0]))]
    return facts


# ============================================================================
# c. Trace writer
# ============================================================================


def _sanitize(name: str) -> str:
    """The directory name OpGuard gives a function's records (same rule as its
    writer, so files written here pair with files it captured)."""
    if not name:
        return "unknown"
    s = name.replace("<", "").replace(">", "")
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s)
    s = re.sub(r"[._]{2,}", "_", s).strip("._")
    if len(s) > 200:
        s = s[:190] + "_" + hashlib.md5(name.encode()).hexdigest()[:8]
    return s or "unknown"


class TraceWriter:
    """Write boundary records made outside OpGuard in its capture layout:

        ROOT/exec_trace/gpu_<rank>/ops/<name>/<name>_line<line>.jsonl

    one JSON record per line with the fields OpGuard's exporter aligns and
    compares: function, file, line, counter (the order), inputs and outputs as
    tensor summaries (shape, dtype, device, fingerprints). ``export_pair`` turns
    two such directories into the comparison export the harness tools read.

        w = TraceWriter("/runs/a")
        w.record("block0.norm", inputs=[x], outputs=[y])            # live tensors
        w.record("block0.attn", inputs=[describe(shape, dtype, dev, sig)], outputs=[...])
        w.close()

    Two records pair when name, file, line, rank and ``meta`` agree and they come
    in the same order at that call site. The exporter compares shape, dtype,
    device and the regional fingerprint; per-vector fingerprints ride along as
    ``xor_rows`` / ``xor_vectors`` for tools that read them.
    """

    def __init__(self, root: Union[str, Path], rank: int = 0, vectors: bool = False,
                 regions: int = 4, impl: str = "auto"):
        self.root, self.rank = Path(root), int(rank)
        self.vectors, self.regions, self.impl = vectors, regions, impl
        self.dir = self.root / "exec_trace" / f"gpu_{self.rank:03d}"
        self.counter, self._files, self._lock = 0, {}, threading.Lock()

    def _summary(self, value):
        if hasattr(value, "detach") and hasattr(value, "dtype"):
            return summarize(value, self.vectors, self.regions, self.impl)
        if isinstance(value, (list, tuple)):
            return [self._summary(v) for v in value]
        if isinstance(value, dict) and "stats" not in value:
            return {k: self._summary(v) for k, v in value.items()}
        return value

    def record(self, name: str, inputs: Sequence = (), outputs: Sequence = (), *, kwargs: Optional[dict] = None,
               file: str = "external", line: int = 1, meta: Optional[dict] = None,
               stack: Optional[str] = None, counter: Optional[int] = None) -> int:
        """Append one boundary record. ``inputs`` / ``outputs`` hold tensors (they
        are fingerprinted now), summaries from ``describe``, or plain JSON values.
        ``counter`` is the global order; by default records are numbered as written."""
        if not name or line < 1:
            raise ValueError("a record needs a name and a line >= 1")
        function = name if name.startswith("<") else f"<{name}>"
        outs = [self._summary(o) for o in outputs]
        with self._lock:
            self.counter = counter if counter is not None else self.counter + 1
            record = {
                "counter": self.counter, "pid": os.getpid(), "thread_id": threading.get_ident(),
                "timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"), "function": function,
                "inputs": {"self": None, "args": [function] + [self._summary(i) for i in inputs],
                           "kwargs": self._summary(kwargs or {})},
                "outputs": outs[0] if len(outs) == 1 else (outs or None),
                "is_async": False, "meta": {"lineno": int(line), **(meta or {})}, "line_num": int(line),
                "dump_mode": "summary", "fname": file, "tag_prefix": file, "gpu_rank": self.rank,
            }
            if stack:
                record["stack_trace"] = stack
            folder = _sanitize(function)
            key = (folder, int(line))
            if key not in self._files:
                path = self.dir / "ops" / folder / f"{folder}_line{int(line)}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                self._files[key] = path.open("a")
            self._files[key].write(json.dumps(record) + "\n")
            return self.counter

    def close(self) -> None:
        with self._lock:
            for handle in self._files.values():
                handle.close()
            self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def export_pair(reference_root: Union[str, Path], suspect_root: Union[str, Path], output: Union[str, Path],
                og: Optional[OpGuardEnv] = None, ranks: Sequence[int] = (0,)) -> dict:
    """Compare two capture directories (OpGuard's own or ``TraceWriter``'s) with
    OpGuard's backend ``compare`` action. ``output`` must not exist. Returns
    OpGuard's result: per-rank ``reports`` and ``verdict: not_evaluated``."""
    return backend("compare", {"output": str(Path(output).resolve()),
                               "baseline": {"output": str(Path(reference_root).resolve())},
                               "candidate": {"output": str(Path(suspect_root).resolve())},
                               "ranks": [int(r) for r in ranks]}, og)


# ============================================================================
# d. From a first divergence to an operator class
# ============================================================================

RECIPES = "skills/batch-invariant-kernel/references/operator-recipes.md"
STANDARD = "handbook/3-kernels/README.md"
PRETRAINING = "handbook/1-pretraining/README.md"

# (class, recipe section, words that name it in an operator, a source line or a caller)
OPERATOR_CLASSES = [
    ("Softmax attention", "4. Softmax attention",
     r"scaled_dot_product_attention|flash_attn|flash_attention|sdpa|\battention\b|\battn\b"),
    ("MoE routing and combine", "8. MoE routing, dispatch and combine",
     r"\bmoe\b|router|gate_score|expert|index_add|scatter_add|index_put|\bcombine\b"),
    ("Sparse attention (top-k)", "6. Sparse attention with selection", r"\btopk\b|argsort|\bsort\b|indexer"),
    ("Collectives", "10. Collectives",
     r"all_reduce|allreduce|reduce_scatter|all_gather|all_to_all|\bnccl\b|\bbroadcast\b"),
    ("Quantized GEMM", "9. Quantized GEMM", r"fp8|nvfp4|mxfp8|quantiz|\bamax\b"),
    ("Linear attention / SSM", "7. Linear attention and SSMs", r"\bgdn\b|mamba|delta_rule|\bssm\b"),
    ("MLA", "5. MLA", r"\bmla\b|latent_attention"),
    ("Grouped GEMM, bmm vs mm", "3. Batched and grouped GEMM", r"\bbmm\b|baddbmm"),
    ("Row-wise reductions", "1. Row-wise reductions",
     r"rmsnorm|rms_norm|layernorm|layer_norm|\bnorm\d*\b|rsqrt|log_softmax|\bsoftmax\b|cross_entropy|logsumexp"),
    ("Dense GEMM", "2. Dense GEMM",
     r"\blinear\b|matmul|\bmm\b|addmm|einsum|\bgemm\b|_proj\b|\bfc\d*\b"),
    ("Deployment wrappers", "12. Deployment wrappers", r"torch\.compile|_dynamo|inductor|cudagraph|cuda_graph"),
]
GRADIENT_WORDS = r"\.grad\b|\bgrad\b|grad_norm|clip_grad|backward|optimizer|\bopt\.step|vector_norm"
RANDOM_WORDS = r"dropout|\bdrop\b|bernoulli|multinomial|randn|\brand\b|randint|randperm|normal_|uniform_"
LOOKUP_WORDS = r"embedding|\bemb\b|_emb\b|tok_emb|pos_emb|index_select|\bgather\b|sparse"
DATA_WORDS = r"dataloader|\bbatch(es)?\b|\bnext\b|dataset|sampler|collate"
CAST_WORDS = r"<None\.(to|float|half|bfloat16|double|type)>|\.to\(|\.float\(|\.half\(|\.bfloat16\(|type_as"


def _lead(kind, title, reason, check, where, fix=None):
    return {"kind": kind, "title": title, "reason": reason, "check": check, "handbook": where,
            "fix_options": fix or []}


def suggest(first_divergence_facts: dict, preflight: Optional[dict] = None) -> dict:
    """Map OpGuard's first divergence to the handbook's operator classes and
    recipes. Returns ordered leads, each with the recorded evidence it rests on,
    the next check to run and where the handbook treats it. A lead is a place to
    look. It is never a verdict: confirm it with an intervention (OpGuard's
    validate-hypothesis skill) before calling anything a cause.

    ``preflight`` may be a ``determinism_preflight.diff`` result or a single
    report: its findings are placed ahead of operator leads.
    """
    facts = first_divergence_facts
    out = {"schema": "opguard-suggest-v1", "verdict": "none: leads only", "evidence": [], "leads": [],
           "handoff": None}
    leads: List[dict] = out["leads"]

    if preflight:
        blocking = preflight.get("blocking") or []
        for item in blocking:
            leads.append(_lead("preflight", f"Preflight difference: {item.get('path')}",
                               f"The two runs differ on a controlled surface ({item.get('a')!r} vs "
                               f"{item.get('b')!r}): {item.get('why')}",
                               "Make the two runs agree on it, or declare it as the difference under test, "
                               "before reading the first divergence.", PRETRAINING + " (step 2)"))
        for item in preflight.get("at_risk") or preflight.get("findings") or []:
            leads.append(_lead("preflight", f"Uncontrolled surface: {item.get('path')}", item.get("why", ""),
                               item.get("fix", "Set it and capture again."), PRETRAINING + " (step 2)"))

    if not facts.get("diverged"):
        out["evidence"].append("OpGuard labelled no paired event as different.")
        unaligned = sum(c.get("unaligned", 0) for c in facts.get("status_counts", {}).values())
        if unaligned:
            leads.append(_lead("coverage", "Unaligned events", f"{unaligned} events have no partner.",
                               "Inspect them (first_diff reports the first one); an unpaired event is a gap, "
                               "not a difference.", PRETRAINING + " (step 4)"))
        out["handoff"] = "none: check coverage, then extend the run or the boundaries"
        return out

    p = facts["pivot"]
    src = p.get("source") or {}
    callers = " ".join(str(c.get("function")) for c in p.get("callers") or [])
    words = " ".join(str(x) for x in (p.get("fn"), src.get("text"), src.get("enclosing"), callers)).lower()
    where = f"{p.get('fn')} at {os.path.basename(str(p.get('file')))}:{p.get('line')}"
    if src.get("enclosing"):
        where += f" in {src['enclosing']}"
    out["evidence"].append(f"First observed difference: {where}, call {p.get('counter')}; "
                           f"fields that differ: {p.get('diff_fields')}.")
    if src.get("text"):
        out["evidence"].append(f"Source line: {src['text']}")
    n_un = facts.get("unaligned_before") or 0
    if n_un:
        out["evidence"].append(f"{n_un} events before it have no partner on the other side.")
        leads.append(_lead("alignment", "Unpaired events before the pivot",
                           f"{n_un} earlier events are undetermined, so an earlier divergence may be hidden.",
                           "Inspect the first unaligned event and its call site on both sides; record a pairing "
                           "with review_pair only with evidence.", PRETRAINING + " (step 4)"))
    lone = [s for s in facts.get("unaligned_sites") or [] if s.get("reason") == "no_corresponding_trace_file"]
    if lone:
        sites = ", ".join(f"{s['file']}:{s['line']} ({s['run']})" for s in lone[:4])
        out["evidence"].append(f"Call sites captured on one side only: {sites}.")

    inputs_differ, outputs_differ = p.get("inputs_differ"), p.get("outputs_differ")
    no_inputs = p.get("inputs_recorded") == 0
    gradient = bool(re.search(GRADIENT_WORDS, words))
    statuses = (facts.get("site_statuses") or {}).get("statuses", "")

    if p.get("dtype_differs") or p.get("shape_differs"):
        what = "dtype" if p.get("dtype_differs") else "shape"
        ref_t = [t.get(what) for t in (p["reference"]["outputs"] + p["reference"]["inputs"])]
        sus_t = [t.get(what) for t in (p["suspect"]["outputs"] + p["suspect"]["inputs"])]
        out["evidence"].append(f"The recorded {what} differs: {ref_t} vs {sus_t}.")
        if what == "dtype":
            leads.append(_lead(
                "rounding-point", "A cast or rounding point moved",
                f"The same call produces {ref_t} on the reference and {sus_t} on the suspect. The two runs "
                "round at different places"
                + (", and one side has call sites the other never runs" if lone else "") + ".",
                "Write down both sequences (where each cast happens, what is fused, the accumulator dtype) and "
                "compare the next boundary with equal dtype element by element (bi_check.first_mismatch).",
                "skills/batch-invariant-kernel/SKILL.md (section 3) and " + RECIPES + " (1. Row-wise reductions)",
                ["Make the suspect round where the reference rounds.",
                 "If the new order is wanted, change both runs together and accept a new reference."]))
        else:
            leads.append(_lead("layout", "The two runs see different shapes here",
                               f"Shapes {ref_t} vs {sus_t}: a different batch, micro-batch or sequence layout.",
                               "Compare per-vector fingerprints (xor_vectors, merge) row by row; a different "
                               "layout is a different configuration unless every kernel on the path is "
                               "batch-invariant (L2).", STANDARD))

    if inputs_differ and not (p.get("dtype_differs") or p.get("shape_differs")):
        out["evidence"].append("Recorded inputs already differ at this call: it is a consumer, not the origin.")
        if gradient:
            agree = statuses.count("o")
            reason = ("The operator reads gradients (" + (src.get("text") or p.get("fn") or "") + "), and its "
                      "input differs while the forward events before it agree: a backward kernel upstream "
                      "produced different bits from the same forward.")
            if statuses:
                reason += (f" At this call site {statuses.count('d')} of {len(statuses)} calls differ and "
                           f"{agree} agree; the calls that agree mark values that do not depend on the "
                           "nondeterministic op.")
            leads.append(_lead(
                "nondeterministic-backward", "A backward kernel upstream",
                reason,
                "Map each call at this site to its parameter (call order) and find the most downstream "
                "parameter that differs. Then repeat that op's backward 5 times on fixed inputs "
                "(bi_check.check_repeat).",
                RECIPES + " (4. Softmax attention, Backward) and handbook/3-kernels/attention.md",
                ["torch.use_deterministic_algorithms(True, warn_only=False): warn_only=True leaves SDPA's "
                 "backward nondeterministic (measured).",
                 "An atomic-free backward (tools/bi_kernels.py attention_fixed_blocks_train).",
                 "For index_add / scatter_add / embedding-style backward: the deterministic flag, or a "
                 "sorted fixed-order accumulation."]))
        else:
            leads.append(_lead("consumer", "The inputs already differ",
                               "The origin is upstream of this call, in an op with no boundary record or in "
                               "state this call reads.",
                               "Follow the differing input to its producer (OpGuard skill trace-upstream); "
                               "check the coverage of the capture between the last ok event and this one.",
                               PRETRAINING + " (step 5)"))
    elif outputs_differ and not inputs_differ and not (p.get("dtype_differs") or p.get("shape_differs")):
        if no_inputs:
            out["evidence"].append("No tensor input is recorded for this call (a method call records its "
                                   "arguments, not its receiver): input equality is unknown.")
        else:
            out["evidence"].append("Recorded inputs agree and the output differs: a candidate origin "
                                   "(fingerprints can collide; weights and state are not inputs).")

    for name, section, pattern in OPERATOR_CLASSES:
        hit = re.search(pattern, words)
        if not hit:
            continue
        if gradient and inputs_differ and name in ("Row-wise reductions", "Dense GEMM"):
            continue                                    # a gradient norm is not the norm of the model
        reason = f"'{hit.group(0)}' names this class in the operator, its source line or its callers."
        if name == "Dense GEMM" and inputs_differ:
            reason += " Its inputs differ already, so read it as the first GEMM that saw the difference."
        leads.append(_lead("operator-class", name, reason,
                           "Run bi_check.check_repeat on the op with the captured shapes (L1), then "
                           "bi_check.check_batch (L2) if the two runs batch differently.",
                           f"{RECIPES} ({section}) and {STANDARD}"))

    if re.search(RANDOM_WORDS, words) and outputs_differ and not inputs_differ:
        leads.append(_lead("rng", "A random op with equal inputs",
                           "The output of a random op differs while its recorded input agrees: the RNG stream "
                           "is at a different position or seed.",
                           "Compare the RNG digests of the two preflight reports at this point; on a resume, "
                           "check that CPU and CUDA generator states are restored.",
                           PRETRAINING + " (step 2, surfaces 1 and 4)"))
    if re.search(LOOKUP_WORDS, words) and outputs_differ and not inputs_differ:
        leads.append(_lead("state", "A lookup with equal indices and a different result",
                           "A table lookup moves data exactly, so equal indices with a different output mean "
                           "the table differs: the weights changed before this call.",
                           "Find the last event that agrees. If it is a gradient consumer of the previous "
                           "step, the optimizer update between them differs: compare optimizer state (step "
                           "counters, moments), learning rate and the restore path.",
                           PRETRAINING + " (step 2, surface 4; root cause 'state bookkeeping')",
                           ["Restore every piece of optimizer and scheduler state on resume; compare "
                            "determinism_preflight reports taken right after the restore."]))
    if re.search(DATA_WORDS, words) and (inputs_differ or (outputs_differ and (p.get("counter") or 99) <= 3)):
        leads.append(_lead("data", "The data itself differs",
                           "The first difference is at the data boundary: the two runs read different batches.",
                           "Compare seeds, the sampler or generator state and the dataloader position "
                           "(determinism_preflight.diff); on a resume, the number of batches skipped.",
                           PRETRAINING + " (step 2, surfaces 1 and 4)"))
    if not any(lead["kind"] not in ("preflight", "alignment") for lead in leads):
        leads.append(_lead("unclassified", "No handbook class matched",
                           "Neither the operator name nor its source line names a known class.",
                           "Read the operator's source and its producers (OpGuard skills investigate-first-diff "
                           "and trace-upstream).", PRETRAINING + " (step 5)"))
    if re.search(CAST_WORDS, str(p.get("fn")) + " " + str(src.get("text"))) and not p.get("dtype_differs"):
        out["evidence"].append("The recorded call is a cast or a layout change: the operator that computed "
                               "the value is before it on the line or in the caller.")
    out["handoff"] = ("trace-upstream" if inputs_differ or n_un else "validate-hypothesis") + \
        " (OpGuard harness skill), entered through investigate-first-diff"
    return out


# ============================================================================
# CLI
# ============================================================================


def selftest(device: Optional[str] = None, verbose: bool = True) -> List[dict]:
    """Plain-torch fingerprints against the pure-Python reference of the documented
    formats, and against OpGuard's functions when importable. Returns one dict per
    case with ``passed``."""
    torch = _torch()
    devices = [device] if device else ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    gen = torch.Generator().manual_seed(0)
    cases = []

    def tensor(shape, dtype):
        if dtype == torch.bool:
            return torch.rand(shape, generator=gen) > 0.5
        if dtype.is_floating_point:
            return (torch.randn(shape, generator=gen) * 100).to(dtype)
        return torch.randint(-2 ** 31, 2 ** 31 - 1, shape, generator=gen).to(dtype)
    specs = [("bf16 [38, 32, 128]", (38, 32, 128), torch.bfloat16), ("fp16 [7, 33]", (7, 33), torch.float16),
             ("fp32 [5, 6, 7]", (5, 6, 7), torch.float32), ("fp64 [9, 3]", (9, 3), torch.float64),
             ("int64 [11, 4]", (11, 4), torch.int64), ("bool [1001]", (1001,), torch.bool),
             ("uint8 [13, 7]", (13, 7), torch.uint8), ("uint8 [3]", (3,), torch.uint8),
             ("bf16 [3] (6 bytes)", (3,), torch.bfloat16), ("empty [0, 4]", (0, 4), torch.float32),
             ("0-dim fp32", (), torch.float32), ("0-dim bf16", (), torch.bfloat16)]
    for dev in devices:
        for label, shape, dtype in specs:
            base = tensor(shape, dtype).to(dev)
            variants = [(label, base)]
            if base.dim() >= 2:
                variants.append((label + " transposed", base.transpose(0, -1)))
                variants.append((label + " strided", base[::2]))
            for name, t in variants:
                data = bytes(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).tolist())
                for regions in (1, 4, 7):
                    got = _unsigned_list(xor_signature(t, regions, impl="torch"))
                    ok = got == reference_signature(data, regions)
                    if opguard_available():
                        ok = ok and got == _unsigned_list(xor_signature(t, regions, impl="opguard"))
                    cases.append({"case": f"{dev} {name} regions={regions}", "passed": ok})
                layout = vector_layout(t)
                vec = xor_vectors(t, impl="torch")
                if "skipped" in layout:
                    cases.append({"case": f"{dev} {name} vectors skipped ({layout['skipped']})", "passed": vec is None})
                    continue
                got = _unsigned_list(vec)
                ok = got == reference_vectors(data, layout["vector_bytes"]) and len(got) == layout["count"]
                if opguard_available():
                    ok = ok and got == _unsigned_list(xor_vectors(t, impl="opguard"))
                ok = ok and merge(got) == reference_signature(data, 1)
                ok = ok and decode_vectors(encode_vectors(vec, layout["vector_bytes"])) == got
                ok = ok and decode_vectors([_signed32(v) for v in got]) == got
                cases.append({"case": f"{dev} {name} vectors", "passed": ok})
    if verbose:
        bad = [c["case"] for c in cases if not c["passed"]]
        print(f"fingerprint selftest: {len(cases) - len(bad)}/{len(cases)} cases pass "
              f"(reference: pure Python{'; and OpGuard' if opguard_available() else '; OpGuard not importable'})")
        for name in bad:
            print("  FAIL", name)
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The handbook's interface to OpGuard.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="fingerprints against the documented formats (no OpGuard needed)")
    fd = sub.add_parser("first-divergence", help="facts OpGuard reports about one comparison export")
    fd.add_argument("artifact")
    fd.add_argument("--root", default="./opguard-runs", help="harness store")
    fd.add_argument("--opguard-root")
    fd.add_argument("--python")
    fd.add_argument("--suggest", action="store_true", help="also print the handbook leads")
    sg = sub.add_parser("suggest", help="handbook leads for a first-divergence facts file")
    sg.add_argument("facts")
    args = ap.parse_args(argv)
    if args.command == "selftest":
        return 0 if all(c["passed"] for c in selftest()) else 1
    if args.command == "suggest":
        print(json.dumps(suggest(json.loads(Path(args.facts).read_text())), indent=1))
        return 0
    og = locate_opguard(args.opguard_root, args.python)
    facts = first_divergence(args.artifact, args.root, og)
    print(json.dumps({"facts": facts, "suggest": suggest(facts)} if args.suggest else facts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
