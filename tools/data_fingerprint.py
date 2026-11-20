"""data_fingerprint: bitwise fingerprints of the batches a data loader hands to training.

Two runs can only be compared bitwise if every rank sees the same token bits at
every step. This tool records, per step and rank, fingerprints of every field
of the batch as it leaves the loader, and compares two runs with exact equality.
It names the first differing (step, rank, field) and says which kind of
difference it is:

    order     same samples, different order (inside the batch, or moved to
              another step or rank)
    samples   different samples (some documents exist on one side only)
    packing   same samples (documents), different packing or padding
    masks     same token ids, different masks, positions or labels
    structure different fields or dtypes;  missing: a record on one side only

In a training loop, two lines:

    rec = data_fingerprint.Recorder("fp/run_a", rank=dp_rank)
    for step, batch in enumerate(loader):
        rec.record(step, batch)             # before the batch is used
        ...

``meta=`` adds what the run was built from, compared first:
``Recorder(dir, rank, meta={"tokenizer": tokenizer_fingerprint(tok, add_special_tokens=True),
"source": source_fingerprint(files)})``. ``append=True`` continues a run after a
resume (a later record of the same step replaces the earlier one).
``opguard="dir"`` also writes the records in OpGuard's capture layout
(opguard_bridge.TraceWriter), so OpGuard's harness ``first_diff`` sees the data
boundary in front of the model's.

    python data_fingerprint.py compare fp/run_a fp/run_b            # per (step, rank)
    python data_fingerprint.py compare fp/run_a fp/run_b --global   # per step, ranks concatenated
    python data_fingerprint.py export fp/run_a og/run_a             # OpGuard capture layout
    python data_fingerprint.py selftest

What one record holds, per field (each tensor of the batch; dim 0 is the sample):

    shape, dtype
    xor    ``flat-u32-le-v1`` fingerprint of the whole field, one region
           (opguard_bridge.xor_signature: OpGuard's format)
    rows   one XOR fingerprint per sample (opguard_bridge.xor_vectors on the
           sample's bytes, zero-padded to whole 32-bit words; for a 2-D field of
           whole words these are OpGuard's per-row values, and they XOR to ``xor``)
    keys   one position-keyed 64-bit key per sample. XOR does not see a
           permutation of tokens inside a sample (a ^ b = b ^ a), so every
           comparison also uses these: each 8-byte word is mixed with its index
           (splitmix64) before the XOR.

and, for the token field, the documents in the batch ("segments"): one
position-keyed key and the length of each document. A document starts at the
first unmasked token of a row, where ``position_ids`` returns to 0, after a
masked gap, and after ``sep_id``. Masked tokens (attention mask 0) and
``pad_id`` tokens, when no mask is given, are left out. The same documents
packed or padded differently give the same segment keys.

A difference in any fingerprint proves the bits differ. Equal fingerprints are
equal bits up to a hash collision (64-bit keys), not a proof.

Records are JSON lines, one file per rank: ``DIR/rank00000.jsonl``, a header
line (``kind: header``) and one line per batch (``kind: batch``).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opguard_bridge as ob  # noqa: E402

SCHEMA = "data-fp-v1"
COMPARE_SCHEMA = "data-fp-compare-v1"
MASK32 = 0xFFFFFFFF
CLASSES = {
    "order": "same samples, different order",
    "samples": "different samples",
    "packing": "same samples, different packing or padding",
    "masks": "same token ids, different masks, positions or labels",
    "structure": "different fields or dtypes",
    "missing": "a record on one side only",
}

# ----------------------------------------------------------------------------- keys
_G = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _mix(z):
    """splitmix64's finalizer on a uint64 array (arithmetic mod 2**64)."""
    with np.errstate(over="ignore"):
        z = np.asarray(z, dtype=np.uint64)
        z = z ^ (z >> np.uint64(30))
        z = z * _M1
        z = z ^ (z >> np.uint64(27))
        z = z * _M2
        return z ^ (z >> np.uint64(31))


def _keyed(values, positions, length):
    """Position-keyed key of a word sequence: mix(XOR_j mix(w_j + G*(j+1)) ^ mix(length))."""
    with np.errstate(over="ignore"):
        h = _mix(values.astype(np.uint64) + _G * (positions.astype(np.uint64) + np.uint64(1)))
    return h, _mix(np.asarray(length, dtype=np.uint64))


def _hex(values) -> List[str]:
    return [f"{int(v):016x}" for v in np.asarray(values, dtype=np.uint64).reshape(-1)]


def sample_keys(raw: np.ndarray) -> List[str]:
    """Position-keyed 64-bit key of each row of a uint8 [samples, bytes] array."""
    rows, nbytes = raw.shape
    if rows == 0:
        return []
    pad = (-nbytes) % 8
    if pad:
        raw = np.concatenate([raw, np.zeros((rows, pad), np.uint8)], axis=1)
    words = np.ascontiguousarray(raw).view("<u8").astype(np.uint64)
    idx = np.broadcast_to(np.arange(words.shape[1], dtype=np.uint64), words.shape)
    h, tail = _keyed(words, idx, np.full(rows, nbytes))
    acc = np.bitwise_xor.reduce(h, axis=1) if words.shape[1] else np.zeros(rows, np.uint64)
    return _hex(_mix(acc ^ tail))


def segments(tokens: np.ndarray, mask: Optional[np.ndarray] = None, positions: Optional[np.ndarray] = None,
             sep_id: Optional[int] = None, pad_id: Optional[int] = None) -> dict:
    """The documents in a [rows, tokens] batch: one position-keyed key, length and
    row each, in row order (see the module docstring for the boundary rule)."""
    keys, lens, rows = [], [], []
    for r in range(tokens.shape[0]):
        tok = tokens[r]
        keep = np.ones(tok.shape, bool)
        if mask is not None:
            keep &= mask[r] != 0
        elif pad_id is not None:
            keep &= tok != pad_id
        idx = np.nonzero(keep)[0]
        if not len(idx):
            continue
        start = np.zeros(len(idx), bool)
        start[0] = True
        start[1:] |= np.diff(idx) != 1                  # a masked gap
        if positions is not None:
            start |= positions[r][idx] == 0
        if sep_id is not None:
            start[1:] |= tok[idx[:-1]] == sep_id        # the token after a separator
        starts = np.nonzero(start)[0]
        seg = np.cumsum(start) - 1
        rel = np.arange(len(idx)) - starts[seg]
        length = np.diff(np.append(starts, len(idx)))
        h, tail = _keyed(tok[idx].astype(np.int64).view(np.uint64), rel, length)
        keys += _hex(_mix(np.bitwise_xor.reduceat(h, starts) ^ tail))
        lens += [int(x) for x in length]
        rows += [r] * len(starts)
    return {"keys": keys, "lens": lens, "rows": rows}


# ----------------------------------------------------------------------------- one field
def _torch():
    import torch
    return torch


def _as_tensor(value):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        return torch.from_numpy(np.ascontiguousarray(value))
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, (int, float, bool)) for v in value):
        try:
            return torch.as_tensor(value)
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, (list, tuple)) for v in value):
        try:
            return torch.as_tensor(value)
        except Exception:
            return None
    return None


def _sample_bytes(t):
    """uint8 [samples, bytes] on the tensor's device, in logical order."""
    torch = _torch()
    t = t.detach()
    if t.dim() == 0:
        t = t.reshape(1)
    n = t.shape[0]
    if t.numel() == 0:
        return torch.zeros((n, 0), dtype=torch.uint8, device=t.device)
    return ob._raw_bytes(t).reshape(n, -1)


def fingerprint_field(value) -> dict:
    """Fingerprints of one batch field (see the module docstring)."""
    t = _as_tensor(value)
    if t is None:                                       # strings, objects: one hash of their JSON
        blob = json.dumps(value, sort_keys=True, default=repr).encode()
        return {"kind": "object", "sha256": hashlib.sha256(blob).hexdigest()}
    torch = _torch()
    raw = _sample_bytes(t)
    n, nbytes = raw.shape
    rows: List[int] = []
    if n and nbytes:
        words = raw if nbytes % 4 == 0 else torch.cat(
            [raw, torch.zeros((n, (-nbytes) % 4), dtype=torch.uint8, device=raw.device)], dim=1)
        rows = [int(v) & MASK32 for v in ob.xor_vectors(words).reshape(-1).tolist()]
    elif n:
        rows = [0] * n
    xor = int(ob.xor_signature(t, regions=1).reshape(-1)[0].item()) & MASK32
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "xor": xor, "rows": rows,
            "keys": sample_keys(np.ascontiguousarray(raw.cpu().numpy()))}


def _flatten(batch, prefix="") -> List[Tuple[str, Any]]:
    """(name, value) of every field of a batch: a mapping (nested names joined by
    '.'), a namedtuple, a tuple or list of tensors, or one tensor ("batch")."""
    torch = _torch()
    if isinstance(batch, Mapping):
        out = []
        for k, v in batch.items():
            name = f"{prefix}{k}"
            out += _flatten(v, name + ".") if isinstance(v, Mapping) else [(name, v)]
        return out
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        return _flatten(dict(zip(batch._fields, batch)), prefix)
    if isinstance(batch, (list, tuple)) and batch and all(isinstance(v, (torch.Tensor, np.ndarray, Mapping)) for v in batch):
        return _flatten({str(i): v for i, v in enumerate(batch)}, prefix)
    return [((prefix[:-1] if prefix else "batch"), batch)]


def _numpy_int(value) -> Optional[np.ndarray]:
    t = _as_tensor(value)
    if t is None or t.dtype.is_floating_point or t.dtype.is_complex:
        return None
    a = t.detach().cpu().to(_torch().int64).numpy()
    return a.reshape(1, -1) if a.ndim <= 1 else a.reshape(a.shape[0], -1)


def fingerprint_batch(batch, *, token_field: str = "input_ids", mask_field: str = "attention_mask",
                      position_field: str = "position_ids", sep_id: Optional[int] = None,
                      pad_id: Optional[int] = None) -> dict:
    """``{"fields": {name: fingerprint}, "segments": {...}}`` for one batch."""
    items = _flatten(batch)
    fields = {name: fingerprint_field(v) for name, v in items}
    values = dict(items)
    tok_name = token_field if token_field in values else next(
        (n for n, f in fields.items() if f.get("dtype", "").startswith(("int", "uint")) and len(f["shape"]) >= 1), None)
    out: Dict[str, Any] = {"fields": fields}
    tokens = _numpy_int(values[tok_name]) if tok_name else None
    if tokens is not None:
        aux = {}
        for key, name in (("mask", mask_field), ("positions", position_field)):
            a = _numpy_int(values[name]) if name in values else None
            aux[key] = a if a is not None and a.shape == tokens.shape else None
        out["segments"] = {"field": tok_name, **segments(tokens, aux["mask"], aux["positions"], sep_id, pad_id)}
    return out


# ----------------------------------------------------------------------------- meta
def tokenizer_fingerprint(tok, **call_kwargs) -> dict:
    """A hash of a tokenizer's full definition (vocabulary, normalizer,
    pre-tokenizer, model, post-processor, special tokens) and of the arguments it
    is called with (``add_special_tokens``, ``truncation``, ...), which the
    definition does not hold. Accepts a ``tokenizers.Tokenizer`` or a
    transformers tokenizer."""
    parts: Dict[str, Any] = {"class": type(tok).__name__}
    backend = getattr(tok, "backend_tokenizer", None)
    if hasattr(tok, "to_str"):
        parts["definition"] = tok.to_str()
    elif backend is not None:
        parts["definition"] = backend.to_str()
    elif hasattr(tok, "get_vocab"):
        parts["vocab"] = sorted(tok.get_vocab().items())
    for attr in ("special_tokens_map", "init_kwargs", "padding_side", "truncation_side", "model_max_length",
                 "add_bos_token", "add_eos_token", "clean_up_tokenization_spaces"):
        if hasattr(tok, attr):
            parts[attr] = getattr(tok, attr)
    parts["call"] = call_kwargs
    blob = json.dumps(parts, sort_keys=True, default=repr).encode()
    return {"sha256": hashlib.sha256(blob).hexdigest(), "class": parts["class"],
            "call": json.loads(json.dumps(call_kwargs, default=repr))}


def source_fingerprint(paths: Union[str, Path, Iterable[Union[str, Path]]], max_bytes: Optional[int] = None) -> dict:
    """A hash of the data files: name (relative to their common parent), size and
    content. ``max_bytes`` hashes only the first and last ``max_bytes`` bytes of
    each file, for corpora too large to read in full."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: List[Path] = []
    for p in map(Path, paths):
        files += sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    base = Path(os.path.commonpath([str(f.resolve().parent) for f in files])) if files else Path(".")
    h, total = hashlib.sha256(), 0
    for f in files:
        size = f.stat().st_size
        total += size
        h.update(f"{f.resolve().relative_to(base)}\0{size}\0".encode())
        with f.open("rb") as fh:
            if max_bytes is None or size <= 2 * max_bytes:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            else:
                h.update(fh.read(max_bytes))
                fh.seek(size - max_bytes)
                h.update(fh.read(max_bytes))
    return {"sha256": h.hexdigest(), "files": len(files), "bytes": total,
            "content": "full" if max_bytes is None else f"first and last {max_bytes} bytes"}


# ----------------------------------------------------------------------------- recorder
class Recorder:
    """Fingerprint every batch at the loader's exit and append one JSON line per
    batch to ``out_dir/rank{rank:05d}.jsonl``.

        rec = Recorder("fp/run_a", rank=dp_rank, meta={...})
        rec.record(step, batch)                 # or rec(step, batch); micro= for micro-batches

    ``token_field``, ``mask_field``, ``position_field``, ``sep_id``, ``pad_id``
    say how documents are found (module docstring). ``append=True`` keeps the
    lines already there (a resumed run). ``opguard`` is a directory for OpGuard
    capture records (``export``). Each line is flushed when written.
    """

    def __init__(self, out_dir: Union[str, Path], rank: int = 0, *, meta: Optional[dict] = None,
                 token_field: str = "input_ids", mask_field: str = "attention_mask",
                 position_field: str = "position_ids", sep_id: Optional[int] = None,
                 pad_id: Optional[int] = None, append: bool = False,
                 opguard: Optional[Union[str, Path]] = None):
        self.rank = int(rank)
        self.opts = {"token_field": token_field, "mask_field": mask_field, "position_field": position_field,
                     "sep_id": sep_id, "pad_id": pad_id}
        self.path = Path(out_dir) / f"rank{self.rank:05d}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a" if append else "w")
        self._write({"kind": "header", "schema": SCHEMA, "rank": self.rank, "meta": meta or {}, "options": self.opts})
        self._og = ob.TraceWriter(opguard, rank=self.rank) if opguard else None

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._fh.flush()

    def record(self, step: int, batch, *, micro: int = 0) -> dict:
        rec = {"kind": "batch", "step": int(step), "micro": int(micro), "rank": self.rank,
               **fingerprint_batch(batch, **self.opts)}
        self._write(rec)
        if self._og is not None:
            emit_opguard(self._og, rec)
        return rec

    __call__ = record

    def close(self) -> None:
        self._fh.close()
        if self._og is not None:
            self._og.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------------------- reading
def load(run_dir: Union[str, Path]) -> dict:
    """``{"headers": {rank: [header, ...]}, "records": {(step, micro, rank): record},
    "replaced": n}``. A later line for the same (step, micro, rank) replaces the
    earlier one (a resumed run that redid a step)."""
    headers: Dict[int, List[dict]] = {}
    records: Dict[Tuple[int, int, int], dict] = {}
    replaced = 0
    files = sorted(Path(run_dir).glob("rank*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no rank*.jsonl in {run_dir}")
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("kind") == "header":
                headers.setdefault(obj["rank"], []).append(obj)
            elif obj.get("kind") == "batch":
                key = (obj["step"], obj.get("micro", 0), obj["rank"])
                replaced += key in records
                records[key] = obj
    return {"dir": str(run_dir), "headers": headers, "records": records, "replaced": replaced}


def _merge_ranks(recs: List[dict]) -> dict:
    """One record for a step: every rank's samples concatenated in rank order."""
    names: List[str] = []
    for r in recs:
        names += [n for n in r["fields"] if n not in names]
    fields = {}
    for n in names:
        parts = [r["fields"].get(n) for r in recs]
        if any(p is None or "sha256" in p for p in parts):
            fields[n] = {"per_rank": parts}
            continue
        shapes = [p["shape"] for p in parts]
        same_tail = all(len(s) >= 1 and s[1:] == shapes[0][1:] for s in shapes)
        xor = 0
        for p in parts:
            xor ^= p["xor"]
        dtypes = sorted({p["dtype"] for p in parts})
        fields[n] = {"shape": [sum(s[0] for s in shapes)] + shapes[0][1:] if same_tail else shapes,
                     "dtype": dtypes[0] if len(dtypes) == 1 else dtypes, "xor": xor,
                     "rows": [v for p in parts for v in p["rows"]], "keys": [v for p in parts for v in p["keys"]]}
    out = {"kind": "batch", "step": recs[0]["step"], "micro": recs[0].get("micro", 0), "rank": "all",
           "ranks": [r["rank"] for r in recs], "fields": fields}
    if all("segments" in r for r in recs):
        seg = {"field": recs[0]["segments"]["field"], "keys": [], "lens": [], "rows": []}
        offset = 0
        for r in recs:
            s = r["segments"]
            seg["keys"] += s["keys"]
            seg["lens"] += s["lens"]
            seg["rows"] += [offset + i for i in s["rows"]]
            offset += len(r["fields"][s["field"]]["keys"])
        out["segments"] = seg
    return out


def _view(run: dict, global_: bool) -> Dict[tuple, dict]:
    if not global_:
        return dict(run["records"])
    groups: Dict[tuple, List[dict]] = {}
    for (step, micro, rank), rec in sorted(run["records"].items()):
        groups.setdefault((step, micro, "all"), []).append(rec)
    return {k: _merge_ranks(v) for k, v in groups.items()}


def _index(view: Dict[tuple, dict]) -> Dict[str, List[tuple]]:
    """Segment key -> the records (step, micro, rank) that hold it."""
    idx: Dict[str, List[tuple]] = {}
    for loc, rec in view.items():
        for k in rec.get("segments", {}).get("keys", []):
            idx.setdefault(k, []).append(loc)
    return idx


# ----------------------------------------------------------------------------- classify
def classify(ra: dict, rb: dict, *, index_a: Optional[dict] = None, index_b: Optional[dict] = None) -> Optional[dict]:
    """Why record ``rb`` (suspect) differs from ``ra`` (reference), or None when
    every field is equal. ``index_a`` / ``index_b`` (segment key -> locations)
    let it see samples that moved to another step or rank."""
    fa, fb = ra["fields"], rb["fields"]
    tok = (ra.get("segments") or rb.get("segments") or {}).get("field")
    names = ([tok] if tok in fa else []) + [n for n in fa if n != tok] + [n for n in fb if n not in fa]
    diff = [n for n in names if fa.get(n) != fb.get(n)]
    if not diff:
        return None
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    dtypes = {n: [fa[n].get("dtype"), fb[n].get("dtype")] for n in fa if n in fb and fa[n].get("dtype") != fb[n].get("dtype")}
    layout = {"only_reference": only_a, "only_suspect": only_b, "dtype": dtypes}
    if tok is None or tok not in diff or tok not in fb or tok in dtypes:
        if only_a or only_b or dtypes:
            return {"class": "structure", "field": diff[0], "detail": layout}
        rows = {}
        for n in diff:
            ka, kb = fa[n].get("keys"), fb[n].get("keys")
            if ka is not None and kb is not None and len(ka) == len(kb):
                rows[n] = [i for i, (x, y) in enumerate(zip(ka, kb)) if x != y]
        return {"class": "masks", "field": diff[0], "detail": {"fields": diff, "rows": rows}}
    extra = {"fields_differ": layout} if (only_a or only_b or dtypes) else {}
    ka, kb = fa[tok]["keys"], fb[tok]["keys"]
    if collections.Counter(ka) == collections.Counter(kb):
        where: Dict[str, List[int]] = {}
        for i, k in enumerate(ka):
            where.setdefault(k, []).append(i)
        perm = [where[k].pop(0) for k in kb]
        follow = all(n not in diff or (n in fa and n in fb and "keys" in fa[n] and "keys" in fb[n]
                                       and len(fa[n]["keys"]) == len(perm)
                                       and [fa[n]["keys"][i] for i in perm] == fb[n]["keys"])
                     for n in names if n != tok)
        return {"class": "order", "field": tok,
                "detail": {"scope": "inside the batch", "suspect_row_from_reference_row": perm,
                           "other_fields_follow": follow, **extra}}
    sa, sb = ra.get("segments") or {}, rb.get("segments") or {}
    ca, cb = collections.Counter(sa.get("keys", [])), collections.Counter(sb.get("keys", []))
    if ca and ca == cb:
        return {"class": "packing", "field": tok,
                "detail": {"documents": sum(ca.values()), "shape_reference": fa[tok]["shape"],
                           "shape_suspect": fb[tok]["shape"], **extra}}
    gone, new = list((ca - cb).elements()), list((cb - ca).elements())
    nowhere_b = [k for k in gone if index_b is not None and k not in index_b]
    nowhere_a = [k for k in new if index_a is not None and k not in index_a]
    if ca and index_a is not None and index_b is not None and not nowhere_a and not nowhere_b:
        here = (rb["step"], rb.get("micro", 0))
        found_a = sorted({loc for k in new for loc in index_a[k]}, key=str)     # where the suspect's extras were
        found_b = sorted({loc for k in gone for loc in index_b[k]}, key=str)    # where the reference's went
        steps = sorted({loc[0] for k in new for loc in index_a[k] if (loc[0], loc[1]) != here})
        moved = steps or any((loc[0], loc[1]) != here for loc in found_b)
        seen_before = sorted({loc[0] for k in new for loc in index_b[k] if loc[0] < rb["step"]})
        return {"class": "order", "field": tok,
                "detail": {"scope": "across steps" if moved else "across ranks, same step",
                           "suspect_documents_found_in_reference_at": [list(x) for x in found_a[:8]],
                           "reference_documents_found_in_suspect_at": [list(x) for x in found_b[:8]],
                           "step_offset": steps[0] - rb["step"] if len(steps) == 1 else None,
                           "suspect_repeats_its_own_steps": seen_before[:8], **extra}}
    detail: Dict[str, Any] = {"documents_reference": sum(ca.values()), "documents_suspect": sum(cb.values()),
                              "reference_only": len(nowhere_b) if index_b is not None else len(gone),
                              "suspect_only": len(nowhere_a) if index_a is not None else len(new), **extra}
    la, lb = sa.get("lens", []), sb.get("lens", [])
    if la and len(la) == len(lb):
        deltas = {b - a for a, b in zip(la, lb)}
        if len(deltas) == 1 and 0 not in deltas:
            d = deltas.pop()
            detail["hint"] = (f"every document is {abs(d)} token(s) {'longer' if d > 0 else 'shorter'} in the suspect: "
                              "a special token added or removed; compare the tokenizer fingerprints and call arguments")
        elif deltas == {0}:
            detail["hint"] = ("same document lengths, different token ids: a vocabulary, normalizer or "
                              "data-source change, or a different shuffle of equal-length documents")
    return {"class": "samples", "field": tok, "detail": detail}


def _meta_values(run: dict) -> Dict[str, Dict[str, Any]]:
    """Header key -> {its value as JSON: the value}, over every rank and session."""
    out: Dict[str, Dict[str, Any]] = {}
    for heads in run["headers"].values():
        for h in heads:
            for k, v in h.get("meta", {}).items():
                out.setdefault(k, {})[json.dumps(v, sort_keys=True, default=str)] = v
    return out


def _meta_diff(a: dict, b: dict) -> List[dict]:
    """Header keys whose set of values (over ranks and sessions) differs between
    the runs. Ranks are pooled, so runs at different data-parallel sizes compare."""
    ma, mb = _meta_values(a), _meta_values(b)
    out = []
    for k in sorted(set(ma) | set(mb)):
        va, vb = ma.get(k, {}), mb.get(k, {})
        if set(va) != set(vb):
            one = lambda d: (list(d.values())[0] if len(d) == 1 else list(d.values())) if d else None  # noqa: E731
            out.append({"key": k, "reference": one(va), "suspect": one(vb)})
    return out


def compare(reference: Union[str, Path, dict], suspect: Union[str, Path, dict], *, global_: bool = False,
            limit: int = 20) -> dict:
    """Compare two recorded runs with exact equality. ``global_`` merges the ranks
    of each step (rank order) first, for runs at different data-parallel sizes.
    Returns ``data-fp-compare-v1``: ``equal``, ``first`` (step, micro, rank,
    field, class, detail), ``classes`` (differing records per class), the first
    ``limit`` differing records, and ``meta_diff`` (header differences)."""
    a = reference if isinstance(reference, dict) else load(reference)
    b = suspect if isinstance(suspect, dict) else load(suspect)
    va, vb = _view(a, global_), _view(b, global_)
    ia, ib = _index(va), _index(vb)
    keys = sorted(set(va) | set(vb), key=lambda k: (k[0], k[1], str(k[2]) if global_ else k[2]))
    classes: Dict[str, int] = collections.Counter()
    differing, equal_n = [], 0
    for k in keys:
        ra, rb = va.get(k), vb.get(k)
        if ra is None or rb is None:
            res = {"class": "missing", "field": None,
                   "detail": {"present_in": "reference" if rb is None else "suspect"}}
        else:
            res = classify(ra, rb, index_a=ia, index_b=ib)
            if res is None:
                equal_n += 1
                continue
        classes[res["class"]] += 1
        if len(differing) < limit:
            differing.append({"step": k[0], "micro": k[1], "rank": k[2], **res,
                              "meaning": CLASSES[res["class"]]})
    meta = _meta_diff(a, b)
    return {"schema": COMPARE_SCHEMA, "mode": "global" if global_ else "rank",
            "reference": a.get("dir"), "suspect": b.get("dir"),
            "equal": not differing and not classes and not meta,
            "records": {"reference": len(va), "suspect": len(vb), "equal": equal_n,
                        "differ": sum(classes.values()), "replaced_reference": a.get("replaced", 0),
                        "replaced_suspect": b.get("replaced", 0)},
            "meta_diff": meta, "first": differing[0] if differing else None,
            "classes": dict(classes), "differing": differing}


def format_report(res: dict) -> str:
    lines = [f"data fingerprints, {res['mode']} mode: "
             f"{res['records']['equal']} equal, {res['records']['differ']} differing records"]
    for d in res["meta_diff"]:
        lines.append(f"  header differs: {d['key']}")
    f = res["first"]
    if f is None:
        lines.append("  no record differs" + ("" if not res["meta_diff"] else " (the headers do)"))
    else:
        lines.append(f"  first difference: step {f['step']} micro {f['micro']} rank {f['rank']} "
                     f"field {f['field']}: {f['class']} ({f['meaning']})")
        lines.append("    " + json.dumps(f["detail"], default=str)[:400])
        lines.append("  by class: " + ", ".join(f"{k} {v}" for k, v in sorted(res["classes"].items())))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- OpGuard
def _order_keys(keys: Sequence[str]):
    """The per-sample keys mixed with their row index, as an int64 tensor whose
    XOR fingerprint changes when the samples change order."""
    torch = _torch()
    k = np.array([int(x, 16) for x in keys], dtype=np.uint64)
    mixed = _mix(k ^ _mix(np.arange(1, len(k) + 1, dtype=np.uint64)))
    return torch.from_numpy(mixed.view(np.int64).copy())


def emit_opguard(writer, rec: dict) -> None:
    """One OpGuard boundary record per field (name ``data.<field>``, meta step and
    micro): output 0 is the field (shape, dtype, its whole-field fingerprint),
    output 1 the order-keyed sample keys, so a change of order differs too."""
    for name, f in rec["fields"].items():
        if "sha256" in f:
            continue
        whole = ob.describe(f["shape"], f["dtype"], "cpu", [f["xor"]])
        writer.record(f"data.{name}", outputs=[whole, _order_keys(f["keys"])], file="data_fingerprint",
                      line=1, meta={"step": rec["step"], "micro": rec.get("micro", 0)})


def export(run_dir: Union[str, Path], root: Union[str, Path]) -> dict:
    """Write a recorded run in OpGuard's capture layout under ``root``, one
    ``gpu_<rank>`` directory per rank, records in step order."""
    run = load(run_dir)
    counts: Dict[int, int] = collections.Counter()
    by_rank: Dict[int, List[dict]] = {}
    for (step, micro, rank), rec in sorted(run["records"].items()):
        by_rank.setdefault(rank, []).append(rec)
    for rank, recs in by_rank.items():
        with ob.TraceWriter(root, rank=rank) as w:
            for rec in recs:
                emit_opguard(w, rec)
                counts[rank] += len(rec["fields"])
    return {"root": str(root), "records_per_rank": dict(counts)}


# ----------------------------------------------------------------------------- selftest
def selftest(verbose: bool = True) -> List[dict]:
    """The fingerprints against opguard_bridge's pure-Python reference, and the
    classifier on planted pairs."""
    torch = _torch()
    g = torch.Generator().manual_seed(0)
    cases = []

    def case(name, ok):
        cases.append({"case": name, "passed": bool(ok)})

    for dtype in (torch.int64, torch.int32, torch.bool, torch.bfloat16, torch.uint8):
        t = (torch.randint(0, 1000, (4, 7), generator=g) % (2 if dtype == torch.bool else 1000)).to(dtype)
        f = fingerprint_field(t)
        data = bytes(t.contiguous().view(torch.uint8).reshape(-1).tolist())
        case(f"{dtype} whole field = reference", [f["xor"]] == ob.reference_signature(data, 1))
        per = len(data) // 4                            # bytes per sample (4 samples)
        case(f"{dtype} rows = reference", f["rows"] == [ob.reference_signature(data[i * per:(i + 1) * per], 1)[0]
                                                        for i in range(4)])
        if per % 4 == 0:                                # whole words per sample: rows XOR to the field
            merged = 0
            for v in f["rows"]:
                merged ^= v
            case(f"{dtype} rows merge to the whole field", merged == f["xor"])
    t = torch.arange(32, dtype=torch.int64).reshape(4, 8)
    swapped = t.clone()
    swapped[0, [0, 1]] = swapped[0, [1, 0]]
    f0, f1 = fingerprint_field(t), fingerprint_field(swapped)
    case("a swap inside a row: XOR blind, keys see it", f0["rows"] == f1["rows"] and f0["keys"] != f1["keys"])

    def rec(step, batch, rank=0):
        return {"kind": "batch", "step": step, "micro": 0, "rank": rank, **fingerprint_batch(batch)}
    docs = [torch.arange(1, 6), torch.arange(10, 13), torch.arange(20, 24), torch.arange(30, 36)]

    def padded(ds, T=8):
        ids = torch.zeros(len(ds), T, dtype=torch.int64)
        mask = torch.zeros_like(ids)
        for i, d in enumerate(ds):
            ids[i, :len(d)], mask[i, :len(d)] = d, 1
        return {"input_ids": ids, "attention_mask": mask}

    def packed(ds, T=9):
        rows, pos = [[]], [[]]
        for d in ds:
            if len(rows[-1]) + len(d) > T:
                rows.append([])
                pos.append([])
            rows[-1] += d.tolist()
            pos[-1] += list(range(len(d)))
        used = [len(r) for r in rows]
        ids = torch.tensor([r + [0] * (T - len(r)) for r in rows])
        p = torch.tensor([q + [0] * (T - len(q)) for q in pos])
        mask = (torch.arange(T)[None, :] < torch.tensor(used)[:, None]).long()
        return {"input_ids": ids, "attention_mask": mask, "position_ids": p}
    a = rec(0, padded(docs))
    case("equal", classify(a, rec(0, padded(docs))) is None)
    case("order inside the batch", (classify(a, rec(0, padded(docs[::-1]))) or {}).get("class") == "order")
    other = padded([docs[0], docs[1], docs[2], torch.arange(40, 46)])
    case("samples", (classify(a, rec(0, other)) or {}).get("class") == "samples")
    case("packing", (classify(a, rec(0, packed(docs))) or {}).get("class") == "packing")
    lab = padded(docs)
    lab["labels"] = lab["input_ids"].clone()
    lab2 = {k: v.clone() for k, v in lab.items()}
    lab2["labels"][0, 0] = -100
    case("masks", (classify(rec(0, lab), rec(0, lab2)) or {}).get("class") == "masks")
    if verbose:
        bad = [c["case"] for c in cases if not c["passed"]]
        print(f"data_fingerprint selftest: {len(cases) - len(bad)}/{len(cases)} cases pass")
        for name in bad:
            print("  FAIL", name)
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fingerprints of the batches a data loader hands to training.")
    sub = ap.add_subparsers(dest="command", required=True)
    cp = sub.add_parser("compare", help="first differing (step, rank, field) and its class; exit 1 when they differ")
    cp.add_argument("reference")
    cp.add_argument("suspect")
    cp.add_argument("--global", dest="global_", action="store_true", help="merge the ranks of each step first")
    cp.add_argument("--limit", type=int, default=20, help="differing records to list")
    cp.add_argument("--json", help="write the full result here")
    ex = sub.add_parser("export", help="write a recorded run in OpGuard's capture layout")
    ex.add_argument("run")
    ex.add_argument("root")
    sub.add_parser("selftest", help="fingerprints against the reference and the classifier on planted pairs")
    args = ap.parse_args(argv)
    if args.command == "selftest":
        return 0 if all(c["passed"] for c in selftest()) else 1
    if args.command == "export":
        print(json.dumps(export(args.run, args.root)))
        return 0
    res = compare(args.reference, args.suspect, global_=args.global_, limit=args.limit)
    print(format_report(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))
    return 0 if res["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
