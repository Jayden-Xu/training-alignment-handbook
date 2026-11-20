"""ckpt_compare: compare two checkpoints of different layouts, row by row, bit for bit.

A checkpoint conversion (HF <-> Megatron-style, fused QKV or gate/up, TP/PP resharding,
a dtype cast) must copy every row of every logical parameter exactly. This tool checks it:

  For each logical parameter, the rows (last-dimension vectors) on side A and on side B
  are matched as multisets by their exact bits, so repacking (rows permuted, split,
  concatenated, sharded) is not a difference. Every row left over is listed by logical
  parameter and row index, with the physical tensor and row it came from.

It also reports dtype changes (a widening cast is exact and compared after widening; any
other change is compared in the common wider type, so every row that lost bits is named),
replicated copies that differ, tensors no logical parameter uses, and, where both sides
have the same shape, whether the tensors are bit-equal position by position.

What a row multiset cannot see: a row moved to another place inside the same logical
parameter (two heads swapped, a RoPE row permutation). Run a forward on fixed inputs at
the target layout for that (handbook/5-scenarios/checkpoint-conversion.md).

Row keys. ``--rows digest`` (the default) hashes each row's bytes in order (BLAKE2b,
128 bits). ``--rows xor`` uses OpGuard's ``lastdim-u32le-v1`` fingerprints from
opguard_bridge.py, the values OpGuard's weight_sync records hold, cheap on a GPU. XOR has
two blind spots inside a row: 32-bit words reordered (row-parallel shards joined in the
wrong order), and two changes with the same bit pattern in the same lane, which cancel
(a cast that rounds two elements of a row by the same last bit). Both were measured
(examples/ckpt_conversion/REPORT.md).

Usage
    python tools/ckpt_compare.py A B                              # same names on both sides
    python tools/ckpt_compare.py A B --map spec.json --layouts hf mcore --set b:r=2 --json out.json
    python tools/ckpt_compare.py A B --scope global               # every row of A somewhere in B
    python tools/ckpt_compare.py A B --rows xor                   # OpGuard's row fingerprints

    from ckpt_compare import load, load_map, compare, format_report
    rep = compare(load(a_dir), load(b_dir), load_map("spec.json"), layouts=("hf", "mcore"),
                  vars_b={"r": 4})

A and B: a .pt/.pth/.bin file (torch.load, weights only unless --unsafe-load), a
.safetensors file, a directory of them, or (Python API) a dict of tensors. In a directory,
a file in a subdirectory contributes its tensors as "<subdir>/<name>"; nested dicts are
flattened with "."; values that are not tensors are skipped.

Mapping spec (JSON, schema ckpt-map-v1):

    {"schema": "ckpt-map-v1",
     "vars":    {"i": 2, "hd": 16, "qrows": 32},     # defaults; --set [a:|b:]name=value
     "layouts": {"hf": {}, "mcore": {"r": 2}},       # per-layout var defaults
     "params":  {"model.layers.{i}.self_attn.q_proj.weight": {
                    "hf": "=",
                    "mcore": {"cat": {"over": "r", "each": {
                        "part": "mp_rank_{r:02d}/model.decoder.layers.{i}.self_attention.linear_qkv.weight",
                        "pattern": ["{qrows}", "{hd}", "{hd}"], "take": 0}}}}}}

A parameter name is expanded over every var it mentions (i takes range(vars["i"])). An
expression is one of
    "name"                       a tensor; "=" is the logical name itself; str.format with vars
    {"cat": [e, ...] | {"over": v, "each": e}, "dim": 0}
                                 concatenation along dim (over: v takes range(vars[v]))
    {"part": e, "dim": 0, "pattern": [n0, n1, ...] | "parts": k, "take": j}
                                 dim is a repetition of groups with these sizes (or one group
                                 of k equal parts); keep piece j of every group
    {"slice": e, "dim": 0, "start": a, "stop": b}       e.g. vocabulary padding
    {"replicated": [e, ...] | {"over": v, "each": e}}   copies that must be bit-equal; the first
Sizes are ints, or strings formatted with vars and evaluated as integer arithmetic ("{hd}*2").

The report (schema ckpt-compare-v1) states facts; equality is exact bits, no tolerance.
Exit status: 0 when every logical parameter is equal or has all rows matched, 1 otherwise
(with --strict, also when a tensor is not used by any logical parameter).
"""
from __future__ import annotations

import argparse
import ast
import bisect
import collections
import hashlib
import itertools
import json
import operator
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opguard_bridge import MASK32, VECTOR_SCHEME, xor_vectors  # noqa: E402

SCHEMA = "ckpt-compare-v1"
MAP_SCHEMA = "ckpt-map-v1"
EXTENSIONS = (".pt", ".pth", ".bin", ".safetensors")
OK_STATUS = ("equal", "rows_matched")
NOTE = ("Rows are last-dimension vectors, matched per logical parameter as multisets by exact "
        "(row bytes, dtype, key): permuted, split, concatenated or sharded rows are not differences. "
        "A row moved inside one logical parameter is not seen. With rows='xor', 32-bit words "
        "reordered inside a row, and equal changes in the same lane of a row, are not seen either. "
        "Zero rows match as values but are never evidence of a correct placement.")


def _torch():
    import torch
    return torch


# ============================================================================ loading
def _flatten(obj, prefix: str, out: list, skipped: list) -> None:
    torch = _torch()
    if isinstance(obj, torch.Tensor):
        out.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out, skipped)
    else:
        skipped.append(prefix)


def _read(path: Path, unsafe: bool):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    torch = _torch()
    try:
        return torch.load(str(path), map_location="cpu", weights_only=not unsafe)
    except TypeError:                                   # torch without weights_only
        return torch.load(str(path), map_location="cpu")


def load(path, unsafe: bool = False) -> Dict[str, Any]:
    """{name: tensor} from a checkpoint file or directory (see the module docstring)."""
    if isinstance(path, dict):
        return dict(path)
    root = Path(path).expanduser()
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix in EXTENSIONS)
    else:
        raise FileNotFoundError(f"no checkpoint at {path}")
    if not files:
        raise FileNotFoundError(f"no {'/'.join(EXTENSIONS)} files under {path}")
    tensors: Dict[str, Any] = {}
    for f in files:
        prefix = "" if root.is_file() or f.parent == root else f.parent.relative_to(root).as_posix() + "/"
        items, skipped = [], []
        _flatten(_read(f, unsafe), "", items, skipped)
        for name, t in items:
            key = prefix + name
            if key in tensors:
                raise ValueError(f"tensor {key!r} appears twice under {path}")
            tensors[key] = t.detach().cpu()
    return tensors


def load_map(path) -> dict:
    with open(Path(path).expanduser()) as f:
        spec = json.load(f)
    if spec.get("schema") != MAP_SCHEMA:
        raise ValueError(f"{path}: schema must be {MAP_SCHEMA}")
    return spec


# ============================================================================ row keys
def _rows2d(t):
    """The tensor as [rows, width]; a 0-dim tensor is one row of one element."""
    t = t.reshape(1) if t.dim() == 0 else t
    return t.contiguous().reshape(-1, t.shape[-1]) if t.numel() else t.reshape(0, max(t.shape[-1], 1))


def row_keys(t, rows: str = "digest") -> List[tuple]:
    """One hashable key per row: (kind, row bytes, dtype, value). rows: 'digest' or 'xor'.

    'digest' is BLAKE2b-128 of the row's bytes in order. 'xor' is OpGuard's lastdim-u32le-v1
    fingerprint (opguard_bridge.xor_vectors); a row whose byte width is not a multiple of 4
    falls back to 'digest' (both sides alike, since only rows of equal width can match).
    """
    torch = _torch()
    r2 = _rows2d(t)
    n, width = r2.shape
    vb = width * r2.element_size()
    dtype = str(r2.dtype).replace("torch.", "")
    if n == 0:
        return []
    if rows == "xor" and vb % 4 == 0:
        v = xor_vectors(r2, impl="torch")
        return [("xor", vb, dtype, x & MASK32) for x in v.to(torch.int64).tolist()]
    if rows not in ("xor", "digest"):
        raise ValueError("rows must be 'xor' or 'digest'")
    raw = r2.view(torch.uint8).numpy().tobytes()
    return [("blake2b", vb, dtype, hashlib.blake2b(raw[i * vb:(i + 1) * vb], digest_size=16).hexdigest())
            for i in range(n)]


def _zero_rows(t) -> int:
    torch = _torch()
    r2 = _rows2d(t)
    if not r2.numel():
        return 0
    return int((r2.view(torch.uint8) == 0).all(dim=1).sum())


def _bits_equal(a, b) -> bool:
    torch = _torch()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if not a.numel():
        return True
    size = a.element_size()
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[size]
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


# ============================================================================ expressions
_FIELDS = string.Formatter()
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}


def _fields(template: str) -> List[str]:
    return [f.split(".")[0].split("[")[0] for _, f, _, _ in _FIELDS.parse(template) if f]


def _int_expr(text: str) -> int:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            num, den = ev(node.left), ev(node.right)
            if num % den:
                raise ValueError(f"{text!r}: {num}/{den} is not an integer")
            return num // den
        raise ValueError(f"size expression {text!r}: only integers and + - * / //")
    return ev(ast.parse(text, mode="eval"))


class _Registry:
    """Global row ids: every physical tensor's rows get a contiguous id range."""

    def __init__(self):
        self.names: List[str] = []
        self.starts: List[int] = []
        self.base: Dict[str, int] = {}
        self.next = 0

    def ids(self, name: str, t):
        torch = _torch()
        r2 = _rows2d(t)
        if name not in self.base:
            self.base[name] = self.next
            self.names.append(name)
            self.starts.append(self.next)
            self.next += r2.shape[0]
        shape = tuple(t.shape[:-1]) if t.dim() else ()
        return (self.base[name] + torch.arange(r2.shape[0], dtype=torch.int64)).reshape(shape + (1,))

    def origin(self, gid: int) -> str:
        k = bisect.bisect_right(self.starts, gid) - 1
        return f"{self.names[k]}[{gid - self.starts[k]}]"


class _Side:
    """Evaluates the expressions of one layout against one checkpoint."""

    def __init__(self, tensors: Dict[str, Any], layout: Optional[str], env: Dict[str, int]):
        self.tensors, self.layout, self.env = tensors, layout, env
        self.used: set = set()
        self.registry = _Registry()

    def _fmt(self, template: str, env: dict) -> str:
        try:
            return template.format(**env)
        except KeyError as e:
            raise ValueError(f"layout {self.layout!r}: var {e.args[0]!r} is not set (--set)") from None

    def _size(self, v, env) -> int:
        return v if type(v) is int else _int_expr(self._fmt(str(v), env))

    def _over(self, spec, env) -> List[Tuple[dict, Any]]:
        if isinstance(spec, list):
            return [(env, e) for e in spec]
        var = spec["over"]
        if var not in env:
            raise ValueError(f"layout {self.layout!r}: var {var!r} is not set (--set)")
        return [({**env, var: k}, spec["each"]) for k in range(int(env[var]))]

    def eval(self, expr, env: dict, logical: str, problems: list):
        """(tensor, ids) where ids[..., p] are the global row ids of the parts each row came from,
        or None when a tensor is missing (recorded in problems)."""
        torch = _torch()
        if isinstance(expr, str):
            name = logical if expr == "=" else self._fmt(expr, env)
            if name not in self.tensors:
                problems.append({"kind": "missing", "tensor": name})
                return None
            self.used.add(name)
            t = self.tensors[name]
            return t, self.registry.ids(name, t)
        if not isinstance(expr, dict) or len({"cat", "part", "slice", "replicated"} & set(expr)) != 1:
            raise ValueError(f"bad expression for {logical!r}: {expr!r}")
        if "cat" in expr:
            parts = [self.eval(e, en, logical, problems) for en, e in self._over(expr["cat"], env)]
            if any(p is None for p in parts):
                return None
            nd = parts[0][0].dim()
            dim = expr.get("dim", 0) % max(nd, 1)
            t = torch.cat([p[0] for p in parts], dim=dim)
            if dim == nd - 1:                             # joined along the row: a row has several parts
                ids = torch.cat([p[1] for p in parts], dim=-1)
            else:
                ids = torch.cat([p[1] for p in parts], dim=dim)
            return t, ids
        if "part" in expr or "slice" in expr:
            got = self.eval(expr["part"] if "part" in expr else expr["slice"], env, logical, problems)
            if got is None:
                return None
            t, ids = got
            dim = expr.get("dim", 0) % max(t.dim(), 1)
            n = t.shape[dim]
            if "slice" in expr:
                lo = self._size(expr.get("start", 0), env)
                hi = self._size(expr.get("stop", n), env)
                index = torch.arange(lo, min(hi, n))
            else:
                if "parts" in expr:
                    k = self._size(expr["parts"], env)
                    if n % k:
                        raise ValueError(f"{logical!r}: dim {dim} of size {n} does not split into {k} parts")
                    pattern = [n // k] * k
                else:
                    pattern = [self._size(s, env) for s in expr["pattern"]]
                group = sum(pattern)
                if n % group:
                    problems.append({"kind": "pattern", "detail": f"dim {dim} has {n} entries, not a multiple "
                                                                  f"of the group size {group}"})
                    return None
                j = int(expr["take"])
                off = sum(pattern[:j])
                index = torch.cat([torch.arange(g * group + off, g * group + off + pattern[j])
                                   for g in range(n // group)])
            t = t.index_select(dim, index)
            if dim < t.dim() - 1:
                ids = ids.index_select(dim, index)
            return t, ids
        copies = [self.eval(e, en, logical, problems) for en, e in self._over(expr["replicated"], env)]
        if any(c is None for c in copies):
            return None
        first, first_ids = copies[0]
        for t, ids in copies[1:]:
            if not _bits_equal(first, t):
                rows = []
                if t.shape == first.shape and t.dtype == first.dtype:
                    a2, b2 = _rows2d(first), _rows2d(t)
                    diff = (a2.view(torch.uint8) != b2.view(torch.uint8)).any(dim=1)
                    rows = diff.nonzero().reshape(-1).tolist()
                problems.append({"kind": "replica_mismatch",
                                 "copy": self.registry.origin(int(ids.reshape(-1)[0])).rsplit("[", 1)[0],
                                 "first": self.registry.origin(int(first_ids.reshape(-1)[0])).rsplit("[", 1)[0],
                                 "rows": rows[:64], "row_count": len(rows),
                                 "shapes": [list(first.shape), list(t.shape)]})
        return first, first_ids


def _expand(name_template: str, env: dict) -> List[Tuple[str, dict]]:
    fields = sorted(set(_fields(name_template)))
    for f in fields:
        if f not in env:
            raise ValueError(f"parameter {name_template!r}: var {f!r} is not set (--set)")
    out = []
    for values in itertools.product(*[range(int(env[f])) for f in fields]):
        e = {**env, **dict(zip(fields, values))}
        out.append((name_template.format(**e), e))
    return out


# ============================================================================ comparison
def _match(keys_a: list, keys_b: list) -> Tuple[List[int], List[int]]:
    """Indices of rows of a and of b left over after multiset matching by exact key."""
    pool = collections.Counter(keys_b)
    left_a = []
    for i, k in enumerate(keys_a):
        if pool[k]:
            pool[k] -= 1
        else:
            left_a.append(i)
    pool = collections.Counter(keys_a)
    left_b = []
    for i, k in enumerate(keys_b):
        if pool[k]:
            pool[k] -= 1
        else:
            left_b.append(i)
    return left_a, left_b


def _common(ta, tb):
    """Both tensors in one dtype for row matching, and a description of the dtype change."""
    torch = _torch()
    da, db = ta.dtype, tb.dtype
    if da == db:
        return ta, tb, None
    info = {"a": str(da).replace("torch.", ""), "b": str(db).replace("torch.", "")}
    if not (da.is_floating_point and db.is_floating_point):
        info.update(change="incompatible", compared_as="stored")
        return ta, tb, info
    common = torch.promote_types(da, db)
    info["compared_as"] = str(common).replace("torch.", "")
    info["change"] = ("widening (exact): rows compared after widening A" if common == db else
                      "narrowing: rows compared after widening B; a row that lost bits is unmatched" if common == da
                      else "different formats: both widened; a row whose value changed is unmatched")
    return ta.to(common), tb.to(common), info


def _origins(registry: _Registry, ids, rows: Sequence[int]) -> List[str]:
    flat = ids.reshape(-1, ids.shape[-1])
    return [" + ".join(registry.origin(int(g)) for g in flat[i].tolist()) for i in rows]


def _compare_one(name: str, got_a, got_b, sa: _Side, sb: _Side, rows: str, max_rows: int,
                 problems_a: list, problems_b: list) -> dict:
    item: Dict[str, Any] = {"name": name}
    if problems_a or problems_b:
        item["problems"] = {"a": problems_a, "b": problems_b}
    if got_a is None or got_b is None:
        item["status"] = "missing" if any(p["kind"] == "missing" for p in problems_a + problems_b) else "unreadable"
        return item
    (ta, ida), (tb, idb) = got_a, got_b
    item["a"] = {"shape": list(ta.shape), "dtype": str(ta.dtype).replace("torch.", "")}
    item["b"] = {"shape": list(tb.shape), "dtype": str(tb.dtype).replace("torch.", "")}
    ca, cb, dinfo = _common(ta, tb)
    if dinfo:
        item["dtype_change"] = dinfo
    wa = ta.shape[-1] if ta.dim() else 1
    wb = tb.shape[-1] if tb.dim() else 1
    if wa != wb:
        item["status"] = "width_mismatch"
        item["note"] = ("rows have different widths; join shards split along the last dimension with "
                        "{\"cat\": ..., \"dim\": -1} in the map")
        return item
    ka, kb = row_keys(ca, rows), row_keys(cb, rows)
    left_a, left_b = _match(ka, kb)
    for side, t, ids, left, s in (("a", ca, ida, left_a, sa), ("b", cb, idb, left_b, sb)):
        item[side].update({"rows": len(ka if side == "a" else kb), "zero_rows": _zero_rows(t),
                           "unmatched_count": len(left), "unmatched_rows": left[:max_rows],
                           "unmatched_origin": _origins(s.registry, ids, left[:max_rows]),
                           "truncated": len(left) > max_rows})
    if ta.shape == tb.shape:
        item["positional_equal"] = _bits_equal(ca, cb)
    if any(p["kind"] == "replica_mismatch" for p in problems_a + problems_b):
        item["status"] = "replica_mismatch"
    elif left_a or left_b:
        item["status"] = "rows_unmatched"
    elif item.get("positional_equal") and not dinfo:
        item["status"] = "equal"
    else:
        item["status"] = "rows_matched"
    return item


def compare(a, b, spec: Optional[dict] = None, layouts: Optional[Sequence[str]] = None,
            vars_a: Optional[dict] = None, vars_b: Optional[dict] = None, *, scope: str = "logical",
            rows: str = "digest", max_rows: int = 64) -> dict:
    """Compare checkpoints a and b ({name: tensor} or paths). See the module docstring."""
    a, b = load(a), load(b)
    if scope == "global":
        return _compare_global(a, b, rows, max_rows)
    if scope != "logical":
        raise ValueError("scope must be 'logical' or 'global'")
    if spec is None:
        spec = {"schema": MAP_SCHEMA, "params": {n: {"a": "=", "b": "="} for n in sorted(set(a) | set(b))}}
        layouts = ("a", "b")
    if not layouts or len(layouts) != 2:
        names = sorted(spec.get("layouts", {}))
        raise ValueError(f"give two layouts of the map (it has {names})")
    base = {k: int(v) for k, v in spec.get("vars", {}).items()}
    envs = []
    for lay, extra in zip(layouts, (vars_a, vars_b)):
        env = {**base, **{k: int(v) for k, v in spec.get("layouts", {}).get(lay, {}).items()},
               **{k: int(v) for k, v in (extra or {}).items()}}
        envs.append(env)
    sa, sb = _Side(a, layouts[0], envs[0]), _Side(b, layouts[1], envs[1])
    expand_env = {k: v for k, v in envs[0].items() if envs[1].get(k) == v}
    params = []
    for template, by_layout in spec["params"].items():
        for lay in layouts:
            if lay not in by_layout:
                raise ValueError(f"parameter {template!r} has no expression for layout {lay!r}")
        for name, env in _expand(template, expand_env):
            pa, pb = [], []
            got_a = sa.eval(by_layout[layouts[0]], {**envs[0], **{k: env[k] for k in _fields(template)}}, name, pa)
            got_b = sb.eval(by_layout[layouts[1]], {**envs[1], **{k: env[k] for k in _fields(template)}}, name, pb)
            params.append(_compare_one(name, got_a, got_b, sa, sb, rows, max_rows, pa, pb))
    counts = collections.Counter(p["status"] for p in params)
    unused_a = sorted(set(a) - sa.used)
    unused_b = sorted(set(b) - sb.used)
    dtype_changes = collections.Counter(p["dtype_change"]["change"].split(":")[0] for p in params
                                        if "dtype_change" in p)
    summary = {"params": len(params), **{s: counts.get(s, 0) for s in
               ("equal", "rows_matched", "rows_unmatched", "width_mismatch", "replica_mismatch", "missing",
                "unreadable")},
               "unmatched_rows_a": sum(p.get("a", {}).get("unmatched_count", 0) for p in params),
               "unmatched_rows_b": sum(p.get("b", {}).get("unmatched_count", 0) for p in params),
               "dtype_changes": dict(dtype_changes), "unused_a": len(unused_a), "unused_b": len(unused_b)}
    return {"schema": SCHEMA, "scope": "logical", "rows": rows, "fingerprint_scheme":
            VECTOR_SCHEME if rows == "xor" else "blake2b-128 of row bytes", "layouts": list(layouts),
            "vars": {"a": envs[0], "b": envs[1]}, "summary": summary,
            "all_rows_matched": all(p["status"] in OK_STATUS for p in params),
            "params": params, "unused": {"a": unused_a[:200], "b": unused_b[:200]}, "note": NOTE}


def _compare_global(a: dict, b: dict, rows: str, max_rows: int) -> dict:
    """Every row of every tensor of a against all rows of b, as one multiset per side."""
    sides = {}
    for side, tensors in (("a", a), ("b", b)):
        keyed = []
        for name in sorted(tensors):
            keyed.append((name, tensors[name], row_keys(tensors[name], rows)))
        sides[side] = keyed
    pools = {s: collections.Counter(k for _, _, keys in sides[s] for k in keys) for s in sides}
    report = {}
    for side, other in (("a", "b"), ("b", "a")):
        pool, items = pools[other].copy(), []
        for name, t, keys in sides[side]:
            left = []
            for i, k in enumerate(keys):
                if pool[k]:
                    pool[k] -= 1
                else:
                    left.append(i)
            items.append({"name": name, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                          "rows": len(keys), "zero_rows": _zero_rows(t), "unmatched_count": len(left),
                          "unmatched_rows": left[:max_rows], "truncated": len(left) > max_rows,
                          "status": "rows_unmatched" if left else "rows_matched"})
        report[side] = items
    summary = {f"{s}_{k}": v for s in ("a", "b") for k, v in
               (("tensors", len(report[s])), ("rows", sum(i["rows"] for i in report[s])),
                ("unmatched_rows", sum(i["unmatched_count"] for i in report[s])))}
    return {"schema": SCHEMA, "scope": "global", "rows": rows, "summary": summary,
            "all_rows_matched": not (summary["a_unmatched_rows"] or summary["b_unmatched_rows"]),
            "a": report["a"], "b": report["b"],
            "note": NOTE + " Global scope: rows of equal width and dtype only; rows split along the last "
                           "dimension (row-parallel shards) cannot match their full rows."}


# ============================================================================ text report
def format_report(rep: dict, limit: int = 8) -> str:
    lines = []
    if rep["scope"] == "global":
        s = rep["summary"]
        lines.append(f"global: {s['a_rows']} rows in A, {s['b_rows']} in B; unmatched: "
                     f"{s['a_unmatched_rows']} in A, {s['b_unmatched_rows']} in B ({rep['rows']})")
        for side in ("a", "b"):
            for it in rep[side]:
                if it["unmatched_count"]:
                    lines.append(f"  {side.upper()} {it['name']}: {it['unmatched_count']} of {it['rows']} rows "
                                 f"unmatched, e.g. rows {it['unmatched_rows'][:limit]}")
        return "\n".join(lines)
    s = rep["summary"]
    lines.append(f"{s['params']} logical parameters ({rep['layouts'][0]} vs {rep['layouts'][1]}, rows={rep['rows']}): "
                 f"{s['equal']} equal, {s['rows_matched']} rows matched, {s['rows_unmatched']} with unmatched rows, "
                 f"{s['width_mismatch']} width mismatch, {s['replica_mismatch']} replica mismatch, "
                 f"{s['missing']} missing")
    if s["dtype_changes"]:
        lines.append(f"  dtype changes: {s['dtype_changes']}")
    for p in rep["params"]:
        if p["status"] in OK_STATUS:
            continue
        lines.append(f"  {p['name']}: {p['status']}")
        for side in ("a", "b"):
            info = p.get(side) or {}
            if info.get("unmatched_count"):
                shown = [f"row {r} <- {o}" for r, o in zip(info["unmatched_rows"][:limit],
                                                            info["unmatched_origin"][:limit])]
                more = info["unmatched_count"] - len(shown)
                lines.append(f"    {side.upper()} {info['unmatched_count']} of {info['rows']} rows unmatched: "
                             + "; ".join(shown) + (f"; ... {more} more" if more > 0 else ""))
        for side in ("a", "b"):
            for prob in (p.get("problems") or {}).get(side, []):
                lines.append(f"    {side.upper()} {prob['kind']}: " +
                             ", ".join(f"{k}={v}" for k, v in prob.items() if k != "kind"))
        if p["status"] == "width_mismatch":
            lines.append(f"    shapes {p['a']['shape']} vs {p['b']['shape']}: {p['note']}")
    for side in ("a", "b"):
        if rep["unused"][side]:
            n = s[f"unused_{side}"]
            lines.append(f"  {n} tensors of {side.upper()} used by no logical parameter, e.g. "
                         f"{rep['unused'][side][:limit]}")
    lines.append("all rows matched" if rep["all_rows_matched"] else "NOT all rows matched")
    return "\n".join(lines)


def _parse_set(items: Sequence[str]) -> Tuple[dict, dict]:
    va, vb = {}, {}
    for item in items or []:
        side, _, rest = item.rpartition(":")
        name, eq, value = rest.partition("=")
        if not eq or side not in ("", "a", "b"):
            raise SystemExit(f"--set {item!r}: use [a:|b:]name=value")
        for target in ((va, vb) if side == "" else (va,) if side == "a" else (vb,)):
            target[name] = int(value)
    return va, vb


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="checkpoint A: file or directory")
    ap.add_argument("b", help="checkpoint B: file or directory")
    ap.add_argument("--map", help="mapping spec (ckpt-map-v1); without it, logical names are tensor names")
    ap.add_argument("--layouts", nargs=2, metavar=("LAYOUT_A", "LAYOUT_B"), help="the map's layouts of A and B")
    ap.add_argument("--set", action="append", default=[], metavar="[a:|b:]NAME=VALUE", help="set a map var")
    ap.add_argument("--scope", choices=("logical", "global"), default="logical")
    ap.add_argument("--rows", choices=("digest", "xor"), default="digest",
                    help="row key: a hash of the row bytes in order (default), or OpGuard's lastdim-u32le-v1 "
                         "XOR fingerprint (blind to words reordered in a row and to equal changes in one lane)")
    ap.add_argument("--max-rows", type=int, default=64, help="unmatched rows listed per parameter and side")
    ap.add_argument("--strict", action="store_true", help="also fail when a tensor is used by no parameter")
    ap.add_argument("--unsafe-load", action="store_true", help="torch.load with weights_only=False (trusted files)")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)
    va, vb = _parse_set(args.set)
    spec = load_map(args.map) if args.map else None
    rep = compare(load(args.a, args.unsafe_load), load(args.b, args.unsafe_load), spec,
                  args.layouts, va, vb, scope=args.scope, rows=args.rows, max_rows=args.max_rows)
    print(format_report(rep))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"wrote {args.json}")
    unused = rep["scope"] == "logical" and (rep["summary"]["unused_a"] or rep["summary"]["unused_b"])
    return 0 if rep["all_rows_matched"] and not (args.strict and unused) else 1


if __name__ == "__main__":
    sys.exit(main())
