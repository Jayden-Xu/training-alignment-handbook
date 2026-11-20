"""bi_audit: which operators does a model run, and which of its modules are batch-variant?

For an nn.Module whose tensors are batch-first, and a function that builds a batch
from rows, the audit does three things:

1. Inventory. Every operator the forward runs (and the backward, when a loss is
   given), per module, with shapes and dtypes, through a torch dispatch mode. Each
   operator is put in an operator class of handbook/3-kernels/README.md,
   with that class's default risk.
2. A per-module L2 test. Each module's inputs are recorded in one batched forward.
   The module is then re-run on the target row alone and inside sub-batches of other
   sizes, positions and fillers, built from the recorded inputs, so the target row's
   input bits are identical every time. The target row's output bits are compared.
   Operators inside the module are compared too, in their own dtype: an fp32
   reduction that changes with the batch is reported even when the module's bf16
   output happens to round to the same bits at the rows tested. Every module gets
   its own verdict.
3. First divergence. The model runs on the target row alone and on the full batch;
   the first module, in forward order, whose target-row output differs is named,
   and every module whose inputs agree but whose output differs is listed.

    from bi_audit import audit, to_markdown
    rep = audit(model, rows, build_batch)          # rows: list; build_batch(rows) -> model input
    open("audit.md", "w").write(to_markdown(rep))

    python bi_audit.py --demo
    python bi_audit.py --factory path/to/file.py:function --json out.json --md out.md
    (the function returns dict(model=, rows=, build_batch=, and optionally name=, forward=,
     loss_fn=, sizes=, target=))

Assumptions
- Batch-first tensors: a tensor argument or output whose leading size equals the
  batch size is one row per sequence; anything else is passed unchanged. Choose a
  batch size no other leading dimension equals (the audit warns when it sees one).
- Rows have equal shapes (no padding to the longest row inside build_batch).
- Modules are pure functions of their inputs in eval mode: no cache or running
  statistic is updated by a forward, nothing is random.
- Sub-batch sizes cannot exceed the number of rows given.
- A 2-D kernel sees batch x tokens rows. Audit with one-token rows to reach small
  row counts, and with long rows for large ones.

What it cannot see
- Operators that do not go through the torch dispatcher: Triton or CUDA kernels
  called directly. Their module is still tested as a black box (step 2), and code
  can report them with bi_audit.note(name, opclass, inputs, output).
- Compiled graphs (torch.compile, TorchScript, CUDA graphs, exported engines): the
  dispatcher sees the eager ops, not what the compiler fused. Audit the deployed
  wrapper as a black box with tools/bi_check.py.
- Code between modules belongs to the enclosing module and is only tested as part
  of it. Rows regrouped inside a module (sorted by expert, packed) cannot be
  compared operator by operator; only the module's output is.
- "No difference" is an observation at the sizes and rows tested, not a proof.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.modules.setdefault("bi_audit", sys.modules[__name__])       # one module object under either name
_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
RECIPES = "skills/batch-invariant-kernel/references/operator-recipes.md"   # relative to the repository root
RECIPES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", RECIPES)


def recipes_link(md_dir: str) -> str:
    """The recipes file as a link that works from a Markdown file written in md_dir."""
    return os.path.relpath(os.path.normpath(RECIPES_FILE), os.path.abspath(md_dir))
DEFAULT_SIZES = (1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64)

# class id -> (title, default risk, recipe anchor)
CLASSES = OrderedDict([
    ("attention", ("Softmax attention", "high: SDPA changed a sequence from batch 3 and between prefill and decode (A40)",
                   "#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv")),
    ("gemm", ("Dense GEMM", "high: cuBLAS bf16 rows changed from M = 31, per shape (A40)", "#2-dense-gemm-bf16fp16")),
    ("bmm", ("Batched or grouped GEMM", "high: bmm runs other kernels than mm and varies with the piece count",
             "#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm")),
    ("scatter", ("Scatter-add / combine", "high: float atomics break run-to-run determinism", "#8-moe-routing-dispatch-and-combine")),
    ("batchstat", ("Batch statistics", "high by design: every row depends on the others", "#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy")),
    ("rowwise", ("Row-wise reduction", "medium: torch's sum/mean changes its tree below 16 rows (fp32 variance of a 1024-wide row "
                 "differed from 8 rows, 4096-wide from 2); softmax and log_softmax passed", "#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy")),
    ("selection", ("Selection (top-k, sort)", "medium: exact itself, flips when its scores vary; tie order undocumented",
                   "#6-sparse-attention-with-selection-indexer-plus-top-k")),
    ("conv", ("Convolution (closed kernel)", "unknown: library picks the algorithm by shape", "#11-closed-or-precompiled-kernels")),
    ("quant", ("Quantization", "high if scales span the batch", "#9-quantized-gemm-fp8-block-scaled-mxfp8-nvfp4")),
    ("random", ("Random", "run-to-run: needs fixed seeds and state", "")),
    ("opaque", ("Opaque (no dispatcher op seen)", "unknown: custom or closed kernel", "#11-closed-or-precompiled-kernels")),
    ("elementwise", ("Elementwise", "none expected: one element at a time", "")),
    ("cast", ("Cast (rounding point)", "none for batch invariance; a rounding point across engines", "")),
    ("exact", ("Order-free reduction (max, min, any)", "none expected", "")),
    ("move", ("Data movement", "none: exact", "")),
])
_OPS = {
    "attention": "scaled_dot_product_attention _scaled_dot_product_flash_attention _scaled_dot_product_efficient_attention "
                 "_scaled_dot_product_cudnn_attention _scaled_dot_product_flash_attention_for_cpu _native_multi_head_attention "
                 "_scaled_dot_product_flash_attention_backward _scaled_dot_product_efficient_attention_backward",
    "gemm": "mm addmm matmul linear mv addmv dot vdot inner outer ger tensordot addr",
    "bmm": "bmm baddbmm addbmm einsum",
    "scatter": "index_add index_add_ scatter_add scatter_add_ scatter_reduce scatter_reduce_ index_put index_put_ "
               "_index_put_impl_ put_ embedding_bag _embedding_bag embedding_dense_backward",
    "batchstat": "batch_norm native_batch_norm _native_batch_norm_legit _native_batch_norm_legit_no_training cudnn_batch_norm "
                 "instance_norm _batch_norm_impl_index",
    "rowwise": "sum mean var std norm linalg_vector_norm logsumexp softmax _softmax log_softmax _log_softmax prod cumsum cumprod "
               "logcumsumexp layer_norm native_layer_norm group_norm native_group_norm rms_norm cross_entropy_loss nll_loss "
               "nll_loss_forward nll_loss_nd mse_loss var_mean std_mean nansum nanmean trace "
               "_softmax_backward_data _log_softmax_backward_data nll_loss_backward native_layer_norm_backward",
    "selection": "topk sort argsort kthvalue unique unique_consecutive _unique2 msort mode median nanmedian",
    "conv": "convolution _convolution conv1d conv2d conv3d cudnn_convolution conv_transpose1d conv_transpose2d convolution_backward",
    "quant": "quantize_per_tensor quantize_per_channel fake_quantize_per_tensor_affine fake_quantize_per_channel_affine",
    "random": "dropout native_dropout bernoulli bernoulli_ multinomial rand randn randint rand_like randn_like normal_ uniform_ randperm",
    "cast": "_to_copy to type_as float half bfloat16 double",
    "exact": "max min amax amin argmax argmin any all aminmax count_nonzero",
    "move": "view reshape _unsafe_view view_as transpose permute t expand expand_as slice select unsqueeze squeeze cat stack split "
            "split_with_sizes chunk unbind index_select gather index embedding clone contiguous copy_ empty empty_like zeros "
            "zeros_like ones ones_like full full_like arange tril triu detach alias lift_fresh lift_fresh_copy pad constant_pad_nd "
            "roll flip repeat repeat_interleave unfold narrow masked_select nonzero bincount scatter scatter_ index_fill "
            "masked_scatter flatten unflatten new_empty new_zeros new_ones new_full _reshape_alias as_strided diagonal "
            "resize_ set_ fill_ zero_ linspace eye one_hot meshgrid broadcast_to _local_scalar_dense item is_nonzero "
            "result_type numel size dim movedim swapaxes mT mH T tile take take_along_dim eq ne lt le gt ge isnan isinf "
            "isfinite logical_and logical_or logical_not logical_xor bitwise_and bitwise_or bitwise_not bitwise_xor",
}
OP_CLASS = {op: cls for cls, ops in _OPS.items() for op in ops.split()}
# after one of these, a module's rows are no longer batch-major: stop comparing its intermediates
_REGROUP = set("argsort sort nonzero unique _unique2 unique_consecutive masked_select randperm index_put index_put_ _index_put_impl_ "
               "index_add index_add_ scatter_add scatter_add_ scatter scatter_ bincount".split())
# outputs of these hold uninitialized memory until a later kernel writes them: never compared
_UNINIT = set("empty empty_like new_empty empty_strided new_empty_strided".split())
_PRIORITY = list(CLASSES)                                         # earlier = more specific risk


def classify(op: str) -> str:
    base = op.split("::")[-1]
    if base in OP_CLASS:
        return OP_CLASS[base]
    if base.endswith("_") and base[:-1] in OP_CLASS:
        return OP_CLASS[base[:-1]]
    return "elementwise"


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    if t.dtype == torch.bool:
        return t.to(torch.uint8)
    if t.is_complex():
        t = torch.view_as_real(t)
    return t.view(_INT[t.element_size()])


def _mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel(), 1)
    return int((bits(a) != bits(b)).sum())


# ----------------------------------------------------------------------------- tracing
ACTIVE: Optional["_Tracer"] = None


def note(name: str, opclass: str, inputs, output) -> None:
    """Report a kernel launch the dispatcher cannot see (a Triton or CUDA kernel called directly).
    A no-op unless an audit is running. opclass is a key of CLASSES."""
    if ACTIVE is not None and not ACTIVE.paused:
        ins = [t for t in (inputs if isinstance(inputs, (tuple, list)) else (inputs,)) if torch.is_tensor(t)]
        ACTIVE.on_op(name, opclass if opclass in CLASSES else "opaque", ins, output)


class _Tracer(TorchDispatchMode):
    """Records every dispatcher op with the module that ran it. In replay mode it instead compares
    each op's target-row output with a reference trace."""

    def __init__(self):
        super().__init__()
        self.stack, self.records, self.paused = ["(top level)"], [], False
        self.bwd_module, self.in_backward = None, False
        self.mode, self.trace, self.i, self.cmp = "inventory", [], 0, None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if not self.paused:
            name = func._schema.name
            ins = [a for a in args if torch.is_tensor(a)]
            self.on_op(name, classify(name), ins, out)
        return out

    def on_op(self, name, cls, ins, out):
        outs = [o for o in (out if isinstance(out, (tuple, list)) else (out,)) if torch.is_tensor(o)]
        if self.mode == "inventory":
            mod = self.bwd_module if self.in_backward else self.stack[-1]
            self.records.append({"module": mod or "(outside modules)", "op": name, "class": cls, "backward": self.in_backward,
                                 "in": [(list(t.shape), str(t.dtype).replace("torch.", "")) for t in ins],
                                 "out": [(list(t.shape), str(t.dtype).replace("torch.", "")) for t in outs]})
        elif self.mode == "ref":
            self.trace.append((name, cls, [o.detach().clone() for o in outs]))
        elif self.mode == "cmp":
            c = self.cmp
            i, self.i = self.i, self.i + 1
            if c["stop"]:
                return
            if i >= len(self.trace) or self.trace[i][0] != name:
                c["stop"] = f"op sequence changes with the batch at op {i}: {name} vs {self.trace[i][0] if i < len(self.trace) else 'end'}"
                return
            if name.split("::")[-1] in _REGROUP:
                c["stop"] = f"rows regrouped by {name}: later intermediates not compared"
                return
            if name.split("::")[-1] in _UNINIT or (cls in ("move", "cast") and not any(o.is_floating_point() for o in outs)):
                return
            n, p = c["n"], c["p"]
            for r, o in zip(self.trace[i][2], outs):
                if o.dim() == 0 or r.dim() == 0 or r.shape[0] == 0 or o.shape[0] != n * r.shape[0] or o.shape[1:] != r.shape[1:]:
                    continue
                k = r.shape[0]
                bad = _mismatch(o[p * k:(p + 1) * k], r)
                c["compared"] += 1
                if bad and c["first"] is None:
                    c["first"] = {"op_index": i, "op": name, "class": cls, "dtype": str(o.dtype).replace("torch.", ""),
                                  "mismatched_elements": bad, "of": r.numel()}


def _attach_stack(model, tracer):
    hooks = []
    for name, mod in model.named_modules():
        label = name or "(root)"
        hooks.append(mod.register_forward_pre_hook(lambda m, a, k, label=label: (tracer.stack.append(label), None)[1], with_kwargs=True))
        hooks.append(mod.register_forward_hook(lambda m, a, k, o, label=label: (tracer.stack.pop(), None)[1],
                                               with_kwargs=True, always_call=True))
    return hooks


# ----------------------------------------------------------------------------- batch-first slicing
def _map(obj, fn):
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*[_map(o, fn) for o in obj])
    if isinstance(obj, (tuple, list)):
        return type(obj)(_map(o, fn) for o in obj)
    if isinstance(obj, dict):
        try:
            return type(obj)((k, _map(v, fn)) for k, v in obj.items())
        except Exception:  # noqa: BLE001
            return {k: _map(v, fn) for k, v in obj.items()}
    return obj


def _leaves(obj, out=None):
    out = [] if out is None else out
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _leaves(o, out)
    elif isinstance(obj, dict):
        for o in obj.values():
            _leaves(o, out)
    return out


def _take(obj, idx, B, filler="recorded", p=0):
    """Rows idx of every batch-first tensor in obj. Fillers: the other rows as recorded, zeroed, or x1000."""
    index = None

    def fn(t):
        nonlocal index
        if t.dim() == 0 or t.shape[0] != B:
            return t
        if index is None or index.device != t.device:
            index = torch.as_tensor(idx, device=t.device)
        s = t.index_select(0, index)
        if filler != "recorded" and s.is_floating_point() and len(idx) > 1:
            keep = s[p:p + 1].clone()
            s = torch.zeros_like(s) if filler == "zeros" else s * 1e3
            s[p:p + 1] = keep
        return s
    return _map(obj, fn)


def _rows(obj, n):
    return [t for t in _leaves(obj) if t.dim() > 0 and t.shape[0] == n]


def _default_forward(model, batch):
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


# ----------------------------------------------------------------------------- the three steps
def _inventory(model, batch, forward, loss_fn):
    tracer = _Tracer()
    hooks = _attach_stack(model, tracer)
    claimed, entry = {}, {}

    def seq_now():
        tracer.paused = True
        try:
            return torch.zeros(1, requires_grad=True).clone().grad_fn._sequence_nr()
        finally:
            tracer.paused = False

    def claim(label, outputs, since):
        todo = [t.grad_fn for t in _leaves(outputs) if t.grad_fn is not None]
        seen = set()
        while todo:
            node = todo.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            if node.name().endswith("AccumulateGrad") or node._sequence_nr() < since:
                continue
            if node not in claimed:
                claimed[node] = label
                node.register_prehook(lambda g, label=label: setattr(tracer, "bwd_module", label))
            todo.extend(f for f, _ in node.next_functions)

    if loss_fn is not None:
        for name, mod in model.named_modules():
            label = name or "(root)"
            hooks.append(mod.register_forward_pre_hook(
                lambda m, a, label=label: (entry.setdefault((label, len(tracer.stack)), seq_now()), None)[1]))
            hooks.append(mod.register_forward_hook(
                lambda m, a, o, label=label: claim(label, o, entry.pop((label, len(tracer.stack)), 0))))
    global ACTIVE
    ACTIVE = tracer
    try:
        with torch.set_grad_enabled(loss_fn is not None), tracer:
            out = forward(model, batch)
            if loss_fn is not None:
                loss = loss_fn(out, batch)
                claim("(loss, outside modules)", loss, 0)
                tracer.in_backward = True
                loss.backward()
    finally:
        ACTIVE = None
        for h in hooks:
            h.remove()
    return tracer.records


def _capture(model, batch, forward):
    """One forward; every module call's inputs and output, in completion order."""
    calls, counts, hooks = [], {}, []

    def hook(label):
        def fn(mod, args, kwargs, out):
            i = counts.get(label, 0)
            counts[label] = i + 1
            calls.append({"module": label, "call": i, "mod": mod, "args": args, "kwargs": kwargs, "out": out})
        return fn
    for name, mod in model.named_modules():
        hooks.append(mod.register_forward_hook(hook(name or "(root)"), with_kwargs=True))
    try:
        with torch.no_grad():
            forward(model, batch)
    finally:
        for h in hooks:
            h.remove()
    return calls


def _subbatch(target, n, pos, B):
    p = {"first": 0, "middle": n // 2, "last": n - 1}[pos]
    others = [i for i in range(B) if i != target]
    rot = (n * 7) % max(len(others), 1)
    others = (others[rot:] + others[:rot])[:n - 1]
    return others[:p] + [target] + others[p:], p


def _test_module(call, B, target, sizes, positions, fillers):
    """Replay one recorded module call on the target row alone and in sub-batches."""
    global ACTIVE
    mod, args, kwargs = call["mod"], call["args"], call["kwargs"]
    res = {"comparisons": 0, "failing_sizes": [], "first_failing_size": None, "max_fraction": 0.0,
           "inside": None, "inside_first_size": None, "notes": []}
    if not _rows((args, kwargs), B):
        return dict(res, verdict="not tested", notes=["no batch-first tensor input"])
    if not _rows(call["out"], B):
        return dict(res, verdict="not tested", notes=["no batch-first tensor output"])
    tracer = _Tracer()
    try:
        with torch.no_grad():
            a1, k1 = _take((args, kwargs), [target], B)
            tracer.mode = "ref"
            ACTIVE = tracer
            with tracer:
                ref = mod(*a1, **k1)
            ACTIVE = None
            ref_rows = _rows(ref, 1)
            again = _rows(mod(*a1, **k1), 1)
            if any(_mismatch(x, y) for x, y in zip(ref_rows, again)):
                return dict(res, verdict="not deterministic (L1)", notes=["two runs on the same row differ"])
            rec = [t[target:target + 1] for t in _rows(call["out"], B)]
            if len(rec) == len(ref_rows) and any(_mismatch(x, y) for x, y in zip(ref_rows, rec)):
                res["notes"].append("alone differs from the recorded batch output")
            for n in sizes:
                if n < 2 or n > B:
                    continue
                for pos in positions:
                    for filler in fillers:
                        idx, p = _subbatch(target, n, pos, B)
                        an, kn = _take((args, kwargs), idx, B, filler, p)
                        tracer.mode, tracer.i = "cmp", 0
                        tracer.cmp = {"n": n, "p": p, "stop": None, "first": None, "compared": 0}
                        ACTIVE = tracer
                        with tracer:
                            out = mod(*an, **kn)
                        ACTIVE = None
                        got = _rows(out, n)
                        res["comparisons"] += 1
                        bad = sum(_mismatch(g[p:p + 1], r) for g, r in zip(got, ref_rows)) if len(got) == len(ref_rows) else -1
                        tot = sum(r.numel() for r in ref_rows)
                        if bad:
                            if n not in res["failing_sizes"]:
                                res["failing_sizes"].append(n)
                            res["max_fraction"] = max(res["max_fraction"], 1.0 if bad < 0 else bad / max(tot, 1))
                        c = tracer.cmp
                        if c["first"] is not None and res["inside"] is None:
                            res["inside"], res["inside_first_size"] = c["first"], n
                        if c["stop"] and c["stop"] not in res["notes"]:
                            res["notes"].append(c["stop"])
    except Exception as ex:  # noqa: BLE001
        ACTIVE = None
        return dict(res, verdict="error", notes=[f"{type(ex).__name__}: {str(ex)[:200]}"])
    if res["failing_sizes"]:
        res["first_failing_size"] = min(res["failing_sizes"])
        res["verdict"] = "variant"
    elif res["inside"] is not None:
        res["verdict"] = "variant inside"
        res["first_failing_size"] = res["inside_first_size"]
        res["max_fraction"] = res["inside"]["mismatched_elements"] / max(res["inside"]["of"], 1)
    else:
        res["verdict"] = "no difference"
    return res


def _first_divergence(alone, batched, B, target, leaf):
    key = lambda c: (c["module"], c["call"])  # noqa: E731
    solo = {key(c): c for c in alone}
    rows, first = [], None
    for c in batched:
        s = solo.get(key(c))
        if s is None:
            continue
        ob, oa = [t[target:target + 1] for t in _rows(c["out"], B)], _rows(s["out"], 1)
        ib, ia = [t[target:target + 1] for t in _rows((c["args"], c["kwargs"]), B)], _rows((s["args"], s["kwargs"]), 1)
        if not ob or len(ob) != len(oa):
            continue
        bad = sum(_mismatch(x, y) for x, y in zip(ob, oa))
        tot = sum(t.numel() for t in oa)
        in_eq = len(ib) == len(ia) and not any(_mismatch(x, y) for x, y in zip(ib, ia))
        row = {"module": c["module"], "call": c["call"], "output_differs": bool(bad), "fraction": bad / max(tot, 1),
               "inputs_equal": in_eq, "leaf": leaf.get(c["module"], True)}
        rows.append(row)
        if bad and first is None:
            first = row
    origins = [r for r in rows if r["output_differs"] and r["inputs_equal"] and r["leaf"]]
    return {"first": first, "modules_compared": len(rows), "modules_differing": sum(r["output_differs"] for r in rows),
            "origins": origins[:50]}


def audit(model, rows: Sequence[Any], build_batch: Callable, *, target: int = 0, sizes: Sequence[int] = DEFAULT_SIZES,
          positions: Sequence[str] = ("first", "middle", "last"), fillers: Sequence[str] = ("recorded",),
          forward: Optional[Callable] = None, loss_fn: Optional[Callable] = None, name: str = "model",
          recipes: str = RECIPES) -> dict:
    """Run the three steps and return a JSON-serializable report. See the module docstring."""
    forward = forward or _default_forward
    B = len(rows)
    batch = build_batch(list(rows))
    was_training = model.training
    records = _inventory(model, batch, forward, loss_fn)
    for p in model.parameters():
        p.grad = None
    batched = _capture(model, batch, forward)
    alone = _capture(model, build_batch([rows[target]]), forward)
    leaf = {(n or "(root)"): not any(True for _ in m.children()) for n, m in model.named_modules()}
    divergence = _first_divergence(alone, batched, B, target, leaf)
    del alone

    types = {(n or "(root)"): type(m).__name__ for n, m in model.named_modules()}
    declared = {(n or "(root)"): getattr(m, "bi_class", None) for n, m in model.named_modules()}
    own = {}
    for r in records:
        e = own.setdefault(r["module"], {})
        k = (r["op"], r["class"], r["backward"])
        d = e.setdefault(k, {"op": r["op"], "class": r["class"], "backward": r["backward"], "count": 0, "in": r["in"], "out": r["out"]})
        d["count"] += 1
    warn = []
    lead = {s[0] for r in records for s, _ in r["in"] + r["out"] if s}
    modules = []
    for c in batched:
        label = c["module"]
        ops = sorted(own.get(label, {}).values(), key=lambda d: (d["backward"], _PRIORITY.index(d["class"]), d["op"]))
        fwd = [d for d in ops if not d["backward"]]
        rounding = [d["class"] for d in fwd if d["class"] not in ("move", "cast", "exact", "elementwise")]
        if rounding:
            cls = min(rounding, key=_PRIORITY.index)
        elif not leaf[label]:
            cls = "container"
        elif declared[label] in CLASSES:
            cls = declared[label]
        elif any(d["class"] == "elementwise" for d in fwd):
            cls = "elementwise"
        elif any(d["class"] == "cast" for d in fwd):
            cls = "cast"
        elif any(True for _ in c["mod"].parameters()) and not any(d["op"].endswith("embedding") for d in fwd):
            cls = "opaque"
        else:
            cls = "move"
        test = _test_module(c, B, target, sizes, positions, fillers)
        title, risk, anchor = CLASSES.get(cls, ("Container", "see its children and its own ops", ""))
        modules.append({"module": label, "call": c["call"], "type": types[label], "leaf": leaf[label], "class": cls,
                        "class_title": title, "default_risk": risk, "recipe": (recipes + anchor) if anchor else "",
                        "ops": ops, **test})
    if B in {s for s in lead if s != B} or any(B == d for r in records for s, _ in r["out"] for d in s[1:2]):
        warn.append(f"a non-leading dimension equals the batch size {B}: a tensor may be mistaken for batch-first")
    by_class = {}
    for r in records:
        e = by_class.setdefault(r["class"], {"ops": {}, "modules": set(), "backward_ops": {}})
        e["backward_ops" if r["backward"] else "ops"][r["op"]] = e["backward_ops" if r["backward"] else "ops"].get(r["op"], 0) + 1
        e["modules"].add(r["module"])
    inventory = [{"class": k, "title": CLASSES[k][0], "default_risk": CLASSES[k][1], "ops": v["ops"],
                  "backward_ops": v["backward_ops"], "modules": len(v["modules"])}
                 for k, v in sorted(by_class.items(), key=lambda kv: _PRIORITY.index(kv[0]))]
    if was_training:
        warn.append("the model is in training mode: dropout and batch statistics make replays differ")
    env = {"torch": torch.__version__, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    row_shapes = sorted({str(list(t.shape[1:])) for t in _rows(batch, B)})
    return {"schema": "bi-audit-v1", "name": name, "env": env, "batch_size": B, "target_row": target,
            "row_shapes": row_shapes, "sizes": [s for s in sizes if 2 <= s <= B], "positions": list(positions),
            "fillers": list(fillers), "backward_inventory": loss_fn is not None, "warnings": warn,
            "first_divergence": divergence, "inventory": inventory, "modules": modules,
            "summary": {v: sum(m["verdict"] == v for m in modules) for v in sorted({m["verdict"] for m in modules})}}


# ----------------------------------------------------------------------------- output
def to_markdown(rep: dict, leaf_only: bool = False) -> str:
    L = [f"# bi_audit: {rep['name']}", "",
         f"- {rep['env']['device']}, torch {rep['env']['torch']}. Batch of {rep['batch_size']} rows, row shapes "
         f"{', '.join(rep['row_shapes'])}; target row {rep['target_row']}.",
         f"- Per-module test: the target row alone against sub-batches of {rep['sizes']}, positions "
         f"{rep['positions']}, fillers {rep['fillers']}, with identical input bits.",
         f"- Verdicts: " + ", ".join(f"{v}: {n}" for v, n in rep["summary"].items()) + "."]
    L += [f"- Warning: {w}" for w in rep["warnings"]]
    d = rep["first_divergence"]
    L += ["", "## First divergence in the full forward (alone against the batch)", ""]
    if d["first"] is None:
        L.append(f"None: the target row's output is bitwise equal at all {d['modules_compared']} module calls.")
    else:
        f = d["first"]
        L.append(f"`{f['module']}` is the first module whose target-row output differs ({f['fraction']:.2%} of elements; "
                 f"its inputs were {'equal' if f['inputs_equal'] else 'already different'}). "
                 f"{d['modules_differing']} of {d['modules_compared']} module calls differ.")
        if d["origins"]:
            L.append("")
            L.append("Leaf modules whose inputs agree and whose output differs (where variance enters): "
                     + ", ".join(f"`{o['module']}` ({o['fraction']:.2%})" for o in d["origins"][:12])
                     + (" ..." if len(d["origins"]) > 12 else "") + ".")
    L += ["", "## Modules", "",
          "| Module | Type | Class | Verdict | First batch size that changes it | Fraction of elements | Recipe |",
          "|---|---|---|---|---|---|---|"]
    for m in rep["modules"]:
        if leaf_only and not m["leaf"]:
            continue
        verdict = m["verdict"]
        if m["verdict"] == "variant inside":
            i = m["inside"]
            verdict = f"variant inside: `{i['op']}` ({i['dtype']}) differs, the output did not at these rows"
        elif m["verdict"] == "variant" and m["inside"]:
            verdict = f"variant (first op: `{m['inside']['op']}`)"
        elif m["verdict"] in ("not tested", "error"):
            verdict = f"{m['verdict']}: {'; '.join(m['notes'])}"
        frac = f"{m['max_fraction']:.2%}" if m["first_failing_size"] else ""
        link = f"[recipe]({m['recipe']})" if m["recipe"] else ""
        L.append(f"| `{m['module']}`{'' if m['call'] == 0 else ' #' + str(m['call'])} | {m['type']} | {m['class_title']} | "
                 f"{verdict} | {m['first_failing_size'] or ''} | {frac} | {link} |")
    L += ["", "## Operators by class", "", "| Class | Default risk | Forward ops (calls) | Backward ops (calls) |", "|---|---|---|---|"]
    for e in rep["inventory"]:
        f = ", ".join(f"`{k.split('::')[-1]}` ({v})" for k, v in sorted(e["ops"].items(), key=lambda kv: -kv[1])[:12])
        b = ", ".join(f"`{k.split('::')[-1]}` ({v})" for k, v in sorted(e["backward_ops"].items(), key=lambda kv: -kv[1])[:12])
        L.append(f"| {e['title']} | {e['default_risk']} | {f} | {b} |")
    L += ["", "The audit sees operators that pass through the torch dispatcher or are reported with `bi_audit.note`. "
          "Compiled graphs and kernels called outside any module are not visible. \"No difference\" is an observation at "
          "the sizes and rows tested.", ""]
    return "\n".join(L)


def _jsonable(rep: dict) -> dict:
    return json.loads(json.dumps(rep, default=str))


# ----------------------------------------------------------------------------- demo and CLI
def _demo():
    """A small model with one batch-variant module by design (BatchNorm in training mode)."""
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 256)
            self.proj = nn.Linear(256, 256)
            self.norm = nn.LayerNorm(256)
            self.bn = nn.BatchNorm1d(256)
            self.head = nn.Linear(256, 100)

        def forward(self, ids):
            h = self.norm(self.proj(self.embed(ids)))
            h = self.bn(h.transpose(1, 2)).transpose(1, 2)
            return self.head(h)
    torch.manual_seed(0)
    model = Net().to(dev)
    model.bn.train()                                            # batch statistics: variant by design
    rows = [torch.randint(0, 100, (24,), device=dev) for _ in range(33)]
    return dict(model=model, rows=rows, build_batch=lambda r: torch.stack(r), name="demo (BatchNorm in training mode)",
                loss_fn=lambda out, batch: out.float().pow(2).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true", help="audit a small built-in model")
    ap.add_argument("--factory", help="file.py:function returning dict(model=, rows=, build_batch=, ...)")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--sizes", type=int, nargs="*")
    ap.add_argument("--fillers", nargs="*", default=None, help="recorded zeros large")
    ap.add_argument("--leaf-only", action="store_true", help="markdown table: leaf modules only")
    args = ap.parse_args(argv)
    if args.demo:
        spec = _demo()
    elif args.factory:
        path, fn = args.factory.rsplit(":", 1)
        sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
        s = importlib.util.spec_from_file_location("bi_audit_factory", path)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        spec = getattr(m, fn)()
    else:
        ap.print_help()
        return
    kw = {k: spec[k] for k in ("target", "sizes", "positions", "fillers", "forward", "loss_fn", "name") if k in spec}
    if args.sizes:
        kw["sizes"] = args.sizes
    if args.fillers:
        kw["fillers"] = args.fillers
    if args.md:
        kw["recipes"] = recipes_link(os.path.dirname(os.path.abspath(args.md)))
    rep = _jsonable(audit(spec["model"], spec["rows"], spec["build_batch"], **kw))
    md = to_markdown(rep, leaf_only=args.leaf_only)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md)
    d = rep["first_divergence"]["first"]
    print(f"{rep['name']}: {rep['summary']}; first divergence: {d['module'] if d else 'none'}")
    if not args.md:
        print(md)


if __name__ == "__main__":
    main()
