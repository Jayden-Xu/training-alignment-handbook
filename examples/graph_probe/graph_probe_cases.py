"""graph_probe_cases: fingerprints from inside torch.compile and CUDA graphs, on the whole model.

A static-shape decode step built from examples/whole_model/model.py's modules
(B rows, one token per row, preallocated KV cache) carries tools/graph_probe.py
probes after each layer's attention norm, attention residual, MLP norm and
output, and on the logits. The probe records are written with
tools/opguard_bridge.py's TraceWriter and compared by OpGuard (backend
``compare``, harness first_diff / inspect_event / compare_events).

Parts
 0. probe     the probe's fingerprints equal opguard_bridge's (and OpGuard's own when
              importable) for the same tensors, in eager and compiled runs
 1. modes     stock op set: eager vs torch.compile (default, max-autotune-no-cudagraphs,
              reduce-overhead) vs a manually captured torch.cuda.CUDAGraph; where the
              compiled and graph bits first part from eager, layer by layer
 2. buckets   one request in row 0 of CUDA graphs captured at padded batch 1, 8, 16
              (stock and bi op sets); the first differing layer
 3. opaque    bi op set with its kernels (and then every rounding op) behind
              torch.library custom ops: compile = eager through the probes
 4. cost      probes on vs off: logits bitwise, Inductor kernel count, ms per step

    python graph_probe_cases.py --quick --opguard-root ~/OpGuard --json results-quick.json
    python graph_probe_cases.py --opguard-root ~/OpGuard --json results-a40.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tools"))
sys.path.insert(0, str(HERE.parent / "whole_model"))
import model as wm  # noqa: E402
import graph_probe as gp  # noqa: E402
import opguard_bridge as ob  # noqa: E402
from bi_report import Results  # noqa: E402


# ----------------------------------------------------------------------------- opaque wrappers
_FNS = {}


@torch.library.custom_op("gp_demo::call", mutates_args=())
def _call(fid: int, xs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [y.contiguous() for y in _FNS[fid][0](*xs)]


@_call.register_fake
def _(fid, xs):
    return list(_FNS[fid][1](*xs))


def opaque(fn, fake):
    """fn(*tensors) -> tuple of tensors, run as one opaque custom op; fake gives empty outputs."""
    fid = len(_FNS)
    _FNS[fid] = (fn, fake)
    return lambda *xs: torch.ops.gp_demo.call(fid, list(xs))


# ----------------------------------------------------------------------------- the decode step
def attn_stock(q, kc, vc, pos, scale):
    """q [B, Hq, D]; kc, vc [B, cap, Hk, D]; keys at index <= pos. SDPA on K/V expanded to Hq heads."""
    B, Hq, D = q.shape
    cap, Hk = kc.shape[1], kc.shape[2]
    K = wm._repeat_kv(kc.transpose(1, 2), Hq // Hk)
    V = wm._repeat_kv(vc.transpose(1, 2), Hq // Hk)
    mask = (torch.arange(cap, device=q.device)[None, :] <= pos[:, None])[:, None, None, :]
    return F.scaled_dot_product_attention(q[:, :, None, :], K, V, attn_mask=mask, scale=scale)[:, :, 0]


def attn_bi(q, kc, vc, pos, scale):
    """The model's per-sequence-position kernel (examples/attention_serving), one page per row."""
    B, dev = q.shape[0], q.device
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    bt = torch.arange(B, device=dev, dtype=torch.int32).view(-1, 1)
    return wm._serve_launch(q, kc, vc, cu, pos.to(torch.int32), 1, wm.MODE_PAGED, block_table=bt, page=kc.shape[1])


class Kit:
    """How each op of the decode step runs. The modules are examples/whole_model's."""

    def __init__(self, model, kind: str):
        self.kind = kind
        ops = "stock" if kind == "stock" else "bi"
        model.set_ops(ops)
        self.attn_fn = attn_stock if kind == "stock" else attn_bi
        self.norm_fn = lambda mod, x: mod(x)
        self.linear_fn = lambda mod, x: mod(x)
        self.rope_fn = lambda mod, q, k, pos: mod(q, k, pos)
        self.silu_fn = lambda mod, g, u: mod(g, u)
        self.add_fn = lambda blk, s, o: blk.add(s, o)
        if kind in ("bi_opaque_kernels", "bi_opaque_all"):
            self._wrap_kernels(model)
        if kind == "bi_opaque_all":
            self._wrap_all(model)

    def _wrap_kernels(self, model):
        """GEMMs, the RMSNorm row sum and attention as opaque custom ops; one wrapper per
        module, attached to it (Dynamo reads the attribute, the op runs the module in eager)."""
        for mod in model.modules():
            if isinstance(mod, wm.Linear):
                n = mod.weight.shape[0]
                object.__setattr__(mod, "_gp_op", opaque(lambda x, m=mod: (m(x),),
                                                         lambda x, n=n: (x.new_empty(*x.shape[:-1], n),)))
        self.linear_fn = lambda mod, x: mod._gp_op(x)[0]
        op_rs = opaque(lambda x: (wm.row_sum(x, square=True),), lambda x: (x.new_empty(x.shape[0]),))

        def norm(mod, x):                                       # RMSNorm math as in model.py, row sum opaque
            xf = x.float()
            K = xf.shape[-1]
            ms = op_rs(xf.reshape(-1, K))[0].view(*xf.shape[:-1], 1) * (1.0 / K)
            y = xf * torch.rsqrt(ms + mod.ctx.cfg.rms_eps)
            return mod.weight * y.to(mod.ctx.dt)
        self.norm_fn = norm
        sc = model.cfg.head_dim ** -0.5
        op_attn = opaque(lambda q, kc, vc, pos: (attn_bi(q, kc, vc, pos, sc),),
                         lambda q, kc, vc, pos: (q.new_empty(q.shape),))
        self.attn_fn = lambda q, kc, vc, pos, scale: op_attn(q, kc, vc, pos)[0]

    def _wrap_all(self, model):
        """Every rounding op opaque too: RMSNorm, RoPE, SiLU-mul, residual add."""
        for mod in model.modules():
            if isinstance(mod, wm.RMSNorm):
                object.__setattr__(mod, "_gp_op", opaque(lambda x, m=mod: (m(x),), lambda x: (x.new_empty(x.shape),)))
        self.norm_fn = lambda mod, x: mod._gp_op(x)[0]
        rope_mod = model.model.layers[0].self_attn.rope
        op_rope = opaque(lambda q, k, pos: rope_mod(q, k, pos),
                         lambda q, k, pos: (q.new_empty(q.shape), k.new_empty(k.shape)))
        self.rope_fn = lambda mod, q, k, pos: tuple(op_rope(q, k, pos))
        act = model.model.layers[0].mlp.act
        op_silu = opaque(lambda g, u: (act(g, u),), lambda g, u: (g.new_empty(g.shape),))
        self.silu_fn = lambda mod, g, u: op_silu(g, u)[0]
        blk0 = model.model.layers[0]
        op_add = opaque(lambda s, o: (blk0.add(s, o),), lambda s, o: (s.new_empty(s.shape),))
        self.add_fn = lambda blk, s, o: op_add(s, o)[0]


class Decoder:
    """A static-shape decode step: ids [B], pos [B] (int64, device) -> logits [B, V]. KV cache
    [L, B, cap, Hk, D] preallocated. ``ps`` gets probes on rows [:probe_rows] of each value."""

    def __init__(self, model, kit: Kit, B: int, cap: int, ps=None, probe_rows=None):
        c, dev = model.cfg, model.lm_head.weight.device
        self.m, self.kit, self.B, self.ps, self.rows = model, kit, B, ps, probe_rows or B
        self.kc = torch.zeros(c.layers, B, cap, c.kv_heads, c.head_dim, device=dev, dtype=model.ctx.dt)
        self.vc = torch.zeros_like(self.kc)
        self.bidx = torch.arange(B, device=dev)
        for t in (self.kc, self.vc, self.bidx):
            torch._dynamo.mark_static_address(t)

    def reset(self):
        self.kc.zero_()
        self.vc.zero_()

    def P(self, name, x):
        if self.ps is not None:
            self.ps.probe(name, x[: self.rows])

    def __call__(self, ids, pos):
        m, k, c = self.m, self.kit, self.m.cfg
        if self.ps is not None:
            self.ps.step()
        B, scale = ids.shape[0], c.head_dim ** -0.5
        s = m.model.embed_tokens(ids)                                         # [B, H]
        for i, blk in enumerate(m.model.layers):
            at = blk.self_attn
            a = k.norm_fn(blk.input_layernorm, s)
            self.P(f"L{i}.attn_norm", a)
            q = k.linear_fn(at.q_proj, a).view(B, c.heads, c.head_dim)
            kk = k.linear_fn(at.k_proj, a).view(B, c.kv_heads, c.head_dim)
            v = k.linear_fn(at.v_proj, a).view(B, c.kv_heads, c.head_dim)
            q, kk = k.rope_fn(at.rope, q, kk, pos)
            self.kc[i].index_put_((self.bidx, pos), kk)
            self.vc[i].index_put_((self.bidx, pos), v)
            o = k.attn_fn(q, self.kc[i], self.vc[i], pos, scale)
            s = k.add_fn(blk, s, k.linear_fn(at.o_proj, o.reshape(B, -1)))
            self.P(f"L{i}.attn_res", s)
            h = k.norm_fn(blk.post_attention_layernorm, s)
            self.P(f"L{i}.mlp_norm", h)
            mlp = blk.mlp
            u = k.linear_fn(mlp.down_proj, k.silu_fn(mlp.act, k.linear_fn(mlp.gate_proj, h), k.linear_fn(mlp.up_proj, h)))
            s = k.add_fn(blk, s, u)
            self.P(f"L{i}.out", s)
        logits = k.linear_fn(m.lm_head, k.norm_fn(m.model.norm, s))
        self.P("logits", logits)
        return logits


# ----------------------------------------------------------------------------- runners
@torch.no_grad()
def run_mode(dec: Decoder, mode: str, ids, pos, timing=False):
    """Run len(ids) decode steps. mode: eager | graph | compile:<torch.compile mode>.
    Returns (logits per step [T, B, V], ms per step list, info)."""
    torch._dynamo.reset()
    info = {}
    T = ids.shape[0]
    if mode == "eager":
        fn = dec
    elif mode == "graph":
        sid, spos = ids[0].clone(), pos[0].clone()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2):
                dec(sid, spos)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sout = dec(sid, spos)

        def fn(i_, p_):
            sid.copy_(i_)
            spos.copy_(p_)
            g.replay()
            return sout
    else:
        cm = mode.split(":", 1)[1]
        torch._inductor.metrics.reset()
        t0 = time.perf_counter()
        fn = torch.compile(dec, mode=cm, fullgraph=dec.kit.kind != "bi")
        for _ in range(3 if cm == "reduce-overhead" else 1):                 # compile (and record graphs)
            fn(ids[0], pos[0])
        torch.cuda.synchronize()
        info.update(compile_s=round(time.perf_counter() - t0, 1),
                    inductor_kernels=torch._inductor.metrics.generated_kernel_count)
    dec.reset()
    if dec.ps is not None:
        dec.ps.reset()
    outs, ms = [], []
    for t in range(T):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        y = fn(ids[t], pos[t])
        e1.record()
        outs.append(y.clone())
        if timing:
            torch.cuda.synchronize()
            ms.append(e0.elapsed_time(e1))
    torch.cuda.synchronize()
    return torch.stack(outs), ms, info


def bits_equal(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    it = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(it), b.contiguous().view(it))


def per_probe(ref, other):
    """{name: steps whose fingerprints differ} for two records() lists."""
    kb = {(r["step"], r["name"], r["index"], r["call"]): r for r in other}
    out = {}
    for r in ref:
        o = kb.get((r["step"], r["name"], r["index"], r["call"]))
        d = o is None or o["signature"] != r["signature"] or o["vectors"] != r["vectors"]
        out.setdefault(r["name"], []).append(int(d))
    return {n: sum(v) for n, v in out.items()}


class OG:
    """OpGuard comparison of two probe record sets (TraceWriter + export_pair + first_divergence)."""

    def __init__(self, og, work: Path):
        self.og, self.work, self.n = og, work, 0

    def write(self, tag, ps, data):
        root = self.work / "traces" / tag
        if root.exists():
            shutil.rmtree(root)
        with ob.TraceWriter(root) as w:
            ps.emit(w, data)
        return root

    def compare(self, a: Path, b: Path) -> dict:
        self.n += 1
        out = self.work / "exports" / f"{a.name}__vs__{b.name}"
        if out.exists():
            shutil.rmtree(out)
        t0 = time.perf_counter()
        exp = ob.export_pair(a, b, out, self.og)
        facts = ob.first_divergence(exp["reports"][0], self.work, self.og, walk=False)
        pv = facts.get("pivot") or {}
        return {"diverged": facts["diverged"], "fn": pv.get("fn"), "counter": pv.get("counter"),
                "diff_fields": pv.get("diff_fields"), "status_counts": facts["status_counts"],
                "tools": facts["tools"], "regional": [r.get("differing_regions") for r in pv.get("regional") or []],
                "seconds": round(time.perf_counter() - t0, 1)}


def ordinal(records, first):
    """1-based position of a record (the TraceWriter counter) in a records() list."""
    if first is None:
        return None
    for i, r in enumerate(records):
        if (r["step"], r["name"], r["index"], r["call"]) == (first["step"], first["name"], first["index"], first["call"]):
            return i + 1


def opguard_agrees(host_first, og_res, records):
    if og_res is None:
        return None
    if host_first is None:
        return not og_res["diverged"]
    return og_res["diverged"] and og_res["fn"] == f"<{host_first['name']}>" and og_res["counter"] == ordinal(records, host_first)


# ----------------------------------------------------------------------------- parts
def part_probe(res, model, ids, pos):
    """0. probe fingerprints vs opguard_bridge (and OpGuard) on the kept tensors."""
    out = {}
    impls = ["torch"] + (["opguard"] if ob.opguard_available() else [])
    for mode in ("eager", "compile:default"):
        ps = gp.ProbeSet(depth=ids.shape[0] + 1, vectors=True, keep=True)
        dec = Decoder(model, Kit(model, "stock"), ids.shape[1], 64, ps)
        run_mode(dec, mode, ids, pos)
        recs = ps.records()
        assert len(recs) == len(ps.kept), (len(recs), len(ps.kept))
        bad = {i: 0 for i in impls}
        for r, (key, t) in zip(recs, ps.kept):
            assert key == (r["name"], r["index"], r["call"])
            for impl in impls:
                sig = ob._unsigned_list(ob.xor_signature(t, 4, impl=impl))
                vec = ob._unsigned_list(ob.xor_vectors(t, impl=impl))
                bad[impl] += int(sig != r["signature"] or vec != r["vectors"])
        out[mode] = {"records": len(recs), "mismatching": bad}
        res.check(f"probe fingerprints = opguard_bridge.xor_signature/xor_vectors ({mode})",
                  all(v == 0 for v in bad.values()), level="other", impl="graph_probe",
                  detail={"records": len(recs), "mismatching_by_impl": bad})
    # synthetic tensors: odd byte counts, non-contiguous, 0-dim, inside a CUDA graph
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(3, 5, 64, generator=g).bfloat16(), torch.randn(7, 33, generator=g),
          torch.randn(5, 3, generator=g).half(), torch.randn((), generator=g), torch.randn(6, 4, generator=g).t()]
    xs = [x.cuda() for x in xs]
    ps = gp.ProbeSet(depth=4, vectors=True)

    def f():
        ps.step()
        for i, x in enumerate(xs):
            ps.probe(f"t{i}", x * 1.5)
    f()
    torch.cuda.synchronize()
    ps.reset()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        f()
    for _ in range(3):
        gr.replay()
    recs = [r for r in ps.records() if r["step"] == 3]
    bad = sum(int(r["signature"] != ob._unsigned_list(ob.xor_signature(x * 1.5, 4))
                  or r["vectors"] != (None if ob.xor_vectors(x * 1.5) is None else ob._unsigned_list(ob.xor_vectors(x * 1.5))))
              for r, x in zip(recs, xs))
    res.check("probe in a CUDA graph: replay 3 writes step 3; fingerprints = opguard_bridge (5 dtypes/layouts)",
              len(recs) == 5 and bad == 0, level="other", impl="graph_probe",
              detail={"records": len(recs), "mismatching": bad, "bridge_impl": "opguard" if ob.opguard_available() else "torch"})
    return out


def part_modes(res, model, ids, pos, modes, ogc, work):
    """1. stock: eager vs compiled vs CUDA graph, probes at every layer."""
    B, T = ids.shape[1], ids.shape[0]
    base = None
    table = {}
    for mode in modes:
        ps = gp.ProbeSet(depth=T + 1, vectors=True)
        dec = Decoder(model, Kit(model, "stock"), B, 64 if T < 64 else 2 * T, ps)
        logits, _, info = run_mode(dec, mode, ids, pos)
        data = ps.read()
        recs = ps.records(data)
        root = ogc.write(f"stock_{mode.replace(':', '_')}", ps, data) if ogc else None
        if base is None:
            base = (mode, logits, recs, root)
            table[mode] = {"records": len(recs), **info}
            continue
        first = gp.first_difference(base[2], recs)
        og_res = ogc.compare(base[3], root) if ogc else None
        row = {"logits_equal_eager": bits_equal(base[1], logits), "first_diff": first,
               "differing_steps_by_probe": per_probe(base[2], recs), "opguard": og_res,
               "opguard_matches_host": opguard_agrees(first, og_res, base[2]), **info}
        table[mode] = row
        res.check(f"stock {mode} vs eager: logits bitwise", row["logits_equal_eager"], level="other",
                  impl="stock", expect="pass" if mode == "graph" else "info",
                  detail={"first_diff": first})
        if og_res is not None:
            res.check(f"stock {mode} vs eager: OpGuard first_diff = host first difference", bool(row["opguard_matches_host"]),
                      level="other", impl="graph_probe+OpGuard", detail=og_res)
    return table


def part_buckets(res, model, ids, pos, kinds, buckets, ogc):
    """2. one request in row 0 of graphs captured at padded batch sizes."""
    T = ids.shape[0]
    g = torch.Generator().manual_seed(11)
    out = {}
    for kind in kinds:
        ref, rows = None, {}
        for Bk in buckets:
            pad = torch.randint(0, model.cfg.vocab_size, (T, Bk), generator=g).cuda()
            pad[:, 0] = ids[:, 0]
            ppos = pos[:, :1].expand(T, Bk).contiguous()
            ps = gp.ProbeSet(depth=T + 1, vectors=True)
            dec = Decoder(model, Kit(model, kind), Bk, 64 if T < 64 else 2 * T, ps, probe_rows=1)
            logits, _, _ = run_mode(dec, "graph", pad, ppos)
            data = ps.read()
            recs = ps.records(data)
            root = ogc.write(f"bucket_{kind}_{Bk}", ps, data) if ogc else None
            if ref is None:
                ref = (Bk, logits[:, :1], recs, root)
                rows[Bk] = {"reference": True}
                continue
            first = gp.first_difference(ref[2], recs)
            og_res = ogc.compare(ref[3], root) if ogc else None
            rows[Bk] = {"row0_logits_equal_B1": bits_equal(ref[1], logits[:, :1]), "first_diff": first,
                        "differing_steps_by_probe": per_probe(ref[2], recs), "opguard": og_res,
                        "opguard_matches_host": opguard_agrees(first, og_res, ref[2])}
            res.check(f"{kind} graph bucket {Bk} vs 1: row 0 bitwise through every probe", first is None,
                      level="L2", impl=kind, expect="pass" if kind != "stock" else "info", detail={"first_diff": first})
            if og_res is not None:
                res.check(f"{kind} graph bucket {Bk} vs 1: OpGuard first_diff = host first difference",
                          bool(rows[Bk]["opguard_matches_host"]), level="other", impl="graph_probe+OpGuard", detail=og_res)
        out[kind] = rows
    return out


def part_opaque(res, model, ids, pos, kinds, modes, ogc):
    """3. bi op set, kernels or all rounding ops opaque: compiled = eager through the probes."""
    B, T = ids.shape[1], ids.shape[0]
    out = {}
    for kind in kinds:
        base, rows = None, {}
        for mode in ["eager"] + modes:
            ps = gp.ProbeSet(depth=T + 1, vectors=True)
            dec = Decoder(model, Kit(model, kind), B, 64 if T < 64 else 2 * T, ps)
            try:
                logits, _, info = run_mode(dec, mode, ids, pos)
            except Exception as exc:                                          # recorded, not hidden
                rows[mode] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                res.check(f"{kind} {mode} runs", False, level="other", impl=kind, expect="info", detail=rows[mode])
                continue
            data = ps.read()
            recs = ps.records(data)
            root = ogc.write(f"opaque_{kind}_{mode.replace(':', '_')}", ps, data) if ogc else None
            if base is None:
                base = (logits, recs, root)
                rows[mode] = {"records": len(recs)}
                continue
            first = gp.first_difference(base[1], recs)
            og_res = ogc.compare(base[2], root) if ogc else None
            rows[mode] = {"first_diff": first, "differing_steps_by_probe": per_probe(base[1], recs),
                          "logits_equal_eager": bits_equal(base[0], logits), "opguard": og_res,
                          "opguard_matches_host": opguard_agrees(first, og_res, base[1]), **info}
            res.check(f"{kind} {mode} = eager through every probe", first is None, level="other", impl=kind,
                      expect="pass" if kind == "bi_opaque_all" else "info", detail={"first_diff": first, **info})
            if og_res is not None:
                res.check(f"{kind} {mode} vs eager: OpGuard first_diff = host first difference",
                          bool(rows[mode]["opguard_matches_host"]), level="other", impl="graph_probe+OpGuard", detail=og_res)
        out[kind] = rows
    return out


def part_cost(res, model, ids, pos, kinds_modes):
    """4. probes on vs off: logits bitwise, Inductor kernel count, time per step."""
    B, T = ids.shape[1], ids.shape[0]
    out = {}
    for kind, mode in kinds_modes:
        row = {}
        logits = {}
        variants = [("no_probes", None), ("probes", True), ("no_probes_again", None)]
        if mode.startswith("compile"):
            variants.insert(2, ("probes_not_isolated", False))
        for tag, iso in variants:
            ps = gp.ProbeSet(depth=T + 1, vectors=True, isolate=iso) if iso is not None else None
            dec = Decoder(model, Kit(model, kind), B, 64 if T < 64 else 2 * T, ps)
            lg, ms, info = run_mode(dec, mode, ids, pos, timing=True)
            logits[tag] = lg
            row[tag] = {"ms_per_step_median": round(statistics.median(ms[2:]), 4), **info}
        row["logits_equal"] = bits_equal(logits["no_probes"], logits["probes"])
        row["control_no_probes_twice_equal"] = bits_equal(logits["no_probes"], logits["no_probes_again"])
        if "probes_not_isolated" in logits:
            row["not_isolated_logits_equal"] = bits_equal(logits["no_probes"], logits["probes_not_isolated"])
            res.check(f"{kind} {mode}: probes without isolation do not change the logits", row["not_isolated_logits_equal"],
                      level="other", impl="graph_probe(isolate=False)", expect="info",
                      detail={"inductor_kernels": row["probes_not_isolated"].get("inductor_kernels")})
        out[f"{kind} {mode}"] = row
        res.check(f"{kind} {mode}: probes do not change the logits (bitwise, {T} steps)", row["logits_equal"],
                  level="other", impl="graph_probe", detail=row)
        res.measure(f"{kind} {mode} ms/step without probes", row["no_probes"]["ms_per_step_median"], "ms")
        res.measure(f"{kind} {mode} ms/step with probes", row["probes"]["ms_per_step_median"], "ms")
    return out


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="tiny model, eager vs compile vs graph, ~2 min")
    ap.add_argument("--json", help="write results here (bi-results-v1)")
    ap.add_argument("--opguard-root", default=os.environ.get("OPGUARD_ROOT"), help="OpGuard source checkout")
    ap.add_argument("--work", help="directory for traces and exports (default: a temporary one)")
    ap.add_argument("--parts", default="probe,modes,buckets,opaque,cost",
                    help="comma list; full mode is run one part per process (each under ~5 min) and merged")
    ap.add_argument("--merge", nargs="+", help="merge these result files into --json and exit")
    args = ap.parse_args()
    if args.merge:
        res = Results("graph_probe", quick=False)
        for path in args.merge:
            d = json.load(open(path))
            res.checks += d["checks"]
            res.measurements += d["measurements"]
            for k, v in d["extra"].items():
                if isinstance(v, dict) and isinstance(res.extra.get(k), dict):
                    res.extra[k].update(v)
                else:
                    res.extra[k] = v
        res.extra["merged_from"] = [os.path.basename(p) for p in args.merge]
        res.write(args.json)
        return res.exit_code()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU")
        return 2
    torch._dynamo.config.cache_size_limit = 64
    torch._inductor.config.fx_graph_cache = False
    res = Results("graph_probe", quick=args.quick)
    og = None
    if args.opguard_root:
        sys.path.insert(0, str(Path(args.opguard_root).expanduser()))
        try:
            og = ob.locate_opguard(args.opguard_root)
        except ob.OpGuardError as exc:
            print(f"OpGuard not usable, comparisons by host only: {exc}")
    work = Path(args.work).expanduser().resolve() if args.work else Path(tempfile.mkdtemp(prefix="graph-probe-"))
    work.mkdir(parents=True, exist_ok=True)
    ogc = OG(og, work) if og else None
    quick = args.quick
    cfg = wm.Config.tiny() if quick else wm.Config()
    T = 6 if quick else 24
    model = wm.build(cfg, ops="stock", seed=0)
    # The model builds its RoPE tables lazily on first use. Built inside a compiled graph they
    # come from Inductor's cos/sin and stay cached for every later run, eager included, so
    # build them in eager first. Recorded: are the two builds equal?
    eager_tab = model.ctx.tables(model.lm_head.weight.device)
    if "probe" in args.parts.split(","):
        ctx2 = wm._Ctx(cfg, "stock")
        torch._dynamo.reset()
        try:
            comp = torch.compile(lambda d: ctx2.tables(d)["cos"])(model.lm_head.weight.device)
        except Exception as exc:
            comp = None
            print(f"RoPE table compile failed: {exc}"[:300])
        res.check("lazy RoPE cos table built inside torch.compile = built in eager",
                  comp is not None and bits_equal(comp, eager_tab["cos"]),
                  level="other", impl="model.py _Ctx.tables", expect="info",
                  detail={"elements_differing": None if comp is None else
                          int((comp.view(torch.int16) != eager_tab["cos"].view(torch.int16)).sum()),
                          "elements": eager_tab["cos"].numel()})
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, cfg.vocab_size, (T, 1), generator=g).cuda()           # [T, B=1], teacher-forced
    pos = torch.arange(T).view(T, 1).cuda()
    parts = args.parts.split(",")
    res.extra.update(config={"hidden": cfg.hidden, "layers": cfg.layers, "heads": cfg.heads, "kv_heads": cfg.kv_heads,
                             "intermediate": cfg.intermediate, "vocab": cfg.vocab_size, "dtype": cfg.dtype},
                     steps=T, opguard=bool(og), bridge_impl="opguard" if ob.opguard_available() else "torch",
                     probes_per_step=4 * cfg.layers + 1)
    t0 = time.perf_counter()
    if "probe" in parts:
        res.extra["probe"] = part_probe(res, model, ids[:3], pos[:3])
    if "modes" in parts:
        modes = ["eager", "compile:default", "graph"] if quick else \
            ["eager", "compile:default", "compile:max-autotune-no-cudagraphs", "compile:reduce-overhead", "graph"]
        res.extra["modes"] = part_modes(res, model, ids, pos, modes, ogc, work)
    if "buckets" in parts:
        res.extra["buckets"] = part_buckets(res, model, ids, pos, ["stock", "bi_opaque_all"] if quick else
                                            ["stock", "bi", "bi_opaque_all"], [1, 8, 16], ogc)
    if "opaque" in parts:
        res.extra["opaque"] = part_opaque(res, model, ids, pos, ["bi_opaque_all"] if quick else
                                          ["bi", "bi_opaque_kernels", "bi_opaque_all"],
                                          ["compile:default"] if quick else
                                          ["compile:default", "compile:max-autotune-no-cudagraphs", "compile:reduce-overhead"], ogc)
    if "cost" in parts:
        km = [("stock", "compile:default")] if quick else \
            [("stock", "eager"), ("stock", "graph"), ("stock", "compile:default"), ("stock", "compile:max-autotune-no-cudagraphs"),
             ("stock", "compile:reduce-overhead"), ("bi_opaque_all", "compile:default")]
        res.extra["cost"] = part_cost(res, model, ids, pos, km)
    res.extra["runtime_s"] = round(time.perf_counter() - t0, 1)
    if ogc:
        res.extra["opguard_exports"] = ogc.n
    res.write(args.json)
    if not args.work:
        shutil.rmtree(work, ignore_errors=True)
    return res.exit_code()


if __name__ == "__main__":
    sys.exit(main())
