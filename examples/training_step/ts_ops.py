"""Section 1 and 2: every op of an LM training step, forward plus backward, run to run.

`op_cases(quick)` lists (op, case, stock factory, fixed factory). A factory builds the
inputs once and returns run() -> {name: tensor} (forward results and gradients). The worker
runs each case 20 times in one mode and compares every run with the first, bit for bit.

Fixes use ts_kernels.seg_sum (one written-down order for every scatter-style reduction) and
are also compared with a NumPy loop that states the same order on the CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ts_kernels import (bincount_fixed, gather_rows_fixed, index_add_fixed, scatter_add_fixed,
                        seg_sum, seq_cumsum)
from ts_util import BF, DEV, bit_equal, err_line, first_mismatch, repeat_l1, sync, timed_ms

F32 = torch.float32


def gen_for(seed: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed)


def randn(g, *shape, dtype=F32, scale=1.0):
    return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)


def zipf(g, n: int, V: int, power: float = 4.0) -> torch.Tensor:
    """Token ids with a heavy head, so that many indices repeat."""
    return (torch.rand(n, generator=g, device=DEV) ** power * V).long().clamp_(max=V - 1)


def randint(g, shape, hi: int) -> torch.Tensor:
    return torch.randint(0, hi, shape, generator=g, device=DEV)


class Leaves:
    """Leaf tensors that are differentiated again on every run."""

    def __init__(self, **tensors):
        self.t = {k: v.detach().clone().requires_grad_(True) for k, v in tensors.items()}

    def __getitem__(self, k):
        return self.t[k]

    def reset(self):
        for v in self.t.values():
            v.grad = None

    def grads(self) -> dict:
        return {"d" + k: v.grad for k, v in self.t.items()}


def name_dtype(dt) -> str:
    return {F32: "fp32", BF: "bf16", torch.float16: "fp16", torch.int64: "int64"}[dt]


# ----------------------------------------------------------------------------- factories
def f_embedding(N, V, H, dtype, seed, distinct=None, fixed=False):
    def make():
        g = gen_for(seed)
        idx = zipf(g, N, V) if distinct is None else randint(g, (N,), distinct)
        G = randn(g, N, H, dtype=dtype)
        L = Leaves(W=torch.zeros(V, H, device=DEV, dtype=dtype))

        def run():
            L.reset()
            out = gather_rows_fixed(L["W"], idx) if fixed else F.embedding(idx, L["W"])
            out.backward(G)
            return L.grads()
        return run
    return make


def f_embedding_tied(N, V, H, seed, fixed=False):
    """One weight used as the token embedding and as the LM head, bf16 autocast, fp32 weight.
    Two gradients meet in W.grad: the embedding backward and the head's dW."""
    def make():
        g = gen_for(seed)
        idx, y = zipf(g, N, V), zipf(g, N, V)
        L = Leaves(W=randn(g, V, H, scale=0.05))

        def run():
            L.reset()
            with torch.autocast("cuda", dtype=BF):
                e = gather_rows_fixed(L["W"], idx) if fixed else F.embedding(idx, L["W"])
                logits = F.linear(F.silu(e), L["W"])
                loss = F.cross_entropy(logits.float(), y)
            loss.backward()
            return {"loss": loss.detach(), **L.grads()}
        return run
    return make


def f_rows_bwd(kind, N, V, H, dtype, seed):
    """Backward of a row gather written three ways: index_select, W[idx], fixed."""
    def make():
        g = gen_for(seed)
        idx = zipf(g, N, V)
        G = randn(g, N, H, dtype=dtype)
        L = Leaves(W=torch.zeros(V, H, device=DEV, dtype=dtype))

        def run():
            L.reset()
            W = L["W"]
            out = {"index_select": lambda: W.index_select(0, idx), "getitem": lambda: W[idx],
                   "fixed": lambda: gather_rows_fixed(W, idx)}[kind]()
            out.backward(G)
            return L.grads()
        return run
    return make


def _acc_inputs(R, N, H, dtype, seed, per_row=None):
    g = gen_for(seed)
    if per_row:                                     # exactly per_row contributions per row
        idx = torch.arange(R, device=DEV).repeat(per_row)[torch.randperm(R * per_row, generator=g, device=DEV)]
        N = R * per_row
    else:
        idx = randint(g, (N,), R)
    shape = (N,) if H == 0 else (N, H)
    src = randn(g, *shape, dtype=dtype)
    base = torch.zeros((R,) if H == 0 else (R, H), device=DEV, dtype=dtype)
    return idx, src, base


def f_accumulate(kind, R, N, H, dtype, seed, per_row=None):
    """out = base + rows of src at idx. H == 0 means 1-D tensors."""
    def make():
        idx, src, base = _acc_inputs(R, N, H, dtype, seed, per_row)
        idx2 = idx if H == 0 else idx[:, None].expand(-1, H)

        def run():
            if kind == "index_add":
                out = base.index_add(0, idx, src)
            elif kind == "scatter_add":
                out = base.scatter_add(0, idx2, src)
            elif kind == "index_put":
                out = base.index_put((idx,), src, accumulate=True)
            elif kind == "fixed":
                out = (scatter_add_fixed(base, idx, src) if H == 0 else index_add_fixed(base, idx, src))
            else:
                raise KeyError(kind)
            return {"out": out}
        return run
    return make


def f_gather_bwd(kind, seed, fixed=False):
    def make():
        g = gen_for(seed)
        if kind == "lm":                            # one pick per row: no repeated destination
            N, V = 4096, 32000
            L = Leaves(x=randn(g, N, V))
            idx = zipf(g, N, V)[:, None]
            G = randn(g, N, 1)
            dim = 1
        elif kind == "dup1d":                       # many picks of each table entry
            R, N = 64, 32768
            L = Leaves(x=randn(g, R))
            idx = randint(g, (N,), R)
            G = randn(g, N)
            dim = 0
        else:                                       # dup2d: repeated picks inside a row
            T, E, K = 4096, 64, 8
            L = Leaves(x=randn(g, T, E))
            idx = randint(g, (T, K), 4)
            G = randn(g, T, K)
            dim = 1

        def run():
            L.reset()
            x = L["x"]
            if not fixed:
                x.gather(dim, idx).backward(G)
                return L.grads()
            # fixed: the backward as one segmented sum over linear destinations
            if dim == 0:
                dx = seg_sum(G.reshape(-1, 1), idx, x.shape[0]).reshape(-1)
            else:
                lin = torch.arange(x.shape[0], device=DEV)[:, None] * x.shape[1] + idx
                dx = seg_sum(G.reshape(-1, 1), lin.reshape(-1), x.numel()).reshape(x.shape)
            return {"dx": dx}
        return run
    return make


def f_cross_entropy(form, N, V, seed):
    def make():
        g = gen_for(seed)
        L = Leaves(logits=randn(g, N, V, scale=2.0))
        y = zipf(g, N, V)
        y[torch.rand(N, generator=g, device=DEV) < 0.1] = -100     # padding

        def run():
            L.reset()
            x = L["logits"]
            loss = F.cross_entropy(x, y) if form == "cross_entropy" else F.nll_loss(F.log_softmax(x, -1), y)
            loss.backward()
            return {"loss": loss.detach(), **L.grads()}
        return run
    return make


def rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def f_norm(kind, N, H, dtype, seed):
    def make():
        g = gen_for(seed)
        L = Leaves(x=randn(g, N, H, dtype=dtype), w=1 + randn(g, H, dtype=dtype, scale=0.1),
                   b=randn(g, H, dtype=dtype, scale=0.1))
        G = randn(g, N, H, dtype=dtype)

        def run():
            L.reset()
            y = rmsnorm(L["x"], L["w"]) if kind == "rmsnorm" else F.layer_norm(L["x"], (H,), L["w"], L["b"])
            y.backward(G)
            out = {"y": y.detach(), **L.grads()}
            if kind == "rmsnorm":
                out.pop("db")
            return out
        return run
    return make


def f_linear(M, K, N, amp, seed):
    def make():
        g = gen_for(seed)
        L = Leaves(x=randn(g, M, K), W=randn(g, N, K, scale=K ** -0.5), b=randn(g, N, scale=0.1))
        G = randn(g, M, N, dtype=BF if amp else F32)

        def run():
            L.reset()
            with torch.autocast("cuda", dtype=BF, enabled=amp):
                y = F.linear(L["x"], L["W"], L["b"])
            y.backward(G)
            return {"y": y.detach(), **L.grads()}
        return run
    return make


def f_cumsum(shape, dtype, seed, fixed=False):
    def make():
        g = gen_for(seed)
        if dtype == torch.int64:
            x = randint(g, shape, 3)
            return lambda: {"y": torch.cumsum(x, -1)}
        L = Leaves(x=randn(g, *shape, dtype=dtype))
        G = randn(g, *shape, dtype=dtype)

        def run():
            L.reset()
            if fixed:                               # forward only; fp32 out
                return {"y": seq_cumsum(L["x"].detach())}
            y = torch.cumsum(L["x"], -1)
            y.backward(G)
            return {"y": y.detach(), **L.grads()}
        return run
    return make


def f_topk(ties, seed):
    def make():
        g = gen_for(seed)
        s = randn(g, 4096, 64)
        if ties:
            s = (s * 2).round() / 2                 # many equal scores
        L = Leaves(s=s)
        G = randn(g, 4096, 8)

        def run():
            L.reset()
            v, i = L["s"].topk(8, -1)
            v.backward(G)
            return {"values": v.detach(), "indices": i, **L.grads()}
        return run
    return make


def f_counts(kind, seed, fixed=False):
    def make():
        g = gen_for(seed)
        E, N = 64, 32768
        idx = randint(g, (N,), E)
        w = torch.rand(N, generator=g, device=DEV)

        def run():
            if kind == "bincount":
                return {"y": torch.bincount(idx, minlength=E)}
            if kind == "bincount_weights":
                return {"y": bincount_fixed(idx, w, E) if fixed else torch.bincount(idx, weights=w, minlength=E)}
            return {"y": torch.histc(idx.float(), bins=E, min=0, max=E)}
        return run
    return make


def moe_forward(x, P, gather, combine, K):
    """Softmax router, top-K, renormalized weights, one small MLP per expert."""
    E = P["Wr"].shape[0]
    probs = F.linear(x, P["Wr"]).float().softmax(-1)
    w, ids = probs.topk(K, -1)
    w = w / w.sum(-1, keepdim=True)
    flat = ids.reshape(-1)
    order = flat.argsort(stable=True)               # dispatch list: by expert, then by token
    tok = order // K
    counts = torch.bincount(flat, minlength=E).tolist()
    xs = gather(x, tok)
    ys, s = [], 0
    for e, c in enumerate(counts):
        if c:
            ys.append(F.linear(F.silu(F.linear(xs[s:s + c], P["W1"][e])), P["W2"][e]))
            s += c
    y = torch.cat(ys) * w.reshape(-1)[order].unsqueeze(1).to(x.dtype)
    return combine(torch.zeros_like(x), tok, y)


MOE_GATHER = {"getitem": lambda x, tok: x[tok], "index_select": lambda x, tok: x.index_select(0, tok),
              "fixed": gather_rows_fixed}
MOE_COMBINE = {"index_add": lambda base, tok, y: base.index_add(0, tok, y),
               "scatter_add": lambda base, tok, y: base.scatter_add(0, tok[:, None].expand_as(y), y),
               "fixed": index_add_fixed}


def f_moe(gather, combine, seed, T=2048, H=256, I=512, E=16, K=4, dtype=BF):
    def make():
        g = gen_for(seed)
        L = Leaves(x=randn(g, T, H, dtype=dtype), Wr=randn(g, E, H, dtype=dtype, scale=H ** -0.5),
                   W1=randn(g, E, I, H, dtype=dtype, scale=H ** -0.5),
                   W2=randn(g, E, H, I, dtype=dtype, scale=I ** -0.5))
        G = randn(g, T, H, dtype=dtype)

        def run():
            L.reset()
            out = moe_forward(L["x"], L.t, MOE_GATHER[gather], MOE_COMBINE[combine], K)
            out.backward(G)
            return {"out": out.detach(), **L.grads()}
        return run
    return make


def f_repeat_kv(kind, seed, B=4, Hk=2, T=512, D=128, group=4):
    def make():
        g = gen_for(seed)
        L = Leaves(k=randn(g, B, Hk, T, D, dtype=BF))
        G = randn(g, B, Hk * group, T, D, dtype=BF)

        def run():
            L.reset()
            k = L["k"]
            out = (k[:, :, None].expand(B, Hk, group, T, D).reshape(B, Hk * group, T, D) if kind == "expand"
                   else k.repeat_interleave(group, dim=1))
            out.backward(G)
            return L.grads()
        return run
    return make


def f_sum(kind, seed):
    def make():
        g = gen_for(seed)
        L = Leaves(x=randn(g, 4096, 1024))

        def run():
            L.reset()
            y = L["x"].sum() if kind == "sum" else L["x"].mean()
            y.backward()
            return {"y": y.detach(), **L.grads()}
        return run
    return make


# ----------------------------------------------------------------------------- the list
def op_cases(quick: bool):
    """[(op, case, stock factory or None, fixed factory or None)]. `op` is the row of the case
    page's table; `case` is the size and dtype."""
    V, H = 32000, 256
    out = []

    def add(op, case, stock, fixed=None):
        out.append((op, case, stock, fixed))

    emb_n = (3072, 3073, 16384) if quick else (512, 3071, 3072, 3073, 16384, 65536)
    for n in emb_n:
        add("embedding backward", f"N={n} fp32", f_embedding(n, V, H, F32, 10), f_embedding(n, V, H, F32, 10, fixed=True))
    for n in ((3072, 3073) if quick else (3072, 3073, 65536)):
        add("embedding backward", f"N={n} bf16", f_embedding(n, V, H, BF, 11), f_embedding(n, V, H, BF, 11, fixed=True))
        add("embedding backward", f"N={n} fp32 64 distinct rows", f_embedding(n, V, H, F32, 12, distinct=64),
            f_embedding(n, V, H, F32, 12, distinct=64, fixed=True))
    add("tied embedding and head (bf16 autocast)", "N=4096 V=8192", f_embedding_tied(4096, 8192, H, 13),
        f_embedding_tied(4096, 8192, H, 13, fixed=True))

    for n in ((4096,) if quick else (4096, 65536)):
        for dt in (F32, BF):
            fx = f_rows_bwd("fixed", n, V, H, dt, 14)
            add("index_select backward", f"N={n} {name_dtype(dt)}", f_rows_bwd("index_select", n, V, H, dt, 14), fx)
            add("W[idx] backward", f"N={n} {name_dtype(dt)}", f_rows_bwd("getitem", n, V, H, dt, 14), fx)

    for kind in ("index_add", "scatter_add", "index_put"):
        label = {"index_add": "index_add_", "scatter_add": "scatter_add_", "index_put": "index_put_(accumulate=True)"}[kind]
        for dt in (F32, BF):
            add(label, f"R=1024 N=16384 H=256 {name_dtype(dt)}", f_accumulate(kind, 1024, 16384, 256, dt, 15),
                f_accumulate("fixed", 1024, 16384, 256, dt, 15))
        add(label, "R=4096 3 per row H=256 bf16", f_accumulate(kind, 4096, 0, 256, BF, 16, per_row=3),
            f_accumulate("fixed", 4096, 0, 256, BF, 16, per_row=3))
        add(label, "1-D R=64 N=32768 fp32", f_accumulate(kind, 64, 32768, 0, F32, 17),
            f_accumulate("fixed", 64, 32768, 0, F32, 17))

    add("gather backward", "one pick per row (LM loss) N=4096 V=32000", f_gather_bwd("lm", 18))
    add("gather backward", "repeated picks 1-D R=64 N=32768", f_gather_bwd("dup1d", 19), f_gather_bwd("dup1d", 19, fixed=True))
    add("gather backward", "repeated picks in a row T=4096 K=8", f_gather_bwd("dup2d", 20), f_gather_bwd("dup2d", 20, fixed=True))

    nce = 1024 if quick else 4096
    add("cross_entropy backward", f"N={nce} V=32000 fp32", f_cross_entropy("cross_entropy", nce, 32000, 21))
    add("log_softmax + nll_loss backward", f"N={nce} V=32000 fp32", f_cross_entropy("lsm_nll", nce, 32000, 21))

    for dt in (BF, F32):
        add("RMSNorm backward", f"N=4096 H=1024 {name_dtype(dt)}", f_norm("rmsnorm", 4096, 1024, dt, 22))
        add("LayerNorm backward", f"N=4096 H=1024 {name_dtype(dt)}", f_norm("layernorm", 4096, 1024, dt, 23))
    add("linear backward (dW, db, dX)", "4096x1024 -> 2048 bf16 autocast", f_linear(4096, 1024, 2048, True, 24))
    add("linear backward (dW, db, dX)", "4096x1024 -> 2048 fp32", f_linear(4096, 1024, 2048, False, 24))

    add("cumsum", "[64, 4096] fp32 forward+backward", f_cumsum((64, 4096), F32, 25), f_cumsum((64, 4096), F32, 25, fixed=True))
    add("cumsum", "[64, 4096] bf16 forward+backward", f_cumsum((64, 4096), BF, 25))
    add("cumsum", "[1048576] fp32 forward+backward", f_cumsum((1 << 20,), F32, 26))
    add("cumsum", "[1048576] int64 forward", f_cumsum((1 << 20,), torch.int64, 27))

    add("topk backward", "[4096, 64] k=8 fp32", f_topk(False, 28))
    add("topk backward", "[4096, 64] k=8 fp32, tied scores", f_topk(True, 28))

    add("bincount", "N=32768 E=64, no weights", f_counts("bincount", 29))
    add("bincount", "N=32768 E=64, fp32 weights", f_counts("bincount_weights", 29), f_counts("bincount_weights", 29, fixed=True))
    add("histc", "N=32768 bins=64", f_counts("histc", 29))

    add("MoE dispatch -> experts -> combine, backward", "x[tok] + index_add_, top-4 bf16",
        f_moe("getitem", "index_add", 30), f_moe("fixed", "fixed", 30))
    add("MoE dispatch -> experts -> combine, backward", "index_select + scatter_add_, top-4 bf16",
        f_moe("index_select", "scatter_add", 30), f_moe("fixed", "fixed", 30))

    add("GQA repeat_kv backward", "expand+reshape, group 4 bf16", f_repeat_kv("expand", 31))
    add("GQA repeat_kv backward", "repeat_interleave, group 4 bf16", f_repeat_kv("interleave", 31))
    add("sum / mean (full reduction)", "[4096, 1024] fp32 sum", f_sum("sum", 32))
    add("sum / mean (full reduction)", "[4096, 1024] fp32 mean", f_sum("mean", 32))
    return out


def run_ops(quick: bool, runs: int, with_fixed: bool) -> list:
    """One record per (op, case): L1 over `runs` runs and the median time, for the stock op in
    the current mode, and for the fixed version when with_fixed is set."""
    recs = []
    for op, case, stock, fixed in op_cases(quick):
        rec = {"op": op, "case": case}
        for label, factory in (("stock", stock), ("fixed", fixed if with_fixed else None)):
            if factory is None:
                continue
            try:
                run = factory()
                r = repeat_l1(run, runs)
                r["ms"] = round(timed_ms(run), 4)
                rec[label] = r
            except RuntimeError as e:
                sync()
                rec[label] = {"raised": err_line(e)}
            torch.cuda.empty_cache()
        recs.append(rec)
        s = rec.get("stock", {})
        print(f"  {op:44s} {case:44s} "
              + ("RAISES " + s["raised"][:60] if "raised" in s else f"{'repeat' if s['passed'] else 'DIFFERS'} {s['ms']:.3f} ms")
              + (f" | fixed {'repeat' if rec['fixed'].get('passed') else rec['fixed']} {rec['fixed'].get('ms', 0):.3f} ms"
                 if "fixed" in rec else ""), flush=True)
    return recs


# ----------------------------------------------------------------------------- references
def np_seg_sum(src: torch.Tensor, index: torch.Tensor, num_dst: int, base: torch.Tensor = None) -> torch.Tensor:
    """The fixed order, stated as a plain loop on the CPU (NumPy, fp32):
    for i in ascending order: out[index[i]] += src[i]."""
    s = src.detach().float().cpu().numpy()
    if s.ndim == 1:
        s = s[:, None]
    out = (np.zeros((num_dst, s.shape[1]), np.float32) if base is None
           else base.detach().float().cpu().numpy().reshape(num_dst, -1).copy())
    ix = index.cpu().numpy()
    for i in range(s.shape[0]):
        out[ix[i]] += s[i]
    return torch.from_numpy(out)


def np_embedding_small_path(src, index, num_dst, dtype) -> torch.Tensor:
    """Model of torch's CUDA embedding backward for <= 3072 indices: source rows in chunks of 32;
    inside a chunk the rows of one destination are summed in order in fp32; each chunk's sum is
    rounded to the weight dtype and added to the destination in the weight dtype."""
    s = src.detach().float().cpu().numpy()
    ix = index.cpu().numpy()
    out = torch.zeros(num_dst, s.shape[1], dtype=dtype)
    for c0 in range(0, s.shape[0], 32):
        part = {}
        for i in range(c0, min(c0 + 32, s.shape[0])):
            d = int(ix[i])
            part[d] = s[i].copy() if d not in part else part[d] + s[i]
        for d, v in part.items():
            out[d] = (out[d].float() + torch.from_numpy(v).to(dtype).float()).to(dtype)
    return out


def np_embedding_large_path(src, index, num_dst, dtype) -> torch.Tensor:
    """Model of torch's CUDA embedding backward for > 3072 indices: stable sort by destination;
    each destination's rows in pieces of 10, each piece summed in order in fp32; the piece sums
    added in order in fp32; one rounding to the weight dtype."""
    s = src.detach().float().cpu().numpy()
    ix = index.cpu().numpy()
    order = np.argsort(ix, kind="stable")
    out = np.zeros((num_dst, s.shape[1]), np.float32)
    i = 0
    while i < len(order):
        d = ix[order[i]]
        j = i
        while j < len(order) and ix[order[j]] == d:
            j += 1
        total = np.zeros(s.shape[1], np.float32)
        for p in range(i, j, 10):
            piece = np.zeros(s.shape[1], np.float32)
            for q in range(p, min(p + 10, j)):
                piece = piece + s[order[q]]
            total = total + piece
        out[d] = total
        i = j
    return torch.from_numpy(out).to(dtype)


def seq_in_dtype(src, index, num_dst, base) -> torch.Tensor:
    """Sequential adds in ascending source position, rounding to the tensor's dtype after every
    add (what a load-add-store loop on the destination does)."""
    out = base.detach().cpu().clone().reshape(num_dst, -1)
    s = src.detach().cpu().reshape(src.shape[0], -1)
    ix = index.cpu().tolist()
    for i, d in enumerate(ix):
        out[d] = (out[d].float() + s[i].float()).to(out.dtype)
    return out


def sum_then_base(src, index, num_dst, base) -> torch.Tensor:
    """Each destination's rows summed in order in fp32 from zero, then added to the base once."""
    s = np_seg_sum(src, index, num_dst)
    return (base.detach().float().cpu().reshape(num_dst, -1) + s).to(base.dtype)


def lanes32(src, index, num_dst, base) -> torch.Tensor:
    """Model of torch's sort-based accumulate for 1-element slices: a destination with n rows
    takes floor(n/32) passes of 32 lanes (lane l sums rows l, l+32, ...), folds the lanes with a
    shuffle-down tree (16, 8, 4, 2, 1), adds the n % 32 remaining rows in order, then adds the
    base once. All in fp32."""
    s = src.detach().float().cpu().numpy().reshape(-1)
    ix = index.cpu().numpy()
    order = np.argsort(ix, kind="stable")
    out = base.detach().float().cpu().numpy().reshape(-1).copy()
    i = 0
    while i < len(order):
        d = ix[order[i]]
        j = i
        while j < len(order) and ix[order[j]] == d:
            j += 1
        n = j - i
        passes = n // 32
        lane = np.zeros(32, np.float32)
        for p in range(passes):
            lane = lane + s[order[i + p * 32:i + p * 32 + 32]]
        off = 16
        while off > 0:
            lane[:32 - off] = lane[:32 - off] + lane[off:]
            off //= 2
        acc = lane[0]
        for q in range(i + passes * 32, j):
            acc = np.float32(acc + s[order[q]])
        out[d] = np.float32(out[d] + acc)
        i = j
    return torch.from_numpy(out).to(base.dtype)


def fixes_checks(quick: bool) -> list:
    """Section 2: the fixed-order kernel against its written definition, across layouts."""
    recs = []

    def rec(name, passed, **detail):
        recs.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'ok' if passed else 'FAIL'}] {name} {detail if not passed else ''}", flush=True)

    # 1. seg_sum == the NumPy loop, for the slice widths around torch's own thresholds (1, 32)
    #    and index counts around the embedding threshold (3072)
    g = gen_for(40)
    for H in (1, 2, 31, 32, 33, 64, 256):
        for N, R in ((3071, 97), (3072, 97), (3073, 97)) + (() if quick else ((20000, 1000),)):
            for dt in (F32, BF):
                idx = randint(g, (N,), R)
                src = randn(g, N, H, dtype=dt)
                got = seg_sum(src, idx, R)
                mm = first_mismatch(np_seg_sum(src, idx, R), got.cpu())
                if mm:
                    rec(f"seg_sum equals the NumPy loop H={H} N={N} {name_dtype(dt)}", False, **mm)
    rec("seg_sum equals the NumPy loop: H in 1,2,31,32,33,64,256; N in 3071,3072,3073"
        + ("" if quick else ",20000") + "; fp32 and bf16 sources",
        not any(not r["passed"] for r in recs))

    # 2. with a base (index_add_ into a non-zero tensor), rounded once to the base dtype
    ok = True
    for dt in (F32, BF):
        idx = randint(g, (5000,), 300)
        src, base = randn(g, 5000, 48, dtype=dt), randn(g, 300, 48, dtype=dt)
        ref = np_seg_sum(src, idx, 300, base=base).to(dt)
        ok &= bit_equal(ref, index_add_fixed(base, idx, src).cpu())
    rec("index_add_fixed with a non-zero base equals the NumPy loop (fp32, bf16)", ok)

    # 3. layout: column block size, strided source rows, int32 index, extra destinations
    idx = zipf(g, 9000, 500)
    big = randn(g, 18000, 200, dtype=BF)
    src = big[::2]                                   # rows with stride 2
    ref = seg_sum(src.contiguous(), idx, 500)
    ok = all(bit_equal(ref, seg_sum(src, idx, 500, BH=bh)) for bh in (16, 64, 128))
    rec("seg_sum is the same for column blocks 16/64/128 and strided source rows", ok)
    rec("seg_sum is the same for an int32 index", bit_equal(ref, seg_sum(src, idx.int(), 500)))
    rec("seg_sum is the same with more destination rows", bit_equal(ref, seg_sum(src, idx, 4000)[:500]))

    # 4. a destination's result depends only on its own rows, in order (L2 for the backward):
    #    the probe token's gradient row alone vs inside batches of other tokens
    V, H = 32000, 256
    n_probe, probe = 100, 7
    Gp = randn(g, n_probe, H)
    alone = seg_sum(Gp, torch.full((n_probe,), probe, device=DEV), V)[probe]
    stock_alone = _stock_emb_row(Gp, torch.full((n_probe,), probe, device=DEV), V, probe)
    sizes = (512, 3072, 3073, 16384) if quick else (128, 512, 2048, 3071, 3072, 3073, 4096, 16384, 65536)
    fixed_ok, stock_fail, stock_same_large = True, [], []
    large_ref = None
    for N in sizes:
        for layout in ("front", "spread", "back"):
            pos = {"front": torch.arange(n_probe, device=DEV),
                   "spread": torch.linspace(0, N - 1, n_probe, device=DEV).long(),
                   "back": torch.arange(N - n_probe, N, device=DEV)}[layout]
            idx = 8 + zipf(g, N, V - 8)              # other tokens, never the probe
            idx[pos] = probe
            Gall = randn(g, N, H)
            Gall[pos] = Gp
            fixed_ok &= bit_equal(alone, seg_sum(Gall, idx, V)[probe])
            row = _stock_emb_row(Gall, idx, V, probe)
            if not bit_equal(stock_alone, row):
                stock_fail.append(f"{N}/{layout}")
            if N > 3072:
                large_ref = row if large_ref is None else large_ref
                stock_same_large.append(bit_equal(large_ref, row))
    rec("fixed embedding backward: a token's gradient row is the same alone and inside batches of "
        f"{sizes[0]}..{sizes[-1]} tokens, at the front, spread, or at the back", fixed_ok)
    recs.append({"name": "stock embedding backward: a token's gradient row alone vs inside batches",
                 "passed": not stock_fail, "expect": "fail",
                 "detail": {"sizes": list(sizes), "failing": stock_fail,
                            "all_batches_over_3072_agree_with_each_other": all(stock_same_large)}})
    print(f"  stock embedding row vs alone: failing {stock_fail}; >3072 agree with each other: {all(stock_same_large)}")
    return recs


def _stock_emb_row(G, idx, V, row):
    W = torch.zeros(V, G.shape[1], device=DEV, requires_grad=True)
    F.embedding(idx, W).backward(G)
    return W.grad[row].clone()


def stock_order_models(quick: bool) -> list:
    """Which written order reproduces the stock op's bits (run in default and strict mode)."""
    recs = []
    g = gen_for(50)
    V, H = 2000, 48

    def rec(name, **detail):
        recs.append({"name": name, "detail": detail})
        print(f"  {name}: {detail}", flush=True)

    # embedding backward, both sides of the 3072 threshold
    for N in (3072, 3073):
        for dt in (F32, BF):
            idx = zipf(g, N, V)
            G = randn(g, N, H, dtype=dt)
            W = torch.zeros(V, H, device=DEV, dtype=dt, requires_grad=True)
            F.embedding(idx, W).backward(G)
            got = W.grad.cpu()
            rec(f"embedding backward N={N} {name_dtype(dt)}",
                chunks_of_32_source_rows=bit_equal(np_embedding_small_path(G, idx, V, dt), got),
                pieces_of_10_per_destination=bit_equal(np_embedding_large_path(G, idx, V, dt), got),
                sequential_fp32=bit_equal(np_seg_sum(G, idx, V).to(dt), got))

    # accumulate ops: slice widths around 1 and 32, zero and non-zero base
    for H in (1, 2, 32, 33):
        for dt in (F32, BF):
            for per_row in (3, 31, 32, 33, 100):
                R = 64
                idx = torch.arange(R, device=DEV).repeat(per_row)[torch.randperm(R * per_row, generator=g, device=DEV)]
                N = idx.numel()
                src = randn(g, N, H, dtype=dt)
                base = randn(g, R, H, dtype=dt)
                refs = {"sequential_fp32_round_once": np_seg_sum(src, idx, R, base=base).to(dt),
                        "sum_from_zero_then_base": sum_then_base(src, idx, R, base),
                        "sequential_in_tensor_dtype": seq_in_dtype(src, idx, R, base)}
                if H == 1:
                    refs["lanes_of_32_then_tree"] = lanes32(src, idx, R, base).reshape(R, 1)
                for opname, fn in (("index_add_", lambda: base.index_add(0, idx, src)),
                                   ("index_put_(accumulate)", lambda: base.index_put((idx,), src, accumulate=True)),
                                   ("scatter_add_", lambda: base.scatter_add(0, idx[:, None].expand(-1, H), src))):
                    try:
                        outs = [fn().cpu() for _ in range(3)]
                    except RuntimeError as e:
                        rec(f"{opname} H={H} {name_dtype(dt)} {per_row} per row", raised=err_line(e))
                        continue
                    stable = all(bit_equal(outs[0], o) for o in outs[1:])
                    rec(f"{opname} H={H} {name_dtype(dt)} {per_row} per row", repeats=stable,
                        **{k: bit_equal(v, outs[0]) for k, v in refs.items()})
    return recs
