"""MoE layer under expert parallelism (EP), checked bitwise at L1, L2 and L5.

Layer: bf16, hidden 2048, SwiGLU experts with FFN 1408, E experts, top-k, a
softmax router with renormalized fp32 weights. Router and expert GEMMs run
through tools/bi_kernels.mm_one_config (one tile configuration for every M).
Top-k is a stable sort: score descending, then expert id ascending.

EP: W ranks, E/W experts per rank, tokens sharded across ranks. Dispatch sends
one row per (token, expert) with all_to_all_single, after an all_to_all of
per-expert counts. Expert outputs go back the same way.

Combine variants (all share routing, dispatch and expert GEMMs):
  arrival_atomic    index_add_ of the returned rows into a bf16 output (atomics),
                    as common EP code does after the return all_to_all
  arrival_det       the same under torch.use_deterministic_algorithms(True)
  fixed_slot        the recipe: each returned row has a fixed slot (token,
                    top-k rank); one Triton program per token sums its slots in
                    rank order in fp32 and rounds to bf16 once
  fixed_slot_torch  the same sum written with torch ops, as in the single-GPU
                    case (examples/moe_combine); a cross-check of the kernel
  rank_partial      the expert rank pre-reduces each token's local experts in
                    fp32 (top-k rank order) and returns one fp32 row per
                    (token, rank); the source sums those in rank order.
                    Deterministic, but the grouping depends on the EP degree
  capacity          fixed_slot plus a per-expert capacity ceil(cf*T_local*K/E)
                    decided on the source rank; assignments past it (first
                    come, first served in local token order) are dropped

Run (one launch per EP degree W, then a comparison):
  torchrun --nproc_per_node=W moe_ep.py --E 32 --K 4 --dump DIR
  python moe_ep.py --compare DIR --json results-a16.json
--quick (the regression suite, see run_all.sh): fewer co-batched sizes in L2 and fewer batch
sizes in the GEMM checks. The layer, the global batch and the comparisons are the same.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
from bi_check import DEFAULT_SIZES, bits, check_batch, first_mismatch  # noqa: E402
from bi_kernels import mm_one_config  # noqa: E402

H, I = 2048, 1408
BF = torch.bfloat16
VARIANTS = ("arrival_atomic", "arrival_det", "fixed_slot", "fixed_slot_torch", "rank_partial", "capacity")
CAP_FACTOR = 1.0
BIG = 1 << 32
L2_SIZES = (1, 2, 3, 8, 31, 64, 129, 512)
QUICK_L2_SIZES = (1, 3, 31, 129)
QUICK_GEMM_SIZES = (1, 2, 3, 8, 31, 64, 129)


# ----------------------------------------------------------------------------- parameters
def _randn(shape, seed, scale):
    g = torch.Generator().manual_seed(seed)                       # CPU: same values on every rank
    return (torch.randn(*shape, generator=g) * scale).to(BF)


def make_params(E, rank, W, dev):
    El = E // W
    lo = rank * El
    P = {"Wr": _randn((H, E), 1, H ** -0.5).to(dev)}
    for name, seed0, shp, sc in (("W1", 1000, (H, I), H ** -0.5), ("W3", 2000, (H, I), H ** -0.5),
                                 ("W2", 3000, (I, H), I ** -0.5)):
        P[name] = torch.stack([_randn(shp, seed0 + e, sc) for e in range(lo, lo + El)]).to(dev)
    return P


# ----------------------------------------------------------------------------- routing
def route(x, Wr, K):
    """Softmax router, stable top-k (score desc, expert id asc), fp32 weights renormalized
    with a sum in fixed top-k rank order."""
    probs = mm_one_config(x, Wr).float().softmax(-1)
    vals, ids = torch.sort(probs, dim=-1, descending=True, stable=True)
    w, ids = vals[:, :K], ids[:, :K].contiguous()
    s = w[:, 0]
    for j in range(1, K):
        s = s + w[:, j]
    return (w / s[:, None]).contiguous(), ids


def capacity_keep(ids, E, cf=CAP_FACTOR):
    """Per-expert capacity from the local token count; first come, first served in
    (token, top-k rank) order. Returns a [T, K] keep mask."""
    T, K = ids.shape
    if T == 0:
        return torch.ones_like(ids, dtype=torch.bool)
    C = math.ceil(cf * T * K / E)
    oh = F.one_hot(ids.reshape(-1), E)
    seen = (oh.cumsum(0) * oh).sum(1) - 1                          # earlier assignments to this expert
    return (seen < C).view(T, K)


# ----------------------------------------------------------------------------- fixed-slot sum
@triton.jit(do_not_specialize=["T"])
def _slot_sum_kernel(Y, Wt, ROW, OUT, T, H: tl.constexpr, S: tl.constexpr, BH: tl.constexpr):
    """OUT[t] = sum over s = 0..S-1 (fixed order, fp32) of Wt[t, s] * Y[ROW[t, s]], skipping
    ROW < 0. One program per token; nothing is shared between programs. The accumulator
    starts at -0.0 so the first add is an exact identity (-0.0 + x == x for every x)."""
    t = tl.program_id(0).to(tl.int64)
    for h0 in tl.static_range(0, H, BH):
        rh = h0 + tl.arange(0, BH)
        acc = tl.full([BH], -0.0, dtype=tl.float32)
        for s in tl.static_range(S):
            row = tl.load(ROW + t * S + s).to(tl.int64)
            wt = tl.load(Wt + t * S + s)
            y = tl.load(Y + row * H + rh, mask=(rh < H) & (row >= 0), other=0.0).to(tl.float32)
            acc = tl.where(row >= 0, acc + wt * y, acc)
        tl.store(OUT + t * H + rh, acc.to(OUT.dtype.element_ty), mask=rh < H)


def slot_sum(y, wt, row, out_dtype):
    """y [N, H] (bf16 or fp32), wt [T, S] fp32, row [T, S] int32 -> [T, H] out_dtype.
    enable_fp_fusion=False keeps product and add as two roundings, as the torch-op sum does."""
    T, S = row.shape
    out = torch.empty(T, H, device=row.device, dtype=out_dtype)
    if T:
        if y.numel() == 0:
            y = torch.zeros(1, H, device=row.device, dtype=y.dtype)
        _slot_sum_kernel[(T,)](y, wt.contiguous(), row.contiguous(), out, T, H=H, S=S, BH=1024,
                               num_warps=4, enable_fp_fusion=False)
    return out


# ----------------------------------------------------------------------------- EP layer
def moe_ep(x, P, E, K, variant, group=None):
    """One MoE layer on this rank's tokens x [T, H]. Returns (out [T, H] bf16, ids, keep).
    `group` is the EP group (default: all ranks); P holds this rank's experts in it."""
    W, dev = dist.get_world_size(group), x.device
    El = E // W
    T = x.shape[0]
    w, ids = route(x, P["Wr"], K)
    keep = capacity_keep(ids, E) if variant == "capacity" else torch.ones_like(ids, dtype=torch.bool)
    flat = ids.reshape(-1)                                         # index t*K + j
    kept = keep.reshape(-1).nonzero().squeeze(1)
    order = kept[torch.sort(flat[kept], stable=True).indices]      # by expert, then (token, rank)
    tok, slot, exp = order // K, order % K, flat[order]

    # counts: cnt[e] rows for expert e; chunk d (experts of rank d) goes to rank d
    cnt = torch.bincount(exp, minlength=E)
    rcnt = torch.empty_like(cnt)
    dist.all_to_all_single(rcnt, cnt, group=group)
    hc = torch.cat([cnt, rcnt]).tolist()
    send = [sum(hc[d * El:(d + 1) * El]) for d in range(W)]
    rmat = [hc[E + s * El:E + (s + 1) * El] for s in range(W)]     # rows from source s, per local expert
    recv = [sum(r) for r in rmat]

    # dispatch: data movement only
    xs = x[tok]
    xr = x.new_empty(sum(recv), H)
    dist.all_to_all_single(xr, xs, recv, send, group=group)
    if variant == "rank_partial":
        meta = torch.stack([tok, slot, w[tok, slot].view(torch.int32).long()], 1)
        rmeta = meta.new_empty(sum(recv), 3)
        dist.all_to_all_single(rmeta, meta, recv, send, group=group)

    # received rows are source-major; regroup expert-major for the expert GEMMs
    src_off = [sum(recv[:s]) for s in range(W)]
    segs = [(src_off[s] + sum(rmat[s][:l]), rmat[s][l]) for l in range(El) for s in range(W)]
    perm = (torch.cat([torch.arange(a, a + n, device=dev) for a, n in segs if n])
            if sum(recv) else torch.zeros(0, dtype=torch.long, device=dev))
    xe = xr[perm]
    ye = torch.empty_like(xe)
    o = 0
    for l in range(El):
        m = sum(rmat[s][l] for s in range(W))
        if m:
            a = xe[o:o + m]
            h = F.silu(mm_one_config(a, P["W1"][l])) * mm_one_config(a, P["W3"][l])
            ye[o:o + m] = mm_one_config(h, P["W2"][l])
            o += m
    yr = torch.empty_like(ye)
    yr[perm] = ye                                                  # back to source-major

    if variant == "rank_partial":
        return _rank_partial(yr, rmeta, recv, tok, exp, El, W, T, K, dev, group), ids, keep

    yb = x.new_empty(len(order), H)                               # row i belongs to order[i]
    dist.all_to_all_single(yb, yr, send, recv, group=group)

    if variant in ("arrival_atomic", "arrival_det"):
        out = torch.zeros_like(x)
        contrib = yb * w[tok, slot].unsqueeze(1).to(BF)
        if variant == "arrival_det":
            prev = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
            try:
                out.index_add_(0, tok, contrib)
            finally:
                torch.use_deterministic_algorithms(prev)
        else:
            out.index_add_(0, tok, contrib)                       # atomics
    elif variant == "fixed_slot_torch":
        per = torch.zeros(T, K, H, device=dev, dtype=torch.float32)
        per[tok, slot] = yb.float() * w[tok, slot].unsqueeze(1)   # unique (token, rank) slots
        acc = per[:, 0]
        for j in range(1, K):
            acc = acc + per[:, j]
        out = acc.to(BF)
    else:                                                          # fixed_slot, capacity
        row = torch.full((T, K), -1, dtype=torch.int32, device=dev)
        row[tok, slot] = torch.arange(len(order), dtype=torch.int32, device=dev)
        out = slot_sum(yb, w, row, BF)
    return out, ids, keep


def _rank_partial(yr, rmeta, recv, tok, exp, El, W, T, K, dev, group=None):
    # expert side: one fp32 partial per (source rank, token), its local experts in top-k rank order
    n = yr.shape[0]
    src = torch.repeat_interleave(torch.arange(W, device=dev), torch.tensor(recv, device=dev))
    rt, rj = rmeta[:, 0], rmeta[:, 1]
    rw = rmeta[:, 2].contiguous().int().view(torch.float32)
    uq, inv = torch.unique(src * BIG + rt, sorted=True, return_inverse=True)
    rowp = torch.full((len(uq), K), -1, dtype=torch.int32, device=dev)
    rowp[inv, rj] = torch.arange(n, dtype=torch.int32, device=dev)
    wp = torch.zeros(len(uq), K, dtype=torch.float32, device=dev)
    wp[inv, rj] = rw
    part = slot_sum(yr, wp, rowp, torch.float32)
    # source side: the same (destination rank, token) list in the same (sorted) order
    uq2 = torch.unique((exp // El) * BIG + tok, sorted=True)
    hc = torch.cat([torch.bincount(uq // BIG, minlength=W), torch.bincount(uq2 // BIG, minlength=W)]).tolist()
    pb = part.new_empty(len(uq2), H)
    dist.all_to_all_single(pb, part, hc[W:], hc[:W], group=group)
    rowq = torch.full((T, W), -1, dtype=torch.int32, device=dev)
    rowq[uq2 % BIG, uq2 // BIG] = torch.arange(len(uq2), dtype=torch.int32, device=dev)
    return slot_sum(pb, torch.ones(T, W, device=dev), rowq, BF)   # rank order, fp32, one rounding


def ref_fp32(x, P, E, K):
    """Same routing, fp32 experts and combine: a numerical sanity check only (EP=1)."""
    w, ids = route(x, P["Wr"], K)
    out = torch.zeros(x.shape[0], H, device=x.device)
    for e in range(E):
        t, j = (ids == e).nonzero(as_tuple=True)
        if t.numel():
            xe = x[t].float()
            ye = (F.silu(xe @ P["W1"][e].float()) * (xe @ P["W3"][e].float())) @ P["W2"][e].float()
            out.index_add_(0, t, ye * w[t, j, None])
    return out


# ----------------------------------------------------------------------------- helpers
def gather_global(t, parts, Tg, group=None):
    """Rank 0 of the group gets the rows of every rank, placed at their global token indices."""
    W, r = dist.get_world_size(group), dist.get_rank(group)
    mx = max(len(p) for p in parts)
    buf = t.new_zeros((mx,) + tuple(t.shape[1:]))
    buf[:t.shape[0]] = t
    bufs = [torch.empty_like(buf) for _ in range(W)]
    dist.all_gather(bufs, buf, group=group)
    if r != 0:
        return None
    g = t.new_empty((Tg,) + tuple(t.shape[1:]))
    for p, b in zip(parts, bufs):
        g[p.to(t.device)] = b[:len(p)]
    return g


def shardings(Tg, W):
    perm = torch.randperm(Tg, generator=torch.Generator().manual_seed(5))
    wts = [d + 1 for d in range(W)]
    sizes = [Tg * v // sum(wts) for v in wts]
    sizes[-1] += Tg - sum(sizes)
    uneven, o = [], 0
    for s in sizes:
        uneven.append(perm[o:o + s])
        o += s
    return {"contig": list(torch.arange(Tg).tensor_split(W)),
            "strided": [torch.arange(d, Tg, W) for d in range(W)],
            "uneven": uneven}


def filler(kind, n, seed, dev):
    if kind == "zeros":
        return torch.zeros(n, H, device=dev, dtype=BF)
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(n, H, generator=g, device=dev)
    return (x * 1e3 if kind == "large" else x).to(BF)


def allsum(v, dev):
    t = torch.tensor([float(v)], device=dev, dtype=torch.float64)
    dist.all_reduce(t)
    return t.item()


# ----------------------------------------------------------------------------- tests
def gemm_checks(P, K, dev, sizes=DEFAULT_SIZES):
    """L2 of the building blocks on this GPU (rank 0 of the EP=1 run)."""
    res = {}
    torch.manual_seed(1)
    cases = (("router H->E", P["Wr"]), ("expert up H->I", P["W1"][0]), ("expert down I->H", P["W2"][0]))
    for name, Wm in cases:
        rows = torch.randn(2, Wm.shape[0], device=dev).to(BF)
        for gname, fn in (("mm_one_config", lambda a, Wm=Wm: mm_one_config(a, Wm)),
                          ("cublas", lambda a, Wm=Wm: a @ Wm)):
            rep = check_batch(fn, rows, sizes, name=f"{gname} {name}")
            res[rep.name] = {"summary": rep.summary(), "failures": rep.failures[:10]}
            print(rep.summary(), flush=True)
    rows = torch.randn(2, H, device=dev).to(BF)
    for nm, fn in (("router weights", lambda a: route(a, P["Wr"], K)[0]),
                   ("router ids", lambda a: route(a, P["Wr"], K)[1])):
        rep = check_batch(fn, rows, sizes, name=nm)
        res[nm] = {"summary": rep.summary(), "failures": rep.failures[:10]}
        print(rep.summary(), flush=True)
    return res


def run_l1(P, E, K, variants, Xg, dev, n=5):
    W, r = dist.get_world_size(), dist.get_rank()
    x = Xg[shardings(Xg.shape[0], W)["contig"][r].to(dev)]
    res = {}
    for v in variants:
        ref = moe_ep(x, P, E, K, v)[0]
        bad_runs, bad_elems = 0, 0
        for _ in range(n - 1):
            mm = first_mismatch(ref, moe_ep(x, P, E, K, v)[0])
            m = mm["mismatched_elements"] if mm else 0
            tot = allsum(m, dev)
            bad_runs += tot > 0
            bad_elems = max(bad_elems, int(tot))
        res[v] = {"runs": n, "failed_runs": int(bad_runs), "max_mismatched_elements": bad_elems,
                  "of": Xg.shape[0] * H, "passed": bad_runs == 0}
        if r == 0:
            print(f"L1 {v:18s} failed {bad_runs}/{n - 1} max mismatched {bad_elems}", flush=True)
    return res


def run_l2(P, E, K, variants, probe, dev, sizes=L2_SIZES):
    W, r = dist.get_world_size(), dist.get_rank()
    Pn = probe.shape[0]
    wheres = ("same",) if W == 1 else ("same", "others", "both")
    res = {}
    for v in variants:
        x0 = probe if r == 0 else probe[:0]
        ref, rids, rkeep = moe_ep(x0, P, E, K, v)
        checked, fails = 0, []
        for where in wheres:
            for n in sizes:
                for pos in ("first", "last"):
                    for fill in ("random", "large", "zeros"):
                        has = (where in ("same", "both")) if r == 0 else (where in ("others", "both"))
                        seed = 1 + r * 1_000_003 + n * 1009 + ("first", "last").index(pos) * 17 + \
                            ("random", "large", "zeros").index(fill) * 3 + wheres.index(where) * 7
                        f = filler(fill, n if has else 0, seed, dev)
                        x = (torch.cat([probe, f]) if pos == "first" else torch.cat([f, probe])) if r == 0 else f
                        out, ids, keep = moe_ep(x, P, E, K, v)
                        if r == 0:
                            p0 = 0 if pos == "first" else f.shape[0]
                            checked += 1
                            mm = first_mismatch(ref, out[p0:p0 + Pn])
                            if mm:
                                fails.append({"where": where, "tokens_per_rank": n, "position": pos, "filler": fill,
                                              "mismatched_elements": mm["mismatched_elements"],
                                              "ids_changed": not torch.equal(rids, ids[p0:p0 + Pn]),
                                              "dropped_changed": not torch.equal(rkeep, keep[p0:p0 + Pn])})
        if r == 0:
            summ = {"checked": checked, "failed": len(fails), "passed": not fails, "of_elements": Pn * H}
            if fails:
                summ["failing_where"] = sorted({f["where"] for f in fails})
                summ["failing_sizes"] = sorted({f["tokens_per_rank"] for f in fails})
                summ["failing_fillers"] = sorted({f["filler"] for f in fails})
                summ["failing_positions"] = sorted({f["position"] for f in fails})
                summ["max_mismatched_elements"] = max(f["mismatched_elements"] for f in fails)
                summ["fails_with_dropped_changed"] = sum(f["dropped_changed"] for f in fails)
                summ["fails_with_ids_changed"] = sum(f["ids_changed"] for f in fails)
                summ["by_where"] = {w_: sum(f["where"] == w_ for f in fails) for w_ in wheres}
                summ["failures"] = fails[:40]
            res[v] = summ
            print(f"L2 {v:18s} {'PASS' if not fails else 'FAIL'} {len(fails)}/{checked}"
                  + (f" where={summ['failing_where']} sizes={summ['failing_sizes']} fill={summ['failing_fillers']}"
                     f" maxel={summ['max_mismatched_elements']}" if fails else ""), flush=True)
    return res


def run_dumps(P, E, K, variants, Xg, probe, dev, outdir):
    W, r = dist.get_world_size(), dist.get_rank()
    Tg = Xg.shape[0]
    sh = shardings(Tg, W)
    sanity = {}
    ref = ref_fp32(Xg, P, E, K) if W == 1 else None
    for v in variants:
        for name, parts in sh.items():
            out, ids, keep = moe_ep(Xg[parts[r].to(dev)], P, E, K, v)
            go, gi, gk = (gather_global(t, parts, Tg) for t in (out, ids, keep.to(torch.uint8)))
            if r == 0:
                torch.save({"out": go.cpu(), "ids": gi.cpu(), "keep": gk.cpu()}, os.path.join(outdir, f"{v}__{name}.pt"))
                if ref is not None and name == "contig":
                    err = (go.float() - ref).abs()
                    sanity[v] = {"max_abs": float(err.max()), "mean_abs": float(err.mean()),
                                 "ref_mean_abs": float(ref.abs().mean())}
        out, ids, keep = moe_ep(probe if r == 0 else probe[:0], P, E, K, v)
        if r == 0:
            torch.save({"out": out.cpu(), "ids": ids.cpu(), "keep": keep.to(torch.uint8).cpu()},
                       os.path.join(outdir, f"{v}__alone.pt"))
    return sanity


def run_timing(P, E, K, variants, dev, tokens, iters=10, reps=2):
    W, r = dist.get_world_size(), dist.get_rank()
    res = {}
    for n in tokens:
        x = filler("random", n, 99 + r, dev)
        res[n] = {}
        best = {v: float("inf") for v in variants}
        for _ in range(reps):
            for v in variants:
                for _ in range(3):
                    moe_ep(x, P, E, K, v)
                torch.cuda.synchronize()
                dist.barrier()
                t0 = time.perf_counter()
                for _ in range(iters):
                    moe_ep(x, P, E, K, v)
                torch.cuda.synchronize()
                t = torch.tensor([(time.perf_counter() - t0) / iters * 1e3], device=dev)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                best[v] = min(best[v], t.item())
        res[n] = best
        if r == 0:
            print(f"timing tokens/rank={n}: " + " ".join(f"{v}={best[v]:.2f}ms" for v in variants), flush=True)
    return res


# ----------------------------------------------------------------------------- comparison
def _span(ids, El):
    rk = (ids // El).sort(1).values
    return 1 + (rk[:, 1:] != rk[:, :-1]).sum(1)


def compare(root, json_out):
    res = {}
    for cdir in sorted(glob.glob(os.path.join(root, "E*K*"))):
        cfg = os.path.basename(cdir)
        E = int(cfg[1:cfg.index("K")])
        runs = sorted(d for d in os.listdir(cdir) if os.path.isfile(os.path.join(cdir, d, "run.json")))
        meta = {d: json.load(open(os.path.join(cdir, d, "run.json"))) for d in runs}
        entry = {"runs": meta, "L5": {}, "cross_variant_ep1": {}}
        load = lambda d, v, s: torch.load(os.path.join(cdir, d, f"{v}__{s}.pt"), weights_only=True)
        for v in VARIANTS:
            entry["L5"][v] = {}
            base = {s: load("ep1", v, s) for s in ("contig", "alone")}
            for d in runs:
                W = meta[d]["world"]
                entry["L5"][v][d] = {}
                for s in ("contig", "strided", "uneven", "alone"):
                    p = os.path.join(cdir, d, f"{v}__{s}.pt")
                    if not os.path.exists(p):
                        continue
                    a, b = base["alone" if s == "alone" else "contig"], torch.load(p, weights_only=True)
                    diff = bits(a["out"]) != bits(b["out"])
                    rec = {"mismatched_elements": int(diff.sum()), "of": diff.numel(),
                           "mismatched_tokens": int(diff.any(1).sum()), "tokens": diff.shape[0],
                           "ids_equal": torch.equal(a["ids"], b["ids"]), "keep_equal": torch.equal(a["keep"], b["keep"])}
                    if rec["mismatched_elements"] and s != "alone":
                        span = _span(a["ids"], E // W)
                        bad = diff.any(1)
                        rec["by_rank_span"] = {int(k): [int((bad & (span == k)).sum()), int((span == k).sum())]
                                               for k in span.unique().tolist()}
                        kept_same = (a["keep"] == b["keep"]).all(1)
                        rec["mismatched_tokens_with_same_drops"] = int((bad & kept_same).sum())
                    entry["L5"][v][d][s] = rec
            print(cfg, v, {d: {s: entry["L5"][v][d][s]["mismatched_elements"] for s in entry["L5"][v][d]}
                           for d in runs}, flush=True)
        fs = load("ep1", "fixed_slot", "contig")["out"]
        for v in VARIANTS:
            if v != "fixed_slot":
                diff = bits(fs) != bits(load("ep1", v, "contig")["out"])
                entry["cross_variant_ep1"][f"fixed_slot vs {v}"] = {"mismatched_elements": int(diff.sum()),
                                                                     "of": diff.numel()}
        print(cfg, "cross-variant", entry["cross_variant_ep1"], flush=True)
        res[cfg] = entry
    with open(json_out, "w") as fh:
        json.dump(res, fh, indent=1)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=32)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--dump", default=os.path.join(HERE, "dumps"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--tg", type=int, default=1024, help="global tokens for L1 and L5")
    ap.add_argument("--parts", default="gemm,l1,l2,l5,timing")
    ap.add_argument("--timing-tokens", type=int, nargs="*", default=[64, 512, 4096])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--quick", action="store_true", help="fewer L2 and GEMM-check sizes (regression suite)")
    ap.add_argument("--compare")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare, args.json or os.path.join(HERE, "results-a16.json"))

    dist.init_process_group("nccl")
    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    W, r = dist.get_world_size(), dist.get_rank()
    assert args.E % W == 0
    parts = set(args.parts.split(","))
    variants = args.variants.split(",")
    cfg = f"E{args.E}K{args.K}" + (f"T{args.tg}" if args.tg != 1024 else "")
    outdir = os.path.join(args.dump, cfg, f"ep{W}{args.tag}")
    if r == 0:
        os.makedirs(outdir, exist_ok=True)
    P = make_params(args.E, r, W, dev)
    probe = _randn((2, H), 11, 1.0).to(dev)
    Xg = _randn((args.tg, H), 7, 1.0).to(dev)
    res = {"world": W, "tag": args.tag, "E": args.E, "K": args.K, "H": H, "I": I, "cap_factor": CAP_FACTOR,
           "global_tokens_L1_L5": args.tg,
           "device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
           "nccl": ".".join(map(str, torch.cuda.nccl.version())),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all")}
    if "gemm" in parts and W == 1:
        res["gemm_checks"] = gemm_checks(P, args.K, dev, QUICK_GEMM_SIZES if args.quick else DEFAULT_SIZES)
    if "l1" in parts:
        res["L1"] = run_l1(P, args.E, args.K, variants, Xg, dev)
    if "l2" in parts:
        res["L2"] = run_l2(P, args.E, args.K, variants, probe, dev,
                           sizes=QUICK_L2_SIZES if args.quick else L2_SIZES)
    if "l5" in parts:
        res["vs_fp32_ref"] = run_dumps(P, args.E, args.K, variants, Xg, probe, dev, outdir)
    if "timing" in parts:
        res["timing_ms"] = run_timing(P, args.E, args.K, variants, dev, args.timing_tokens)
    if r == 0:
        with open(os.path.join(outdir, "run.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
