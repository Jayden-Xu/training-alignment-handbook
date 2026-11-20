"""Collectives and tensor-parallel (TP) linear layers: batch (L2) and parallel-degree (L5) invariance.

Run under torch.distributed.run on one 8-GPU node, one launch per NCCL setting (NCCL reads
NCCL_ALGO / NCCL_PROTO at init). run_all.sh does every launch and the merge.

Part `coll` (8 ranks):
  1. all-reduce    a fixed slice whose per-rank values are fixed (4 rows x 4096, and 512 elements),
                   reduced alone and inside larger messages (1 KB .. 256 MB), at the first, middle
                   and last position. Compared bitwise to the slice reduced alone; every rank's copy
                   compared to rank 0's; every result element matched against the 16 sequential
                   ring orders (start rank, direction) to see which order produced it.
  2. reduce-scatter  the same slice at a fixed place inside every rank's output chunk, chunks of
                   4 .. 4096 rows; also the chunk owned by rank j against the chunk owned by rank 0.
  3. fallbacks     all-gather + local fp32 sum in rank order; all-to-all + local fp32 sum + all-gather.
  4. repeat (L1)   5 identical calls.
Part `tp` (8 ranks; TP groups of 2 and 4 are subgroups of ranks 0..t-1):
  1. fp32-output GEMM variant: rounding its output to bf16 gives exactly mm_one_config.
  2. L2  row-parallel (mm_one_config per K shard, then all-reduce), row-parallel with the invariant
         fallback in place of the all-reduce, row-parallel with sequence parallelism (reduce-scatter
         over tokens, all-gathered for the check), the column-parallel control (N shard, then
         all-gather), and the logical-TP-8 schemes, over M = 1 .. 4096.
  3. L1  repeat.
  4. L5  TP = 1, 2, 4, 8 against the unsharded GEMM; logical TP 8 at every degree.
  5. accuracy against fp64; timing of collectives, fallbacks and TP schemes.

Logical TP 8: K is split into 8 fixed chunks. Every configuration computes the 8 chunk partials
with the same kernel in fp32 and moves them without in-network reduction, then sums them in fp32
in a fixed order and rounds to bf16 once. TP=1 runs the same 8 chunks on one GPU.
  lt8_gather  all-gather every partial, sum in chunk order 0..7 on every rank
  lt8_a2a     all-to-all the partials by column block, sum in chunk order 0..7, all-gather bf16
  lt8_tree    a fixed pairwise tree over the 8 chunks: each rank reduces its own aligned subtree
              locally, all-to-all one fp32 partial by column block, finishes the tree, all-gathers bf16

Two nodes (16 ranks, ~/bi-check-xnode/launch2.sh, see run_2node.sh): --xnode caps messages at
64 MB, uses logical TP 16, and also runs the 8-chunk scheme at TP 16 two ways: lt8_2d (the two
ranks of a chunk split its N columns: bitwise the same partials) and lt8_splitk (they split its K
range: different partials, shown as the counterexample).

Merge:  python collectives_tp.py --merge out --json results-a16.json

--quick (the regression suite: `run_all.sh out --quick`, `XARGS=--quick run_2node.sh out`): messages
up to 4 MB instead of 256 MB (64 MB on two nodes), L2 over M = 1 .. 64, L5 at M = 1 and 256, and a
shorter timing. Every sweep keeps all its sizes below the cap.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import torch
import torch.distributed as dist
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from bi_check import FILLERS, bits, bit_equal, check_batch, check_repeat, first_mismatch  # noqa: E402
from bi_kernels import BF, mm_one_config  # noqa: E402

F32 = torch.float32
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}
M_SIZES = (1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512, 1024, 2048, 4096)
ROW = 4096                     # row width of the collective sweeps


# ----------------------------------------------------------------------------- fp32-output GEMM
@triton.jit(do_not_specialize=["M"])
def _mm_f32_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # tools/bi_kernels.py::_mm_kernel with the fp32 accumulator stored as is (no bf16 rounding).
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def mm_one_config_f32(a, b, out=None, BM=64, BN=64, BK=32):
    M, K = a.shape
    N = b.shape[1]
    c = out if out is not None else torch.empty(M, N, device=a.device, dtype=F32)
    if M:
        _mm_f32_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
            a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return c


# ----------------------------------------------------------------------------- helpers
def sha(t):
    return hashlib.sha1(bits(t).cpu().numpy().tobytes()).hexdigest()[:16]


def mismatched(a, b):
    mm = first_mismatch(a, b)
    return 0 if mm is None else mm["mismatched_elements"]


def max_abs_diff(a, b):
    return float((a.double() - b.double()).abs().max())


def timed(fn, group, iters, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    if group is not False:
        dist.barrier(group=group)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def summarize(rep):
    """check_batch report -> pass/fail, failing sizes, mismatched elements per size/position."""
    by = {}
    for f in rep.failures:
        by.setdefault(str(f["size"]), {}).setdefault(f["position"], []).append(f["mismatched_elements"])
    sizes = sorted({f["size"] for f in rep.failures})
    return {"passed": rep.passed, "checked": rep.checked, "failed": len(rep.failures),
            "failing_sizes": sizes, "of": rep.failures[0]["of"] if rep.failures else None,
            "max_mismatched": max((f["mismatched_elements"] for f in rep.failures), default=0),
            "by_size": by}


# ----------------------------------------------------------------------------- collectives
def all_reduce(m, group=None):
    dist.all_reduce(m, group=group)
    return m


def reduce_scatter(inp, world, group=None):
    out = torch.empty(inp.numel() // world, device=inp.device, dtype=inp.dtype)
    dist.reduce_scatter_tensor(out, inp, group=group)
    return out


def fallback_gather(m, world, group=None):
    """All-gather every rank's tensor, then sum in rank order 0..W-1 in fp32, round once."""
    g = torch.empty(world, m.numel(), device=m.device, dtype=m.dtype)
    dist.all_gather_into_tensor(g, m, group=group)
    acc = g[0].float().clone()
    for r in range(1, world):
        acc.add_(g[r])
    return acc.to(m.dtype)


def fallback_rs_a2a(inp, world, group=None):
    """Deterministic reduce-scatter: all-to-all, then sum the W pieces in rank order in fp32."""
    recv = torch.empty_like(inp)
    dist.all_to_all_single(recv, inp, group=group)
    recv = recv.view(world, -1)
    acc = recv[0].float().clone()
    for r in range(1, world):
        acc.add_(recv[r])
    return acc.to(inp.dtype)


def fallback_a2a(m, world, group=None):
    """Deterministic all-reduce at ring-like traffic: the reduce-scatter above, then all-gather."""
    part = fallback_rs_a2a(m, world, group)
    out = torch.empty_like(m)
    dist.all_gather_into_tensor(out, part, group=group)
    return out


def hier_groups(rank, world, local=8):
    """Groups for the two-level fixed order on 2 nodes x `local` ranks: the rank's node, and the pair
    (i, i + local) of ranks with the same local index. Every rank creates every group, in order."""
    if world != 2 * local:
        return None
    nodes = [dist.new_group(list(range(i * local, (i + 1) * local))) for i in range(2)]
    pairs = [dist.new_group([i, i + local]) for i in range(local)]
    return {"local": local, "node": nodes[rank // local], "pair": pairs[rank % local]}


def fallback_hier(m, hg):
    """Two-level fixed-order all-reduce: inside each node, all-to-all and an fp32 sum over its ranks
    in rank order; across nodes, node 0's sum + node 1's sum (one fp32 exchange with the partner);
    one rounding; all-gather inside the node. Every element gets the same order at every size."""
    L, n = hg["local"], m.numel()
    recv = torch.empty_like(m)
    dist.all_to_all_single(recv, m, group=hg["node"])
    recv = recv.view(L, n // L)
    acc = recv[0].float().clone()
    for r in range(1, L):
        acc.add_(recv[r])
    both = torch.empty(2, n // L, device=m.device, dtype=F32)
    dist.all_gather_into_tensor(both, acc, group=hg["pair"])
    part = (both[0] + both[1]).to(m.dtype)
    out = torch.empty_like(m)
    dist.all_gather_into_tensor(out, part, group=hg["node"])
    return out


def slice_values(rank, S, dtype, geom):
    g = torch.Generator().manual_seed(1000 + 100 * geom + rank)
    return torch.randn(S, generator=g).to(dtype)


def order_candidates(slices, dtype):
    """The slice summed over ranks in every ring order: start rank a, direction +1 (fwd) or -1
    (bwd), rounding to dtype after every add, as a ring or chain reduction does. For bf16 also the
    fallback's arithmetic (fp32 sum in rank order, one rounding)."""
    W = len(slices)
    names, outs = [], []
    for d, tag in ((1, "fwd"), (-1, "bwd")):
        for a in range(W):
            order = [(a + d * i) % W for i in range(W)]
            s = slices[order[0]]
            for k in order[1:]:
                s = (s.float() + slices[k].float()).to(dtype)
            names.append(f"{tag}{a}")
            outs.append(s)
    if W == 16:
        # Two nodes of 8 (NCCL INIT log): per node a chain into its first rank (7->0, 15->8); channel 0
        # has root 0 with remote child 8, channel 1 root 8 with remote child 0. The root combines its
        # own value, its chain child's sum and the remote node's sum in some order: try the plausible ones.
        def add(a, b):
            return (a.float() + b.float()).to(dtype)

        def chain(order):
            s = slices[order[0]]
            for k in order[1:]:
                s = add(s, slices[k])
            return s
        c0, c1 = chain(range(7, -1, -1)), chain(range(15, 7, -1))      # whole-node chains
        s0, s1 = chain(range(7, 0, -1)), chain(range(15, 8, -1))       # chains below each root
        for name, v in (("tree:C(7..0)+C(15..8)", add(c0, c1)),
                        ("tree:(x0+C(15..8))+C(7..1)", add(add(slices[0], c1), s0)),
                        ("tree:(x8+C(7..0))+C(15..9)", add(add(slices[8], c0), s1)),
                        ("tree:(C(7..1)+C(15..8))+x0", add(add(s0, c1), slices[0])),
                        ("tree:(C(15..9)+C(7..0))+x8", add(add(s1, c0), slices[8]))):
            names.append(name)
            outs.append(v)
    if dtype != F32:
        acc = slices[0].float()
        for k in range(1, W):
            acc = acc + slices[k].float()
        names.append("fp32_rank_order")
        outs.append(acc.to(dtype))
    return names, torch.stack(outs)


def order_cover(names, cands, res, top=8):
    """Greedy cover of the result's elements by candidate orders: which sequential orders explain
    the bits, how many are needed, and how many elements no candidate explains."""
    match = bits(cands) == bits(res)[None]
    remaining = torch.ones(res.numel(), dtype=torch.bool, device=res.device)
    chosen = []
    while True:
        gains = (match & remaining).sum(1)
        c = int(gains.argmax())
        if int(gains[c]) == 0:
            break
        chosen.append([names[c], int(gains[c])])
        remaining &= ~match[c]
    return {"orders": chosen[:top], "n_orders": len(chosen), "unexplained": int(remaining.sum())}


def positions(n, S, unit):
    return {"first": 0, "middle": ((n - S) // 2) // unit * unit, "last": n - S}


SWEEPS = {
    "rows": {"S": 4 * ROW, "unit": ROW, "geom": 0,
             "n": {"bf16": [r * ROW for r in (4, 5, 8, 12, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)],
                   "fp32": [r * ROW for r in (4, 5, 8, 12, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)]}},
    "small": {"S": 512, "unit": 64, "geom": 1,
              "n": {k: [512, 640, 1024, 2048, 4096, 8192, 16384, 65536, 262144, 1048576] for k in DT}},
}
RS_ROWS = {"bf16": (4, 5, 8, 16, 64, 256, 1024, 4096), "fp32": (4, 5, 8, 16, 64, 256, 1024, 2048)}
POS_ID = {"first": 0, "middle": 1, "last": 2}


def message(n, S, off, sl, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = torch.randn(n, device="cuda", generator=g).to(dtype)
    m[off:off + S] = sl
    return m


def coll_part(rank, world, args):
    dev = torch.device("cuda")
    quick, cap, gcap = args.quick, args.max_bytes, args.gather_max_bytes
    out = {"allreduce": [], "fallback": [], "reduce_scatter": [], "fallback_rs": [], "repeat": {}}
    hg = hier_groups(rank, world)
    for di, (dname, dtype) in enumerate(DT.items()):
        esize = torch.tensor([], dtype=dtype).element_size()
        # ------------------------------------------------ all-reduce and its fallbacks
        for sname, sw in SWEEPS.items():
            S, unit = sw["S"], sw["unit"]
            sizes = [n for n in sw["n"][dname] if not cap or n * esize <= cap]
            sl = slice_values(rank, S, dtype, sw["geom"]).to(dev)
            if rank == 0:
                every = [slice_values(r, S, dtype, sw["geom"]).to(dev) for r in range(world)]
                names, cands = order_candidates(every, dtype)
            ref = ref_fb = ref_fh = None
            for n in sizes:
                for pos, off in positions(n, S, unit).items():
                    if n == S and pos != "first":
                        continue
                    seed = (rank + 1) * 1_000_003 + n * 31 + POS_ID[pos] * 7 + di
                    m = message(n, S, off, sl, dtype, seed)
                    keep = m.clone()
                    res = all_reduce(m)[off:off + S].clone()
                    fba_full = fallback_a2a(keep, world)
                    if not gcap or n * esize <= gcap:      # the all-gather form holds world x the message
                        fbg_full = fallback_gather(keep, world)
                        same_fb = bit_equal(fbg_full, fba_full)
                    else:
                        fbg_full, same_fb = fba_full, None
                    fbg = fbg_full[off:off + S].clone()
                    fbh = fallback_hier(keep, hg)[off:off + S].clone() if hg else None
                    del m, keep, fbg_full, fba_full
                    every_res = torch.empty(world, S, device=dev, dtype=dtype)
                    dist.all_gather_into_tensor(every_res, res)
                    if rank == 0:
                        if ref is None:
                            ref, ref_fb = res.clone(), fbg.clone()
                            ref_fh = fbh.clone() if fbh is not None else None
                        out["allreduce"].append({
                            "dtype": dname, "sweep": sname, "n": n, "bytes": n * res.element_size(),
                            "pos": pos, "offset": off, "of": S,
                            "mismatched": mismatched(ref, res), "max_abs_diff": max_abs_diff(ref, res),
                            "ranks_disagree": sum(int(not bit_equal(every_res[r], every_res[0])) for r in range(world)),
                            "cover": order_cover(names, cands, res), "hash": sha(res)})
                        out["fallback"].append({
                            "dtype": dname, "sweep": sname, "n": n, "bytes": n * res.element_size(),
                            "pos": pos, "of": S, "mismatched": mismatched(ref_fb, fbg),
                            "gather_equals_a2a": same_fb, "matches_fp32_rank_order_or_fwd0":
                                bit_equal(fbg, cands[names.index("fp32_rank_order" if dtype != F32 else "fwd0")]),
                            "hash": sha(fbg)})
                        if fbh is not None:
                            out["fallback"][-1].update({"hier_mismatched": mismatched(ref_fh, fbh),
                                                        "hier_hash": sha(fbh)})
                    torch.cuda.empty_cache()
            if rank == 0:
                print(f"allreduce {dname} {sname}: " + " ".join(
                    f"{e['n']}/{e['pos'][0]}:{e['mismatched']}" for e in out["allreduce"]
                    if e["dtype"] == dname and e["sweep"] == sname), flush=True)

        # ------------------------------------------------ reduce-scatter and its fallback
        S = 4 * ROW
        sl = slice_values(rank, S, dtype, 0).to(dev)
        if rank == 0:
            every = [slice_values(r, S, dtype, 0).to(dev) for r in range(world)]
            names, cands = order_candidates(every, dtype)
        ref = ref_fb = None
        for rp in [r for r in RS_ROWS[dname] if not cap or world * r * ROW * esize <= cap]:
            c = rp * ROW
            for pos, off in positions(c, S, ROW).items():
                if c == S and pos != "first":
                    continue
                seed = (rank + 1) * 2_000_003 + c * 31 + POS_ID[pos] * 7 + di
                inp = message(world * c, S, off, sl, dtype, seed)
                for j in range(world):
                    inp[j * c + off:j * c + off + S] = sl
                res = reduce_scatter(inp, world)[off:off + S].clone()
                fb = fallback_rs_a2a(inp, world)[off:off + S].clone()
                del inp
                g_res = torch.empty(world, S, device=dev, dtype=dtype)
                g_fb = torch.empty(world, S, device=dev, dtype=dtype)
                dist.all_gather_into_tensor(g_res, res)
                dist.all_gather_into_tensor(g_fb, fb)
                if rank == 0:
                    if ref is None:
                        ref, ref_fb = g_res.clone(), g_fb.clone()
                    out["reduce_scatter"].append({
                        "dtype": dname, "rows_per_rank": rp, "bytes_total": world * c * res.element_size(),
                        "pos": pos, "of": S,
                        "mismatched_per_owner": [mismatched(ref[j], g_res[j]) for j in range(world)],
                        "owner_vs_owner0": [mismatched(g_res[0], g_res[j]) for j in range(world)],
                        "cover_per_owner": [order_cover(names, cands, g_res[j], top=3) for j in range(world)]})
                    out["fallback_rs"].append({
                        "dtype": dname, "rows_per_rank": rp, "pos": pos,
                        "mismatched_per_owner": [mismatched(ref_fb[j], g_fb[j]) for j in range(world)],
                        "owner_vs_owner0": [mismatched(g_fb[0], g_fb[j]) for j in range(world)]})
                torch.cuda.empty_cache()
        if rank == 0:
            print(f"reduce_scatter {dname}: " + " ".join(
                f"{e['rows_per_rank']}/{e['pos'][0]}:{sum(e['mismatched_per_owner'])}"
                for e in out["reduce_scatter"] if e["dtype"] == dname), flush=True)

        # ------------------------------------------------ repeat (L1)
        big = max(n for n in SWEEPS["rows"]["n"][dname] if not cap or n * esize <= cap) // ROW
        for R in ((4, 1024) if quick else (4, 1024, big)):
            m = message(R * ROW, 0, 0, sl[:0], dtype, rank * 13 + R)
            rep = check_repeat(lambda x: all_reduce(x.clone()), m, n=5, name=f"allreduce {dname} {R} rows")
            fails = torch.tensor([len(rep.failures)], device=dev)
            dist.all_reduce(fails)                    # integer sum over ranks: exact
            if rank == 0:
                out["repeat"][rep.name] = {**rep.to_dict(), "failures_all_ranks": int(fails)}
            del m
        for rp in (4, 1024):
            inp = message(world * rp * ROW, 0, 0, sl[:0], dtype, rank * 17 + rp)
            rep = check_repeat(lambda x: reduce_scatter(x, world), inp, n=5, name=f"reduce_scatter {dname} {rp} rows/rank")
            fails = torch.tensor([len(rep.failures)], device=dev)
            dist.all_reduce(fails)                    # integer sum over ranks: exact
            if rank == 0:
                out["repeat"][rep.name] = {**rep.to_dict(), "failures_all_ranks": int(fails)}
            del inp
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- tensor parallel
class TP:
    """Row-parallel, column-parallel and logical-TP linear layers y = x @ W on ranks 0..t-1.
    Logical TP C splits K into C fixed chunks; `chunk_counts` lists the values of C in use."""

    def __init__(self, W, rank, groups, chunk_counts=(8,), hg=None):
        self.W, self.hg = W, hg
        self.K, self.N = W.shape
        self.rank, self.groups = rank, groups
        self.w_row, self.w_col = {}, {}
        for t in groups:
            if rank < t:
                kt, nt = self.K // t, self.N // t
                self.w_row[t] = W[rank * kt:(rank + 1) * kt].contiguous()
                self.w_col[t] = W[:, rank * nt:(rank + 1) * nt].contiguous()
        self.w_ch = {C: [W[c * (self.K // C):(c + 1) * (self.K // C)].contiguous() for c in range(C)]
                     for C in chunk_counts}
        self._cache = {}

    def my_chunks(self, t, C):
        per = C // t
        return list(range(self.rank * per, (self.rank + 1) * per))

    # inputs as a rank holds them
    def shard_k(self, t, x):
        kt = self.K // t
        return x[:, self.rank * kt:(self.rank + 1) * kt].contiguous()

    def chunk_inputs(self, t, x, C):
        kc = self.K // C
        return [x[:, c * kc:(c + 1) * kc].contiguous() for c in self.my_chunks(t, C)]

    # layers on sharded inputs
    def row(self, t, xs):
        y = mm_one_config(xs, self.w_row[t])
        if t > 1:
            dist.all_reduce(y, group=self.groups[t])
        return y

    def row_fallback(self, t, xs):
        """Row-parallel with the all-reduce replaced by the invariant fallback (all-to-all, fp32 sum in
        rank order, all-gather). Partials are still rounded to bf16 by the GEMM, as in row_parallel."""
        y = mm_one_config(xs, self.w_row[t])
        return fallback_a2a(y.view(-1), t, self.groups[t]).view(y.shape)

    def row_fbhier(self, t, xs):
        """Row-parallel at TP = world = 2 nodes, with the two-level fixed-order fallback."""
        y = mm_one_config(xs, self.w_row[t])
        return fallback_hier(y.view(-1), self.hg).view(y.shape)

    def row_sp(self, t, xs):
        """Row-parallel with sequence parallelism: reduce-scatter over tokens (rows padded to a
        multiple of t), so each rank owns M/t tokens; then all-gather to compare the whole output."""
        y = mm_one_config(xs, self.w_row[t])
        M = y.shape[0]
        mp = -(-M // t) * t
        if mp != M:
            y = torch.cat([y, y.new_zeros(mp - M, y.shape[1])])
        mine = torch.empty(mp // t, y.shape[1], device=y.device, dtype=y.dtype)
        dist.reduce_scatter_tensor(mine, y, group=self.groups[t])
        out = torch.empty_like(y)
        dist.all_gather_into_tensor(out, mine, group=self.groups[t])
        return out[:M]

    def col(self, t, x):
        y = mm_one_config(x, self.w_col[t])
        if t == 1:
            return y
        out = torch.empty(t, *y.shape, device=y.device, dtype=y.dtype)
        dist.all_gather_into_tensor(out, y, group=self.groups[t])
        return out.permute(1, 0, 2).reshape(y.shape[0], self.N)

    def _partials(self, t, xcs, C):
        M = xcs[0].shape[0]
        P = torch.empty(len(xcs), M, self.N, device=xcs[0].device, dtype=F32)
        for i, c in enumerate(self.my_chunks(t, C)):
            mm_one_config_f32(xcs[i], self.w_ch[C][c], out=P[i])
        return P

    @staticmethod
    def _chunk_sum(G):
        acc = G[0].clone()
        for c in range(1, G.shape[0]):           # fixed chunk order 0..C-1, fp32
            acc.add_(G[c])
        return acc.to(BF)

    @staticmethod
    def _tree_sum(G):
        """Fixed pairwise tree over a power-of-two stack: (G0+G1), (G2+G3), ... level by level."""
        while G.shape[0] > 1:
            G = G[0::2] + G[1::2]
        return G[0]

    def lt_gather(self, t, xcs, C):
        P = self._partials(t, xcs, C)
        if t > 1:
            G = torch.empty(C, *P.shape[1:], device=P.device, dtype=F32)
            dist.all_gather_into_tensor(G, P, group=self.groups[t])
        else:
            G = P
        return self._chunk_sum(G)

    def lt_a2a(self, t, xcs, C):
        P = self._partials(t, xcs, C)            # [C/t, M, N]
        if t == 1:
            return self._chunk_sum(P)
        per, M, N = P.shape
        send = P.view(per, M, t, N // t).permute(2, 0, 1, 3).contiguous()   # [t, C/t, M, N/t]
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.groups[t])            # recv[s]: rank s's chunks
        part = self._chunk_sum(recv.view(C, M, N // t))                     # my N/t columns
        out = torch.empty(t, M, N // t, device=P.device, dtype=BF)
        dist.all_gather_into_tensor(out, part, group=self.groups[t])
        return out.permute(1, 0, 2).reshape(M, N)

    def lt_tree(self, t, xcs, C):
        """Logical TP C with a fixed pairwise tree over the C chunk partials, in fp32. A rank's chunks
        form one aligned subtree, so it reduces them locally, exchanges one fp32 partial (all-to-all
        by column block), finishes the upper tree levels in the same order, rounds once, all-gathers."""
        P = self._partials(t, xcs, C)            # [C/t, M, N]
        sub = self._tree_sum(P)                  # this rank's subtree, [M, N] fp32
        if t == 1:
            return sub.to(BF)
        M, N = sub.shape
        send = sub.view(M, t, N // t).permute(1, 0, 2).contiguous()       # [t, M, N/t]
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.groups[t])          # recv[s]: rank s's subtree
        part = self._tree_sum(recv).to(BF)                                # my N/t columns
        out = torch.empty(t, M, N // t, device=sub.device, dtype=BF)
        dist.all_gather_into_tensor(out, part, group=self.groups[t])
        return out.permute(1, 0, 2).reshape(M, N)

    def lt_treeh(self, t, xcs, C):
        """lt_tree at TP = world = 2 nodes, routed hierarchically: the node's chunks form one subtree,
        reduced after an all-to-all inside the node; the two node subtrees are added after one fp32
        exchange with the partner rank. Same pairwise tree, so the same bits as lt_tree."""
        P = self._partials(t, xcs, C)
        sub = self._tree_sum(P)                                           # this rank's subtree
        L = self.hg["local"]
        M, N = sub.shape
        send = sub.view(M, L, N // L).permute(1, 0, 2).contiguous()       # [L, M, N/L]
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.hg["node"])
        node = self._tree_sum(recv).contiguous()                          # this node's subtree, my columns
        both = torch.empty(2, M, N // L, device=sub.device, dtype=F32)
        dist.all_gather_into_tensor(both, node, group=self.hg["pair"])
        part = self._tree_sum(both).to(BF)                                # node 0 + node 1
        out = torch.empty(L, M, N // L, device=sub.device, dtype=BF)
        dist.all_gather_into_tensor(out, part, group=self.hg["node"])
        return out.permute(1, 0, 2).reshape(M, N)

    def _sub_weight(self, key, rows, cols):
        if key not in self._cache:
            self._cache[key] = self.W[rows, cols].contiguous()
        return self._cache[key]

    def lt_2d(self, t, x, C):
        """Logical TP C on t > C ranks: the g = t/C ranks of chunk c split its N columns, which leaves
        every chunk partial bitwise unchanged. All-gather, then the same pairwise tree as lt_tree."""
        g = t // C
        c, h = divmod(self.rank, g)
        kc, ng = self.K // C, self.N // g
        w = self._sub_weight(("2d", C, t), slice(c * kc, (c + 1) * kc), slice(h * ng, (h + 1) * ng))
        P = mm_one_config_f32(x[:, c * kc:(c + 1) * kc].contiguous(), w)        # [M, N/g]
        M = P.shape[0]
        G = torch.empty(t, M, ng, device=P.device, dtype=F32)
        dist.all_gather_into_tensor(G, P, group=self.groups[t])
        full = G.view(C, g, M, ng).permute(0, 2, 1, 3).reshape(C, M, self.N)
        return self._tree_sum(full).to(BF)

    def lt_splitk(self, t, x, C):
        """Logical TP C on t > C ranks by splitting each chunk's K range over g = t/C ranks and adding
        the g fp32 partials in rank order. This changes the chunk partials: shown, not a fix."""
        g = t // C
        c, h = divmod(self.rank, g)
        kc = self.K // C
        kg = kc // g
        lo = c * kc + h * kg
        w = self._sub_weight(("splitk", C, t), slice(lo, lo + kg), slice(None))
        P = mm_one_config_f32(x[:, lo:lo + kg].contiguous(), w)                  # [M, N]
        G = torch.empty(t, *P.shape, device=P.device, dtype=F32)
        dist.all_gather_into_tensor(G, P, group=self.groups[t])
        G = G.view(C, g, *P.shape)
        chunks = G[:, 0].clone()
        for j in range(1, g):
            chunks.add_(G[:, j])
        return self._tree_sum(chunks).to(BF)

    # full-x entry points (each rank shards the replicated input itself)
    def fn(self, scheme, t):
        if scheme.startswith("lt"):                         # e.g. lt8_tree, lt16_tree, lt8_2d
            head, form = scheme.split("_", 1)
            C = int(head[2:])
            if form in ("2d", "splitk"):
                return lambda x: getattr(self, "lt_" + form)(t, x, C)
            return lambda x: getattr(self, "lt_" + form)(t, self.chunk_inputs(t, x, C), C)
        return {"unsharded": lambda x: mm_one_config(x, self.w_row[1]),
                "row_parallel": lambda x: self.row(t, self.shard_k(t, x)),
                "row_fallback": lambda x: self.row_fallback(t, self.shard_k(t, x)),
                "row_fbhier": lambda x: self.row_fbhier(t, self.shard_k(t, x)),
                "row_sp": lambda x: self.row_sp(t, self.shard_k(t, x)),
                "col_parallel": lambda x: self.col(t, x)}[scheme]


def tp_plan(world, xnode):
    """What the tp part runs: the one-node plan (8 ranks, logical TP 8), or the two-node plan
    (16 ranks, logical TP 16, plus the 8-chunk scheme stretched to TP 16 two ways)."""
    degrees = [t for t in (2, 4, 8, 16) if t <= world]
    if not xnode:
        return {"degrees": degrees, "chunks": (8,), "fillers": FILLERS, "l5_m": (1, 256, 4096),
                "l2": [("unsharded", 1), ("row_parallel", 2), ("row_parallel", 4), ("row_parallel", 8),
                       ("row_fallback", 4), ("row_fallback", 8), ("row_sp", 4), ("row_sp", 8), ("col_parallel", 8),
                       ("lt8_gather", 1), ("lt8_gather", 2), ("lt8_gather", 8), ("lt8_a2a", 8),
                       ("lt8_tree", 1), ("lt8_tree", 2), ("lt8_tree", 4), ("lt8_tree", 8)],
                "l1": [("row_parallel", 4), ("row_parallel", 8), ("lt8_gather", 8), ("lt8_a2a", 8), ("lt8_tree", 8)],
                "l5": [("unsharded", (1,)), ("row_parallel", degrees), ("row_fallback", degrees),
                       ("col_parallel", degrees), ("lt8_gather", [1] + degrees), ("lt8_a2a", degrees),
                       ("lt8_tree", [1] + degrees)],
                "timing": [("unsharded", (1,)), ("row_parallel", degrees), ("row_fallback", degrees),
                           ("col_parallel", (8,)), ("lt8_gather", [1] + degrees), ("lt8_a2a", degrees),
                           ("lt8_tree", [1] + degrees)],
                "coll_bytes": (32 << 10, 1 << 20, 16 << 20, 256 << 20), "coll_dtypes": tuple(DT)}
    return {"degrees": degrees, "chunks": (8, 16), "fillers": ("random",), "l5_m": (1, 256, 4096),
            "l2": [("unsharded", 1), ("row_parallel", 8), ("row_parallel", 16), ("row_fallback", 16),
                   ("row_fbhier", 16), ("row_sp", 16), ("col_parallel", 16), ("lt16_tree", 1), ("lt16_tree", 8),
                   ("lt16_tree", 16), ("lt16_treeh", 16)],
            "l1": [("row_parallel", 16), ("lt16_tree", 16), ("lt16_treeh", 16)],
            "l5": [("unsharded", (1,)), ("row_parallel", degrees), ("row_fallback", (16,)), ("row_fbhier", (16,)),
                   ("col_parallel", (16,)), ("lt16_tree", [1] + degrees), ("lt16_treeh", (16,)),
                   ("lt8_tree", (1, 8)), ("lt8_2d", (16,)), ("lt8_splitk", (16,))],
            "timing": [("unsharded", (1,)), ("row_parallel", (8, 16)), ("row_fallback", (16,)), ("row_fbhier", (16,)),
                       ("col_parallel", (16,)), ("lt16_tree", (1, 8, 16)), ("lt16_treeh", (16,)), ("lt8_tree", (8,)),
                       ("lt8_2d", (16,))],
            "coll_bytes": (1 << 20, 16 << 20, 64 << 20), "coll_dtypes": ("bf16",)}


def _l5_reference(key):
    scheme = key.split("/")[0]
    if scheme.startswith("lt"):
        head, form = scheme.split("_", 1)
        return f"{head}_gather/tp1" if form in ("gather", "a2a") else f"{head}_tree/tp1"
    return "unsharded/tp1"


def tp_part(rank, world, args):
    dev = torch.device("cuda")
    K = N = args.k
    plan = tp_plan(world, args.xnode)
    gcpu = torch.Generator().manual_seed(7)
    W = (torch.randn(K, N, generator=gcpu) / K ** 0.5).to(BF).to(dev)
    probe = torch.randn(2, K, generator=gcpu).to(BF).to(dev)
    X = torch.randn(4096, K, generator=gcpu).to(BF).to(dev)
    degrees = plan["degrees"]
    groups = {1: None}
    for t in degrees:
        groups[t] = dist.group.WORLD if t == world else dist.new_group(list(range(t)))
    tp = TP(W, rank, groups, plan["chunks"], hier_groups(rank, world))
    res = {"K": K, "N": N, "chunks": list(plan["chunks"]), "plan": {k: v for k, v in plan.items()}}

    # 1. the fp32-output variant does the same arithmetic as mm_one_config; splitting K does not
    if rank == 0:
        res["f32_variant_rounds_to_mm_one_config"] = {
            str(M): bit_equal(mm_one_config_f32(X[:M], W).to(BF), mm_one_config(X[:M], W)) for M in (1, 65, 257, 4096)}
        kc = K // 8
        whole = mm_one_config_f32(X[:, :kc].contiguous(), W[:kc].contiguous())
        halves = (mm_one_config_f32(X[:, :kc // 2].contiguous(), W[:kc // 2].contiguous())
                  + mm_one_config_f32(X[:, kc // 2:kc].contiguous(), W[kc // 2:kc].contiguous()))
        cols = mm_one_config_f32(X[:, :kc].contiguous(), W[:kc, :N // 2].contiguous())
        res["chunk0_of_8_M4096"] = {
            "split_K_in_two_mismatched": mismatched(whole, halves), "of": whole.numel(),
            "split_N_in_two_equal": bit_equal(cols, whole[:, :N // 2].contiguous())}
        print("f32 variant:", res["f32_variant_rounds_to_mm_one_config"], "chunk split:", res["chunk0_of_8_M4096"],
              flush=True)

    # 2. L2
    sizes = M_SIZES[:12] if args.quick else M_SIZES
    res["l2"] = {}
    for scheme, t in plan["l2"]:
        if t > world:
            continue
        if rank < t:
            t0 = time.time()
            rep = check_batch(tp.fn(scheme, t), probe, sizes, fillers=plan["fillers"], name=f"{scheme} tp={t}")
            if rank == 0:
                res["l2"][f"{scheme}/tp{t}"] = summarize(rep)
                print(f"L2 {rep.summary()}  ({time.time() - t0:.0f} s)", flush=True)
        dist.barrier()

    # 3. L1
    res["l1"] = {}
    for scheme, t in plan["l1"]:
        if t > world:
            continue
        if rank < t:
            rep = check_repeat(tp.fn(scheme, t), X, n=5, name=f"{scheme} tp={t} M=4096")
            if rank == 0:
                res["l1"][rep.name] = rep.to_dict()
                print("L1", rep.summary(), flush=True)
        dist.barrier()

    # 4. L5 and accuracy
    res["l5"], res["accuracy"] = {}, {}
    for M in ((1, 256) if args.quick else plan["l5_m"]):
        x = X[:M]
        outs = {}
        for scheme, ts in plan["l5"]:
            for t in ts:
                if t > world:
                    continue
                if rank < t:
                    y = tp.fn(scheme, t)(x)
                    if rank == 0:
                        outs[f"{scheme}/tp{t}"] = y
                dist.barrier()
        if rank == 0:
            base = outs["unsharded/tp1"]
            table = {}
            for k, y in outs.items():
                vs = _l5_reference(k)
                ref = outs.get(vs, base)
                table[k] = {"vs": vs if vs in outs else "unsharded/tp1",
                            "mismatched": mismatched(ref, y), "of": y.numel(),
                            "max_abs_diff": max_abs_diff(ref, y),
                            "vs_unsharded_mismatched": mismatched(base, y)}
            res["l5"][str(M)] = table
            print(f"L5 M={M}: " + ", ".join(f"{k} {v['mismatched']}/{v['of']} (vs unsharded {v['vs_unsharded_mismatched']})"
                                            for k, v in table.items()), flush=True)
            if M == 256:
                y64 = x.double() @ W.double()
                for k, y in outs.items():
                    d = y.double() - y64
                    res["accuracy"][k] = {"rel_fro": float(d.norm() / y64.norm()), "max_abs": float(d.abs().max())}
        del outs
        torch.cuda.empty_cache()

    # 5. timing
    res["timing"] = {} if args.no_timing else timing(rank, world, tp, X, plan, args)
    return res


def timing(rank, world, tp, X, plan, args):
    dev = torch.device("cuda")
    quick, xnode = args.quick, args.xnode
    out = {"collectives": {}, "tp": {}}
    # collectives at the full world size: all-reduce, reduce-scatter, and the two invariant fallbacks
    for dname in plan["coll_dtypes"]:
        dtype = DT[dname]
        for nbytes in (plan["coll_bytes"][:2] if quick else plan["coll_bytes"]):
            n = nbytes // torch.tensor([], dtype=dtype).element_size()
            m = torch.randn(n, device=dev).to(dtype)
            it = 20 if nbytes <= (1 << 20) else (3 if xnode else 5)
            row = {}
            row["allreduce"] = timed(lambda: all_reduce(m.clone()), None, it)
            row["reduce_scatter"] = timed(lambda: reduce_scatter(m, world), None, it)
            row["fallback_gather_sum"] = timed(lambda: fallback_gather(m, world), None, it)
            row["fallback_a2a_sum_gather"] = timed(lambda: fallback_a2a(m, world), None, it)
            if tp.hg:
                row["fallback_hier"] = timed(lambda: fallback_hier(m, tp.hg), None, it)
            row["clone_only"] = timed(lambda: m.clone(), None, it)
            if rank == 0:
                out["collectives"][f"{dname}/{nbytes}"] = {
                    "dtype": dname, "bytes": nbytes, "world": world,
                    "ms": {k: round(v, 4) for k, v in row.items()},
                    "algbw_GBps": {k: round(nbytes / v / 1e6, 3) for k, v in row.items() if k != "clone_only"}}
                print(f"timing {dname} {nbytes} B: " + ", ".join(f"{k} {v:.3f} ms" for k, v in row.items()), flush=True)
            del m
            torch.cuda.empty_cache()
    # TP layers on pre-sharded inputs (GEMM + communication)
    for M in ((16, 512) if quick else (16, 512, 4096)):
        x = X[:M].contiguous()
        it = 10 if M <= 512 else (3 if xnode else 5)
        for scheme, ts in plan["timing"]:
            for t in ts:
                if t > world:
                    continue
                if rank < t:
                    if scheme == "unsharded" or (scheme == "row_parallel" and t == 1):
                        f = lambda: mm_one_config(x, tp.w_row[1])                     # noqa: E731
                    elif scheme in ("row_parallel", "row_fallback", "row_fbhier"):
                        xs = tp.shard_k(t, x)
                        meth = {"row_parallel": tp.row, "row_fallback": tp.row_fallback, "row_fbhier": tp.row_fbhier}[scheme]
                        f = lambda: meth(t, xs)                                       # noqa: E731
                    elif scheme == "col_parallel":
                        f = lambda: tp.col(t, x)                                      # noqa: E731
                    else:
                        head, form = scheme.split("_", 1)
                        C = int(head[2:])
                        if form in ("2d", "splitk"):
                            f = lambda: getattr(tp, "lt_" + form)(t, x, C)            # noqa: E731
                        else:
                            xcs = tp.chunk_inputs(t, x, C)
                            f = lambda: getattr(tp, "lt_" + form)(t, xcs, C)          # noqa: E731
                    ms = timed(f, False if t == 1 else tp.groups[t], it)
                    if rank == 0:
                        out["tp"][f"M{M}/{scheme}/tp{t}"] = round(ms, 4)
                        print(f"timing M={M} {scheme} tp={t}: {ms:.3f} ms", flush=True)
                dist.barrier()
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- merge
ALGO = {0: "Tree", 1: "Ring", 2: "CollNetDirect", 3: "CollNetChain", 4: "NVLS", 5: "NVLSTree"}
PROTO = {0: "LL", 1: "LL128", 2: "Simple"}
NCCL_DT = {7: "fp32", 9: "bf16"}
# Send/Recv (all-to-all) take no algorithm choice, so only collectives are queued.
RE_COLL = re.compile(r"\[0\] NCCL INFO (AllReduce|ReduceScatter|AllGather|Broadcast|Reduce): "
                     r"opCount \w+ .*? count (\d+) datatype (\d+)")
RE_ALGO = re.compile(r"\[0\] NCCL INFO (\d+) Bytes -> Algo (\d+) proto (\d+) time ([\d.]+)")


def parse_nccl_log(d, tag):
    """Rank 0's NCCL log -> the (algorithm, protocol) NCCL chose per collective, dtype and size."""
    files = [f for f in glob.glob(os.path.join(d, f"{tag}-coll.nccl.*.log"))
             if " [0] NCCL INFO" in open(f, errors="replace").read(1 << 20)]
    if not files:
        return {"error": "no rank-0 NCCL log"}
    lines = open(files[0], errors="replace").read().splitlines()
    pending, choices, info = [], {}, []
    for ln in lines:
        m = RE_COLL.search(ln)
        if m:
            pending.append([m.group(1), int(m.group(2)), int(m.group(3))])
            continue
        m = RE_ALGO.search(ln)
        if m and pending:
            coll, count, dt = pending.pop(0)
            key = f"{coll}/{NCCL_DT.get(dt, dt)}/{count}"
            choices.setdefault(key, set()).add(f"{ALGO.get(int(m.group(2)), m.group(2))}/{PROTO.get(int(m.group(3)), m.group(3))}")
            continue
        if any(s in ln for s in ("Channel 00/", "Channel 01/", "Trees [", "set by environment", "via P2P", "via SHM",
                                 "NCCL version", "nChannels")) and "[0] NCCL INFO" in ln:
            info.append(ln.split("NCCL INFO", 1)[1].strip())
    return {"choices": {k: sorted(v) for k, v in choices.items()}, "init": sorted(set(info)),
            "unmatched_ops": len(pending)}


def _compact_json(obj):
    """Indented JSON with lists of scalars kept on one line."""
    txt = json.dumps(obj, indent=1, default=list)
    return re.sub(r"\[\s*([^\[\]{}]*?)\s*\]", lambda m: "[" + re.sub(r"\s*\n\s*", " ", m.group(1)) + "]", txt)


def merge(d, out_json, compact=0):
    runs, summ, fb_hashes = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(d, "*-coll.json")) + glob.glob(os.path.join(d, "*-tp.json"))):
        r = json.load(open(f))
        key = f"{r['tag']}/{r['part']}"
        if r["part"] == "tp":
            runs[key] = r
            continue
        log = parse_nccl_log(d, r["tag"])
        ch = log.get("choices", {})
        res = r["result"]
        s = {}
        for dname in DT:
            for sw in SWEEPS:
                es = [e for e in res["allreduce"] if e["dtype"] == dname and e["sweep"] == sw]
                fs = [e for e in res["fallback"] if e["dtype"] == dname and e["sweep"] == sw]
                s[f"allreduce/{dname}/{sw}"] = {
                    "calls": len(es), "failing": sum(e["mismatched"] > 0 for e in es),
                    "distinct_results": len({e["hash"] for e in es}),
                    "max_mismatched": max(e["mismatched"] for e in es), "of": es[0]["of"],
                    "max_abs_diff": max(e["max_abs_diff"] for e in es),
                    "ranks_disagree": sum(e["ranks_disagree"] for e in es),
                    "unexplained_by_ring_orders": sum(e["cover"]["unexplained"] for e in es),
                    "bytes": [es[0]["bytes"], es[-1]["bytes"]],
                    "nccl_choices": sorted({c for e in es for c in ch.get(f"AllReduce/{dname}/{e['n']}", [])})}
                s[f"fallback/{dname}/{sw}"] = {
                    "calls": len(fs), "failing": sum(e["mismatched"] > 0 for e in fs),
                    "distinct_results": len({e["hash"] for e in fs}),
                    "gather_equals_a2a": all(e["gather_equals_a2a"] is not False for e in fs),
                    "gather_checked": sum(e["gather_equals_a2a"] is not None for e in fs),
                    "equals_fp32_rank_order_sum": all(e["matches_fp32_rank_order_or_fwd0"] for e in fs)}
                if any("hier_hash" in e for e in fs):
                    s[f"fallback/{dname}/{sw}"].update({
                        "hier_failing": sum(e["hier_mismatched"] > 0 for e in fs),
                        "hier_distinct_results": len({e["hier_hash"] for e in fs})})
                for e in fs:
                    fb_hashes.setdefault(f"{dname}/{sw}", set()).add(e["hash"])
                    if "hier_hash" in e:
                        fb_hashes.setdefault(f"{dname}/{sw}/hier", set()).add(e["hier_hash"])
            rs = [e for e in res["reduce_scatter"] if e["dtype"] == dname]
            frs = [e for e in res["fallback_rs"] if e["dtype"] == dname]
            s[f"reduce_scatter/{dname}"] = {
                "calls": len(rs), "failing": sum(sum(e["mismatched_per_owner"]) > 0 for e in rs),
                "max_mismatched": max(max(e["mismatched_per_owner"]) for e in rs),
                "owners_differ_from_owner0": max(sum(x > 0 for x in e["owner_vs_owner0"]) for e in rs),
                "unexplained_by_ring_orders": sum(c["unexplained"] for e in rs for c in e["cover_per_owner"]),
                "nccl_choices": sorted({c for e in rs for c in ch.get(f"ReduceScatter/{dname}/{e['rows_per_rank'] * ROW}", [])})}
            s[f"fallback_rs/{dname}"] = {
                "calls": len(frs), "failing": sum(sum(e["mismatched_per_owner"]) > 0 for e in frs),
                "owners_differ_from_owner0": max(sum(x > 0 for x in e["owner_vs_owner0"]) for e in frs)}
        s["repeat"] = {k: {"passed": v["passed"], "checked": v["checked"], "failures_all_ranks": v.get("failures_all_ranks")}
                       for k, v in res["repeat"].items()}
        summ[key] = s
        # per-call rows, slimmed: the order diagnostic keeps the 3 largest orders
        cols_ar = ["dtype", "sweep", "n", "bytes", "pos", "mismatched", "of", "max_abs_diff", "ranks_disagree",
                   "nccl", "n_orders", "top_orders", "unexplained"]
        rows_ar = [[e["dtype"], e["sweep"], e["n"], e["bytes"], e["pos"], e["mismatched"], e["of"], e["max_abs_diff"],
                    e["ranks_disagree"], ch.get(f"AllReduce/{e['dtype']}/{e['n']}"), e["cover"]["n_orders"],
                    [o[0] for o in e["cover"]["orders"][:3]], e["cover"]["unexplained"]] for e in res["allreduce"]]
        cols_rs = ["dtype", "rows_per_rank", "bytes_total", "pos", "mismatched_per_owner", "owner_vs_owner0",
                   "order_per_owner", "nccl"]
        rows_rs = [[e["dtype"], e["rows_per_rank"], e["bytes_total"], e["pos"], e["mismatched_per_owner"],
                    e["owner_vs_owner0"], [c["orders"][0][0] if c["n_orders"] == 1 else f"{c['n_orders']} orders"
                                           for c in e["cover_per_owner"]],
                    ch.get(f"ReduceScatter/{e['dtype']}/{e['rows_per_rank'] * ROW}")] for e in res["reduce_scatter"]]
        runs[key] = {k: r[k] for k in ("tag", "part", "world", "device", "torch", "triton", "nccl", "env", "seconds")}
        runs[key]["nccl_init"] = (log.get("init") or [])[:40]
        runs[key]["nccl_choices"] = ch
        if compact:
            # failing calls only (at most `compact` per run), a tally of which orders explained the
            # results, and the sizes at which NCCL changed algorithm or protocol
            tally = {}
            for row in rows_ar:
                rec = dict(zip(cols_ar, row))
                o = rec["top_orders"][0] if rec["n_orders"] == 1 else f"{rec['n_orders']} orders"
                k = f"{rec['dtype']}/{'+'.join(rec['nccl'] or ['?'])}/{o}"
                tally[k] = tally.get(k, 0) + 1
            fails = [row for row in rows_ar if row[cols_ar.index("mismatched")] > 0]
            runs[key]["allreduce_order_tally"] = tally
            runs[key]["allreduce_failing"] = {"columns": cols_ar, "rows": fails[:compact], "total": len(fails)}
            runs[key]["reduce_scatter"] = {"columns": cols_rs, "rows": rows_rs[-2:]}
            switch = {}
            for k, v in sorted(ch.items(), key=lambda kv: (kv[0].split("/")[0], kv[0].split("/")[1], int(kv[0].split("/")[2]))):
                coll, dt, count = k.split("/")
                lst = switch.setdefault(f"{coll}/{dt}", [])
                if not lst or lst[-1][1] != "+".join(v):
                    lst.append([int(count), "+".join(v)])
            runs[key]["nccl_choices"] = {"note": "[count where the choice starts, choice]", **switch}
            runs[key]["nccl_init"] = [ln for ln in (log.get("init") or []) if "/02 :" in ln or "Trees" in ln]
        else:
            runs[key]["allreduce"] = {"columns": cols_ar, "rows": rows_ar}
            runs[key]["reduce_scatter"] = {"columns": cols_rs, "rows": rows_rs}
    out = {"about": "collectives_tp.py on A16s; see handbook/3-kernels/collectives.md. Order names: fwdA = "
                    "sequential sum over ranks A, A+1, ... (mod world), bwdA = A, A-1, ..., rounding to the dtype "
                    "after every add; tree:... = two-node tree orders, C(a..b) = chain sum from rank a to b.",
           "summary": summ,
           "fallback_distinct_results_across_all_runs": {k: len(v) for k, v in fb_hashes.items()},
           "runs": runs}
    with open(out_json, "w") as f:
        f.write(_compact_json(out))
    for key, s in summ.items():
        print(f"== {key}")
        for k, v in s.items():
            print(f"  {k}: {v}")
    print("fallback distinct results across runs:", out["fallback_distinct_results_across_all_runs"])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("coll", "tp"))
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out", default="out")
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--xnode", action="store_true",
                    help="two-node preset: messages <= 64 MB, all-gather fallback <= 16 MB in the sweep, "
                         "logical TP 16 (and 8 stretched to TP 16), random fillers only in L2")
    ap.add_argument("--max-bytes", type=int, default=0, help="largest message in the collective sweeps")
    ap.add_argument("--gather-max-bytes", type=int, default=0, help="largest message for the all-gather fallback")
    ap.add_argument("--merge", help="directory of per-launch JSONs (and NCCL logs) to merge")
    ap.add_argument("--json", default="results-a16.json")
    ap.add_argument("--compact", type=int, default=0,
                    help="merge: keep at most this many failing all-reduce calls per run, plus tallies")
    args = ap.parse_args()
    if args.quick:                                # every sweep keeps all its sizes up to 4 MB
        args.max_bytes = args.max_bytes or (4 << 20)
        args.gather_max_bytes = args.gather_max_bytes or (4 << 20)
    if args.xnode:
        args.max_bytes = args.max_bytes or (64 << 20)
        args.gather_max_bytes = args.gather_max_bytes or (16 << 20)
    if args.merge:
        merge(args.merge, args.json, args.compact)
        return
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    t0 = time.time()
    result = coll_part(rank, world, args) if args.part == "coll" else tp_part(rank, world, args)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        meta = {"tag": args.tag, "part": args.part, "world": world, "device": torch.cuda.get_device_name(),
                "torch": torch.__version__, "triton": triton.__version__,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_") and k != "NCCL_DEBUG_FILE"},
                "seconds": round(time.time() - t0, 1), "result": result}
        with open(os.path.join(args.out, f"{args.tag}-{args.part}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"done {args.tag} {args.part} in {meta['seconds']} s", flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
