"""Gradients under data parallelism (DDP) and FSDP/ZeRO-style sharding, checked bitwise.

A small decoder LM (fp32, stock torch ops, strict deterministic mode) is trained data-parallel on
d GPUs. The script measures where the averaged gradient, the sharded gradient, the global
grad-norm and the final weights depend on things that should not matter (the bucket layout, the
owning rank, the degree d, the placement), and checks a scheme that removes each dependence.

Stock paths (documented variance, expect="fail" where the bits change):
  nccl_allreduce        DDP-style: each rank's gradient times 1/d, packed into messages, NCCL
                        all-reduce. Packings: one message per parameter; DDP buckets of several
                        caps and parameter orders; one flat message, with padding in front.
  torch_ddp             torch.nn.parallel.DistributedDataParallel itself, bucket_cap_mb varied.
  nccl_reduce_scatter   FSDP-style: the flat gradient, NCCL reduce-scatter, rank j keeps slice j.
  sharded_norm_stock    FSDP-style grad-norm: local sum of squares per rank, NCCL all-reduce.
  loss_mean_stock       per-rank batch mean, NCCL all-reduce, divided by d.
The scheme (expect="pass"):
  ordered_allreduce     all-to-all (rank r receives only slice r), an fp32 sum in rank order on the
                        owner, all-gather. No reduction happens in the network. Also as a DDP
                        comm hook (torch_ddp+ordered_hook).
  ordered_reduce_scatter  the same without the all-gather: the data volume of a reduce-scatter.
  logical_shards        S = 16 logical micro-batches. A rank computes its shards one at a time
                        (a backward per shard) and adds them in the fixed pairwise tree over the
                        shard index, as far as its shards form aligned subtrees. The remaining
                        tree is finished on the owner of each slice after an all-to-all. The tree
                        does not depend on d. 2level: the same tree routed node first, then one
                        exchange across the link (two nodes).
  logical_norm          sum of squares per fixed block of 4096 elements (a pairwise tree of
                        elementwise fp64 adds), block sums moved by all-gather, one pairwise tree
                        over the blocks. Slices are cut on block boundaries.
  e2e                   10 AdamW steps (torch.optim.AdamW, foreach=False, fused=False: the
                        single-tensor path) with logical shards, ordered reductions and clipping
                        by the logical norm, replicated (dp) or sharded (fsdp), then elastic
                        resume through a checkpoint file at another degree.

Driver (this file without --worker) launches one torch.distributed.run per NCCL setting (NCCL
reads NCCL_ALGO at init), merges the launches and writes one bi-results-v1 file:
  python training_dp.py --quick --json out.json     one node, up to 8 GPUs, about 3 minutes;
                                                    two-node checks are skipped and listed
  python training_dp.py --json results-a16.json     one node, full
  python training_dp.py --two-node --json results-a16-2node.json [--ref out/ref-default.pt]
        16 ranks on two nodes through a launcher (default ~/bi-check-xnode/launch2.sh, which
        takes the cluster lock itself); run on node 0, same tree at the same path on node 1.
Exit status is nonzero when a check with expect="pass" fails or a launch does not finish.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import copy  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
from bi_check import bit_equal, bits, first_mismatch  # noqa: E402

S = 16                          # logical shards (micro-batches) of the global batch
BLK = 4096                      # logical block of the flat gradient, in elements (a power of two)
STEPS = 10
OPT = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
MAX_NORM = 0.5                  # clipping threshold on the global grad-norm
CFGS = {                        # vocab, width, layers, heads, context
    "small": (256, 256, 4, 4, 128),
    "medium": (256, 512, 6, 8, 256),
}
SETTINGS = {"default": {}, "ring": {"NCCL_ALGO": "Ring"}, "tree": {"NCCL_ALGO": "Tree"}}
MIB = 1 << 20


# ----------------------------------------------------------------------------- model and data
class Block(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.ln1, self.qkv, self.proj = nn.LayerNorm(D), nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.ln2, self.fc, self.out = nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class LM(nn.Module):
    """Pre-LN decoder LM with tied input and output embeddings. forward(tokens [B, T+1]) returns the
    mean next-token cross-entropy over the B * T predictions."""

    def __init__(self, V, D, L, H, T):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.ModuleList([Block(D, H) for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.wte.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.wte(idx[:, :-1]) + self.wpe(torch.arange(idx.shape[1] - 1, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))


def make_tokens(n, T, V, seed):
    """n sequences of T+1 tokens from a fixed random 4-successor chain. Integer-only, on the CPU."""
    g = torch.Generator().manual_seed(seed)
    succ = torch.randint(0, V, (V, 4), generator=g)
    x = torch.empty(n, T + 1, dtype=torch.long)
    x[:, 0] = torch.randint(0, V, (n,), generator=g)
    pick = torch.randint(0, 4, (n, T), generator=g)
    for t in range(T):
        x[:, t + 1] = succ[x[:, t], pick[:, t]]
    return x


class Flat:
    """Canonical flat layout of a model's parameters: named_parameters() order, no gaps, zero-padded
    to a whole number of blocks that is a multiple of 16. Slices for d ranks are cut on block
    boundaries; for d in 1, 2, 4, 8, 16 they are equal."""

    def __init__(self, model, blk=BLK, mult=16):
        self.items = [(n, p.numel(), tuple(p.shape)) for n, p in model.named_parameters()]
        self.off, o = {}, 0
        for n, k, _ in self.items:
            self.off[n] = o
            o += k
        self.P, self.blk = o, blk
        nb = -(-o // blk)                              # blocks, rounded up
        self.NB = -(-nb // mult) * mult
        self.n = self.NB * blk

    def _cat(self, ts):
        return torch.cat(ts + [ts[0].new_zeros(self.n - self.P)])

    def grads(self, model):
        return self._cat([p.grad.reshape(-1) for p in model.parameters()])

    def params(self, model):
        return self._cat([p.detach().reshape(-1) for p in model.parameters()])

    def load(self, model, vec):
        with torch.no_grad():
            for (n, k, shp), p in zip(self.items, model.parameters()):
                p.copy_(vec[self.off[n]:self.off[n] + k].view(shp))

    def set_grads(self, model, vec):
        for (n, k, shp), p in zip(self.items, model.parameters()):
            p.grad = vec[self.off[n]:self.off[n] + k].view(shp)

    def bounds(self, d):
        return [(self.NB * j // d) * self.blk for j in range(d + 1)]

    def sizes(self, d):
        b = self.bounds(d)
        return [b[j + 1] - b[j] for j in range(d)]


def kind_of(name):
    if name.startswith("wte"):
        return "wte+head"
    if name.startswith("wpe"):
        return "wpe"
    mod, leaf = name.split(".")[-2:]
    return ("ln." if mod.startswith("ln") else mod + ".") + leaf


def batch_grad(model, flat, x):
    """One forward and backward over x: the flat gradient of the mean loss, and the loss."""
    model.zero_grad(set_to_none=True)
    loss = model(x)
    loss.backward()
    return flat.grads(model), loss.detach()


# ----------------------------------------------------------------------------- the fixed tree
def aligned_blocks(shards):
    """Maximal aligned blocks (level, index) of the pairwise tree inside a sorted list of shard ids.
    Block (l, i) covers shards [i << l, (i + 1) << l)."""
    st = []
    for s in shards:
        st.append((0, s))
        while len(st) > 1 and st[-1][0] == st[-2][0] and st[-2][1] % 2 == 0 and st[-1][1] == st[-2][1] + 1:
            st[-2:] = [(st[-2][0] + 1, st[-2][1] // 2)]
    return st


class TreeAcc:
    """Adds leaves pushed in ascending shard order as far as they form aligned subtrees."""

    def __init__(self):
        self.keys, self.vals = [], []

    def push(self, s, t):
        self.keys.append((0, s))
        self.vals.append(t)
        k = self.keys
        while len(k) > 1 and k[-1][0] == k[-2][0] and k[-2][1] % 2 == 0 and k[-1][1] == k[-2][1] + 1:
            r = self.vals.pop()
            self.vals[-1].add_(r)                      # left + right, fp32
            k[-2:] = [(k[-2][0] + 1, k[-2][1] // 2)]


def tree_finish(nodes):
    """Finish the pairwise tree from nodes {(level, index): tensor} that tile the leaves."""
    nodes = dict(nodes)
    while len(nodes) > 1:
        lv = min(k[0] for k in nodes)
        for i in sorted(k[1] for k in nodes if k[0] == lv):
            if i % 2 == 0:
                nodes[(lv + 1, i // 2)] = nodes.pop((lv, i)) + nodes.pop((lv, i + 1))
    return next(iter(nodes.values()))


def pairwise(x):
    """Pairwise tree over dim 0 (a power of two): (x0 + x1) + (x2 + x3), ..."""
    while x.shape[0] > 1:
        x = x[0::2] + x[1::2]
    return x[0]


def tree_sum(v):
    """Pairwise tree over a vector, zero-padded to a power of two (adding +0.0 is exact here)."""
    n = 1 << max(0, (v.numel() - 1).bit_length())
    if n != v.numel():
        v = torch.cat([v, v.new_zeros(n - v.numel())])
    while v.numel() > 1:
        v = v[0::2] + v[1::2]
    return v[0]


def block_sumsq(x, blk):
    """Sum of squares of every block of blk elements in fp64, by a pairwise tree of elementwise adds.
    A block's result depends on that block's values only, not on how many blocks there are."""
    y = x.view(-1, blk).double()
    y = y * y
    while y.shape[1] > 1:
        y = y[:, 0::2] + y[:, 1::2]
    return y[:, 0]


def logical_norm(g, blk=BLK):
    return tree_sum(block_sumsq(g, blk)).sqrt()


def shard_assign(d, strided=False):
    if strided:
        return [list(range(m, S, d)) for m in range(d)]
    return [list(range(m * S // d, (m + 1) * S // d)) for m in range(d)]


# ----------------------------------------------------------------------------- groups and collectives
class Grp:
    """A set of ranks acting as one data-parallel job. me: this rank's index in it (None when not a
    member). hier: for 2 nodes x L ranks, this rank's node group and its pair across the link."""

    def __init__(self, name, members, group, me, hier):
        self.name, self.members, self.group, self.me, self.hier = name, members, group, me, hier
        self.d = len(members)


def gather_slices(mine, sizes, grp):
    """All-gather of per-rank slices of given sizes (a copy), concatenated in rank order."""
    if grp.d == 1:
        return mine
    mx = max(sizes)
    buf = mine if mine.numel() == mx else torch.cat([mine, mine.new_zeros(mx - mine.numel())])
    out = torch.empty(grp.d, mx, device=mine.device, dtype=mine.dtype)
    dist.all_gather_into_tensor(out, buf.contiguous(), group=grp.group)
    if min(sizes) == mx:
        return out.view(-1)
    return torch.cat([out[j, :sizes[j]] for j in range(grp.d)])


def a2a_exchange(parts, sizes, counts, grp):
    """parts [n_me, n]: this rank's vectors. Every rank j receives slice j of every vector of every
    rank; nothing is reduced on the wire. Returns, per source rank s, a [counts[s], sizes[me]] view."""
    me, d = grp.me, grp.d
    if d == 1:
        return [parts]
    if parts.shape[0] == 1:
        send = parts.reshape(-1)
    else:
        b = [0]
        for s in sizes:
            b.append(b[-1] + s)
        send = torch.cat([parts[:, b[j]:b[j + 1]].reshape(-1) for j in range(d)])
    recv = torch.empty(sum(counts) * sizes[me], device=parts.device, dtype=parts.dtype)
    dist.all_to_all_single(recv, send, output_split_sizes=[c * sizes[me] for c in counts],
                           input_split_sizes=[counts[me] * s for s in sizes], group=grp.group)
    out, o = [], 0
    for s in range(d):
        out.append(recv[o:o + counts[s] * sizes[me]].view(counts[s], sizes[me]))
        o += counts[s] * sizes[me]
    return out


def ordered_reduce(x, grp, order="rank", scatter=False):
    """Sum of one 1-D message over the group's ranks with the order fixed by the caller: all-to-all,
    then on the owner of each slice an fp32 sum in rank order (or the pairwise tree over ranks, d a
    power of two). scatter=True returns this rank's slice (an ordered reduce-scatter; the message
    length must be a multiple of d); otherwise the slices are all-gathered (an ordered all-reduce)."""
    n, d = x.numel(), grp.d
    if d == 1:
        return x.clone()
    pad = (-n) % d
    xp = torch.cat([x, x.new_zeros(pad)]) if pad else x
    sz = xp.numel() // d
    pieces = a2a_exchange(xp[None], [sz] * d, [1] * d, grp)
    if order == "rank":
        acc = pieces[0][0].clone()
        for s in range(1, d):
            acc.add_(pieces[s][0])
    else:
        acc = pairwise(torch.stack([p[0] for p in pieces]))
    if scatter:
        return acc
    return gather_slices(acc, [sz] * d, grp)[:n]


def hier_reduce(part, grp, scatter=False):
    """The pairwise tree over the ranks of a 2 x L group, routed in two levels: inside each node an
    all-to-all and the node's subtree; across the link one fp32 exchange with the rank of the same
    local index; then node 0's subtree + node 1's. Same tree as the flat route, less link traffic."""
    h, d, me = grp.hier, grp.d, grp.me
    L = h["L"]
    node, n = me // L, part.numel()
    if scatter:                                           # rank (node, i) owns slice node * L + i
        sz = n // d
        send = part.view(2, L, sz).permute(1, 0, 2).contiguous()        # [dest local index, owner node, sz]
        recv = send
        if L > 1:
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=h["node"])          # [source local index, owner node, sz]
        sums = pairwise(recv)                                            # this node's subtree: mine and my partner's slice
        both = torch.empty(2, sz, device=part.device, dtype=part.dtype)
        dist.all_gather_into_tensor(both, sums[1 - node].contiguous(), group=h["pair"])
        return sums[node] + both[1 - node] if node == 0 else both[1 - node] + sums[node]
    sz = n // L
    recv = part.view(L, sz)
    if L > 1:
        recv = torch.empty(L, sz, device=part.device, dtype=part.dtype)
        dist.all_to_all_single(recv, part.view(L, sz), group=h["node"])
    sub = pairwise(recv).contiguous()                                    # this node's subtree, slice i of L
    both = torch.empty(2, sz, device=part.device, dtype=part.dtype)
    dist.all_gather_into_tensor(both, sub, group=h["pair"])
    tot = both[0] + both[1]
    if L == 1:
        return tot
    out = torch.empty(L, sz, device=part.device, dtype=part.dtype)
    dist.all_gather_into_tensor(out, tot, group=h["node"])
    return out.view(-1)


def nccl_allreduce(x, grp):
    y = x.clone()
    if grp.d > 1:
        dist.all_reduce(y, group=grp.group)
    return y


def nccl_reduce_scatter(x, grp):
    if grp.d == 1:
        return x.clone()
    out = torch.empty(x.numel() // grp.d, device=x.device, dtype=x.dtype)
    dist.reduce_scatter_tensor(out, x, group=grp.group)
    return out


# ----------------------------------------------------------------------------- packings of the gradient
class Packing:
    """How the canonical gradient is laid out in messages: a list of messages, each a list of
    (parameter name or None for zero padding, elements). roll: rotate a message by that many slices
    before a reduce-scatter (every element moves to the next owner)."""

    def __init__(self, name, msgs, roll=0):
        self.name, self.msgs, self.roll = name, msgs, roll


def ordered(plist, how, seed=7):
    if how == "rev":
        return plist[::-1]
    if how == "size":
        return sorted(plist, key=lambda nk: nk[1])
    if how == "shuf":
        out = list(plist)
        random.Random(seed).shuffle(out)
        return out
    return list(plist)


def buckets(plist, cap, first=None):
    """torch DDP's rule: a bucket is closed once it holds at least `cap` bytes (the first: `first`)."""
    msgs, cur, size, limit = [], [], 0, first or cap
    for n, k in plist:
        cur.append((n, k))
        size += 4 * k
        if size >= limit:
            msgs.append(cur)
            cur, size, limit = [], 0, cap
    return msgs + ([cur] if cur else [])


def by_layer(plist):
    groups = {}
    for n, k in plist:
        groups.setdefault(".".join(n.split(".")[:2]) if n.startswith("blocks.") else "rest", []).append((n, k))
    return list(groups.values())


def run_packing(pk, g, flat, op, d, pad_d=False):
    """Reduce every message of a packing of this rank's canonical gradient g with op (message -> the
    reduced message) and return the result in the canonical layout."""
    out = torch.empty(flat.P, device=g.device, dtype=g.dtype)
    for msg in pk.msgs:
        x = torch.cat([g[flat.off[n]:flat.off[n] + k] if n else g.new_zeros(k) for n, k in msg])
        n0 = x.numel()
        if pad_d and n0 % d:
            x = torch.cat([x, x.new_zeros((-n0) % d)])
        sh = pk.roll * (x.numel() // d)
        y = op(torch.roll(x, sh) if sh else x)
        if sh:
            y = torch.roll(y, -sh)
        o = 0
        for n, k in msg:
            if n:
                out[flat.off[n]:flat.off[n] + k] = y[o:o + k]
            o += k
    return out


def owner_map(pk, flat, d, dev):
    """Owning rank of every canonical element when each message of the packing is reduce-scattered."""
    own = torch.empty(flat.P, dtype=torch.int16, device=dev)
    for msg in pk.msgs:
        n = sum(k for _, k in msg)
        n += (-n) % d
        sz, o = n // d, 0
        for nm, k in msg:
            if nm:
                pos = (torch.arange(o, o + k, device=dev) + pk.roll * sz) % n
                own[flat.off[nm]:flat.off[nm] + k] = (pos // sz).to(torch.int16)
            o += k
    return own


# ----------------------------------------------------------------------------- comparisons and diagnostics
def sha(t):
    return hashlib.sha1(bits(t).cpu().numpy().tobytes()).hexdigest()[:16]


def diff(a, b):
    """None when bitwise equal (tools/bi_check.py), else how many elements differ and by how much."""
    if bit_equal(a, b):
        return None
    mm = first_mismatch(a, b)
    if "mismatched_elements" not in mm:
        return mm
    dd = (a.double() - b.double()).abs()
    return {"mismatched": mm["mismatched_elements"], "of": mm["of"], "max_abs_diff": float(dd.max()),
            "rel_fro": float(dd.norm() / b.double().norm().clamp_min(1e-300))}


def by_kind(a, b, flat):
    """Mismatched elements per parameter kind: {kind: [mismatched, elements, tensors that differ]}."""
    c = torch.cat([torch.zeros(1, dtype=torch.long, device=a.device),
                   (bits(a[:flat.P]) != bits(b[:flat.P])).cumsum(0)])
    out = {}
    for n, k, _ in flat.items:
        m = int(c[flat.off[n] + k] - c[flat.off[n]])
        e = out.setdefault(kind_of(n), [0, 0, 0])
        e[0] += m
        e[1] += k
        e[2] += int(m > 0)
    return out


def order_candidates(inp):
    """inp [d, n]: what each rank contributes. The sum over ranks in every ring order (start rank a,
    direction), in the two-node tree orders NCCL may use, and in the pairwise tree, fp32."""
    d = inp.shape[0]
    names, outs = [], []

    def chain(order):
        s = inp[order[0]]
        for k in order[1:]:
            s = s + inp[k]
        return s
    for step, tag in ((1, "fwd"), (-1, "bwd")):
        for a in range(d):
            names.append(f"{tag}{a}")
            outs.append(chain([(a + step * i) % d for i in range(d)]))
    if d == 16:
        c0, c1 = chain(range(7, -1, -1)), chain(range(15, 7, -1))
        s0, s1 = chain(range(7, 0, -1)), chain(range(15, 8, -1))
        for name, v in (("tree:C(7..0)+C(15..8)", c0 + c1), ("tree:(x0+C(15..8))+C(7..1)", (inp[0] + c1) + s0),
                        ("tree:(x8+C(7..0))+C(15..9)", (inp[8] + c0) + s1),
                        ("tree:(C(7..1)+C(15..8))+x0", (s0 + c1) + inp[0]),
                        ("tree:(C(15..9)+C(7..0))+x8", (s1 + c0) + inp[8])):
            names.append(name)
            outs.append(v)
    if d & (d - 1) == 0:
        names.append("pairwise")
        outs.append(pairwise(inp))
    return names, torch.stack(outs)


def order_cover(names, cands, res, top=4):
    """Greedy cover of the result's elements by candidate orders."""
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


def median(ts):
    ts = sorted(ts)
    return ts[len(ts) // 2]


# ----------------------------------------------------------------------------- worker
class Worker:
    def __init__(self, args):
        self.args, self.quick, self.xnode = args, args.quick, args.xnode
        self.local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local)
        self.dev = torch.device("cuda", self.local)
        dist.init_process_group("nccl", device_id=self.dev, timeout=datetime.timedelta(minutes=30))
        self.rank, self.W = dist.get_rank(), dist.get_world_size()
        assert self.W in (4, 8, 16), "run on 4 or 8 GPUs (one node) or 16 (two nodes)"
        torch.use_deterministic_algorithms(True, warn_only=False)
        self.votes, self.meas, self.extra, self.keep, self._setups, self._refs = [], [], {}, {}, {}, {}
        self.t0 = time.time()
        self.setup_groups()

    def log(self, *a):
        if self.rank == 0:
            print(f"[{self.args.tag} {time.time() - self.t0:6.1f}s]", *a, flush=True)

    # -- groups
    def setup_groups(self):
        W, q = self.W, self.quick
        if self.xnode:
            n0, n1 = list(range(8)), list(range(8, 16))
            pl = {"d1": [0], "d1@n1": [8], "d2": [0, 1], "d2x": [0, 8], "d4": n0[:4], "d4x": [0, 1, 8, 9],
                  "d8": n0, "d8@n1": n1, "d8x": n0[:4] + n1[:4], "d12x": n0[:6] + n1[:6], "d16": n0 + n1}
            if q:
                pl = {k: pl[k] for k in ("d1", "d1@n1", "d2x", "d4", "d8", "d8x", "d16")}
        else:
            pl = {f"d{d}": list(range(d)) for d in ((1, 2, 3, 4, 8) if q else range(1, 9)) if d <= W}
            if W == 8 and not q:
                pl.update({"d1@g5": [5], "d2x": [0, 4], "d4x": [0, 1, 4, 5]})
        made = {}

        def get(mem):
            key = tuple(mem)
            if key not in made:
                made[key] = dist.group.WORLD if len(key) == W else (None if len(key) == 1 else dist.new_group(list(key)))
            return made[key]
        self.G = {}
        for name, mem in pl.items():
            g, d, hier = get(mem), len(mem), None
            L = d // 2
            if self.xnode and d % 2 == 0 and d & (d - 1) == 0 and max(mem[:L]) < 8 <= min(mem[L:]):
                nodes = [get(mem[:L]), get(mem[L:])]
                pairs = [get([mem[i], mem[i + L]]) for i in range(L)]
                if self.rank in mem:
                    me = mem.index(self.rank)
                    hier = {"L": L, "node": nodes[me // L], "pair": pairs[me % L]}
            self.G[name] = Grp(name, mem, g, mem.index(self.rank) if self.rank in mem else None, hier)
        self.top = f"d{W}"
        self.pow2 = [n for n in ("d1", "d2", "d4", "d8", "d16") if n in self.G]      # ranks 0..d-1

    def mine(self, names=None):
        """The placements this rank is a member of, in a fixed order."""
        return [g for n, g in self.G.items() if (names is None or n in names) and g.me is not None]

    def routes(self, grp):
        return ("a2a", "2level") if grp.hier else ("a2a",)

    # -- records
    def vote(self, name, ok, *, level, impl, expect="pass", **info):
        self.votes.append({"name": name, "ok": bool(ok), "level": level, "impl": impl, "expect": expect,
                           "rank": self.rank, "info": info})

    def measure(self, name, value, unit, **meta):
        self.meas.append({"name": name, "value": float(f"{float(value):.5g}"), "unit": unit, **meta})

    def bcast(self, t):
        """Rank 0's tensor on every rank (a copy)."""
        dist.broadcast(t, 0)
        return t

    def from_leader(self, t, grp):
        """Member 0's copy of t on every member (a copy inside the group)."""
        if grp.d > 1:
            ref = t.clone()
            dist.broadcast(ref, grp.members[0], group=grp.group)
            return ref
        return t

    def timeit(self, fn, grp, n=7, warm=2):
        """Median wall time of fn in ms over n synchronized runs after warm-up; the slowest member."""
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        if grp.d > 1:
            dist.barrier(group=grp.group)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        t = torch.tensor([median(ts) * 1e3], device=self.dev, dtype=torch.float64)
        if grp.d > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=grp.group)
        return float(t)

    # -- models
    def setup(self, cfg):
        if cfg not in self._setups:
            V, D, L, H, T = CFGS[cfg]
            torch.manual_seed(1234)
            model = LM(V, D, L, H, T).to(self.dev)
            for p in model.parameters():                   # rank 0's initial weights everywhere
                self.bcast(p.data)
            m = argparse.Namespace(cfg=cfg, model=model, flat=Flat(model), T=T, V=V)
            m.theta0 = m.flat.params(model).clone()
            m.tokens = make_tokens(STEPS * 16 + 64, T, V, 4321).to(self.dev)
            self._setups[cfg] = m
        return self._setups[cfg]

    # -- logical shards
    def shard_grads(self, model, flat, X, shards):
        acc, losses = TreeAcc(), {}
        per = X.shape[0] // S
        for s in shards:
            g, loss = batch_grad(model, flat, X[s * per:(s + 1) * per])
            acc.push(s, g)
            losses[s] = loss
        return acc, losses

    def ls_reduce(self, grp, flat, acc, allk, route="a2a", scatter=False):
        if grp.d == 1:
            return tree_finish(dict(zip(acc.keys, acc.vals)))
        if route == "2level":
            return hier_reduce(acc.vals[0], grp, scatter)
        sizes = flat.sizes(grp.d)
        parts = torch.stack(acc.vals) if len(acc.vals) > 1 else acc.vals[0][None]
        pieces = a2a_exchange(parts, sizes, [len(k) for k in allk], grp)
        mine = tree_finish({k: pieces[m][i] for m in range(grp.d) for i, k in enumerate(allk[m])})
        return mine if scatter else gather_slices(mine, sizes, grp)

    def shard_mean(self, losses, assign, grp):
        """Mean of the S shard losses: values moved by all-gather, summed in the pairwise tree."""
        v = torch.zeros(S, device=self.dev)
        for s, t in losses.items():
            v[s] = t
        if grp.d > 1:
            allv = torch.empty(grp.d, S, device=self.dev)
            dist.all_gather_into_tensor(allv, v, group=grp.group)
            for m, sh in enumerate(assign):
                v[sh] = allv[m, sh]
        return tree_sum(v) * (1.0 / S)

    def ls_grad(self, M, X, grp, route="a2a", scatter=False, strided=False, model=None):
        """Logical shards on this group: the mean gradient over the S shards (the full canonical
        vector, or this rank's slice with scatter=True) and the mean loss."""
        assign = shard_assign(grp.d, strided)
        allk = [aligned_blocks(a) for a in assign]
        acc, losses = self.shard_grads(model or M.model, M.flat, X, assign[grp.me])
        g = self.ls_reduce(grp, M.flat, acc, allk, route, scatter)
        return g.mul_(1.0 / S), self.shard_mean(losses, assign, grp)

    def ls_ref(self, cfg, G):
        """The d = 1 result on rank 0, broadcast to every rank."""
        key = (cfg, G)
        if key not in self._refs:
            M = self.setup(cfg)
            if self.rank == 0:
                g, loss = self.ls_grad(M, M.tokens[:G], self.G["d1"])
            else:
                g, loss = torch.empty(M.flat.n, device=self.dev), torch.empty((), device=self.dev)
            self._refs[key] = (self.bcast(g), self.bcast(loss))
        return self._refs[key]

    def norm_sharded(self, gs, flat, grp):
        """The logical norm from this rank's block-aligned slice: block sums moved by all-gather."""
        sizes = [s // flat.blk for s in flat.sizes(grp.d)]
        return tree_sum(gather_slices(block_sumsq(gs, flat.blk), sizes, grp)).sqrt()

    # ------------------------------------------------------------------------- part: copy
    def part_copy(self):
        """All-gather, all-to-all and broadcast move raw bit patterns unchanged (NaN payloads, -0.0,
        subnormals included)."""
        def pattern(r, n):
            g = torch.Generator().manual_seed(977 * n + r)
            return torch.randint(-2 ** 31, 2 ** 31 - 1, (n,), generator=g, dtype=torch.int64).to(torch.int32).view(torch.float32)
        sizes = (255, 65536 + 16) if self.quick else (255, 4096 + 16, 65536 + 16, (1 << 20) + 16, (1 << 22) + 16)
        for grp in self.mine([self.top, "d3", "d4", "d8x"]):
            d, mem = grp.d, grp.members
            for n in sizes:
                n -= n % 16
                x = pattern(self.rank, n).to(self.dev)
                out = torch.empty(d, n, device=self.dev)
                dist.all_gather_into_tensor(out, x, group=grp.group)
                ok = all(bit_equal(out[i], pattern(mem[i], n).to(self.dev)) for i in range(d))
                self.vote(f"all-gather is a bit-exact copy ({grp.name})", ok, level="other", impl="all_gather", n=n)
                recv = torch.empty(n // 16 * d, device=self.dev)
                dist.all_to_all_single(recv, x[:n // 16 * d], group=grp.group)
                sz = n // 16
                ok = all(bit_equal(recv[i * sz:(i + 1) * sz], pattern(mem[i], n).to(self.dev)[grp.me * sz:(grp.me + 1) * sz])
                         for i in range(d))
                self.vote(f"all-to-all is a bit-exact copy ({grp.name})", ok, level="other", impl="all_to_all", n=n)
                y = x.clone()
                dist.broadcast(y, mem[0], group=grp.group)
                self.vote(f"broadcast is a bit-exact copy ({grp.name})", bit_equal(y, pattern(mem[0], n).to(self.dev)),
                          level="other", impl="broadcast", n=n)

    # ------------------------------------------------------------------------- part: pre
    def part_pre(self):
        """Precondition of logical shards: a shard's gradient has the same bits on every rank (and
        node), run after run."""
        for cfg in (("small",) if self.quick else ("small", "medium")):
            M = self.setup(cfg)
            for G in ((16,) if self.quick else (16, 64)):
                per = G // S
                for s in ((0, 11) if self.quick else (0, 5, 11, 15)):
                    x = M.tokens[s * per:(s + 1) * per]
                    g, loss = batch_grad(M.model, M.flat, x)
                    reps = [batch_grad(M.model, M.flat, x) for _ in range(2 if self.quick else 4)]
                    self.vote(f"a shard's gradient and loss repeat bit for bit on every rank ({cfg})",
                              all(bit_equal(g, r[0]) and bit_equal(loss, r[1]) for r in reps),
                              level="L1", impl="shard_backward", case=f"G{G}/shard{s}")
                    ref, lref = self.bcast(g.clone()), self.bcast(loss.clone())
                    dd = diff(g, ref)
                    self.vote(f"a shard's gradient and loss are the same bits on all {self.W} ranks ({cfg})",
                              dd is None and bit_equal(loss, lref), level="L5", impl="shard_backward",
                              case=f"G{G}/shard{s}", **(dd or {}))
        if not self.quick:
            # without strict deterministic mode torch's SDPA backward is not run-to-run stable here
            M = self.setup("medium")
            x = M.tokens[:1]
            torch.use_deterministic_algorithms(False)
            g = batch_grad(M.model, M.flat, x)[0]
            reps = [batch_grad(M.model, M.flat, x)[0] for _ in range(7)]
            torch.use_deterministic_algorithms(True, warn_only=False)
            bad = [dd["mismatched"] for dd in (diff(g, r) for r in reps) if dd]
            self.vote("a shard's gradient repeats without torch.use_deterministic_algorithms (medium, 8 runs)",
                      not bad, level="L1", impl="shard_backward_nondeterministic", expect="fail",
                      runs_differing=len(bad), max_mismatched=max(bad, default=0), of=M.flat.n)

    # ------------------------------------------------------------------------- part: ddp
    def local_scaled(self, M, X, grp):
        """What a DDP rank contributes: the gradient of its local batch times 1/d, and its loss."""
        G, d = X.shape[0], grp.d
        g, loss = batch_grad(M.model, M.flat, X[grp.me * G // d:(grp.me + 1) * G // d])
        return g[:M.flat.P] * (1.0 / d), loss

    def torch_ddp(self, M, grp, xl, cap, iters, hook=None):
        """torch DDP on a copy of the model: the gradient after each of `iters` identical iterations
        and the bucket layout each one used."""
        from torch.nn.parallel import DistributedDataParallel as DDP
        m = copy.deepcopy(M.model)
        names = {id(p): n for n, p in m.named_parameters()}
        ddp = DDP(m, device_ids=[self.local], process_group=grp.group, bucket_cap_mb=cap)
        if hook is not None:
            ddp.register_comm_hook(grp, hook)
        outs, lays = [], []
        for _ in range(iters):
            m.zero_grad(set_to_none=True)
            ddp(xl).backward()
            outs.append(M.flat.grads(m)[:M.flat.P].clone())
            lays.append([[names[id(p)] for p in b.parameters()] for b in ddp.reducer._get_zeros_like_grad_buckets()])
        del ddp
        return outs, lays

    @staticmethod
    def ordered_hook(grp, bucket):
        """DDP comm hook: the ordered all-reduce on one bucket. DDP's own path scales by 1/d first."""
        x = bucket.buffer()
        x.mul_(1.0 / grp.d)
        fut = torch.futures.Future()
        fut.set_result(ordered_reduce(x, grp, "rank"))
        return fut

    def ar_packings(self, flat, full=True):
        pl = [(n, k) for n, k, _ in flat.items]
        pk = [Packing("flat/fwd", [ordered(pl, "fwd")]), Packing("flat/rev", [ordered(pl, "rev")]),
              Packing("flat/fwd+pad1000", [[(None, 1000)] + pl]),
              Packing("ddp-default(1MiB,25MiB)/rev", buckets(ordered(pl, "rev"), 25 * MIB, MIB)),
              Packing("bucket128K/rev", buckets(ordered(pl, "rev"), 128 << 10)),
              Packing("bucket1M/rev", buckets(ordered(pl, "rev"), MIB)), Packing("per-layer", by_layer(pl))]
        if full:
            pk += [Packing("flat/size", [ordered(pl, "size")]), Packing("flat/shuf", [ordered(pl, "shuf")]),
                   Packing("flat/fwd+pad1", [[(None, 1)] + pl]), Packing("flat/fwd+pad4096", [[(None, 4096)] + pl]),
                   Packing("flat/fwd+pad100003", [[(None, 100003)] + pl])]
            pk += [Packing(f"bucket{c >> 10}K/rev", buckets(ordered(pl, "rev"), c)) for c in (64 << 10, 192 << 10, 256 << 10)]
            pk += [Packing("bucket4M/rev", buckets(ordered(pl, "rev"), 4 * MIB)),
                   Packing("bucket1M/fwd", buckets(pl, MIB)), Packing("bucket1M/shuf", buckets(ordered(pl, "shuf"), MIB)),
                   Packing("bucket256K/size", buckets(ordered(pl, "size"), 256 << 10))]
        return Packing("per-parameter", [[(n, k)] for n, k in pl]), pk

    def ddp_packing(self, M, grp, full=True):
        """Measure 1 on one group: repeats, and the same gradients reduced under many packings."""
        flat, d, tree = M.flat, grp.d, self.args.tag == "tree"
        tag = f"{M.cfg}, d={d}"
        g, _ = self.local_scaled(M, M.tokens[:16], grp)
        per, pks = self.ar_packings(flat, full and not self.quick)
        probe = max(flat.items, key=lambda it: it[1])[0]            # the largest tensor
        psl = slice(flat.off[probe], flat.off[probe] + dict((n, k) for n, k, _ in flat.items)[probe])
        for impl, op, expect in (("nccl_allreduce", lambda x: nccl_allreduce(x, grp), "pass" if tree else "fail"),
                                 ("ordered_allreduce", lambda x: ordered_reduce(x, grp, "rank"), "pass")):
            ref = run_packing(per, g, flat, op, d)
            reps = [run_packing(p, g, flat, op, d) for p in (per, per, pks[0], pks[3])]
            reps2 = [run_packing(p, g, flat, op, d) for p in (pks[0], pks[3])]
            self.vote(f"repeats: the same message reduced again gives the same bits ({tag})",
                      bit_equal(ref, reps[0]) and bit_equal(ref, reps[1]) and bit_equal(reps[2], reps2[0])
                      and bit_equal(reps[3], reps2[1]), level="L1", impl=impl,
                      expect="pass" if impl == "ordered_allreduce" else "info")
            rows, hashes, lead = {}, {sha(ref[psl])}, None
            for p in pks:
                out = run_packing(p, g, flat, op, d)
                dd = diff(out, ref)
                hashes.add(sha(out[psl]))
                if dd:
                    kinds = by_kind(out, ref, flat)
                    rows[p.name] = [dd["mismatched"], sum(v[2] for v in kinds.values()), len(p.msgs), dd["max_abs_diff"]]
                else:
                    rows[p.name] = [0, 0, len(p.msgs), 0.0]
                if p.name.startswith("ddp-default"):
                    lead = out
            failing = [k for k, v in rows.items() if v[0]]
            self.vote(f"packing: a parameter's averaged gradient does not depend on the bucket layout ({tag})",
                      not failing, level="L2", impl=impl, expect=expect, packings=len(pks), failing=len(failing),
                      of=flat.P, tensors=len(flat.items), probe_tensor=probe, probe_distinct_results=len(hashes),
                      columns=["elements differing from per-parameter all-reduce", "tensors differing", "messages", "max abs diff"],
                      rows=rows)
            same = bit_equal(self.from_leader(ref, grp), ref) and bit_equal(self.from_leader(lead, grp), lead)
            self.vote(f"all ranks hold the same reduced gradient ({tag})", same, level="other", impl=impl,
                      expect="pass" if impl == "ordered_allreduce" else "info")
            self.keep[(impl, M.cfg, grp.name)] = ref
            if impl == "nccl_allreduce" and M.cfg == "small":
                # which order produced each element: per-parameter messages against one flat message
                every = torch.empty(d, flat.P, device=self.dev)
                dist.all_gather_into_tensor(every, g, group=grp.group)
                one = run_packing(pks[0], g, flat, op, d)
                if grp.me == 0:
                    names, cands = order_candidates(every)
                    cov = {"per-parameter": order_cover(names, cands, ref), "flat/fwd": order_cover(names, cands, one)}
                    del cands
                    self.vote(f"order diagnostic: every element equals a sequential ring or tree order ({tag})",
                              all(c["unexplained"] == 0 for c in cov.values()), level="other", impl=impl,
                              expect="info", **cov)
                del every
        return g

    def ddp_torch(self, M, grp, g):
        """Measure 1 with torch DDP itself: the first iteration against the next, bucket_cap_mb, the
        ordered comm hook, and the emulation above against DDP."""
        flat, d, tree = M.flat, grp.d, self.args.tag == "tree"
        tag, expect = f"{M.cfg}, d={d}", "pass" if tree else "fail"
        X = M.tokens[:16]
        xl = X[grp.me * 16 // d:(grp.me + 1) * 16 // d]
        caps = (None, 1) if self.quick else (None, 1, 0.25, 4)
        iters = 3 if self.quick else 4
        runs = {cap: self.torch_ddp(M, grp, xl, cap, iters) for cap in caps}
        outs, lays = runs[None]
        dd = diff(outs[0], outs[1])
        self.vote(f"torch DDP: the first iteration and the second give the same gradient for the same input ({tag})",
                  dd is None, level="L1", impl="torch_ddp", expect=expect,
                  buckets_iteration1=[sum(4 * dict((n, k) for n, k, _ in flat.items)[x] for x in b) for b in lays[0]],
                  buckets_iteration2=[sum(4 * dict((n, k) for n, k, _ in flat.items)[x] for x in b) for b in lays[1]],
                  **(dd or {}))
        self.vote(f"torch DDP: iterations 2 to {iters} repeat bit for bit ({tag})",
                  all(bit_equal(outs[1], o) for o in outs[2:]) and all(
                      bit_equal(runs[c][0][1], o) for c in caps for o in runs[c][0][2:]),
                  level="L1", impl="torch_ddp", expect="info")
        rows = {}
        for cap in caps[1:]:
            dc = diff(runs[cap][0][1], outs[1])
            rows[f"bucket_cap_mb={cap}"] = [dc["mismatched"] if dc else 0, len(runs[cap][1][1]), dc["max_abs_diff"] if dc else 0.0]
        self.vote(f"torch DDP: bucket_cap_mb does not change the gradient ({tag})", not any(v[0] for v in rows.values()),
                  level="L2", impl="torch_ddp", expect=expect, of=flat.P,
                  columns=["elements differing from the default cap", "buckets", "max abs diff"], rows=rows)
        # the emulation reproduces DDP when it uses DDP's own bucket layout
        size = {n: k for n, k, _ in flat.items}
        emu = []
        for cap in caps:
            for it in (0, 1):
                pk = Packing("ddp", [[(n, size[n]) for n in b] for b in runs[cap][1][it]])
                emu.append(bit_equal(run_packing(pk, g, flat, lambda x: nccl_allreduce(x, grp), d), runs[cap][0][it]))
        self.vote(f"the packing emulation with DDP's bucket layout equals torch DDP ({tag})", all(emu), level="other",
                  impl="nccl_allreduce", expect="info", compared=len(emu), equal=sum(emu))
        # the ordered all-reduce as a comm hook
        href = self.keep.get(("ordered_allreduce", M.cfg, grp.name))
        ok, n = True, 0
        for cap in caps[:2] if self.quick else caps[:3]:
            ho, _ = self.torch_ddp(M, grp, xl, cap, 3, hook=Worker.ordered_hook)
            ok &= all(bit_equal(o, href) for o in ho)
            n += len(ho)
        self.vote(f"torch DDP with the ordered comm hook: first iteration, later ones and every bucket_cap_mb "
                  f"equal the ordered all-reduce ({tag})", ok, level="L2", impl="torch_ddp+ordered_hook", gradients=n)
        self.keep[("torch_ddp", M.cfg, grp.name)] = outs[1]
        return outs[1]

    def plain_ref(self, M):
        """One batched backward over the 16 sequences on rank 0 (d = 1), broadcast: gradient and loss."""
        key = ("plain", M.cfg, "d1")
        if key not in self.keep:
            if self.rank == 0:
                g, loss = batch_grad(M.model, M.flat, M.tokens[:16])
                g = g[:M.flat.P].clone()
            else:
                g, loss = torch.empty(M.flat.P, device=self.dev), torch.empty((), device=self.dev)
            self.keep[key] = self.bcast(g)
            self.keep[key + ("loss",)] = self.bcast(loss)
        return self.keep[key], self.keep[key + ("loss",)]

    def part_ddp(self):
        M = self.setup("small")
        flat, X = M.flat, M.tokens[:16]
        fixed = [self.top] if self.quick else [self.top, "d4"] + (["d8"] if self.xnode else [])
        for grp in self.mine(fixed):
            g = self.ddp_packing(M, grp)
            if grp.name == self.top or not self.xnode:
                self.ddp_torch(M, grp, g)
            self.log(f"ddp packing {grp.name} done")
        if not self.quick:
            Mm = self.setup("medium")
            for grp in self.mine([self.top]):
                self.ddp_packing(Mm, grp, full=False)
            if "d3" in self.G and self.G["d3"].me is not None:
                # torch's allreduce_hook divides by d; DDP's default path multiplies by 1/d
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
                grp = self.G["d3"]
                xl = X[grp.me * 5:(grp.me + 1) * 5]
                a = self.torch_ddp(M, grp, xl, None, 2)[0][1]
                b = self.torch_ddp(M, grp, xl, None, 2, hook=lambda state, bucket: allreduce_hook(state.group, bucket))[0][1]
                dd = diff(b, a)
                self.vote("torch DDP at d=3: allreduce_hook gives the bits of the default path", dd is None,
                          level="other", impl="torch_ddp", expect="info", **(dd or {}))
        # measure 2: the same global batch at every degree
        ref, lref = self.plain_ref(M)
        for grp in self.mine(self.pow2[1:]):
            d = grp.d
            g, loss = self.local_scaled(M, X, grp)
            out = self.keep.get(("torch_ddp", "small", grp.name))
            if out is None:
                out = self.torch_ddp(M, grp, X[grp.me * 16 // d:(grp.me + 1) * 16 // d], None, 2)[0][1]
            for impl, res in (("torch_ddp", out), ("batched+ordered_allreduce", ordered_reduce(g, grp, "rank"))):
                dd = diff(res, ref)
                self.vote("degree: the averaged gradient of the same 16 sequences equals the d=1 gradient (small)",
                          dd is None, level="L5", impl=impl, expect="fail", case=grp.name,
                          row={**(dd or {}), "by_kind": by_kind(res, ref, flat) if dd else None})
            mean = loss.clone()
            dist.all_reduce(mean, group=grp.group)
            mean = mean * (1.0 / d)
            self.vote("degree: the loss mean (per-rank mean, all-reduce, divide by d) equals the d=1 loss (small)",
                      bit_equal(mean, lref), level="L5", impl="loss_mean_stock", expect="fail", case=grp.name,
                      row=[float(mean), float(lref)])
        self.log("ddp degrees done")

    # ------------------------------------------------------------------------- part: ls
    def part_ls(self):
        runs = [("small", 16), ("small", 64)] + ([] if self.quick else [("medium", 16)])
        for cfg, G in runs:
            M = self.setup(cfg)
            X = M.tokens[:G]
            ref, lref = self.ls_ref(cfg, G)
            tag = f"{cfg}, {G} sequences"
            names = None if cfg == "small" else self.pow2 + ["d3", "d8x", "d12x"]
            for grp in self.mine(names):
                for route in self.routes(grp):
                    first = None
                    for rep in range(2 if self.quick or cfg != "small" else 3):
                        g, loss = self.ls_grad(M, X, grp, route)
                        if rep == 0:
                            first = g
                            dd = diff(g, ref)
                            self.vote(f"logical shards: the gradient at every degree and placement equals d=1 ({tag})",
                                      dd is None, level="L5", impl="logical_shards", case=f"{grp.name}/{route}", **(dd or {}))
                            self.vote(f"logical shards: the loss mean at every degree and placement equals d=1 ({tag})",
                                      bit_equal(loss, lref), level="L5", impl="loss_mean_ordered", case=f"{grp.name}/{route}")
                        else:
                            self.vote(f"logical shards: repeats give the same gradient ({tag})", bit_equal(g, first),
                                      level="L1", impl="logical_shards", case=f"{grp.name}/{route}")
                    self.vote(f"logical shards: all ranks hold the same gradient ({tag})",
                              bit_equal(self.from_leader(first, grp), first), level="other", impl="logical_shards",
                              case=f"{grp.name}/{route}")
            self.log(f"ls {tag} done")
        M = self.setup("small")
        X, flat = M.tokens[:16], M.flat
        ref, _ = self.ls_ref("small", 16)
        for grp in self.mine(self.pow2[1:]):
            d = grp.d
            # rejected: the cross-rank part of the tree left to NCCL
            acc, _ = self.shard_grads(M.model, flat, X, shard_assign(d)[grp.me])
            out = nccl_allreduce(acc.vals[0], grp).mul_(1.0 / S)
            dd = diff(out, ref)
            self.vote("logical shards with an NCCL all-reduce across ranks equal d=1, at three or more ranks (small)"
                      if d > 2 else "logical shards with an NCCL all-reduce across ranks equal d=1, at two ranks (small)",
                      dd is None, level="L5", impl="logical_shards+nccl_allreduce", expect="fail" if d > 2 else "info",
                      case=grp.name, row=dd)
            # strided shard assignment: leaves sent without pre-reduction pass, a per-rank pre-sum does not
            g, _ = self.ls_grad(M, X, grp, strided=True)
            self.vote("logical shards, strided assignment (rank r holds shards r, r+d, ...), leaves sent unreduced: equals d=1 (small)",
                      bit_equal(g, ref), level="L5", impl="logical_shards", case=grp.name)
            if d < S:
                acc, _ = self.shard_grads(M.model, flat, X, shard_assign(d, True)[grp.me])
                pre = acc.vals[0].clone()
                for v in acc.vals[1:]:
                    pre.add_(v)
                out = ordered_reduce(pre, grp, "tree").mul_(1.0 / S)
                dd = diff(out, ref)
                self.vote("strided shards pre-summed per rank, then the ordered tree across ranks, equal d=1 (small)",
                          dd is None, level="L5", impl="strided_presum+ordered_allreduce", expect="fail", case=grp.name, row=dd)
        # accuracy against fp64 (rank 0)
        if self.rank == 0:
            m64 = copy.deepcopy(M.model).double()
            m64.zero_grad(set_to_none=True)
            m64(X).backward()
            g64 = torch.cat([p.grad.reshape(-1) for p in m64.parameters()])
            del m64
            for impl, deg, v in (("one_batched_backward", 1, self.keep.get(("plain", "small", "d1"))),
                                 ("torch_ddp", self.W, self.keep.get(("torch_ddp", "small", self.top))),
                                 ("logical_shards", "any", ref[:flat.P])):
                if v is not None:
                    e = v.double() - g64
                    self.measure("gradient error against fp64, relative Frobenius", float(e.norm() / g64.norm()), "rel",
                                 impl=impl, max_abs=float(e.abs().max()), degree=deg)
            dd = diff(ref[:flat.P], self.keep[("plain", "small", "d1")]) if ("plain", "small", "d1") in self.keep else None
            if dd:
                self.vote("logical shards at d=1 give the bits of one batched backward over the 16 sequences (small)",
                          False, level="other", impl="logical_shards", expect="info", **dd)

    # ------------------------------------------------------------------------- part: fsdp
    def rs_packings(self, flat, d, full=True):
        pl = [(n, k) for n, k, _ in flat.items]
        sz = -(-flat.P // d)
        pk = [Packing("pad-front-1000", [[(None, 1000)] + pl]), Packing("pad-front-half-slice", [[(None, sz // 2)] + pl]),
              Packing("reversed", [ordered(pl, "rev")]), Packing("owners-rotated-by-1", [pl], roll=1),
              Packing("per-layer", by_layer(pl))]
        if full:
            pk += [Packing("pad-front-1", [[(None, 1)] + pl]), Packing("shuffled", [ordered(pl, "shuf")]),
                   Packing("per-parameter", [[(n, k)] for n, k in pl])]
        return Packing("canonical", [pl]), pk

    def fsdp_layouts(self, M, grp):
        """Measure 4b on one group: the flat gradient under shifted layouts, stock and ordered."""
        flat, d = M.flat, grp.d
        tag = f"{M.cfg}, d={d}"
        g, _ = self.local_scaled(M, M.tokens[:16], grp)
        canon, pks = self.rs_packings(flat, d, not self.quick)
        own0 = owner_map(canon, flat, d, self.dev)
        sizes = lambda x: [x.numel() // d] * d                                       # noqa: E731
        stock = lambda x: gather_slices(nccl_reduce_scatter(x, grp), sizes(x), grp)  # noqa: E731
        ours = lambda x: gather_slices(ordered_reduce(x, grp, "rank", scatter=True), sizes(x), grp)   # noqa: E731
        res = {}
        for impl, op in (("nccl_reduce_scatter", stock), ("ordered_reduce_scatter", ours)):
            ref = run_packing(canon, g, flat, op, d, pad_d=True)
            again = [run_packing(canon, g, flat, op, d, pad_d=True) for _ in range(2 if self.quick else 4)]
            self.vote(f"repeats: the same reduce-scatter gives the same bits ({tag})", all(bit_equal(ref, a) for a in again),
                      level="L1", impl=impl, expect="pass" if impl.startswith("ordered") else "info")
            rows, clean = {}, True
            for p in pks:
                out = run_packing(p, g, flat, op, d, pad_d=True)
                moved = owner_map(p, flat, d, self.dev) != own0
                bad = bits(out) != bits(ref)
                rows[p.name] = [int(bad.sum()), int(moved.sum()), int((bad & ~moved).sum())]
                clean &= rows[p.name][2] == 0
            failing = [k for k, v in rows.items() if v[0]]
            self.vote(f"layout: a parameter's reduced gradient does not depend on which rank owns it ({tag})",
                      not failing, level="L2", impl=impl, expect="pass" if impl.startswith("ordered") else "fail",
                      layouts=len(pks), failing=len(failing), of=flat.P,
                      columns=["elements differing from the canonical layout", "elements whose owner changed",
                               "differing elements whose owner did not change"], rows=rows)
            if impl == "nccl_reduce_scatter":
                self.vote(f"reduce-scatter: elements that keep their owner keep their bits under every layout ({tag})",
                          clean, level="other", impl=impl, expect="info")
            res[impl] = ref
        # the ordered reduce-scatter is the ordered all-reduce, and NCCL's two collectives disagree
        oar = ordered_reduce(g, grp, "rank")
        self.vote(f"the ordered reduce-scatter, gathered, equals the ordered all-reduce ({tag})",
                  bit_equal(res["ordered_reduce_scatter"], oar), level="other", impl="ordered_reduce_scatter")
        ar = nccl_allreduce(g, grp)
        per_owner = lambda a, b: [int((bits(a) != bits(b))[own0 == j].sum()) for j in range(d)]    # noqa: E731
        self.vote(f"NCCL reduce-scatter equals NCCL all-reduce on the same inputs ({tag})",
                  bit_equal(res["nccl_reduce_scatter"], ar), level="other", impl="nccl_reduce_scatter", expect="info",
                  mismatched_by_owner=per_owner(res["nccl_reduce_scatter"], ar), elements_per_owner=int((own0 == 0).sum()))
        self.vote(f"NCCL reduce-scatter equals the rank-order sum 0..d-1 ({tag})",
                  bit_equal(res["nccl_reduce_scatter"], oar), level="other", impl="nccl_reduce_scatter", expect="info",
                  mismatched_by_owner=per_owner(res["nccl_reduce_scatter"], oar))
        if M.cfg == "small":
            every = torch.empty(d, flat.P, device=self.dev)
            dist.all_gather_into_tensor(every, g, group=grp.group)
            if grp.me == 0:
                cov = []
                for j in sorted({0, 1, d // 2, d - 1}):
                    idx = (own0 == j).nonzero().squeeze(1)[:200000]
                    names, cands = order_candidates(every[:, idx])
                    c = order_cover(names, cands, res["nccl_reduce_scatter"][idx], top=2)
                    cov.append([j, c["orders"], c["n_orders"], c["unexplained"]])
                self.vote(f"order diagnostic: the slice kept by rank j is summed from rank j+1 around the ring ({tag})",
                          all(c[2] == 1 and c[1][0][0] == f"fwd{(c[0] + 1) % d}" for c in cov), level="other",
                          impl="nccl_reduce_scatter", expect="info", columns=["owner", "orders", "orders needed", "unexplained"],
                          rows=cov)
            del every

    def part_fsdp(self):
        M = self.setup("small")
        flat, X = M.flat, M.tokens[:16]
        fixed = [self.top] if self.quick else [self.top, "d4"] + (["d8"] if self.xnode else [])
        for grp in self.mine(fixed):
            self.fsdp_layouts(M, grp)
            self.log(f"fsdp layouts {grp.name} done")
        # 4a: stock FSDP-style gradient (batched backward, reduce-scatter of the flat gradient) across d
        ref, _ = self.plain_ref(M)
        for grp in self.mine(self.pow2[1:]):
            d = grp.d
            g, _ = self.local_scaled(M, X, grp)
            pad = (-flat.P) % d
            x = torch.cat([g, g.new_zeros(pad)]) if pad else g
            out = gather_slices(nccl_reduce_scatter(x, grp), [x.numel() // d] * d, grp)[:flat.P]
            dd = diff(out, ref)
            self.vote("degree: the reduce-scattered gradient of the same 16 sequences equals the d=1 gradient (small)",
                      dd is None, level="L5", impl="nccl_reduce_scatter", expect="fail", case=grp.name, row=dd)
        # logical shards with the ordered reduce-scatter: the sharded gradient is the replicated one
        lref, _ = self.ls_ref("small", 16)
        for grp in self.mine():
            for route in self.routes(grp):
                gs, _ = self.ls_grad(M, X, grp, route, scatter=True)
                full = gather_slices(gs, flat.sizes(grp.d), grp)
                dd = diff(full, lref)
                self.vote("logical shards with the ordered reduce-scatter: the gathered slices equal the d=1 gradient (small)",
                          dd is None, level="L5", impl="logical_shards+ordered_reduce_scatter", case=f"{grp.name}/{route}",
                          **(dd or {}))
                # the all-gather of the sharded parameters returns the parameters
                b = flat.bounds(grp.d)
                th = gather_slices(M.theta0[b[grp.me]:b[grp.me + 1]].contiguous(), flat.sizes(grp.d), grp)
                self.vote("all-gather of the parameter slices returns the parameters bit for bit (small)",
                          bit_equal(th, M.theta0), level="other", impl="all_gather", case=grp.name)
        for grp in self.mine(self.pow2[1:]):
            d = grp.d
            acc, _ = self.shard_grads(M.model, flat, X, shard_assign(d)[grp.me])
            part = acc.vals[0]
            # rejected: logical shards, but NCCL's reduce-scatter across ranks
            out = gather_slices(nccl_reduce_scatter(part, grp), flat.sizes(d), grp).mul_(1.0 / S)
            dd = diff(out, lref)
            own = torch.arange(flat.n, device=self.dev) // (flat.n // d)
            self.vote("logical shards with an NCCL reduce-scatter across ranks equal d=1, at three or more ranks (small)"
                      if d > 2 else "logical shards with an NCCL reduce-scatter across ranks equal d=1, at two ranks (small)",
                      dd is None, level="L5", impl="logical_shards+nccl_reduce_scatter", expect="fail" if d > 2 else "info",
                      case=grp.name, row={**(dd or {}), "mismatched_by_owner":
                                          [int((bits(out) != bits(lref))[own == j].sum()) for j in range(d)] if dd else None})
            # the tree-ordered reduce-scatter under shifted layouts
            if grp.name == self.top:
                canon, pks = self.rs_packings(flat, d, not self.quick)
                op = lambda x: gather_slices(ordered_reduce(x, grp, "tree", scatter=True), [x.numel() // d] * d, grp)   # noqa: E731
                ok = all(bit_equal(run_packing(p, part, flat, op, d, pad_d=True).mul_(1.0 / S), lref[:flat.P]) for p in [canon] + pks)
                self.vote(f"logical shards with the ordered reduce-scatter under shifted layouts equal the d=1 gradient (small, d={d})",
                          ok, level="L2", impl="logical_shards+ordered_reduce_scatter", layouts=len(pks) + 1)
        self.log("fsdp done")

    # ------------------------------------------------------------------------- part: glob
    def part_glob(self):
        """Measure 5: the global grad-norm. The same vectors at every degree, so that only the
        reduction of the norm differs."""
        M = self.setup("small")
        flat = M.flat
        vecs = {"gradient, 16 sequences": self.ls_ref("small", 16)[0], "gradient, 64 sequences": self.ls_ref("small", 64)[0],
                "initial weights": M.theta0}
        for vname, v in vecs.items():
            # stock: FSDP-style, slices of P padded to a multiple of d, local sum of squares, all-reduce
            def stock(grp, shift=0):
                d = grp.d
                x = torch.cat([v.new_zeros(shift), v[:flat.P]])
                x = torch.cat([x, x.new_zeros((-x.numel()) % d)])
                sz = x.numel() // d
                t = torch.linalg.vector_norm(x[grp.me * sz:(grp.me + 1) * sz]) ** 2
                if d > 1:
                    dist.all_reduce(t, group=grp.group)
                return t.sqrt()
            ref = self.bcast(stock(self.G["d1"]) if self.rank == 0 else torch.empty((), device=self.dev))
            lref = self.bcast(logical_norm(v) if self.rank == 0 else torch.empty((), device=self.dev, dtype=torch.float64))
            for grp in self.mine():
                if grp.name != "d1":
                    t = stock(grp)
                    self.vote("grad-norm, sharded the FSDP way (local sum of squares, NCCL all-reduce): equals the d=1 norm",
                              bit_equal(t, ref), level="L5", impl="sharded_norm_stock", expect="fail",
                              case=f"{vname}/{grp.name}", row=[float(t), float(ref)])
                b = flat.bounds(grp.d)
                n1 = self.norm_sharded(v[b[grp.me]:b[grp.me + 1]], flat, grp)
                self.vote("logical norm from block-aligned slices equals the norm of the whole vector at every degree",
                          bit_equal(n1, lref) and bit_equal(logical_norm(v), lref), level="L5", impl="logical_norm",
                          case=f"{vname}/{grp.name}")
            for grp in self.mine([self.top]):
                vals = {sh: float(stock(grp, sh)) for sh in (0, 1, 1000, 4096)}
                self.vote(f"grad-norm, sharded the FSDP way: padding in front of the flat vector does not change it (d={grp.d})",
                          len({repr(x) for x in vals.values()}) == 1, level="L2", impl="sharded_norm_stock", expect="fail",
                          case=vname, row=vals)
            if self.rank == 0:
                exact = float(v.double().pow(2).sum().sqrt())
                self.measure("grad-norm error against an fp64 sum, relative", abs(float(lref) - exact) / exact, "rel",
                             impl="logical_norm", vector=vname)
                self.measure("grad-norm error against an fp64 sum, relative", abs(float(ref) - exact) / exact, "rel",
                             impl="sharded_norm_stock", degree=1, vector=vname)
        if not self.quick:
            # another block size: another fixed arithmetic, invariant in the same way
            f2 = Flat(M.model, blk=1024)
            v = torch.cat([vecs["gradient, 16 sequences"][:flat.P], torch.zeros(f2.n - flat.P, device=self.dev)])
            r2 = self.bcast(logical_norm(v, 1024) if self.rank == 0 else torch.empty((), device=self.dev, dtype=torch.float64))
            for grp in self.mine():
                b = f2.bounds(grp.d)
                self.vote("logical norm with blocks of 1024 elements: the same at every degree", bit_equal(
                    self.norm_sharded(v[b[grp.me]:b[grp.me + 1]], f2, grp), r2), level="L5", impl="logical_norm", case=grp.name)
            if self.rank == 0:
                self.vote("logical norm: blocks of 1024 and of 4096 elements give the same bits",
                          bit_equal(r2, logical_norm(vecs["gradient, 16 sequences"])), level="other", impl="logical_norm",
                          expect="info")
        self.log("glob done")

    # ------------------------------------------------------------------------- part: e2e
    def new_opt(self, params, path="single"):
        kw = {"single": dict(foreach=False, fused=False), "foreach": dict(foreach=True, fused=False),
              "fused": dict(fused=True)}[path]
        return torch.optim.AdamW(params, **OPT, **kw)

    def train(self, M, grp, mode, steps, state=None, route="a2a", path="single", t0=0):
        """Steps t0..t0+steps-1 with logical shards, ordered reductions, clipping by the logical norm
        and AdamW. mode dp: every rank holds all weights. mode fsdp: every rank holds a slice of the
        flat weights and of the optimizer state. Returns the canonical state and per-step loss, norm."""
        flat, d, me = M.flat, grp.d, grp.me
        state = state or {"theta": M.theta0, "m": None, "v": None, "step": 0}
        model = copy.deepcopy(M.model)
        b = flat.bounds(d)
        if mode == "dp":
            flat.load(model, state["theta"])
            opt = self.new_opt(model.parameters(), path)
            targets = [(p, flat.off[n], k) for (n, k, _), p in zip(flat.items, model.parameters())]
        else:
            shard = nn.Parameter(state["theta"][b[me]:b[me + 1]].clone())
            opt = self.new_opt([shard], path)
            targets = [(shard, b[me], b[me + 1] - b[me])]
        if state["m"] is not None:                             # resume: optimizer state from the checkpoint
            for p, o, k in targets:
                dev_step = path == "fused"
                opt.state[p] = {"step": torch.tensor(float(state["step"]), dtype=torch.float32, device=self.dev if dev_step else None),
                                "exp_avg": state["m"][o:o + k].view(p.shape).clone(),
                                "exp_avg_sq": state["v"][o:o + k].view(p.shape).clone()}
        losses, norms, clipped = [], [], 0
        for t in range(t0, t0 + steps):
            X = M.tokens[16 * t:16 * t + 16]
            if mode == "fsdp":
                flat.load(model, gather_slices(shard.data, flat.sizes(d), grp))
                gs, loss = self.ls_grad(M, X, grp, route, scatter=True, model=model)
                norm = self.norm_sharded(gs, flat, grp)
            else:
                gs, loss = self.ls_grad(M, X, grp, route, model=model)
                norm = logical_norm(gs)
            coef = min(1.0, MAX_NORM / (float(norm) + 1e-6))
            clipped += coef < 1.0
            gs.mul_(coef)
            if mode == "fsdp":
                shard.grad = gs
            else:
                flat.set_grads(model, gs)
            opt.step()
            losses.append(loss)
            norms.append(norm)

        def full(key):
            if mode == "fsdp":
                return gather_slices(opt.state[shard][key], flat.sizes(d), grp)
            return flat._cat([opt.state[p][key].reshape(-1) for p in model.parameters()])
        theta = gather_slices(shard.data, flat.sizes(d), grp) if mode == "fsdp" else flat.params(model)
        return {"theta": theta.clone(), "m": full("exp_avg"), "v": full("exp_avg_sq"), "step": state["step"] + steps,
                "losses": torch.stack(losses), "norms": torch.stack(norms), "clipped": clipped}

    def stock_train(self, M, grp, steps, cap=None, restart_at=None):
        """torch DDP, torch clip_grad_norm_, the same pinned AdamW: the stock data-parallel loop.
        restart_at: rebuild the model, DDP and the optimizer from state_dicts before that step."""
        from torch.nn.parallel import DistributedDataParallel as DDP
        flat, d = M.flat, grp.d

        def wrap(m):
            return DDP(m, device_ids=[self.local], process_group=grp.group, bucket_cap_mb=cap) if d > 1 else m
        model = copy.deepcopy(M.model)
        net, opt = wrap(model), self.new_opt(model.parameters())
        for t in range(steps):
            if restart_at == t:
                sd, od = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
                del net
                model = copy.deepcopy(M.model)
                model.load_state_dict(sd)
                net, opt = wrap(model), self.new_opt(model.parameters())
                opt.load_state_dict(od)
            X = M.tokens[16 * t:16 * t + 16]
            opt.zero_grad(set_to_none=True)
            net(X[grp.me * 16 // d:(grp.me + 1) * 16 // d]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()
        theta = flat.params(model).clone()
        del net
        return theta

    def part_e2e(self):
        M = self.setup("small")
        flat = M.flat
        keys = ("theta", "m", "v", "losses", "norms")
        if self.rank == 0:
            ref = self.train(M, self.G["d1"], "dp", STEPS)
            self.extra["e2e"] = {"losses": [float(x) for x in ref["losses"]], "grad_norms": [float(x) for x in ref["norms"]],
                                 "steps_clipped": int(ref["clipped"]), "max_norm": MAX_NORM, "optimizer": {**OPT, "path": "torch.optim.AdamW(foreach=False, fused=False)"},
                                 "model": dict(zip(("vocab", "width", "layers", "heads", "context"), CFGS["small"])),
                                 "parameters": flat.P, "sequences_per_step": 16, "logical_shards": S}
        else:
            ref = {"theta": torch.empty(flat.n, device=self.dev), "m": torch.empty(flat.n, device=self.dev),
                   "v": torch.empty(flat.n, device=self.dev), "losses": torch.empty(STEPS, device=self.dev),
                   "norms": torch.empty(STEPS, device=self.dev, dtype=torch.float64)}
        for k in keys:
            self.bcast(ref[k])
        self._e2e_ref = ref

        def same(out):
            return {k: diff(out[k], ref[k]) for k in keys if not bit_equal(out[k], ref[k])}
        half = f"d{self.W // 2}"
        only = {"dp": ("d2", "d3", self.top), "fsdp": (half, self.top)} if self.quick and not self.xnode else {}
        for mode in ("dp", "fsdp"):
            for grp in self.mine(only.get(mode)):
                if grp.name == "d1" and mode == "dp":
                    continue                                   # the reference itself
                for route in self.routes(grp):
                    out = self.train(M, grp, mode, STEPS, route=route)
                    bad = same(out)
                    self.vote(f"end to end: weights, AdamW state, losses and grad-norms after {STEPS} steps equal the d=1 run",
                              not bad, level="L5", impl=f"e2e_{mode}", case=f"{grp.name}/{route}", **bad)
            self.log(f"e2e {mode} done")
        # elastic resume through a checkpoint file: 5 steps at one degree, 5 at another, in every mode pairing
        a, b_ = half, self.top
        pairs = [("dp", "dp")] if self.quick else [("dp", "dp"), ("fsdp", "fsdp"), ("dp", "fsdp"), ("fsdp", "dp")]
        for m1, m2 in pairs:
            ga, gb = self.G[a], self.G[b_]
            path = os.path.join(self.args.out, f"ckpt-{self.args.tag}.pt")
            if ga.me is not None:
                st = self.train(M, ga, m1, STEPS // 2)
                if ga.me == 0:
                    torch.save({k: st[k].cpu() if torch.is_tensor(st[k]) else st[k] for k in ("theta", "m", "v", "step")}, path)
            dist.barrier()
            if gb.me is not None:
                if self.rank == gb.members[0]:
                    ck = torch.load(path, weights_only=True)
                    st = {k: ck[k].to(self.dev) if torch.is_tensor(ck[k]) else ck[k] for k in ck}
                else:
                    st = {k: torch.empty(flat.n, device=self.dev) for k in ("theta", "m", "v")}
                    st["step"] = STEPS // 2
                for k in ("theta", "m", "v"):
                    dist.broadcast(st[k], gb.members[0], group=gb.group)
                for route in self.routes(gb):
                    out = self.train(M, gb, m2, STEPS - STEPS // 2, state=st, route=route, t0=STEPS // 2)
                    bad = {k: diff(out[k], ref[k]) for k in ("theta", "m", "v") if not bit_equal(out[k], ref[k])}
                    bad.update({k: True for k in ("losses", "norms") if not bit_equal(out[k], ref[k][STEPS // 2:])})
                    self.vote(f"elastic resume: {STEPS // 2} steps at {a}, checkpoint file, {STEPS - STEPS // 2} steps at {b_} "
                              f"equal {STEPS} steps at d=1 (and so at every degree that passed above)", not bad, level="L5", impl="e2e_elastic",
                              case=f"{m1}->{m2}/{route}", **bad)
            dist.barrier()
        if self.rank == 0 and os.path.exists(os.path.join(self.args.out, f"ckpt-{self.args.tag}.pt")):
            os.remove(os.path.join(self.args.out, f"ckpt-{self.args.tag}.pt"))
        self.log("e2e elastic done")
        # optimizer paths: any one of them is degree-invariant, but they are not the same arithmetic
        if not self.quick:
            for path in ("foreach", "fused"):
                outs = {}
                for grp in self.mine(["d1", self.top]):
                    outs[grp.name] = self.train(M, grp, "dp", STEPS, path=path)["theta"]
                if self.rank == 0:
                    dd = diff(outs["d1"], ref["theta"])
                    self.vote(f"AdamW path {path} gives the weights of the single-tensor path after {STEPS} steps (d=1)",
                              dd is None, level="other", impl="adamw_paths", expect="info", **(dd or {}))
                    p1 = self.bcast(outs["d1"])
                else:
                    p1 = self.bcast(torch.empty(flat.n, device=self.dev))
                if self.top in outs:
                    self.vote(f"AdamW path {path}: weights at d={self.W} equal its own d=1 weights", bit_equal(outs[self.top], p1),
                              level="L5", impl="adamw_paths", expect="info")
        # the stock loop
        sref = self.bcast(self.stock_train(M, self.G["d1"], STEPS) if self.rank == 0 else torch.empty(flat.n, device=self.dev))
        tree = self.args.tag == "tree"
        for grp in self.mine(self.pow2[1:]):
            th = self.stock_train(M, grp, STEPS)
            dd = diff(th, sref)
            self.vote(f"stock loop (torch DDP, clip_grad_norm_, AdamW): weights after {STEPS} steps equal the d=1 run",
                      dd is None, level="L5", impl="e2e_stock_ddp", expect="fail", case=grp.name, row=dd)
            if grp.name == self.top:
                again = self.stock_train(M, grp, STEPS)
                self.vote(f"stock loop at d={grp.d}: a second run gives the same weights", bit_equal(again, th), level="L1",
                          impl="e2e_stock_ddp", expect="info")
                rs = self.stock_train(M, grp, STEPS, restart_at=STEPS // 2)
                dd = diff(rs, th)
                self.vote(f"stock loop at d={grp.d}: restarting from state_dicts after {STEPS // 2} steps gives the weights "
                          f"of the uninterrupted run", dd is None, level="L1", impl="e2e_stock_ddp",
                          expect="pass" if tree else "fail", **(dd or {}))
                if not self.quick:
                    c1 = self.stock_train(M, grp, STEPS, cap=1)
                    dd = diff(c1, th)
                    self.vote(f"stock loop at d={grp.d}: bucket_cap_mb=1 gives the weights of the default cap", dd is None,
                              level="L2", impl="e2e_stock_ddp", expect="pass" if tree else "fail", **(dd or {}))
        if self.rank == 0:
            dd = diff(sref, ref["theta"])
            self.vote("the stock loop at d=1 and the logical-shard loop give the same weights", dd is None, level="other",
                      impl="e2e_dp", expect="info", **(dd or {}))
            save = {f"ls_grad_{c}_{g}": v[0].cpu() for (c, g), v in self._refs.items()}
            save.update({f"ls_loss_{c}_{g}": v[1].cpu() for (c, g), v in self._refs.items()})
            save.update({"e2e_" + k: ref[k].cpu() for k in keys})
            torch.save(save, os.path.join(self.args.out, f"ref-{self.args.tag}.pt"))
        self.log("e2e stock done")

    # ------------------------------------------------------------------------- part: time
    def part_time(self):
        from torch.nn.parallel import DistributedDataParallel as DDP
        runs = [("small", 16)] if self.quick else [("small", 16), ("medium", 16), ("medium", 64)]
        for cfg, G in runs:
            M = self.setup(cfg)
            flat, X = M.flat, M.tokens[:G]
            n = 5 if cfg == "medium" else 7
            names = [x for x in self.pow2 if not self.quick or x in ("d1", self.top)]
            for name in names:
                grp = self.G[name]
                if grp.me is not None:
                    d = grp.d
                    xl = X[grp.me * G // d:(grp.me + 1) * G // d]
                    meta = dict(model=cfg, sequences=G, degree=d, parameters=flat.P)
                    row = {}
                    model = copy.deepcopy(M.model)
                    net = DDP(model, device_ids=[self.local], process_group=grp.group) if d > 1 else model

                    def stock():
                        model.zero_grad(set_to_none=True)
                        net(xl).backward()
                    row["torch_ddp"] = self.timeit(stock, grp, n)
                    del net
                    if d > 1:
                        hm = copy.deepcopy(M.model)
                        hnet = DDP(hm, device_ids=[self.local], process_group=grp.group)
                        hnet.register_comm_hook(grp, Worker.ordered_hook)

                        def hooked():
                            hm.zero_grad(set_to_none=True)
                            hnet(xl).backward()
                        row["torch_ddp+ordered_hook"] = self.timeit(hooked, grp, n)
                        del hnet, hm
                    row["backward_then_nccl_allreduce"] = self.timeit(
                        lambda: nccl_allreduce(batch_grad(M.model, flat, xl)[0], grp), grp, n)
                    row["logical_shards"] = self.timeit(lambda: self.ls_grad(M, X, grp), grp, n)
                    if grp.hier:
                        row["logical_shards_2level"] = self.timeit(lambda: self.ls_grad(M, X, grp, "2level"), grp, n)
                    sh = shard_assign(d)[grp.me]
                    row["logical_shards: backward passes only"] = self.timeit(lambda: self.shard_grads(M.model, flat, X, sh), grp, n)
                    row["logical_shards+ordered_reduce_scatter"] = self.timeit(
                        lambda: self.ls_grad(M, X, grp, scatter=True), grp, n)
                    if grp.me == 0:
                        for impl, ms in row.items():
                            self.measure("gradient of one global batch, all ranks", ms, "ms", impl=impl, **meta,
                                         ratio_to_torch_ddp=round(ms / row["torch_ddp"], 3))
                        self.log(f"time {cfg} G={G} {name}: " + ", ".join(f"{k} {v:.1f}" for k, v in row.items()))
                dist.barrier()
        # the collectives alone, on gradient-sized fp32 messages
        sizes = [3257856, 19177472] if self.quick else [262144, 3257856, 19177472, 1 << 26]
        if self.xnode:
            sizes = sizes[:-1] + ([] if self.quick else [1 << 24])
        for name in ([self.top] if self.quick or not self.xnode else [self.top, "d8"]):
            grp = self.G[name]
            if grp.me is None:
                dist.barrier()
                continue
            d = grp.d
            for nel in sorted(sizes):
                nel -= nel % (16 * BLK)
                x = torch.randn(nel, device=self.dev)
                it = 7 if nel <= (1 << 22) else 5
                sz = [nel // d] * d
                row = {"nccl_allreduce": self.timeit(lambda: nccl_allreduce(x, grp), grp, it),
                       "ordered_allreduce": self.timeit(lambda: ordered_reduce(x, grp, "rank"), grp, it),
                       "ordered_allreduce_tree": self.timeit(lambda: ordered_reduce(x, grp, "tree"), grp, it),
                       "nccl_reduce_scatter": self.timeit(lambda: nccl_reduce_scatter(x, grp), grp, it),
                       "ordered_reduce_scatter": self.timeit(lambda: ordered_reduce(x, grp, "rank", scatter=True), grp, it),
                       "ordered_reduce_scatter_tree": self.timeit(lambda: ordered_reduce(x, grp, "tree", scatter=True), grp, it),
                       "all_gather": self.timeit(lambda: gather_slices(x[:nel // d], sz, grp), grp, it),
                       "clone": self.timeit(lambda: x.clone(), grp, it)}
                if grp.hier:
                    row["ordered_allreduce_2level"] = self.timeit(lambda: hier_reduce(x, grp), grp, it)
                    row["ordered_reduce_scatter_2level"] = self.timeit(lambda: hier_reduce(x, grp, scatter=True), grp, it)
                if grp.me == 0:
                    for impl, ms in row.items():
                        base = row["nccl_reduce_scatter" if "scatter" in impl else "nccl_allreduce"]
                        self.measure("collective on one fp32 message", ms, "ms", impl=impl, bytes=4 * nel, degree=d,
                                     ratio_to_nccl=round(ms / base, 3) if impl not in ("all_gather", "clone") else None)
                    self.log(f"time coll {name} {4 * nel / MIB:.0f} MiB: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items()))
                del x
            dist.barrier()
        # the logical norm against torch's
        v = self.ls_ref("small", 16)[0]
        vecs = [("small", v)] + ([] if self.quick else [("medium", self.setup("medium").theta0)])
        for nm, vec in vecs:
            if self.rank == 0:
                g1 = self.G["d1"]
                self.measure("grad-norm of the full vector, one GPU", self.timeit(lambda: logical_norm(vec), g1), "ms",
                             impl="logical_norm", elements=vec.numel())
                self.measure("grad-norm of the full vector, one GPU", self.timeit(lambda: torch.linalg.vector_norm(vec), g1),
                             "ms", impl="torch.linalg.vector_norm", elements=vec.numel())

    # ------------------------------------------------------------------------- run
    def run(self):
        self.t0 = time.time()
        hosts = [None] * self.W
        dist.all_gather_object(hosts, os.uname().nodename)
        two = len(set(hosts)) == 2 and len(set(hosts[:8])) == 1 and len(set(hosts[8:])) == 1
        assert not self.xnode or two, "two-node mode expects ranks 0-7 on one node and 8-15 on the other"
        for part in self.args.parts.split(","):
            getattr(self, "part_" + part)()
            allv, allm = [None] * self.W, [None] * self.W
            dist.all_gather_object(allv, self.votes)
            dist.all_gather_object(allm, self.meas)
            if self.rank == 0:
                self.done = getattr(self, "done", {"votes": [], "meas": []})
                self.done["votes"] += [v for lst in allv for v in lst]
                self.done["meas"] += [m for lst in allm for m in lst]
                self.log(f"part {part} finished")
            self.votes, self.meas = [], []
            torch.cuda.empty_cache()
        if self.rank == 0:
            out = {"tag": self.args.tag, "world": self.W, "two_nodes": two, "quick": self.quick, "parts": self.args.parts,
                   "nccl_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_") and k != "NCCL_DEBUG_FILE"},
                   "seconds": round(time.time() - self.t0, 1), "placements": {n: g.members for n, g in self.G.items()},
                   "checks": aggregate(self.done["votes"]), "measurements": self.done["meas"], "extra": self.extra}
            with open(os.path.join(self.args.out, f"launch-{self.args.tag}.json"), "w") as f:
                json.dump(out, f, indent=1)
            self.log(f"done in {out['seconds']} s")
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()


def aggregate(votes):
    """Votes from every rank -> one check per (name, impl): passed when every vote passed. `case`
    values are listed, `row` values are kept per case, other details come from the first failing vote."""
    groups = {}
    for v in votes:
        groups.setdefault((v["name"], v["impl"]), []).append(v)
    out = []
    for (name, impl), vs in groups.items():
        bad = [v for v in vs if not v["ok"]]
        detail = {"votes": len(vs), "failed_votes": len(bad)}
        cases = sorted({v["info"]["case"] for v in vs if "case" in v["info"]})
        if cases:
            detail["cases"] = cases
            detail["failing_cases"] = sorted({v["info"]["case"] for v in bad if "case" in v["info"]})
        rows = {}
        for v in sorted(vs, key=lambda v: v["rank"]):
            if "row" in v["info"] and v["info"]["case"] not in rows:
                rows[v["info"]["case"]] = v["info"]["row"]
        if rows:
            detail["rows_by_case"] = rows
        src = min(bad or vs, key=lambda v: v["rank"])["info"]
        detail.update({k: v for k, v in src.items() if k not in ("case", "row")})
        out.append({"name": name, "impl": impl, "level": vs[0]["level"], "expect": vs[0]["expect"], "passed": not bad,
                    "detail": detail})
    return out


# ----------------------------------------------------------------------------- driver
def foreign_gpu_processes():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
        return len([ln for ln in out.splitlines() if ln.strip()])
    except Exception:
        return None


def launch(args, tag, parts, out):
    env = {**os.environ, **SETTINGS[tag]}
    for k in ("NCCL_ALGO", "NCCL_PROTO"):
        if k not in SETTINGS[tag]:
            env.pop(k, None)
    wargs = ["--worker", "--tag", tag, "--parts", parts, "--out", out] + (["--quick"] if args.quick else [])
    if args.two_node:
        env["TIMEOUT"] = str(args.timeout)
        cmd = [os.path.expanduser(args.launcher), HERE, os.path.basename(__file__)] + wargs + ["--xnode"]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.nproc}",
               os.path.abspath(__file__)] + wargs
    log = os.path.join(out, f"launch-{tag}.log")
    before = foreign_gpu_processes()
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True)
        try:
            rc = p.wait(timeout=args.timeout + 120)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            rc = -9
    res = os.path.join(out, f"launch-{tag}.json")
    ok = rc == 0 and os.path.exists(res) and os.path.getmtime(res) >= t0
    tail = [ln for ln in open(log, errors="replace").read().splitlines() if ln.startswith("[")][-3:]
    print(f"launch {tag}: parts {parts}, exit {rc}, {time.time() - t0:.0f} s; {tail[-1] if tail else ''}", flush=True)
    return (json.load(open(res)) if ok else None), {"exit": rc, "seconds": round(time.time() - t0, 1),
                                                     "gpu_processes_before": before, "gpu_processes_after": foreign_gpu_processes()}


def driver(args):
    from bi_report import Results
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if not args.two_node:
        n = torch.cuda.device_count()
        args.nproc = args.nproc or (8 if n >= 8 else 4)
        if n < 4:
            print("training_dp needs at least 4 GPUs: sums over fewer than three ranks hide order bugs")
            return 2
    res = Results("training_dp", quick=args.quick)
    every = "copy,pre,ddp,ls,fsdp,glob,e2e"
    plan = {"default": every + ("" if args.no_timing else ",time")}
    for tag in args.settings.split(","):
        if tag != "default":
            plan[tag] = "copy,ddp,fsdp,glob" if args.quick else every + ("" if args.no_timing or tag != "tree" else ",time")
    runs = {}
    for tag, parts in plan.items():
        r, meta = launch(args, tag, parts, out)
        runs[tag] = meta
        res.check(f"launch finished [{tag}]", r is not None, level="other", impl="driver", detail=meta)
        if r is None:
            continue
        meta.update({k: r[k] for k in ("world", "two_nodes", "parts", "nccl_env", "seconds", "placements")})
        for c in r["checks"]:
            res.check(f"{c['name']} [{tag}]", c["passed"], level=c["level"], impl=c["impl"], expect=c["expect"], detail=c["detail"])
        for m in r["measurements"]:
            res.measure(m.pop("name"), m.pop("value"), m.pop("unit"), nccl=tag, **m)
        if r["extra"]:
            res.extra.setdefault("launch_extra", {})[tag] = r["extra"]
    # the scheme's results must not depend on the NCCL setting, nor on one node against two
    refs = {t: os.path.join(out, f"ref-{t}.pt") for t in plan}
    refs = {t: torch.load(p, weights_only=True) for t, p in refs.items() if os.path.exists(p)}
    if len(refs) > 1:
        base = refs["default"]
        bad = [f"{t}/{k}" for t, r in refs.items() for k in base if not bit_equal(r[k], base[k])]
        res.check("logical shards: gradients, weights, AdamW state, losses and norms are the same bits under every NCCL setting",
                  not bad, level="other", impl="logical_shards", detail={"settings": list(refs), "tensors": list(base), "differing": bad})
    if args.ref and "default" in refs:
        other = torch.load(args.ref, weights_only=True)
        bad = [k for k in refs["default"] if not bit_equal(other[k], refs["default"][k])]
        res.check("placement: the two-node run gives the bits of the one-node run (gradients, final weights, AdamW state, "
                  "losses, norms)", not bad, level="L5", impl="logical_shards", detail={"tensors": list(other), "differing": bad})
    if "default" in refs:
        res.extra["sha1_of_reference_tensors"] = {k: hashlib.sha1(bits(v).numpy().tobytes()).hexdigest()[:16]
                                                  for k, v in refs["default"].items()}
    res.extra["launches"] = runs
    res.extra["scope"] = "two nodes, 16 ranks" if args.two_node else f"one node, {args.nproc} ranks"
    if not args.two_node:
        res.extra["skipped"] = ["two-node checks (d=16, cross-node placements, the two-level route, one node against two): "
                                "run with --two-node"]
        print("skipped: two-node checks (run with --two-node on node 0)")
    res.write(args.json)
    return res.exit_code()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="one node, fewer cases, about 3 minutes")
    ap.add_argument("--json", help="write the bi-results-v1 file here")
    ap.add_argument("--two-node", action="store_true", help="16 ranks on two nodes through --launcher")
    ap.add_argument("--launcher", default="~/bi-check-xnode/launch2.sh", help="two-node launcher: <workdir> <script> [args]")
    ap.add_argument("--ref", help="ref-default.pt of a one-node run, to compare the two-node bits with")
    ap.add_argument("--settings", default="default,ring,tree", help="NCCL settings to launch")
    ap.add_argument("--nproc", type=int, default=0, help="ranks on one node (default 8, or 4)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"), help="work directory for per-launch files")
    ap.add_argument("--timeout", type=int, default=1500, help="seconds per launch")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--xnode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tag", default="default", help=argparse.SUPPRESS)
    ap.add_argument("--parts", default="copy,pre,ddp,ls,fsdp,glob,e2e,time", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        Worker(args).run()
        return 0
    return driver(args)


if __name__ == "__main__":
    sys.exit(main())
