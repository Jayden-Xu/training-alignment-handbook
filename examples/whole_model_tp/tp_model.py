"""Megatron-style tensor parallelism for the dense model of examples/whole_model/model.py.

Each rank of a TP group holds:
  column-parallel  q_proj, k_proj, v_proj (whole heads), gate_proj, up_proj (columns of the
                   intermediate size), lm_head (a vocab shard)
  row-parallel     o_proj (the K rows of its heads), down_proj (the K rows of its intermediate columns)
  replicated       embedding, the RMSNorm weights, RoPE tables
KV heads: with kv_heads >= TP each rank owns kv_heads/TP of them; with fewer, each rank keeps the one
KV head its query heads use (Megatron's replication). The weights are sliced from a model.py
WholeModel, so TP = 1 runs model.py's weights.

Two variants:
  "stock"  model.py's stock op set (cuBLAS F.linear, SDPA, torch reductions). Row-parallel layers
           round each rank's partial product to bf16 and sum them with NCCL all-reduce (bf16).
           log_softmax over the vocab shards: all-reduce MAX, local torch sum of exp, all-reduce SUM.
  "inv"    model.py's bi op set, plus the logical-TP scheme for every row-parallel GEMM: K is cut into
           C fixed chunks (C = 16 by default, whatever the degree); a rank computes the fp32 partials of
           its C/TP chunks with one launch, adds them with a fixed pairwise tree, exchanges fp32 subtrees
           by column block (all-to-all), finishes the same tree, rounds to bf16 once, all-gathers.
           Vocab-parallel log_softmax: max (exact in any order) by all-reduce MAX; sum of exp per fixed
           vocab chunk (Cv = 16 chunks) with model.py's fixed-tree row sum; chunk sums all-gathered and
           added with a fixed pairwise tree. Every order depends on C and Cv, never on the degree.
Under autograd ("inv" only, kv_heads >= TP), the column-parallel input gradient dX = sum over output
columns of dY W is reduced with the same logical-TP scheme (chunks: heads for q/k/v, C for gate/up,
Cv for lm_head); the row-parallel input gradient needs no reduction. bwd_nccl=True replaces that
reduction by Megatron's (bf16 partial dX per rank, NCCL all-reduce): the negative control.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "whole_model"))
from bi_kernels import BF, mm_one_config  # noqa: E402
import model as wm  # noqa: E402

F32 = torch.float32


# ----------------------------------------------------------------------------- kernels
@triton.jit(do_not_specialize=["M"])
def _mm_chunks_kernel(a_ptr, b_ptr, c_ptr, M, N, KC, sam, sak, sbk, sbn, scc, scm, scn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """P[c] = A[:, c*KC:(c+1)*KC] @ B[c*KC:(c+1)*KC, :] in fp32, one chunk per grid z. The K loop of
    each chunk is bi_kernels._mm_kernel's (BK-wide steps from the chunk start); nothing depends on M."""
    pid_m, pid_n, c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr += c * KC * sak
    b_ptr += c * KC * sbk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, KC, BK):
        kk = k0 + rk
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak,
                    mask=(rm[:, None] < M) & (kk[None, :] < KC), other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(kk[:, None] < KC) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + c * scc + rm[:, None] * scm + rn[None, :] * scn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def mm_chunks(a: torch.Tensor, b: torch.Tensor, n: int, BM=64, BN=64, BK=32) -> torch.Tensor:
    """a [M, n*KC] bf16, b [n*KC, N] bf16 -> fp32 [n, M, N]: the n K-chunk partial products."""
    M, Kt = a.shape
    N = b.shape[1]
    KC = Kt // n
    assert KC * n == Kt and b.shape[0] == Kt, (a.shape, b.shape, n)
    P = torch.empty(n, M, N, device=a.device, dtype=F32)
    if M:
        _mm_chunks_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), n)](
            a, b, P, M, N, KC, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            P.stride(0), P.stride(1), P.stride(2), BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return P


def tree(G: torch.Tensor) -> torch.Tensor:
    """Fixed pairwise tree over dim 0 of a power-of-two stack: (G0+G1), (G2+G3), ... level by level."""
    while G.shape[0] > 1:
        G = G[0::2] + G[1::2]
    return G[0]


# ----------------------------------------------------------------------------- groups
class Grp:
    """A TP group: global ranks `ranks` (index i = TP rank i). pg is None for one rank."""

    def __init__(self, ranks: Sequence[int], pg=None):
        self.ranks = list(ranks)
        self.t = len(self.ranks)
        me = dist.get_rank() if dist.is_initialized() else 0
        self.member = me in self.ranks
        self.idx = self.ranks.index(me) if self.member else -1
        self.pg = pg

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(self.t, *x.shape, device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(out, x.contiguous(), group=self.pg)
        return out

    def __repr__(self):
        return f"Grp({self.ranks})"


def make_group(ranks: Sequence[int]) -> Grp:
    """Every rank must call this for every group, in the same order."""
    ranks = list(ranks)
    return Grp(ranks, dist.new_group(ranks) if len(ranks) > 1 else None)


def gather_cols(x: torch.Tensor, grp: Grp) -> torch.Tensor:
    """[M, n] on each rank -> [M, t*n], rank order (an exact copy)."""
    if grp.t == 1:
        return x
    G = grp.all_gather(x)
    return G.permute(1, 0, 2).reshape(x.shape[0], -1)


def lt_reduce(P: torch.Tensor, grp: Grp) -> torch.Tensor:
    """Logical-TP sum. P [C/t, M, N] fp32: this rank's chunks, which form one aligned subtree of the
    pairwise tree over all C chunks. Returns the bf16 sum [M, N], the same bits for every t | C."""
    sub = tree(P)
    if grp.t == 1:
        return sub.to(BF)
    t = grp.t
    M, N = sub.shape
    send = sub.view(M, t, N // t).permute(1, 0, 2).contiguous()           # [t, M, N/t]
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=grp.pg)                      # recv[s]: rank s's subtree
    part = tree(recv).to(BF)                                              # my N/t columns, rounded once
    return grp.all_gather(part).permute(1, 0, 2).reshape(M, N)


# ----------------------------------------------------------------------------- layers
_BOUND = threading.local()


def _bind_ctx():
    """Autograd runs CUDA backward on its own thread. A Triton launch there fails with "invalid device
    context" until a CUDA runtime call has made the primary context current; do one, once per thread."""
    if not getattr(_BOUND, "done", False):
        torch.cuda.synchronize()
        _BOUND.done = True


class _ColFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, lin):
        ctx.lin = lin
        return mm_one_config(x2, lin.wt)

    @staticmethod
    def backward(ctx, g):
        _bind_ctx()
        lin = ctx.lin
        if lin.bwd_chunks is None:
            raise RuntimeError("backward needs whole, unreplicated chunks on every rank")
        if lin.bwd_nccl:                                                  # Megatron's: bf16 partial, all-reduce
            dx = mm_one_config(g.contiguous(), lin.w)
            if lin.grp.t > 1:
                dist.all_reduce(dx, group=lin.grp.pg)
            return dx, None
        return lt_reduce(mm_chunks(g.contiguous(), lin.w, lin.bwd_chunks), lin.grp), None


class _RowFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2, lin):
        ctx.lin = lin
        return lin.reduce_inv(x2)

    @staticmethod
    def backward(ctx, g):
        _bind_ctx()
        return mm_one_config(g.contiguous(), ctx.lin.w), None            # dX_local = dY W_local: no sum


class ColLinear:
    """y_local = x @ W[rows]^T. w [n_loc, K] (cuBLAS layout, and the backward's B), wt [K, n_loc]."""

    def __init__(self, w_rows: torch.Tensor, grp: Grp, variant: str, bwd_chunks: Optional[int]):
        self.w = w_rows.detach().contiguous()
        self.wt = self.w.t().contiguous() if variant == "inv" else None
        self.grp, self.variant, self.bwd_chunks = grp, variant, bwd_chunks
        self.bwd_nccl = False

    def __call__(self, x):
        if self.variant == "stock":
            return F.linear(x, self.w)
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        if torch.is_grad_enabled() and x.requires_grad:
            y = _ColFn.apply(x2, self)
        else:
            y = mm_one_config(x2, self.wt)
        return y.view(*x.shape[:-1], self.w.shape[0])


class RowLinear:
    """y = sum over ranks of x_local @ W[:, cols]^T. w [N, k_loc] (cuBLAS layout, backward B), wt [k_loc, N]."""

    def __init__(self, w_cols: torch.Tensor, grp: Grp, variant: str, chunks: int):
        self.w = w_cols.detach().contiguous()
        self.wt = self.w.t().contiguous() if variant == "inv" else None
        self.grp, self.variant, self.chunks = grp, variant, chunks

    def reduce_inv(self, x2):
        return lt_reduce(mm_chunks(x2, self.wt, self.chunks), self.grp)

    def __call__(self, x):
        lead, K = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, K)
        if self.variant == "stock":
            y = F.linear(x2, self.w)
            if self.grp.t > 1:
                dist.all_reduce(y, group=self.grp.pg)
        elif torch.is_grad_enabled() and x.requires_grad:
            y = _RowFn.apply(x2, self)
        else:
            y = self.reduce_inv(x2.contiguous())
        return y.view(*lead, self.w.shape[0])


# ----------------------------------------------------------------------------- vocab-parallel log_softmax
def fixed_vocab_sum(e: torch.Tensor, grp: Grp, c_loc: int) -> torch.Tensor:
    """e [M, V/t] fp32 -> [M]: the sum over the whole vocab of Cv = t*c_loc fixed chunks. Each chunk by
    the fixed-tree row sum (order set by the chunk width), chunk sums by a pairwise tree."""
    M, Vl = e.shape
    w = Vl // c_loc
    s = wm.row_sum(e.contiguous().view(M * c_loc, w)).view(M, c_loc)
    if grp.t > 1:
        s = grp.all_gather(s).permute(1, 0, 2).reshape(M, -1)            # [M, Cv], chunk order
    return tree(s.t().contiguous())


def lsm_inv(x: torch.Tensor, grp: Grp, c_loc: int) -> torch.Tensor:
    xf = x.float()
    m = xf.amax(-1, keepdim=True)
    if grp.t > 1:
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=grp.pg)
    z = xf - m
    return z - torch.log(fixed_vocab_sum(torch.exp(z), grp, c_loc))[:, None]


def lsm_stock(x: torch.Tensor, grp: Grp) -> torch.Tensor:
    """As Megatron's vocab-parallel cross entropy: global max, local sum of exp, all-reduce SUM."""
    xf = x.float()
    m = xf.amax(-1, keepdim=True)
    if grp.t > 1:
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=grp.pg)
    z = xf - m
    s = torch.exp(z).sum(-1, keepdim=True)
    if grp.t > 1:
        dist.all_reduce(s, group=grp.pg)
    return z - torch.log(s)


class _LsmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grp, c_loc):
        lp = lsm_inv(x, grp, c_loc)
        ctx.save_for_backward(lp)
        ctx.grp, ctx.c_loc = grp, c_loc
        return lp

    @staticmethod
    def backward(ctx, g):
        _bind_ctx()
        (lp,) = ctx.saved_tensors
        sg = fixed_vocab_sum(g.float(), ctx.grp, ctx.c_loc)
        return (g - torch.exp(lp) * sg[:, None]).to(BF), None, None


# ----------------------------------------------------------------------------- the model
def heads_of(cfg: wm.Config, t: int, i: int):
    """(query heads, kv heads) of TP rank i at degree t."""
    hq = cfg.heads // t
    qh = list(range(i * hq, (i + 1) * hq))
    if cfg.kv_heads >= t:
        hk = cfg.kv_heads // t
        kh = list(range(i * hk, (i + 1) * hk))
    else:
        kh = [qh[0] // (cfg.heads // cfg.kv_heads)]
    return qh, kh


class TPModel:
    """One rank's shard of a model.py WholeModel. Duck-types the parts of WholeModel that
    examples/whole_model/whole_model.py's engine helpers use (cfg, ctx.dt, lm_head.weight, prefill,
    logprobs, forward_cached); cfg.kv_heads is the local count, so model.py's KVCache holds local heads."""

    def __init__(self, ref: wm.WholeModel, grp: Grp, variant: str, C: int = 16, Cv: int = 16, bwd_nccl: bool = False):
        cfg = ref.cfg
        assert not (cfg.moe or cfg.qkv_bias or cfg.tie_embeddings), "dense, untied, no bias only"
        assert variant in ("inv", "stock") and grp.member
        t, i = grp.t, grp.idx
        assert cfg.heads % t == 0 and C % t == 0 and Cv % t == 0 and cfg.hidden % t == 0
        assert cfg.intermediate % C == 0 and cfg.vocab_size % Cv == 0 and (cfg.heads * cfg.head_dim) % C == 0
        self.ref, self.grp, self.variant, self.C, self.Cv = ref, grp, variant, C, Cv
        self.opsname = "bi" if variant == "inv" else "stock"
        D = cfg.head_dim
        self.qh, self.kh = heads_of(cfg, t, i)
        self.nq, self.nk = len(self.qh), len(self.kh)
        rq = torch.cat([torch.arange(h * D, (h + 1) * D) for h in self.qh]).to(ref.lm_head.weight.device)
        rk = torch.cat([torch.arange(h * D, (h + 1) * D) for h in self.kh]).to(rq.device)
        il = cfg.intermediate // t
        isl = slice(i * il, (i + 1) * il)
        kv_bwd = cfg.kv_heads // t if cfg.kv_heads >= t else None
        self.layers = []
        for blk in ref.model.layers:
            a, m = blk.self_attn, blk.mlp
            self.layers.append(SimpleNamespace(
                q=ColLinear(a.q_proj.weight[rq], grp, variant, cfg.heads // t),
                k=ColLinear(a.k_proj.weight[rk], grp, variant, kv_bwd),
                v=ColLinear(a.v_proj.weight[rk], grp, variant, kv_bwd),
                o=RowLinear(a.o_proj.weight[:, rq], grp, variant, C // t),
                gate=ColLinear(m.gate_proj.weight[isl], grp, variant, C // t),
                up=ColLinear(m.up_proj.weight[isl], grp, variant, C // t),
                down=RowLinear(m.down_proj.weight[:, isl], grp, variant, C // t)))
        vl = cfg.vocab_size // t
        self.vlo = i * vl
        self.lm = ColLinear(ref.lm_head.weight[i * vl:(i + 1) * vl], grp, variant, Cv // t)
        if bwd_nccl:                      # negative control: column-parallel input grads by NCCL all-reduce
            for L in self.layers:
                for lin in (L.q, L.k, L.v, L.gate, L.up):
                    lin.bwd_nccl = True
            self.lm.bwd_nccl = True
        self.full_cfg = cfg
        self.cfg = replace(cfg, kv_heads=self.nk)              # for model.py's KVCache: local KV heads
        self.ctx = SimpleNamespace(dt=ref.ctx.dt)
        self.lm_head = SimpleNamespace(weight=self.lm.w)

    # ---- the layers
    def _block(self, li, s, pos, attn):
        blk, L = self.ref.model.layers[li], self.layers[li]
        lead, D = s.shape[:-1], self.full_cfg.head_dim
        a = blk.input_layernorm(s)
        q = L.q(a).view(*lead, self.nq, D)
        k = L.k(a).view(*lead, self.nk, D)
        v = L.v(a).view(*lead, self.nk, D)
        q, k = blk.self_attn.rope(q, k, pos)
        o = attn(li, q, k, v)
        s = blk.add(s, L.o(o.reshape(*lead, self.nq * D)))
        mm = blk.post_attention_layernorm(s)
        return blk.add(s, L.down(blk.mlp.act(L.gate(mm), L.up(mm))))

    def logits_local(self, ids, taps: Optional[list] = None):
        """Trainer forward: ids [B, T] -> this rank's vocab shard of the logits [B, T, V/t]. With
        `taps`, the residual stream entering each layer and the final norm is appended (for grads)."""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).expand(B, T)
        scale = self.full_cfg.head_dim ** -0.5
        with self.ref.using(ops=self.opsname):
            o = self.ref.ctx.ops
            s = self.ref.model.embed_tokens(ids)
            if taps is not None:
                s = s.detach().requires_grad_(True)
            for li in range(len(self.layers)):
                if taps is not None:
                    if s.grad_fn is not None:
                        s.retain_grad()
                    taps.append(s)
                s = self._block(li, s, pos, lambda i, q, k, v: o.attn_full(q, k, v, scale))
            if taps is not None:
                s.retain_grad()
                taps.append(s)
            return self.lm(self.ref.model.norm(s))

    def lsm_local(self, x_local):
        """Vocab-parallel log_softmax of [M, V/t] -> fp32 [M, V/t]."""
        if self.variant == "stock":
            return lsm_stock(x_local, self.grp)
        if torch.is_grad_enabled() and x_local.requires_grad:
            return _LsmFn.apply(x_local, self.grp, self.Cv // self.grp.t)
        return lsm_inv(x_local, self.grp, self.Cv // self.grp.t)

    # ---- the WholeModel surface used by whole_model.py's helpers (full, gathered tensors)
    @torch.no_grad()
    def prefill(self, ids, ops=None):
        B, T = ids.shape
        loc = self.logits_local(ids)
        return gather_cols(loc.reshape(B * T, -1), self.grp).view(B, T, -1)

    def logprobs(self, logits):
        """log_softmax of full (gathered) logits on one GPU, with the variant's formula at t = 1. For "inv"
        this is bitwise the vocab-parallel result at any degree (checked in the tp section)."""
        one = Grp([dist.get_rank()]) if dist.is_initialized() else Grp([0])
        lead = logits.shape[:-1]
        x = logits.reshape(-1, logits.shape[-1])
        lp = lsm_stock(x, one) if self.variant == "stock" else lsm_inv(x, one, self.Cv)
        return lp.view(*lead, -1)

    @torch.no_grad()
    def forward_cached(self, cache, slots, tokens, ops=None) -> List[torch.Tensor]:
        """model.py's forward_cached on the shard: the cache holds this rank's KV heads."""
        assert len(set(slots)) == len(slots) == len(tokens)
        counts = [int(t.numel()) for t in tokens]
        pos0 = [cache.length[s] for s in slots]
        assert all(c >= 1 for c in counts) and max(p + c for p, c in zip(pos0, counts)) <= cache.capacity
        dev = cache.k.device
        ids = torch.cat([t.reshape(-1) for t in tokens]).to(dev)
        cnt = torch.as_tensor(counts, device=dev)
        row = torch.repeat_interleave(torch.arange(len(slots), device=dev), cnt)
        first = torch.cumsum(cnt, 0) - cnt
        pos = torch.as_tensor(pos0, device=dev)[row] + (torch.arange(ids.numel(), device=dev) - first[row])
        row_slot = torch.as_tensor(list(slots), device=dev)[row]
        scale = self.full_cfg.head_dim ** -0.5
        with self.ref.using(ops=self.opsname):
            o = self.ref.ctx.ops

            def attn(i, q, k, v):
                cache.k[i][row_slot, pos] = k
                cache.v[i][row_slot, pos] = v
                return o.attn_cached(q, cache.k[i], cache.v[i], list(slots), counts, pos0, scale)
            s = self.ref.model.embed_tokens(ids)
            for li in range(len(self.layers)):
                s = self._block(li, s, pos, attn)
            loc = self.lm(self.ref.model.norm(s))
        for sl, c in zip(slots, counts):
            cache.length[sl] += c
        return list(torch.split(gather_cols(loc, self.grp), counts))
