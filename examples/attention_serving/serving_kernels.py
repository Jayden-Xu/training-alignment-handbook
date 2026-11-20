"""Serving-side layouts for the fixed-block attention (Triton): paged KV, varlen
packing, an atomic-free varlen backward, and opaque custom ops for deployment
wrappers. Small and untuned on purpose; extends tools/bi_kernels.py without
editing it.

Layouts. Queries are packed: q [NQ, Hq, D], sequence b owns rows
cu_q[b]..cu_q[b+1]-1, and its first query sits at absolute position pos0[b]
(the length of its cached context). Its KV length is pos0[b] + q_len[b].

attention_paged(q, k_cache, v_cache, block_table, cu_q, pos0, max_q, BN=64)
    K/V are read through a block table: k_cache [num_pages, P, Hk, D], and
    position t of sequence b lives in page block_table[b, t // P], slot t % P.
    Compute blocks are BN = 64 tokens aligned to absolute position 0, whatever
    the page size P: a block gathers from several pages (P < 64) or a page holds
    several blocks (P > 64). The arithmetic per query row is that of
    bi_kernels.attention_fixed_blocks. Passing BN=P gives the contrast where the
    compute block follows the page size.

attention_varlen(q, k, v, cu_q, cu_k, pos0, max_q, align_pack=False)
    K/V packed: k [NK, Hk, D], sequence b owns rows cu_k[b]..cu_k[b+1]-1.
    Blocks start at the sequence's own position 0, independent of where the
    sequence sits in the pack. align_pack=True is the naive contrast: blocks are
    aligned to multiples of 64 in pack coordinates, so the first block of a
    sequence also covers (masked) rows of its neighbour.

attention_varlen_train(q, k, v, cu, max_len)
    Packed self-attention (pos0 = 0, cu_q = cu_k) with the atomic-free two-pass
    backward of bi_kernels, one program per (sequence, head, block); query and key
    blocks are aligned to each sequence's start.

Masked loads return 0 and are never issued past a sequence's KV length, so
unwritten cache slots (even NaN) are never read.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from bi_kernels import BF, LOG2E, _attn_kernel, mm_one_config  # noqa: E402

MODE_VARLEN, MODE_PAGED, MODE_PACK_ALIGNED = 0, 1, 2


@triton.jit(do_not_specialize=["NQ"])
def _serve_fwd(Q, K, V, O, LSE, BT, CUQ, CUK, POS0, scale_log2, Hq, NQ,
               sq_t, sq_h, sk_p, sk_t, sk_h, sv_p, sv_t, sv_h, so_t, so_h, s_bt,
               GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
               PAGE: tl.constexpr, MODE: tl.constexpr, STORE_LSE: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // Hq, bh % Hq
    kh = h // GROUP
    q0 = tl.load(CUQ + b)
    q_len = tl.load(CUQ + b + 1) - q0
    p0 = tl.load(POS0 + b)                            # absolute position of the first query
    kv_len = p0 + q_len
    rm = pid_m * BM + tl.arange(0, BM)
    pos = p0 + rm
    rd = tl.arange(0, D)
    q = tl.load(Q + (q0 + rm[:, None]) * sq_t + h * sq_h + rd[None, :], mask=rm[:, None] < q_len, other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    hi = tl.minimum(p0 + (pid_m + 1) * BM, kv_len)    # keys any row of this block may see
    hi = tl.where(pid_m * BM < q_len, hi, 0)          # programs past the sequence do nothing
    k0 = 0
    if MODE != 1:
        k0 = tl.load(CUK + b)                         # sequence start in the KV pack
    lo = 0
    if MODE == 2:
        lo = -(k0 % BN)                               # naive: block grid of the pack
    for n0 in range(lo, hi, BN):                      # fixed blocks from absolute position 0
        rn = n0 + tl.arange(0, BN)
        kvm = rn < kv_len
        if MODE == 2:
            kvm = kvm & (rn >= 0)
        if MODE == 1:
            if PAGE >= BN:                            # the whole block sits in one page
                pg = tl.load(BT + b * s_bt + n0 // PAGE).to(tl.int64)
                slot = n0 % PAGE + tl.arange(0, BN)
            else:                                     # the block gathers from BN // PAGE pages
                pg = tl.load(BT + b * s_bt + rn // PAGE, mask=kvm, other=0).to(tl.int64)
                slot = rn % PAGE
            k_row = pg * sk_p + slot * sk_t
            v_row = pg * sv_p + slot * sv_t
        else:
            k_row = (k0 + rn) * sk_t
            v_row = (k0 + rn) * sv_t
        k = tl.load(K + k_row[:, None] + kh * sk_h + rd[None, :], mask=kvm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        s = tl.where((rn[None, :] <= pos[:, None]) & kvm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + v_row[:, None] + kh * sv_h + rd[None, :], mask=kvm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(O + (q0 + rm[:, None]) * so_t + h * so_h + rd[None, :], o.to(tl.bfloat16), mask=rm[:, None] < q_len)
    if STORE_LSE:                                     # log2-domain log-sum-exp, [Hq, NQ]
        tl.store(LSE + h * NQ + q0 + rm, m_i + tl.log2(l_i), mask=rm < q_len)


def _launch(q, k, v, cu_q, pos0, max_q, mode, block_table=None, cu_k=None, page=64, BN=64, BM=64,
            lse=None, num_stages=2):
    NQ, Hq, D = q.shape
    Hk = k.shape[-2]
    B = cu_q.numel() - 1
    o = torch.empty(NQ, Hq, D, device=q.device, dtype=q.dtype)
    if mode == MODE_PAGED:
        sk_p, sk_t, sk_h = k.stride(0), k.stride(1), k.stride(2)
        sv_p, sv_t, sv_h = v.stride(0), v.stride(1), v.stride(2)
        bt, s_bt, cuk = block_table, block_table.stride(0), cu_q
    else:
        sk_p, sk_t, sk_h = 0, k.stride(0), k.stride(1)
        sv_p, sv_t, sv_h = 0, v.stride(0), v.stride(1)
        bt, s_bt, cuk = cu_q, 0, cu_k
    if max_q > 0:
        _serve_fwd[(triton.cdiv(max_q, BM), B * Hq)](
            q, k, v, o, o if lse is None else lse, bt, cu_q, cuk, pos0, LOG2E / math.sqrt(D), Hq, NQ,
            q.stride(0), q.stride(1), sk_p, sk_t, sk_h, sv_p, sv_t, sv_h, o.stride(0), o.stride(1), s_bt,
            GROUP=Hq // Hk, BM=BM, BN=BN, D=D, PAGE=page, MODE=mode, STORE_LSE=lse is not None,
            num_warps=4, num_stages=num_stages)
    return o


def attention_paged(q, k_cache, v_cache, block_table, cu_q, pos0, max_q, BN=64, lse=None, num_stages=2):
    """q [NQ, Hq, D]; k_cache, v_cache [num_pages, P, Hk, D]; block_table int32 [B, max_pages];
    cu_q int32 [B+1]; pos0 int32 [B]; max_q = max query rows of any sequence (host int)."""
    return _launch(q, k_cache, v_cache, cu_q, pos0, max_q, MODE_PAGED, block_table=block_table,
                   page=k_cache.shape[1], BN=BN, lse=lse, num_stages=num_stages)


def attention_varlen(q, k, v, cu_q, cu_k, pos0, max_q, align_pack=False, lse=None):
    """q [NQ, Hq, D]; k, v [NK, Hk, D]; cu_q, cu_k int32 [B+1]; pos0 int32 [B];
    cu_k[b+1] - cu_k[b] must equal pos0[b] + q_len[b]."""
    return _launch(q, k, v, cu_q, pos0, max_q, MODE_PACK_ALIGNED if align_pack else MODE_VARLEN,
                   cu_k=cu_k, lse=lse)


# ----------------------------------------------------------------------------- varlen backward
# Packed layouts: q, o, do, dq [NT, Hq, D]; k, v, dk, dv [NT, Hk, D]; LSE, Delta fp32 [Hq, NT].
# Self-attention, pos0 = 0. Every loop runs over one sequence's rows, with blocks
# aligned to that sequence's start, and rows past its length load as zeros.

@triton.jit(do_not_specialize=["NT"])
def _vl_bwd_delta(O, DO, Delta, CU, NT, Hq, s_t, s_h, D: tl.constexpr, BM: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // Hq, bh % Hq
    c0 = tl.load(CU + b)
    T = tl.load(CU + b + 1) - c0
    rm = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    off = (c0 + rm[:, None]) * s_t + h * s_h + rd[None, :]
    o = tl.load(O + off, mask=rm[:, None] < T, other=0.0).to(tl.float32)
    do = tl.load(DO + off, mask=rm[:, None] < T, other=0.0).to(tl.float32)
    tl.store(Delta + h * NT + c0 + rm, tl.sum(o * do, 1), mask=rm < T)


@triton.jit(do_not_specialize=["NT"])
def _vl_bwd_dkdv(Q, K, V, DO, LSE, Delta, DK, DV, CU, sm_scale, scale_log2, NT, Hq, Hk,
                 sq_t, sq_h, sk_t, sk_h,
                 GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_n, bkh = tl.program_id(0), tl.program_id(1)
    b, kh = bkh // Hk, bkh % Hk
    c0 = tl.load(CU + b)
    T = tl.load(CU + b + 1) - c0
    rn = pid_n * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    kv_off = (c0 + rn[:, None]) * sk_t + kh * sk_h + rd[None, :]
    k = tl.load(K + kv_off, mask=rn[:, None] < T, other=0.0)
    v = tl.load(V + kv_off, mask=rn[:, None] < T, other=0.0)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    lo = (pid_n * BN // BM) * BM                      # first query block that sees these keys
    for g in tl.static_range(GROUP):                  # fixed head order inside the GQA group
        h = kh * GROUP + g
        for m0 in range(lo, T, BM):                   # fixed query-block order, from the sequence start
            rm = m0 + tl.arange(0, BM)
            q_off = (c0 + rm[:, None]) * sq_t + h * sq_h + rd[None, :]
            q = tl.load(Q + q_off, mask=rm[:, None] < T, other=0.0)
            do = tl.load(DO + q_off, mask=rm[:, None] < T, other=0.0)
            lse = tl.load(LSE + h * NT + c0 + rm, mask=rm < T, other=0.0)
            delta = tl.load(Delta + h * NT + c0 + rm, mask=rm < T, other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale_log2
            valid = (rn[None, :] <= rm[:, None]) & (rm[:, None] < T) & (rn[None, :] < T)
            p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
            dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q)
    tl.store(DK + kv_off, (dk * sm_scale).to(tl.bfloat16), mask=rn[:, None] < T)
    tl.store(DV + kv_off, dv.to(tl.bfloat16), mask=rn[:, None] < T)


@triton.jit(do_not_specialize=["NT"])
def _vl_bwd_dq(Q, K, V, DO, LSE, Delta, DQ, CU, sm_scale, scale_log2, NT, Hq, Hk,
               sq_t, sq_h, sk_t, sk_h,
               GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // Hq, bh % Hq
    kh = h // GROUP
    c0 = tl.load(CU + b)
    T = tl.load(CU + b + 1) - c0
    rm = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    q_off = (c0 + rm[:, None]) * sq_t + h * sq_h + rd[None, :]
    q = tl.load(Q + q_off, mask=rm[:, None] < T, other=0.0)
    do = tl.load(DO + q_off, mask=rm[:, None] < T, other=0.0)
    lse = tl.load(LSE + h * NT + c0 + rm, mask=rm < T, other=0.0)
    delta = tl.load(Delta + h * NT + c0 + rm, mask=rm < T, other=0.0)
    dq = tl.zeros([BM, D], tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, T)
    for n0 in range(0, hi, BN):                       # key blocks from the sequence's position 0
        rn = n0 + tl.arange(0, BN)
        kv_off = (c0 + rn[:, None]) * sk_t + kh * sk_h + rd[None, :]
        k = tl.load(K + kv_off, mask=rn[:, None] < T, other=0.0)
        v = tl.load(V + kv_off, mask=rn[:, None] < T, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        valid = (rn[None, :] <= rm[:, None]) & (rm[:, None] < T) & (rn[None, :] < T)
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.bfloat16), k)
    tl.store(DQ + q_off, (dq * sm_scale).to(tl.bfloat16), mask=rm[:, None] < T)


def attention_varlen_backward(q, k, v, o, lse, do, cu, max_len, BM=64, BN=64):
    """Gradients of packed causal self-attention. All tensors contiguous."""
    NT, Hq, D = q.shape
    Hk = k.shape[1]
    B = cu.numel() - 1
    sm_scale = 1.0 / math.sqrt(D)
    delta = torch.empty(Hq, NT, device=q.device, dtype=torch.float32)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _vl_bwd_delta[(triton.cdiv(max_len, BM), B * Hq)](o, do, delta, cu, NT, Hq, o.stride(0), o.stride(1),
                                                     D=D, BM=BM, num_warps=4)
    _vl_bwd_dkdv[(triton.cdiv(max_len, BN), B * Hk)](
        q, k, v, do, lse, delta, dk, dv, cu, sm_scale, sm_scale * LOG2E, NT, Hq, Hk,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        GROUP=Hq // Hk, BM=BM, BN=BN, D=D, num_warps=8, num_stages=2)
    _vl_bwd_dq[(triton.cdiv(max_len, BM), B * Hq)](
        q, k, v, do, lse, delta, dq, cu, sm_scale, sm_scale * LOG2E, NT, Hq, Hk,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        GROUP=Hq // Hk, BM=BM, BN=BN, D=D, num_warps=4, num_stages=2)
    return dq, dk, dv


class _VarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, max_len):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        lse = torch.empty(q.shape[1], q.shape[0], device=q.device, dtype=torch.float32)
        pos0 = torch.zeros(cu.numel() - 1, device=q.device, dtype=torch.int32)
        o = attention_varlen(q, k, v, cu, cu, pos0, max_len, lse=lse)
        ctx.save_for_backward(q, k, v, o, lse, cu)
        ctx.max_len = max_len
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, cu = ctx.saved_tensors
        dq, dk, dv = attention_varlen_backward(q, k, v, o, lse, do.contiguous(), cu, ctx.max_len)
        return dq, dk, dv, None, None


def attention_varlen_train(q, k, v, cu, max_len) -> torch.Tensor:
    """Packed causal self-attention for training: q [NT, Hq, D], k, v [NT, Hk, D]."""
    return _VarlenAttention.apply(q, k, v, cu, max_len)


# ----------------------------------------------------------------------------- opaque custom ops
def attention_fixed_blocks_out(q, k, v, start: int, BM=64, BN=64) -> torch.Tensor:
    """bi_kernels.attention_fixed_blocks with a contiguous output (same kernel, same
    arguments); a custom op's fake and real outputs must agree on strides."""
    B, Hq, Tq, D = q.shape
    Hk = k.shape[1]
    o = torch.empty(B, Hq, Tq, D, device=q.device, dtype=q.dtype)
    _attn_kernel[(triton.cdiv(Tq, BM), B * Hq)](
        q, k, v, o, o, LOG2E / math.sqrt(D), start, Tq, start + Tq, Hq,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
        GROUP=Hq // Hk, BM=BM, BN=BN, D=D, STORE_LSE=False, num_warps=4, num_stages=2)
    return o


@torch.library.custom_op("bi_serving::mm", mutates_args=())
def bi_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return mm_one_config(a, b)


@bi_mm.register_fake
def _(a, b):
    return a.new_empty(a.shape[0], b.shape[1])


@torch.library.custom_op("bi_serving::attn", mutates_args=())
def bi_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int) -> torch.Tensor:
    return attention_fixed_blocks_out(q, k, v, start)


@bi_attn.register_fake
def _(q, k, v, start):
    return q.new_empty(q.shape)


def rmsnorm_torch(x, g, eps: float = 1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(BF) * g


def silu_mul_torch(gu, inter: int):
    return torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:]


@torch.library.custom_op("bi_serving::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return rmsnorm_torch(x, g)


@rmsnorm_op.register_fake
def _(x, g):
    return x.new_empty(x.shape)


@torch.library.custom_op("bi_serving::silu_mul", mutates_args=())
def silu_mul_op(gu: torch.Tensor, inter: int) -> torch.Tensor:
    return silu_mul_torch(gu, inter)


@silu_mul_op.register_fake
def _(gu, inter):
    return gu.new_empty(gu.shape[0], inter)
