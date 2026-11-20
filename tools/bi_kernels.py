"""Reference batch-invariant kernels (Triton), small and untuned on purpose.

mm_one_config(a, b)
    bf16 GEMM with one tile configuration for every M: fixed 64x64x32 tiles,
    K walked in ascending order into an fp32 accumulator, no split-K, and no
    specialization on M. A row's result cannot depend on the batch.

attention_fixed_blocks(q, k, v, start)
    Causal attention forward for queries at absolute positions start..start+Tq-1
    against keys 0..start+Tq-1 (the KV cache plus the current chunk). Every query
    visits KV in blocks of a fixed size from absolute position 0, with an
    online-softmax update per block, and no split-KV. Prefill (start=0, all
    queries), chunked prefill and one-token decode therefore run the same
    arithmetic per position. Blocks past a query's position are fully masked and
    are exact no-ops (scale 1, added probability 0). GQA by head grouping.
    With `lse` given, it also stores each query row's log-sum-exp (log2 units)
    for the backward; that is a compile-time variant whose output is checked
    bitwise against the plain forward.

attention_fixed_blocks_train(q, k, v)
    Full causal attention (start=0) with a deterministic, batch-invariant
    backward and no atomics. dK/dV: one program per (sequence, KV head, key
    block) walks the GQA group's query heads, then the query blocks, in a fixed
    order, accumulating in registers. dQ: one program per (sequence, query head,
    query block) walks key blocks from position 0. Delta = rowsum(dO * O) is a
    row-wise reduction in its own kernel. Nothing depends on the batch size or
    on other sequences.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

BF = torch.bfloat16
LOG2E = 1.4426950408889634


@triton.jit(do_not_specialize=["M"])
def _mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.bfloat16),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def mm_one_config(a: torch.Tensor, b: torch.Tensor, BM=64, BN=64, BK=32) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=BF)
    if M:
        _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
            a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return c


@triton.jit(do_not_specialize=["start", "Tq", "Tk"])
def _attn_kernel(Q, K, V, O, LSE, scale_log2, start, Tq, Tk, Hq,
                 sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sob, soh, sot,
                 GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                 STORE_LSE: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // Hq, bh % Hq
    kh = h // GROUP
    rm = pid_m * BM + tl.arange(0, BM)
    pos = start + rm                                  # absolute query positions
    rd = tl.arange(0, D)
    q = tl.load(Q + b * sqb + h * sqh + rm[:, None] * sqt + rd[None, :], mask=rm[:, None] < Tq, other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    hi = tl.minimum(start + (pid_m + 1) * BM, Tk)     # keys any row of this block may see
    for n0 in range(0, hi, BN):                       # fixed blocks from absolute position 0
        rn = n0 + tl.arange(0, BN)
        k = tl.load(K + b * skb + kh * skh + rn[:, None] * skt + rd[None, :], mask=rn[:, None] < Tk, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        s = tl.where((rn[None, :] <= pos[:, None]) & (rn[None, :] < Tk), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + b * svb + kh * svh + rn[:, None] * svt + rd[None, :], mask=rn[:, None] < Tk, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(O + b * sob + h * soh + rm[:, None] * sot + rd[None, :], o.to(tl.bfloat16), mask=rm[:, None] < Tq)
    if STORE_LSE:                                     # log2-domain log-sum-exp, [B*Hq, Tq]
        tl.store(LSE + bh * Tq + rm, m_i + tl.log2(l_i), mask=rm < Tq)


def attention_fixed_blocks(q, k, v, start: int, BM=64, BN=64, lse=None) -> torch.Tensor:
    """q [B, Hq, Tq, D]; k, v [B, Hk, >= start+Tq, D] (last dim contiguous).
    lse: optional fp32 [B*Hq, Tq] to receive the log2-domain log-sum-exp."""
    B, Hq, Tq, D = q.shape
    Hk = k.shape[1]
    o = torch.empty_like(q)
    _attn_kernel[(triton.cdiv(Tq, BM), B * Hq)](
        q, k, v, o, o if lse is None else lse, LOG2E / math.sqrt(D), start, Tq, start + Tq, Hq,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
        GROUP=Hq // Hk, BM=BM, BN=BN, D=D, STORE_LSE=lse is not None, num_warps=4, num_stages=2)
    return o


# ----------------------------------------------------------------------------- backward
# Layouts: q, o, do, dq contiguous [B, Hq, T, D]; k, v, dk, dv contiguous [B, Hk, T, D];
# LSE and Delta contiguous fp32 [B*Hq, T]. Masked score entries give p = 0 exactly.

@triton.jit(do_not_specialize=["T"])
def _attn_bwd_delta(O, DO, Delta, T, D: tl.constexpr, BM: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    off = bh * T * D + rm[:, None] * D + rd[None, :]
    o = tl.load(O + off, mask=rm[:, None] < T, other=0.0).to(tl.float32)
    do = tl.load(DO + off, mask=rm[:, None] < T, other=0.0).to(tl.float32)
    tl.store(Delta + bh * T + rm, tl.sum(o * do, 1), mask=rm < T)


@triton.jit(do_not_specialize=["T"])
def _attn_bwd_dkdv(Q, K, V, DO, LSE, Delta, DK, DV, sm_scale, scale_log2, T, Hq, Hk,
                   GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_n, bkh = tl.program_id(0), tl.program_id(1)
    b, kh = bkh // Hk, bkh % Hk
    rn = pid_n * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    kv_off = bkh * T * D + rn[:, None] * D + rd[None, :]
    k = tl.load(K + kv_off, mask=rn[:, None] < T, other=0.0)
    v = tl.load(V + kv_off, mask=rn[:, None] < T, other=0.0)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    lo = (pid_n * BN // BM) * BM                      # first query block that sees these keys
    for g in tl.static_range(GROUP):                  # fixed head order inside the GQA group
        bh = b * Hq + kh * GROUP + g
        for m0 in range(lo, T, BM):                   # fixed query-block order
            rm = m0 + tl.arange(0, BM)
            q_off = bh * T * D + rm[:, None] * D + rd[None, :]
            q = tl.load(Q + q_off, mask=rm[:, None] < T, other=0.0)
            do = tl.load(DO + q_off, mask=rm[:, None] < T, other=0.0)
            lse = tl.load(LSE + bh * T + rm, mask=rm < T, other=0.0)
            delta = tl.load(Delta + bh * T + rm, mask=rm < T, other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale_log2
            valid = (rn[None, :] <= rm[:, None]) & (rm[:, None] < T) & (rn[None, :] < T)
            p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
            dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q)
    tl.store(DK + kv_off, (dk * sm_scale).to(tl.bfloat16), mask=rn[:, None] < T)
    tl.store(DV + kv_off, dv.to(tl.bfloat16), mask=rn[:, None] < T)


@triton.jit(do_not_specialize=["T"])
def _attn_bwd_dq(Q, K, V, DO, LSE, Delta, DQ, sm_scale, scale_log2, T, Hq, Hk,
                 GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // Hq, bh % Hq
    bkh = b * Hk + h // GROUP
    rm = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    q_off = bh * T * D + rm[:, None] * D + rd[None, :]
    q = tl.load(Q + q_off, mask=rm[:, None] < T, other=0.0)
    do = tl.load(DO + q_off, mask=rm[:, None] < T, other=0.0)
    lse = tl.load(LSE + bh * T + rm, mask=rm < T, other=0.0)
    delta = tl.load(Delta + bh * T + rm, mask=rm < T, other=0.0)
    dq = tl.zeros([BM, D], tl.float32)
    hi = tl.minimum((pid_m + 1) * BM, T)
    for n0 in range(0, hi, BN):                       # key blocks from absolute position 0
        rn = n0 + tl.arange(0, BN)
        kv_off = bkh * T * D + rn[:, None] * D + rd[None, :]
        k = tl.load(K + kv_off, mask=rn[:, None] < T, other=0.0)
        v = tl.load(V + kv_off, mask=rn[:, None] < T, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        valid = (rn[None, :] <= rm[:, None]) & (rm[:, None] < T) & (rn[None, :] < T)
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.bfloat16), k)
    tl.store(DQ + q_off, (dq * sm_scale).to(tl.bfloat16), mask=rm[:, None] < T)


def attention_fixed_blocks_backward(q, k, v, o, lse, do, BM=64, BN=64):
    """Gradients of full causal attention (start=0). All tensors contiguous."""
    B, Hq, T, D = q.shape
    Hk = k.shape[1]
    sm_scale = 1.0 / math.sqrt(D)
    delta = torch.empty(B * Hq, T, device=q.device, dtype=torch.float32)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _attn_bwd_delta[(triton.cdiv(T, BM), B * Hq)](o, do, delta, T, D=D, BM=BM, num_warps=4)
    _attn_bwd_dkdv[(triton.cdiv(T, BN), B * Hk)](
        q, k, v, do, lse, delta, dk, dv, sm_scale, sm_scale * LOG2E, T, Hq, Hk,
        GROUP=Hq // Hk, BM=BM, BN=BN, D=D, num_warps=8, num_stages=2)
    _attn_bwd_dq[(triton.cdiv(T, BM), B * Hq)](
        q, k, v, do, lse, delta, dq, sm_scale, sm_scale * LOG2E, T, Hq, Hk,
        GROUP=Hq // Hk, BM=BM, BN=BN, D=D, num_warps=4, num_stages=2)
    return dq, dk, dv


class _FixedBlockAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        lse = torch.empty(q.shape[0] * q.shape[1], q.shape[2], device=q.device, dtype=torch.float32)
        o = attention_fixed_blocks(q, k, v, 0, lse=lse)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        return attention_fixed_blocks_backward(q, k, v, o, lse, do.contiguous())


def attention_fixed_blocks_train(q, k, v) -> torch.Tensor:
    """Causal attention for training: q [B, Hq, T, D], k, v [B, Hk, T, D]."""
    return _FixedBlockAttention.apply(q, k, v)
