"""Batch-invariant Triton kernels for MLA and DSA-like sparse attention.

Small and untuned on purpose, like tools/bi_kernels.py.

mm_grouped(a, b)
    G independent GEMMs in one launch: c[:, g] = a[:, g] @ b[g]. The body is
    tools/bi_kernels.mm_one_config's: fixed 64x64x32 tiles, K ascending into an
    fp32 accumulator, no split-K, no specialization on M. Used for MLA's per-head
    absorption (q_nope @ W_UK^T and o_latent @ W_UV).

rmsnorm(x, w)
    One row per program, the whole row in one block, a fixed in-block tree.

mla_attention(qn, qr, kn, kr, v, start)
    Causal attention whose score is split into two dot products,
    s = qn . kn + qr . kr, so head dims need not be a power of two:
    explicit MLA uses 128 + 64 (V 128), absorbed MLA uses 512 + 64 (V = the
    512-dim latent). The rope key kr is shared by all heads. Rows of a program
    are (position, head) pairs: G query heads share one KV head, so the absorbed
    form (G = 16, one latent KV head) loads each latent block once for all heads.
    KV is visited in fixed BN-token blocks from absolute position 0 with an
    online-softmax update per block, as in bi_kernels.attention_fixed_blocks.
    Blocks past a row's position are exact no-ops.

indexer_scores(qi, ki, w, start)
    DSA-like indexer: score[t, s] = sum_h w[t, h] * relu(qi[t, h] . ki[s]),
    fp32, with the heads summed in a fixed order inside the program; keys after
    the query's absolute position get -inf.

sparse_attention(q, k, v, idx, start)
    Attention of each query over its selected keys only. idx holds absolute key
    positions in ascending order, -1 (invalid) at the end. The selected keys are
    visited in that order in fixed BN-key blocks with an online softmax. All
    query heads of a token share one selection and one KV head (MQA).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

BF = torch.bfloat16
LOG2E = 1.4426950408889634


# ----------------------------------------------------------------------------- grouped GEMM
@triton.jit(do_not_specialize=["M"])
def _mm_grouped_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sag, sam, sak, sbg, sbk, sbn, scg, scm, scn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n, g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    a_ptr += g * sag
    b_ptr += g * sbg
    c_ptr += g * scg
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


def mm_grouped(a: torch.Tensor, b: torch.Tensor, BM=64, BN=64, BK=32) -> torch.Tensor:
    """a [M, G, K] (any strides), b [G, K, N] -> c [M, G, N] bf16 with c[:, g] = a[:, g] @ b[g]."""
    M, G, K = a.shape
    N = b.shape[2]
    c = torch.empty(M, G, N, device=a.device, dtype=BF)
    if M:
        _mm_grouped_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), G)](
            a, b, c, M, N, K, a.stride(1), a.stride(0), a.stride(2), b.stride(0), b.stride(1), b.stride(2),
            c.stride(1), c.stride(0), c.stride(2), BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return c


# ----------------------------------------------------------------------------- RMSNorm
@triton.jit
def _rmsnorm_kernel(X, W, Y, sx, sy, eps, N: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.arange(0, N)
    x = tl.load(X + r * sx + cols).to(tl.float32)
    var = tl.sum(x * x, 0) / N
    y = (x * tl.rsqrt(var + eps)).to(tl.bfloat16)
    w = tl.load(W + cols)
    tl.store(Y + r * sy + cols, (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16))


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """x [R, N] with unit last-dim stride, N a power of two."""
    R, N = x.shape
    y = torch.empty(R, N, device=x.device, dtype=BF)
    if R:
        _rmsnorm_kernel[(R,)](x, w, y, x.stride(0), y.stride(0), eps, N=N, num_warps=4)
    return y


# ----------------------------------------------------------------------------- MLA attention
@triton.jit(do_not_specialize=["start", "Tq", "Tk"])
def _mla_attn_kernel(QN, QR, KN, KR, V, O, scale_log2, start, Tq, Tk, Hkv,
                     sqn_b, sqn_t, sqn_h, sqr_b, sqr_t, sqr_h,
                     skn_b, skn_t, skn_h, skr_b, skr_t,
                     sv_b, sv_t, sv_h, so_b, so_t, so_h,
                     G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     DN: tl.constexpr, DR: tl.constexpr, DV: tl.constexpr, V_IS_KN: tl.constexpr):
    pid_m, bk = tl.program_id(0), tl.program_id(1)
    b, kvh = bk // Hkv, bk % Hkv
    r = pid_m * BM + tl.arange(0, BM)                 # row = t * G + g
    t = r // G
    h = kvh * G + r % G
    pos = start + t                                   # absolute query positions
    row_ok = t < Tq
    rdn = tl.arange(0, DN)
    rdr = tl.arange(0, DR)
    rdv = tl.arange(0, DV)
    qn = tl.load(QN + b * sqn_b + t[:, None] * sqn_t + h[:, None] * sqn_h + rdn[None, :],
                 mask=row_ok[:, None], other=0.0)
    qr = tl.load(QR + b * sqr_b + t[:, None] * sqr_t + h[:, None] * sqr_h + rdr[None, :],
                 mask=row_ok[:, None], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, DV], tl.float32)
    t_last = (pid_m * BM + BM - 1) // G               # last position any row of this block holds
    hi = tl.minimum(start + t_last + 1, Tk)
    for n0 in range(0, hi, BN):                       # fixed blocks from absolute position 0
        rn = n0 + tl.arange(0, BN)
        kmask = rn[:, None] < Tk
        kn = tl.load(KN + b * skn_b + rn[:, None] * skn_t + kvh * skn_h + rdn[None, :], mask=kmask, other=0.0)
        kr = tl.load(KR + b * skr_b + rn[:, None] * skr_t + rdr[None, :], mask=kmask, other=0.0)
        s = tl.dot(qn, tl.trans(kn))                  # nope part, then rope part: fixed order
        s = tl.dot(qr, tl.trans(kr), s)
        s = s * scale_log2
        s = tl.where((rn[None, :] <= pos[:, None]) & (rn[None, :] < Tk), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if V_IS_KN:
            v = kn
        else:
            v = tl.load(V + b * sv_b + rn[:, None] * sv_t + kvh * sv_h + rdv[None, :], mask=kmask, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(O + b * so_b + t[:, None] * so_t + h[:, None] * so_h + rdv[None, :], o.to(tl.bfloat16),
             mask=row_ok[:, None])


def mla_attention(qn, qr, kn, kr, v, start: int, sm_scale: float, BM=64, BN=64, num_warps=4, num_stages=2):
    """qn [B, Tq, Hq, DN], qr [B, Tq, Hq, DR] queries at absolute positions start..start+Tq-1;
    kn [B, Tk, Hkv, DN], kr [B, Tk, DR] (shared by all heads), v [B, Tk, Hkv, DV] or None
    (then V is kn, the absorbed form). Tk = start + Tq. Last dims contiguous. Returns
    [B, Tq, Hq, DV] bf16."""
    B, Tq, Hq, DN = qn.shape
    DR = qr.shape[-1]
    Hkv = kn.shape[2]
    G = Hq // Hkv
    v_is_kn = v is None
    vv = kn if v_is_kn else v
    DV = vv.shape[-1]
    Tk = start + Tq
    o = torch.empty(B, Tq, Hq, DV, device=qn.device, dtype=BF)
    _mla_attn_kernel[(triton.cdiv(Tq * G, BM), B * Hkv)](
        qn, qr, kn, kr, vv, o, sm_scale * LOG2E, start, Tq, Tk, Hkv,
        qn.stride(0), qn.stride(1), qn.stride(2), qr.stride(0), qr.stride(1), qr.stride(2),
        kn.stride(0), kn.stride(1), kn.stride(2), kr.stride(0), kr.stride(1),
        vv.stride(0), vv.stride(1), vv.stride(2), o.stride(0), o.stride(1), o.stride(2),
        G=G, BM=BM, BN=BN, DN=DN, DR=DR, DV=DV, V_IS_KN=v_is_kn, num_warps=num_warps, num_stages=num_stages)
    return o


# ----------------------------------------------------------------------------- DSA indexer
@triton.jit(do_not_specialize=["start", "Tq", "Tk"])
def _indexer_kernel(QI, KI, W, S, start, Tq, Tk,
                    sq_b, sq_t, sq_h, sk_b, sk_t, sw_b, sw_t, ss_b, ss_t,
                    HI: tl.constexpr, DI: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m, pid_n, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rd = tl.arange(0, DI)
    pos = start + rm
    k = tl.load(KI + b * sk_b + rn[:, None] * sk_t + rd[None, :], mask=rn[:, None] < Tk, other=0.0)
    acc = tl.zeros([BM, BN], tl.float32)
    for h in tl.static_range(HI):                     # fixed head order
        q = tl.load(QI + b * sq_b + rm[:, None] * sq_t + h * sq_h + rd[None, :], mask=rm[:, None] < Tq, other=0.0)
        w = tl.load(W + b * sw_b + rm * sw_t + h, mask=rm < Tq, other=0.0)
        lg = tl.dot(q, tl.trans(k))
        acc += tl.maximum(lg, 0.0) * w[:, None]
    acc = tl.where((rn[None, :] <= pos[:, None]) & (rn[None, :] < Tk), acc, float("-inf"))
    tl.store(S + b * ss_b + rm[:, None] * ss_t + rn[None, :], acc, mask=(rm[:, None] < Tq) & (rn[None, :] < Tk))


def indexer_scores(qi, ki, w, start: int, BM=64, BN=64):
    """qi [B, Tq, HI, DI] bf16, ki [B, Tk, DI] bf16, w [B, Tq, HI] fp32 (contiguous last dim).
    Returns fp32 [B, Tq, Tk], -inf after each query's absolute position."""
    B, Tq, HI, DI = qi.shape
    Tk = start + Tq
    s = torch.empty(B, Tq, Tk, device=qi.device, dtype=torch.float32)
    _indexer_kernel[(triton.cdiv(Tq, BM), triton.cdiv(Tk, BN), B)](
        qi, ki, w, s, start, Tq, Tk, qi.stride(0), qi.stride(1), qi.stride(2), ki.stride(0), ki.stride(1),
        w.stride(0), w.stride(1), s.stride(0), s.stride(1), HI=HI, DI=DI, BM=BM, BN=BN, num_warps=4)
    return s


# ----------------------------------------------------------------------------- sparse attention
@triton.jit(do_not_specialize=["Tq"])
def _sparse_attn_kernel(Q, K, V, IDX, O, scale_log2, Tq,
                        sq_b, sq_t, sq_h, sk_b, sk_t, sv_b, sv_t, si_b, si_t, so_b, so_t, so_h,
                        H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr,
                        KSEL: tl.constexpr, BN: tl.constexpr):
    t, b = tl.program_id(0), tl.program_id(1)
    rh = tl.arange(0, HP)
    rd = tl.arange(0, D)
    q = tl.load(Q + b * sq_b + t * sq_t + rh[:, None] * sq_h + rd[None, :], mask=rh[:, None] < H, other=0.0)
    m_i = tl.full([HP], float("-inf"), tl.float32)
    l_i = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for n0 in tl.static_range(0, KSEL, BN):           # selected keys in ascending index order
        rn = n0 + tl.arange(0, BN)
        idx = tl.load(IDX + b * si_b + t * si_t + rn)
        ok = idx >= 0
        k = tl.load(K + b * sk_b + idx[:, None] * sk_t + rd[None, :], mask=ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + b * sv_b + idx[:, None] * sv_t + rd[None, :], mask=ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(O + b * so_b + t * so_t + rh[:, None] * so_h + rd[None, :], o.to(tl.bfloat16), mask=rh[:, None] < H)


def sparse_attention(q, k, v, idx, sm_scale: float, BN=32):
    """q [B, Tq, H, D]; k, v [B, Tk, D] (one KV head); idx int32 [B, Tq, KSEL], ascending
    absolute key positions with -1 padding at the end. Returns [B, Tq, H, D] bf16."""
    B, Tq, H, D = q.shape
    KSEL = idx.shape[-1]
    idx = idx.to(torch.int32).contiguous()
    o = torch.empty(B, Tq, H, D, device=q.device, dtype=BF)
    HP = max(16, triton.next_power_of_2(H))
    _sparse_attn_kernel[(Tq, B)](
        q, k, v, idx, o, sm_scale * LOG2E, Tq, q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), v.stride(0), v.stride(1), idx.stride(0), idx.stride(1),
        o.stride(0), o.stride(1), o.stride(2), H=H, HP=HP, D=D, KSEL=KSEL, BN=BN, num_warps=4, num_stages=2)
    return o
