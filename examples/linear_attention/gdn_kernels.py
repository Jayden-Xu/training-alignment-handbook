"""Gated delta rule (Gated DeltaNet-style) linear attention kernels in Triton.

Per head, with state S in R^{Dk x Dv} (fp32), decay g_t = exp(glog_t) in (0, 1),
write strength beta_t and an L2-normalized key k_t:

    S_t = g_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t * Dk^-1/2

(the Dk x Dv layout puts the projection on the left; it is the same rule as
S_t = S_{t-1} g_t (I - beta_t k_t k_t^T) + beta_t v_t k_t^T in the Dv x Dk layout).

Layouts: q, k [B, T, H, Dk] bf16; v [B, T, H, Dv] bf16; glog [B, T, H] fp32 (log of
the decay, <= 0); beta [B, T, H] bf16; h0 [B, H, Dk, Dv] fp32. All contiguous.
Outputs: o [B, T, H, Dv] fp32 (cast to bf16 by the caller), final state [B, H, Dk, Dv] fp32.

gdn_recurrent(q, k, v, glog, beta, h0, BV, num_warps, prefetch)
    One program per (sequence, head, Dv tile of BV columns). It steps t in order
    with its [Dk, BV] state tile in fp32 registers: decay, delta correction,
    rank-1 write, output. It loads the initial state and stores the final state,
    so prefill, chunked prefill and decode are the same loop body run over
    different t ranges. Nothing depends on the batch size or on other sequences.
    With prefetch, the next token's loads are issued before the current token's
    arithmetic; the arithmetic is unchanged. The sum over Dk inside a tile
    depends on BV and num_warps, so both sides of a parity check must use the
    same configuration.

gdn_chunked(q, k, v, glog, beta, h0, prec, C)
    The chunked (WY / UT) form with a fixed chunk size C, chunk boundaries at
    multiples of C from the start of the call. Three kernels:
      prepare  one program per (sequence, head, chunk): in-chunk cumulative
               log-decay G; A = strictly lower beta_i (k_i . k_j) exp(G_i - G_j);
               T = (I + A)^-1 by row-by-row forward substitution in fp32, over
               the chunk's valid rows only; W = T (beta exp(G) k), U = T (beta v),
               BW columns at a time.
      state    one program per (sequence, head, Dv tile): walks the chunks in
               order. Stores the chunk-start state, v_new = U - W S, then
               S = exp(G_last) S + (k exp(G_last - G))^T v_new.
      out      one program per (sequence, head, chunk, Dv tile):
               o = (exp(G) q) S_start + ((q k^T) * exp(G_i - G_j), j <= i) v_new.
    prec="bf16": the matmuls take bf16 operands with fp32 accumulation (the state
    and intermediates are rounded to bf16 at each matmul, as common chunked
    kernels do). prec="fp32": fp32 operands, input_precision="ieee" (no tensor
    cores; FMA dots, untuned). Returns o, the final state and the chunk-start
    states [B, H, NT, Dk, Dv]. A partial last chunk masks its tail to zero, and
    every row's arithmetic is the same as in a full chunk (verified bitwise
    through AlignedChunked).

AlignedChunked(h0, prec, C)
    A cache that makes every call's chunk boundaries sit at absolute multiples
    of C, so decode and chunked prefill of any size reproduce full prefill.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BF = torch.bfloat16
F32 = torch.float32


# ----------------------------------------------------------------------------- recurrent
@triton.jit(do_not_specialize=["T"])
def _recurrent_kernel(Q, K, V, GL, BETA, H0, O, HT, scale, T,
                      H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
                      PREFETCH: tl.constexpr):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    b = i_bh // H
    h = i_bh % H
    rk = tl.arange(0, DK)
    rv = i_v * BV + tl.arange(0, BV)
    st = i_bh * DK * DV + rk[:, None] * DV + rv[None, :]
    S = tl.load(H0 + st)                                    # [DK, BV] fp32
    tok = b * T * H + h
    qn = tl.load(Q + tok * DK + rk)
    kn = tl.load(K + tok * DK + rk)
    vn = tl.load(V + tok * DV + rv)
    gn = tl.load(GL + tok)
    bn = tl.load(BETA + tok)
    for t in range(T):                                      # tokens in order
        tok = (b * T + t) * H + h
        if PREFETCH:                                        # issue the next token's loads now
            qr, kr, vr, g, br = qn, kn, vn, gn, bn
            nxt = (b * T + tl.minimum(t + 1, T - 1)) * H + h
            qn = tl.load(Q + nxt * DK + rk)
            kn = tl.load(K + nxt * DK + rk)
            vn = tl.load(V + nxt * DV + rv)
            gn = tl.load(GL + nxt)
            bn = tl.load(BETA + nxt)
        else:
            qr = tl.load(Q + tok * DK + rk)
            kr = tl.load(K + tok * DK + rk)
            vr = tl.load(V + tok * DV + rv)
            g = tl.load(GL + tok)
            br = tl.load(BETA + tok)
        q = qr.to(tl.float32) * scale
        k = kr.to(tl.float32)
        v = vr.to(tl.float32)
        beta = br.to(tl.float32)
        S = S * tl.exp(g)                                   # decay
        u = (v - tl.sum(S * k[:, None], 0)) * beta          # delta correction
        S = S + k[:, None] * u[None, :]                     # rank-1 write
        tl.store(O + tok * DV + rv, tl.sum(S * q[:, None], 0))
    tl.store(HT + st, S)


def gdn_recurrent(q, k, v, glog, beta, h0, BV=8, num_warps=1, prefetch=True):
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    for t in (q, k, v, glog, beta, h0):
        assert t.is_contiguous()
    o = torch.empty(B, T, H, DV, device=q.device, dtype=F32)
    ht = torch.empty(B, H, DK, DV, device=q.device, dtype=F32)
    if T == 0:
        ht.copy_(h0)
        return o, ht
    _recurrent_kernel[(DV // BV, B * H)](q, k, v, glog, beta, h0, o, ht, DK ** -0.5, T,
                                         H=H, DK=DK, DV=DV, BV=BV, PREFETCH=prefetch,
                                         num_warps=num_warps)
    return o, ht


# ----------------------------------------------------------------------------- chunked
@triton.jit
def _mm(a, b, PREC: tl.constexpr):
    if PREC == 1:
        r = tl.dot(a, b, input_precision="ieee")
    else:
        r = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    return r


@triton.jit(do_not_specialize=["T"])
def _chunk_prepare(K, V, GL, BETA, GC, W, U, T,
                   H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
                   BW: tl.constexpr, PREC: tl.constexpr):
    i_n, i_bh = tl.program_id(0), tl.program_id(1)
    b = i_bh // H
    h = i_bh % H
    rc = tl.arange(0, C)
    rk = tl.arange(0, DK)
    t = i_n * C + rc
    valid = t < T
    tok = (b * T + t) * H + h
    g = tl.load(GL + tok, mask=valid, other=0.0)
    G = tl.cumsum(g, 0)                                     # in-chunk cumulative log-decay
    tl.store(GC + tok, G, mask=valid)
    beta = tl.load(BETA + tok, mask=valid, other=0.0).to(tl.float32)
    k = tl.load(K + tok[:, None] * DK + rk[None, :], mask=valid[:, None], other=0.0)
    kk = tl.dot(k, tl.trans(k))                             # bf16 operands, exact products
    lower = rc[:, None] > rc[None, :]
    X = tl.where(lower, -(kk * beta[:, None] * tl.exp(G[:, None] - G[None, :])), 0.0)
    # X <- (I + A)^-1 - I, one row at a time: x_i = n_i + sum_{j<i} n_ij x_j. Rows past
    # the chunk's valid length are zero and are never read by valid rows: skip them.
    for i in range(1, tl.minimum(T - i_n * C, C)):
        sel = rc == i
        a = tl.sum(tl.where(sel[:, None], X, 0.0), 0)       # row i, not yet updated
        a = a + tl.sum(a[:, None] * X, 0)
        a = tl.where(rc < i, a, 0.0)
        X = tl.where(sel[:, None], a[None, :], X)
    Tm = X + tl.where(rc[:, None] == rc[None, :], 1.0, 0.0)
    sc = beta * tl.exp(G)
    for j in tl.static_range(0, DK, BW):                   # W = T (beta exp(G) k), BW columns at a time
        rj = j + tl.arange(0, BW)
        kj = tl.load(K + tok[:, None] * DK + rj[None, :], mask=valid[:, None], other=0.0)
        tl.store(W + tok[:, None] * DK + rj[None, :], _mm(Tm, kj.to(tl.float32) * sc[:, None], PREC),
                 mask=valid[:, None])
    for j in tl.static_range(0, DV, BW):                   # U = T (beta v)
        rj = j + tl.arange(0, BW)
        vj = tl.load(V + tok[:, None] * DV + rj[None, :], mask=valid[:, None], other=0.0)
        tl.store(U + tok[:, None] * DV + rj[None, :], _mm(Tm, vj.to(tl.float32) * beta[:, None], PREC),
                 mask=valid[:, None])


@triton.jit(do_not_specialize=["T", "NT"])
def _chunk_state(K, W, U, GC, H0, HS, VN, HT, T, NT,
                 H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
                 BV: tl.constexpr, PREC: tl.constexpr):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    b = i_bh // H
    h = i_bh % H
    rc = tl.arange(0, C)
    rk = tl.arange(0, DK)
    rv = i_v * BV + tl.arange(0, BV)
    st = rk[:, None] * DV + rv[None, :]
    S = tl.load(H0 + i_bh * DK * DV + st)
    for n in range(NT):                                     # chunks in order
        tl.store(HS + (i_bh * NT + n) * DK * DV + st, S)    # chunk-start state
        t = n * C + rc
        valid = t < T
        tok = (b * T + t) * H + h
        w = tl.load(W + tok[:, None] * DK + rk[None, :], mask=valid[:, None], other=0.0)
        u = tl.load(U + tok[:, None] * DV + rv[None, :], mask=valid[:, None], other=0.0)
        vn = u - _mm(w, S, PREC)
        tl.store(VN + tok[:, None] * DV + rv[None, :], vn, mask=valid[:, None])
        last = tl.minimum(T - n * C, C) - 1
        g_last = tl.load(GC + (b * T + n * C + last) * H + h)
        G = tl.load(GC + tok, mask=valid, other=0.0)
        k = tl.load(K + tok[:, None] * DK + rk[None, :], mask=valid[:, None], other=0.0)
        kd = k.to(tl.float32) * tl.where(valid, tl.exp(g_last - G), 0.0)[:, None]
        S = S * tl.exp(g_last) + _mm(tl.trans(kd), vn, PREC)
    tl.store(HT + i_bh * DK * DV + st, S)


@triton.jit(do_not_specialize=["T", "NT"])
def _chunk_out(Q, K, GC, HS, VN, O, scale, T, NT,
               H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
               BV: tl.constexpr, PREC: tl.constexpr):
    i_v, i_n, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b = i_bh // H
    h = i_bh % H
    rc = tl.arange(0, C)
    rk = tl.arange(0, DK)
    rv = i_v * BV + tl.arange(0, BV)
    t = i_n * C + rc
    valid = t < T
    tok = (b * T + t) * H + h
    q = tl.load(Q + tok[:, None] * DK + rk[None, :], mask=valid[:, None], other=0.0)
    k = tl.load(K + tok[:, None] * DK + rk[None, :], mask=valid[:, None], other=0.0)
    G = tl.load(GC + tok, mask=valid, other=0.0)
    S = tl.load(HS + (i_bh * NT + i_n) * DK * DV + rk[:, None] * DV + rv[None, :])
    vn = tl.load(VN + tok[:, None] * DV + rv[None, :], mask=valid[:, None], other=0.0)
    qk = tl.dot(q, tl.trans(k))
    P = tl.where(rc[:, None] >= rc[None, :], qk * tl.exp(G[:, None] - G[None, :]), 0.0)
    qg = q.to(tl.float32) * tl.exp(G)[:, None]
    o = (_mm(qg, S, PREC) + _mm(P, vn, PREC)) * scale
    tl.store(O + tok[:, None] * DV + rv[None, :], o, mask=valid[:, None])


def gdn_chunked(q, k, v, glog, beta, h0, prec="bf16", C=64, events=None):
    """Returns o [B, T, H, Dv] fp32, final state [B, H, Dk, Dv], chunk-start states
    [B, H, NT, Dk, Dv]. `events`: optional list that receives CUDA events around
    each of the three kernels (for timing). Tile sizes are fixed per precision."""
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    for t in (q, k, v, glog, beta, h0):
        assert t.is_contiguous()
    P = {"bf16": 0, "fp32": 1}[prec]
    BW, BV_STATE, BV_OUT = (32, 16, 64) if P else (128, 32, 64)  # fp32 (FMA dots): smaller tiles
    NT = triton.cdiv(T, C)
    dev = q.device
    gc = torch.empty(B, T, H, device=dev, dtype=F32)
    w = torch.empty(B, T, H, DK, device=dev, dtype=F32)
    u = torch.empty(B, T, H, DV, device=dev, dtype=F32)
    hs = torch.empty(B, H, NT, DK, DV, device=dev, dtype=F32)
    vn = torch.empty(B, T, H, DV, device=dev, dtype=F32)
    ht = torch.empty(B, H, DK, DV, device=dev, dtype=F32)
    o = torch.empty(B, T, H, DV, device=dev, dtype=F32)
    if T == 0:
        ht.copy_(h0)
        return o, ht, hs

    def mark():
        if events is not None:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            events.append(e)
    mark()
    _chunk_prepare[(NT, B * H)](k, v, glog, beta, gc, w, u, T, H=H, DK=DK, DV=DV, C=C,
                                BW=BW, PREC=P, num_warps=4)
    mark()
    _chunk_state[(DV // BV_STATE, B * H)](k, w, u, gc, h0, hs, vn, ht, T, NT, H=H, DK=DK, DV=DV,
                                          C=C, BV=BV_STATE, PREC=P, num_warps=4, num_stages=1)
    mark()
    _chunk_out[(DV // BV_OUT, NT, B * H)](q, k, gc, hs, vn, o, DK ** -0.5, T, NT, H=H, DK=DK, DV=DV,
                                          C=C, BV=BV_OUT, PREC=P, num_warps=4)
    mark()
    return o, ht, hs


FIELDS = ("q", "k", "v", "glog", "beta")


class AlignedChunked:
    """Chunked GDN whose chunk boundaries sit at absolute multiples of C in every
    mode. It keeps the state at the last boundary and the tokens since then; each
    call (a decode step or a prefill chunk of any size) reruns the chunked
    algorithm from that boundary over the pending plus the new tokens, and returns
    the new tokens' outputs. At most C-1 tokens are recomputed per call."""

    def __init__(self, h0, prec="bf16", C=64):
        self.S, self.pend, self.prec, self.C = h0, None, prec, C
        self.final = h0

    def __call__(self, x):
        n_old = 0 if self.pend is None else self.pend["q"].shape[1]
        full = x if self.pend is None else {f: torch.cat([self.pend[f], x[f]], 1).contiguous() for f in FIELDS}
        L = full["q"].shape[1]
        o, ht, hs = gdn_chunked(*(full[f] for f in FIELDS), self.S, prec=self.prec, C=self.C)
        nb = L // self.C
        if L % self.C == 0:
            self.S, self.pend = ht, None
        else:
            if nb > 0:
                self.S = hs[:, :, nb].contiguous()
            self.pend = {f: full[f][:, nb * self.C:].contiguous() for f in FIELDS}
        self.final = ht
        return o[:, n_old:]
