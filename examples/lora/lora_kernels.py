"""Triton kernels for the LoRA linear case (examples/lora/lora_bi.py).

Layouts (contiguous unless a stride says otherwise):
  x     [T, H]    bf16   one row per token
  W     [H, N]    bf16   base weight in the layout of tools/bi_kernels.mm_one_config: y = x @ W
  A     [L, R, H] bf16   adapter l's lora_A.weight in rows [:r_l], zeros above (rank padding)
  B     [L, R, N] bf16   adapter l's lora_B.weight, transposed, in rows [:r_l], zeros above
  idx   [T]       int32  adapter of each row, -1 for none
  scale [L]       fp32   alpha / r per adapter

Part 1: a batch-invariant LoRA linear, one kernel for the trainer and the engine.

  lora_shrink(x, A, idx) -> u [T, R] fp32
      u[t, j] = sum_k x[t, k] * A[idx[t], j, k]
      One program per (row, 16 ranks). It reads the row's own adapter. K is walked
      in ascending chunks of 256; each chunk's products (exact in fp32) are summed
      by one fixed tree and added to an fp32 accumulator. u is not rounded to bf16.

  lora_linear(x, W, A, B, idx, scale) -> y [T, N] bf16
      y[t, n] = bf16(fma(s, upd, base)), s = scale[idx[t]]
        base = sum_k x[t, k] * W[k, n]             mm_one_config's loop and tiles, fp32
        upd  = sum_j u[t, j] * B[idx[t], j, n]     j ascending, one fp32 fma per rank
      Base and update meet in one fp32 fused multiply-add, followed by the only
      cast. Rows with idx = -1 get bf16(base): the bits of mm_one_config(x, W).

  lora_linear_backward(x, W, A, B, idx, scale, u, g) -> dx, dA, dB
      du[t, j]    = sum_n g[t, n] * B[idx[t], j, n]                 lora_shrink(g, B)
      dx[t, k]    = bf16(fma(s, sum_j du[t, j] * A[idx[t], j, k], sum_n g[t, n] * W[k, n]))
                    the forward kernel on (g, W^T, du, A): per row
      dA[l, j, k] = s_l * sum_t du[t, j] * x[t, k]
      dB[l, j, n] = s_l * sum_t u[t, j]  * g[t, n]
                    over the rows t of adapter l whose g is not all zero, in ascending
                    row order: mm_one_config's loop with K = those rows. The fp32
                    factor enters as two bf16 terms, its bf16 rounding and then the
                    bf16 rounding of what is left, so about 16 of its bits take part.
                    One program owns each output tile and writes it once: no atomics.
                    fp32 out.

Nothing here depends on the number of rows, on a row's position, or on what the
other rows hold. A row of another adapter adds programs; it changes no reduction.

Part 2: models of an engine's gathered LoRA kernels, to measure that form here.
Their structure (grids, block sizes, split factors, dtypes, where the atomics are)
was read from vLLM 0.6.3.post1: vllm/lora/ops/{sgmv,bgmv}_{shrink,expand}.py and
vllm/lora/punica.py. They are not vLLM's code and nothing here ran inside vLLM.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BF = torch.bfloat16

# One configuration, for every batch and every stack of adapters.
SHRINK_BR, SHRINK_BK = 16, 256
LIN_BM, LIN_BN, LIN_BK = 64, 64, 32
GRAD_BJ, GRAD_BC, GRAD_BK = 16, 64, 32


# ----------------------------------------------------------------------------- part 1: forward
@triton.jit(do_not_specialize=["T", "R"])
def _shrink_kernel(X, A, IDX, U, T, R, K, sxt, sxk, sal, saj, sak, sut, suj,
                   BR: tl.constexpr, BK: tl.constexpr):
    t, pid_r = tl.program_id(0), tl.program_id(1)
    a_id = tl.load(IDX + t)
    rj = pid_r * BR + tl.arange(0, BR)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BR,), dtype=tl.float32)
    if a_id >= 0:
        for k0 in range(0, K, BK):                    # K ascending, the row's own adapter
            kk = k0 + rk
            xv = tl.load(X + t * sxt + kk * sxk, mask=kk < K, other=0.0).to(tl.float32)
            av = tl.load(A + a_id * sal + rj[:, None] * saj + kk[None, :] * sak,
                         mask=(rj[:, None] < R) & (kk[None, :] < K), other=0.0).to(tl.float32)
            acc += tl.sum(av * xv[None, :], axis=1)   # bf16 x bf16 products are exact in fp32
    tl.store(U + t * sut + rj * suj, acc, mask=rj < R)


@triton.jit(do_not_specialize=["T", "R"])
def _linear_kernel(X, W, U, B, IDX, S, Y, T, N, K, R,
                   sxt, sxk, swk, swn, sut, suj, sbl, sbj, sbn, syt, syn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, COMBINE: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    row_ok = rm < T
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):                        # base GEMM: mm_one_config's loop
        kk = k0 + rk
        a = tl.load(X + rm[:, None] * sxt + kk[None, :] * sxk,
                    mask=row_ok[:, None] & (kk[None, :] < K), other=0.0)
        b = tl.load(W + kk[:, None] * swk + rn[None, :] * swn,
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    ids = tl.load(IDX + rm, mask=row_ok, other=-1)
    has = ids >= 0
    safe = tl.maximum(ids, 0)
    s = tl.load(S + safe, mask=has, other=0.0)
    upd = tl.zeros((BM, BN), dtype=tl.float32)
    bmask = has[:, None] & (rn[None, :] < N)
    for j in range(0, R):                             # ascending rank, each row's own adapter
        uj = tl.load(U + rm * sut + j * suj, mask=has, other=0.0)
        bj = tl.load(B + safe[:, None] * sbl + j * sbj + rn[None, :] * sbn, mask=bmask, other=0.0)
        upd = tl.fma(uj[:, None], bj.to(tl.float32), upd)
    omask = row_ok[:, None] & (rn[None, :] < N)
    optr = Y + rm[:, None] * syt + rn[None, :] * syn
    if COMBINE == 0:                                  # the kernel: one fp32 fma, one cast
        y = tl.where(has[:, None], tl.fma(s[:, None], upd, acc), acc)
        tl.store(optr, y.to(tl.bfloat16), mask=omask)
    elif COMBINE == 1:                                # rejected: two bf16 values, then added
        d = (s[:, None] * upd).to(tl.bfloat16).to(tl.float32)
        tl.store(optr, (acc.to(tl.bfloat16).to(tl.float32) + d).to(tl.bfloat16), mask=omask)
    elif COMBINE == 2:                                # rejected: update cast, scaled in bf16, added
        d = (upd.to(tl.bfloat16).to(tl.float32) * s[:, None]).to(tl.bfloat16).to(tl.float32)
        tl.store(optr, (acc.to(tl.bfloat16).to(tl.float32) + d).to(tl.bfloat16), mask=omask)
    elif COMBINE == 3:                                # debug: the fp32 base accumulator
        tl.store(optr, acc, mask=omask)
    elif COMBINE == 4:                                # debug: the fp32 update before the scale
        tl.store(optr, upd, mask=omask)
    else:                                             # debug: the fp32 value before the cast
        tl.store(optr, tl.where(has[:, None], tl.fma(s[:, None], upd, acc), acc), mask=omask)


def as_idx(idx: torch.Tensor) -> torch.Tensor:
    return idx if idx.dtype == torch.int32 else idx.to(torch.int32)


def lora_shrink(x: torch.Tensor, A: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    T, K = x.shape
    R = A.shape[1]
    u = torch.empty(T, R, device=x.device, dtype=torch.float32)
    if T:
        _shrink_kernel[(T, triton.cdiv(R, SHRINK_BR))](
            x, A, as_idx(idx), u, T, R, K, x.stride(0), x.stride(1),
            A.stride(0), A.stride(1), A.stride(2), u.stride(0), u.stride(1),
            BR=SHRINK_BR, BK=SHRINK_BK, num_warps=4)
    return u


def lora_expand_add(x, W, u, B, idx, scale, combine: int = 0) -> torch.Tensor:
    """The base GEMM fused with the update. combine=0 is the kernel. 1 and 2 move the
    rounding points (rejected designs, kept to measure them). 3, 4 and 5 return the
    fp32 base, the fp32 update and the fp32 value before the cast."""
    T, K = x.shape
    N = W.shape[1]
    R = B.shape[1]
    y = torch.empty(T, N, device=x.device, dtype=BF if combine < 3 else torch.float32)
    if T:
        _linear_kernel[(triton.cdiv(T, LIN_BM), triton.cdiv(N, LIN_BN))](
            x, W, u, B, as_idx(idx), scale, y, T, N, K, R,
            x.stride(0), x.stride(1), W.stride(0), W.stride(1), u.stride(0), u.stride(1),
            B.stride(0), B.stride(1), B.stride(2), y.stride(0), y.stride(1),
            BM=LIN_BM, BN=LIN_BN, BK=LIN_BK, COMBINE=combine, num_warps=4, num_stages=3)
    return y


def lora_linear(x, W, A, B, idx, scale) -> torch.Tensor:
    """y = x @ W + scale[idx] * (x @ A[idx]^T) @ B[idx], per row, bf16."""
    idx = as_idx(idx)
    return lora_expand_add(x, W, lora_shrink(x, A, idx), B, idx, scale)


# ----------------------------------------------------------------------------- part 1: backward
@triton.jit(do_not_specialize=["R", "C"])
def _grad_kernel(V, Z, ORDER, START, COUNT, S, OUT, R, C, svt, svj, szt, szc, sol, soj, soc,
                 BJ: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr):
    """out[l, j, c] = s[l] * sum over adapter l's listed rows t, ascending, of
    v[t, j] * z[t, c]: mm_one_config's loop with K = the adapter's rows."""
    l, pid_j, pid_c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rj = pid_j * BJ + tl.arange(0, BJ)
    rc = pid_c * BC + tl.arange(0, BC)
    rq = tl.arange(0, BK)
    start = tl.load(START + l)
    n = tl.load(COUNT + l)
    acc = tl.zeros((BJ, BC), dtype=tl.float32)
    for q0 in range(0, n, BK):                        # the adapter's rows, ascending
        q = q0 + rq
        live = q < n
        t = tl.load(ORDER + start + q, mask=live, other=0)
        v = tl.load(V + t[None, :] * svt + rj[:, None] * svj,
                    mask=live[None, :] & (rj[:, None] < R), other=0.0)
        z = tl.load(Z + t[:, None] * szt + rc[None, :] * szc,
                    mask=live[:, None] & (rc[None, :] < C), other=0.0)
        v_hi = v.to(tl.bfloat16)                      # fp32 factor as two bf16 terms
        v_lo = (v - v_hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(v_hi, z, acc)
        acc = tl.dot(v_lo, z, acc)
    s = tl.load(S + l)
    tl.store(OUT + l * sol + rj[:, None] * soj + rc[None, :] * soc, acc * s,
             mask=(rj[:, None] < R) & (rc[None, :] < C))


def adapter_rows(idx: torch.Tensor, L: int, keep: torch.Tensor = None):
    """(order, start, count): the rows of each adapter in ascending row order. Rows
    with idx = -1, and rows where keep is False, belong to no adapter. Integer
    bookkeeping only, no host sync."""
    key = idx.long()
    if keep is not None:
        key = torch.where(keep, key, torch.full_like(key, -1))
    order = torch.argsort(key, stable=True).to(torch.int32)
    count = torch.bincount(key + 1, minlength=L + 1)
    start = torch.cumsum(count, 0)[:-1].to(torch.int32)
    return order, start, count[1:].to(torch.int32)


def lora_grad(v, z, rows, scale, L: int) -> torch.Tensor:
    R, C = v.shape[1], z.shape[1]
    out = torch.empty(L, R, C, device=v.device, dtype=torch.float32)
    order, start, count = rows
    _grad_kernel[(L, triton.cdiv(R, GRAD_BJ), triton.cdiv(C, GRAD_BC))](
        v, z, order, start, count, scale, out, R, C, v.stride(0), v.stride(1), z.stride(0), z.stride(1),
        out.stride(0), out.stride(1), out.stride(2), BJ=GRAD_BJ, BC=GRAD_BC, BK=GRAD_BK,
        num_warps=4, num_stages=1)                    # rows are read through `order`: no pipelining
    return out


def lora_linear_backward(x, W, A, B, idx, scale, u, g, Wt=None):
    """g = dL/dy [T, N] bf16; u from the forward. Returns dx bf16, dA fp32 [L, R, H],
    dB fp32 [L, R, N]. Wt: W.t().contiguous(), if the caller keeps one (W is frozen)."""
    idx = as_idx(idx)
    du = lora_shrink(g, B, idx)
    dx = lora_expand_add(g, W.t() if Wt is None else Wt, du, A, idx, scale)
    L = A.shape[0]
    rows = adapter_rows(idx, L, keep=(g != 0).any(dim=1))     # rows with no gradient are skipped
    return dx, lora_grad(du, x, rows, scale, L), lora_grad(u, g, rows, scale, L)


class _LoraLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, A, B, idx, scale, Wt):
        A16, B16 = A.to(BF), B.to(BF)                 # the values the engine loads
        idx = as_idx(idx)
        u = lora_shrink(x, A16, idx)
        ctx.save_for_backward(x, W, A16, B16, idx, scale, u, Wt)
        return lora_expand_add(x, W, u, B16, idx, scale)

    @staticmethod
    def backward(ctx, g):
        x, W, A16, B16, idx, scale, u, Wt = ctx.saved_tensors
        torch.cuda.current_stream(g.device).query()   # a runtime call binds the CUDA context to autograd's worker thread; Triton needs it
        dx, dA, dB = lora_linear_backward(x, W, A16, B16, idx, scale, u, g.contiguous(), Wt)
        return dx, None, dA, dB, None, None, None


def lora_linear_train(x, W, A, B, idx, scale, Wt=None) -> torch.Tensor:
    """The same forward, with autograd. A and B may be fp32 masters: the forward runs
    on their bf16 values, and dA and dB come back in fp32 (autograd casts them to the
    leaf's dtype). W is frozen and gets no gradient."""
    return _LoraLinear.apply(x, W, A, B, idx, scale, W.t() if Wt is None else Wt)


# ----------------------------------------------------------------------------- part 2: engine models
# Prefill model ("segmented"): consecutive rows with the same adapter form a segment.
#   shrink: tiles 32x16x32; SPLIT_K = 8 programs per output tile, each summing every
#           8th K block in fp32, then an atomic add of its partial into an fp32 buffer.
#   expand: tiles 32x32x16; the buffer is cast to the adapter dtype; fp32 accumulate;
#           cast to bf16; added to the bf16 base output already in y.
# Decode model ("per row"): one output row per program column.
#   shrink: 64 programs per row, each covering 256 of K, products and sum in the
#           adapter dtype, then an atomic add into the fp32 buffer.
#   expand: SPLIT_N programs per row; the buffer is cast to the adapter dtype;
#           products and sum in the adapter dtype, plus the bf16 base output.
# The scale is not in these kernels (they get 1.0): it is folded into B at load.

@triton.jit(do_not_specialize=["R"])
def _seg_shrink_kernel(X, A, U, SEG_START, SEG_LEN, SEG_ID, R, K, scaling, sxt, sxk, sal, saj, sak, sut, suj,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr):
    pid, pid_sk, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tiles_n = tl.cdiv(R, BN)
    pid_m, pid_n = pid // tiles_n, pid % tiles_n
    M = tl.load(SEG_LEN + seg)
    a_id = tl.load(SEG_ID + seg)
    if (pid_m * BM < M) & (a_id >= 0):
        start = tl.load(SEG_START + seg)
        lm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = pid_sk * BK + tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK * SPLIT_K):
            kk = k0 + rk
            a = tl.load(X + (start + lm)[:, None] * sxt + kk[None, :] * sxk,
                        mask=(lm[:, None] < M) & (kk[None, :] < K), other=0.0)
            b = tl.load(A + a_id * sal + kk[:, None] * sak + rn[None, :] * saj,
                        mask=(kk[:, None] < K) & (rn[None, :] < R), other=0.0)
            acc += tl.dot(a, b)
        acc *= scaling
        tl.atomic_add(U + (start + lm)[:, None] * sut + rn[None, :] * suj, acc,
                      mask=(lm[:, None] < M) & (rn[None, :] < R))


@triton.jit(do_not_specialize=["R"])
def _seg_expand_kernel(U, B, Y, SEG_START, SEG_LEN, SEG_ID, N, R, sut, suj, sbl, sbj, sbn, syt, syn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid, seg = tl.program_id(0), tl.program_id(1)
    tiles_n = tl.cdiv(N, BN)
    pid_m, pid_n = pid // tiles_n, pid % tiles_n
    M = tl.load(SEG_LEN + seg)
    a_id = tl.load(SEG_ID + seg)
    if (pid_m * BM < M) & (a_id >= 0):
        start = tl.load(SEG_START + seg)
        lm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, R, BK):
            kk = k0 + rk
            a = tl.load(U + (start + lm)[:, None] * sut + kk[None, :] * suj,
                        mask=(lm[:, None] < M) & (kk[None, :] < R), other=0.0).to(tl.bfloat16)
            b = tl.load(B + a_id * sbl + kk[:, None] * sbj + rn[None, :] * sbn,
                        mask=(kk[:, None] < R) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        c = acc.to(tl.bfloat16)
        optr = Y + (start + lm)[:, None] * syt + rn[None, :] * syn
        omask = (lm[:, None] < M) & (rn[None, :] < N)
        c += tl.load(optr, mask=omask, other=0.0)     # bf16 + bf16
        tl.store(optr, c, mask=omask)


@triton.jit(do_not_specialize=["R"])
def _row_shrink_kernel(X, A, U, IDX, R, K, scaling, sxt, sxk, sal, saj, sak, sut, suj,
                       BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_sk, t = tl.program_id(0), tl.program_id(1)
    a_id = tl.load(IDX + t)
    if a_id >= 0:
        rn = tl.arange(0, BN)
        rk = pid_sk * BK + tl.arange(0, BK)
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, K, BK * SPLIT_K):
            kk = k0 + rk
            a = tl.load(X + t * sxt + kk * sxk, mask=kk < K, other=0.0)
            b = tl.load(A + a_id * sal + rn[:, None] * saj + kk[None, :] * sak,
                        mask=(rn[:, None] < R) & (kk[None, :] < K), other=0.0)
            acc += tl.sum(a * b, 1)                   # products and sum in the operands' dtype
        acc *= scaling
        tl.atomic_add(U + t * sut + rn * suj, acc, mask=rn < R)


@triton.jit(do_not_specialize=["R"])
def _row_expand_kernel(U, B, Y, IDX, N, R, sut, suj, sbl, sbj, sbn, syt, syn,
                       BN: tl.constexpr, BK: tl.constexpr, SPLIT_N: tl.constexpr):
    pid_sn, t = tl.program_id(0), tl.program_id(1)
    a_id = tl.load(IDX + t)
    if a_id >= 0:
        rk = tl.arange(0, BK)
        a = tl.load(U + t * sut + rk * suj, mask=rk < R, other=0.0).to(tl.bfloat16)
        split_n = tl.cdiv(N, SPLIT_N)
        n_off = pid_sn * split_n
        for n0 in range(0, split_n, BN):
            ln = n0 + tl.arange(0, BN)
            nmask = ln < split_n
            b = tl.load(B + a_id * sbl + (n_off + ln)[:, None] * sbn + rk[None, :] * sbj,
                        mask=nmask[:, None] & (rk[None, :] < R), other=0.0)
            optr = Y + t * syt + (n_off + ln) * syn
            out = tl.load(optr, mask=nmask, other=0.0)
            tl.store(optr, tl.sum(a * b, 1) + out, mask=nmask)


def segments(idx: torch.Tensor):
    """Consecutive rows with the same adapter: (start, length, adapter), int32."""
    ids, lens = torch.unique_consecutive(idx, return_counts=True)
    starts = torch.cumsum(lens, 0) - lens
    return starts.to(torch.int32), lens.to(torch.int32), ids.to(torch.int32)


def seg_lora_add(y, x, A, Bs, idx) -> torch.Tensor:
    """Prefill model: y += (x @ A[idx]^T) @ Bs[idx] in place. Bs holds scale * B."""
    T, K = x.shape
    N, R = y.shape[1], A.shape[1]
    start, length, ids = segments(idx)
    max_len, n_seg = int(length.max()), start.shape[0]
    u = torch.zeros(T, R, device=x.device, dtype=torch.float32)
    _seg_shrink_kernel[(triton.cdiv(max_len, 32) * triton.cdiv(R, 16), 8, n_seg)](
        x, A, u, start, length, ids, R, K, 1.0, x.stride(0), x.stride(1),
        A.stride(0), A.stride(1), A.stride(2), u.stride(0), u.stride(1), BM=32, BN=16, BK=32, SPLIT_K=8)
    _seg_expand_kernel[(triton.cdiv(max_len, 32) * triton.cdiv(N, 32), n_seg)](
        u, Bs, y, start, length, ids, N, R, u.stride(0), u.stride(1),
        Bs.stride(0), Bs.stride(1), Bs.stride(2), y.stride(0), y.stride(1), BM=32, BN=32, BK=16)
    return y


def _split_n(n: int) -> int:
    for d in (64, 32, 16, 8, 4, 2):
        if n % d == 0:
            return d
    return 1


def row_lora_add(y, x, A, Bs, idx) -> torch.Tensor:
    """Decode model: y += (x @ A[idx]^T) @ Bs[idx] in place, one row per program column."""
    T, K = x.shape
    N, R = y.shape[1], A.shape[1]
    idx = as_idx(idx)
    u = torch.zeros(T, R, device=x.device, dtype=torch.float32)
    bk = triton.next_power_of_2(R)
    _row_shrink_kernel[(64, T)](
        x, A, u, idx, R, K, 1.0, x.stride(0), x.stride(1), A.stride(0), A.stride(1), A.stride(2),
        u.stride(0), u.stride(1), BN=bk, BK=256, SPLIT_K=64, num_warps=8)
    sn = _split_n(N)
    _row_expand_kernel[(sn, T)](
        u, Bs, y, idx, N, R, u.stride(0), u.stride(1), Bs.stride(0), Bs.stride(1), Bs.stride(2),
        y.stride(0), y.stride(1), BN=256, BK=bk, SPLIT_N=sn, num_warps=8)
    return y
