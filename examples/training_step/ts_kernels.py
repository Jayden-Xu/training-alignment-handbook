"""Fixed-order reductions for the training step.

Every reduction here has an order that is written down, so two frameworks that
implement the same rule produce the same bits, not only two runs of one.

seg_sum(src, index, num_dst, base=None)                       (Triton)
    out[d] = fl32(base[d]) + src[i1] + src[i2] + ...  over the positions
    i1 < i2 < ... with index[i] == d. One add at a time, in ascending source
    position, in an fp32 accumulator (round to nearest even). One program per
    (destination row, block of columns): no atomics, one write per element.
    The result is fp32; the caller rounds once to its storage dtype.
    This one rule replaces index_add_, scatter_add_, index_put_(accumulate),
    weighted bincount, and the backward of embedding, index_select, gather and
    an MoE dispatch.

tree_sum(x)                                                   (torch ops)
    Sum over the last dim with a fixed pairwise tree on absolute positions:
    level by level, element 2i + element 2i+1, missing positions are +0.0.
    The tree of a sequence does not depend on how far it is padded, because
    x + (+0.0) == x for every x except -0.0.

tree_mean_loss(per_token, lengths)                            (torch ops)
    The loss reduction: tree_sum per sequence over its own positions, then
    tree_sum over the sequences in batch order, then one fp32 division by the
    token count held in an fp32 tensor (see divide_by_count).

seq_cumsum(x)                                                 (Triton)
    Prefix sums along the last dim, strictly left to right in fp32.

xor32 / sum64
    Fingerprints of a tensor's raw 32-bit words, computed on the device.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------- seg_sum
@triton.jit(do_not_specialize=["H", "s_src", "s_out"])
def _seg_sum_kernel(src_ptr, order_ptr, offs_ptr, dst_ptr, out_ptr, H, s_src, s_out,
                    BH: tl.constexpr, HAS_BASE: tl.constexpr):
    u = tl.program_id(0)                               # one non-empty destination row
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    cmask = cols < H
    start = tl.load(offs_ptr + u)
    end = tl.load(offs_ptr + u + 1)
    d = tl.load(dst_ptr + u)
    if HAS_BASE:                                       # out was initialised with the base rows
        acc = tl.load(out_ptr + d * s_out + cols, mask=cmask, other=0.0)
    else:
        acc = tl.zeros([BH], tl.float32)
    for i in range(start, end):                        # ascending source position
        r = tl.load(order_ptr + i)
        acc += tl.load(src_ptr + r * s_src + cols, mask=cmask, other=0.0).to(tl.float32)
    tl.store(out_ptr + d * s_out + cols, acc, mask=cmask)


def seg_plan(index: torch.Tensor):
    """Stable sort by destination. Returns (order, offs, dst): order[offs[u]:offs[u+1]] are the
    source positions of destination dst[u], ascending. Integer ops only (no cumsum, which
    raises in strict deterministic mode)."""
    index = index.reshape(-1).to(torch.int64)
    n = index.numel()
    sidx, order = torch.sort(index, stable=True)
    first = torch.ones(n, dtype=torch.bool, device=index.device)
    first[1:] = sidx[1:] != sidx[:-1]
    starts = first.nonzero().flatten()
    offs = torch.cat([starts, starts.new_tensor([n])])
    return order, offs, sidx[starts]


def seg_sum(src: torch.Tensor, index: torch.Tensor, num_dst: int, base: torch.Tensor = None,
            BH: int = 64, plan=None) -> torch.Tensor:
    """src [N, H] (any float dtype, rows may be strided, last dim contiguous), index [N] ->
    fp32 [num_dst, H]. See the module docstring for the order."""
    if src.dim() == 1:
        src = src.unsqueeze(1)
    N, H = src.shape
    if src.stride(1) != 1:
        src = src.contiguous()
    out = (torch.zeros(num_dst, H, device=src.device, dtype=torch.float32) if base is None
           else base.reshape(num_dst, H).to(torch.float32, copy=True).contiguous())
    if N == 0:
        return out
    order, offs, dst = seg_plan(index) if plan is None else plan
    _seg_sum_kernel[(dst.numel(), triton.cdiv(H, BH))](
        src, order, offs, dst, out, H, src.stride(0), out.stride(0),
        BH=BH, HAS_BASE=base is not None, num_warps=1 if BH <= 64 else 4)
    return out


class _GatherRowsFixed(torch.autograd.Function):
    """y = table[index] (rows). Backward: the fixed-order segmented sum."""

    @staticmethod
    def forward(ctx, table, index):
        ctx.save_for_backward(index)
        ctx.rows = table.shape[0]
        return table[index]

    @staticmethod
    def backward(ctx, g):
        (index,) = ctx.saved_tensors
        H = g.shape[-1]
        return seg_sum(g.reshape(-1, H), index, ctx.rows).to(g.dtype), None


def gather_rows_fixed(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Embedding lookup, index_select(0, .) or an MoE dispatch with a fixed-order backward.
    table [R, H], index any shape -> [*index.shape, H]."""
    return _GatherRowsFixed.apply(table, index)


class _IndexAddFixed(torch.autograd.Function):
    """out = base.index_add(0, index, src) in the fixed order. Backward is a gather."""

    @staticmethod
    def forward(ctx, base, index, src):
        ctx.save_for_backward(index)
        return seg_sum(src, index, base.shape[0], base=base).to(base.dtype)

    @staticmethod
    def backward(ctx, g):
        (index,) = ctx.saved_tensors
        return g, None, g[index]


def index_add_fixed(base: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """base [R, H] + rows of src [N, H] added at index [N], fixed order, fp32 accumulator,
    one rounding to base.dtype."""
    return _IndexAddFixed.apply(base, index, src)


def scatter_add_fixed(base: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """base.scatter_add(0, index, src) for 1-D or 2-D tensors in the fixed order. The source
    order is row-major position in src. Not differentiable (use index_add_fixed for rows)."""
    if base.dim() == 1:
        return seg_sum(src.reshape(-1, 1), index, base.shape[0], base=base.reshape(-1, 1)).reshape(-1).to(base.dtype)
    R, H = base.shape
    lin = index.to(torch.int64) * H + torch.arange(H, device=base.device)
    return seg_sum(src.reshape(-1, 1), lin.reshape(-1), R * H, base=base.reshape(-1, 1)).reshape(R, H).to(base.dtype)


def bincount_fixed(index: torch.Tensor, weights: torch.Tensor, minlength: int) -> torch.Tensor:
    """Weighted bincount in the fixed order (fp32)."""
    return seg_sum(weights.reshape(-1, 1), index, minlength).reshape(-1)


# ----------------------------------------------------------------------------- fixed tree
def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def tree_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum over the last dim with the fixed pairwise tree. Positions that are not part of the
    sum must already hold +0.0."""
    n = x.shape[-1]
    p = _next_pow2(n)
    if p != n:
        x = torch.nn.functional.pad(x, (0, p - n))
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


def tree_mean_loss(per_token: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """per_token fp32 [B, T], mask bool [B, T] (True = counted). Fixed tree per sequence, then
    the same tree over sequences in batch order, then one division by the count."""
    if mask is not None:
        per_token = torch.where(mask, per_token, per_token.new_zeros(()))
        count = int(mask.sum())
    else:
        count = per_token.numel()
    return divide_by_count(tree_sum(tree_sum(per_token)), count)


def divide_by_count(total: torch.Tensor, count: int) -> torch.Tensor:
    """One IEEE fp32 division by the token count, with the count as an fp32 tensor. Measured:
    dividing a CUDA tensor by a Python number gave different bits from dividing by the same
    number held in a tensor (B=32 case of the loss section), so the definition fixes the form."""
    return total / torch.tensor(float(count), dtype=torch.float32, device=total.device)


def tree_mean_loss_packed(values: torch.Tensor, lengths) -> torch.Tensor:
    """The same reduction from a packed layout: values fp32 [sum(lengths)] holds the sequences
    one after another. Each sequence is moved to its own row (no arithmetic), then reduced."""
    lengths = torch.as_tensor(lengths, device=values.device)
    B, T = lengths.numel(), int(lengths.max())
    seq = torch.repeat_interleave(torch.arange(B, device=values.device), lengths)
    start = torch.zeros(B, dtype=torch.int64, device=values.device)
    start[1:] = torch.cumsum(lengths.cpu(), 0)[:-1].to(values.device)   # host cumsum of ints
    pos = torch.arange(values.numel(), device=values.device) - start[seq]
    grid = values.new_zeros(B, T)
    grid[seq, pos] = values
    return divide_by_count(tree_sum(tree_sum(grid)), values.numel())


# ----------------------------------------------------------------------------- cumsum
@triton.jit(do_not_specialize=["R", "T"])
def _seq_cumsum_kernel(x_ptr, out_ptr, R, T, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rmask = rows < R
    acc = tl.zeros([BR], tl.float32)
    for t in range(0, T):                              # strictly left to right
        acc += tl.load(x_ptr + rows * T + t, mask=rmask, other=0.0).to(tl.float32)
        tl.store(out_ptr + rows * T + t, acc, mask=rmask)


def seq_cumsum(x: torch.Tensor, BR: int = 32) -> torch.Tensor:
    """Prefix sums along the last dim, left to right, fp32 accumulator. x [R, T] or [T]."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    R, T = x2.shape
    out = torch.empty(R, T, device=x.device, dtype=torch.float32)
    _seq_cumsum_kernel[(triton.cdiv(R, BR),)](x2, out, R, T, BR=BR, num_warps=1)
    return out.reshape(shape)


# ----------------------------------------------------------------------------- fingerprints
def _words(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous().reshape(-1)
    if t.element_size() == 4:
        return t.view(torch.int32)
    if t.element_size() == 2:
        return t.view(torch.int16).to(torch.int32) & 0xFFFF
    if t.element_size() == 8:
        return t.view(torch.int32)
    return t.to(torch.int32)


def xor32(t: torch.Tensor) -> int:
    """XOR of the tensor's raw words (32-bit; 16-bit dtypes are zero-extended)."""
    w = _words(t)
    n = w.numel()
    if n == 0:
        return 0
    while n > 1:
        h = n // 2
        nxt = w[:h] ^ w[h:2 * h]
        if n % 2:
            nxt[0] ^= w[2 * h]
        w, n = nxt, h
    return int(w[0].item()) & 0xFFFFFFFF


def sum64(t: torch.Tensor) -> int:
    """Wrap-around int64 sum of the raw words: a second fingerprint that does not cancel pairs."""
    return int(_words(t).sum(dtype=torch.int64).item())


def fingerprint(t: torch.Tensor) -> str:
    return f"{xor32(t):08x}:{sum64(t) & 0xFFFFFFFFFFFFFFFF:016x}"
