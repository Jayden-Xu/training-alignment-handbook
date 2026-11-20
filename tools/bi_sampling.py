"""bi_sampling: a reference sampler whose result for a request does not depend on the batch.

For every row, the sampled token is a function of
    the row's logit bits, its seed, its absolute token position, a purpose tag,
    a stream index, and its sampling parameters (temperature, top-k, top-p)
and of nothing else: not the batch size, the row's index, what the other rows
hold or ask for, how many draws were made before, or which process runs it.

How each source of variance is removed

noise       Counter-based (Philox4x32-10): one 53-bit uniform per
            (seed, position, purpose, stream). No generator state exists, so
            preemption, recompute, chunked verification and resume in another
            process reproduce the same draw. `purpose` separates the token draw
            from the speculative accept and residual draws; `stream` separates
            the n samples of one request. Computed on the CPU in integers.
order       Top-k and top-p work on the order (value descending, index
            ascending), from `torch.sort(stable=True)`.
sums        Probabilities become integers: w = floor(exp(z - max z) * 2**S),
            with S = 52 - ceil(log2 V). Every sum and cumulative sum is int64,
            so it is exact and has no order to fix: any kernel, tile or batch
            gives the same bits. Totals stay below 2**52 and convert to float64
            exactly.
nucleus     Keep a token iff the integer mass before it is < top_p * total.
draw        R = floor(u * total); the token is the first index, in vocabulary
            order, whose cumulative integer weight exceeds R.
greedy      temperature 0: the lowest index among the maxima.

Floating-point functions it relies on

exp="torch" (default)  `torch.exp` on float32, elementwise: the same input must
            give the same bits wherever the element sits in the batch. No log
            anywhere. Everything else is one IEEE-754 operation per element
            (fp32 divide by the temperature, subtract, multiply by a power of
            two, floor; fp64 multiply per row), integer arithmetic, and the
            stability of the sort.
exp="basic" no transcendental function: exp is 2**n times a degree-7
            polynomial after a two-constant range reduction, each multiply and
            add its own elementwise op (nothing to contract into an FMA), and
            2**n is built from exponent bits. The result is then defined by
            IEEE-754 alone, so a CPU replays a GPU bit for bit. It costs about
            20 more elementwise passes over [B, V].
            The two modes are different samplers (their weights differ in the
            last bits): pick one per deployment.

What the integer weights cost in fidelity: a token whose probability is below
2**-S of the row maximum gets weight 0 and is never drawn (S = 34 at
V = 151,936, 37 at 32,000). The mass moved is below V * 2**-S <= 2**-33 of the
maximum's.

Logits must be finite or -inf (a masked token), with at least one finite value
per row. NaN and +inf are not supported.

Speculative decoding
    spec_verify_exact      the emitted token is always sample(target logits):
                           a draft token is accepted iff it equals that token.
                           Speculation cannot change what is sampled.
    spec_verify_rejection  the classical accept/reject rule with counter-based
                           accept and residual draws. Lossless and independent
                           of batch, chunk size and preemption, but the tokens
                           are a function of the draft logits too, so they
                           differ from decoding without a draft.

Verified on an A40 with torch 2.5.1: handbook/2-rl/sampling.md,
examples/sampling/sampling_bi.py.
"""
from __future__ import annotations

import math
import struct
from typing import NamedTuple, Optional, Tuple, Union

import torch

TOKEN, ACCEPT, RESIDUAL, DRAFT = 0, 1, 2, 3          # purposes; any other 32-bit value is free for callers

_M32 = 0xFFFFFFFF
_PHILOX_M0, _PHILOX_M1 = 0xD2511F53, 0xCD9E8D57
_PHILOX_W0, _PHILOX_W1 = 0x9E3779B9, 0xBB67AE85


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# exp="basic": fp32 constants of the classic expf range reduction. n * _LN2_HI is exact for
# |n| < 128 because _LN2_HI has seven trailing zero bits.
_INV_LN2 = _f32(0x3FB8AA3B)                           # 1.4426950
_LN2_HI = _f32(0x3F317180)                            # 0.693138123
_LN2_LO = _f32(0x3717F7D1)                            # 9.0580006e-06
_EXP_COEFFS = [1.0 / math.factorial(k) for k in range(7, -1, -1)]   # Horner order, 1/7! first

IntLike = Union[int, list, torch.Tensor]
FloatLike = Union[float, list, torch.Tensor]


# ----------------------------------------------------------------------------- counter-based noise
def _mulhilo32(m: int, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(high, low) 32-bit halves of m * x, for a 32-bit constant m and int64 x in [0, 2**32).
    Split so that no intermediate reaches 2**63: int64 overflow is never relied on."""
    t1 = (x & 0xFFFF) * m                             # < 2**48
    t2 = (x >> 16) * m                                # < 2**48
    s = t1 + ((t2 & 0xFFFF) << 16)                    # < 2**49
    return (t2 >> 16) + (s >> 32), s & _M32


def philox4x32(counter, key, rounds: int = 10):
    """Philox4x32 (Salmon et al., Random123). counter: 4 int64 tensors, key: 2 int64 tensors,
    all holding 32-bit values. Returns 4 int64 tensors of 32-bit values. Integer-only."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for r in range(rounds):
        if r:
            k0, k1 = (k0 + _PHILOX_W0) & _M32, (k1 + _PHILOX_W1) & _M32
        hi0, lo0 = _mulhilo32(_PHILOX_M0, c0)
        hi1, lo1 = _mulhilo32(_PHILOX_M1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


def _cpu_i64(v: IntLike, n: int) -> torch.Tensor:
    t = torch.as_tensor(v, dtype=torch.int64).cpu()
    return t.expand(n) if t.dim() == 0 else t.reshape(n)


def uniform53(seeds: IntLike, positions: IntLike, purpose: IntLike = TOKEN, stream: IntLike = 0,
              n: Optional[int] = None) -> torch.Tensor:
    """One 53-bit uniform integer per row, in [0, 2**53), keyed by (seed, position, purpose,
    stream) only. seed and position: 64 bits each; purpose and stream: 32 bits each.
    Computed on the CPU (a few hundred tiny integer ops); returns a CPU int64 tensor [n]."""
    if n is None:
        n = max((torch.as_tensor(t).numel() for t in (seeds, positions, purpose, stream)), default=1)
    s, p = _cpu_i64(seeds, n), _cpu_i64(positions, n)
    pu, st = _cpu_i64(purpose, n), _cpu_i64(stream, n)
    o0, o1, _, _ = philox4x32((p & _M32, (p >> 32) & _M32, pu & _M32, st & _M32),
                              (s & _M32, (s >> 32) & _M32))
    return (o0 << 21) | (o1 >> 11)


# ----------------------------------------------------------------------------- integer weights
def scale_bits(vocab: int) -> int:
    """S such that vocab * 2**S <= 2**52: every total is exact in int64 and in float64."""
    return 52 - max(vocab - 1, 1).bit_length()


def _exp_scaled_basic(d: torch.Tensor, S: int) -> torch.Tensor:
    """floor(exp(d) * 2**S) for float32 d <= 0, from IEEE fp32 multiply, add, subtract, floor and
    exponent bits only: exp(d) = 2**n * exp(y), n = round(d / ln 2), y = d - n * ln 2."""
    d = torch.clamp(d, min=-(S + 2) * _LN2_HI)        # below this the weight is 0 in any case
    n = torch.floor(d * _INV_LN2 + 0.5)
    y = (d - n * _LN2_HI) - n * _LN2_LO               # |y| <= 0.35
    acc = torch.full_like(y, _EXP_COEFFS[0])
    for c in _EXP_COEFFS[1:]:                         # Horner; mul_ and add_ are separate kernels
        acc.mul_(y).add_(c)
    pow2 = ((n.to(torch.int32) + (S + 127)) << 23).view(torch.float32)   # 2**(n + S), exact
    return acc.mul_(pow2).floor_()


def _weights(z: torch.Tensor, exp: str) -> torch.Tensor:
    """int64 [B, V] weights floor(exp(z - max z) * 2**S). The maximum gets exactly 2**S."""
    S = scale_bits(z.shape[-1])
    d = z - z.max(dim=-1, keepdim=True).values        # fp32, <= 0
    if exp == "torch":
        w = d.exp_().mul_(2.0 ** S).floor_()          # the scaling is exact
    elif exp == "basic":
        w = _exp_scaled_basic(d, S)
    else:
        raise ValueError("exp must be 'torch' or 'basic'")
    return w.to(torch.int64)


def _param(v, n: int, dtype, device) -> torch.Tensor:
    t = torch.as_tensor(v, dtype=dtype).to(device)
    return t.expand(n) if t.dim() == 0 else t.reshape(n)


def kept_weights(logits: torch.Tensor, temperature: FloatLike = 1.0, top_k: IntLike = 0,
                 top_p: FloatLike = 1.0, exp: str = "torch") -> Tuple[torch.Tensor, torch.Tensor]:
    """The sampling distribution of each row as exact integers: (w [B, V] int64, total [B] int64),
    with P(token i) = w[i] / total after temperature, top-k and top-p. top_k <= 0 disables top-k.
    A row with temperature 0 is treated as temperature 1 here (sample() returns its argmax)."""
    if logits.dim() != 2:
        raise ValueError("logits must be [B, V]")
    B, V = logits.shape
    dev = logits.device
    temp = _param(temperature, B, torch.float32, "cpu")
    temp = torch.where(temp == 0, torch.ones_like(temp), temp)
    k = _param(top_k, B, torch.int64, "cpu")
    k = torch.where(k <= 0, torch.full_like(k, V), k).clamp(max=V)
    p = _param(top_p, B, torch.float64, "cpu")
    truncate = bool((k < V).any()) or bool((p < 1).any())               # decided on the CPU: no sync

    z = logits.to(torch.float32) / temp.to(dev)[:, None]                 # fp32 IEEE divide
    w = _weights(z, exp)
    if truncate:
        k, p = k.to(dev), p.to(dev)
        order = torch.sort(z, dim=-1, descending=True, stable=True).indices   # (value desc, index asc)
        rank = torch.arange(V, device=dev)[None, :]
        in_k = rank < k[:, None]
        ws = torch.where(in_k, w.gather(1, order), 0)
        total = ws.sum(-1, keepdim=True)                                 # int64: exact, order-free
        before = ws.cumsum(-1).sub_(ws)                                  # integer mass before each token
        bound = torch.ceil(p[:, None] * total.to(torch.float64)).to(torch.int64)
        keep_sorted = in_k & (before < bound)
        keep = torch.zeros_like(keep_sorted).scatter_(1, order, keep_sorted)   # a permutation: exact
        w = torch.where(keep, w, 0)
    return w, w.sum(-1)


def _draw(w: torch.Tensor, total: torch.Tensor, u53: torch.Tensor) -> torch.Tensor:
    """First index, in vocabulary order, whose cumulative integer weight exceeds floor(u * total)."""
    u = u53.to(w.device).to(torch.float64) * (2.0 ** -53)                # exact
    r = torch.floor(u * total.to(torch.float64)).to(torch.int64)         # one fp64 multiply per row
    r = torch.minimum(r, total - 1)
    idx = torch.searchsorted(w.cumsum(-1), r[:, None], right=True)[:, 0]
    return idx.clamp_(max=w.shape[-1] - 1)


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """argmax with ties to the lowest index, by construction (not by a library's tie rule)."""
    V = logits.shape[-1]
    idx = torch.arange(V, device=logits.device)[None, :]
    is_max = logits == logits.max(dim=-1, keepdim=True).values
    return torch.where(is_max, idx, V).amin(-1)


class SampleInfo(NamedTuple):
    tokens: torch.Tensor      # [B] int64
    weight: torch.Tensor      # [B] int64, integer weight of the token
    total: torch.Tensor       # [B] int64, total kept weight: P(token) = weight / total


def sample(logits: torch.Tensor, seeds: IntLike, positions: IntLike, *, temperature: FloatLike = 1.0,
           top_k: IntLike = 0, top_p: FloatLike = 1.0, purpose: IntLike = TOKEN, stream: IntLike = 0,
           exp: str = "torch", return_info: bool = False):
    """Sample one token per row. logits [B, V]; seeds, positions, purpose, stream, temperature,
    top_k, top_p: a scalar or one value per row (pass them as CPU values: the noise is computed
    on the CPU). Returns tokens [B] int64, or SampleInfo with return_info=True."""
    B, V = logits.shape
    is_greedy = _param(temperature, B, torch.float32, "cpu") == 0
    any_greedy = bool(is_greedy.any())
    if any_greedy and not return_info and bool(is_greedy.all()):
        return greedy(logits.to(torch.float32))                          # same result, less work
    w, total = kept_weights(logits, temperature, top_k, top_p, exp)
    tokens = _draw(w, total, uniform53(seeds, positions, purpose, stream, n=B))
    if any_greedy:
        tokens = torch.where(is_greedy.to(logits.device), greedy(logits.to(torch.float32)), tokens)
    if not return_info:
        return tokens
    return SampleInfo(tokens, w.gather(1, tokens[:, None])[:, 0], total)


# ----------------------------------------------------------------------------- speculative decoding
def spec_verify_exact(target_logits: torch.Tensor, draft_tokens: torch.Tensor, seeds: IntLike,
                      positions: IntLike, **params) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verification that cannot change the sample. One row per position. Returns (tokens,
    accepted): tokens is sample(target_logits, ...) whatever the draft proposed, and a draft
    token is accepted iff it equals it. The engine keeps the accepted prefix plus the token at
    the first rejected position. No accept/reject draw is consumed."""
    tokens = sample(target_logits, seeds, positions, purpose=TOKEN, **params)
    return tokens, tokens == draft_tokens.to(tokens.device)


def spec_verify_rejection(target_logits: torch.Tensor, draft_logits: torch.Tensor,
                          draft_tokens: torch.Tensor, seeds: IntLike, positions: IntLike, *,
                          stream: IntLike = 0, exp: str = "torch", **params
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classical speculative sampling, one row per position: accept the draft token x with
    probability min(1, p(x) / q(x)), else draw from max(0, p - q). The accept draw is keyed by
    (seed, position, ACCEPT), the residual draw by (seed, position, RESIDUAL), so the result for
    a position is a function of its target logits, its draft logits, the draft token, the seed
    and the position. The draft token must come from sample(draft_logits, ..., purpose=DRAFT)
    with the same parameters for the output to follow p. Temperature 0 is not handled here: use
    spec_verify_exact for greedy requests. Returns (tokens, accepted)."""
    B = target_logits.shape[0]
    dev = target_logits.device
    wp, zp = kept_weights(target_logits, exp=exp, **params)
    wq, zq = kept_weights(draft_logits, exp=exp, **params)
    p = wp.to(torch.float64) / zp.to(torch.float64)[:, None]             # IEEE divide per element
    q = wq.to(torch.float64) / zq.to(torch.float64)[:, None]
    x = draft_tokens.to(dev).to(torch.int64)[:, None]
    u = uniform53(seeds, positions, ACCEPT, stream, n=B).to(dev).to(torch.float64) * (2.0 ** -53)
    accepted = u * q.gather(1, x)[:, 0] < p.gather(1, x)[:, 0]
    rw = torch.floor(torch.clamp(p - q, min=0.0) * (2.0 ** 51)).to(torch.int64)
    rtotal = rw.sum(-1)
    accepted = accepted | (rtotal == 0)                                  # p == q up to rounding
    resid = _draw(rw, rtotal.clamp(min=1), uniform53(seeds, positions, RESIDUAL, stream, n=B))
    return torch.where(accepted, x[:, 0], resid), accepted
