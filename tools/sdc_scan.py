#!/usr/bin/env python
"""sdc_scan: check GPUs for silent data corruption (SDC) with bitwise fingerprints.

A device that gives different bits for identical inputs and an identical kernel is
faulty. The scan runs a fixed, seeded "golden workload" on every GPU, reduces each
output on the device to three fingerprints, and compares fingerprints only:

    python tools/sdc_scan.py --quick --json out.json     # 3 repeats + fault-injection self-test
    python tools/sdc_scan.py --json out.json             # full: 20 repeats, NCCL pairs, self-test
    python tools/sdc_scan.py --phases stress --stress 240 --part stress.json
    python tools/sdc_scan.py --merge a.json b.json --json fleet.json    # fleet across hosts
    python tools/sdc_scan.py --list-ops

Scan modes
  self-replay   N repeats per device; every repeat must equal the first.
  known answer  ops that do not round (integer, gather/scatter, sort, copies, single IEEE
                operations) must equal the same op run on the CPU.
  fleet         devices of one group must agree. A group is the same GPU model, SM count,
                memory size, driver, torch, CUDA build, cuDNN, cuBLAS, Triton and NCCL.
                Devices of different groups are never compared for a verdict.
  golden file   compare with fingerprints recorded earlier for the same group
                (--record-golden, --golden).
  stress        the same workload for a set time under a concurrent load, with every tensor
                at other device addresses in every pass, plus a memory sweep.
  NCCL pairs    with two or more GPUs: all-reduce, all-gather, broadcast and reduce-scatter
                between every pair of GPUs.

Fingerprints of a tensor, over its bytes read as 32-bit little-endian words w_0..w_{n-1}:
  xor32    XOR of the words (the fingerprint OpGuard uses);
  sum64    sum of the words as signed integers, modulo 2^64;
  wsum64   sum of (2i+1) * w_i, modulo 2^64.
xor32 misses two flips of the same bit; sum64 sees them unless they go in opposite
directions; wsum64 sees those, and also values that moved to another position.

Every op runs under torch.use_deterministic_algorithms(True, warn_only=False), with
CUBLAS_WORKSPACE_CONFIG=:4096:8 and TF32 off. Ops that are not run-to-run deterministic are
either pinned to a deterministic variant or kept as probes: probes are recorded as
information and never judged.

In-job checks (import as a library; importing changes no global state):
    from sdc_scan import sdc_guard, cross_rank_check, replica_check
    logits = sdc_guard(model, tokens)            # recompute the forward, compare fingerprints
    rep = cross_rank_check(loss_fn, batch)       # every rank computes rank 0's micro-batch
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import fnmatch
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

WORKLOAD_VERSION = 1
DEFAULT_SEED = 20261119
CUBLAS_WS = ":4096:8"
PART_SCHEMA = "sdc-part-v1"
GOLDEN_SCHEMA = "sdc-golden-v1"

F32, F64, F16, BF16 = torch.float32, torch.float64, torch.float16, torch.bfloat16
I64, I32, I16, I8, U8 = torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8
_DT = {F32: "f32", F64: "f64", F16: "f16", BF16: "bf16", I64: "i64", I32: "i32", I16: "i16",
       I8: "i8", U8: "u8", torch.bool: "b1"}
_M64 = (1 << 64) - 1


def _s64(v: int) -> int:
    v &= _M64
    return v - (1 << 64) if v >= (1 << 63) else v


def _s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


# ----------------------------------------------------------------------------- fingerprints
_COLS = 4096                 # words per row of the weighted sum
_CHUNK = 1 << 24             # words handled at once (64 MB)
_colw: Dict[Any, torch.Tensor] = {}


def _as_words(t: torch.Tensor) -> torch.Tensor:
    """The tensor's bytes as a 1-D int32 tensor, zero-padded to a multiple of 4 bytes."""
    t = t.detach()
    if t.dtype == torch.bool:
        t = t.to(U8)
    t = t.contiguous().reshape(-1)
    es = t.element_size()
    if es < 4 and t.numel() % (4 // es):
        t = torch.cat([t, t.new_zeros(4 // es - t.numel() % (4 // es))])
    return t.view(I32)


def _xor_fold(c: torch.Tensor) -> torch.Tensor:
    n = c.numel()
    while n > 1:
        h = n >> 1
        f = torch.bitwise_xor(c[:h], c[h:2 * h])
        if n & 1:
            f[0].bitwise_xor_(c[n - 1])
        c, n = f, h
    return c[0]


def fingerprint_tensor(t: torch.Tensor) -> torch.Tensor:
    """int64[3] on t's device: xor32, sum64, wsum64. Does not synchronize the device.

    The weighted sum uses row and column sums of the words laid out 4096 per row:
    sum (2i+1) w_i = 2*4096 * sum_r r*rowsum_r + sum_c (2c+1)*colsum_c."""
    w = _as_words(t)
    n, dev = w.numel(), w.device
    colw = _colw.get(dev)
    if colw is None:
        colw = _colw[dev] = 2 * torch.arange(_COLS, device=dev, dtype=I64) + 1
    x = torch.zeros((), dtype=I32, device=dev)
    s = torch.zeros((), dtype=I64, device=dev)
    ws = torch.zeros((), dtype=I64, device=dev)
    for o in range(0, n, _CHUNK):
        c = w[o:o + _CHUNK]
        m = c.numel()
        x = x ^ _xor_fold(c)
        c64 = c.to(I64)
        rows = m // _COLS
        if rows:
            b = c64[:rows * _COLS].view(rows, _COLS)
            rs, cs = b.sum(1), b.sum(0)
            sb = rs.sum()
            ws = ws + (2 * _COLS) * (torch.arange(rows, device=dev, dtype=I64) * rs).sum() \
                + (colw * cs).sum() + (2 * o) * sb
            s = s + sb
        tail = c64[rows * _COLS:]
        if tail.numel():
            st = tail.sum()
            ws = ws + (colw[:tail.numel()] * tail).sum() + (2 * (o + rows * _COLS)) * st
            s = s + st
    return torch.stack([x.to(I64) & 0xFFFFFFFF, s, ws])


def _flatten(obj, prefix=""):
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}/{i}" if prefix else str(i))
    elif obj is not None:
        raise TypeError(f"cannot fingerprint {type(obj).__name__}")


def _fp_text(t: torch.Tensor, vals: Sequence[int]) -> str:
    x, s, w = vals
    shape = ",".join(str(d) for d in t.shape)
    return f"{_DT.get(t.dtype, str(t.dtype))}[{shape}]:{x & 0xFFFFFFFF:08x}:{s & _M64:016x}:{w & _M64:016x}"


def fingerprint(obj) -> Dict[str, str]:
    """Fingerprints of a tensor or of a nested tuple/dict of tensors, as text:
    label -> 'dtype[shape]:xor32:sum64:wsum64'. Synchronizes the device."""
    items = list(_flatten(obj))
    fps = [fingerprint_tensor(t) for _, t in items]
    return {label: _fp_text(t, f.tolist()) for (label, t), f in zip(items, fps)}


def fingerprint_reference(t: torch.Tensor) -> str:
    """The same three fingerprints through numpy and unsigned 64-bit arithmetic. Used by the
    self-test to check the torch implementation on the CPU and on every device."""
    import numpy as np
    c = t.detach().cpu()
    if c.dtype == torch.bool:
        c = c.to(U8)
    raw = c.contiguous().reshape(-1).view(U8).numpy().tobytes()
    raw += b"\0" * (-len(raw) % 4)
    w = np.frombuffer(raw, dtype="<i4")
    u = w.astype(np.int64).astype(np.uint64)
    idx = np.arange(w.size, dtype=np.uint64)
    x = int(np.bitwise_xor.reduce(w.view("<u4"))) if w.size else 0
    s = int(u.sum(dtype=np.uint64)) if w.size else 0
    ws = int((u * (2 * idx + 1)).sum(dtype=np.uint64)) if w.size else 0
    return _fp_text(t, (x, s, ws))


def _fp_parts(text: str):
    return text.split(":")


def _diff_words(bad: torch.Tensor, good: torch.Tensor, limit: int = 4) -> dict:
    """Evidence for one output: which 32-bit words differ between two tensors, and how."""
    if bad.shape != good.shape or bad.dtype != good.dtype:
        return {"shape_or_dtype": [list(bad.shape), str(bad.dtype), list(good.shape), str(good.dtype)]}
    wb, wg = _as_words(bad), _as_words(good.to(bad.device))
    d = (wb != wg).nonzero().flatten()
    es = 1 if bad.dtype == torch.bool else bad.element_size()
    first = [int(i) for i in d[:limit].tolist()]
    return {"words_differing": int(d.numel()), "of_words": int(wb.numel()),
            "first": [{"word": i, "element": i * 4 // es,
                       "xor": f"{(int(wb[i]) ^ int(wg[i])) & 0xFFFFFFFF:08x}"} for i in first]}


# ----------------------------------------------------------------------------- inputs
def N(*shape, dtype=F32, scale=1.0):
    """Bell-shaped floats in (-2, 2) * scale. scale must be a power of two."""
    return ("normal", shape, dtype, scale)


def NZ(*shape, dtype=F32):
    """As N, with exact zeros replaced by 1 (safe divisors)."""
    return ("nonzero", shape, dtype, 1.0)


def U(*shape):
    """Distinct fp32 values in random order (no ties for sort, top-k, argmax)."""
    return ("unique", shape, F32, None)


def I(*shape, hi, lo=0, dtype=I64):     # noqa: E743
    return ("ints", shape, dtype, (lo, hi))


def R(*shape):
    """Random 64-bit integers."""
    return ("bits", shape, I64, None)


def P(n):
    return ("perm", (n,), I64, None)


def K(*shape, num=64):
    """Boolean mask, true with probability num/256."""
    return ("mask", shape, torch.bool, num)


def L(*shape):
    """fp32 with random mantissas, exponents in [-32, 31] and random signs."""
    return ("logscale", shape, F32, None)


def W(*shape):
    """Fixed memory-test words: zeros, ones, checkerboards, walking ones and zeros."""
    return ("patterns", shape, I64, None)


class Gen:
    """Seeded CPU generator. Integers come from torch's CPU generator. Floats are built from
    those integers with exact steps only (integer adds, int-to-float conversion, scaling by
    powers of two, IEEE rounding to the target dtype), so the bytes do not depend on a host's
    math library or SIMD level. A tensor's content depends on the seed and on its own name,
    not on which other ops are selected."""

    BASE_N = 1 << 25
    BITS_N = 1 << 23

    def __init__(self, seed: int):
        self.seed = seed
        g = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.randint(-(1 << 23), 1 << 23, (self.BASE_N,), generator=g, dtype=I32)
        a += torch.randint(-(1 << 23), 1 << 23, (self.BASE_N,), generator=g, dtype=I32)
        self.tri = a.to(F32).mul_(2.0 ** -23)            # |a| <= 2^24: exact
        hi = torch.randint(0, 1 << 32, (self.BITS_N,), generator=g, dtype=I64)
        lo = torch.randint(0, 1 << 32, (self.BITS_N,), generator=g, dtype=I64)
        self.bits = (hi << 32) | lo

    def _h(self, name: str) -> int:
        return int(hashlib.sha256(f"{self.seed}/{name}".encode()).hexdigest()[:15], 16)

    def _bits(self, name: str, n: int) -> torch.Tensor:
        if n <= self.BITS_N:
            o = self._h(name) % (self.BITS_N - n + 1)
            return self.bits[o:o + n].clone()
        t = self.bits.repeat(-(-n // self.BITS_N))[:n].clone()
        t ^= torch.arange(n, dtype=I64) * _s64(0x9E3779B97F4A7C15) + self._h(name)
        return t

    def make(self, name: str, spec) -> torch.Tensor:
        kind, shape, dtype, arg = spec
        n = math.prod(shape)
        if kind in ("normal", "nonzero"):
            o = self._h(name) % (self.BASE_N - n + 1)
            t = self.tri[o:o + n].clone()
            if dtype == F64:                              # 49 significant bits, still exact
                o2 = self._h(name + "#2") % (self.BASE_N - n + 1)
                t = t.to(F64) + self.tri[o2:o2 + n].to(F64) * 2.0 ** -24
            if arg != 1.0:
                t = t * arg
            if kind == "nonzero":
                t = torch.where(t == 0, torch.ones_like(t), t)
            return t.to(dtype).reshape(shape)
        if kind == "unique":
            g = torch.Generator(device="cpu").manual_seed(self._h(name))
            p = torch.randperm(n, generator=g)
            return ((p - n // 2).to(F32) * 2.0 ** -10).reshape(shape)       # n <= 2^24: exact, distinct
        if kind == "ints":
            lo, hi = arg
            return ((self._bits(name, n) & ((1 << 62) - 1)) % (hi - lo) + lo).to(dtype).reshape(shape)
        if kind == "bits":
            return self._bits(name, n).reshape(shape)
        if kind == "perm":
            return torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(self._h(name)))
        if kind == "mask":
            return ((self._bits(name, n) & 255) < arg).reshape(shape)
        if kind == "logscale":
            b = self._bits(name, n)
            word = (((b >> 31) & 1) << 31) | ((((b >> 24) & 63) + 95) << 23) | (b & ((1 << 23) - 1))
            return word.to(I32).view(F32).reshape(shape)                    # sign | exponent | mantissa
        if kind == "patterns":
            words = [0, -1, 0x5555555555555555, _s64(0xAAAAAAAAAAAAAAAA), 0x3333333333333333,
                     _s64(0xCCCCCCCCCCCCCCCC), 0x0F0F0F0F0F0F0F0F, _s64(0xF0F0F0F0F0F0F0F0)]
            words += [_s64(1 << k) for k in range(64)] + [_s64(~(1 << k)) for k in range(64)]
            return torch.tensor(words, dtype=I64).repeat(-(-n // len(words)))[:n].reshape(shape)
        raise ValueError(kind)


# ----------------------------------------------------------------------------- the golden workload
class Op:
    def __init__(self, name, cls, fn, inputs, exact, full, needs, probe, det, why):
        self.name, self.cls, self.fn, self.inputs = name, cls, fn, inputs
        self.exact, self.full, self.needs, self.probe, self.det, self.why = exact, full, needs, probe, det, why


OPS: List[Op] = []


def op(name, cls, exact=False, full=False, needs=(), probe=False, det=True, why="", **inputs):
    """Register an op of the golden workload.

    name     unique; also seeds the op's inputs
    cls      class shown in reports (gemm, attention, norm, ...)
    exact    the outputs involve no rounding choice, so the same function on the CPU gives
             the known answer
    full     only in full mode (heavy)
    needs    ("triton",) when the op uses tools/bi_kernels.py
    probe    an excluded candidate: recorded as information, never judged
    det      run under strict deterministic mode (False only for probes of default kernels)
    why      for probes and pinned ops: why
    inputs   keyword -> input spec (N, U, I, R, P, K, L, W)
    The function takes the inputs as device tensors and returns a tensor or a dict of tensors.
    It must not modify its inputs."""
    def deco(fn):
        if any(o.name == name for o in OPS):
            raise ValueError(f"duplicate op {name}")
        OPS.append(Op(name, cls, fn, inputs, exact, full, tuple(needs), probe, det, why))
        return fn
    return deco


def _bk():
    import bi_kernels
    return bi_kernels


def _sdpa_backends():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return SDPBackend, sdpa_kernel


sdpa = F.scaled_dot_product_attention


def _grads(fn, do, names, *leaves):
    leaves = [t.detach().clone().requires_grad_(True) for t in leaves]
    o = fn(*leaves)
    g = torch.autograd.grad(o, leaves, do)
    return {"o": o.detach(), **{n: x for n, x in zip(names, g)}}


# --- GEMM (cuBLAS through torch, and the one-config Triton GEMM). TF32 is off.
def _mm(a, b):
    return a @ b


for _n, _a, _b, _dt, _full in (
        ("gemm_fp32_1024", (1024, 1024), (1024, 1024), F32, False),
        ("gemm_fp32_odd", (1000, 777), (777, 1333), F32, False),
        ("gemm_fp32_4096", (4096, 4096), (4096, 4096), F32, True),
        ("gemm_bf16_2048", (2048, 2048), (2048, 2048), BF16, False),
        ("gemm_fp16_2048", (2048, 2048), (2048, 2048), F16, False),
        ("gemm_fp16_odd", (1000, 777), (777, 1333), F16, False),
        ("gemm_fp64_512", (512, 512), (512, 512), F64, False),
        ("bmm_bf16", (64, 128, 256), (64, 256, 128), BF16, False),
        ("bmm_fp32", (32, 200, 300), (32, 300, 100), F32, False)):
    op(_n, "gemm", full=_full, a=N(*_a, dtype=_dt), b=N(*_b, dtype=_dt))(_mm)


@op("gemm_bf16_tokens", "gemm", x=N(512, 4096, dtype=BF16), w=N(4096, 4096, dtype=BF16, scale=2.0 ** -6))
def _gemm_tokens(x, w):
    return {"m512": x @ w, "m31": x[:31] @ w, "m1": x[:1] @ w}


@op("gemm_fp32_tn", "gemm", x=N(2048, 1024), g=N(2048, 512))
def _gemm_tn(x, g):                                   # the shape of a weight gradient: x^T g
    return x.t() @ g


@op("gemv_fp32", "gemm", a=N(2048, 4096), v=N(4096))
def _gemv(a, v):
    return torch.mv(a, v)


@op("linear_bf16_bias", "gemm", x=N(1024, 2048, dtype=BF16), w=N(4096, 2048, dtype=BF16, scale=2.0 ** -5),
    b=N(4096, dtype=BF16))
def _linear(x, w, b):
    return F.linear(x, w, b)


@op("mm_one_config_bf16", "gemm", needs=("triton",), x=N(512, 1024, dtype=BF16),
    w=N(1024, 1024, dtype=BF16, scale=2.0 ** -5))
def _mm_triton(x, w):
    return {"m512": _bk().mm_one_config(x, w), "m1": _bk().mm_one_config(x[:1], w)}


# --- attention: SDPA backends, and the fixed-block kernels of tools/bi_kernels.py
_QKV = dict(q=N(2, 8, 512, 64, dtype=BF16), k=N(2, 8, 512, 64, dtype=BF16), v=N(2, 8, 512, 64, dtype=BF16))
_QKV32 = dict(q=N(2, 4, 256, 64), k=N(2, 4, 256, 64), v=N(2, 4, 256, 64))


@op("sdpa_flash_bf16", "attention", **_QKV)
def _sdpa_flash(q, k, v):
    B, ctx = _sdpa_backends()
    with ctx([B.FLASH_ATTENTION]):
        return sdpa(q, k, v, is_causal=True)


@op("sdpa_flash_fp16", "attention", q=N(2, 4, 256, 128, dtype=F16), k=N(2, 4, 256, 128, dtype=F16),
    v=N(2, 4, 256, 128, dtype=F16))
def _sdpa_flash16(q, k, v):
    B, ctx = _sdpa_backends()
    with ctx([B.FLASH_ATTENTION]):
        return sdpa(q, k, v)


@op("sdpa_efficient_fp32", "attention", **_QKV32)
def _sdpa_eff(q, k, v):
    B, ctx = _sdpa_backends()
    with ctx([B.EFFICIENT_ATTENTION]):
        return sdpa(q, k, v, is_causal=True)


@op("sdpa_math_fp32_mask", "attention", m=K(256, 256, num=192), **_QKV32)
def _sdpa_math(q, k, v, m):
    B, ctx = _sdpa_backends()
    with ctx([B.MATH]):
        return sdpa(q, k, v, attn_mask=m)


def _sdpa_bwd(backend_name):
    def run(q, k, v, do):
        B, ctx = _sdpa_backends()
        with ctx([getattr(B, backend_name)]):
            return _grads(lambda q, k, v: sdpa(q, k, v, is_causal=True), do, ("dq", "dk", "dv"), q, k, v)
    return run


_WHY_SDPA = ("SDPA backward adds dQ with atomics; measured in this handbook: dQ differs run to run "
             "unless strict deterministic mode is on")
op("sdpa_bwd_flash_det", "attention", why="pinned: strict deterministic mode selects flash's ordered dQ",
   do=N(2, 8, 512, 64, dtype=BF16), **_QKV)(_sdpa_bwd("FLASH_ATTENTION"))
op("sdpa_bwd_efficient_det", "attention", why="pinned: strict deterministic mode",
   do=N(2, 4, 256, 64), **_QKV32)(_sdpa_bwd("EFFICIENT_ATTENTION"))
op("sdpa_bwd_flash_default", "attention", probe=True, det=False, why=_WHY_SDPA,
   do=N(2, 8, 512, 64, dtype=BF16), **_QKV)(_sdpa_bwd("FLASH_ATTENTION"))
op("sdpa_bwd_efficient_default", "attention", probe=True, det=False, why=_WHY_SDPA,
   do=N(2, 4, 256, 64), **_QKV32)(_sdpa_bwd("EFFICIENT_ATTENTION"))

_GQA = dict(q=N(2, 8, 512, 128, dtype=BF16), k=N(2, 2, 512, 128, dtype=BF16), v=N(2, 2, 512, 128, dtype=BF16))


@op("attn_fixed_blocks_bf16", "attention", needs=("triton",), **_GQA)
def _afb(q, k, v):
    f = _bk().attention_fixed_blocks
    return {"prefill": f(q, k, v, 0), "decode": f(q[:, :, 511:].contiguous(), k, v, 511)}


@op("attn_fixed_blocks_train", "attention", needs=("triton",), do=N(2, 8, 512, 128, dtype=BF16), **_GQA)
def _afb_train(q, k, v, do):
    return _grads(_bk().attention_fixed_blocks_train, do, ("dq", "dk", "dv"), q, k, v)


# --- norms
def _ln(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w + 1, b)


op("layernorm_fp32", "norm", x=N(4096, 1024), w=N(1024, scale=0.25), b=N(1024, scale=0.25))(_ln)
op("layernorm_bf16", "norm", x=N(4096, 1024, dtype=BF16), w=N(1024, dtype=BF16, scale=0.25),
   b=N(1024, dtype=BF16, scale=0.25))(_ln)


@op("rmsnorm_bf16", "norm", x=N(4096, 1024, dtype=BF16), g=N(1024, dtype=BF16))
def _rmsnorm(x, g):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * g


@op("layernorm_bwd_fp32", "norm", x=N(4096, 1024), w=N(1024, scale=0.25), b=N(1024, scale=0.25), go=N(4096, 1024))
def _ln_bwd(x, w, b, go):
    return _grads(lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b), go, ("dx", "dw", "db"), x, w + 1, b)


# --- softmax and losses
@op("softmax_fp32", "softmax", x=N(512, 32000, scale=4.0))
def _softmax(x):
    return {"softmax": torch.softmax(x, -1), "log_softmax": torch.log_softmax(x, -1),
            "logsumexp": torch.logsumexp(x, -1)}


@op("softmax_bf16", "softmax", x=N(512, 32000, dtype=BF16, scale=4.0))
def _softmax_bf16(x):
    return {"softmax": torch.softmax(x, -1), "log_softmax": torch.log_softmax(x, -1)}


@op("cross_entropy_fp32", "softmax", x=N(256, 32000, scale=4.0), t=I(256, hi=32000))
def _ce(x, t):
    x = x.detach().clone().requires_grad_(True)
    per = F.cross_entropy(x, t, reduction="none")
    loss = per.mean()
    (g,) = torch.autograd.grad(loss, x)
    return {"per_token": per.detach(), "mean": loss.detach(), "grad": g}


# --- exp, log and the other elementwise functions
def _elem(x):
    p = x.abs() + 2.0 ** -12
    return {"exp": torch.exp(x), "log": torch.log(p), "expm1": torch.expm1(x), "log1p": torch.log1p(p),
            "tanh": torch.tanh(x), "sigmoid": torch.sigmoid(x), "erf": torch.erf(x), "gelu": F.gelu(x),
            "silu": F.silu(x), "rsqrt": torch.rsqrt(p), "sqrt": torch.sqrt(p), "sin": torch.sin(x),
            "cos": torch.cos(x), "pow": torch.pow(p, 1.5)}


op("elementwise_fp32", "exp_log", x=N(1 << 22))(_elem)
op("elementwise_bf16", "exp_log", x=N(1 << 22, dtype=BF16))(_elem)
op("elementwise_fp16", "exp_log", x=N(1 << 22, dtype=F16))(_elem)
op("elementwise_fp64", "exp_log", x=N(1 << 20, dtype=F64))(_elem)


@op("elementwise_wide_fp32", "exp_log", x=L(1 << 22))
def _wide(x):
    p = x.abs()
    return {"log": torch.log(p), "sqrt": torch.sqrt(p), "rsqrt": torch.rsqrt(p), "recip": torch.reciprocal(x),
            "square": x * x, "exp": torch.exp(x.clamp(-80, 80)), "tanh": torch.tanh(x)}


def _ieee(a, b):                                       # one correctly rounded operation per output
    # sqrt is left out: torch's CPU sqrt differed from the GPU's in 0.6-0.7% of elements on the
    # A40 host (fp32 and fp64), so the CPU cannot give its known answer. It stays in _elem.
    return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b,
            "max": torch.maximum(a, b), "min": torch.minimum(a, b), "neg": -a}


op("ieee_fp32", "exp_log", exact=True, a=N(1 << 22), b=NZ(1 << 22))(_ieee)
op("ieee_fp64", "exp_log", exact=True, a=N(1 << 20, dtype=F64), b=NZ(1 << 20, dtype=F64))(_ieee)
op("ieee_bf16", "exp_log", exact=True, a=N(1 << 22, dtype=BF16), b=NZ(1 << 22, dtype=BF16))(_ieee)
op("ieee_fp16", "exp_log", exact=True, a=N(1 << 22, dtype=F16), b=NZ(1 << 22, dtype=F16))(_ieee)


@op("cast", "integer", exact=True, x=N(1 << 22), i=R(1 << 22))
def _cast(x, i):
    return {"f32_bf16": x.to(BF16), "f32_f16": x.to(F16), "f32_f64": x.to(F64), "f32_i32": (x * 1024).to(I32),
            "i64_f32": i.to(F32), "i64_f64": i.to(F64), "i64_i32": i.to(I32), "bf16_f32": x.to(BF16).to(F32)}


# --- embedding gather and scatter
@op("embedding_bf16", "gather_scatter", exact=True, table=N(32000, 512, dtype=BF16), ids=I(8192, hi=32000))
def _emb(table, ids):
    return F.embedding(ids, table)


@op("gather_fp32", "gather_scatter", exact=True, x=N(2048, 4096), idx=I(2048, 512, hi=4096),
    rows=I(1024, hi=2048), cols=I(1024, hi=4096))
def _gather(x, idx, rows, cols):
    return {"gather": torch.gather(x, 1, idx), "index_select": x.index_select(0, rows), "advanced": x[rows, cols]}


@op("scatter_unique_fp32", "gather_scatter", exact=True, src=N(1 << 22), perm=P(1 << 22))
def _scatter_unique(src, perm):
    z = torch.zeros_like(src)
    return {"scatter": z.scatter(0, perm, src), "index_copy": z.index_copy(0, perm, src), "permute": src[perm],
            "index_put": z.index_put((perm,), src)}


@op("scatter_add_int64", "gather_scatter", exact=True, idx=I(1 << 22, hi=65536), src=R(1 << 22))
def _scatter_add_int(idx, src):                        # integer adds commute: any order gives the same bits
    z = torch.zeros(65536, dtype=I64, device=src.device)
    return {"scatter_add": z.scatter_add(0, idx, src), "index_add": z.index_add(0, idx, src),
            "bincount": torch.bincount(idx, minlength=65536), "index_put_acc": z.index_put((idx,), src, accumulate=True)}


@op("masked_fp32", "gather_scatter", exact=True, x=N(1 << 22), m=K(1 << 22, num=64), src=N(1 << 22))
def _masked(x, m, src):
    return {"masked_select": torch.masked_select(x, m), "nonzero": m.nonzero(), "where": torch.where(m, x, -x),
            "masked_fill": x.masked_fill(m, 0.0), "masked_scatter": x.masked_scatter(m, src)}


def _index_add_f(idx, src):
    z = torch.zeros(4096, device=src.device)
    return {"index_add": z.index_add(0, idx, src), "scatter_add": z.scatter_add(0, idx, src)}


_WHY_ACC = "float accumulation over duplicate indices: torch documents the default CUDA kernels as non-deterministic"
op("index_add_fp32_det", "gather_scatter", why="pinned: strict deterministic mode selects the ordered kernels",
   idx=I(1 << 20, hi=4096), src=N(1 << 20))(_index_add_f)
op("index_add_fp32_default", "gather_scatter", probe=True, det=False, why=_WHY_ACC,
   idx=I(1 << 20, hi=4096), src=N(1 << 20))(_index_add_f)


def _emb_bwd(table, ids, go):
    t = table.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(F.embedding(ids, t), t, go)
    return g


op("embedding_bwd_fp32", "gather_scatter", table=N(32000, 256), ids=I(16384, hi=4096), go=N(16384, 256))(_emb_bwd)
op("embedding_bwd_fp32_default", "gather_scatter", probe=True, det=False,
   why="listed in handbook chapter 1 as a racy kernel in production; probed here outside deterministic mode",
   table=N(32000, 256), ids=I(16384, hi=4096), go=N(16384, 256))(_emb_bwd)


# --- sort and top-k
@op("sort_fp32", "sort_topk", exact=True, x=U(1 << 22))
def _sort(x):
    v, i = torch.sort(x)
    return {"values": v, "indices": i, "argsort_desc": torch.argsort(x, descending=True)}


@op("sort_rows_fp32", "sort_topk", exact=True, x=U(256, 32000))
def _sort_rows(x):
    v, i = torch.sort(x, dim=-1, descending=True)
    tv, ti = torch.topk(x, 64, dim=-1)
    return {"values": v, "indices": i, "topk_values": tv, "topk_indices": ti, "argmax": x.argmax(-1),
            "max": x.amax(-1), "min": x.amin(-1)}


@op("sort_stable_int32", "sort_topk", exact=True, k=I(1 << 22, hi=1024, dtype=I32))
def _sort_stable(k):
    v, i = torch.sort(k, stable=True)
    u, inv, cnt = torch.unique(k, return_inverse=True, return_counts=True)
    return {"values": v, "indices": i, "unique": u, "inverse": inv, "counts": cnt}


@op("searchsorted_fp32", "sort_topk", exact=True, x=U(1 << 20), q=N(1 << 22, scale=256.0))
def _search(x, q):
    s = torch.sort(x).values
    return {"searchsorted": torch.searchsorted(s, q), "bucketize_right": torch.bucketize(q, s, right=True)}


# --- cumsum. Integer cumsum is exact; float cumsum on CUDA is excluded (probe only).
@op("cumsum_int64", "cumsum", exact=True, x=R(1 << 22))
def _cumsum_int(x):
    m = x.view(2048, 2048)
    return {"flat": x.cumsum(0), "dim0": m.cumsum(0), "dim1": m.cumsum(1), "int32": x.to(I32).cumsum(0),
            "cummax": torch.cummax(x, 0).values}


def _cumsum_f(x):
    m = x.view(2048, 2048)
    return {"flat": x.cumsum(0), "dim0": m.cumsum(0), "dim1": m.cumsum(1)}


_WHY_CUMSUM = ("torch documents float cumsum on CUDA as non-deterministic (pytorch issue 75240); "
               "strict deterministic mode raises an error for it")
op("cumsum_fp32", "cumsum", probe=True, det=False, why=_WHY_CUMSUM, x=N(1 << 22))(_cumsum_f)
op("cumsum_bf16", "cumsum", probe=True, det=False, why=_WHY_CUMSUM, x=N(1 << 22, dtype=BF16))(_cumsum_f)


# --- integer ops
def _int_arith(a, b, c):
    sh = c & (a.element_size() * 8 - 1)
    return {"muladd": a * b + c, "xorshift": (a ^ (a >> 13)) ^ (b << 7), "logic": (a & b) | (~c), "sub": a - b,
            "shift_var": torch.bitwise_right_shift(a, sh), "cmp": a < b,
            "minmax": torch.maximum(a, b) - torch.minimum(a, c)}


op("int_arith_int64", "integer", exact=True, a=R(1 << 22), b=R(1 << 22), c=R(1 << 22))(_int_arith)
op("int_arith_int32", "integer", exact=True, a=I(1 << 22, lo=-(1 << 31), hi=1 << 31, dtype=I32),
   b=I(1 << 22, lo=-(1 << 31), hi=1 << 31, dtype=I32), c=I(1 << 22, lo=-(1 << 31), hi=1 << 31, dtype=I32))(_int_arith)


@op("int_div_int64", "integer", exact=True, a=R(1 << 22), d=I(1 << 22, lo=1, hi=1000))
def _int_div(a, d):
    a = a >> 1
    d = torch.where((a & 2) == 0, d, -d)
    return {"floor": torch.div(a, d, rounding_mode="floor"), "trunc": torch.div(a, d, rounding_mode="trunc"),
            "remainder": torch.remainder(a, d), "fmod": torch.fmod(a, d)}


@op("int_small", "integer", exact=True, a=I(1 << 22, hi=256, dtype=U8), b=I(1 << 22, hi=256, dtype=U8),
    s=I(1 << 22, lo=-32768, hi=32768, dtype=I16))
def _int_small(a, b, s):
    return {"u8_muladd": a * b + a, "u8_logic": (a ^ b) & ~a, "i16_mul": s * s, "i8_sub": a.to(I8) - b.to(I8),
            "bool": (a > b) ^ (s < 0)}


@op("int_reduce", "integer", exact=True, x=R(1 << 22))
def _int_reduce(x):
    m = x.view(2048, 2048)
    return {"sum": x.sum(), "sum_dim": m.sum(1), "max": x.max(), "min_dim": m.amin(0), "argmax": x.argmax(),
            "count": (x > 0).sum()}


# --- memory patterns: write (upload), copy on the device five ways, complement, read back
def _mem(p):
    n = p.numel()
    c1 = p.clone()
    buf = torch.empty(n + 5, dtype=p.dtype, device=p.device)
    buf[5:].copy_(c1)                                  # copy to an address 40 bytes into a block
    c2 = buf[5:]
    c3 = c2.flip(0)
    c4 = torch.roll(c3, 12345 % n)
    c5 = c4.view(2, -1).t().contiguous()               # strided copy
    c6 = ~c5
    return {"clone": c1, "offset_copy": c2, "flip": c3, "roll": c4, "transpose": c5, "complement": c6,
            "readback": (~c6).cpu()}


for _n, _sz, _full in (("mem_roundtrip_4k", 1 << 9, False), ("mem_roundtrip_1m", 1 << 17, False),
                       ("mem_roundtrip_64m", 1 << 23, False), ("mem_roundtrip_256m", 1 << 25, True)):
    op(_n, "memory", exact=True, full=_full, p=R(_sz))(_mem)
op("mem_patterns_1m", "memory", exact=True, p=W(1 << 17))(_mem)


# --- long reductions
@op("reduce_fp32_32m", "reduction", x=N(1 << 25))
def _reduce(x):
    m = x.view(4096, -1)
    h = x.numel() // 2
    return {"sum": x.sum(), "mean": x.mean(), "sum_dim0": m.sum(0), "sum_dim1": m.sum(1),
            "norm": torch.linalg.vector_norm(x), "var": x.var(), "dot": torch.dot(x[:h], x[h:]),
            "logsumexp": torch.logsumexp(m, 1)}


op("reduce_fp64_4m", "reduction", x=N(1 << 22, dtype=F64))(_reduce)


def _reduce_lowp(x):
    m = x.view(4096, -1)
    return {"sum": x.sum(), "sum_f32": x.sum(dtype=F32), "mean_dim1": m.mean(1), "sum_dim0": m.sum(0),
            "norm_f32": torch.linalg.vector_norm(x.float())}


op("reduce_bf16_16m", "reduction", x=N(1 << 24, dtype=BF16))(_reduce_lowp)
op("reduce_fp16_16m", "reduction", x=N(1 << 24, dtype=F16))(_reduce_lowp)


def _minmax(x):                                        # reductions that do not round
    m = x.view(4096, -1)
    return {"max": x.max(), "min": x.min(), "amax_rows": m.amax(1), "amin_cols": m.amin(0)}


op("minmax_fp32_16m", "reduction", exact=True, x=N(1 << 24))(_minmax)
op("minmax_bf16_16m", "reduction", exact=True, x=N(1 << 24, dtype=BF16))(_minmax)


# --- device random numbers (dropout masks, sampling)
@op("rng_philox", "rng", anchor=I(4, hi=2))
def _rng(anchor):
    dev, n = anchor.device, 1 << 22
    g = torch.Generator(device=dev).manual_seed(DEFAULT_SEED)
    return {"rand": torch.rand(n, device=dev, generator=g), "randn": torch.randn(n, device=dev, generator=g),
            "randint": torch.randint(0, 32000, (n,), device=dev, generator=g),
            "bernoulli": torch.bernoulli(torch.full((n,), 0.1, device=dev), generator=g)}


# --- optimizer and a whole training step
@op("adamw_fp32", "train", p=N(1 << 22, scale=2.0 ** -3), g=N(1 << 22, scale=2.0 ** -6))
def _adamw(p, g):
    out = {}
    for mode in ("fused", "foreach"):
        w = p.detach().clone().requires_grad_(True)
        opt = torch.optim.AdamW([w], lr=1e-3, weight_decay=0.01, **{mode: True})
        for s in range(3):
            w.grad = g * (s + 1)
            opt.step()
        out[mode] = w.detach()
    return out


class TinyLM:
    """A small pre-norm transformer LM written with functional ops, so its weights can come from
    the seeded generator. Used by the train_step op and by examples/sdc_scan/guard_demo.py."""

    def __init__(self, vocab=4096, d=256, heads=4, layers=2, T=128):
        self.vocab, self.d, self.heads, self.layers, self.T = vocab, d, heads, layers, T
        s = {"emb": (vocab, d), "pos": (T, d)}
        for i in range(layers):
            s.update({f"l{i}.ln1.w": (d,), f"l{i}.ln1.b": (d,), f"l{i}.qkv": (3 * d, d), f"l{i}.proj": (d, d),
                      f"l{i}.ln2.w": (d,), f"l{i}.ln2.b": (d,), f"l{i}.fc1": (4 * d, d), f"l{i}.fc2": (d, 4 * d)})
        s.update({"lnf.w": (d,), "lnf.b": (d,)})
        self.shapes = s
        self.nparams = sum(math.prod(v) for v in s.values())

    def carve(self, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Leaf parameters cut from one flat tensor (norm weights are 1 + the slice)."""
        out, o = {}, 0
        for name, shape in self.shapes.items():
            n = math.prod(shape)
            t = flat[o:o + n].view(shape).clone()
            if name.endswith(".w"):
                t = t + 1.0
            out[name] = t.requires_grad_(True)
            o += n
        return out

    def logits(self, p, tok):
        B, T = tok.shape
        d, h = self.d, self.heads
        with torch.autocast("cuda", dtype=BF16):
            x = F.embedding(tok, p["emb"]) + p["pos"][:T]
            for i in range(self.layers):
                y = F.layer_norm(x, (d,), p[f"l{i}.ln1.w"], p[f"l{i}.ln1.b"])
                q, k, v = F.linear(y, p[f"l{i}.qkv"]).view(B, T, 3, h, d // h).permute(2, 0, 3, 1, 4)
                a = sdpa(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
                x = x + F.linear(a, p[f"l{i}.proj"])
                y = F.layer_norm(x, (d,), p[f"l{i}.ln2.w"], p[f"l{i}.ln2.b"])
                x = x + F.linear(F.gelu(F.linear(y, p[f"l{i}.fc1"])), p[f"l{i}.fc2"])
            x = F.layer_norm(x, (d,), p["lnf.w"], p["lnf.b"])
            return F.linear(x, p["emb"])

    def loss(self, p, tok, tgt):
        return F.cross_entropy(self.logits(p, tok).float().view(-1, self.vocab), tgt.reshape(-1))


TINY = TinyLM()


@op("train_step_bf16", "train", flat=N(TINY.nparams, scale=2.0 ** -5), tok=I(8, 128, hi=4096), tgt=I(8, 128, hi=4096))
def _train_step(flat, tok, tgt):
    p = TINY.carve(flat)
    opt = torch.optim.AdamW(list(p.values()), lr=1e-3, weight_decay=0.01, fused=True)
    losses = []
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = TINY.loss(p, tok, tgt)
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    return {"loss": torch.stack(losses), "grad": torch.cat([t.grad.reshape(-1) for t in p.values()]),
            "params": torch.cat([t.detach().reshape(-1) for t in p.values()])}


# NCCL pair collectives: (name, dtype, element counts). Rank roles, not rank numbers, pick the
# inputs, so every pair on every host computes the same thing.
NCCL_SIZES = (256, 1 << 16, 1 << 24)                  # elements: LL-sized, mid, large messages
NCCL_OPS = (("allreduce_sum_i32", I32), ("allreduce_sum_f32", F32), ("allreduce_sum_bf16", BF16),
            ("allgather_bf16", BF16), ("allgather_i64", I64), ("broadcast_f32", F32), ("reduce_scatter_f32", F32))

SELFTEST_OPS = ("gemm_fp32_1024", "gemm_bf16_2048", "gemm_fp16_2048", "layernorm_fp32", "elementwise_fp32",
                "int_arith_int64")


def select_ops(quick: bool, patterns: Optional[Sequence[str]] = None, have_triton: bool = True):
    """(ops to run, {skipped op: reason})."""
    run, skipped = [], {}
    for o in OPS:
        if patterns and not any(fnmatch.fnmatch(o.name, p) for p in patterns):
            continue
        if quick and o.full and not patterns:
            continue
        if "triton" in o.needs and not have_triton:
            skipped[o.name] = "Triton is not installed"
            continue
        run.append(o)
    return run, skipped


# ----------------------------------------------------------------------------- environment and groups
def _smi(fields: str) -> List[List[str]]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


_ECC = ("ecc.mode.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,"
        "ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total,retired_pages.pending")


def list_devices(only: Optional[Sequence[int]] = None) -> List[dict]:
    """GPUs from nvidia-smi, limited to CUDA_VISIBLE_DEVICES and to `only` (nvidia-smi indices)."""
    rows = _smi("index,uuid,name,pci.bus_id,driver_version,vbios_version,memory.total,memory.used," + _ECC)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [t.strip() for t in vis.split(",")] if vis else None
    devs = []
    for r in rows:
        if len(r) < 8 or not r[0].isdigit():
            continue
        d = {"index": int(r[0]), "uuid": r[1], "name": r[2], "pci": r[3].split(":", 1)[-1], "driver": r[4],
             "vbios": r[5], "memory_mib": r[6], "used_mib_at_start": r[7], "ecc_before": r[8:]}
        if tokens is not None and not any(t == r[0] or (t.startswith("GPU-") and r[1].startswith(t)) for t in tokens):
            continue
        if only is not None and d["index"] not in only:
            continue
        devs.append(d)
    return devs


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def software_key() -> dict:
    """The software half of a group key. Needs no CUDA context."""
    try:
        import triton
        tri = triton.__version__
    except Exception:
        tri = None
    try:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception:
        nccl = None
    return {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cublas": _pkg_version("nvidia-cublas-cu12"), "triton": tri, "nccl": nccl}


def group_id(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]


def _pin_settings():
    """The scan's fixed settings. Called in worker processes only, never on import."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


@contextlib.contextmanager
def _det(on: bool):
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(on, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


# ----------------------------------------------------------------------------- worker: one device
def _call(o: Op, pool: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    with _det(o.det):
        out = o.fn(**{k: pool[f"{o.name}.{k}"] for k in o.inputs})
    return {(f"{o.name}/{label}" if label else o.name): t for label, t in _flatten(out)}


def _apply_fault(t: torch.Tensor, kind: str, word: int, bit: int) -> dict:
    """Software fault injection into a tensor on the device, in place."""
    w = _as_words(t)
    if w.data_ptr() != t.data_ptr():
        raise RuntimeError("fault injection needs a contiguous tensor of a multiple of 4 bytes")
    word %= w.numel()
    mask = _s32(1 << bit)
    if kind in ("bitflip", "persistent", "input"):
        w[word] ^= mask
        return {"words": [word], "bit": bit}
    if kind in ("pair_same", "pair_opposite"):          # two flips of one bit: XOR cannot see them
        b = (w >> bit) & 1
        cand = (b == b[word]) if kind == "pair_same" else (b != b[word])
        cand[word] = False
        j = int(cand.nonzero()[0])
        w[word] ^= mask
        w[j] ^= mask
        return {"words": [word, j], "bit": bit}
    if kind == "swap":                                  # two values trade places
        j = int((w != w[word]).nonzero()[0])
        a, c = w[word].clone(), w[j].clone()
        w[word], w[j] = c, a
        return {"words": [word, j]}
    raise ValueError(kind)


def _sweep_pattern(c: int, n: int, dev) -> torch.Tensor:
    """Chunk c of the memory sweep: n int64 words whose value depends on their global index."""
    z = torch.arange(c * n, (c + 1) * n, device=dev, dtype=I64)
    z.mul_(_s64(0x9E3779B97F4A7C15)).add_(_s64(0xD1B54A32D192ED03))
    z.bitwise_xor_((z >> 29) & ((1 << 35) - 1))
    z.mul_(_s64(0xBF58476D1CE4E5B9))
    z.bitwise_xor_((z >> 32) & 0xFFFFFFFF)
    return z


class _Sweep:
    """Fill most of the free memory with an address-dependent pattern, hold it while the stress
    passes run, then verify it, complement it in place and verify again."""

    CHUNK = 1 << 25                                    # int64 words: 256 MiB

    def __init__(self, dev, reserve_mib: int):
        self.dev, self.bufs, self.t_write = dev, [], time.time()
        free = torch.cuda.mem_get_info(dev)[0] >> 20
        n = max(0, (free - reserve_mib) // 256)
        try:
            for c in range(n):
                self.bufs.append(_sweep_pattern(c, self.CHUNK, dev))
        except torch.cuda.OutOfMemoryError:
            pass
        torch.cuda.synchronize(dev)
        self.write_s = time.time() - self.t_write

    def verify(self) -> dict:
        t0 = time.time()
        held = t0 - self.t_write
        bad = []
        for phase in ("held", "complement"):
            cnt = []
            for c, b in enumerate(self.bufs):
                pat = _sweep_pattern(c, self.CHUNK, self.dev)
                if phase == "complement":
                    b.bitwise_not_()
                    pat.bitwise_not_()
                cnt.append((b != pat).sum())
                del pat
            for c, k in enumerate(torch.stack(cnt).tolist() if cnt else []):
                if k:
                    pat = _sweep_pattern(c, self.CHUNK, self.dev)
                    if phase == "complement":
                        pat.bitwise_not_()
                    bad.append({"phase": phase, "chunk": c, "addr": hex(self.bufs[c].data_ptr()),
                                **_diff_words(self.bufs[c], pat)})
        out = {"gib": len(self.bufs) * 0.25, "chunks": len(self.bufs), "held_s": round(held, 1),
               "write_s": round(self.write_s, 2), "verify_s": round(time.time() - t0, 2), "bad": bad}
        self.bufs = []
        return out


def _burner(stop: threading.Event, dev, counter: list):
    """Sustained load on a second stream: bf16 and fp32 GEMMs, exp, and a 256 MiB copy."""
    try:
        stream = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(stream):
            g = torch.Generator(device=dev).manual_seed(1)
            a = torch.randn(4096, 4096, device=dev, generator=g).to(BF16)
            b = torch.randn(2048, 2048, device=dev, generator=g)
            big = torch.randn(1 << 26, device=dev, generator=g)
            c, d, e = torch.empty_like(a), torch.empty_like(b), torch.empty_like(big)
            while not stop.is_set():
                torch.matmul(a, a, out=c)
                torch.matmul(b, b, out=d)
                torch.exp(b, out=d)
                e.copy_(big)
                stream.synchronize()
                counter[0] += 1
    except Exception as ex:                             # noqa: BLE001
        counter.append(f"{type(ex).__name__}: {ex}"[:300])


def _fp_selfcheck(dev) -> dict:
    """The torch fingerprints on the CPU and on the device against the numpy reference."""
    g = torch.Generator(device="cpu").manual_seed(7)
    cases, bad = 0, []
    for dt in (F32, F64, F16, BF16, I64, I32, I16, I8, U8, torch.bool):
        for n in (1, 3, 7, 4095, 4097, 100003):
            raw = torch.randint(0, 256, (n * 8,), generator=g, dtype=U8)
            t = (raw[:n] > 127) if dt == torch.bool else raw[:n * torch.empty((), dtype=dt).element_size()].view(dt)
            ref = fingerprint_reference(t)
            got = (fingerprint(t)[""], fingerprint(t.to(dev))[""])
            cases += 1
            if got != (ref, ref):
                bad.append({"dtype": _DT[dt], "n": n, "ref": ref, "cpu": got[0], "gpu": got[1]})
    big = torch.randint(-(1 << 31), 1 << 31, (_CHUNK + 5001,), generator=g, dtype=I32)   # crosses a chunk
    ref = fingerprint_reference(big)
    got = (fingerprint(big)[""], fingerprint(big.to(dev))[""])
    cases += 1
    if got != (ref, ref):
        bad.append({"dtype": "i32", "n": big.numel(), "ref": ref, "cpu": got[0], "gpu": got[1]})
    return {"cases": cases, "bad": bad}


def _worker_main(spec_path: str, slot: int) -> None:
    with open(spec_path) as f:
        spec = json.load(f)
    signal.alarm(int(spec["timeout"]))                  # never outlive the scan
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WS
    out: dict = {"phase": spec["phase"], "errors": []}
    try:
        _worker_body(spec, spec["devices"][slot], out)
    except BaseException as ex:                         # noqa: BLE001
        out["errors"].append({"where": "worker", "error": f"{type(ex).__name__}: {ex}"[:1500],
                              "trace": traceback.format_exc()[-1500:]})
    path = os.path.join(spec["tmpdir"], f"worker-{slot}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)


def _worker_body(spec: dict, me: dict, out: dict) -> None:
    t_start = time.time()
    _pin_settings()
    torch.set_num_threads(int(spec.get("threads", 4)))
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    try:
        import triton  # noqa: F401
        have_triton = True
    except Exception:
        have_triton = False
    key = {"gpu": props.name, "sm": f"{props.major}{props.minor}", "sm_count": props.multi_processor_count,
           "memory_mib": props.total_memory >> 20, "driver": me["driver"], **software_key()}
    out.update(group=key, gid=group_id(key), seed=spec["seed"], workload=WORKLOAD_VERSION)
    ops, skipped = select_ops(spec["quick"], spec.get("ops"), have_triton)
    out["skipped"] = skipped
    stress = spec.get("stress") or {}
    inject = [j for j in spec.get("inject", []) if j["device"] == me["index"]]

    # inputs: generated on the CPU, hashed, pinned
    gen = Gen(spec["seed"])
    pool = {f"{o.name}.{k}": gen.make(f"{o.name}.{k}", s) for o in ops for k, s in o.inputs.items()}
    sha = hashlib.sha256()
    for k in sorted(pool):
        t = pool[k].to(U8) if pool[k].dtype == torch.bool else pool[k]
        sha.update(k.encode())
        sha.update(t.contiguous().reshape(-1).view(U8).numpy().data)
    pool_fp = {k: fingerprint(v)[""] for k, v in pool.items()}
    out["inputs"] = {"sha256": sha.hexdigest()[:32], "tensors": len(pool),
                     "mib": round(sum(v.numel() * v.element_size() for v in pool.values()) / 2 ** 20, 1)}
    t_gen = time.time()

    # known answers: the exact ops on the CPU
    known: Dict[str, str] = {}
    known_err: Dict[str, str] = {}
    if spec.get("known", True):
        for o in ops:
            if o.exact and not o.probe:
                try:
                    known.update(fingerprint(_call(o, pool)))
                except Exception as ex:                 # noqa: BLE001
                    known_err[o.name] = f"{type(ex).__name__}: {ex}"[:300]
    t_known = time.time()
    del gen
    pool = {k: v.pin_memory() for k, v in pool.items()}
    out["fp_selfcheck"] = _fp_selfcheck(dev)

    lock = open(os.path.join(spec["tmpdir"], "triton.lock"), "w")
    ref: Dict[str, str] = {}
    odd: List[dict] = []
    seen: Dict[str, set] = {}
    pool_bad: List[dict] = []
    op_err: Dict[str, str] = {}
    injected: List[dict] = []
    addrs: Dict[str, set] = {k: set() for k in pool}
    pass_s: List[float] = []
    op_ms: Dict[str, float] = {o.name: 0.0 for o in ops}
    fp_s = upload_s = 0.0
    out["startup_s"] = {"generate": round(t_gen - t_start, 2), "known_answers": round(t_known - t_gen, 2),
                        "pin_and_selfcheck": round(time.time() - t_known, 2)}

    burn_stop, burn_count, burn_thread, sweep = threading.Event(), [0], None, None
    deadline = float("inf")
    repeats, rep = int(spec["repeats"]), 0
    t_loop = time.time()

    def more() -> bool:
        if not stress.get("seconds"):
            return rep < repeats
        return rep < 2 or time.time() < deadline

    def evidence(o: Op, key_: str, bad: Dict[str, torch.Tensor], dpool) -> dict:
        ev: dict = {}
        ins = [f"{o.name}.{k}" for k in o.inputs]
        ev["inputs_corrupted"] = [k for k in ins if fingerprint(dpool[k])[""] != pool_fp[k]]
        ev["recompute_matches_first"] = []
        for _ in range(3):
            again = _call(o, dpool)
            same = fingerprint(again[key_])[""] == ref.get(key_)
            ev["recompute_matches_first"].append(same)
            if same:
                ev["diff"] = _diff_words(bad[key_], again[key_])
                break
        return ev

    while more():
        if os.getppid() != spec["parent"]:
            raise RuntimeError("the scan's parent process is gone")
        t_pass = time.time()
        pad = None
        if stress.get("pad"):                           # other device addresses in every pass
            torch.cuda.empty_cache()
            mib = (37 * rep * rep + 101 * rep + 1) % int(stress.get("pad_max_mib", 1024))
            pad = torch.empty((mib << 20) + 512 * (rep % 5), dtype=U8, device=dev)
        t0 = time.time()
        dpool = {}
        for i, (k, v) in enumerate(pool.items()):
            # 256-byte steps keep the alignment fixed. Measured on the A40: offsets of 1-13 elements
            # aborted the process (exit -6, misaligned vector loads), and 16-byte steps changed the
            # bits of an fp64 sum over dim 0 between 16- and 32-byte aligned inputs (stable at each).
            off = (1 + (rep * 7 + i) % 13) * 256 // v.element_size() if stress.get("offsets") and v.dtype != torch.bool else 0
            if off:                                     # start the tensor `off` elements into its block
                buf = torch.empty(v.numel() + off, dtype=v.dtype, device=dev)
                buf[off:].copy_(v.reshape(-1), non_blocking=True)
                dpool[k] = buf[off:].view(v.shape)
            else:
                dpool[k] = v.to(dev, non_blocking=True)
            addrs[k].add(dpool[k].data_ptr())
        for j in inject:
            if j["kind"] == "input" and j["repeat"] == rep:
                injected.append({**j, **_apply_fault(dpool[j["op"]], "input", j["word"], j["bit"])})
        torch.cuda.synchronize(dev)
        upload_s += time.time() - t0

        for o in ops:
            if o.name in op_err:
                continue
            t0 = time.time()
            try:
                if "triton" in o.needs and rep == 0:    # one process compiles, the others read the cache
                    fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    res = _call(o, dpool)
                    torch.cuda.synchronize(dev)
                finally:
                    if "triton" in o.needs and rep == 0:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                t1 = time.time()
                for j in inject:
                    if j["kind"] != "input" and j["op"] in res and (j["repeat"] == rep or j["kind"] == "persistent"):
                        injected.append({**j, "repeat": rep, **_apply_fault(res[j["op"]], j["kind"], j["word"], j["bit"])})
                fps = fingerprint(res)
                t2 = time.time()
                op_ms[o.name] += (t1 - t0) * 1e3
                fp_s += t2 - t1
                if o.probe:
                    for k, v in fps.items():
                        seen.setdefault(k, set()).add(v)
                    continue
                for k, v in fps.items():
                    if k not in ref:
                        if rep:
                            odd.append({"key": k, "repeat": rep, "fp": v, "first": None})
                        ref[k] = v
                    elif v != ref[k]:
                        rec = {"key": k, "repeat": rep, "fp": v, "first": ref[k]}
                        try:
                            rec["evidence"] = evidence(o, k, res, dpool)
                        except Exception as ex:         # noqa: BLE001
                            rec["evidence"] = {"error": f"{type(ex).__name__}: {ex}"[:300]}
                        odd.append(rec)
                del res
            except Exception as ex:                     # noqa: BLE001
                op_err[o.name] = f"repeat {rep}: {type(ex).__name__}: {ex}"[:600]
        # the inputs must still be what the CPU generated
        t0 = time.time()
        for k, v in dpool.items():
            got = fingerprint(v)[""]
            if got != pool_fp[k]:
                pool_bad.append({"input": k, "repeat": rep, "fp": got, "cpu": pool_fp[k],
                                 **_diff_words(v, pool[k])})
        fp_s += time.time() - t0
        del dpool, pad
        pass_s.append(time.time() - t_pass)
        rep += 1
        if rep == 1 and stress.get("seconds"):          # the first pass ran alone and sized the workload
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated(dev) >> 20
            reserve = int(peak * 1.15) + int(stress.get("pad_max_mib", 1024)) + 1024 + (900 if stress.get("burner") else 0)
            out["stress_setup"] = {"first_pass_peak_mib": peak, "reserve_mib": reserve}
            if stress.get("sweep"):
                sweep = _Sweep(dev, reserve)
            if stress.get("burner"):
                burn_thread = threading.Thread(target=_burner, args=(burn_stop, dev, burn_count), daemon=True)
                burn_thread.start()
            deadline = time.time() + float(stress["seconds"])

    out["loop_s"] = round(time.time() - t_loop, 2)
    if burn_thread is not None:
        burn_stop.set()
        burn_thread.join(60)
        out["burner"] = {"iterations": burn_count[0], "errors": burn_count[1:]}
    if sweep is not None:
        out["sweep"] = sweep.verify()
    known_bad = [{"key": k, "cpu": v, "gpu": ref.get(k)} for k, v in known.items() if ref.get(k) != v]
    n_addr = sorted(len(a) for a in addrs.values())
    out.update(
        repeats=rep, fp=ref, odd=odd[:200], odd_total=len(odd), op_errors=op_err,
        known={"checked": len(known), "bad": known_bad[:50], "cpu_errors": known_err},
        pool={"checked": rep * len(pool), "bad": pool_bad[:50]},
        probes={k: len(v) for k, v in seen.items()}, injected=injected,
        addresses={"min": n_addr[0], "median": n_addr[len(n_addr) // 2], "max": n_addr[-1]} if n_addr else {},
        timing={"pass_s": [round(x, 3) for x in pass_s], "upload_s": round(upload_s, 2), "fingerprint_s": round(fp_s, 2),
                "op_ms": {k: round(v / max(rep, 1), 2) for k, v in op_ms.items()},
                "peak_mib": torch.cuda.max_memory_allocated(dev) >> 20, "total_s": round(time.time() - t_start, 2)})


# ----------------------------------------------------------------------------- worker: NCCL pairs
def _round_robin(n: int) -> List[List[tuple]]:
    """Rounds of disjoint pairs covering every pair of n ranks once."""
    ids = list(range(n)) + ([None] if n % 2 else [])
    rounds = []
    for _ in range(len(ids) - 1):
        pairs = [(ids[i], ids[-1 - i]) for i in range(len(ids) // 2)]
        rounds.append([tuple(sorted(p)) for p in pairs if None not in p])
        ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    return rounds


def _nccl_inputs(seed: int):
    gen = Gen(seed)
    n = max(NCCL_SIZES)
    ins = {}
    for role in (0, 1):
        ins[role] = {F32: gen.make(f"nccl.f32.{role}", N(n)), BF16: gen.make(f"nccl.bf16.{role}", N(n, dtype=BF16)),
                     I32: gen.make(f"nccl.i32.{role}", I(n, lo=-(1 << 31), hi=1 << 31, dtype=I32)),
                     I64: gen.make(f"nccl.i64.{role}", R(n))}
    return ins


def _nccl_expected(name: str, a: torch.Tensor, b: torch.Tensor, role: int) -> torch.Tensor:
    """What a healthy pair must hold after the collective (a: role 0's tensor, b: role 1's)."""
    if name.startswith("allreduce"):
        return a + b
    if name.startswith("allgather"):
        return torch.cat([a, b])
    if name.startswith("broadcast"):
        return a
    h = a.numel() // 2                                  # reduce-scatter: rank r keeps half r of the sum
    return (a + b)[role * h:(role + 1) * h]


def _nccl_worker_main(spec_path: str, rank: int) -> None:
    with open(spec_path) as f:
        spec = json.load(f)
    signal.alarm(int(spec["timeout"]))
    out: dict = {"rank": rank, "errors": [], "sides": {}}
    try:
        _nccl_body(spec, rank, out)
    except BaseException as ex:                         # noqa: BLE001
        out["errors"].append({"where": "nccl worker", "error": f"{type(ex).__name__}: {ex}"[:1500],
                              "trace": traceback.format_exc()[-1500:]})
    path = os.path.join(spec["tmpdir"], f"nccl-{rank}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)


def _nccl_body(spec: dict, rank: int, out: dict) -> None:
    import torch.distributed as dist
    t_start = time.time()
    _pin_settings()
    torch.set_num_threads(4)
    world = len(spec["devices"])
    dev = torch.device("cuda:0")
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{spec['port']}", rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=180))
    pairs = [(i, j) for i in range(world) for j in range(i + 1, world)]
    groups = {p: dist.new_group(list(p)) for p in pairs}          # every rank, the same order
    ins = _nccl_inputs(spec["seed"])
    sizes = [s for s in NCCL_SIZES if s <= int(spec.get("nccl_max", max(NCCL_SIZES)))]
    known: Dict[tuple, str] = {}
    for name, dt in NCCL_OPS:
        for n in sizes:
            for role in (0, 1):
                known[(name, n, role)] = fingerprint(_nccl_expected(name, ins[0][dt][:n], ins[1][dt][:n], role))[""]
    t_init = time.time()
    for rep in range(int(spec["repeats"])):
        for rnd in _round_robin(world):
            pair = next((p for p in rnd if rank in p), None)
            if pair is None:
                continue
            role, g = pair.index(rank), groups[pair]
            side = out["sides"].setdefault(f"{pair[0]}-{pair[1]}", {"fp": {}, "odd": [], "known_bad": [], "s": 0.0})
            t0 = time.time()
            for name, dt in NCCL_OPS:
                for n in sizes:
                    x = ins[role][dt][:n].to(dev)
                    if name.startswith("allreduce"):
                        dist.all_reduce(x, group=g)
                        res = x
                    elif name.startswith("allgather"):
                        res = torch.empty(2 * n, dtype=dt, device=dev)
                        dist.all_gather_into_tensor(res, x, group=g)
                    elif name.startswith("broadcast"):
                        dist.broadcast(x, src=pair[0], group=g)
                        res = x
                    else:
                        res = torch.empty(n // 2, dtype=dt, device=dev)
                        dist.reduce_scatter_tensor(res, x, group=g)
                    for j in spec.get("inject", []):
                        if j["rank"] == rank and j["op"] == name and j["n"] == n and j["repeat"] == rep \
                                and j["pair"] == list(pair):
                            _apply_fault(res, "bitflip", j["word"], j["bit"])
                    fp = fingerprint(res)[""]
                    key = f"{name}/{n}" + (f"/role{role}" if name.startswith("reduce_scatter") else "")
                    if key not in side["fp"]:
                        side["fp"][key] = fp
                        if fp != known[(name, n, role)]:
                            side["known_bad"].append({"key": key, "expected": known[(name, n, role)], "got": fp})
                    elif fp != side["fp"][key]:
                        side["odd"].append({"key": key, "repeat": rep, "fp": fp, "first": side["fp"][key]})
            side["s"] += time.time() - t0
    out.update(repeats=int(spec["repeats"]), init_s=round(t_init - t_start, 2), total_s=round(time.time() - t_start, 2))
    dist.destroy_process_group()


# ----------------------------------------------------------------------------- parent: run workers
_children: List[subprocess.Popen] = []


def _kill_children(*_):
    for p in _children:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    if _:
        raise SystemExit(143)


def run_procs(jobs: Sequence[tuple], timeout: float, poll: Optional[Callable] = None, fail_fast: bool = False) -> List[Optional[int]]:
    """Run (argv, env, logpath) jobs in parallel, each in its own session. Everything is killed on
    timeout, on exit, and (fail_fast) when one job fails."""
    procs, files = [], []
    try:
        for argv, env, log in jobs:
            f = open(log, "w")
            files.append(f)
            p = subprocess.Popen(argv, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(p)
            _children.append(p)
        t0 = time.time()
        while True:
            rcs = [p.poll() for p in procs]
            if all(rc is not None for rc in rcs) or time.time() - t0 > timeout:
                break
            if fail_fast and any(rc not in (None, 0) for rc in rcs):
                time.sleep(2)
                break
            if poll:
                poll()
            time.sleep(0.25)
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                p.wait()
        for f in files:
            f.close()
    return [p.returncode for p in procs]


class _Monitor:
    """Temperature, utilization and power from nvidia-smi while the workers run."""

    def __init__(self, every: float = 5.0):
        self.every, self.last, self.rows = every, 0.0, {}

    def poll(self):
        if time.time() - self.last < self.every:
            return
        self.last = time.time()
        for r in _smi("index,temperature.gpu,utilization.gpu,power.draw"):
            try:
                self.rows.setdefault(int(r[0]), []).append((float(r[1]), float(r[2]), float(r[3])))
            except (ValueError, IndexError):
                pass

    def summary(self, index: int) -> dict:
        rows = self.rows.get(index)
        if not rows:
            return {}
        return {"temp_c": [min(r[0] for r in rows), max(r[0] for r in rows)],
                "util_pct_mean": round(sum(r[1] for r in rows) / len(rows), 1),
                "power_w_max": max(r[2] for r in rows), "samples": len(rows)}


def _tail(path: str, n: int = 12) -> str:
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])[-1500:]
    except OSError:
        return ""


def _label(node: str, d: dict) -> str:
    return f"{node}:gpu{d['index']}"


def scan_devices(devs: List[dict], node: str, phase: str, repeats: int, seed: int, quick: bool,
                 ops: Optional[List[str]] = None, inject: Optional[List[dict]] = None,
                 stress: Optional[dict] = None, timeout: float = 1800, known: bool = True) -> List[dict]:
    """Run the golden workload on every device, one worker process per device. Returns one
    run record per device."""
    tmp = tempfile.mkdtemp(prefix="sdc-scan-")
    spec = {"devices": devs, "phase": phase, "repeats": repeats, "seed": seed, "quick": quick, "ops": ops,
            "inject": inject or [], "stress": stress, "tmpdir": tmp, "parent": os.getpid(), "timeout": int(timeout),
            "known": known, "threads": max(2, min(8, (os.cpu_count() or 8) // max(1, len(devs))))}
    spec_path = os.path.join(tmp, "spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    jobs = []
    for k, d in enumerate(devs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=d["uuid"], CUBLAS_WORKSPACE_CONFIG=CUBLAS_WS, PYTHONUNBUFFERED="1")
        jobs.append(([sys.executable, os.path.abspath(__file__), "--worker", spec_path, str(k)], env,
                     os.path.join(tmp, f"worker-{k}.log")))
    mon = _Monitor()
    t0 = time.time()
    rcs = run_procs(jobs, timeout + 30, poll=mon.poll)
    wall = time.time() - t0
    runs = []
    for k, d in enumerate(devs):
        path = os.path.join(tmp, f"worker-{k}.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
        else:
            r = {"phase": phase, "errors": [{"where": "worker", "error": f"no result, exit code {rcs[k]}",
                                             "log": _tail(os.path.join(tmp, f"worker-{k}.log"))}]}
        r.update(smi=mon.summary(d["index"]), wall_s=round(wall, 1))
        if stress:
            r["stress"] = {k_: v for k_, v in stress.items()}
        runs.append(r)
    shutil.rmtree(tmp, ignore_errors=True)
    return runs


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def scan_nccl(devs: List[dict], node: str, repeats: int, seed: int, inject: Optional[List[dict]] = None,
              timeout: float = 900, nccl_max: Optional[int] = None) -> dict:
    """All-reduce, all-gather, broadcast and reduce-scatter between every pair of GPUs of this
    host: one rank per GPU, one 2-rank NCCL group per pair."""
    tmp = tempfile.mkdtemp(prefix="sdc-nccl-")
    spec = {"devices": devs, "repeats": repeats, "seed": seed, "inject": inject or [], "tmpdir": tmp,
            "port": _free_port(), "timeout": int(timeout), "nccl_max": nccl_max or max(NCCL_SIZES)}
    spec_path = os.path.join(tmp, "spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    jobs = []
    for k, d in enumerate(devs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=d["uuid"], CUBLAS_WORKSPACE_CONFIG=CUBLAS_WS, PYTHONUNBUFFERED="1")
        jobs.append(([sys.executable, os.path.abspath(__file__), "--nccl-worker", spec_path, str(k)], env,
                     os.path.join(tmp, f"nccl-{k}.log")))
    t0 = time.time()
    rcs = run_procs(jobs, timeout + 30, fail_fast=True)
    sides, errors, init_s = [], [], []
    for k, d in enumerate(devs):
        path = os.path.join(tmp, f"nccl-{k}.json")
        if not os.path.exists(path):
            errors.append({"rank": k, "device": _label(node, d), "error": f"no result, exit code {rcs[k]}",
                           "log": _tail(os.path.join(tmp, f"nccl-{k}.log"))})
            continue
        with open(path) as f:
            r = json.load(f)
        errors += [{"rank": k, "device": _label(node, d), **e} for e in r["errors"]]
        init_s.append(r.get("init_s"))
        for pair, s in r["sides"].items():
            i, j = (int(v) for v in pair.split("-"))
            sides.append({"pair": [devs[i]["index"], devs[j]["index"]], "at": d["index"],
                          "label": f"{node}:gpu{devs[i]['index']}+gpu{devs[j]['index']}@gpu{d['index']}",
                          "repeats": r.get("repeats", repeats), **s})
    shutil.rmtree(tmp, ignore_errors=True)
    return {"sides": sides, "errors": errors, "wall_s": round(time.time() - t0, 1), "repeats": repeats,
            "pairs": len(devs) * (len(devs) - 1) // 2, "collectives_per_pair": len(NCCL_OPS) * len(NCCL_SIZES),
            "init_s_max": max([s for s in init_s if s is not None], default=None)}


# ----------------------------------------------------------------------------- judging
def _odd_summary(run: dict) -> List[dict]:
    """Odd repeats per fingerprint. When every later repeat differs from the first in the same
    way, the first repeat is the odd one."""
    by: Dict[str, List[dict]] = {}
    for o in run.get("odd", []):
        by.setdefault(o["key"], []).append(o)
    out = []
    for key, items in by.items():
        reps = [o["repeat"] for o in items]
        rec = {"key": key, "odd_repeats": reps, "first": items[0]["first"], "fp": items[0]["fp"]}
        if len(reps) == run["repeats"] - 1 and len({o["fp"] for o in items}) == 1 and run["repeats"] > 2:
            rec.update(odd_repeats=[0], fp=items[0]["first"], first=items[0]["fp"],
                       note="every later repeat agrees; the first repeat is the odd one")
        ev = next((o["evidence"] for o in items if o.get("evidence")), None)
        if ev:
            rec["evidence"] = ev
        out.append(rec)
    return out


def _canonical(runs: List[dict]) -> Dict[str, str]:
    """A unit's fingerprint per key: the value most repeats gave."""
    votes: Dict[str, Counter] = {}
    for run in runs:
        odd: Dict[str, List[str]] = {}
        for o in run.get("odd", []):
            odd.setdefault(o["key"], []).append(o["fp"])
        for key, fp in run.get("fp", {}).items():
            c = votes.setdefault(key, Counter())
            c[fp] += run.get("repeats", 1) - len(odd.get(key, []))
            for f in odd.get(key, []):
                c[f] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def _majority(values: List[tuple]) -> tuple:
    """(majority value or None, [labels that differ from it])."""
    c = Counter(v for _, v in values)
    top, n = c.most_common(1)[0]
    if len(c) == 1:
        return top, []
    if 2 * n <= len(values):
        return None, [lab for lab, _ in values]
    return top, [lab for lab, v in values if v != top]


PROCEDURE = ("A disagreement inside a group is a fact, not yet a verdict of SDC. 1. Rule out software: "
             "compare the group keys and the host stacks (driver, libraries, Python packages). 2. Repeat on the "
             "disagreeing device: a fingerprint that changes between repeats points at hardware; a stable one "
             "can be software or a hard fault. 3. To settle it: for an exact op the CPU's known answer says "
             "which side is right; otherwise move the card to another slot or host and repeat, and capture "
             "the op's output on both devices to see how the values differ (single flipped bits, or rounding "
             "differences across many elements).")


def judge(parts: List[dict], golden: Optional[dict] = None, cross_group: bool = False, quick: bool = False,
          argv: Optional[list] = None):
    """Turn raw facts (parts) into checks. Returns (Results, golden entries of the clean groups)."""
    from bi_report import Results
    res = Results("sdc_scan", quick=quick, argv=argv)
    units, pair_units, verdicts = [], [], []
    by_label: Dict[str, dict] = {}
    for part in parts:
        for u in part.get("units", []):
            if u["label"] in by_label:                  # the same device in several parts: one unit
                by_label[u["label"]]["runs"] += u["runs"]
                continue
            by_label[u["label"]] = u
            units.append(u)
        for n in part.get("nccl", []):
            n["node"] = part["node"]
            pair_units.append(n)
        for c in part.get("checks", []):
            res.check(c["name"], c["passed"], level=c.get("level", "other"), impl=c["impl"],
                      expect=c.get("expect", "pass"), detail=c.get("detail"))
        for m in part.get("measurements", []):
            res.measure(**m)

    canon: Dict[str, Dict[str, str]] = {}
    probes: Dict[str, Dict[str, int]] = {}
    for u in units:
        v = {"device": u["label"], "gpu": u["name"], "uuid": u["uuid"][:12], "pci": u["pci"], "group": u.get("gid"),
             "checks": {}}
        good = [r for r in u["runs"] if "fp" in r]
        v["ecc"] = {"before": u.get("ecc_before"), "after": u.get("ecc_after")}
        for run in u["runs"]:
            tag = run["phase"]
            if tag in v["checks"]:
                tag = f"{tag} #{sum(1 for t in v['checks'] if t.split(' #')[0] == run['phase']) + 1}"
            errs = run.get("errors", []) + [{"op": k, "error": e} for k, e in run.get("op_errors", {}).items()]
            if "fp" not in run:
                v["checks"][tag] = res.check(f"{tag}: the worker finished", False, level="other", impl=u["label"],
                                             detail={"errors": errs})
                continue
            odd = _odd_summary(run)
            name = f"{tag}: {run['repeats']} repeats x {len(run['fp'])} fingerprints, each equal to the first"
            if tag == "stress":
                st = run.get("stress", {})
                name = (f"stress {st.get('seconds')} s: {run['repeats']} passes x {len(run['fp'])} fingerprints, "
                        f"each equal to the first")
            ok = res.check(name, not odd and not errs, level="L1", impl=u["label"],
                           detail={"odd": odd[:10], "odd_total": run.get("odd_total", 0), "errors": errs[:5],
                                   "inputs_sha256": run["inputs"]["sha256"], "pass_s_median": _median(run["timing"]["pass_s"]),
                                   "addresses_per_tensor": run.get("addresses"), "burner": run.get("burner"),
                                   "smi": run.get("smi"), "skipped_ops": run.get("skipped")})
            kn, pl, sc = run["known"], run["pool"], run["fp_selfcheck"]
            ok &= res.check(f"{tag}: {kn['checked']} exact fingerprints equal the CPU's known answer", not kn["bad"],
                            level="other", impl=u["label"], detail={"bad": kn["bad"][:10]} if kn["bad"] else None)
            ok &= res.check(f"{tag}: inputs on the device still equal the CPU's after every pass "
                            f"({pl['checked']} tensor checks)", not pl["bad"], level="other", impl=u["label"],
                            detail={"bad": pl["bad"][:10]} if pl["bad"] else None)
            ok &= res.check(f"{tag}: fingerprint code equals the numpy reference on CPU and GPU ({sc['cases']} cases)",
                            not sc["bad"], level="other", impl=u["label"], detail={"bad": sc["bad"][:5]} if sc["bad"] else None)
            if "sweep" in run:
                sw = run["sweep"]
                ok &= res.check(f"stress: memory sweep {sw['gib']:.2f} GiB, held {sw['held_s']:.0f} s, verified, "
                                f"complemented, verified", not sw["bad"] and sw["chunks"] > 0, level="other",
                                impl=u["label"], detail=sw)
            v["checks"][tag] = bool(ok)
            for k, n in run.get("probes", {}).items():
                probes.setdefault(k, {})[f"{u['label']} {tag}"] = n
        canon[u["label"]] = _canonical(good)
        if len(good) > 1:                               # separate processes must agree as well
            diff = sorted({k for r in good for k, f in _canonical([r]).items() if canon[u["label"]].get(k) != f})
            v["checks"]["across runs"] = res.check(
                f"the {len(good)} runs of this device (separate processes) gave the same fingerprints",
                not diff, level="L1", impl=u["label"], detail={"differing": diff[:20]} if diff else None)
        u["_verdict"] = v
        verdicts.append(v)

    # every worker must have generated the same input bytes
    by_set: Dict[tuple, Dict[str, List[str]]] = {}
    for u in units:
        for r in u["runs"]:
            if "inputs" in r:
                by_set.setdefault((r["inputs"]["tensors"], r["inputs"]["mib"]), {}).setdefault(r["inputs"]["sha256"], []).append(u["label"])
    for (n_t, mib), shas_ in by_set.items():
        res.check(f"inputs: every worker generated the same bytes ({n_t} tensors, {mib} MiB, SHA-256)", len(shas_) == 1,
                  level="other", impl="inputs", detail={"sha256": {k: len(v_) for k, v_ in shas_.items()}})

    # fleet: inside a group only
    groups: Dict[str, List[dict]] = {}
    for u in units:
        if u.get("gid") and canon.get(u["label"]):
            groups.setdefault(u["gid"], []).append(u)
    ginfo, new_golden = {}, {}
    for gid, us in groups.items():
        keys = sorted({k for u in us for k in canon[u["label"]]})
        ref, out = {}, {u["label"]: [] for u in us}
        for k in keys:
            vals = [(u["label"], canon[u["label"]][k]) for u in us if k in canon[u["label"]]]
            ref[k], bad = _majority(vals)
            for lab in bad:
                out[lab].append({"key": k, "fp": canon[lab][k], "majority": ref[k]})
        shas = sorted({r["inputs"]["sha256"] for u in us for r in u["runs"] if "inputs" in r})
        ginfo[gid] = {"key": us[0]["group"], "devices": [u["label"] for u in us], "fingerprints": len(keys),
                      "inputs_sha256": shas, "fp": ref}
        if len(us) > 1:
            for u in us:
                u["_verdict"]["checks"]["fleet"] = res.check(
                    f"fleet: {len(canon[u['label']])} fingerprints equal the group's ({len(us)} devices, group {gid})",
                    not out[u["label"]], level="other", impl=u["label"],
                    detail={"differing": out[u["label"]][:10], "count": len(out[u["label"]]), "procedure": PROCEDURE}
                    if out[u["label"]] else None)
        clean = all(all(c for c in u["_verdict"]["checks"].values()) for u in us)
        if clean and all(v is not None for v in ref.values()):
            new_golden[gid] = {"key": us[0]["group"], "inputs_sha256": shas, "devices": len(us), "fp": ref}
        if golden is not None:
            g = golden.get("groups", {}).get(gid)
            for u in us:
                if g is None:
                    res.check(f"golden file: no entry for group {gid}", True, level="other", impl=u["label"], expect="info")
                    continue
                common = [k for k in canon[u["label"]] if k in g["fp"]]
                bad = [{"key": k, "fp": canon[u["label"]][k], "golden": g["fp"][k]} for k in common
                       if canon[u["label"]][k] != g["fp"][k]]
                u["_verdict"]["checks"]["golden"] = res.check(
                    f"golden file: {len(common)} fingerprints equal the ones recorded for group {gid}",
                    not bad and bool(common), level="other", impl=u["label"],
                    detail={"differing": bad[:10], "count": len(bad), "procedure": PROCEDURE} if bad else None)

    # NCCL pairs: per host
    for n in pair_units:
        node, sides = n["node"], n["sides"]
        impl = f"{node} nccl"
        per = f"{n['pairs']} pairs x {n['collectives_per_pair']} collectives"
        res.check(f"nccl pairs: every rank finished ({per})", not n["errors"] and bool(sides), level="other",
                  impl=impl, detail={"errors": n["errors"][:5]} if n["errors"] else None)
        if not sides:
            continue
        odd = [{"side": s["label"], **o} for s in sides for o in s["odd"]]
        res.check(f"nccl pairs: {n['repeats']} repeats, each equal to the first ({per}, both ranks)", not odd,
                  level="L1", impl=impl, detail={"odd": odd[:10]} if odd else None)
        kb = [{"side": s["label"], **o} for s in sides for o in s["known_bad"]]
        res.check(f"nccl pairs: results equal the known answer computed on the CPU ({per}, both ranks)", not kb,
                  level="other", impl=impl, detail={"bad": kb[:10]} if kb else None)
        keys = sorted({k for s in sides for k in s["fp"]})
        bad = []
        for k in keys:
            vals = [(s["label"], s["fp"][k]) for s in sides if k in s["fp"]]
            top, labs = _majority(vals)
            bad += [{"side": lab, "key": k, "majority": top} for lab in labs]
        res.check(f"nccl pairs: every pair and both of its ranks hold the same result ({len(sides)} rank views)",
                  not bad, level="other", impl=impl, detail={"differing": bad[:10], "procedure": PROCEDURE} if bad else None)
        failing = {s["label"] for s in sides if s["odd"] or s["known_bad"]} | {b["side"] for b in bad}
        for v in verdicts:
            if v["device"].startswith(node + ":"):
                idx = int(v["device"].rsplit("gpu", 1)[1])
                mine = [s for s in sides if idx in s["pair"]]
                if mine:
                    v["checks"]["nccl pairs"] = not any(s["label"] in failing for s in mine) and not n["errors"]
        gid = next((u.get("gid") for u in units if u["label"].startswith(node + ":") and u.get("gid")), None)
        if gid and not bad and not kb and not odd:
            ginfo.setdefault(gid, {}).setdefault("nccl_fp", {k: next(s["fp"][k] for s in sides if k in s["fp"]) for k in keys})
            if gid in new_golden:
                new_golden[gid]["nccl_fp"] = ginfo[gid]["nccl_fp"]
        if golden is not None and gid and golden.get("groups", {}).get(gid, {}).get("nccl_fp"):
            gf = golden["groups"][gid]["nccl_fp"]
            gb = [{"side": s["label"], "key": k} for s in sides for k, f in s["fp"].items() if k in gf and gf[k] != f]
            res.check(f"nccl pairs: results equal the golden file's for group {gid}", not gb, level="other", impl=impl,
                      detail={"differing": gb[:10]} if gb else None)

    # probes: excluded ops, information only
    why = {o.name: o.why for o in OPS if o.probe}
    for name in sorted({k.split("/")[0] for k in probes}):
        mine = {k: v for k, v in probes.items() if k.split("/")[0] == name}
        worst = max(max(v.values()) for v in mine.values())
        res.check(f"excluded op {name}: identical fingerprints over the repeats", worst == 1, level="L1", impl="probe",
                  expect="info", detail={"why_excluded": why.get(name), "max_distinct_fingerprints": worst,
                                         "distinct_by_output": {k: max(v.values()) for k, v in mine.items()}})

    # different groups: information only, never a verdict
    gids = sorted(ginfo)
    if cross_group:
        for a in range(len(gids)):
            for b in range(a + 1, len(gids)):
                fa, fb = ginfo[gids[a]].get("fp", {}), ginfo[gids[b]].get("fp", {})
                common = [k for k in fa if k in fb and fa[k] is not None and fb[k] is not None]
                diff = [k for k in common if fa[k] != fb[k]]
                ka, kb_ = ginfo[gids[a]]["key"], ginfo[gids[b]]["key"]
                res.check(f"observation, not a verdict: groups {gids[a]} and {gids[b]} gave the same fingerprints "
                          f"({len(common)} compared)", not diff and bool(common), level="other", impl="cross-group",
                          expect="info", detail={"key_differences": {k: [ka[k], kb_[k]] for k in ka if ka[k] != kb_.get(k)},
                                                 "differing": len(diff), "differing_keys": diff[:40]})
    for v in verdicts:
        v["verdict"] = "pass" if v["checks"] and all(v["checks"].values()) else "FINDING"
    res.extra.update(devices=verdicts, groups=ginfo, workload={"version": WORKLOAD_VERSION, "ops": _op_table()},
                     nodes=[{k: p.get(k) for k in ("node", "time", "env", "argv", "runtime_s")} for p in parts])
    return res, new_golden


def _median(xs: List[float]) -> Optional[float]:
    xs = sorted(xs[1:] or xs)                            # the first pass includes compilation
    return xs[len(xs) // 2] if xs else None


def _op_table() -> List[dict]:
    return [{"op": o.name, "class": o.cls, "mode": "probe" if o.probe else "full" if o.full else "quick+full",
             "exact": o.exact, **({"why": o.why} if o.why else {})} for o in OPS]


# ----------------------------------------------------------------------------- fault-injection self-test
def self_test(devs: List[dict], node: str, seed: int, timeout: float = 900) -> List[dict]:
    """Inject known faults in software and require the scan to name each one, and nothing else."""
    lab = [_label(node, d) for d in devs]
    A, C = devs[0]["index"], devs[-1]["index"]
    ops = list(SELFTEST_OPS)
    inj = [
        {"id": "transient", "device": A, "op": "gemm_fp32_1024", "repeat": 2, "kind": "bitflip", "word": 123457, "bit": 7},
        {"id": "first", "device": A, "op": "gemm_bf16_2048", "repeat": 0, "kind": "bitflip", "word": 77, "bit": 3},
        {"id": "persistent", "device": C, "op": "layernorm_fp32", "repeat": -1, "kind": "persistent", "word": 4242, "bit": 30},
        {"id": "pair_same", "device": A, "op": "elementwise_fp32/exp", "repeat": 1, "kind": "pair_same", "word": 1000, "bit": 12},
        {"id": "pair_opposite", "device": A, "op": "elementwise_fp32/tanh", "repeat": 1, "kind": "pair_opposite", "word": 1000, "bit": 12},
        {"id": "swap", "device": A, "op": "int_arith_int64/muladd", "repeat": 3, "kind": "swap", "word": 2000, "bit": 0},
        {"id": "input", "device": A, "op": "gemm_fp16_2048.a", "repeat": 2, "kind": "input", "word": 31337, "bit": 9},
    ]
    clean = scan_devices(devs, node, "self-test", 2, seed, True, ops=ops, timeout=timeout, known=False)
    runs = scan_devices(devs, node, "self-test", 4, seed, True, ops=ops, inject=inj, timeout=timeout, known=False)
    checks = []

    def add(name, passed, detail=None):
        checks.append({"name": f"self-test: {name}", "passed": bool(passed), "level": "other", "impl": "fault injection",
                       "expect": "pass", "detail": detail})

    if any("fp" not in r for r in clean + runs):
        add("the injected runs finished", False, {"errors": [r.get("errors") for r in clean + runs if "fp" not in r]})
        return checks
    golden = {lab[i]: _canonical([r]) for i, r in enumerate(clean)}
    found = {(lab[i], o["key"]): o for i, r in enumerate(runs) for o in _odd_summary(r)}
    a = lab[0]

    def comps(o):          # which of xor32 / sum64 / wsum64 changed
        x, y = _fp_parts(o["fp"]), _fp_parts(o["first"])
        return [n for n, i in (("xor32", 1), ("sum64", 2), ("wsum64", 3)) if x[i] != y[i]]

    o = found.get((a, "gemm_fp32_1024"))
    ev = (o or {}).get("evidence", {})
    add(f"a bit flip in one repeat is named: device {a}, op gemm_fp32_1024, repeat 2",
        o is not None and o["odd_repeats"] == [2] and comps(o) == ["xor32", "sum64", "wsum64"], o)
    add("its evidence is one word, the injected word and bit (word 123457, bit 7)",
        ev.get("diff", {}).get("words_differing") == 1 and ev["diff"]["first"][0] == {"word": 123457, "element": 123457, "xor": "00000080"}
        and ev.get("inputs_corrupted") == [], ev)
    o = found.get((a, "gemm_bf16_2048"))
    add(f"a bit flip in the first repeat is named as repeat 0: device {a}, op gemm_bf16_2048",
        o is not None and o["odd_repeats"] == [0], o)
    o = found.get((a, "elementwise_fp32/exp"))
    add("two flips of the same bit, same direction: xor32 does not see them, sum64 and wsum64 do (repeat 1 named)",
        o is not None and o["odd_repeats"] == [1] and comps(o) == ["sum64", "wsum64"], o)
    o = found.get((a, "elementwise_fp32/tanh"))
    add("two flips of the same bit, opposite directions: only wsum64 sees them (repeat 1 named)",
        o is not None and o["odd_repeats"] == [1] and comps(o) == ["wsum64"], o)
    o = found.get((a, "int_arith_int64/muladd"))
    add("two values swapped: only wsum64 sees it (repeat 3 named)",
        o is not None and o["odd_repeats"] == [3] and comps(o) == ["wsum64"], o)
    o = found.get((a, "gemm_fp16_2048"))
    pb = runs[0]["pool"]["bad"]
    add(f"a corrupted input is named: device {a}, input gemm_fp16_2048.a, repeat 2, and the op that read it",
        len(pb) == 1 and pb[0]["input"] == "gemm_fp16_2048.a" and pb[0]["repeat"] == 2 and pb[0]["words_differing"] == 1
        and o is not None and o["odd_repeats"] == [2] and o.get("evidence", {}).get("inputs_corrupted") == ["gemm_fp16_2048.a"],
        {"pool": pb, "op": o})
    # the persistent fault: every repeat agrees on the device, so only the golden file or the fleet can see it
    c = lab[-1]
    canon = {lab[i]: _canonical([r]) for i, r in enumerate(runs)}
    gbad = [(l, k) for l in lab for k, f in canon[l].items() if golden[l].get(k) != f and (l, k) not in found]
    add(f"a fault present in every repeat passes self-replay and is named by the golden file: device {c}, op layernorm_fp32",
        (c, "layernorm_fp32") not in found and gbad == [(c, "layernorm_fp32")], {"differs_from_golden": gbad})
    if len(devs) >= 3:
        vals = [(l, canon[l]["layernorm_fp32"]) for l in lab]
        top, out = _majority(vals)
        add(f"the same fault is named by the fleet majority: device {c} of {len(devs)}", out == [c], {"outliers": out})
    expected = {(a, "gemm_fp32_1024"), (a, "gemm_bf16_2048"), (a, "elementwise_fp32/exp"), (a, "elementwise_fp32/tanh"),
                (a, "int_arith_int64/muladd"), (a, "gemm_fp16_2048")}
    extra = sorted(set(found) - expected)
    add(f"nothing else is reported ({len(devs)} devices x {len(runs[0]['fp'])} fingerprints x 4 repeats)",
        not extra and all(not r["op_errors"] and not r["errors"] for r in runs), {"unexpected": extra})
    return checks


# ----------------------------------------------------------------------------- in-job checks
class SDCMismatch(RuntimeError):
    def __init__(self, report: dict):
        super().__init__(f"{report.get('name', 'sdc_guard')}: recomputed fingerprints differ: {report}")
        self.report = report


def _rng_get():
    return torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _rng_set(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def sdc_guard(fn: Callable, *inputs, repeats: int = 1, no_grad: bool = True, restore_rng: bool = True,
              on_mismatch: Any = "raise", name: str = "sdc_guard", **kwargs):
    """Call fn(*inputs) and return its result; then recompute it and compare fingerprints.

    For sampled training steps. fn must be deterministic and must not update state (no running
    statistics, no optimizer step). The random-number state is put back before each recompute
    and left afterwards where the first call left it, so the job's stream is unchanged.
    Cost: `repeats` extra calls of fn (without autograd when no_grad) plus two fingerprint
    passes over the outputs and one device synchronization.
    on_mismatch: "raise" (SDCMismatch), "return" (returns (out, report)), or a callable(report)."""
    state = _rng_get() if restore_rng else None
    out = fn(*inputs, **kwargs)
    items = list(_flatten(out))
    first = [fingerprint_tensor(t) for _, t in items]
    after = _rng_get() if restore_rng else None
    bad = []
    for r in range(repeats):
        if restore_rng:
            _rng_set(state)
        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            again = list(_flatten(fn(*inputs, **kwargs)))
        if len(again) != len(items):
            bad.append({"repeat": r + 1, "outputs": [len(items), len(again)]})
            continue
        eq = torch.stack([(a == fingerprint_tensor(t)).all() for a, (_, t) in zip(first, again)]).tolist()
        for (label, t), a, (_, t2), ok in zip(items, first, again, eq):
            if not ok:
                bad.append({"repeat": r + 1, "output": label, "first": _fp_text(t, a.tolist()),
                            "recomputed": fingerprint(t2)[""], **_diff_words(t2, t)})
        del again
    if restore_rng:
        _rng_set(after)
    report = {"name": name, "ok": not bad, "outputs": len(items), "repeats": repeats, "mismatches": bad}
    if on_mismatch == "return":
        return out, report
    if bad:
        if callable(on_mismatch):
            on_mismatch(report)
        else:
            raise SDCMismatch(report)
    return out


def _gather_rows(row: torch.Tensor, group=None) -> torch.Tensor:
    import torch.distributed as dist
    table = torch.empty(dist.get_world_size(group) * row.numel(), dtype=row.dtype, device=row.device)
    dist.all_gather_into_tensor(table, row, group=group)
    return table.view(dist.get_world_size(group), -1)


def _rank_report(table: torch.Tensor, n_in: int, ranks: Sequence[int]) -> dict:
    """Rows: [group hash, participated, input fingerprints (3 each), output fingerprints]."""
    rows = table.tolist()
    by_group: Dict[int, List[int]] = {}
    for r in ranks:
        if rows[r][1]:
            by_group.setdefault(rows[r][0], []).append(r)
    outliers, input_outliers, undecided = [], [], []
    for members in by_group.values():
        for lo, hi, dest in ((2, 2 + 3 * n_in, input_outliers), (2 + 3 * n_in, None, outliers)):
            top, bad = _majority([(r, tuple(rows[r][lo:hi])) for r in members])
            if top is None and len(members) > 1:
                undecided += [r for r in bad if r not in undecided]
            else:
                dest += bad
    return {"ok": not outliers and not input_outliers and not undecided, "outliers": sorted(outliers),
            "input_outliers": sorted(input_outliers), "undecided": sorted(undecided),
            "compared": {str(g): m for g, m in by_group.items()}, "groups": len(by_group)}


def _group_hash() -> int:
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    key = {"gpu": p.name, "sm_count": p.multi_processor_count, "memory_mib": p.total_memory >> 20, **software_key()}
    try:
        with open("/proc/driver/nvidia/version") as f:
            key["driver"] = f.readline().strip()
    except OSError:
        key["driver"] = None
    return int(group_id(key), 16)


def cross_rank_check(fn: Callable, *inputs, src: int = 0, ranks: Optional[Sequence[int]] = None, group=None,
                     no_grad: bool = True) -> dict:
    """Duplicate check for data-parallel jobs. Every rank calls this at the same step with its own
    micro-batch. Rank `src`'s tensors are broadcast, the ranks in `ranks` (default: all) compute fn
    on them, and the fingerprints of the received inputs and of the outputs are all-gathered and
    compared between ranks whose devices are in the same group. Two ranks can only say that they
    differ; three or more name the outlier. Tensor inputs must have the same shape and dtype on
    every rank. The random-number state is restored. Returns the same report on every rank."""
    import torch.distributed as dist
    me, world = dist.get_rank(group), dist.get_world_size(group)
    ranks = list(range(world)) if ranks is None else list(ranks)
    state = _rng_get()
    bufs = []
    for t in inputs:
        if isinstance(t, torch.Tensor):
            t = t.detach().clone()
            dist.broadcast(t, src=src, group=group)
        bufs.append(t)
    tin = [t for t in bufs if isinstance(t, torch.Tensor)]
    dev = tin[0].device
    if me in ranks:
        with (torch.no_grad() if no_grad else contextlib.nullcontext()):
            out = [t for _, t in _flatten(fn(*bufs))]
        row = torch.cat([torch.tensor([_group_hash(), 1], dtype=I64, device=dev)]
                        + [fingerprint_tensor(t) for t in tin + out])
        n_out = len(out)
    else:
        row, n_out = None, None
    # ranks that do not compute still take part in the gather with an empty row of the right size
    width = torch.tensor([0 if row is None else row.numel()], dtype=I64, device=dev)
    dist.all_reduce(width, op=dist.ReduceOp.MAX, group=group)
    if row is None:
        row = torch.zeros(int(width), dtype=I64, device=dev)
    table = _gather_rows(row, group)
    _rng_set(state)
    rep = _rank_report(table, len(tin), ranks)
    rep.update(src=src, ranks=ranks, outputs=n_out)
    return rep


def replica_check(tensors: Sequence[torch.Tensor], group=None) -> dict:
    """Data-parallel replicas hold the same parameters. Every rank fingerprints its copy and the
    fingerprints are compared between ranks of the same device group."""
    import torch.distributed as dist
    tensors = list(tensors)
    fps = torch.stack([fingerprint_tensor(t) for t in tensors])
    row = torch.cat([torch.tensor([_group_hash(), 1], dtype=I64, device=fps.device), fingerprint_tensor(fps)])
    rep = _rank_report(_gather_rows(row, group), 0, range(dist.get_world_size(group)))
    rep["tensors"] = len(tensors)
    return rep


# ----------------------------------------------------------------------------- command line
def _host_env() -> dict:
    """The stack of this host, read without creating a CUDA context."""
    drv = _smi("driver_version")
    try:
        import numpy
        npv = numpy.__version__
    except Exception:
        npv = None
    import platform
    return {"python": platform.python_version(), "numpy": npv, "driver": drv[0][0] if drv else None,
            "cublas_workspace_config": CUBLAS_WS, **software_key()}


def _print_verdicts(res) -> None:
    for v in res.extra.get("devices", []):
        marks = ", ".join(f"{k} {'ok' if ok else 'FAIL'}" for k, ok in v["checks"].items())
        print(f"  {v['device']:14} {v['gpu']:11} {v['uuid']:13} group {v['group']}  {v['verdict']:8} {marks}")
    for gid, g in res.extra.get("groups", {}).items():
        k = g.get("key", {})
        print(f"  group {gid}: {k.get('gpu')} sm{k.get('sm')} x{k.get('sm_count')} SMs, {k.get('memory_mib')} MiB, driver "
              f"{k.get('driver')}, torch {k.get('torch')}, CUDA {k.get('cuda_build')}, cuDNN {k.get('cudnn')}, cuBLAS "
              f"{k.get('cublas')}, Triton {k.get('triton')}, NCCL {k.get('nccl')}: {len(g.get('devices', []))} devices")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="self-replay with 3 repeats on the quick ops, plus the self-test")
    ap.add_argument("--json", help="write the bi-results-v1 file here")
    ap.add_argument("--phases", help="comma list of self-replay, nccl, stress, self-test "
                    "(default: quick: self-replay,self-test; full: self-replay,nccl,self-test; stress when --stress is set)")
    ap.add_argument("--repeats", type=int, help="self-replay repeats (default 20, quick 3)")
    ap.add_argument("--nccl-repeats", type=int, default=5)
    ap.add_argument("--stress", type=float, default=0.0, help="seconds of stress: load, padding offsets, memory sweep")
    ap.add_argument("--no-burner", action="store_true")
    ap.add_argument("--no-pad", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--offsets", action="store_true", help="stress: also start every input 256..3328 bytes into its block")
    ap.add_argument("--pad-max-mib", type=int, default=1024)
    ap.add_argument("--reserve-mib", type=int, default=7000, help="stress: memory the sweep leaves for the workload")
    ap.add_argument("--devices", help="nvidia-smi indices, comma separated (default: every visible GPU)")
    ap.add_argument("--ops", help="only these ops (comma list, shell patterns)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--node", default="node0", help="name of this host in the results")
    ap.add_argument("--golden", help="compare with this golden file")
    ap.add_argument("--record-golden", help="write the fingerprints of every clean group here")
    ap.add_argument("--part", help="write the raw facts here, for --merge")
    ap.add_argument("--merge", nargs="+", help="judge these part files together (fleet across hosts)")
    ap.add_argument("--cross-group-info", action="store_true", help="also record, as information, whether different groups agree")
    ap.add_argument("--inject", action="append", default=[], help="software fault for a manual test: "
                    "device=0,op=gemm_fp32_1024,repeat=1,kind=bitflip,word=5,bit=3")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    ap.add_argument("--nccl-worker", nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker:
        _worker_main(args.worker[0], int(args.worker[1]))
        return 0
    if args.nccl_worker:
        _nccl_worker_main(args.nccl_worker[0], int(args.nccl_worker[1]))
        return 0
    if args.list_ops:
        for o in _op_table():
            print(f"{o['op']:28} {o['class']:15} {o['mode']:11} exact={o['exact']}" + (f"  ({o['why']})" if o.get("why") else ""))
        print(f"nccl pairs: {', '.join(n for n, _ in NCCL_OPS)} at {NCCL_SIZES} elements")
        return 0

    golden = None
    if args.golden and os.path.exists(args.golden):
        with open(args.golden) as f:
            golden = json.load(f)
    elif args.golden:
        golden = {"groups": {}}

    if args.merge:
        parts = []
        for p in args.merge:
            with open(p) as f:
                parts.append(json.load(f))
        quick = all(p.get("quick") for p in parts)
    else:
        signal.signal(signal.SIGTERM, _kill_children)
        t_all = time.time()
        only = [int(x) for x in args.devices.split(",")] if args.devices else None
        devs = list_devices(only)
        if not devs:
            print("no GPU visible to nvidia-smi")
            return 2
        default = "self-replay,self-test" if args.quick else "self-replay,nccl,self-test"
        phases = (args.phases or (default + (",stress" if args.stress else ""))).split(",")
        repeats = args.repeats or (3 if args.quick else 20)
        ops = args.ops.split(",") if args.ops else None
        inject = []
        for s in args.inject:
            j = dict(kv.split("=", 1) for kv in s.split(","))
            inject.append({"device": int(j.get("device", devs[0]["index"])), "op": j["op"], "repeat": int(j.get("repeat", 1)),
                           "kind": j.get("kind", "bitflip"), "word": int(j.get("word", 0)), "bit": int(j.get("bit", 0))})
        part = {"schema": PART_SCHEMA, "node": args.node, "quick": args.quick, "argv": list(sys.argv),
                "time": datetime.datetime.now().isoformat(timespec="seconds"), "env": _host_env(),
                "units": [], "nccl": [], "checks": [], "measurements": []}
        units = {d["index"]: {"label": _label(args.node, d), "node": args.node, **d, "runs": []} for d in devs}
        runtime = {}

        def attach(runs):
            for d, r in zip(devs, runs):
                u = units[d["index"]]
                u["runs"].append(r)
                if r.get("gid"):
                    u["gid"], u["group"] = r["gid"], r["group"]

        if "self-replay" in phases:
            t0 = time.time()
            attach(scan_devices(devs, args.node, "self-replay", repeats, args.seed, args.quick, ops, inject,
                                timeout=args.timeout))
            runtime["self-replay"] = round(time.time() - t0, 1)
            print(f"self-replay: {len(devs)} devices, {repeats} repeats, {runtime['self-replay']} s")
        if "stress" in phases and args.stress > 0:
            t0 = time.time()
            stress = {"seconds": args.stress, "burner": not args.no_burner, "pad": not args.no_pad,
                      "sweep": not args.no_sweep, "offsets": args.offsets, "pad_max_mib": args.pad_max_mib,
                      "reserve_mib": args.reserve_mib}
            attach(scan_devices(devs, args.node, "stress", 2, args.seed, args.quick, ops, inject, stress,
                                timeout=args.timeout + args.stress))
            runtime["stress"] = round(time.time() - t0, 1)
            print(f"stress: {len(devs)} devices, {args.stress} s requested, {runtime['stress']} s")
        if "nccl" in phases and len(devs) >= 2:
            n = scan_nccl(devs, args.node, 2 if args.quick else args.nccl_repeats, args.seed, timeout=args.timeout)
            part["nccl"].append(n)
            runtime["nccl"] = n["wall_s"]
            print(f"nccl pairs: {n['pairs']} pairs, {n['repeats']} repeats, {n['wall_s']} s")
        if "self-test" in phases:
            t0 = time.time()
            part["checks"] += self_test(devs, args.node, args.seed, timeout=args.timeout)
            runtime["self-test"] = round(time.time() - t0, 1)
            print(f"self-test: {runtime['self-test']} s")
        ecc = {int(r[0]): r[1:] for r in _smi("index," + _ECC)}
        for d in devs:
            units[d["index"]]["ecc_after"] = ecc.get(d["index"])
        runtime["total"] = round(time.time() - t_all, 1)
        part["units"], part["runtime_s"] = list(units.values()), runtime
        for k, v in runtime.items():
            part["measurements"].append({"name": f"runtime {k}", "value": v, "unit": "s", "node": args.node,
                                         "devices": len(devs), "quick": args.quick})
        if args.part:
            with open(args.part, "w") as f:
                json.dump(part, f)
            print(f"wrote {args.part}")
        parts, quick = [part], args.quick

    res, new_golden = judge(parts, golden, args.cross_group_info, quick, list(sys.argv))
    _print_verdicts(res)
    res.write(args.json)
    if args.record_golden:
        g = {"schema": GOLDEN_SCHEMA, "workload": WORKLOAD_VERSION, "groups": {}}
        if os.path.exists(args.record_golden):
            with open(args.record_golden) as f:
                g = json.load(f)
        for gid, entry in new_golden.items():
            old = g["groups"].get(gid, {})
            entry["fp"] = {**old.get("fp", {}), **entry["fp"]}
            if "nccl_fp" in old and "nccl_fp" not in entry:
                entry["nccl_fp"] = old["nccl_fp"]
            g["groups"][gid] = entry
        with open(args.record_golden, "w") as f:
            json.dump(g, f, indent=1, sort_keys=True)
        print(f"recorded {len(new_golden)} clean group(s) in {args.record_golden}")
    return res.exit_code()


if __name__ == "__main__":
    sys.exit(main())
