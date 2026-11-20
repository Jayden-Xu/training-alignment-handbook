"""bi_check: bitwise invariance checks for one GPU op.

Checks that an op gives bit-identical results for a row whatever batch it runs
in (batch invariance), across repeated runs (run-to-run determinism), and across
alternative execution modes (prefill vs decode, chunk sizes, page sizes).

Equality is bitwise: tensors are compared as integers of the same width, so
-0.0 vs +0.0 and NaN payloads count as differences. There is no tolerance.

Library use:
    from bi_check import check_batch, check_repeat, check_modes
    rep = check_batch(lambda x: x @ W, rows)        # rows: [R, ...] on the GPU
    print(rep.summary())

Demo (needs CUDA):
    python bi_check.py --demo [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

DEFAULT_SIZES = (1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129,
                 255, 256, 257, 511, 512, 1024, 2048)
POSITIONS = ("first", "middle", "last")
FILLERS = ("random", "zeros", "large")


def bits(t: torch.Tensor) -> torch.Tensor:
    """The tensor's bit pattern as integers of the same width."""
    t = t.detach().contiguous()
    return t.view(_INT[t.element_size()])


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def first_mismatch(a: torch.Tensor, b: torch.Tensor) -> Optional[dict]:
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"shape_or_dtype": [list(a.shape), str(a.dtype), list(b.shape), str(b.dtype)]}
    diff = (bits(a) != bits(b)).nonzero()
    if diff.numel() == 0:
        return None
    idx = tuple(diff[0].tolist())
    return {"index": list(idx), "a": float(a[idx]), "b": float(b[idx]),
            "mismatched_elements": int(diff.shape[0]), "of": a.numel()}


@dataclass
class Report:
    name: str
    checked: int = 0
    failures: List[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        if self.passed:
            return f"{self.name}: PASS ({self.checked} comparisons)"
        sizes = sorted({f.get("size") for f in self.failures if f.get("size") is not None})
        modes = sorted({f.get("mode") for f in self.failures if f.get("mode") is not None})
        where = (f" failing batch sizes {sizes}" if sizes else "") + (f" failing modes {modes}" if modes else "")
        return f"{self.name}: FAIL {len(self.failures)}/{self.checked}{where}"

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "checked": self.checked,
                "failures": self.failures[:50]}


def _filler(kind: str, n: int, like: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    shape = (n,) + tuple(like.shape[1:])
    if kind == "zeros":
        return torch.zeros(shape, device=like.device, dtype=like.dtype)
    x = torch.randn(shape, generator=gen, device=like.device, dtype=torch.float32)
    return (x * 1e3 if kind == "large" else x).to(like.dtype)


def check_batch(fn: Callable[[torch.Tensor], torch.Tensor], rows: torch.Tensor,
                sizes: Sequence[int] = DEFAULT_SIZES, positions: Sequence[str] = POSITIONS,
                fillers: Sequence[str] = FILLERS, seed: int = 0, name: str = "op") -> Report:
    """Row r of fn(batch) must equal fn(rows[r:r+1])[0] bit for bit, for every batch
    size, every position of the row in the batch, and whatever the other rows hold.
    `fn` maps a [M, ...] tensor to a [M, ...] tensor (close over weights)."""
    rep = Report(name)
    gen = torch.Generator(device=rows.device).manual_seed(seed)
    for r in range(rows.shape[0]):
        row = rows[r:r + 1]
        ref = fn(row)[0]
        for size in sizes:
            for pos in (positions if size > 1 else ("first",)):
                p = {"first": 0, "middle": size // 2, "last": size - 1}[pos]
                for kind in (fillers if size > 1 else ("random",)):
                    others = _filler(kind, size - 1, rows, gen) if size > 1 else rows[:0]
                    batch = torch.cat([others[:p], row, others[p:]])
                    out = fn(batch)[p]
                    rep.checked += 1
                    mm = first_mismatch(ref, out)
                    if mm:
                        rep.failures.append({"row": r, "size": size, "position": pos, "filler": kind, **mm})
    return rep


def check_repeat(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, n: int = 5,
                 name: str = "op") -> Report:
    """Run-to-run determinism: the same input gives the same bits every time."""
    rep = Report(name)
    ref = fn(x)
    for i in range(1, n):
        rep.checked += 1
        mm = first_mismatch(ref, fn(x))
        if mm:
            rep.failures.append({"run": i, **mm})
    return rep


def check_modes(modes: Dict[str, Callable[[], torch.Tensor]], name: str = "op") -> Report:
    """Alternative execution modes (e.g. prefill, decode, chunked prefill with several
    chunk sizes) must produce the same bits. Every mode is compared to the first."""
    rep = Report(name)
    items = list(modes.items())
    base_name, base = items[0][0], items[0][1]()
    for mode, run in items[1:]:
        rep.checked += 1
        mm = first_mismatch(base, run())
        if mm:
            rep.failures.append({"mode": f"{mode} vs {base_name}", **mm})
    return rep


# ----------------------------------------------------------------------------- demo
def _demo(sizes: Sequence[int]) -> List[Report]:
    torch.manual_seed(0)
    dev, bf = "cuda", torch.bfloat16
    H, V = 4096, 32000
    W = (torch.randn(H, H, device=dev) / 64).to(bf)
    g = torch.randn(H, device=dev).to(bf)
    rows = torch.randn(2, H, device=dev).to(bf)
    reports = []

    reports.append(check_batch(lambda x: x @ W, rows, sizes, name="matmul (torch.mm / cuBLAS)"))
    small = [s for s in sizes if s <= 129]
    reports.append(check_batch(lambda x: torch.cat([x[i:i + 1] @ W for i in range(x.shape[0])]), rows, small,
                               positions=("first", "last"), fillers=("random",),
                               name="matmul row by row (invariant by construction)"))

    def rmsnorm(x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(bf) * g
    reports.append(check_batch(rmsnorm, rows, sizes, name="rmsnorm (torch ops)"))

    logits_rows = torch.randn(2, V, device=dev)
    reports.append(check_batch(lambda x: torch.log_softmax(x, -1), logits_rows, sizes,
                               name="log_softmax fp32 (torch)"))

    reports.append(check_repeat(lambda x: x @ W, torch.randn(512, H, device=dev).to(bf),
                                name="matmul run-to-run"))

    # prefill vs decode for causal attention: position t attends to keys 0..t
    T, NH, D = 64, 8, 128
    q, k, v = (torch.randn(1, NH, T, D, device=dev).to(bf) for _ in range(3))
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def prefill():
        return sdpa(q, k, v, is_causal=True)

    def decode():
        return torch.cat([sdpa(q[:, :, t:t + 1], k[:, :, :t + 1], v[:, :, :t + 1]) for t in range(T)], 2)

    def chunked(c):
        def run():
            outs = []
            for s in range(0, T, c):
                e = min(s + c, T)
                qs = q[:, :, s:e]
                mask = torch.ones(e - s, e, device=dev, dtype=torch.bool).tril(diagonal=s)
                outs.append(sdpa(qs, k[:, :, :e], v[:, :, :e], attn_mask=mask))
            return torch.cat(outs, 2)
        return run
    reports.append(check_modes({"prefill": prefill, "decode": decode,
                                "chunked-16": chunked(16), "chunked-7": chunked(7)},
                               name="causal attention (torch SDPA) prefill/decode/chunked"))
    return reports


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true", help="check a few stock torch ops on this GPU")
    ap.add_argument("--sizes", type=int, nargs="*", default=list(DEFAULT_SIZES))
    ap.add_argument("--json", help="write the reports here")
    args = ap.parse_args(argv)
    if not args.demo:
        ap.print_help()
        return
    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    reports = _demo(args.sizes)
    for r in reports:
        print(r.summary())
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                       "reports": [r.to_dict() for r in reports]}, f, indent=1)


if __name__ == "__main__":
    main()
