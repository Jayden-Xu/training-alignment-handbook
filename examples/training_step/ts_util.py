"""Shared helpers: modes, bitwise repeat checks, timing."""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
from bi_check import bit_equal, bits, first_mismatch  # noqa: E402,F401

DEV = "cuda"
BF = torch.bfloat16
MODES = ("default", "strict", "fixed")
CUBLAS_CFG = ":4096:8"


def mode_env(mode: str) -> dict:
    """Environment for a worker process in this mode. cuBLAS reads CUBLAS_WORKSPACE_CONFIG when
    its handle is created, so the variable is set per process, never changed inside one."""
    env = dict(os.environ)
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if mode == "strict":
        env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CFG
    return env


def apply_mode(mode: str) -> None:
    """default: stock torch. strict: use_deterministic_algorithms(True, warn_only=False).
    fixed: stock torch settings; the caller swaps in the fixed-order kernels."""
    assert mode in MODES, mode
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if mode == "strict":
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CFG
        torch.use_deterministic_algorithms(True, warn_only=False)
    else:
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        torch.use_deterministic_algorithms(False)


def leaf(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


def sync() -> None:
    torch.cuda.synchronize()


def timed_ms(fn, warm: int = 3, iters: int = 9) -> float:
    """Median wall time of fn() in ms, each run bracketed by a device synchronize."""
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def repeat_l1(run, n: int = 20) -> dict:
    """Run-to-run check: run() returns {name: tensor}; every run is compared bitwise with the
    first. Returns counts only (no tensors)."""
    ref = run()
    sync()
    bad_runs, per = 0, {}
    for i in range(1, n):
        out = run()
        bad = False
        for k, a in ref.items():
            mm = first_mismatch(a, out[k])
            if mm:
                bad = True
                e = per.setdefault(k, {"runs": 0, "max_elems": 0, "of": mm.get("of")})
                e["runs"] += 1
                e["max_elems"] = max(e["max_elems"], mm.get("mismatched_elements", 0))
        bad_runs += bad
    return {"runs": n, "differing_runs": bad_runs, "passed": bad_runs == 0, "tensors": per}


def err_line(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:220]}"


def _monotone(t: torch.Tensor) -> torch.Tensor:
    """Map float bit patterns to integers ordered like the floats (sign-magnitude -> signed)."""
    raw = bits(t).to(torch.int64)                 # negative iff the float's sign bit is set
    mag = raw & ((1 << (8 * t.element_size() - 1)) - 1)
    return torch.where(raw < 0, -mag, mag)


def ulp_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """How far apart two tensors of one dtype are: differing elements and the largest distance
    in units in the last place."""
    n = int((bits(a) != bits(b)).sum())
    out = {"mismatched": n, "of": a.numel()}
    if n:
        out["max_ulp"] = int((_monotone(a) - _monotone(b)).abs().max())
    return out
