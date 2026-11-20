"""bi_report: one result format for every example, so the regression suite can read them.

An example records checks (bitwise pass/fail facts) and measurements (costs), then
writes one JSON file in the ``bi-results-v1`` schema.

    from bi_report import Results
    res = Results("attention_backward", quick=args.quick)
    res.check("batch 1 vs 2..16", rep.passed, level="L2", impl="fixed_blocks", detail=rep.to_dict())
    res.check("batch 1 vs 2..16", False, level="L2", impl="sdpa_flash", expect="fail")
    res.measure("fwd+bwd B=4 T=2048", 2.94, "ms", impl="fixed_blocks")
    res.write(args.json)
    sys.exit(res.exit_code())

``expect`` says what a check is for:
    "pass"  the handbook claims this holds (our kernel or fix). A failure is a regression.
    "fail"  documented variance of a stock op or of a rejected design. A pass is a
            change in the stack: the claim needs a new look, it is not an error.
    "info"  recorded, never judged.

A check is a fact about bits: ``passed`` means zero mismatching bits, with no tolerance.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import Any, Optional

SCHEMA = "bi-results-v1"
LEVELS = ("L1", "L2", "L3", "L4", "L5", "other")
EXPECT = ("pass", "fail", "info")


def env_info() -> dict:
    """The stack a result was measured on. No host or user names."""
    info: dict = {"python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        try:
            info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception:
            info["nccl"] = None
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gpus.append({"name": p.name, "sm": f"{p.major}{p.minor}",
                             "memory_gb": round(p.total_memory / 2**30, 1),
                             "sm_count": getattr(p, "multi_processor_count", None)})
        info["gpus"] = gpus
    except ImportError:
        info["torch"] = None
        info["gpus"] = []
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        info["driver"] = out[0] if out else None
    except Exception:
        info["driver"] = None
    return info


def gpu_key(env: Optional[dict] = None) -> str:
    """Short name of the GPU set, e.g. ``A40`` or ``A16x8``. Used to name result files."""
    gpus = (env or env_info()).get("gpus") or []
    if not gpus:
        return "cpu"
    name = gpus[0]["name"].replace("NVIDIA", "").strip().replace(" ", "")
    return name if len(gpus) == 1 else f"{name}x{len(gpus)}"


class Results:
    def __init__(self, example: str, quick: bool = False, argv: Optional[list] = None):
        self.example, self.quick = example, bool(quick)
        self.argv = list(sys.argv if argv is None else argv)
        self.env = env_info()
        self.checks: list = []
        self.measurements: list = []
        self.extra: dict = {}

    def check(self, name: str, passed: bool, *, level: str, impl: str,
              expect: str = "pass", detail: Any = None) -> bool:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        if expect not in EXPECT:
            raise ValueError(f"expect must be one of {EXPECT}")
        passed = bool(passed)
        status = ("info" if expect == "info" else
                  "ok" if passed == (expect == "pass") else
                  "regression" if expect == "pass" else "changed")
        self.checks.append({"name": name, "impl": impl, "level": level, "expect": expect,
                            "passed": passed, "status": status, "detail": detail})
        return passed

    def measure(self, name: str, value: float, unit: str, **meta) -> None:
        self.measurements.append({"name": name, "value": value, "unit": unit, **meta})

    def counts(self) -> dict:
        out = {"ok": 0, "regression": 0, "changed": 0, "info": 0}
        for c in self.checks:
            out[c["status"]] += 1
        return out

    def exit_code(self) -> int:
        return 1 if self.counts()["regression"] else 0

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "example": self.example, "quick": self.quick, "argv": self.argv,
                "env": self.env, "counts": self.counts(), "checks": self.checks,
                "measurements": self.measurements, "extra": self.extra}

    def summary(self) -> str:
        lines = [f"{self.example} on {gpu_key(self.env)} "
                 f"(torch {self.env.get('torch')}, triton {self.env.get('triton')})"]
        for c in self.checks:
            mark = {"ok": "ok  ", "regression": "FAIL", "changed": "CHG ", "info": "info"}[c["status"]]
            lines.append(f"  [{mark}] {c['level']:5} {c['impl']:28} {c['name']}"
                         f"  (expect {c['expect']}, {'pass' if c['passed'] else 'fail'})")
        n = self.counts()
        lines.append(f"  {n['ok']} ok, {n['regression']} regressions, {n['changed']} changed, {n['info']} info")
        return "\n".join(lines)

    def write(self, path: Optional[str]) -> None:
        print(self.summary())
        if path:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=1, default=str)
            print(f"wrote {path}")
