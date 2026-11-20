"""Adapters: the first eight examples' own result formats -> bi-results-v1.

The examples written before tools/bi_report.py keep their result files: the committed
results-*.json, and whatever their ``--json`` writes. An adapter reads that structure
and records one check per claim of the example's case page, so the suite reads them
like any other example. Three callers use the same adapter:

    the example itself      python examples/<x>/<x>.py --quick --bi-json out.json
    a multi-launch driver   python suite/adapters/convert.py <adapter> merged.json --json out.json
    a committed result      python suite/adapters/convert.py <adapter> examples/<x>/results-a40.json

``expect`` follows tools/bi_report.py:
    pass   kernels and fixes this handbook provides or recommends, at the level the case
           page claims. A failure is a regression.
    fail   the main documented failures of stock ops and of rejected designs.
    info   stock results that happened to pass, controls and diagnostics.

A check's name states the claim, not the sizes: quick and full mode give the same
names, and the sizes go in ``detail``. ``detail["signature"]`` is a short, deterministic
description of a failure (failing batch sizes, mismatch counts); the suite reports when
it differs from the baseline's.

A required claim whose data is missing from the legacy result (a section raised, a
launch died) becomes a failed check, and the reason is appended to extra["errors"].
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from typing import Any, Optional

_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from bi_report import Results  # noqa: E402

ADAPTERS = ("kernel_cases", "moe_combine", "attention_backward", "attention_serving", "mla_sparse",
            "linear_attention", "moe_ep", "moe_ep_2node", "collectives_tp")
TIME_UNITS = ("ms", "s", "us")


def dig(d: Any, *path, default=None):
    """Nested lookup; `default` when any key is absent."""
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


_RE_SIZES = re.compile(r"failing batch sizes \[([^\]]*)\]")
_RE_MODES = re.compile(r"failing modes \[(.*)\]")
_RE_COUNT = re.compile(r"FAIL (\d+)/(\d+)")


def parse_summary(s: Optional[str]) -> Optional[dict]:
    """A bi_check Report.summary() string -> verdict, failing sizes and failing modes."""
    if not isinstance(s, str):
        return None
    if ": PASS" in s:
        return {"passed": True, "sizes": [], "modes": []}
    if ": FAIL" not in s:
        return None
    sizes = _RE_SIZES.search(s)
    modes = _RE_MODES.search(s)
    count = _RE_COUNT.search(s)
    return {"passed": False,
            "sizes": [int(x) for x in sizes.group(1).split(",") if x.strip()] if sizes else [],
            "modes": [m.strip(" '\"") for m in modes.group(1).split(",")] if modes else [],
            "failed": int(count.group(1)) if count else None,
            "checked": int(count.group(2)) if count else None}


class Adapter:
    """What an adapter module's ``convert(legacy, A)`` writes into."""

    def __init__(self, res: Results, legacy: dict):
        self.res, self.legacy = res, legacy
        self.errors: list = []

    def error(self, msg: str) -> None:
        self.errors.append(str(msg)[:400])

    def check(self, name: str, passed: Optional[bool], *, level: str, impl: str, expect: str = "pass",
              detail: Optional[dict] = None, signature: Any = None) -> None:
        """Record one check. ``passed=None`` means the legacy result has no data for it."""
        if passed is None:
            self.error(f"no data for [{level}] {impl}: {name}")
            if expect == "pass":
                self.res.check(name, False, level=level, impl=impl, expect=expect,
                               detail={"missing": True, **(detail or {})})
            return
        d = dict(detail or {})
        if signature is not None and not passed:
            d["signature"] = signature
        self.res.check(name, bool(passed), level=level, impl=impl, expect=expect, detail=d or None)

    def measure(self, name: str, value: Any, unit: str, **meta) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.res.measure(name, value, unit, **meta)

    def section_errors(self) -> None:
        """Sections that raised: `<name>_error` keys, or a section dict that is only an error."""
        for k, v in self.legacy.items():
            if k.endswith("_error") and isinstance(v, str):
                self.error(f"{k}: {v}")
            elif isinstance(v, dict) and isinstance(v.get("error"), str) and len(v) == 1:
                self.error(f"{k}: {v['error']}")


def convert(adapter: str, legacy: dict, quick: bool = False, argv: Optional[list] = None,
            example: Optional[str] = None, live: bool = True) -> Results:
    """Run one adapter. ``live`` says the legacy dict was just measured in this process (the
    environment is this machine's); otherwise the environment is taken from the legacy file."""
    if adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter {adapter!r}; known: {', '.join(ADAPTERS)}")
    mod = importlib.import_module(f"{__name__}.{adapter}")
    res = Results(example or adapter, quick=quick, argv=argv)
    if not live:
        res.env = legacy_env(legacy, mod)
    A = Adapter(res, legacy)
    A.section_errors()
    mod.convert(legacy, A)
    res.extra["adapter"] = adapter
    res.extra["errors"] = A.errors
    return res


def emit(adapter: str, legacy: dict, path: Optional[str], quick: bool = False) -> int:
    """For an example's ``--bi-json``: convert what it just measured, write it, and return the
    exit code (1 when a required check failed or a section did not run)."""
    res = convert(adapter, legacy, quick=quick)
    res.write(path)
    for e in res.extra["errors"]:
        print("error:", e)
    return 1 if res.exit_code() or res.extra["errors"] else 0


def legacy_env(legacy: dict, mod) -> dict:
    """The stack recorded inside a legacy result file (it has no driver or CUDA build)."""
    src = mod.env_source(legacy) if hasattr(mod, "env_source") else legacy
    name, n = src.get("device"), src.get("gpu_count", 1)
    return {"source": "legacy result file", "torch": src.get("torch"), "triton": src.get("triton"),
            "nccl": src.get("nccl"), "driver": None, "cuda_build": None,
            "gpus": [{"name": name}] * n if name else []}
