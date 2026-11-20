#!/usr/bin/env python
"""bi_suite: re-run every verified example on this machine and compare with a baseline.

    python tools/bi_suite.py                      # quick mode, every entry this machine can run
    python tools/bi_suite.py --full               # the sizes the case pages were written from
    python tools/bi_suite.py --list               # what would run here, and what would be skipped
    python tools/bi_suite.py --record-baseline    # after reviewing a run: make it the baseline

What it does:
  1. reads suite/manifest.json (one entry per example: what it needs, a quick and a full
     command, where its bi-results-v1 file lands, a timeout);
  2. detects the GPUs (and, with --peer, a second node), runs the entries that fit and skips
     the others with a stated reason;
  3. writes one combined report (<out>/report.json, by default under suite/runs/) and prints a
     summary;
  4. compares every check with the baseline recorded for the same stack: GPU model and count,
     torch, Triton, NCCL, driver and CUDA build (suite/baselines/*.json).

What it reports:
  regression   a check with expect="pass" failed, or an entry gave no result. Exit code 1.
  changed      a check with expect="fail" passed, or a verdict differs from the baseline.
  as baseline  a check with expect="fail" passed, and passed in this stack's baseline too: the
               documented failure is stack-specific (for example A40-only). Information only.
  pattern      same verdict as the baseline, but the failure has another shape (for example
               other failing batch sizes). Information only.
  new/missing  checks that the baseline does not have, or that this run did not produce.
  timing       measurements beyond --timing-ratio of the baseline. Information only.
Equality inside the examples is bitwise (tools/bi_check.py). The suite compares verdicts; it
never judges a number by closeness. The exit code is nonzero only on regressions.

Shared machines: --lock-file takes a flock for each entry (not for the whole suite) and
releases it between entries. Entries with "own_lock" take the lock themselves, one launch at
a time (suite/bin/python-locked, or a two-node launcher that has its own lock).

Per entry: "python" names another interpreter for that entry only (an engine environment; the
entry is skipped where it does not exist). A command that uses {opguard} needs --opguard-root
(or $OPGUARD_ROOT), an OpGuard source checkout; without it the entry is skipped.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bi_report import SCHEMA as RESULT_SCHEMA, env_info  # noqa: E402

REPORT_SCHEMA = "bi-suite-report-v1"
BASELINE_SCHEMA = "bi-suite-baseline-v1"
MANIFEST_SCHEMA = "bi-suite-manifest-v1"
KEY_FIELDS = ("gpu", "nodes", "torch", "triton", "nccl", "driver", "cuda_build")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30"]


# ----------------------------------------------------------------------------- stack
def local_env() -> dict:
    env = env_info()
    if not env.get("gpus"):                       # torch missing or no CUDA: ask the driver
        try:
            names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=20).stdout.splitlines()
            env["gpus"] = [{"name": n.strip()} for n in names if n.strip()]
        except Exception:
            pass
    return env


def probe_env(python: str) -> dict:
    """This node's stack, read in a child process with the entries' interpreter: the runner itself
    never holds a CUDA context while the entries run."""
    try:
        p = subprocess.run([python, os.path.abspath(__file__), "--env"], capture_output=True, text=True, timeout=300)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as ex:                       # noqa: BLE001
        raise SystemExit(f"could not read the stack with {python}: {ex}")


def peer_env(peer: str, python: str, repo: str) -> tuple:
    """Environment of the second node: the same script, run there over SSH (same repo path)."""
    cmd = SSH + [peer, f"cd {shlex.quote(repo)} && {shlex.quote(python)} tools/bi_suite.py --env"]
    err = "no answer"
    for _ in range(4):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            err = "ssh timed out"
            continue
        if p.returncode == 0:
            try:
                return json.loads(p.stdout.strip().splitlines()[-1]), None
            except (ValueError, IndexError):
                return None, f"unreadable answer: {p.stdout[-200:]!r}"
        err = (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1][:200]
        if p.returncode != 255:                   # 255: ssh itself failed, worth another try
            break
        time.sleep(3)
    return None, err


def gpu_label(env: dict) -> str:
    counts: dict = {}
    for g in env.get("gpus") or []:
        counts[g["name"]] = counts.get(g["name"], 0) + 1
    return ", ".join(f"{n} x{c}" for n, c in sorted(counts.items())) or "no GPU"


def stack_key(envs: list) -> dict:
    """What a baseline is recorded for. Two nodes: fields that differ are joined with ' + '."""
    def field(name):
        vals = [str(e.get(name)) for e in envs]
        return vals[0] if len(set(vals)) == 1 else " + ".join(vals)
    key = {"gpu": " + ".join(gpu_label(e) for e in envs), "nodes": len(envs)}
    key.update({f: field(f) for f in KEY_FIELDS[2:]})
    return key


def key_slug(key: dict) -> str:
    gpu = re.sub(r"NVIDIA\s*", "", key["gpu"]).replace(" x", "x").replace(" ", "")
    drv = key["driver"].replace(" ", "")
    return re.sub(r"[^A-Za-z0-9.+_,-]", "-", f"{gpu}_torch{key['torch']}_triton{key['triton']}"
                  f"_nccl{key['nccl']}_drv{drv}_cuda{key['cuda_build']}")


def key_text(key: dict) -> str:
    return (f"{key['gpu']}; torch {key['torch']}, Triton {key['triton']}, NCCL {key['nccl']}, "
            f"driver {key['driver']}, CUDA build {key['cuda_build']}")


# ----------------------------------------------------------------------------- manifest
def load_manifest(path: str) -> list:
    with open(path) as f:
        m = json.load(f)
    if m.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit(f"{path}: schema must be {MANIFEST_SCHEMA}")
    seen = set()
    for e in m["entries"]:
        for k in ("name", "gpus", "timeout"):
            if k not in e:
                raise SystemExit(f"{path}: entry {e.get('name', '?')} lacks '{k}'")
        if not (e.get("quick") or e.get("full")):
            raise SystemExit(f"{path}: entry {e['name']} has neither a quick nor a full command")
        if e["name"] in seen:
            raise SystemExit(f"{path}: duplicate entry {e['name']}")
        seen.add(e["name"])
        e.setdefault("nodes", 1)
        e.setdefault("result", "{run}/" + e["name"] + ".json")
    return m["entries"]


def needs_text(e: dict) -> str:
    one = "1 GPU" if e["gpus"] == 1 else f"{e['gpus']} GPUs on one node"
    return one if e["nodes"] == 1 else f"{e['nodes']} nodes x {e['gpus']} GPUs"


def applicable(e: dict, mode: str, n_local: int, peer: Optional[dict], args) -> Optional[str]:
    """None when the entry can run here; otherwise the reason it is skipped."""
    if args.only and e["name"] not in args.only:
        return "not selected (--only)"
    if e["name"] in args.skip:
        return "not selected (--skip)"
    if not e.get(mode):
        return f"no {mode} command in the manifest"
    if e.get("python") and not os.access(entry_python(e, args), os.X_OK):
        return f"needs the interpreter {e['python']}, not found here"
    if "{opguard}" in json.dumps(e.get(mode)) and not args.opguard_root:
        return "needs an OpGuard checkout (--opguard-root or $OPGUARD_ROOT)"
    if e["nodes"] == 1:
        return None if n_local >= e["gpus"] else f"needs {needs_text(e)}; this node has {n_local}"
    if e["nodes"] != 2:
        return f"needs {e['nodes']} nodes; the runner supports 1 or 2"
    if n_local < e["gpus"]:
        return f"needs {needs_text(e)}; this node has {n_local}"
    if not args.peer:
        return f"needs {needs_text(e)}; no second node given (--peer)"
    if "{launcher2}" in json.dumps(e.get(mode)) and not args.launcher2:
        return f"needs {needs_text(e)}; no two-node launcher given (--launcher2)"
    if peer is None or "error" in peer:
        return f"needs {needs_text(e)}; second node not usable: {(peer or {}).get('error', 'no answer')}"
    n_peer = len(peer["env"].get("gpus") or [])
    return None if n_peer >= e["gpus"] else f"needs {needs_text(e)}; the second node has {n_peer}"


def entry_python(e: dict, args) -> str:
    """The entry's own interpreter when the manifest names one, else the suite's."""
    return os.path.expanduser(e["python"]) if e.get("python") else args.python


def timeout_of(e: dict, mode: str) -> float:
    t = e["timeout"]
    return float(t[mode] if isinstance(t, dict) else t)


# ----------------------------------------------------------------------------- running
class FileLock:
    """flock(2) on a lock file, as `flock FILE command` takes it. Waiting is not on the clock."""

    def __init__(self, path: Optional[str]):
        self.path, self.fd, self.waited = path, None, 0.0

    def __enter__(self):
        if self.path:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
            t0 = time.time()
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            self.waited = time.time() - t0
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def run_step(cmd: str, log, deadline: float, env: dict) -> tuple:
    """One shell command in its own process group. Returns (return code or None, timed_out)."""
    log.write(f"\n$ {cmd}\n")
    log.flush()
    p = subprocess.Popen(["bash", "-c", cmd], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, env=env)
    try:
        return p.wait(timeout=max(1.0, deadline - time.time())), False
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                break
            try:
                p.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                continue
        return None, True


def run_entry(e: dict, mode: str, out_dir: str, args) -> dict:
    run = os.path.join(out_dir, e["name"])
    os.makedirs(run, exist_ok=True)
    timeout = timeout_of(e, mode)
    py = entry_python(e, args)
    subst = {"python": py, "opguard": args.opguard_root or "", "repo": REPO, "out": out_dir, "run": run, "launcher2": args.launcher2 or "",
             "peer": args.peer or "", "timeout": str(int(timeout)), "mode": mode,
             "pylocked": os.path.join(REPO, "suite", "bin", "python-locked")}
    subst["result"] = e["result"].format(**subst)
    steps = e[mode] if isinstance(e[mode], list) else [e[mode]]
    steps = [s.format(**subst) for s in steps]
    if os.path.exists(subst["result"]):
        os.remove(subst["result"])
    env = dict(os.environ, BI_SUITE_PYTHON=py, BI_SUITE_LOCK=args.lock_file or "",
               PYTHONUNBUFFERED="1")
    rec = {"name": e["name"], "needs": needs_text(e), "status": "ok", "reason": None, "returncodes": [],
           "lock_wait_s": 0.0, "log": os.path.join(run, "log.txt"), "result_file": subst["result"]}
    t_run = 0.0
    with open(rec["log"], "w") as log:
        own = bool(e.get("own_lock"))
        if own and args.lock_file:                # the entry locks each launch itself: only wait our turn
            with FileLock(args.lock_file) as lk:
                rec["lock_wait_s"] += lk.waited
        deadline = time.time() + timeout
        for cmd in steps:
            with FileLock(None if own else args.lock_file) as lk:
                rec["lock_wait_s"] += lk.waited
                deadline += lk.waited             # waiting for the lock is not on the clock
                t0 = time.time()
                rc, timed_out = run_step(cmd, log, deadline, env)
                t_run += time.time() - t0
            rec["returncodes"].append(rc)
            if timed_out:
                rec["status"], rec["reason"] = "timeout", f"no result after {int(timeout)} s"
                break
    rec["seconds"] = round(t_run, 1)
    rec["lock_wait_s"] = round(rec["lock_wait_s"], 1)
    if rec["status"] == "ok":
        rec["result"], err = read_result(subst["result"])
        if err:
            rec["status"], rec["reason"] = "error", f"{err} (exit codes {rec['returncodes']}; see {rec['log']})"
    if rec["status"] != "ok" and e["nodes"] > 1 and args.peer:
        rec["peer_processes"] = leftover_on_peer(args.peer)
    return rec


def leftover_on_peer(peer: str) -> str:
    """After a failed two-node entry: what still runs on the second node. Reported, never killed."""
    try:
        p = subprocess.run(SSH + [peer, "pgrep -u $USER -af '[t]orch[.]distributed' || true"],
                           capture_output=True, text=True, timeout=60)
        return p.stdout.strip() or "none"
    except Exception as ex:                       # noqa: BLE001
        return f"could not check: {ex}"


def read_result(path: str) -> tuple:
    if not os.path.exists(path):
        return None, "no result file"
    try:
        with open(path) as f:
            r = json.load(f)
    except ValueError as ex:
        return None, f"result file is not JSON ({ex})"
    if r.get("schema") != RESULT_SCHEMA or not isinstance(r.get("checks"), list):
        return None, f"result file is not {RESULT_SCHEMA}"
    return r, None


# ----------------------------------------------------------------------------- comparison
def check_id(c: dict) -> str:
    return f"{c['level']} | {c['impl']} | {c['name']}"


def measurement_id(m: dict) -> str:
    meta = "".join(f" | {k}={m[k]}" for k in sorted(m) if k not in ("name", "value", "unit"))
    return f"{m['name']}{meta}"


def slim(result: dict) -> dict:
    """A result reduced to what the suite compares: verdicts, signatures, measurements."""
    checks, dup = {}, []
    for c in result["checks"]:
        cid = check_id(c)
        if cid in checks:                         # two checks must not share a name: keep both, say so
            dup.append(cid)
            n = 2
            while f"{cid} #{n}" in checks:
                n += 1
            cid = f"{cid} #{n}"
        row = {"expect": c["expect"], "passed": bool(c["passed"])}
        d = c.get("detail")
        if isinstance(d, dict):
            if d.get("signature") is not None:
                row["signature"] = d["signature"]
            if d.get("missing"):
                row["missing_data"] = True
        checks[cid] = row
    meas = {}
    for m in result.get("measurements") or []:
        if isinstance(m.get("value"), (int, float)) and not isinstance(m.get("value"), bool):
            meas[measurement_id(m)] = {"value": m["value"], "unit": m.get("unit")}
    return {"checks": checks, "measurements": meas, "duplicate_check_names": dup,
            "errors": list((result.get("extra") or {}).get("errors") or [])}


def compare(cur: dict, base: Optional[dict], ratio: float) -> dict:
    """cur, base: slim results. base is None when no baseline has this entry."""
    out = {"regressions": [], "changed": [], "as_baseline": [], "pattern": [], "new": [], "missing": [],
           "timing": []}
    bchecks = (base or {}).get("checks") or {}
    for cid, c in cur["checks"].items():
        b = bchecks.get(cid)
        if c["expect"] == "pass" and not c["passed"]:
            out["regressions"].append({"check": cid, "no_data": bool(c.get("missing_data")),
                                       "in_baseline": None if b is None else
                                       ("failed there too" if not b["passed"] else "passed there")})
            continue
        if c["expect"] == "fail" and c["passed"] and b is not None and b["passed"] and b["expect"] == "fail":
            # the documented failure does not occur on this stack, and the reviewed baseline says so
            out["as_baseline"].append({"check": cid, "what": "documented failure passes on this stack, as in its baseline"})
        elif c["expect"] == "fail" and c["passed"]:
            out["changed"].append({"check": cid, "what": "documented failure now passes",
                                   "in_baseline": None if b is None else ("passed" if b["passed"] else "failed")})
        elif b is not None and b["passed"] != c["passed"]:
            out["changed"].append({"check": cid, "what": "verdict differs from the baseline",
                                   "in_baseline": "passed" if b["passed"] else "failed",
                                   "now": "passed" if c["passed"] else "failed"})
        elif b is not None and b["expect"] != c["expect"]:
            out["changed"].append({"check": cid, "what": f"expect was {b['expect']}, is {c['expect']}"})
        elif b is not None and not c["passed"] and "signature" in b and "signature" in c \
                and b["signature"] != c["signature"]:
            out["pattern"].append({"check": cid, "baseline": b["signature"], "now": c["signature"]})
        if base is not None and b is None:
            out["new"].append(cid)
    if base is not None:
        out["missing"] = [cid for cid in bchecks if cid not in cur["checks"]]
        for mid, m in cur["measurements"].items():
            b = base.get("measurements", {}).get(mid)
            if b and b.get("unit") == m["unit"] and b["value"] and m["value"] and b["value"] > 0 and m["value"] > 0:
                r = m["value"] / b["value"]
                if r > ratio or r < 1 / ratio:
                    out["timing"].append({"measurement": mid, "unit": m["unit"], "baseline": b["value"],
                                          "now": m["value"], "ratio": round(r, 2)})
    return out


def load_baselines(directory: str, mode: str) -> list:
    out = []
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(directory, name)) as f:
                        b = json.load(f)
                except ValueError:
                    continue
                if b.get("schema") == BASELINE_SCHEMA and b.get("mode") == mode:
                    b["_file"] = os.path.join(directory, name)
                    out.append(b)
    return out


def find_baseline(baselines: list, key: dict) -> Optional[dict]:
    for b in baselines:
        if all(str(b.get("key", {}).get(f)) == str(key[f]) for f in KEY_FIELDS):
            return b
    return None


def committed_slim(spec: Any) -> tuple:
    """The committed result behind a case page, as a slim result. spec: a bi-results-v1 file, or
    {"file": ..., "adapter": ...} for a file in an example's own format."""
    path = spec if isinstance(spec, str) else spec["file"]
    full = os.path.join(REPO, path)
    try:
        with open(full) as f:
            data = json.load(f)
        if not isinstance(spec, str) and spec.get("adapter"):
            sys.path.insert(0, os.path.join(REPO, "suite"))
            import adapters
            data = adapters.convert(spec["adapter"], data, live=False).to_dict()
        if data.get("schema") != RESULT_SCHEMA:
            return None, f"{path} is not {RESULT_SCHEMA}"
        return slim(data), None
    except Exception as ex:                       # noqa: BLE001
        return None, f"{path}: {type(ex).__name__}: {ex}"


def vs_committed(cur: dict, com: dict) -> dict:
    differ = [{"check": cid, "committed": "passed" if b["passed"] else "failed",
               "now": "passed" if cur["checks"][cid]["passed"] else "failed"}
              for cid, b in com["checks"].items() if cid in cur["checks"] and cur["checks"][cid]["passed"] != b["passed"]]
    return {"differ": differ, "compared": sum(cid in cur["checks"] for cid in com["checks"]),
            "only_committed": [cid for cid in com["checks"] if cid not in cur["checks"]],
            "only_now": [cid for cid in cur["checks"] if cid not in com["checks"]]}


# ----------------------------------------------------------------------------- report
def count(slim_result: dict, as_baseline=()) -> dict:
    """as_baseline: ids of expect="fail" checks that pass here and in this stack's baseline (counted ok)."""
    n = {"checks": 0, "required": 0, "ok": 0, "regression": 0, "changed": 0, "info": 0}
    for cid, c in slim_result["checks"].items():
        n["checks"] += 1
        n["required"] += c["expect"] == "pass"
        if c["expect"] == "info":
            n["info"] += 1
        elif c["passed"] == (c["expect"] == "pass") or cid in as_baseline:
            n["ok"] += 1
        else:
            n["regression" if c["expect"] == "pass" else "changed"] += 1
    return n


def print_report(rep: dict, verbose: bool) -> None:
    P = print
    P(f"\nbi_suite, {rep['mode']} mode")
    P(f"stack: {key_text(rep['stack']['key'])}")
    if rep.get("pair_stack"):
        P(f"two-node stack: {key_text(rep['pair_stack']['key'])}")
    for which in ("stack", "pair_stack"):
        s = rep.get(which)
        if s and s.get("used"):
            if s.get("baseline"):
                P(f"baseline ({which.replace('_', ' ')}): {s['baseline']}, recorded {s.get('baseline_recorded')}")
            else:
                P(f"NO BASELINE matches the {which.replace('_', ' ')} ({rep['mode']} mode). Review the results below, "
                  "then record one:  python tools/bi_suite.py --from-report "
                  f"{rep['report_file']} --record-baseline")
    P(f"\n{'entry':24} {'status':8} {'time':>7} {'checks':>6} {'req.':>5} {'ok':>4} {'regr.':>5} {'chg.':>5} {'info':>5}")
    for e in rep["entries"]:
        if e["status"] == "skipped":
            P(f"{e['name']:24} skipped  {e['reason']}")
            continue
        if e["status"] != "ok":
            P(f"{e['name']:24} {e['status'].upper():8} {e.get('seconds', 0):6.0f}s  {e['reason']}")
            continue
        n = e["counts"]
        P(f"{e['name']:24} {'ran':8} {e['seconds']:6.0f}s {n['checks']:6} {n['required']:5} {n['ok']:4} "
          f"{n['regression']:5} {n['changed']:5} {n['info']:5}")
    sections = (("regressions", "REGRESSIONS: a required check (expect=pass) failed"),
                ("changed", "CHANGED: a documented failure passes, or a verdict differs from the baseline"),
                ("as_baseline", "AS BASELINE: documented failures that pass on this stack, as in its baseline "
                                "(stack-specific; information)"),
                ("pattern", "PATTERN: same verdict as the baseline, another failure shape (information)"),
                ("new", "NEW: checks the baseline does not have"),
                ("missing", "MISSING: baseline checks this run did not produce"),
                ("timing", f"TIMING: beyond {rep['timing_ratio']}x of the baseline, either way (information)"))
    for key, title in sections:
        rows = [(e["name"], r) for e in rep["entries"] if e["status"] == "ok" for r in e["comparison"][key]]
        P(f"\n{title}: {len(rows) or 'none'}")
        limit = None if (verbose or key in ("regressions", "changed")) else (0 if key == "as_baseline" else 12)
        for name, r in rows[:limit]:
            if key in ("new", "missing"):
                P(f"  {name}: {r}")
            elif key == "timing":
                P(f"  {name}: {r['measurement']}: {r['baseline']:.4g} -> {r['now']:.4g} {r['unit']} ({r['ratio']}x)")
            elif key == "pattern":
                P(f"  {name}: {r['check']}\n      baseline {json.dumps(r['baseline'])}\n      now      {json.dumps(r['now'])}")
            else:
                extra = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in r.items()
                                  if k != "check" and v not in (None, False))
                P(f"  {name}: {r['check']}" + (f"  [{extra}]" if extra else ""))
        if limit is not None and len(rows) > limit:
            P(f"  ... {len(rows) - limit} {'more ' if limit else ''}listed in {rep['report_file']} (or --verbose)")
    bad = [e for e in rep["entries"] if e["status"] in ("error", "timeout")]
    P(f"\nNO RESULT: entries that did not finish (counted as regressions): {len(bad) or 'none'}")
    for e in bad:
        P(f"  {e['name']}: {e['status']}: {e['reason']}")
        if e.get("peer_processes"):
            P(f"      still running on the second node: {e['peer_processes']}")
    inc = [(e["name"], m) for e in rep["entries"] if e["status"] == "ok" for m in e["errors"]]
    if inc:
        P(f"\nINCOMPLETE: sections that raised inside an entry: {len(inc)}")
        for name, m in inc:
            P(f"  {name}: {m}")
    for e in rep["entries"]:
        vc = e.get("vs_committed")
        if not vc:
            continue
        if "error" in vc:
            P(f"\nvs committed result, {e['name']}: {vc['error']}")
            continue
        P(f"\nvs committed result, {e['name']} ({vc['file']}): {vc['compared']} checks compared, "
          f"{len(vc['differ'])} differ, {len(vc['only_committed'])} only in the committed result, "
          f"{len(vc['only_now'])} only in this run")
        for r in vc["differ"]:
            P(f"  differs: {r['check']}  [committed: {r['committed']}; now: {r['now']}]")
        if verbose:
            for cid in vc["only_committed"]:
                P(f"  only committed: {cid}")
            for cid in vc["only_now"]:
                P(f"  only now: {cid}")
    s = rep["summary"]
    P(f"\nsummary: {s['ran']} entries ran, {s['skipped']} skipped, {s['no_result']} without result; "
      f"{s['checks']} checks ({s['required']} required): {s['regressions']} regressions, {s['changed']} changed, "
      f"{s.get('as_baseline', 0)} as baseline, {s['pattern']} pattern, {s['new']} new, {s['missing']} missing, {s['timing']} timing")
    P(f"report: {rep['report_file']}")
    P(f"exit code {s['exit_code']}" + (" (regressions)" if s["exit_code"] else ""))


# ----------------------------------------------------------------------------- baselines
def record_baselines(rep: dict, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for which in ("stack", "pair_stack"):
        s = rep.get(which)
        entries = {e["name"]: {"seconds": e["seconds"], "checks": e["slim"]["checks"],
                               "measurements": e["slim"]["measurements"]}
                   for e in rep["entries"] if e["status"] == "ok" and e["stack"] == which}
        if not s or not entries:
            continue
        path = os.path.join(directory, f"{key_slug(s['key'])}.{rep['mode']}.json")
        old = {}
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f).get("entries", {})
        kept = sorted(set(old) - set(entries))
        base = {"schema": BASELINE_SCHEMA, "mode": rep["mode"], "key": s["key"],
                "recorded": datetime.date.today().isoformat(),
                "note": "Verdicts and measurements of one reviewed run of tools/bi_suite.py on this stack. "
                        "Record a new one with --record-baseline; do not edit by hand.",
                "entries": {**old, **entries}}
        with open(path, "w") as f:
            json.dump(base, f, indent=1, sort_keys=True)
            f.write("\n")
        failing = sum(not c["passed"] for e in entries.values() for c in e["checks"].values() if c["expect"] == "pass")
        print(f"baseline written: {path} ({len(entries)} entries from this run"
              + (f", {len(kept)} kept from the previous baseline: {', '.join(kept)}" if kept else "") + ")")
        if failing:
            print(f"  note: {failing} required checks fail in it. They stay regressions on every later run.")
    no = [e["name"] for e in rep["entries"] if e["status"] in ("error", "timeout")]
    if no:
        print(f"  not recorded (no result): {', '.join(no)}")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="See handbook/4-operations/regression.md.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="reduced sizes (default)")
    mode.add_argument("--full", action="store_true", help="the sizes behind the case pages")
    ap.add_argument("--manifest", default=os.path.join(REPO, "suite", "manifest.json"))
    ap.add_argument("--only", default="", help="comma-separated entry names to run")
    ap.add_argument("--skip", default="", help="comma-separated entry names to leave out")
    ap.add_argument("--out", help="directory for logs, result files and report.json "
                                  "(default: suite/runs/<mode>-<time>)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the entries (default: this one)")
    ap.add_argument("--lock-file", help="flock this file around each entry (shared GPUs)")
    ap.add_argument("--opguard-root", default=os.environ.get("OPGUARD_ROOT"),
                    help="OpGuard source checkout, for entries whose command uses {opguard}")
    ap.add_argument("--peer", help="second node, reachable by SSH, with the repo at the same path")
    ap.add_argument("--launcher2", default=os.environ.get("BI_SUITE_LAUNCHER2"),
                    help="two-node launcher: <launcher2> <workdir> <script.py> [args] (it takes its own lock)")
    ap.add_argument("--baseline-dir", default=os.path.join(REPO, "suite", "baselines"))
    ap.add_argument("--record-baseline", action="store_true", help="write this run as the baseline for its stack")
    ap.add_argument("--timing-ratio", type=float, default=1.5,
                    help="report measurements beyond this ratio of the baseline, either way (default 1.5)")
    ap.add_argument("--vs-committed", action="store_true",
                    help="also compare verdicts with the committed result files behind the case pages")
    ap.add_argument("--from-report", help="do not run: compare (or record) an existing report.json")
    ap.add_argument("--list", action="store_true", help="show what would run here, then stop")
    ap.add_argument("--env", action="store_true", help="print this machine's stack as JSON, then stop")
    ap.add_argument("--verbose", action="store_true", help="print every row of every section")
    args = ap.parse_args(argv)
    if args.env:
        print(json.dumps(local_env()))
        return 0
    mode = "full" if args.full else "quick"
    args.only = [x for x in args.only.split(",") if x]
    args.skip = [x for x in args.skip.split(",") if x]
    entries = load_manifest(args.manifest)
    unknown = [x for x in args.only + args.skip if x not in {e["name"] for e in entries}]
    if unknown:
        raise SystemExit(f"unknown entry name(s): {', '.join(unknown)}")

    if args.from_report:
        with open(args.from_report) as f:
            rep = json.load(f)
        mode = rep["mode"]
        rep["report_file"] = args.from_report
    else:
        peer = None
        with FileLock(args.lock_file):            # the probe touches the GPUs: take our turn
            env = probe_env(args.python)
            if args.peer:
                penv, err = peer_env(args.peer, args.python, REPO)
                peer = {"env": penv} if penv else {"error": err}
        rep = {"schema": REPORT_SCHEMA, "mode": mode, "argv": sys.argv[1:],
               "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "stack": {"key": stack_key([env]), "env": env},
               "pair_stack": {"key": stack_key([env, peer["env"]]), "env": [env, peer["env"]]}
               if peer and "env" in peer else None, "entries": []}
        n_local = len(env.get("gpus") or [])
        plan = [(e, applicable(e, mode, n_local, peer, args)) for e in entries]
        if args.list:
            print(f"stack: {key_text(rep['stack']['key'])}")
            for e, why in plan:
                print(f"{e['name']:24} {needs_text(e):22} " + (f"skip: {why}" if why else
                      f"run, timeout {int(timeout_of(e, mode))} s"))
            return 0
        out_dir = os.path.abspath(args.out or os.path.join(
            REPO, "suite", "runs", f"{mode}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"))
        os.makedirs(out_dir, exist_ok=True)
        rep["report_file"] = os.path.join(out_dir, "report.json")
        print(f"bi_suite: {mode} mode on {key_text(rep['stack']['key'])}; output in {out_dir}", flush=True)
        for e, why in plan:
            if why:
                rep["entries"].append({"name": e["name"], "needs": needs_text(e), "status": "skipped", "reason": why})
                continue
            print(f"  {e['name']} ({needs_text(e)}) ...", end=" ", flush=True)
            rec = run_entry(e, mode, out_dir, args)
            rec["stack"] = "stack" if e["nodes"] == 1 else "pair_stack"
            if e.get("committed"):
                rec["committed"] = e["committed"]
            print(f"{rec['status']} in {rec['seconds']:.0f} s"
                  + (f" (waited {rec['lock_wait_s']:.0f} s for the lock)" if rec["lock_wait_s"] >= 1 else "")
                  + (f": {rec['reason']}" if rec["reason"] else ""), flush=True)
            rep["entries"].append(rec)

    # ---- compare
    rep["timing_ratio"] = args.timing_ratio
    baselines = load_baselines(args.baseline_dir, mode)
    for which in ("stack", "pair_stack"):
        s = rep.get(which)
        if s:
            b = find_baseline(baselines, s["key"])
            s["baseline"] = os.path.relpath(b["_file"], REPO) if b else None
            s["baseline_recorded"] = b.get("recorded") if b else None
            s["_entries"] = (b or {}).get("entries")
            s["used"] = any(e.get("stack") == which and e["status"] != "skipped" for e in rep["entries"])
    tot = {k: 0 for k in ("checks", "required", "regressions", "changed", "as_baseline", "pattern", "new", "missing",
                          "timing")}
    for e in rep["entries"]:
        if e["status"] != "ok":
            continue
        e["slim"] = slim(e.pop("result")) if "result" in e else e["slim"]
        base_entries = rep[e["stack"]]["_entries"]
        e["comparison"] = compare(e["slim"], None if base_entries is None else base_entries.get(e["name"]),
                                  args.timing_ratio)
        e["counts"] = count(e["slim"], {r["check"] for r in e["comparison"]["as_baseline"]})
        e["errors"] = e["slim"]["errors"]
        if base_entries is not None and e["name"] not in base_entries:
            e["comparison"]["new"] = list(e["slim"]["checks"])      # the baseline lacks the whole entry
        if args.vs_committed and e.get("committed"):
            com, err = committed_slim(e["committed"])
            spec = e["committed"]
            e["vs_committed"] = {"error": err} if err else {
                "file": spec if isinstance(spec, str) else spec["file"], **vs_committed(e["slim"], com)}
        tot["checks"] += e["counts"]["checks"]
        tot["required"] += e["counts"]["required"]
        for k in ("regressions", "changed", "as_baseline", "pattern", "new", "missing", "timing"):
            tot[k] += len(e["comparison"][k])
    for which in ("stack", "pair_stack"):
        if rep.get(which):
            rep[which].pop("_entries", None)
    no_result = sum(e["status"] in ("error", "timeout") for e in rep["entries"])
    rep["summary"] = {"ran": sum(e["status"] == "ok" for e in rep["entries"]),
                      "skipped": sum(e["status"] == "skipped" for e in rep["entries"]),
                      "no_result": no_result, **tot,
                      "exit_code": 1 if (tot["regressions"] or no_result) else 0}
    if not args.from_report:
        with open(rep["report_file"], "w") as f:
            json.dump(rep, f, indent=1)
    print_report(rep, args.verbose)
    if args.record_baseline:
        record_baselines(rep, args.baseline_dir)
    return rep["summary"]["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
