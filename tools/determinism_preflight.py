"""determinism_preflight: what a job can report about its own determinism settings.

Chapter 1 of the handbook asks for the avoidable nondeterminism to be controlled
before two runs are compared. This tool reads back, from inside the job, every
control surface that can be observed there, as one JSON report per rank:

    import determinism_preflight as pf
    rep = pf.report(optimizer=opt, dataloader=loader, scheduler=sched, autocast=amp, model=model)
    json.dump(rep, open(f"preflight-rank{rep['rank']['rank']}.json", "w"))

    pf.findings(rep)             # surfaces this one run leaves uncontrolled
    pf.diff(rep_a, rep_b)        # what differs between two runs, and which
                                 # differences make a bitwise comparison meaningless

Take the report after everything is built and restored, just before the first
step: on a resume that is where unrestored state shows.

``report`` reads and never sets. It initializes CUDA if it is available. When
OpGuard is installed, its doctor output is embedded as ``opguard_doctor`` (run in
a child process) and nothing it covers is probed again here.

    python determinism_preflight.py                  # this interpreter's report
    python determinism_preflight.py --diff a.json b.json [--declare launch.argv]
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA = "determinism-preflight-v1"
DIFF_SCHEMA = "determinism-preflight-diff-v1"
SDPA_CASE = "handbook/3-kernels/attention.md"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _name(obj) -> Optional[str]:
    if obj is None:
        return None
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', type(obj).__qualname__)}"


def _generator(gen) -> Optional[dict]:
    if gen is None:
        return None
    return {"device": str(gen.device), "initial_seed": gen.initial_seed(),
            "state_sha256": _sha(bytes(gen.get_state().tolist()))}


def _fingerprint(tensors: Iterable) -> Optional[List[int]]:
    """XOR of the whole-tensor fingerprints of some tensors: equal state gives an
    equal value, whatever the order. Four regions, one host read."""
    try:
        import torch
        from opguard_bridge import xor_signature
    except Exception:
        return None
    acc = None
    for t in tensors:
        sig = xor_signature(t, 4, impl="torch").to(torch.int64).cpu()
        acc = sig if acc is None else torch.bitwise_xor(acc, sig)
    return None if acc is None else [int(v) & 0xFFFFFFFF for v in acc.tolist()]


# ------------------------------------------------------------------ sections
def _rank_section() -> dict:
    def env_int(*names):
        for n in names:
            try:
                return int(os.environ[n])
            except (KeyError, ValueError):
                pass
        return None
    out = {"rank": env_int("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"),
           "local_rank": env_int("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"),
           "world_size": env_int("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"),
           "distributed_initialized": False, "backend": None}
    dist = sys.modules.get("torch.distributed")
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            out.update(distributed_initialized=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
                       backend=dist.get_backend())
    except Exception:
        pass
    out["rank"] = 0 if out["rank"] is None else out["rank"]
    out["world_size"] = 1 if out["world_size"] is None else out["world_size"]
    return out


def _stack_section() -> dict:
    try:
        from bi_report import env_info
        stack = env_info()
    except Exception:
        stack = {"python": sys.version.split()[0]}
    libs = {}
    try:
        from importlib import metadata
        for dist in metadata.distributions():
            name = (dist.metadata["Name"] or "").lower()
            if name.startswith("nvidia-") or name in ("triton", "numpy"):
                libs[name] = dist.version
    except Exception:
        pass
    stack["libraries"] = dict(sorted(libs.items()))
    return stack


def _torch_section(torch, autocast, grad_scaler, model, gc_scan: bool) -> dict:
    cuda = torch.cuda.is_available()
    sdpa = {}
    for key, fn in (("flash", "flash_sdp_enabled"), ("mem_efficient", "mem_efficient_sdp_enabled"),
                    ("math", "math_sdp_enabled"), ("cudnn", "cudnn_sdp_enabled")):
        probe = getattr(torch.backends.cuda, fn, None)
        sdpa[key] = bool(probe()) if probe else None
    amp: Dict[str, Any] = {"enabled_now": None, "dtype_now": None, "context": None, "grad_scaler": None}
    try:
        kind = "cuda" if cuda else "cpu"
        amp["enabled_now"] = bool(torch.is_autocast_enabled(kind))
        amp["dtype_now"] = str(torch.get_autocast_dtype(kind))
    except Exception:
        try:
            amp["enabled_now"] = bool(torch.is_autocast_enabled())
        except Exception:
            pass
    if autocast is not None:
        amp["context"] = {"device": getattr(autocast, "device", None),
                          "dtype": str(getattr(autocast, "fast_dtype", None)),
                          "enabled": bool(getattr(autocast, "_enabled", True)),
                          "cache_enabled": getattr(autocast, "_cache_enabled", None)}
    if grad_scaler is not None:
        try:
            amp["grad_scaler"] = {"enabled": grad_scaler.is_enabled(), **_jsonable(grad_scaler.state_dict())}
        except Exception as exc:
            amp["grad_scaler"] = {"error": str(exc)}
    compiled: Dict[str, Any] = {"dynamo_loaded": "torch._dynamo" in sys.modules, "compiled_graphs": None,
                                "model_is_compiled": None}
    try:
        if compiled["dynamo_loaded"]:
            from torch._dynamo.utils import counters
            compiled["compiled_graphs"] = int(counters["stats"].get("unique_graphs", 0))
    except Exception:
        pass
    if model is not None:
        try:
            compiled["model_is_compiled"] = any(hasattr(m, "_orig_mod") for m in model.modules())
        except Exception:
            compiled["model_is_compiled"] = hasattr(model, "_orig_mod")
    graphs: Dict[str, Any] = {"capturing_now": None, "graph_objects": None}
    if cuda:
        try:
            graphs["capturing_now"] = bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            pass
        if gc_scan:
            kind = getattr(torch.cuda, "CUDAGraph", None)
            graphs["graph_objects"] = sum(1 for o in gc.get_objects() if type(o) is kind) if kind else None
    return {
        "deterministic_algorithms": {
            "enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "fill_uninitialized_memory": getattr(torch.utils.deterministic, "fill_uninitialized_memory", None)},
        "cudnn": {"enabled": torch.backends.cudnn.enabled, "deterministic": torch.backends.cudnn.deterministic,
                  "benchmark": torch.backends.cudnn.benchmark, "allow_tf32": torch.backends.cudnn.allow_tf32},
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "sdpa_backends": sdpa, "default_dtype": str(torch.get_default_dtype()),
        "amp": amp, "compile": compiled, "cuda_graphs": graphs,
    }


def _seed_section(torch) -> dict:
    out = {"torch_initial_seed": torch.initial_seed(),
           "torch_rng_state_sha256": _sha(bytes(torch.get_rng_state().tolist())),
           "python_random_state_sha256": _sha(repr(random.getstate()).encode()),
           "numpy_state_sha256": None, "cuda_initial_seed": None, "cuda_rng_state_sha256": {}}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        try:
            state = numpy.random.get_state()
            out["numpy_state_sha256"] = _sha(state[1].tobytes() + str(state[2:]).encode())
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            out["cuda_initial_seed"] = torch.cuda.initial_seed()
            for i in range(torch.cuda.device_count()):
                out["cuda_rng_state_sha256"][str(i)] = _sha(bytes(torch.cuda.get_rng_state(i).tolist()))
        except Exception as exc:
            out["cuda_error"] = str(exc)
    return out


def _dataloader_section(loader) -> dict:
    sampler = getattr(loader, "sampler", None)
    out = {"class": type(loader).__name__, "batch_size": loader.batch_size, "drop_last": loader.drop_last,
           "num_workers": loader.num_workers, "persistent_workers": loader.persistent_workers,
           "pin_memory": loader.pin_memory, "prefetch_factor": loader.prefetch_factor,
           "multiprocessing_context": _jsonable(getattr(loader, "multiprocessing_context", None)) and
           type(loader.multiprocessing_context).__name__,
           "sampler": type(sampler).__name__ if sampler is not None else None,
           "batch_sampler": type(loader.batch_sampler).__name__ if loader.batch_sampler is not None else None,
           "shuffle": type(sampler).__name__ in ("RandomSampler", "SubsetRandomSampler")
           or bool(getattr(sampler, "shuffle", False)),
           "generator": _generator(loader.generator),
           "sampler_generator": _generator(getattr(sampler, "generator", None)),
           "worker_init_fn": _name(loader.worker_init_fn), "collate_fn": _name(loader.collate_fn),
           "dataset": {"type": type(loader.dataset).__name__, "len": None}}
    try:
        out["dataset"]["len"] = len(loader.dataset)
    except Exception:
        pass
    out["seed_source"] = "loader.generator" if loader.generator is not None else "global torch RNG"
    if type(sampler).__name__ == "DistributedSampler":
        out["distributed_sampler"] = {k: getattr(sampler, k, None)
                                      for k in ("seed", "epoch", "shuffle", "num_replicas", "rank", "drop_last")}
    return out


def _optimizer_section(torch, opt) -> dict:
    keys = ("lr", "betas", "eps", "weight_decay", "momentum", "dampening", "nesterov", "amsgrad", "maximize",
            "foreach", "fused", "capturable", "differentiable", "initial_lr")
    groups = [{**{k: _jsonable(g[k]) for k in keys if k in g}, "tensors": len(g["params"]),
               "numel": sum(p.numel() for p in g["params"])} for g in opt.param_groups]
    paths = []
    for g in opt.param_groups:
        fused, foreach = g.get("fused"), g.get("foreach")
        if fused is None and foreach is None:
            try:
                from torch.optim.optimizer import _default_to_fused_or_foreach
                fused, foreach = _default_to_fused_or_foreach(g["params"], g.get("differentiable", False), False)
            except Exception:
                paths.append("unknown")
                continue
        paths.append("fused" if fused else "foreach" if foreach else "single")
    steps, kinds = [], {}
    for state in opt.state.values():
        for key, value in state.items():
            if key == "step":
                steps.append(float(value))
            elif torch.is_tensor(value):
                kinds.setdefault(key, []).append(value)
    return {"class": type(opt).__name__, "param_groups": groups, "path": paths,
            "state": {"entries": len(opt.state),
                      "step": {"min": min(steps), "max": max(steps)} if steps else None,
                      "fingerprints": {k: _fingerprint(v) for k, v in sorted(kinds.items())}}}


def _scheduler_section(sched) -> dict:
    out = {"class": type(sched).__name__}
    for key in ("last_epoch", "_step_count", "base_lrs"):
        if hasattr(sched, key):
            out[key.lstrip("_")] = _jsonable(getattr(sched, key))
    try:
        out["last_lr"] = _jsonable(sched.get_last_lr())
    except Exception:
        pass
    return out


def _doctor() -> Optional[dict]:
    """OpGuard's own installation report, from a child process so that nothing is
    loaded into this job. None when OpGuard is not installed."""
    try:
        if importlib.util.find_spec("opguard") is None:
            return None
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPGUARD_")}
        env["OPGUARD_AUTOSTART"] = "0"                  # never start a tracer in the child
        done = subprocess.run([sys.executable, "-m", "opguard.doctor", "--json"], capture_output=True,
                              text=True, timeout=120, env=env)
        return json.loads(done.stdout)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def report(*, optimizer=None, dataloader=None, scheduler=None, autocast=None, grad_scaler=None, model=None,
           extra: Optional[dict] = None, doctor: bool = True, gc_scan: bool = True) -> dict:
    """This process's determinism settings as a JSON-serializable dict.

    Pass the objects the job uses to have their settings and state read back:
    ``optimizer`` (path flags fused / foreach, hyperparameters, step counters and
    state fingerprints), ``dataloader`` (workers, seeding, generator),
    ``scheduler`` (counters), ``autocast`` (the context object the loop enters)
    and ``grad_scaler`` (AMP), ``model`` (parameter fingerprint, compiled or not).
    In a multi-rank job every rank calls this and writes its own file.
    """
    import torch
    rep: Dict[str, Any] = {
        "schema": SCHEMA, "taken_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "launch": {"argv": list(sys.argv), "pid": os.getpid()},
        "rank": _rank_section(), "stack": _stack_section(),
        "torch": _torch_section(torch, autocast, grad_scaler, model, gc_scan),
        "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES",
                                               "CUDA_LAUNCH_BLOCKING", "PYTHONHASHSEED", "OMP_NUM_THREADS",
                                               "OPGUARD_DETERMINISTIC")},
        "nccl": {"version": None, "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")}},
        "seeds": _seed_section(torch),
        "dataloader": _dataloader_section(dataloader) if dataloader is not None else None,
        "optimizer": _optimizer_section(torch, optimizer) if optimizer is not None else None,
        "scheduler": _scheduler_section(scheduler) if scheduler is not None else None,
        "model": None, "opguard_doctor": _doctor() if doctor else None, "extra": _jsonable(extra or {}),
    }
    rep["nccl"]["version"] = rep["stack"].get("nccl")
    if model is not None:
        params = list(model.parameters())
        rep["model"] = {"class": type(model).__name__, "tensors": len(params),
                        "numel": sum(p.numel() for p in params), "training": bool(model.training),
                        "dtypes": sorted({str(p.dtype) for p in params}),
                        "devices": sorted({str(p.device) for p in params}),
                        "parameters_fingerprint": _fingerprint(params)}
    return _jsonable(rep)


# ------------------------------------------------------------------ findings
def findings(rep: dict) -> List[dict]:
    """Surfaces one run leaves uncontrolled, or that limit what a capture sees.
    ``kind`` is "risk" (residual nondeterminism: two launches of this job may
    differ) or "note" (legitimate, but both runs must match on it)."""
    rep = rep.get("preflight", rep)
    out: List[dict] = []

    def add(kind, path, value, why, fix):
        out.append({"kind": kind, "path": path, "value": value, "why": why, "fix": fix})
    t = rep["torch"]
    det = t["deterministic_algorithms"]
    strict = "torch.use_deterministic_algorithms(True, warn_only=False)"
    if not det["enabled"]:
        add("risk", "torch.deterministic_algorithms.enabled", False,
            "Library kernels may take nondeterministic algorithms (index_add, scatter, attention backward).",
            strict)
    elif det["warn_only"]:
        add("risk", "torch.deterministic_algorithms.warn_only", True,
            "Ops without a deterministic implementation only warn and stay nondeterministic. Measured: the "
            f"backward of SDPA flash and memory-efficient attention differs run to run in dQ ({SDPA_CASE}).",
            strict)
    if t["cudnn"]["benchmark"]:
        add("risk", "torch.cudnn.benchmark", True, "cuDNN autotuning picks algorithms by timing.",
            "torch.backends.cudnn.benchmark = False")
    if not t["cudnn"]["deterministic"]:
        add("risk", "torch.cudnn.deterministic", False, "cuDNN may select nondeterministic algorithms.",
            "torch.backends.cudnn.deterministic = True")
    if rep["stack"].get("gpus") and not rep["env"].get("CUBLAS_WORKSPACE_CONFIG"):
        add("risk", "env.CUBLAS_WORKSPACE_CONFIG", None,
            "cuBLAS is deterministic only with a fixed workspace configuration.",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before CUDA starts")
    if t["cuda_matmul_allow_tf32"]:
        add("note", "torch.cuda_matmul_allow_tf32", True, "TF32 changes fp32 GEMM arithmetic.",
            "torch.backends.cuda.matmul.allow_tf32 = False, or the same on both runs")
    if t["float32_matmul_precision"] != "highest":
        add("note", "torch.float32_matmul_precision", t["float32_matmul_precision"],
            "Below 'highest', fp32 matmul may run in TF32.", 'torch.set_float32_matmul_precision("highest")')
    if (rep["rank"]["world_size"] or 1) > 1 and not any(k in rep["nccl"]["env"] for k in ("NCCL_ALGO", "NCCL_PROTO")):
        add("note", "nccl.env", rep["nccl"]["env"],
            "Collective algorithm and protocol are not pinned: the reduction tree can follow message size "
            "and topology (handbook/3-kernels/collectives.md).", "pin NCCL_ALGO and NCCL_PROTO; verify")
    dl = rep.get("dataloader")
    if dl and dl["generator"] is None and (dl["shuffle"] or dl["num_workers"]):
        add("note", "dataloader.generator", None,
            "Shuffle order and worker seeds are drawn from the global torch RNG at iteration time.",
            "pass generator=torch.Generator().manual_seed(seed)")
    opt = rep.get("optimizer")
    if opt and "fused" in opt["path"]:
        add("note", "optimizer.path", opt["path"], "The fused optimizer kernel is its own arithmetic path.",
            "use the same path on both runs")
    comp, graphs = t["compile"], t["cuda_graphs"]
    if comp["model_is_compiled"] or comp["compiled_graphs"] or graphs["graph_objects"] or graphs["capturing_now"]:
        add("note", "torch.compile / torch.cuda_graphs",
            {"compiled_graphs": comp["compiled_graphs"], "model_is_compiled": comp["model_is_compiled"],
             "graph_objects": graphs["graph_objects"]},
            "OpGuard captures eager execution: boundaries inside compiled code and CUDA graphs are not "
            "recorded by it.", "capture those with the handbook's compiled-code tool")
    return out


# ------------------------------------------------------------------ diff
_VOLATILE = ("taken_at", "launch.pid")
# first matching prefix wins: (prefix, effect, blocking, why)
_RULES = [
    ("launch.", "launch", False, "The launch command differs: make sure this is the difference under test."),
    ("rank.world_size", "different_arithmetic", True,
     "A different parallel degree is a different reduction tree (L5)."),
    ("rank.", "informational", False, "Compare reports of the same rank."),
    ("stack.gpus", "different_arithmetic", True,
     "Kernel selection and tile configuration follow the GPU architecture and SM count."),
    ("stack.libraries.numpy", "informational", False, "Host library."),
    ("stack.python", "informational", False, "Host interpreter."),
    ("stack.", "different_arithmetic", True,
     "Library kernels and heuristics change between versions: compare across versions only when the "
     "upgrade is the question."),
    ("torch.deterministic_algorithms.fill_uninitialized_memory", "different_arithmetic", True,
     "Uninitialized memory is filled on one side only."),
    ("torch.deterministic_algorithms", "different_arithmetic", True,
     "The two runs select different algorithms, and at least one may be nondeterministic by itself."),
    ("torch.cudnn", "different_arithmetic", True, "Different cuDNN algorithm selection or precision."),
    ("torch.cuda_matmul_allow_tf32", "different_arithmetic", True, "TF32 on one side changes every fp32 GEMM."),
    ("torch.float32_matmul_precision", "different_arithmetic", True, "Different fp32 matmul precision."),
    ("torch.sdpa_backends", "different_arithmetic", True, "Attention may dispatch to a different backend."),
    ("torch.default_dtype", "different_arithmetic", True, "Different default precision."),
    ("torch.amp.enabled_now", "informational", False, "Depends on where the report was taken."),
    ("torch.amp.dtype_now", "informational", False, "Depends on where the report was taken."),
    ("torch.amp", "different_arithmetic", True, "Different mixed-precision settings."),
    ("torch.compile", "different_arithmetic", True,
     "Compiled code can fuse and move rounding points: it matches eager only if every rounding op is opaque."),
    ("torch.cuda_graphs", "different_arithmetic", True, "One side replays CUDA graphs."),
    ("env.CUBLAS_WORKSPACE_CONFIG", "different_arithmetic", True, "Different cuBLAS workspace configuration."),
    ("env.CUDA_VISIBLE_DEVICES", "informational", False,
     "Different devices. OpGuard compares the device string of every tensor, so both runs should see "
     "their GPU under the same index."),
    ("env.CUDA_LAUNCH_BLOCKING", "informational", False, "Changes timing only."),
    ("env.", "informational", False, "Environment variable."),
    ("nccl.", "different_arithmetic", True, "Different collective algorithm, protocol or library version."),
    ("seeds.", "different_inputs", True,
     "Different seeds or RNG stream positions: random draws (init, dropout, sampling) differ from here on."),
    ("dataloader.num_workers", "different_inputs", True,
     "A different worker count changes which process draws each sample's randomness."),
    ("dataloader.pin_memory", "informational", False, "Host memory option."),
    ("dataloader.prefetch_factor", "informational", False, "Host queue option."),
    ("dataloader.persistent_workers", "informational", False, "Host process option."),
    ("dataloader.", "different_inputs", True, "The two runs do not read the same batches in the same order."),
    ("optimizer.path", "different_arithmetic", True,
     "fused, foreach and single-tensor are different code paths (their bitwise agreement is untested here)."),
    ("optimizer.state", "different_state", True,
     "Optimizer state differs (step counters or moments): updates differ from the first step."),
    ("optimizer.", "different_arithmetic", True, "Different optimizer or hyperparameters."),
    ("scheduler.", "different_state", True, "Scheduler counters differ: the learning rate differs."),
    ("model.training", "different_arithmetic", True, "train() on one side, eval() on the other."),
    ("model.", "different_state", True, "The two runs do not hold the same weights."),
    ("opguard_doctor.", "informational", False, "Capture tooling."),
    ("extra.", "informational", False, "Caller-supplied."),
]


def _flatten(value, path="", out=None) -> dict:
    out = {} if out is None else out
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            out[path] = value
        else:
            for i, v in enumerate(value):
                _flatten(v, f"{path}[{i}]", out)
    else:
        out[path] = value
    return out


def diff(report_a: dict, report_b: dict, declared: Sequence[str] = ()) -> dict:
    """What differs between two runs' reports, and whether a bitwise comparison of
    the runs still means anything.

    Each difference gets an ``effect``:
        different_arithmetic  the runs are configured to compute different bits;
        different_inputs      they do not see the same data or random draws;
        different_state       they do not start from the same weights or counters;
        launch, informational listed, never blocking.
    The first three are ``blocking``: with one of them present, the first
    divergence follows from the configuration and localizes nothing else.
    ``declared`` names path prefixes that are the difference under test (for
    example "launch.argv" or "stack.torch" for an upgrade); they are listed and
    do not block. ``at_risk`` repeats the single-run findings of both sides.
    """
    a, b = report_a.get("preflight", report_a), report_b.get("preflight", report_b)
    fa, fb = _flatten(a), _flatten(b)
    differences = []
    for path in sorted(set(fa) | set(fb)):
        if path in _VOLATILE or fa.get(path, "<absent>") == fb.get(path, "<absent>"):
            continue
        effect, blocking, why = "informational", False, "Not classified."
        for prefix, rule_effect, rule_blocking, rule_why in _RULES:
            if path.startswith(prefix):
                effect, blocking, why = rule_effect, rule_blocking, rule_why
                break
        is_declared = any(path.startswith(d) for d in declared)
        differences.append({"path": path, "a": fa.get(path, "<absent>"), "b": fb.get(path, "<absent>"),
                            "effect": effect, "blocking": blocking and not is_declared,
                            "declared": is_declared, "why": why})
    blocking = [d for d in differences if d["blocking"]]
    risks: Dict[str, dict] = {}
    for side, rep in (("a", a), ("b", b)):
        for item in findings(rep):
            if item["kind"] == "risk":
                risks.setdefault(item["path"], {**item, "sides": []})["sides"].append(side)
    return {"schema": DIFF_SCHEMA, "differences": differences, "blocking": blocking,
            "declared": [d["path"] for d in differences if d["declared"]],
            "at_risk": list(risks.values()),
            "bitwise_comparison": "not_meaningful" if blocking else "meaningful",
            "summary": (f"{len(differences)} differences, {len(blocking)} blocking; "
                        f"{len(risks)} uncontrolled surfaces")}


def diff_ranks(reports_a: Sequence[dict], reports_b: Sequence[dict], declared: Sequence[str] = ()) -> dict:
    """``diff`` for multi-rank runs: reports are paired by their rank."""
    def by_rank(reports):
        return {r.get("preflight", r)["rank"]["rank"]: r for r in reports}
    a, b = by_rank(reports_a), by_rank(reports_b)
    return {"missing_ranks": sorted(set(a) ^ set(b)),
            "ranks": {str(r): diff(a[r], b[r], declared) for r in sorted(set(a) & set(b))}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the report (or the diff) here")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="compare two report files")
    ap.add_argument("--declare", nargs="*", default=[], help="path prefixes that are the difference under test")
    args = ap.parse_args(argv)
    if args.diff:
        a, b = (json.load(open(p)) for p in args.diff)
        result = diff(a, b, args.declare)
        for d in result["differences"]:
            mark = "BLOCK" if d["blocking"] else "decl " if d["declared"] else "     "
            print(f"{mark} {d['path']}: {d['a']!r} vs {d['b']!r}  [{d['effect']}]")
        for r in result["at_risk"]:
            print(f"RISK  {r['path']} = {r['value']!r} (sides {','.join(r['sides'])}): {r['why']}")
        print(f"bitwise comparison: {result['bitwise_comparison']} ({result['summary']})")
    else:
        result = report()
        for item in findings(result):
            print(f"{item['kind'].upper():5} {item['path']} = {item['value']!r}: {item['why']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    elif not args.diff:
        print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
