"""Pre-training alignment end to end: preflight, OpGuard capture, first divergence, class, fix.

A small LM (workload/) trains for a few steps. Three planted divergences from
chapter 1's root-cause table are each run as a reference/suspect pair:

    norm      the norm multiplies by its weight in fp32 and rounds once, where the
              reference rounds before the weight (a rounding point moved)
    backward  the same job launched twice with deterministic algorithms in
              warn_only mode: SDPA's backward stays nondeterministic
    resume    a resume that restores everything but the optimizer's step counter

What runs:
1. fingerprints  tools/opguard_bridge.py in plain torch against the documented
                 format and, when OpGuard is importable, against OpGuard's own
                 functions bit for bit (CPU and GPU, 1 GiB tensors in full mode)
2. preflight     tools/determinism_preflight.py: report, findings, diff
3. scenarios     the three pairs and their fixes, in process, final parameters
                 compared bit for bit (no OpGuard needed)
4. opguard       full mode only: every pair through OpGuard's harness (discover,
                 prepare, trace, compare), then first_divergence and suggest
5. writer        full mode only: records written by TraceWriter, exported by
                 OpGuard and read back through its harness tools

    python pretraining_e2e.py --quick --json out.json          # parts 1-3, no OpGuard
    python pretraining_e2e.py --opguard-root ~/OpGuard --json results-a16.json

--quick checks only what needs no OpGuard and says so. Full mode needs a
runnable OpGuard (source checkout or installed) and fails without one.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent.parent / "tools"
WORKLOAD = HERE / "workload"
sys.path[:0] = [str(TOOLS), str(WORKLOAD)]

import torch  # noqa: E402

from bi_check import bit_equal, first_mismatch  # noqa: E402
from bi_report import Results  # noqa: E402

STEPS, SAVE_AT = 4, 2
QUICK = dict(layers=2, dim=256, heads=4, seq_len=1024, batch=4, vocab=2048, corpus_seqs=64)
FULL = dict(layers=4, dim=384, heads=6, seq_len=1024, batch=8, vocab=4096, corpus_seqs=256)


# ----------------------------------------------------------------------------- helpers
def argv_for(size: dict, **opts) -> list:
    merged = {"steps": STEPS, **size, **opts}
    out = []
    for key, value in merged.items():
        if value is not None:
            out += ["--" + key.replace("_", "-"), str(value)]
    return out


def train(size: dict, **opts) -> dict:
    """One run of the workload in this process."""
    import train_lm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return train_lm.run(train_lm.parse_args(argv_for(size, **opts)))


def same_state(a: dict, b: dict):
    """Bitwise comparison of two parameter dicts through bi_check."""
    bad = {}
    for key in a:
        if not bit_equal(a[key], b[key]):
            mm = first_mismatch(a[key], b[key])
            bad[key] = {"mismatched_elements": mm.get("mismatched_elements"), "of": mm.get("of")}
    return not bad, {"parameters": len(a), "differing": len(bad), "first": dict(list(bad.items())[:3])}


def timed(fn, dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    return out, time.time() - t0


# ----------------------------------------------------------------------------- 1. fingerprints
def fingerprint_tensors(dev):
    gen = torch.Generator().manual_seed(1)

    def make(shape, dtype):
        if dtype == torch.bool:
            return (torch.rand(shape, generator=gen) > 0.5).to(dev)
        if dtype.is_floating_point:
            return (torch.randn(shape, generator=gen) * 100).to(dtype).to(dev)
        return torch.randint(-2 ** 31, 2 ** 31 - 1, shape, generator=gen).to(dtype).to(dev)
    specs = [("bf16", (64, 16, 128), torch.bfloat16), ("fp16 (462 bytes)", (33, 7), torch.float16),
             ("fp32", (5, 6, 7), torch.float32), ("fp64", (9, 3), torch.float64),
             ("int64", (11, 4), torch.int64), ("bool (1001 bytes)", (1001,), torch.bool),
             ("uint8 (91 bytes)", (13, 7), torch.uint8), ("bf16 (6 bytes)", (3,), torch.bfloat16),
             ("empty", (0, 4), torch.float32), ("0-dim fp32", (), torch.float32),
             ("0-dim bf16", (), torch.bfloat16)]
    for label, shape, dtype in specs:
        base = make(shape, dtype)
        yield label, base
        if base.dim() >= 2:
            yield label + ", transposed", base.transpose(0, -1)
            yield label + ", strided rows", base[::2]
        if base.numel() > 4:
            yield label + ", offset by one element", base.reshape(-1)[1:]


def fingerprints(res: Results, quick: bool) -> None:
    import opguard_bridge as ob
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    cases = ob.selftest(verbose=False)
    res.check("plain torch equals the documented format (pure-Python reference)",
              all(c["passed"] for c in cases), level="other", impl="opguard_bridge plain torch",
              detail={"cases": len(cases), "failed": [c["case"] for c in cases if not c["passed"]][:10],
                      "also_against_opguard": ob.opguard_available()})
    if not ob.opguard_available():
        res.check("plain torch equals OpGuard's functions", True, level="other", expect="info",
                  impl="opguard_bridge plain torch", detail="skipped: opguard is not importable in this process")
        return
    native = getattr(ob._opguard_xor(), "_xor_reduce_cuda_ext", None) is not None
    for dev in devices:
        failed, n = [], 0
        for label, t in fingerprint_tensors(dev):
            for regions in (1, 4, 7, 64):
                n += 1
                mine = ob.xor_signature(t, regions, impl="torch")
                theirs = ob.xor_signature(t, regions, impl="opguard")
                if not bit_equal(mine.cpu(), theirs.cpu()):
                    failed.append(f"{label} regions={regions}")
            n += 1
            mine, theirs = ob.xor_vectors(t, impl="torch"), ob.xor_vectors(t, impl="opguard")
            if (mine is None) != (theirs is None) or (mine is not None and not bit_equal(mine.cpu(), theirs.cpu())):
                failed.append(f"{label} vectors")
        res.check(f"xor_signature and xor_vectors: plain torch equals OpGuard bit for bit ({dev})", not failed,
                  level="other", impl="opguard_bridge plain torch",
                  detail={"comparisons": n, "failed": failed[:10], "opguard_native_extension": native,
                          "covered": "bf16, fp16, fp32, fp64, int64, bool, uint8; byte counts 462, 1001, 91, 6; "
                                     "empty; 0-dim; transposed, strided and offset views; regions 1, 4, 7, 64"})
    if quick or not torch.cuda.is_available():
        return
    big = {"fp32 [65536, 4096] (1 GiB)": lambda: torch.randn(65536, 4096, device="cuda"),
           "bf16 [2^29 + 1] (1 GiB + 2 bytes)": lambda: torch.randn(2 ** 29 + 1, device="cuda").to(torch.bfloat16)}
    for label, build in big.items():
        t = build()
        for dev in ("cuda", "cpu"):
            x = t if dev == "cuda" else t.cpu()
            ok, cost = True, {}
            for name, fn in (("signature", lambda impl: ob.xor_signature(x, 4, impl=impl)),
                             ("vectors", lambda impl: ob.xor_vectors(x, impl=impl))):
                fn("torch"), fn("opguard")                                  # warm up
                mine, cost[f"{name}_torch_s"] = timed(lambda: fn("torch"), dev)
                theirs, cost[f"{name}_opguard_s"] = timed(lambda: fn("opguard"), dev)
                if mine is None and theirs is None:
                    cost[name] = "not defined (vector bytes not a multiple of 4)"
                    continue
                ok = ok and bit_equal(mine.cpu(), theirs.cpu())
            res.check(f"1 GiB tensor: plain torch equals OpGuard bit for bit ({dev})", ok, level="other",
                      impl="opguard_bridge plain torch", detail={"tensor": label, **{k: round(v, 4) if isinstance(v, float) else v for k, v in cost.items()}})
            for key, value in cost.items():
                if isinstance(value, float):
                    res.measure(f"{key[:-2]} of {label} on {dev}", round(value * 1000, 2), "ms")
            del x
        del t
        torch.cuda.empty_cache()

    # the plain-torch path also runs inside compiled code and CUDA graphs (OpGuard's native one does not)
    def probe(x):
        y = x * 2
        return ob.xor_signature(y, 4, impl="torch"), ob.xor_vectors(y, impl="torch")
    x = torch.randn(64, 512, device="cuda").to(torch.bfloat16)
    eager = probe(x)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compiled, seconds = timed(lambda: torch.compile(probe)(x), "cuda")
        ok = all(bit_equal(a.cpu(), b.cpu()) for a, b in zip(eager, compiled))
        res.check("fingerprints computed inside torch.compile equal eager", ok, level="other", expect="info",
                  impl="opguard_bridge plain torch", detail={"first_call_seconds": round(seconds, 1)})
    except Exception as exc:                                                # recorded, not judged
        res.check("fingerprints computed inside torch.compile equal eager", False, level="other", expect="info",
                  impl="opguard_bridge plain torch", detail=f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
    try:
        static = x.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                probe(static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = probe(static)
        fresh = torch.randn(64, 512, device="cuda").to(torch.bfloat16)
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        ok = all(bit_equal(a.cpu(), b.cpu()) for a, b in zip(probe(fresh), captured))
        res.check("fingerprints computed in a replayed CUDA graph equal eager", ok, level="other", expect="info",
                  impl="opguard_bridge plain torch")
        del graph
    except Exception as exc:
        res.check("fingerprints computed in a replayed CUDA graph equal eager", False, level="other", expect="info",
                  impl="opguard_bridge plain torch", detail=f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")


def merge_rule(res: Results) -> None:
    import opguard_bridge as ob
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = (torch.randn(38, 4096, device=dev) * 10).to(torch.bfloat16)
    heads = ob.xor_vectors(q.view(38, 32, 128), impl="torch")              # 1216 head vectors
    rows = ob.xor_vectors(q, impl="torch")                                 # 38 token rows
    ok = bit_equal(ob.merge(heads, group=32).cpu(), rows.cpu())
    ok = ok and ob.merge(ob.decode_vectors(ob.encode_vectors(heads, 256)), group=32) == [v & 0xFFFFFFFF for v in rows.cpu().tolist()]
    x = (torch.randn(64, 4096, device=dev)).to(torch.bfloat16)
    sig = [int(v) for v in ob.xor_signature(x, 4, impl="torch").cpu().tolist()]
    ok = ok and ob.merge(ob.xor_vectors(x, impl="torch").cpu().tolist(), regions=4, vector_bytes=8192) == sig
    ok = ok and ob.merge(ob.xor_vectors(x, impl="torch").cpu().tolist()) == [int(ob.xor_signature(x, 1, impl="torch").cpu()[0])]
    legacy = [v - (1 << 32) if v & 0x80000000 else v for v in ob.decode_vectors(ob.encode_vectors(rows, 8192))]
    ok = ok and ob.decode_vectors(legacy) == ob.decode_vectors(ob.encode_vectors(rows, 8192))
    ok = ok and ob.decode_vectors(float(legacy[0])) == [legacy[0] & 0xFFFFFFFF]
    try:
        ob.merge(heads, regions=7, vector_bytes=256)
        ok = False                                                         # 7 regions cut through vectors
    except ValueError:
        pass
    res.check("merge rule: head vectors -> token rows -> regions -> whole tensor; v1 and legacy decode", ok,
              level="other", impl="opguard_bridge plain torch")


# ----------------------------------------------------------------------------- 2. preflight
def preflight(res: Results, doctor: bool) -> dict:
    import determinism_preflight as pf
    from torch.utils.data import DataLoader, TensorDataset
    model = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.zeros(16, 2)), batch_size=4, shuffle=True, num_workers=2,
                        generator=torch.Generator().manual_seed(3))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    kw = dict(optimizer=opt, dataloader=loader, scheduler=sched, model=model,
              autocast=torch.autocast("cuda", dtype=torch.bfloat16))
    base = pf.report(doctor=doctor, **kw)
    needed = ["torch.deterministic_algorithms.enabled", "torch.deterministic_algorithms.warn_only",
              "torch.cudnn.deterministic", "torch.cudnn.benchmark", "torch.cudnn.allow_tf32",
              "torch.cuda_matmul_allow_tf32", "torch.float32_matmul_precision", "env.CUBLAS_WORKSPACE_CONFIG",
              "torch.sdpa_backends.flash", "nccl.version", "nccl.env", "seeds.torch_initial_seed",
              "seeds.torch_rng_state_sha256", "dataloader.num_workers", "dataloader.generator.initial_seed",
              "dataloader.worker_init_fn", "optimizer.path", "optimizer.state.step", "torch.amp.context.dtype",
              "torch.compile.model_is_compiled", "torch.cuda_graphs.graph_objects", "stack.gpus", "stack.driver",
              "stack.cuda_build", "stack.libraries", "rank.rank", "scheduler.last_epoch",
              "model.parameters_fingerprint"]
    flat = pf._flatten(base)
    missing = [k for k in needed if not any(p == k or p.startswith(k + ".") or p.startswith(k + "[") for p in flat)
               and k not in ("optimizer.state.step", "nccl.env")]
    res.check("report holds every control surface of chapter 1 that a job can read back", not missing,
              level="other", impl="determinism_preflight", detail={"missing": missing, "paths": len(flat)})

    # plant differences one at a time and read them back through diff()
    planted = {}
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = not prev
    planted["torch.cuda_matmul_allow_tf32"] = pf.diff(base, pf.report(doctor=False, **kw))
    torch.backends.cuda.matmul.allow_tf32 = prev
    state = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=True)
    warn = pf.report(doctor=False, **kw)
    torch.use_deterministic_algorithms(True, warn_only=False)
    strict = pf.report(doctor=False, **kw)
    torch.use_deterministic_algorithms(state[0], warn_only=state[1])
    planted["torch.deterministic_algorithms.warn_only"] = pf.diff(strict, warn)
    torch.rand(1)                                                          # one draw moves the CPU stream
    planted["seeds.torch_rng_state_sha256"] = pf.diff(base, pf.report(doctor=False, **kw))
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=torch.cuda.is_available() or None)
    planted["optimizer.path"] = pf.diff(base, pf.report(doctor=False, **{**kw, "optimizer": opt2}))
    ok = all(any(d["path"] == path and d["blocking"] for d in result["differences"]) and
             result["bitwise_comparison"] == "not_meaningful" for path, result in planted.items()
             if path != "optimizer.path" or torch.cuda.is_available())
    same = pf.diff(strict, strict)
    declared = pf.diff(strict, warn, declared=["torch.deterministic_algorithms"])
    ok = ok and same["bitwise_comparison"] == "meaningful" and not same["differences"]
    ok = ok and declared["bitwise_comparison"] == "meaningful" and declared["declared"]
    res.check("diff lists each planted difference and marks it blocking; a declared one does not block", ok,
              level="other", impl="determinism_preflight",
              detail={path: [d["path"] for d in result["blocking"]] for path, result in planted.items()})
    risk = [f["path"] for f in pf.findings(warn) if f["kind"] == "risk"]
    res.check("findings flag warn_only=True as residual nondeterminism", "torch.deterministic_algorithms.warn_only" in risk
              and "torch.deterministic_algorithms.warn_only" not in [f["path"] for f in pf.findings(strict)],
              level="other", impl="determinism_preflight", detail={"risks_under_warn_only": risk})
    return base


# ----------------------------------------------------------------------------- 3. scenarios in process
def scenarios(res: Results, size: dict, work: Path, quick: bool) -> dict:
    """The three pairs and their fixes without OpGuard: final parameters, bit for bit."""
    import determinism_preflight as pf
    base = train(size)                                                     # the reference configuration
    again = train(size)
    ok, detail = same_state(base["state"], again["state"])
    res.check("reference configuration, two runs: identical parameters", ok, level="L1",
              impl="strict deterministic, norm rounds before the weight", detail=detail)
    res.measure("train step, reference configuration", round(base["train_seconds"] / STEPS * 1000, 1), "ms", size=size)

    fused = train(size, norm_mul="fp32")
    ok, detail = same_state(base["state"], fused["state"])
    res.check("norm: reference vs suspect, identical parameters", ok, level="L4", expect="fail",
              impl="norm multiplies in fp32, rounds after the weight (planted)", detail=detail)

    warn_a, warn_b = train(size, det="warn"), train(size, det="warn")
    ok, detail = same_state(warn_a["state"], warn_b["state"])
    res.check("backward: two runs of the same job, identical parameters", ok, level="L1", expect="fail",
              impl="SDPA flash backward, deterministic algorithms warn_only=True", detail=detail)
    ok, detail = same_state(base["state"], warn_a["state"])
    res.check("backward: a warn_only run equals a strict run", ok, level="L1", expect="info",
              impl="SDPA flash backward, deterministic algorithms warn_only=True", detail=detail)
    res.measure("train step, warn_only (default SDPA backward)", round(warn_a["train_seconds"] / STEPS * 1000, 1), "ms")
    risk = [f["path"] for f in pf.findings(warn_a["preflight"]) if f["kind"] == "risk"]
    res.check("backward: preflight findings name warn_only before any capture",
              risk == ["torch.deterministic_algorithms.warn_only"], level="other", impl="determinism_preflight",
              detail={"risks": risk, "diff_of_the_two_runs": pf.diff(warn_a, warn_b)["summary"]})

    ckpt = str(work / "phase1.pt")
    train(size, steps=SAVE_AT, save=ckpt)
    resumed = train(size, resume=ckpt)
    ok, detail = same_state(base["state"], resumed["state"])
    res.check(f"resume: {SAVE_AT} steps, save, full restore, {STEPS - SAVE_AT} steps equals {STEPS} uninterrupted steps",
              ok and resumed["loss_bits"] == base["loss_bits"][SAVE_AT:], level="L1", impl="full restore", detail=detail)
    broken = train(size, resume=ckpt, resume_skip="opt-step")
    ok, detail = same_state(base["state"], broken["state"])
    res.check("resume: restore without the optimizer step counter, identical parameters", ok, level="L1",
              expect="fail", impl="restore without optimizer step counter (planted)", detail=detail)
    blocking = [d["path"] for d in pf.diff(resumed, broken)["blocking"]]
    res.check("resume: preflight diff after the restore names the optimizer step counter",
              bool(blocking) and all(p.startswith("optimizer.state.step") for p in blocking), level="other",
              impl="determinism_preflight", detail={"blocking": blocking})
    if not quick:
        for piece in ("sched", "cuda-rng", "cpu-rng", "data-pos"):
            other = train(size, resume=ckpt, resume_skip=piece)
            ok, detail = same_state(base["state"], other["state"])
            blocking = [d["path"] for d in pf.diff(resumed, other)["blocking"]]
            res.check(f"resume: restore without {piece}, identical parameters", ok, level="L1", expect="info",
                      impl=f"restore without {piece}", detail={**detail, "preflight_blocking": blocking[:4]})
    losses = {"reference": base["losses"], "norm suspect": fused["losses"], "resume suspect": broken["losses"]}
    return {"ckpt": ckpt, "base_state": base["state"], "resumed_state": resumed["state"], "losses": losses,
            "param_names": base["param_names"]}


# ----------------------------------------------------------------------------- 4. OpGuard end to end
def trim_facts(facts: dict) -> dict:
    out = {k: v for k, v in facts.items() if k not in ("tools",)}
    for site in out.get("sites", []):
        site["statuses"] = site["statuses"][:200]
    return out


def opguard_pair(res: Results, og, work: Path, name: str, size: dict, ref: dict, sus: dict, gpu: str,
                 declared=(), expect_diverge=True, level="L1", impl="", ref_state=None) -> dict:
    """One reference/suspect pair through OpGuard's harness, then the bridge."""
    import determinism_preflight as pf
    import opguard_bridge as ob
    root = work / name
    ref_dir, sus_dir = ob.stage_checkouts(WORKLOAD, root / "checkouts")

    def side(checkout, tag, opts):
        args = argv_for(size, **opts) + ["--out", str(root / f"{tag}.json"), "--state-out", str(root / f"{tag}.pt")]
        return ob.workload(checkout, "train_lm.py", args, og=og, pythonpath=[TOOLS],
                           env={"CUDA_VISIBLE_DEVICES": gpu, "OPGUARD_ENABLE_CHECKSUM_NONCONTIG": "1"})
    run = ob.run_pair(side(ref_dir, "reference", ref), side(sus_dir, "suspect", sus), root, og, timeout=900,
                      label=name, progress=lambda s: print(f"  {name}: {s['side']} {s['action']} {s['status']} {s['seconds']}s", flush=True))
    out = {"reference_args": ref, "suspect_args": sus, "status": run["status"], "launches": run["launches"],
           "wall_seconds": run["wall_seconds"], "verdict": run["verdict"], "errors": run["errors"],
           "stages": [{k: s[k] for k in ("side", "action", "status", "seconds")} for s in run["stages"]]}
    if run["status"] != "succeeded":
        res.check(f"{name}: OpGuard capture and comparison complete", False, level="other", impl="OpGuard harness",
                  detail=run["errors"])
        return out
    out["coverage"] = ob.coverage(run)
    reports = {tag: json.load(open(root / f"{tag}.json")) for tag in ("reference", "suspect")}
    pdiff = pf.diff(reports["reference"], reports["suspect"], declared=declared)
    out["preflight"] = {"summary": pdiff["summary"], "bitwise_comparison": pdiff["bitwise_comparison"],
                        "blocking": [{k: d[k] for k in ("path", "a", "b", "effect")} for d in pdiff["blocking"]],
                        "declared": [{k: d[k] for k in ("path", "a", "b")} for d in pdiff["differences"] if d["declared"]],
                        "at_risk": [{k: r[k] for k in ("path", "value", "sides")} for r in pdiff["at_risk"]]}
    t0 = time.time()
    facts = ob.first_divergence(run["reports"][0], root, og)
    out["query_seconds"] = round(time.time() - t0, 1)
    leads = ob.suggest(facts, preflight=pdiff)
    out["facts"], out["suggest"] = trim_facts(facts), leads
    out["param_names"] = reports["reference"]["param_names"]
    states = [torch.load(root / f"{tag}.pt", weights_only=True) for tag in ("reference", "suspect")]
    same, detail = same_state(*states)
    out["final_parameters"] = detail
    unaligned = sum(c.get("unaligned", 0) for c in facts["status_counts"].values())
    counts = {run_name: dict(c) for run_name, c in facts["status_counts"].items()}
    if expect_diverge:
        res.check(f"{name}: OpGuard reports no difference between reference and suspect", not facts["diverged"],
                  level=level, expect="fail", impl=impl, detail={"status_counts": counts})
        res.check(f"{name}: final parameters of reference and suspect are identical", same, level=level,
                  expect="fail", impl=impl, detail=detail)
    else:
        res.check(f"{name}: OpGuard reports no difference and no unaligned event", not facts["diverged"] and not unaligned,
                  level=level, impl=impl, detail={"status_counts": counts, "launches": run["launches"]})
        res.check(f"{name}: final parameters of reference and suspect are identical", same, level=level, impl=impl,
                  detail=detail)
    if ref_state is not None:
        same, detail = same_state(ref_state, states[0])
        res.check(f"{name}: the captured reference run ends with the parameters of an uninstrumented run", same,
                  level="other", impl="OpGuard capture", detail=detail)
    res.measure(f"{name}: wall time of the OpGuard workflow", run["wall_seconds"], "s", launches=run["launches"])
    for tag in ("reference", "suspect"):
        (root / f"{tag}.pt").unlink()
    return out


def opguard_e2e(res: Results, og, work: Path, size: dict, local: dict, gpu: str) -> dict:
    out = {}
    base = local["base_state"]
    out["norm"] = opguard_pair(res, og, work, "norm", size, {}, {"norm_mul": "fp32"}, gpu,
                               declared=["launch.argv"], level="L4", ref_state=base,
                               impl="norm multiplies in fp32, rounds after the weight (planted)")
    out["backward"] = opguard_pair(res, og, work, "backward", size, {"det": "warn"}, {"det": "warn"}, gpu,
                                   impl="SDPA flash backward, deterministic algorithms warn_only=True")
    # the fix of both: round before the weight, and strict deterministic algorithms (one configuration)
    out["norm_and_backward_fixed"] = opguard_pair(
        res, og, work, "norm-backward-fixed", size, {}, {}, gpu, expect_diverge=False,
        impl="strict deterministic, norm rounds before the weight (fix)")
    out["resume"] = opguard_pair(res, og, work, "resume", size, {"resume": local["ckpt"]},
                                 {"resume": local["ckpt"], "resume_skip": "opt-step"}, gpu,
                                 declared=["launch.argv"], ref_state=local["resumed_state"],
                                 impl="restore without optimizer step counter (planted)")
    out["resume_fixed"] = opguard_pair(res, og, work, "resume-fixed", size, {"resume": local["ckpt"]},
                                       {"resume": local["ckpt"]}, gpu, expect_diverge=False, impl="full restore (fix)")
    # coverage: the same model with torch.nn.Linear, whose GEMM is launched from torch's own file
    out["nn_linear"] = opguard_pair(res, og, work, "nn-linear", size, {"linear": "nn"}, {"linear": "nn"}, gpu,
                                    expect_diverge=False, impl="strict deterministic, torch.nn.Linear")
    for name in ("norm", "nn_linear"):
        cov = (out[name].get("coverage") or {}).get("baseline")
        if cov:
            res.check(f"coverage with {'own Linear' if name == 'norm' else 'torch.nn.Linear'}: every discovered "
                      "call site of the checkout is planned", not cov["discovered_not_planned"], level="other",
                      expect="info", impl="OpGuard harness", detail=cov)
    return out


# ----------------------------------------------------------------------------- 5. trace writer
def writer(res: Results, og, work: Path) -> dict:
    import opguard_bridge as ob
    root = work / "writer"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(7)
    xs = [torch.randn(4, 8, 64, generator=gen).to(torch.bfloat16).to(dev) for _ in range(5)]
    for tag, flip in (("a", False), ("b", True)):
        with ob.TraceWriter(root / tag, vectors=True, impl="torch") as out:
            for i, x in enumerate(xs):
                y = x * 2
                if flip and i == 3:                                        # one bit of element 1500
                    y.view(-1).view(torch.int16)[1500] ^= 1
                out.record(f"layer{i % 2}.op", inputs=[x], outputs=[y], line=10 + i % 2)
    files = sorted(str(p.relative_to(root / "a")) for p in (root / "a").rglob("*.jsonl"))
    rec = [[json.loads(line) for line in open(root / tag / "exec_trace/gpu_000/ops/layer1.op/layer1.op_line11.jsonl")]
           for tag in ("a", "b")]
    va, vb = (ob.decode_vectors(r[1]["outputs"]["stats"]["xor_vectors"]) for r in rec)
    legacy = ob.decode_vectors(rec[1][1]["outputs"]["stats"]["xor_rows"])
    differing = [i for i, (p, q) in enumerate(zip(va, vb)) if p != q]
    res.check("writer: per-vector fingerprints differ at the planted vector only; v1 and legacy forms agree",
              differing == [1500 // 64] and legacy == vb, level="other", impl="opguard_bridge.TraceWriter",
              detail={"files": files, "differing_vectors": differing})
    if og is None:
        res.check("writer: OpGuard exports the pair and its harness reports the planted record", True, level="other",
                  expect="info", impl="opguard_bridge.TraceWriter", detail="skipped: needs OpGuard")
        return {}
    export = ob.export_pair(root / "a", root / "b", root / "compare", og)
    facts = ob.first_divergence(export["reports"][0], root, og)
    pivot = facts.get("pivot") or {}
    regions = [r["differing_regions"] for r in pivot.get("regional") or [] if r["slot"][0] == "outputs"]
    ok = (facts["diverged"] and pivot.get("fn") == "<layer1.op>" and pivot.get("counter") == 4
          and pivot.get("diff_fields") == ["outputs"] and regions == [[2]] and facts["unaligned_before"] == 0
          and facts["status_counts"]["run1"] == {"ok": 4, "diff": 1})
    res.check("writer: OpGuard exports the pair and its harness reports the planted record", ok, level="other",
              impl="opguard_bridge.TraceWriter",
              detail={"tools": facts["tools"], "pivot": {k: pivot.get(k) for k in ("fn", "counter", "line", "diff_fields")},
                      "differing_regions_of_output": regions, "status_counts": facts["status_counts"],
                      "verdict": export["verdict"]})
    return {"export": {k: export[k] for k in ("verdict", "ranks", "coverage")}, "pivot": pivot}


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="parts 1-3 at reduced sizes; no OpGuard capture")
    ap.add_argument("--json", help="write results here (bi-results-v1)")
    ap.add_argument("--opguard-root", default=os.environ.get("OPGUARD_ROOT"), help="OpGuard source checkout")
    ap.add_argument("--python", default=sys.executable, help="training interpreter for the captured runs")
    ap.add_argument("--work", help="directory for checkouts, captures and exports (default: a temporary one)")
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU")
        return 2
    if args.opguard_root:
        sys.path.insert(0, str(Path(args.opguard_root).expanduser()))
    import opguard_bridge as ob

    res = Results("pretraining_e2e", quick=args.quick)
    og = None
    try:
        og = ob.locate_opguard(args.opguard_root, args.python)
    except ob.OpGuardError as exc:
        if not args.quick:
            print(f"full mode needs a runnable OpGuard: {exc}")
            return 2
    work = Path(args.work).expanduser().resolve() if args.work else Path(tempfile.mkdtemp(prefix="pretraining-e2e-"))
    work.mkdir(parents=True, exist_ok=True)
    size = QUICK if args.quick else FULL
    res.extra.update(mode="quick: no OpGuard capture is run" if args.quick else "full",
                     sizes=size, steps=STEPS, save_at=SAVE_AT,
                     opguard={"root": og and og["root"], "capabilities": og and og["capabilities"],
                              "importable_here": ob.opguard_available()})
    try:
        fingerprints(res, args.quick)
        merge_rule(res)
        report = preflight(res, doctor=og is not None and not args.quick)
        res.extra["opguard_doctor"] = report.get("opguard_doctor")
        local = scenarios(res, size, work, args.quick)
        res.extra["losses"] = local["losses"]
        if args.quick:
            print("quick mode: the OpGuard captures (part 4) and the export of written traces (part 5) are not run")
            writer(res, None, work)
        else:
            res.extra["scenarios"] = opguard_e2e(res, og, work, size, local, args.gpu)
            res.extra["writer"] = writer(res, og, work)
    finally:
        if not args.keep and not args.work:
            shutil.rmtree(work, ignore_errors=True)
    res.write(args.json)
    return res.exit_code()


if __name__ == "__main__":
    sys.exit(main())
