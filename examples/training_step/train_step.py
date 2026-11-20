"""Training-step determinism on one GPU. See handbook/1-pretraining/training-step.md.

Sections (each runs in its own process, in its own mode):
  ops      1-2  every op of an LM step, forward+backward, 20 runs, default and strict mode;
                the fixed-order kernel (ts_kernels.seg_sum) against its written definition
  train    3    a 4-layer decoder LM, AdamW, 20 steps, run twice in separate processes:
                first differing step and tensor in default, strict, fix_l1, fix_all
  optim    4    AdamW and SGD: fused / foreach / single-tensor paths, and a model of each
                path's rounding order checked bit for bit
  misc     5,6,8  loss reduction layouts, GradScaler, gradient clipping
  accum    7    K micro-batches against one batch; fixed split repeats; order changes bits
  resume   9    20 straight steps against 10 + save + new process + 10; drop one piece each
  cost     10   step time in each mode

Run:  python train_step.py [--quick] [--json results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))

from ts_util import MODES, mode_env  # noqa: E402


def spawn(section: str, mode: str, quick: bool, extra: list = ()) -> dict:
    """Run one section in a fresh process in the given mode and return its JSON."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix=f"ts-{section}-{mode}-")
    os.close(fd)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", section, "--mode", mode, "--out", path]
    cmd += ["--quick"] if quick else []
    cmd += list(extra)
    if os.environ.get("BI_LOCK"):                     # shared GPU: one lock per worker process
        cmd = ["flock", os.environ["BI_LOCK"]] + cmd
    t0 = time.time()
    p = subprocess.run(cmd, env=mode_env(mode))
    if p.returncode != 0:
        raise RuntimeError(f"worker {section}/{mode} exited with {p.returncode}")
    with open(path) as f:
        out = json.load(f)
    os.unlink(path)
    out["_seconds"] = round(time.time() - t0, 1)
    return out


# ----------------------------------------------------------------------------- workers
def worker_ops(args) -> dict:
    import ts_ops
    runs = 5 if args.quick else 20
    out = {"runs": runs}
    print(f"[ops/{args.mode}] run-to-run, {runs} runs per case", flush=True)
    out["ops"] = ts_ops.run_ops(args.quick, runs, with_fixed=args.mode == "default")
    print(f"[ops/{args.mode}] which written order reproduces the stock op", flush=True)
    out["orders"] = [] if args.quick else ts_ops.stock_order_models(args.quick)
    if args.mode == "default":
        print("[ops] fixed-order kernel against its definition", flush=True)
        out["fixes"] = ts_ops.fixes_checks(args.quick)
    return out


def train_args(rest):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="default", choices=("default", "strict", "fix_l1", "fix_all"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--scaler", action="store_true")
    ap.add_argument("--growth", type=int, default=2000)
    ap.add_argument("--init-scale", type=float, default=65536.0)
    ap.add_argument("--opt", default="default")
    ap.add_argument("--clip-path", default="foreach")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--accum-reverse", action="store_true")
    ap.add_argument("--save-at", type=int, default=-1)
    ap.add_argument("--ckpt")
    ap.add_argument("--resume")
    ap.add_argument("--omit", default="")
    ap.add_argument("--what", default="trajectory", choices=("trajectory", "time", "repeat"))
    return ap.parse_args(rest)


def build_trainer(a):
    from ts_model import Cfg, TrainCfg, Trainer, preset
    cfg = Cfg(n_layers=a.layers, dropout=a.dropout, max_len=a.seq, **preset(a.preset))
    tc = TrainCfg(batch=a.batch, seq=a.seq, steps=a.steps, amp=a.amp, scaler=a.scaler,
                  scaler_growth_interval=a.growth, scaler_init_scale=a.init_scale, opt=a.opt,
                  clip_path=a.clip_path, accum=a.accum,
                  accum_reverse=a.accum_reverse)
    return Trainer(cfg, tc)


def worker_train(args) -> dict:
    """One training run in a fresh process. Returns per-step fingerprints (trajectory), the
    median step time (time), or the element counts that differ between two backward passes
    from the same state (repeat)."""
    import torch
    from ts_model import cfg_dict
    from ts_util import first_mismatch, sync
    a = train_args(args.rest)
    tr = build_trainer(a)
    out = {"config": cfg_dict(tr.cfg, tr.tc), "preset": a.preset, "mode": args.mode}
    if a.what == "repeat":
        x, y = tr.sampler.next()
        x, y = x.to(DEV), y.to(DEV)
        runs = []
        for _ in range(4):
            tr.opt.zero_grad(set_to_none=True)
            loss = tr.forward_loss(x, y)
            loss.backward()
            runs.append((loss.detach().clone(), {n: p.grad.clone() for n, p in tr.model.backward_order()}))
        out["repeat"] = []
        for i in range(1, len(runs)):
            d = {"loss_equal": bool(torch.equal(runs[0][0], runs[i][0])), "tensors": {}}
            for n in runs[0][1]:
                mm = first_mismatch(runs[0][1][n], runs[i][1][n])
                if mm:
                    d["tensors"][n] = [mm["mismatched_elements"], mm["of"]]
            out["repeat"].append(d)
        return out
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        tr.restore(ck, omit=[o for o in a.omit.split(",") if o])
        start = ck["step"]
        out["resumed_from"] = start
        out["omit"] = a.omit
    if a.what == "time":
        import statistics
        import time as _t
        for _ in range(5):
            tr.step(record=False)
        sync()
        ts = []
        for _ in range(a.steps):
            t0 = _t.perf_counter()
            tr.step(record=False)
            sync()
            ts.append((_t.perf_counter() - t0) * 1e3)
        out["ms_per_step"] = round(statistics.median(ts), 3)
        out["ms_min"] = round(min(ts), 3)
        out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return out
    recs = []
    for s in range(start, a.steps):
        if s == a.save_at:
            torch.save(tr.checkpoint(), a.ckpt)
        recs.append(tr.step())
    out["trajectory"] = recs
    return out


def worker_optim(args) -> dict:
    import ts_optim
    return ts_optim.sec_optim(args.quick)


def worker_misc(args) -> dict:
    import ts_sections
    out = {}
    for name in ("loss", "amp", "clip"):
        print(f"[{name}]", flush=True)
        out[name] = getattr(ts_sections, "sec_" + name)(args.quick)
    return out


WORKERS = {"ops": worker_ops, "train": worker_train, "optim": worker_optim, "misc": worker_misc}
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--worker", choices=sorted(WORKERS))
    ap.add_argument("--mode", choices=MODES, default="default")
    ap.add_argument("--out")
    args, rest = ap.parse_known_args()
    args.rest = rest
    if args.worker:
        import torch
        from ts_util import apply_mode
        apply_mode(args.mode)
        out = WORKERS[args.worker](args)
        torch.cuda.synchronize()
        with open(args.out, "w") as f:
            json.dump(out, f)
        return
    sys.exit(run_all(args))


def pair(preset, mode, steps, extra=()):
    """Two separate processes, same seed and data; their per-step fingerprints compared."""
    from ts_model import first_divergence
    a = spawn("train", mode, False, ["--preset", preset, "--steps", str(steps), *extra])
    b = spawn("train", mode, False, ["--preset", preset, "--steps", str(steps), *extra])
    d = first_divergence(a["trajectory"], b["trajectory"])
    d.pop("grads_equal_here", None) if d.get("equal") else None
    return d, a


def compare_runs(a, b):
    from ts_model import first_divergence
    d = first_divergence(a["trajectory"], b["trajectory"])
    if d.get("equal"):
        d.pop("grads_equal_here", None)
    return d


def run_all(args) -> int:
    from bi_report import Results
    res = Results("training_step", quick=args.quick)
    q = args.quick
    t_all = time.time()

    # ---------------------------------------------------------------- 1-2 ops
    ops = {m: spawn("ops", m, q) for m in ("default", "strict")}
    res.extra["ops"] = {m: {"runs": o["runs"], "ops": o["ops"], "orders": o["orders"]} for m, o in ops.items()}
    for rec_d, rec_s in zip(ops["default"]["ops"], ops["strict"]["ops"]):
        name = f"{rec_d['op']} [{rec_d['case']}]"
        sd = rec_d["stock"]
        if "raised" not in sd:
            res.check(f"{name}: stock, default mode, {ops['default']['runs']} runs", sd["passed"], level="L1",
                      impl="stock", expect="info" if sd["passed"] else "fail")
        ss = rec_s["stock"]
        if "raised" in ss:
            res.check(f"{name}: strict mode raises", True, level="L1", impl="strict", expect="info",
                      detail=ss["raised"])
        else:
            res.check(f"{name}: strict mode, {ops['strict']['runs']} runs", ss["passed"], level="L1", impl="strict")
        if "fixed" in rec_d:
            fx = rec_d["fixed"]
            res.check(f"{name}: fixed-order version, {ops['default']['runs']} runs", fx.get("passed", False),
                      level="L1", impl="fixed", detail=fx.get("raised"))
    for f in ops["default"]["fixes"]:
        res.check(f["name"], f["passed"], level="L4" if "NumPy" in f["name"] else "L2",
                  impl="stock" if f.get("expect") == "fail" else "fixed", expect=f.get("expect", "pass"),
                  detail=f["detail"])

    # ---------------------------------------------------------------- 3 whole step
    steps = 8 if q else 20
    res.extra["train"] = {}
    for preset, mode, expect in (("default", "default", "fail"), ("strict", "strict", "pass"),
                                 ("fix_l1", "default", "pass"), ("fix_all", "default", "pass")):
        d, a = pair(preset, mode, steps)
        res.extra["train"][preset] = {"divergence": d, "loss_bits": [r["loss"] for r in a["trajectory"]],
                                      "config": a["config"]}
        res.check(f"whole step, {preset}: two processes, {steps} steps, loss, every gradient and weight equal",
                  d["equal"], level="L1", impl=preset, expect=expect, detail=d)
        print(f"[train] {preset}: {d}", flush=True)
    lb = {p: v["loss_bits"][0] for p, v in res.extra["train"].items()}
    res.extra["train"]["step0_loss_bits"] = lb

    # ---------------------------------------------------------------- 4 optimizer
    o = spawn("optim", "default", q)
    res.extra["optim"] = o
    for kind in ("adamw", "sgd"):
        for path, ok in o[kind]["repeat"].items():
            res.check(f"{kind} {path}: 100 steps twice from the same state", ok, level="L1", impl=path)
        for pair_name, by in o[kind]["pairs"].items():
            for s_, d in by.items():
                if s_ in ("1", "100"):
                    res.check(f"{kind} {pair_name} after {s_} step(s)", d["equal"], level="L4", impl=pair_name,
                              expect="info", detail={k: v for k, v in d.items() if k != "equal"})
        for path, m in o[kind]["models"].items():
            if path != "fused":
                res.check(f"{kind} {path}: written rounding order reproduces every bit, {m['steps']} steps",
                          bool(m["exact_models"]), level="L4", impl=path, detail=m["exact_models_described"])
            else:
                res.check(f"{kind} fused: fp64 model of the kernel reproduces every bit", bool(m["exact_models"]),
                          level="L4", impl=path, expect="info", detail=m["mismatches_first_steps"])

    # ---------------------------------------------------------------- 5, 6, 8
    m = spawn("misc", "default", q)
    res.extra["loss"], res.extra["amp"], res.extra["clip"] = m["loss"], m["amp"], m["clip"]
    for c in m["loss"]["cases"]:
        tag = f"B={c['B']} T<={c['Tmax']}"
        for k, v in c["stock_equal_to_packed_mean"].items():
            if k != "packed .mean()":
                res.check(f"loss mean {tag}: {k} equals packed .mean()", v, level="L2", impl="stock", expect="info")
        for k, v in c["fixed_equal_to_padded"].items():
            res.check(f"tree_mean_loss {tag}: {k} equals padded", v, level="L2", impl="fixed")
        for k, v in c["division"].items():
            res.check(f"{tag}: {k}", v, level="other", impl="stock", expect="info")
        for k, v in c["cross_entropy"].items():
            res.check(f"{tag}: {k}", v, level="L2", impl="fixed" if "tree" in k else "stock",
                      expect="pass" if "tree" in k else "info")
    for c in m["amp"]["scale_vs_none"]:
        res.check(f"GradScaler {c['amp']} scale {c['scale']}: unscaled gradients equal an unscaled backward",
                  c["equal"], level="other", impl=f"scaler_{c['amp']}",
                  expect="pass" if c["amp"] == "bf16" else "info", detail=c)
    for opt, d in m["amp"]["skip"].items():
        res.check(f"GradScaler inf skip ({opt} AdamW): parameters, moments and step count unchanged",
                  d["params_unchanged"] and d["exp_avg_unchanged"] and d["step_counts"] == [1.0],
                  level="other", impl=opt, detail=d)
    for path in ("foreach", "single", "fixed"):
        res.check(f"clip_grad_norm_ {path}: run to run", m["clip"][path]["passed"], level="L1", impl=path)
    res.check("clip_grad_norm_ foreach vs per-tensor: norm and clipped gradients equal",
              m["clip"]["foreach_vs_single"]["norm_equal"] and m["clip"]["foreach_vs_single"]["grads_equal"],
              level="L4", impl="stock", expect="info")

    # ---------------------------------------------------------------- 7 accumulation
    acc_steps = 4 if q else 10
    base = ["--preset", "fix_all", "--steps", str(acc_steps)]
    one = spawn("train", "default", False, base + ["--batch", "4", "--accum", "1"])
    k4 = spawn("train", "default", False, base + ["--batch", "1", "--accum", "4"])
    k4b = spawn("train", "default", False, base + ["--batch", "1", "--accum", "4"])
    k4r = spawn("train", "default", False, base + ["--batch", "1", "--accum", "4", "--accum-reverse"])
    k2 = spawn("train", "default", False, base + ["--batch", "2", "--accum", "2"])
    acc = {"one batch of 4 vs 4 micro-batches of 1": compare_runs(one, k4),
           "4 micro-batches of 1, twice": compare_runs(k4, k4b),
           "4 micro-batches of 1, reversed order": compare_runs(k4, k4r),
           "2 micro-batches of 2 vs 4 of 1": compare_runs(k2, k4)}
    res.extra["accum"] = acc
    for k, d in acc.items():
        print(f"[accum] {k}: {d}", flush=True)
        res.check(f"gradient accumulation: {k}", d["equal"], level="other", impl="fix_all",
                  expect="pass" if "twice" in k else "fail", detail=d)

    # ---------------------------------------------------------------- 9 resume
    rs = 8 if q else 20
    half = rs // 2
    rargs = ["--preset", "fix_all", "--steps", str(rs), "--dropout", "0.1", "--scaler", "--growth", "3"]
    tmpd = tempfile.mkdtemp(prefix="ts-resume-")
    ck = os.path.join(tmpd, "ck.pt")
    straight = spawn("train", "default", False, rargs + ["--save-at", str(half), "--ckpt", ck])
    tail = {"trajectory": straight["trajectory"][half:]}
    omissions = ["", "model", "optimizer", "optimizer_step", "scheduler", "scaler", "cpu_rng", "cuda_rng", "data"]
    if q:
        omissions = ["", "cuda_rng", "optimizer_step"]
    res.extra["resume"] = {"steps": rs, "saved_at": half, "dropout": 0.1, "scaler_growth_interval": 3, "rows": []}
    for om in omissions:
        r = spawn("train", "default", False, rargs + ["--resume", ck, "--omit", om])
        d = compare_runs(tail, r)
        res.extra["resume"]["rows"].append({"omitted": om or "nothing", **d})
        print(f"[resume] omit {om or 'nothing'}: {d.get('step', 'equal')} {d.get('kind', '')} {d.get('tensor', '')}", flush=True)
        # bf16 with GradScaler: the scale is a power of two and unscaling is exact (section 6), so
        # a wrong scale changes no bits; the omission is recorded, not judged
        res.check(f"resume at step {half}, omitting {om or 'nothing'}: equal to the straight run",
                  d["equal"], level="L1", impl="resume",
                  expect="pass" if not om else "info" if om == "scaler" else "fail", detail=d)
    # cuda RNG only matters with dropout
    if not q:
        rargs0 = [a for a in rargs if a not in ("0.1",)]
        rargs0[rargs0.index("--dropout") + 1:rargs0.index("--dropout") + 1] = ["0.0"]
        straight0 = spawn("train", "default", False, rargs0 + ["--save-at", str(half), "--ckpt", ck])
        r0 = spawn("train", "default", False, rargs0 + ["--resume", ck, "--omit", "cuda_rng"])
        d0 = compare_runs({"trajectory": straight0["trajectory"][half:]}, r0)
        res.extra["resume"]["dropout0_omit_cuda_rng"] = d0
        res.check("resume without the CUDA RNG state, dropout 0: equal", d0["equal"], level="L1", impl="resume",
                  expect="info", detail=d0)
    os.unlink(ck)
    os.rmdir(tmpd)

    # ---------------------------------------------------------------- 10 cost
    res.extra["cost"] = {}
    if not q:
        for preset, mode in (("default", "default"), ("strict", "strict"), ("fix_l1", "default"), ("fix_all", "default")):
            t = spawn("train", mode, False, ["--preset", preset, "--steps", "15", "--what", "time"])
            res.extra["cost"][preset] = {k: t[k] for k in ("ms_per_step", "ms_min", "peak_mem_gb")}
            res.measure(f"training step B=4 T=1024, 4 layers, {preset}", t["ms_per_step"], "ms", impl=preset)
            print(f"[cost] {preset}: {res.extra['cost'][preset]}", flush=True)
    for m_ in ("default", "strict"):
        for rec in ops[m_]["ops"]:
            for lab in ("stock", "fixed"):
                if lab in rec and "ms" in rec[lab]:
                    res.measure(f"{rec['op']} [{rec['case']}]", rec[lab]["ms"], "ms",
                                impl=f"{lab}/{m_}" if lab == "stock" else "fixed")
    res.extra["seconds"] = round(time.time() - t_all, 1)
    res.write(args.json)
    return res.exit_code()


if __name__ == "__main__":
    main()
