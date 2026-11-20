"""Gated delta rule (GDN-style) linear attention: run-to-run, batch and mode
invariance, recurrent vs chunked, and cost.

Kernels in gdn_kernels.py: a recurrent kernel (one program per sequence, head and
Dv tile; fp32 state in registers) and a chunked WY-form kernel (chunk C=64, bf16 or
fp32 matmul operands). Shapes: 8 heads, Dk = Dv = 128; q, k L2-normalized and v in
bf16; log-decay fp32; beta bf16; fp32 initial and final state. Outputs are compared
in fp32, before the cast to bf16 (stricter than comparing the bf16 output).

1. repeat (L1)  B=4, T=1024: the same batch 5 times; output and final state bitwise
2. batch (L2)   bi_check.check_batch: a T=200 sequence with a nonzero initial state,
                alone vs among others (batch 1-65), first/middle/last, fillers
                random/zeros/large
3. modes (L3)   T=300, B=2, every mode continuing from the saved fp32 state.
                recurrent: full prefill vs per-token decode vs chunked prefill 7/16/64/100.
                chunked: full prefill vs
                  calls at multiples of C (64, 128, 64, 44 tokens);
                  call-relative chunks (each call starts its own chunks: decode as
                  1-token calls, 7, 16, 100);
                  aligned (AlignedChunked: each call restarts from the last absolute
                  multiple of C: decode, 7, 16, 64, 100)
4. cross        recurrent vs chunked: elements that differ; error vs an fp64 recurrence
5. configs      recurrent kernel tile/warp configurations: L1, L2, L3 each, and bits vs the default
6. timing       forward B x T in {8x2048, 4x4096, 2x8192, 1x8192}; one decode step at
                B in {1, 16, 64}, eager and CUDA-graph replay

Run (CUDA, Triton):  python gdn_bi.py --json results.json [--only modes cross] [--quick]
Regression suite:    python gdn_bi.py --quick --bi-json out.json
                     (--quick: every case at reduced sizes; --bi-json: the checks in the bi-results-v1
                     schema, through suite/adapters/linear_attention.py)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, HERE)
from bi_check import bits, check_batch, check_repeat, first_mismatch  # noqa: E402
from gdn_kernels import BF, F32, FIELDS, AlignedChunked, gdn_chunked, gdn_recurrent  # noqa: E402

dev = "cuda"
H, D, C = 8, 128, 64
REC_CFG = (8, 1, True)                 # (Dv tile, warps, prefetch) of the recurrent kernel under test
CONFIGS = ((8, 1, True), (8, 1, False), (16, 1, True), (16, 2, True), (32, 2, True), (32, 4, True),
           (64, 4, True))
SIZES = (1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65)
Q = dict(repeat_T=1024, batch_T=200, modes_T=300, cross_T=2048,
         shapes=((8, 2048), (4, 4096), (2, 8192), (1, 8192)), decode_B=(1, 16, 64))


# ----------------------------------------------------------------------------- inputs
def make_inputs(B, T, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=gen, device=dev)
    return {"q": F.normalize(r(B, T, H, D), dim=-1).to(BF),
            "k": F.normalize(r(B, T, H, D), dim=-1).to(BF),
            "v": r(B, T, H, D).to(BF),
            "glog": F.logsigmoid(r(B, T, H) + 3.0),          # decay exp(glog), mostly 0.88-0.98
            "beta": torch.sigmoid(r(B, T, H)).to(BF),
            "h0": r(B, H, D, D) * 0.1}


def sl(x, s, e):
    return {f: x[f][:, s:e].contiguous() for f in FIELDS}


def run_rec(x, cfg=None):
    BV, nw, pf = cfg or REC_CFG
    return gdn_recurrent(*(x[f] for f in FIELDS), x["h0"], BV=BV, num_warps=nw, prefetch=pf)


def cfg_name(cfg):
    return f"BV={cfg[0]},warps={cfg[1]},prefetch={int(cfg[2])}"


def run_chk(prec):
    return lambda x: gdn_chunked(*(x[f] for f in FIELDS), x["h0"], prec=prec)[:2]


IMPLS = {"recurrent": run_rec, "chunked_bf16": run_chk("bf16"), "chunked_fp32": run_chk("fp32")}


# check_batch works on [M, N] rows: pack one sequence's inputs into one fp32 row.
def pack(x):
    B = x["q"].shape[0]
    return torch.cat([x[f].float().reshape(B, -1) for f in FIELDS + ("h0",)], 1)


def unpacker(T):
    shapes = {"q": (T, H, D), "k": (T, H, D), "v": (T, H, D), "glog": (T, H), "beta": (T, H), "h0": (H, D, D)}
    dtypes = {"q": BF, "k": BF, "v": BF, "glog": F32, "beta": BF, "h0": F32}

    def unpack(rows):
        B, off, x = rows.shape[0], 0, {}
        for f in FIELDS + ("h0",):
            n = math.prod(shapes[f])
            x[f] = rows[:, off:off + n].reshape(B, *shapes[f]).to(dtypes[f]).contiguous()
            off += n
        return x
    return unpack


def flat(o, ht):
    return torch.cat([o.reshape(o.shape[0], -1), ht.reshape(ht.shape[0], -1)], 1)


def compare(base, other):
    (o0, s0), (o1, s1) = base, other
    r = {}
    for nm, a, b in (("o", o0, o1), ("state", s0, s1)):
        mm = first_mismatch(a, b)
        r[nm] = {"mismatched": 0 if mm is None else mm["mismatched_elements"], "of": a.numel(),
                 "max_abs_diff": float((a.double() - b.double()).abs().max())}
    r["o_bf16_mismatched"] = int((bits(o0.to(BF)) != bits(o1.to(BF))).sum())
    r["equal"] = r["o"]["mismatched"] == 0 and r["state"]["mismatched"] == 0
    return r


def short(r):
    if r["equal"]:
        return "equal"
    return (f"o {r['o']['mismatched']}/{r['o']['of']} ({100 * r['o']['mismatched'] / r['o']['of']:.1f}%, "
            f"after bf16 cast {r['o_bf16_mismatched']}), "
            f"state {r['state']['mismatched']}/{r['state']['of']} "
            f"({100 * r['state']['mismatched'] / r['state']['of']:.1f}%), max|d| o {r['o']['max_abs_diff']:.1e}")


# ----------------------------------------------------------------------------- L1, L2
def repeat_case():
    T = Q["repeat_T"]
    rows, unpack = pack(make_inputs(4, T, seed=1)), unpacker(T)
    out = {}
    for name, impl in IMPLS.items():
        rep = check_repeat(lambda r: flat(*impl(unpack(r))), rows, n=5, name=f"{name} repeat")
        out[name] = rep.to_dict()
        print(rep.summary())
    return out


def batch_case():
    T = Q["batch_T"]
    rows, unpack = pack(make_inputs(2, T, seed=2)), unpacker(T)
    out = {}
    for name, impl in IMPLS.items():
        rep = check_batch(lambda r: flat(*impl(unpack(r))), rows, SIZES, name=f"{name} batch")
        out[name] = rep.to_dict() | {"sizes": list(SIZES), "T": T}
        print(rep.summary())
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- L3
def calls(T, c):
    """Call boundaries: c is a call size, or a list of sizes cycled until T."""
    sizes = c if isinstance(c, (list, tuple)) else [c]
    out, s, i = [], 0, 0
    while s < T:
        e = min(s + sizes[i % len(sizes)], T)
        out.append((s, e))
        s, i = e, i + 1
    return out


def rec_calls(x, T, c, cfg=None):
    st, outs = x["h0"], []
    for s, e in calls(T, c):
        o, st = run_rec({**sl(x, s, e), "h0": st}, cfg)
        outs.append(o)
    return torch.cat(outs, 1), st


def chk_calls(x, T, c, prec):
    st, outs = x["h0"], []
    for s, e in calls(T, c):
        y = sl(x, s, e)
        o, st, _ = gdn_chunked(*(y[f] for f in FIELDS), st, prec=prec)
        outs.append(o)
    return torch.cat(outs, 1), st


def chk_aligned(x, T, c, prec):
    cache = AlignedChunked(x["h0"], prec, C)
    outs = [cache(sl(x, s, e)) for s, e in calls(T, c)]
    return torch.cat(outs, 1), cache.final


def run_modes(modes, label):
    items = list(modes.items())
    base_name, base = items[0][0], items[0][1]()
    out = {}
    for name, fn in items[1:]:
        out[f"{name} vs {base_name}"] = r = compare(base, fn())
        print(f"  {label}: {name} vs {base_name}: {short(r)}")
    return out


def modes_case():
    T = Q["modes_T"]
    x = make_inputs(2, T, seed=3)
    res = {"T": T, "B": 2}
    modes = {"prefill": lambda: rec_calls(x, T, T), "decode": lambda: rec_calls(x, T, 1)}
    for c in (7, 16, 64, 100):
        modes[f"chunked-prefill-{c}"] = lambda c=c: rec_calls(x, T, c)
    res["recurrent"] = run_modes(modes, "recurrent")
    for prec in ("bf16", "fp32"):
        modes = {"prefill": lambda: chk_calls(x, T, T, prec),
                 "calls-at-multiples-of-64": lambda: chk_calls(x, T, [64, 128, 64], prec)}
        for c in (1, 7, 16, 100):
            modes[f"call-relative-{c}"] = lambda c=c: chk_calls(x, T, c, prec)
        for c in (1, 7, 16, 64, 100):
            modes[f"aligned-{c}"] = lambda c=c: chk_aligned(x, T, c, prec)
        res[f"chunked_{prec}"] = run_modes(modes, f"chunked_{prec}")
    return res


# ----------------------------------------------------------------------------- cross-algorithm
def ref_fp64(x):
    q, k, v = (x[f].double() for f in ("q", "k", "v"))
    q = q * D ** -0.5
    g, beta, S = x["glog"].double().exp(), x["beta"].double(), x["h0"].double().clone()
    B, T = q.shape[:2]
    o = torch.empty(B, T, H, D, dtype=torch.float64, device=dev)
    for t in range(T):
        S = S * g[:, t, :, None, None]
        u = (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S)) * beta[:, t, :, None]
        S = S + k[:, t, :, :, None] * u[:, :, None, :]
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
    return o, S


def err(a, ref):
    d = a.double() - ref
    return {"max_abs": float(d.abs().max()), "rel_fro": float(d.norm() / ref.norm())}


def cross_case():
    T = Q["cross_T"]
    x = make_inputs(1, T, seed=6)
    ro, rs = ref_fp64(x)
    outs = {n: impl(x) for n, impl in IMPLS.items()}
    res = {"T": T, "B": 1, "error_vs_fp64": {}, "differ": {}}
    for n, (o, s) in outs.items():
        res["error_vs_fp64"][n] = e = {"o": err(o, ro), "o_bf16": err(o.to(BF), ro), "state": err(s, rs)}
        print(f"  fp64 error {n}: o max {e['o']['max_abs']:.2e} rel {e['o']['rel_fro']:.2e}; "
              f"o bf16 rel {e['o_bf16']['rel_fro']:.2e}; state max {e['state']['max_abs']:.2e} rel {e['state']['rel_fro']:.2e}")
    for a, b in (("recurrent", "chunked_bf16"), ("recurrent", "chunked_fp32"), ("chunked_fp32", "chunked_bf16")):
        res["differ"][f"{a} vs {b}"] = r = compare(outs[a], outs[b])
        print(f"  {a} vs {b}: {short(r)}")
    return res


def configs_case():
    T = Q["modes_T"]
    x = make_inputs(2, T, seed=3)
    base = rec_calls(x, T, T, REC_CFG)
    rows1, unpack1 = pack(make_inputs(4, Q["repeat_T"], seed=1)), unpacker(Q["repeat_T"])
    rows2, unpack2 = pack(make_inputs(2, Q["batch_T"], seed=2)), unpacker(Q["batch_T"])
    out = {}
    for cfg in CONFIGS:
        modes = {"prefill": lambda: rec_calls(x, T, T, cfg), "decode": lambda: rec_calls(x, T, 1, cfg),
                 "chunked-prefill-7": lambda: rec_calls(x, T, 7, cfg),
                 "chunked-prefill-100": lambda: rec_calls(x, T, 100, cfg)}
        key = cfg_name(cfg)
        r = run_modes(modes, key)
        l1 = check_repeat(lambda rr: flat(*run_rec(unpack1(rr), cfg)), rows1, n=5, name=f"{key} repeat")
        l2 = check_batch(lambda rr: flat(*run_rec(unpack2(rr), cfg)), rows2, SIZES, name=f"{key} batch")
        out[key] = {"L1": l1.summary(), "L2": l2.summary(), "L3_pass": all(v["equal"] for v in r.values()),
                    "modes": r, "vs_default": compare(base, rec_calls(x, T, T, cfg))}
        print(f"  {key}: {l1.summary()}; {l2.summary()}; L3 {'pass' if out[key]['L3_pass'] else 'FAIL'}; "
              f"vs {cfg_name(REC_CFG)}: "
              f"{short(out[key]['vs_default'])}")
    return out


# ----------------------------------------------------------------------------- timing
def gpu_util():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as ex:  # noqa: BLE001
        return f"n/a ({ex})"


def bench(fn, iters=10, inner=1, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / inner)
    return round(statistics.median(ts), 4)


def graph_bench(fn, iters=10, inner=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    t = bench(g.replay, iters, inner)
    del g
    return t


def other_gpu_pids():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return [int(p) for p in out if p.strip().isdigit() and int(p) != os.getpid()]
    except Exception:  # noqa: BLE001
        return ["n/a"]


class UtilSampler:
    """Samples nvidia-smi once a second in a thread: the A40 is shared, so record
    whether another process held the GPU while timing."""

    def __enter__(self):
        import threading
        self.samples, self.stop = [], threading.Event()

        def run():
            while not self.stop.is_set():
                self.samples.append({"util": gpu_util(), "other_pids": other_gpu_pids()})
                self.stop.wait(1.0)
        self.th = threading.Thread(target=run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join()


def timing_case():
    with UtilSampler() as us:
        res = timing_inner()
    res["gpu_util_samples"] = us.samples
    busy = [u for u in us.samples if u["other_pids"]]
    res["samples_with_other_gpu_processes"] = f"{len(busy)}/{len(us.samples)}"
    print(f"  samples with another process on the GPU: {len(busy)}/{len(us.samples)} "
          f"{sorted({p for u in busy for p in u['other_pids']}, key=str)}")
    return res


def timing_inner():
    res = {"gpu_util_before": gpu_util(), "other_pids_before": other_gpu_pids()}
    fwd = {}
    for B, T in Q["shapes"]:
        x = make_inputs(B, T, seed=4)
        row = {}
        for cfg in CONFIGS:
            row[f"recurrent {cfg_name(cfg)}"] = bench(lambda: run_rec(x, cfg), iters=5)
        for prec in ("bf16", "fp32"):
            fn = run_chk(prec)
            row[f"chunked_{prec}"] = bench(lambda: fn(x), iters=5)
            evs = []
            for _ in range(2):
                evs = []
                gdn_chunked(*(x[f] for f in FIELDS), x["h0"], prec=prec, events=evs)
                torch.cuda.synchronize()
            row[f"chunked_{prec} prepare/state/out"] = [round(evs[i].elapsed_time(evs[i + 1]), 4) for i in range(3)]
        rec_best = min(v for k, v in row.items() if k.startswith("recurrent"))
        row["recurrent_best / chunked_bf16"] = round(rec_best / row["chunked_bf16"], 2)
        row["recurrent default / chunked_bf16"] = round(row[f"recurrent {cfg_name(REC_CFG)}"] / row["chunked_bf16"], 2)
        fwd[f"{B}x{T}"] = row
        print(f"  forward {B}x{T}: " + ", ".join(f"{k} {v}" for k, v in row.items()))
        del x
        torch.cuda.empty_cache()
    res["forward_ms"] = fwd
    dec = {}
    for B in Q["decode_B"]:
        x = make_inputs(B, C, seed=5)
        h0 = x["h0"]
        one = sl(x, 0, 1)
        fns = {"recurrent step": lambda: run_rec({**one, "h0": h0})}
        for p in (0, 15, 31, 47, 63):   # p = 0 is also the call-relative (1-token chunk) decode step
            y = sl(x, 0, p + 1)
            fns[f"chunked_bf16, {p + 1}-token call ({p} pending)"] = \
                lambda y=y: gdn_chunked(*(y[f] for f in FIELDS), h0, prec="bf16")
        row = {}
        for name, fn in fns.items():
            row[name] = {"eager_ms": bench(fn, iters=10, inner=20), "graph_ms": graph_bench(fn)}
        al = [v for k, v in row.items() if k.startswith("chunked_bf16")]
        row["chunked_bf16 aligned, mean over 0/15/31/47/63 pending"] = {
            m: round(sum(v[m] for v in al) / len(al), 4) for m in ("eager_ms", "graph_ms")}
        dec[f"B={B}"] = row
        print(f"  decode B={B}: " + "; ".join(f"{k} eager {v['eager_ms']} graph {v['graph_ms']}" for k, v in row.items()))
    res["decode_step_ms"] = dec
    res["gpu_util_after"] = gpu_util()
    return res


CASES = {"repeat": repeat_case, "batch": batch_case, "modes": modes_case, "cross": cross_case,
         "configs": configs_case, "timing": timing_case}


def main():
    global SIZES
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="results file; existing keys are kept unless re-run")
    ap.add_argument("--only", nargs="*", choices=list(CASES))
    ap.add_argument("--quick", action="store_true", help="reduced sizes, for the regression suite")
    ap.add_argument("--bi-json", help="also write the checks as bi-results-v1 (tools/bi_report.py)")
    args = ap.parse_args()
    if args.quick:
        SIZES = (1, 2, 3, 8, 31, 33)
        Q.update(repeat_T=130, batch_T=70, modes_T=150, cross_T=200, shapes=((2, 512),), decode_B=(1,))
    import triton
    res = {}
    if args.json and os.path.exists(args.json):
        with open(args.json) as fh:
            res = json.load(fh)
    res.update({"device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
                "shapes": {"H": H, "Dk": D, "Dv": D, "C": C, "recurrent_cfg": cfg_name(REC_CFG)}})
    for name in args.only or list(CASES):
        t0 = time.time()
        print(f"[{name}]")
        try:
            res[name] = CASES[name]()
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            res[name] = {"error": f"{type(ex).__name__}: {ex}"[:500]}
        print(f"[{name}] {time.time() - t0:.1f}s")
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(res, fh, indent=1)
    if args.bi_json:
        sys.path.insert(0, os.path.join(HERE, "..", "..", "suite"))
        from adapters import emit
        sys.exit(emit("linear_attention", res, args.bi_json, quick=args.quick))


if __name__ == "__main__":
    main()
