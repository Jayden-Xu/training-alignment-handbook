"""Attention backward: run-to-run (L1) and batch (L2) invariance of dQ, dK, dV.

Compares the fixed-block Triton attention with its deterministic, atomic-free
backward (tools/bi_kernels.py) against torch SDPA backward (flash and
memory-efficient backends), each with and without
torch.use_deterministic_algorithms.

1. forward      the forward that also stores log-sum-exp is bitwise the plain forward
2. accuracy     O, dQ, dK, dV against an fp64 reference (GQA, T not a block multiple)
3. repeat (L1)  the same batch, 5 runs, gradients compared bitwise
4. batch (L2)   a sequence alone vs among others (batch 2, 3, 8, 16; first and last)
5. padded (L2)  a 200-token sequence alone vs right-padded to 256 / 320 inside a
                batch, with zero output-gradient on the padding
6. timing       forward and forward+backward, B=4, T=2048, Hq=8, Hk=2, D=128

SDPA runs as multi-head attention on K/V expanded to the query heads (what a
model's repeat_kv does); its invariance is checked on the gradients of the
expanded K/V, so the expansion's own reduction is not part of the test.

Run (CUDA, Triton):  python attn_bwd.py --json results.json
Regression suite:    python attn_bwd.py --quick --bi-json out.json
                     (--quick: batch sizes 2, 3, 8 and fewer timing iterations; --bi-json: the checks
                     in the bi-results-v1 schema, through suite/adapters/attention_backward.py)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import torch  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from bi_check import bit_equal, first_mismatch  # noqa: E402
from bi_kernels import BF, attention_fixed_blocks, attention_fixed_blocks_train  # noqa: E402

dev = "cuda"
sdpa = torch.nn.functional.scaled_dot_product_attention
NAMES = ("o", "dq", "dk", "dv")
QUICK = False


@contextlib.contextmanager
def deterministic(on: bool):
    # warn_only=True is not enough: SDPA backward then only warns and still runs
    # its default (non-deterministic) algorithm. Measured: dQ differed run to run.
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(on, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


class Impl:
    """One attention implementation under one setting. `gqa` says whether it takes
    K/V with Hk heads (True) or K/V already expanded to Hq heads (False)."""

    def __init__(self, name, fn, gqa, backend=None, det=False):
        self.name, self.fn, self.gqa, self.backend, self.det = name, fn, gqa, backend, det
        self.warnings = set()

    def grads(self, q, k, v, do):
        q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        ctx = sdpa_kernel([self.backend]) if self.backend is not None else contextlib.nullcontext()
        with warnings.catch_warnings(record=True) as caught, deterministic(self.det), ctx:
            warnings.simplefilter("always")
            o = self.fn(q, k, v)
            o.backward(do)
            torch.cuda.synchronize()
        self.warnings.update(str(w.message).split("\n")[0][:200] for w in caught)
        return o.detach(), q.grad, k.grad, v.grad


def _sdpa_causal(q, k, v):
    return sdpa(q, k, v, is_causal=True)


IMPLS = [
    Impl("fixed_blocks", attention_fixed_blocks_train, gqa=True),
    Impl("sdpa_flash", _sdpa_causal, gqa=False, backend=SDPBackend.FLASH_ATTENTION),
    Impl("sdpa_flash_det", _sdpa_causal, gqa=False, backend=SDPBackend.FLASH_ATTENTION, det=True),
    Impl("sdpa_efficient", _sdpa_causal, gqa=False, backend=SDPBackend.EFFICIENT_ATTENTION),
    Impl("sdpa_efficient_det", _sdpa_causal, gqa=False, backend=SDPBackend.EFFICIENT_ATTENTION, det=True),
]


def rand(gen, *shape):
    return torch.randn(*shape, generator=gen, device=dev).to(BF)


def inputs(gen, B, T, Hq, Hk, D, impl):
    hk = Hk if impl.gqa else Hq
    return rand(gen, B, Hq, T, D), rand(gen, B, hk, T, D), rand(gen, B, hk, T, D), rand(gen, B, Hq, T, D)


def compare(ref, out, tag):
    fails = []
    for name, a, b in zip(NAMES, ref, out):
        mm = first_mismatch(a, b)
        if mm:
            fails.append({"case": tag, "tensor": name, **mm})
    return fails


# ----------------------------------------------------------------------------- 1. forward
def forward_case():
    gen = torch.Generator(device=dev).manual_seed(1)
    out = {}
    for T in (200, 256, 1000):
        q, k, v = rand(gen, 2, 8, T, 128), rand(gen, 2, 2, T, 128), rand(gen, 2, 2, T, 128)
        plain = attention_fixed_blocks(q, k, v, 0)
        lse = torch.empty(2 * 8, T, device=dev, dtype=torch.float32)
        with_lse = attention_fixed_blocks(q, k, v, 0, lse=lse)
        train = attention_fixed_blocks_train(q, k, v)
        out[f"T={T}"] = {"with_lse_equals_plain": bit_equal(plain, with_lse),
                         "train_equals_plain": bit_equal(plain, train)}
    return out


# ----------------------------------------------------------------------------- 2. accuracy
def reference_fp64(q, k, v, do):
    q, k, v = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    g = q.shape[1] // k.shape[1]
    ke, ve = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    T, D = q.shape[2], q.shape[3]
    s = q @ ke.transpose(-1, -2) / math.sqrt(D)
    s = s.masked_fill(torch.ones(T, T, device=dev, dtype=torch.bool).triu(1), float("-inf"))
    o = torch.softmax(s, -1) @ ve
    o.backward(do.double())
    return o.detach(), q.grad, k.grad, v.grad


def accuracy_case():
    gen = torch.Generator(device=dev).manual_seed(2)
    B, T, Hq, Hk, D = 2, 500, 8, 2, 128
    q, k, v, do = rand(gen, B, Hq, T, D), rand(gen, B, Hk, T, D), rand(gen, B, Hk, T, D), rand(gen, B, Hq, T, D)
    ref = reference_fp64(q, k, v, do)
    g = Hq // Hk

    def gqa_sdpa(q, k, v):
        return sdpa(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True)
    impls = [Impl("fixed_blocks", attention_fixed_blocks_train, True),
             Impl("sdpa_flash", gqa_sdpa, True, SDPBackend.FLASH_ATTENTION),
             Impl("sdpa_efficient", gqa_sdpa, True, SDPBackend.EFFICIENT_ATTENTION)]
    out = {}
    for impl in impls:
        got = impl.grads(q, k, v, do)
        out[impl.name] = {
            name: {"max_abs": float((a.double() - r).abs().max()),
                   "rel_fro": float((a.double() - r).norm() / r.norm())}
            for name, a, r in zip(NAMES, got, ref)}
    return out


# ----------------------------------------------------------------------------- 3. repeat
def repeat_case(impl, n=5):
    gen = torch.Generator(device=dev).manual_seed(3)
    q, k, v, do = inputs(gen, 4, 1024, 8, 2, 128, impl)
    ref = impl.grads(q, k, v, do)
    fails = []
    for i in range(1, n):
        fails += compare(ref, impl.grads(q, k, v, do), f"run {i}")
    return {"checked": n - 1, "passed": not fails, "failures": fails[:20]}


# ----------------------------------------------------------------------------- 4. batch
def batch_case(impl, T=256, sizes=(2, 3, 8, 16), positions=("first", "last")):
    if QUICK:
        sizes = (2, 3, 8)
    gen = torch.Generator(device=dev).manual_seed(4)
    probe = inputs(gen, 1, T, 8, 2, 128, impl)
    ref = impl.grads(*probe)
    fails, checked = [], 0
    for size in sizes:
        for pos in positions:
            p = 0 if pos == "first" else size - 1
            others = inputs(gen, size - 1, T, 8, 2, 128, impl)
            batch = [torch.cat([o[:p], x, o[p:]]) for o, x in zip(others, probe)]
            got = [t[p:p + 1] for t in impl.grads(*batch)]
            checked += 1
            fails += compare(ref, got, f"batch {size} {pos}")
    return {"checked": checked, "passed": not fails, "failing": sorted({f["case"] for f in fails}),
            "failures": fails[:20]}


# ----------------------------------------------------------------------------- 5. padded
def padded_case(impl, L=200, layouts=((256, 4, 0), (256, 4, 3), (320, 3, 1))):
    """A length-L sequence alone (T=L) vs right-padded to T inside a batch of n at
    position p. Padding holds random q/k/v and zero output-gradient, as a loss mask
    gives. Causality keeps padding out of the real tokens' forward."""
    gen = torch.Generator(device=dev).manual_seed(5)
    probe = inputs(gen, 1, L, 8, 2, 128, impl)
    ref = impl.grads(*probe)
    fails = []
    for T, n, p in layouts:
        pad = inputs(gen, 1, T - L, 8, 2, 128, impl)
        padded = [torch.cat([x, y], 2) for x, y in zip(probe, pad)]
        padded[3][:, :, L:] = 0                       # loss mask: no gradient from padding
        others = inputs(gen, n - 1, T, 8, 2, 128, impl)
        batch = [torch.cat([o[:p], x, o[p:]]) for o, x in zip(others, padded)]
        got = [t[p:p + 1, :, :L] for t in impl.grads(*batch)]
        fails += compare(ref, got, f"T={T} batch {n} pos {p}")
    return {"checked": len(layouts), "passed": not fails, "failing": sorted({f["case"] for f in fails}),
            "failures": fails[:20]}


# ----------------------------------------------------------------------------- 6. timing
def timed(fn, iters=20):
    if QUICK:
        iters = 5
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def timing_case(B=4, T=2048, Hq=8, Hk=2, D=128):
    gen = torch.Generator(device=dev).manual_seed(6)
    q, k, v, do = rand(gen, B, Hq, T, D), rand(gen, B, Hk, T, D), rand(gen, B, Hk, T, D), rand(gen, B, Hq, T, D)
    g = Hq // Hk

    def gqa_sdpa(q, k, v):
        return sdpa(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True)
    impls = [Impl("fixed_blocks", attention_fixed_blocks_train, True)] + [
        Impl(i.name, gqa_sdpa, True, i.backend, i.det) for i in IMPLS[1:]]
    out = {}
    for impl in impls:
        ctx = sdpa_kernel([impl.backend]) if impl.backend is not None else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            fwd = timed(lambda: impl.fn(q, k, v))
        try:
            out[impl.name] = {"fwd_ms": round(fwd, 3), "fwd_bwd_ms": round(timed(lambda: impl.grads(q, k, v, do)), 3)}
        except RuntimeError as e:
            out[impl.name] = {"fwd_ms": round(fwd, 3), "error": str(e).split("\n")[0][:300]}
    return out


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true", help="batch sizes 2, 3, 8 and fewer timing iterations")
    ap.add_argument("--bi-json", help="also write the checks as bi-results-v1 (tools/bi_report.py)")
    args = ap.parse_args()
    QUICK = args.quick
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__}
    import triton
    res["triton"] = triton.__version__

    res["forward"] = forward_case()
    print("forward:", res["forward"])
    res["accuracy"] = accuracy_case()
    for name, r in res["accuracy"].items():
        print(f"accuracy {name}: " + ", ".join(f"{t} max {e['max_abs']:.2e} rel {e['rel_fro']:.2e}" for t, e in r.items()))

    for key, case in (("repeat", repeat_case), ("batch", batch_case), ("padded", padded_case)):
        res[key] = {}
        for impl in IMPLS:
            try:
                r = case(impl)
            except RuntimeError as e:
                res[key][impl.name] = {"error": str(e).split("\n")[0][:300]}
                print(f"{key} {impl.name}: ERROR {res[key][impl.name]['error']}")
                continue
            res[key][impl.name] = r
            first = r["failures"][0] if r["failures"] else None
            detail = "" if r["passed"] else f" e.g. {first['case']} {first['tensor']} {first['mismatched_elements']}/{first['of']}"
            extra = f" failing {r['failing']}" if r.get("failing") else ""
            print(f"{key} {impl.name}: {'PASS' if r['passed'] else 'FAIL'} ({r['checked']}){extra}{detail}")

    res["warnings"] = {i.name: sorted(i.warnings) for i in IMPLS if i.warnings}
    for name, w in res["warnings"].items():
        print(f"warnings {name}: {w}")
    res["timing"] = timing_case()
    for name, t in res["timing"].items():
        print(f"timing {name}: {t}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    if args.bi_json:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "suite"))
        from adapters import emit
        sys.exit(emit("attention_backward", res, args.bi_json, quick=QUICK))


if __name__ == "__main__":
    main()
