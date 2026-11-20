"""Section 4: optimizer paths. Same gradients and state through fused, foreach and
single-tensor AdamW and SGD; then a model of each path's rounding order, checked bit for bit.

The models are written from the torch 2.5.1 sources:
  torch/optim/adamw.py, torch/optim/sgd.py           (single-tensor and foreach paths)
  aten/src/ATen/native/cuda/fused_adam_utils.cuh     (fused AdamW)
  aten/src/ATen/native/cuda/FusedSgdKernel.cu        (fused SGD)
A model is only reported as the path's order when it reproduces every bit of the parameter
and of the optimizer state at every step.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from ts_util import DEV, bit_equal, timed_ms, ulp_stats

F32 = torch.float32
ADAMW = dict(lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
SGD = dict(lr=0.1, momentum=0.9, weight_decay=1e-4)
PATHS = {"fused": dict(fused=True), "foreach": dict(foreach=True, fused=False),
         "single": dict(foreach=False, fused=False)}


def shapes(quick: bool):
    s = [(1024, 1024), (1023, 1023), (513, 300), (4096,), (7,), (1,)]
    return s if quick else s + [(2048, 1024)]


def grad_for(step: int, shape, k: int) -> torch.Tensor:
    """The same pseudo-random gradient for every path: it depends on the step and the tensor
    only, never on the parameters. Mixed magnitudes, and some exact zeros."""
    g = torch.Generator(device=DEV).manual_seed(1000 * step + k)
    x = torch.randn(shape, generator=g, device=DEV) * 10.0 ** torch.randint(-5, 1, shape, generator=g, device=DEV)
    x[torch.rand(shape, generator=g, device=DEV) < 0.02] = 0.0
    return x


def init_params(quick: bool):
    g = torch.Generator(device=DEV).manual_seed(7)
    return [torch.randn(s, generator=g, device=DEV) * 0.05 for s in shapes(quick)]


def make_opt(kind: str, path: str, params):
    if kind == "adamw":
        return torch.optim.AdamW(params, **ADAMW, **PATHS[path])
    return torch.optim.SGD(params, **SGD, **PATHS[path])


def state_of(kind: str, opt, params):
    if kind == "adamw":
        return {"exp_avg": [opt.state[p]["exp_avg"].clone() for p in params],
                "exp_avg_sq": [opt.state[p]["exp_avg_sq"].clone() for p in params]}
    return {"momentum_buffer": [opt.state[p]["momentum_buffer"].clone() for p in params]}


def run_path(kind: str, path: str, quick: bool, steps: int, keep):
    """Returns {step: (params, state)} for the steps in `keep` (1-based)."""
    params = [p.clone().requires_grad_(True) for p in init_params(quick)]
    opt = make_opt(kind, path, params)
    snaps = {}
    for s in range(1, steps + 1):
        for k, p in enumerate(params):
            p.grad = grad_for(s, p.shape, k)
        opt.step()
        if s in keep:
            snaps[s] = ([p.detach().clone() for p in params], state_of(kind, opt, params))
    return snaps


def compare(a, b) -> dict:
    """Element counts that differ between two snapshots, parameters and state."""
    pa, sa = a
    pb, sb = b
    out = {"param": _acc([ulp_stats(x, y) for x, y in zip(pa, pb)])}
    for key in sa:
        out[key] = _acc([ulp_stats(x, y) for x, y in zip(sa[key], sb[key])])
    out["equal"] = all(v["mismatched"] == 0 for v in out.values())
    return out


def _acc(stats) -> dict:
    return {"mismatched": sum(s["mismatched"] for s in stats), "of": sum(s["of"] for s in stats),
            "max_ulp": max((s.get("max_ulp", 0) for s in stats), default=0)}


# ----------------------------------------------------------------------------- order models
def f32(x: float) -> torch.Tensor:
    """A Python double rounded to fp32, as a scalar operand of an fp32 kernel."""
    return torch.tensor(np.float32(x), device=DEV)


def fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fl32(a*b + c) with one rounding (a fused multiply-add). The product of two fp32 values
    is exact in fp64; the fp64 sum is corrected to round-to-odd before the final rounding, so
    there is no double rounding."""
    prod = a.double() * b.double()
    cd = c.double().expand_as(prod)
    s = prod + cd
    bb = s - prod
    err = (prod - (s - bb)) + (cd - bb)                 # TwoSum: prod + cd == s + err exactly
    need = (err != 0) & ((s.view(torch.int64) & 1) == 0)
    inf = torch.full_like(s, float("inf"))
    s = torch.where(need, torch.where(err > 0, torch.nextafter(s, inf), torch.nextafter(s, -inf)), s)
    return s.float()


def model_adamw_fused(p, g, m, v, t, lr, b1, b2, eps, wd):
    """fused_adam_utils.cuh, ADAMW mode. lr, betas, eps and weight decay are doubles, the
    tensors are read as fp32, so most lines run in fp64 and round to fp32 once."""
    pd, gd = p.double(), g.double()
    p1 = (pd - (lr * wd) * pd).float()                  # param -= lr * weight_decay * param
    m1 = (b1 * m.double() + (1 - b1) * gd).float()      # exp_avg = beta1*exp_avg + (1-beta1)*grad
    v1 = (b2 * v.double() + ((1 - b2) * gd) * gd).float()   # ... + (1-beta2)*grad*grad
    bc1 = float(np.float32(1 - b1 ** t))                # bias corrections are passed as fp32
    bc2s = f32(math.sqrt(1 - b2 ** t))
    step_size = f32(lr / bc1)                           # fp64 division, stored as fp32
    denom = ((v1.sqrt() / bc2s).double() + eps).float()  # fp32 sqrt and division, fp64 add of eps
    p2 = p1 - (step_size * m1) / denom                  # fp32: (step_size*exp_avg)/denom
    return p2, m1, v1


def model_adamw_unfused(variant):
    """torch/optim/adamw.py: mul_, lerp_, mul_, addcmul_, sqrt, div, add_, addcdiv_ as separate
    fp32 kernels. `variant` says, for each compound kernel, whether its multiply-add is fused
    (one rounding) and whether the division by a scalar is a multiply by the reciprocal."""
    lerp_fma, addcmul_fma, div_recip, addcdiv_fma = variant

    def step(p, g, m, v, t, lr, b1, b2, eps, wd):
        p1 = p * f32(1 - lr * wd)                       # param.mul_(1 - lr*weight_decay)
        w = f32(1 - b1)                                 # exp_avg.lerp_(grad, 1 - beta1)
        m1 = fma32(w, g - m, m) if lerp_fma else m + w * (g - m)
        v0 = v * f32(b2)                                # exp_avg_sq.mul_(beta2)
        a = f32(1 - b2)                                 # .addcmul_(grad, grad, value=1 - beta2)
        v1 = fma32(a, g * g, v0) if addcmul_fma else v0 + a * (g * g)
        bc2s = f32(math.sqrt(1 - b2 ** t))              # (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
        root = v1.sqrt()
        denom = (root * (f32(1.0) / bc2s) if div_recip else root / bc2s) + f32(eps)
        neg_step = f32(-(lr / (1 - b1 ** t)))           # param.addcdiv_(exp_avg, denom, value=-step_size)
        q = m1 / denom
        p2 = fma32(neg_step, q, p1) if addcdiv_fma else p1 + neg_step * q
        return p2, m1, v1
    return step


ADAMW_VARIANTS = list(itertools.product((False, True), repeat=4))
VARIANT_NAMES = ("lerp fused multiply-add", "addcmul fused multiply-add",
                 "division by a scalar as multiply by reciprocal", "addcdiv fused multiply-add")


def model_sgd_fused(p, g, buf, first, lr, mom, wd):
    """FusedSgdKernel.cu: weight decay, momentum and the update run in fp64 (the hyperparameters
    are doubles) and round to fp32 at each assignment."""
    g1 = (g.double() + wd * p.double()).float()         # g += weight_decay * p
    bd = g1.double() if first else mom * buf.double() + (1 - 0.0) * g1.double()
    buf1 = bd.float()                                   # stored momentum buffer
    g2 = bd.float()                                     # g = momentum_buffer
    p1 = (p.double() - lr * g2.double()).float()        # p -= lr * g
    return p1, buf1


def model_sgd_unfused(variant):
    """torch/optim/sgd.py: grad.add(param, alpha=wd); buf.mul_(momentum).add_(grad, alpha=1);
    param.add_(buf, alpha=-lr), as separate fp32 kernels."""
    wd_fma, upd_fma = variant

    def step(p, g, buf, first, lr, mom, wd):
        g1 = fma32(f32(wd), p, g) if wd_fma else g + f32(wd) * p
        buf1 = g1.clone() if first else buf * f32(mom) + g1
        p1 = fma32(f32(-lr), buf1, p) if upd_fma else p + f32(-lr) * buf1
        return p1, buf1
    return step


SGD_VARIANTS = list(itertools.product((False, True), repeat=2))
SGD_VARIANT_NAMES = ("weight-decay add fused multiply-add", "update add fused multiply-add")


def check_models(kind: str, path: str, quick: bool, steps: int) -> dict:
    """Step the real optimizer; at every step feed its state before the step to each model and
    compare the model's output with the optimizer's state after the step."""
    params = [p.clone().requires_grad_(True) for p in init_params(quick)]
    opt = make_opt(kind, path, params)
    if kind == "adamw":
        models = ({"fused kernel order": model_adamw_fused} |
                  {str(v): model_adamw_unfused(v) for v in ADAMW_VARIANTS})
    else:
        models = ({"fused kernel order": model_sgd_fused} | {str(v): model_sgd_unfused(v) for v in SGD_VARIANTS})
    bad = {k: 0 for k in models}
    total = 0
    for s in range(1, steps + 1):
        grads = [grad_for(s, p.shape, k) for k, p in enumerate(params)]
        before = [p.detach().clone() for p in params]
        if kind == "adamw":
            st = [(opt.state[p]["exp_avg"].clone(), opt.state[p]["exp_avg_sq"].clone()) if p in opt.state and opt.state[p]
                  else (torch.zeros_like(p), torch.zeros_like(p)) for p in params]
        else:
            st = [opt.state[p]["momentum_buffer"].clone() if p in opt.state and opt.state[p].get("momentum_buffer") is not None
                  else None for p in params]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()
        for i, p in enumerate(params):
            total += p.numel()
            for name, fn in models.items():
                if bad[name] > 0 and s > 3:
                    continue                             # already ruled out
                if kind == "adamw":
                    h = ADAMW
                    pm, mm, vm = fn(before[i], grads[i], st[i][0], st[i][1], s, h["lr"], h["betas"][0],
                                    h["betas"][1], h["eps"], h["weight_decay"])
                    n = (ulp_stats(pm, p.detach())["mismatched"] + ulp_stats(mm, opt.state[p]["exp_avg"])["mismatched"]
                         + ulp_stats(vm, opt.state[p]["exp_avg_sq"])["mismatched"])
                else:
                    h = SGD
                    first = st[i] is None
                    pm, bm = fn(before[i], grads[i], st[i] if not first else torch.zeros_like(p), first,
                                h["lr"], h["momentum"], h["weight_decay"])
                    n = (ulp_stats(pm, p.detach())["mismatched"]
                         + ulp_stats(bm, opt.state[p]["momentum_buffer"])["mismatched"])
                bad[name] += n
    exact = [k for k, v in bad.items() if v == 0]
    return {"steps": steps, "elements_per_step": total // steps, "exact_models": exact,
            "mismatches_first_steps": {k: v for k, v in bad.items() if v}}


def describe(kind: str, name: str) -> str:
    if name == "fused kernel order":
        return name
    v = eval(name)
    names = VARIANT_NAMES if kind == "adamw" else SGD_VARIANT_NAMES
    on = [n for n, f in zip(names, v) if f]
    return "separate fp32 kernels; " + ("; ".join(on) if on else "no fused multiply-add, true division")


def sec_optim(quick: bool) -> dict:
    out = {"hyper": {"adamw": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ADAMW.items()}, "sgd": SGD},
           "elements": sum(math.prod(s) for s in shapes(quick)), "shapes": [list(s) for s in shapes(quick)]}
    steps, keep = 100, (1, 2, 10, 100)
    for kind in ("adamw", "sgd"):
        snaps = {path: run_path(kind, path, quick, steps, keep) for path in PATHS}
        again = {path: run_path(kind, path, quick, steps, (steps,)) for path in PATHS}
        res = {"pairs": {}, "repeat": {}, "models": {}, "ms_per_step": {}}
        for a, b in (("fused", "foreach"), ("fused", "single"), ("foreach", "single")):
            res["pairs"][f"{a} vs {b}"] = {str(s): compare(snaps[a][s], snaps[b][s]) for s in keep}
        for path in PATHS:
            res["repeat"][path] = compare(snaps[path][steps], again[path][steps])["equal"]
            m = check_models(kind, path, quick, 20 if quick else 100)
            m["exact_models_described"] = [describe(kind, k) for k in m["exact_models"]]
            res["models"][path] = m
            print(f"  {kind} {path}: repeats {res['repeat'][path]}; order models that reproduce every bit: "
                  f"{m['exact_models_described']}", flush=True)
        for pair, by in res["pairs"].items():
            print(f"  {kind} {pair}: " + ", ".join(
                f"step {s}: param {d['param']['mismatched']}/{d['param']['of']} (max {d['param']['max_ulp']} ulp)"
                for s, d in by.items()), flush=True)
        out[kind] = res
    # cost on an LM-sized parameter set (the case's model: 53M parameters in 40 tensors)
    from ts_model import LM, Cfg
    torch.manual_seed(0)
    model = LM(Cfg()).to(DEV)
    plist = list(model.parameters())
    for p in plist:
        p.grad = torch.randn_like(p) * 1e-3
    for kind in ("adamw", "sgd"):
        for path in PATHS:
            opt = make_opt(kind, path, plist)
            out[kind]["ms_per_step"][path] = round(timed_ms(opt.step, warm=3, iters=15), 3)
        print(f"  {kind} step time, {sum(p.numel() for p in plist) / 1e6:.1f}M parameters: {out[kind]['ms_per_step']}", flush=True)
    out["cost_parameters"] = sum(p.numel() for p in plist)
    return out
