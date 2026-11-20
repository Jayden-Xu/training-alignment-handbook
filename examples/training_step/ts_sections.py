"""Sections 5, 6 and 8: loss reduction layouts, GradScaler exactness, gradient clipping."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ts_kernels import divide_by_count, tree_mean_loss, tree_mean_loss_packed
from ts_util import DEV, bit_equal, first_mismatch, repeat_l1, timed_ms, ulp_stats


def _lengths(g, B, Tmax):
    return (torch.randint(Tmax // 4, Tmax + 1, (B,), generator=g)).tolist()


def sec_loss(quick: bool) -> dict:
    """The same per-token losses averaged from several layouts."""
    g = torch.Generator().manual_seed(5)
    out = {"cases": []}
    for B, Tmax in ((8, 1024), (16, 512)) + (() if quick else ((32, 2048), (3, 100))):
        lens = _lengths(g, B, Tmax)
        vals = [torch.rand(n, generator=g).mul(10).to(DEV) for n in lens]   # per-token losses
        packed = torch.cat(vals)
        padded = torch.zeros(B, Tmax, device=DEV)
        mask = torch.zeros(B, Tmax, dtype=torch.bool, device=DEV)
        for i, v in enumerate(vals):
            padded[i, :len(v)] = v
            mask[i, :len(v)] = True
        pad2 = torch.zeros(B, 2 * Tmax, device=DEV)                       # padded further
        mask2 = torch.zeros(B, 2 * Tmax, dtype=torch.bool, device=DEV)
        pad2[:, :Tmax], mask2[:, :Tmax] = padded, mask
        n = packed.numel()
        stock = {
            "packed .mean()": packed.mean(),
            "padded sum/count": padded.sum() / n,
            "padded more sum/count": pad2.sum() / n,
            "padded [B*T] flat sum/count": padded.reshape(-1).sum() / n,
            "masked_select .mean()": padded[mask].mean(),
            "per-sequence sums, then sum/count": torch.stack([v.sum() for v in vals]).sum() / n,
            "two half batches, summed": (torch.cat(vals[:B // 2]).sum() + torch.cat(vals[B // 2:]).sum()) / n,
        }
        ref = stock["packed .mean()"]
        stock_res = {k: bit_equal(ref, v) for k, v in stock.items()}
        fixed = {
            "padded": tree_mean_loss(padded, mask),
            "padded more": tree_mean_loss(pad2, mask2),
            "packed": tree_mean_loss_packed(packed, lens),
        }
        fref = fixed["padded"]
        fixed_res = {k: bit_equal(fref, v) for k, v in fixed.items()}
        # different B: two halves reduced as their own batches, then the two halves' totals
        # combined by the same tree. Equal to the whole batch when B/2 is a power of two.
        if B % 2 == 0 and (B // 2) & (B // 2 - 1) == 0:
            from ts_kernels import tree_sum
            h = B // 2
            t1 = tree_sum(tree_sum(torch.where(mask[:h], padded[:h], 0.0)))
            t2 = tree_sum(tree_sum(torch.where(mask[h:], padded[h:], 0.0)))
            fixed_res["two half batches (B/2 a power of two)"] = bit_equal(fref, divide_by_count(t1 + t2, n))
        # cross_entropy(mean, ignore_index) with padding vs packed: real loss op
        V = 1000
        logits = torch.randn(B, Tmax, V, generator=torch.Generator().manual_seed(B)).to(DEV)
        y = torch.randint(0, V, (B, Tmax), generator=torch.Generator().manual_seed(B + 1)).to(DEV)
        yp = torch.where(mask, y, -100)
        ce_pad = F.cross_entropy(logits.view(-1, V), yp.view(-1))
        ce_pack = F.cross_entropy(logits[mask], y[mask])
        per = F.cross_entropy(logits.view(-1, V), yp.view(-1), reduction="none").view(B, Tmax)
        per_pack = F.cross_entropy(logits[mask], y[mask], reduction="none")
        ce = {"cross_entropy mean: padded+ignore_index vs packed": bit_equal(ce_pad, ce_pack),
              "per-token losses: padded vs packed": bit_equal(per[mask], per_pack),
              "tree mean of per-token losses: padded vs packed":
                  bit_equal(tree_mean_loss(per, mask), tree_mean_loss_packed(per_pack, lens))}
        tot = torch.where(mask, padded, 0.0).sum()
        div = {"tensor / python int equals tensor / fp32 tensor": bit_equal(tot / n, divide_by_count(tot, n))}
        out["cases"].append({"division": div, "B": B, "Tmax": Tmax, "tokens": n, "stock_equal_to_packed_mean": stock_res,
                             "fixed_equal_to_padded": fixed_res, "cross_entropy": ce})
        print(f"  loss B={B} T<={Tmax}: stock {stock_res}\n    fixed {fixed_res}\n    ce {ce}", flush=True)
    x = torch.rand(64, 4096, device=DEV)
    out["ms"] = {"mean": round(timed_ms(lambda: x.mean()), 4), "tree_mean_loss": round(timed_ms(lambda: tree_mean_loss(x)), 4)}
    return out


def _grads_once(tr, x, y, scale):
    tr.opt.zero_grad(set_to_none=True)
    loss = tr.forward_loss(x, y)
    (loss * scale if scale != 1 else loss).backward()
    gr = {}
    for n, p in tr.model.backward_order():
        gr[n] = p.grad.clone() if scale == 1 else p.grad * (1.0 / scale)    # power of two: exact
    return gr


def sec_amp(quick: bool) -> dict:
    """GradScaler: scale and unscale are multiplications by powers of two. Are the unscaled
    gradients the bits of an unscaled backward? And does the inf/nan skip leave state intact?"""
    import train_step  # noqa: F401  (path setup)
    from ts_model import Cfg, TrainCfg, Trainer
    out = {"scale_vs_none": [], "skip": {}}
    for amp in ("bf16", "fp16"):
        attn = "fixed" if amp == "bf16" else "sdpa"   # the fixed kernel is bf16 only
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel([SDPBackend.MATH])             # fp16: SDPA math backend (plain matmuls)
        ctx.__enter__()
        cfg = Cfg(n_layers=2, attn=attn, emb="fixed", loss="fixed")
        tr = Trainer(cfg, TrainCfg(batch=2, seq=512, amp=amp))
        x, y = tr.sampler.next()
        x, y = x.to(DEV), y.to(DEV)
        ref = _grads_once(tr, x, y, 1)
        for e in (0, 4, 16, 24) + (() if quick else (8, 32, 40)):  # 2^0: the reference repeats
            got = _grads_once(tr, x, y, float(2 ** e))
            diff = {n: ulp_stats(ref[n], got[n]) for n in ref}
            bad = {n: d for n, d in diff.items() if d["mismatched"]}
            fin = all(torch.isfinite(t).all() for t in got.values())
            zeros = sum(int((ref[n] == 0).sum()) for n in ref), sum(int((got[n] == 0).sum()) for n in ref)
            out["scale_vs_none"].append({"amp": amp, "attn": attn, "scale": f"2^{e}", "equal": not bad, "finite": bool(fin),
                                         "tensors_differ": len(bad), "elements_differ": sum(d["mismatched"] for d in bad.values()),
                                         "zero_grads_unscaled_vs_scaled": list(zeros)})
            print(f"  amp {amp} scale 2^{e}: equal {not bad} finite {fin} tensors differ {len(bad)} "
                  f"elements {sum(d['mismatched'] for d in bad.values())} zeros {zeros}", flush=True)
        ctx.__exit__(None, None, None)
    # inf skip path: one gradient element set to inf before scaler.step
    cfg = Cfg(n_layers=2, attn="fixed", emb="fixed", loss="fixed")
    for opt in ("single", "foreach", "fused"):
        tr = Trainer(cfg, TrainCfg(batch=2, seq=512, scaler=True, opt=opt))
        tr.step(record=False)                                # creates optimizer state
        before = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
        st_before = {k: (v.clone() if torch.is_tensor(v) else v) for p in tr.model.parameters()
                     for k, v in tr.opt.state[p].items() if k == "step"}
        m_before = [tr.opt.state[p]["exp_avg"].clone() for p in tr.model.parameters()]
        scale0 = tr.scaler.get_scale()
        tr.backward()
        next(tr.model.parameters()).grad.view(-1)[0] = float("inf")
        tr.scaler.unscale_(tr.opt)
        tr.scaler.step(tr.opt)
        tr.scaler.update()
        same_p = all(bit_equal(before[n], p.detach()) for n, p in tr.model.named_parameters())
        same_m = all(bit_equal(a, tr.opt.state[p]["exp_avg"]) for a, p in zip(m_before, tr.model.parameters()))
        steps = sorted({float(tr.opt.state[p]["step"]) for p in tr.model.parameters()})
        out["skip"][opt] = {"params_unchanged": same_p, "exp_avg_unchanged": same_m, "step_counts": steps,
                            "scale_before": scale0, "scale_after": tr.scaler.get_scale()}
        print(f"  skip path {opt}: {out['skip'][opt]}", flush=True)
    return out


def sec_clip(quick: bool) -> dict:
    """clip_grad_norm_ foreach against per-tensor, and run to run."""
    from ts_model import LM, Cfg, clip_fixed_
    torch.manual_seed(3)
    model = LM(Cfg()).to(DEV)
    params = list(model.parameters())
    g0 = [torch.randn_like(p) * 0.05 for p in params]

    def run(path, max_norm=1.0):
        gs = [g.clone() for g in g0]
        for p, g in zip(params, gs):
            p.grad = g
        if path == "fixed":
            total = clip_fixed_(gs, max_norm)
        else:
            total = torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=path == "foreach")
        return {"norm": total.detach().clone(), **{f"g{i}": p.grad.clone() for i, p in enumerate(params)}}
    out = {"parameters": sum(p.numel() for p in params), "tensors": len(params)}
    runs = 5 if quick else 20
    for path in ("foreach", "single", "fixed"):
        out[path] = repeat_l1(lambda: run(path), runs)
        out[path].pop("tensors")
        out[path]["ms"] = round(timed_ms(lambda: run(path)), 3)
    a, b = run("foreach"), run("single")
    out["foreach_vs_single"] = {"norm_equal": bit_equal(a["norm"], b["norm"]),
                                "grads_equal": all(bit_equal(a[k], b[k]) for k in a if k != "norm"),
                                "norm": [float(a["norm"]), float(b["norm"])]}
    out["fixed_vs_foreach_norm_equal"] = bit_equal(run("fixed")["norm"], a["norm"])
    print(f"  clip: {out}", flush=True)
    return out
