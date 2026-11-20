"""Cross-engine (L4) check: model.py as the trainer against vLLM generating with temperature 0.

    python cross_engine.py --quick --json q.json              one engine config, 8 prompts (~3 min, 1 A16)
    python cross_engine.py --json results-A16.json            every config, 64 prompts, batch 1..64
    python cross_engine.py --trainer-only --workdir W ...     reuse engine records (e.g. from torch 2.5.1)

Steps (each engine config runs in its own process, engine_run.py):
 1. engine self-consistency: L1 (largest batch twice) and L2 (each prompt alone vs in batches 4/16/64),
    tokens and realized-token logprobs, bitwise;
 2. trainer vs engine (L4): teacher-force every engine sequence through the trainer, compare the
    logprob of every realized token (prompt and generated) bitwise; max |dlogp|; KL from full logits;
 3. op-by-op replay (eager configs): feed each captured engine op input to the trainer's choices and
    see which one reproduces the engine's output bits; then move the trainer's rounding points to
    vLLM's one at a time and count the tokens that still differ;
 4. throughput of each engine config at batch 1/8/64.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import subprocess
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import trainer as tr                                                # noqa: E402
from bi_report import Results                                       # noqa: E402

MODEL_DIR = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")

# name -> engine_run.py flags. Prefix caching is off everywhere (engine_run.py).
ENGINES = {
    "default": [],
    "eager": ["--eager"],
    "bi": ["--bi", "--backend", "FLASH_ATTN"],                  # BI mode refuses to start without a backend
    "bi_eager": ["--bi", "--eager", "--backend", "FLASH_ATTN"],
    "bi_eager_lmhead": ["--bi", "--eager", "--backend", "FLASH_ATTN", "--patch-lmhead"],
    "triton": ["--backend", "TRITON_ATTN"],
    "flashinfer": ["--backend", "FLASHINFER"],
    "flex": ["--backend", "FLEX_ATTENTION"],
    "bi_triton_eager": ["--bi", "--eager", "--backend", "TRITON_ATTN"],
}
# vLLM 0.6.3 (sglang env). Its V0 engine computes fp32 logits for every prompt row of a step at once
# (prompt logprobs): cap the tokens per step so batch 64 fits on a 15 GB A16.
_OLD = ["--gpu-mem", "0.3", "--max-batched-tokens", "2048"]
OLD_ENGINES = {"v063_default": _OLD, "v063_eager": ["--eager"] + _OLD}
# What the handbook claims at L2 on sm86 (A16): BI eager with the lm_head on vLLM's Triton GEMM holds;
# without that patch the cuBLAS lm_head breaks L2 (documented variance); default and eager fail.
L2_EXPECT = {"bi_eager_lmhead": "pass", "bi_eager": "fail", "bi": "fail", "default": "fail", "eager": "fail"}
TPUT = ("default", "eager", "bi", "bi_eager", "bi_eager_lmhead")
CAPTURE = ("eager", "bi_eager", "bi_eager_lmhead")

# The trainer's rounding points per vLLM config, read from vLLM 0.19.1's source and confirmed by replay.
TARGETS = {
    "eager": dict(norm="vllm_cuda", gemm="cublas_packed", rope_table="vllm", rope="vllm_cuda",
                  attn="fa2_split0", act="vllm_cuda", lm_head="cublas", logsoftmax="torch"),
    "bi_eager": dict(norm="vllm_bi", gemm="vllm_bi", bias="after_cast", rope_table="vllm", rope="vllm_cuda",
                     attn="fa2_split1", act="vllm_cuda", lm_head="cublas", logsoftmax="vllm_bi"),
    "bi_eager_lmhead": dict(norm="vllm_bi", gemm="vllm_bi", bias="after_cast", rope_table="vllm", rope="vllm_cuda",
                            attn="fa2_split1", act="vllm_cuda", lm_head="vllm_bi", logsoftmax="vllm_bi"),
}
# One rounding point per step, cumulative. For the batch-invariant targets the packing, the bias rounding
# point and the GEMM kernel are separate steps.
STEPS = {
    "eager": [("gemm", "cublas_packed"), ("norm", "vllm_cuda"), ("rope_table", "vllm"), ("rope", "vllm_cuda"),
              ("attn", "fa2_split0"), ("act", "vllm_cuda")],
    "bi_eager": [("gemm", "cublas_packed"), ("bias", "after_cast"), ("gemm", "vllm_bi"), ("norm", "vllm_bi"),
                 ("rope_table", "vllm"), ("rope", "vllm_cuda"), ("attn", "fa2_split1"), ("act", "vllm_cuda"),
                 ("logsoftmax", "vllm_bi")],
}
STEPS["bi_eager_lmhead"] = STEPS["bi_eager"][:-1] + [("lm_head", "vllm_bi"), ("logsoftmax", "vllm_bi")]


# ----------------------------------------------------------------------------- engine records
class _Opaque:
    """Stands in for an engine object pickled into a record (e.g. a vLLM module) when vLLM is absent."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        try:
            return super().find_class(mod, name)
        except (ModuleNotFoundError, AttributeError):
            return _Opaque


_PICKLE = types.ModuleType("record_pickle")
_PICKLE.Unpickler = _Unpickler
_PICKLE.load = pickle.load
_PICKLE.__dict__["Pickler"] = pickle.Pickler


def load_record(path):
    return torch.load(path, weights_only=False, pickle_module=_PICKLE)


def run_engine(name, flags, workdir, python, quick, log):
    out = os.path.join(workdir, f"engine-{name}.pt")
    if os.path.exists(out):
        return out
    args = [python, os.path.join(HERE, "engine_run.py"), "--name", name, "--out", out] + flags
    if quick:
        args += ["--nprompts", "8", "--gen", "16", "--batches", "1,8", "--repeat", "--kl", "2"]
    else:
        args += ["--repeat", "--kl", "4"]
        if name in TPUT:
            args.append("--tput")
    if name in CAPTURE and "--eager" in flags:
        args.append("--capture")
    t = time.time()
    with open(log, "a") as f:
        rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"engine {name}: rc={rc} {time.time() - t:.0f}s", flush=True)
    return out if rc == 0 and os.path.exists(out) else None


def tokens_of(rec):
    return rec["ids"][rec["P"]:]


def lp_vec(rec):
    return torch.tensor(rec["plp"] + rec["glp"], dtype=torch.float64).float()


def compare_runs(a, b):
    """Two runs of the same prompts. Returns prompts with different tokens, and logprob positions
    differing over each prompt's common prefix (positions up to the first token divergence)."""
    tok_diff, n, nd, mx = 0, 0, 0, 0.0
    for i in a:
        if i not in b:
            continue
        ra, rb = a[i], b[i]
        ta, tb = ra["ids"], rb["ids"]
        common = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        if ta != tb:
            tok_diff += 1
        m = common - 1                                              # logprobs of tokens 1..common-1
        va, vb = lp_vec(ra)[:m], lp_vec(rb)[:m]
        n += m
        nd += int((va.view(torch.int32) != vb.view(torch.int32)).sum())
        if m:
            mx = max(mx, float((va - vb).abs().max()))
    return {"prompts": len(a), "token_seq_differ": tok_diff, "logprobs": n, "logprob_differ": nd,
            "max_abs_dlogp": mx}


# ----------------------------------------------------------------------------- trainer side
class Trainer:
    def __init__(self, model):
        self.model, self.cache = model, {}

    def logits(self, ids, variant):
        if variant in ("model.py stock", "model.py bi"):
            ops = variant.split()[-1]
            return self.model.prefill(ids[None].cuda(), ops=ops)[0], "torch"
        ch = dict(variant)
        return tr.forward_matched(self.model, ids, ch), ch.get("logsoftmax", "torch")

    def realized(self, ids_list, variant):
        key = (tuple(ids_list), variant if isinstance(variant, str) else tuple(sorted(variant)))
        if key not in self.cache:
            ids = torch.tensor(ids_list)
            with torch.no_grad():
                lg, how = self.logits(ids, variant)
                lp = tr.logprobs(lg, how)
                self.cache[key] = lp[:-1].gather(-1, ids[1:, None].cuda())[:, 0].cpu()
        return self.cache[key]


def vkey(ch):
    return tuple(sorted({**tr.REFERENCE, **ch}.items()))


def l4_compare(trainer, runs, variant, run_names=None):
    n = nd = 0
    mx = 0.0
    seqs = 0
    n1 = nd1 = 0
    for rn, run in runs.items():
        if run_names and rn not in run_names:
            continue
        for i, rec in run.items():
            e = lp_vec(rec)
            t = trainer.realized(rec["ids"], variant)
            n += e.numel()
            nd += int((e.view(torch.int32) != t.view(torch.int32)).sum())
            mx = max(mx, float((e - t).abs().max()))
            seqs += 1
            if rn == "B1":
                n1 += e.numel()
                nd1 += int((e.view(torch.int32) != t.view(torch.int32)).sum())
    return {"sequences": seqs, "tokens": n, "differ": nd, "frac_differ": nd / max(n, 1), "max_abs_dlogp": mx,
            "tokens_B1": n1, "differ_B1": nd1}


def kl_capture_ok(kl, how="torch"):
    """The captured logits reproduce the engine's reported logprobs bit for bit (GPU log_softmax of fp32;
    vLLM's Triton log_softmax in batch-invariant mode)."""
    for r in kl.values():
        G = r["ids"][r["P"]:]
        lp = tr.logprobs(r["logits"].cuda(), how)
        mine = lp[torch.arange(len(G)), torch.tensor(G).cuda()].cpu()
        if not mine.equal(torch.tensor(r["glp"], dtype=torch.float64).float()):
            return False
    return True


def kl_compare(trainer, kl, variant):
    kls, eq_logit_rows, eq_lp, n = [], 0, 0, 0
    for i, r in kl.items():
        ids = torch.tensor(r["ids"])
        with torch.no_grad():
            lg, how = trainer.logits(ids, variant)
            P = r["P"]
            G = r["logits"].shape[0]
            rows = lg[P - 1:P - 1 + G]
            le = r["logits"].cuda()
            eq_logit_rows += int((rows.to(le.dtype).view(torch.int16) == le.view(torch.int16)).all(-1).sum()) \
                if rows.dtype == le.dtype else 0
            pe = torch.log_softmax(le.double(), -1)
            pt = torch.log_softmax(rows.double(), -1)
            kls.append((pe.exp() * (pe - pt)).sum(-1).cpu())
            n += G
    k = torch.cat(kls)
    return {"positions": n, "mean_kl": float(k.mean()), "max_kl": float(k.max()),
            "logit_rows_bit_equal": eq_logit_rows}


# ----------------------------------------------------------------------------- op replay
def _t(x):
    return x.cuda() if torch.is_tensor(x) else x


def _eq(a, b):
    a, b = a.contiguous(), b.contiguous()
    if a.shape != b.shape:
        return 0.0
    iv = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return float((a.view(iv) == b.view(iv)).float().mean())


@torch.no_grad()
def op_replay(model, cap):
    """For each op class: fraction of captured calls whose output each trainer choice reproduces bit for bit
    (and the mean fraction of equal elements)."""
    cfg, calls = model.cfg, cap["calls"]
    bb, eps = model.model, cfg.rms_eps
    out = {}

    def add(op, variant, eq):
        d = out.setdefault(op, {}).setdefault(variant, {"calls": 0, "exact": 0, "elem_eq": 0.0})
        d["calls"] += 1
        d["exact"] += int(eq == 1.0)
        d["elem_eq"] += eq

    hf_tables = model.ctx.tables("cuda")
    for i, L in enumerate(bb.layers):
        P = f"layers.{i}."
        for nm, w in (("input_layernorm", L.input_layernorm.weight), ("post_attention_layernorm", L.post_attention_layernorm.weight)):
            for c in calls[P + nm]:
                xs = [_t(x) for x in c["in"]]
                res = xs[1] if len(xs) > 1 else None
                eo = c["out"][0] if isinstance(c["out"], tuple) else c["out"]
                for how in tr.CHOICES["norm"]:
                    try:
                        y, _ = tr.op_norm(xs[0], res, w, eps, how)
                        add("rmsnorm" + (" (+residual add)" if res is not None else ""), how, _eq(y.cpu(), eo))
                    except Exception:                               # noqa: BLE001
                        pass
        a, m = L.self_attn, L.mlp
        Wqkv, bqkv, Wgu = tr._packed(L)
        for c in calls[P + "self_attn.qkv_proj"]:
            x, eo = _t(c["in"][0]), c["out"][0]
            q_sz, kv_sz = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
            sep = torch.cat([tr.op_linear(x, W, b, "cublas_sep", "epilogue") for W, b in
                             ((a.q_proj.weight, a.q_proj.bias), (a.k_proj.weight, a.k_proj.bias), (a.v_proj.weight, a.v_proj.bias))], -1)
            add("qkv_proj", "cublas_sep", _eq(sep.cpu(), eo))
            add("qkv_proj", "cublas_packed", _eq(tr.op_linear(x, Wqkv, bqkv, "cublas_packed", "epilogue").cpu(), eo))
            add("qkv_proj", "cublas_packed+bias_after_cast", _eq(tr.op_linear(x, Wqkv, bqkv, "cublas_packed", "after_cast").cpu(), eo))
            add("qkv_proj", "vllm_bi", _eq(tr.op_linear(x, Wqkv, bqkv, "vllm_bi", "after_cast").cpu(), eo))
        for nm, W in (("self_attn.o_proj", a.o_proj.weight), ("mlp.down_proj", m.down_proj.weight), ("mlp.gate_up_proj", Wgu)):
            for c in calls[P + nm]:
                x, eo = _t(c["in"][0]), c["out"][0]
                op = nm.split(".")[-1]
                if op == "gate_up_proj":
                    sep = torch.cat((F_linear(x, m.gate_proj.weight), F_linear(x, m.up_proj.weight)), -1)
                    add(op, "cublas_sep", _eq(sep.cpu(), eo))
                add(op, "cublas" if op != "gate_up_proj" else "cublas_packed", _eq(F_linear(x, W).cpu(), eo))
                add(op, "vllm_bi", _eq(tr.op_linear(x, W, None, "vllm_bi", "after_cast").cpu(), eo))
        for c in calls[P + "mlp.act_fn"]:
            x, eo = _t(c["in"][0]), c["out"]
            g, u = x.chunk(2, -1)
            for how in tr.CHOICES["act"]:
                add("silu_and_mul", how, _eq(tr.op_act(g, u, how).cpu(), eo))
        # attention: prefill rows, then each decode step against a trainer prefill of the whole sequence
        ac = calls[P + "self_attn.attn"]
        qs = [c["in"][0] for c in ac]
        ks = [c["in"][1] for c in ac]
        vs = [c["in"][2] for c in ac]
        for how in tr.CHOICES["attn"]:
            for j in range(len(ac)):
                q, k, v = (torch.cat(z[: j + 1]).cuda() for z in (qs, ks, vs))
                try:
                    o = tr.op_attn(q, k, v, cfg, how)
                except Exception:                                   # noqa: BLE001
                    continue
                n_new = qs[j].shape[0]
                add("attention " + ("prefill" if j == 0 else "decode vs trainer prefill"), how,
                    _eq(o[-n_new:].cpu(), ac[j]["out"].reshape(n_new, -1)))
    # rotary: one module shared by all layers; calls in order
    for c in calls["layers.0.self_attn.rotary_emb"]:
        pos, q, k = (_t(x) for x in c["in"])
        eq_, ek = c["out"]
        for table in tr.CHOICES["rope_table"]:
            for how in tr.CHOICES["rope"]:
                rq, rk = tr.op_rope(q, k, pos, cfg, table, how, hf_tables)
                add("rope", f"table={table},op={how}", min(_eq(rq.cpu(), eq_), _eq(rk.cpu(), ek)))
    for c in calls["norm"]:
        xs = [_t(x) for x in c["in"]]
        eo = c["out"][0] if isinstance(c["out"], tuple) else c["out"]
        for how in tr.CHOICES["norm"]:
            y, _ = tr.op_norm(xs[0], xs[1], bb.norm.weight, eps, how)
            add("final norm (+residual add)", how, _eq(y.cpu(), eo))
    # lm_head: the engine's rows (M = rows it sampled) against the trainer's M = all positions
    for j, c in enumerate(calls["logits_processor"]):
        h, eo = _t(c["in"][1]), c["out"]
        for how in tr.CHOICES["lm_head"]:
            add("lm_head (engine M)", how, _eq(tr.op_lm_head(h, model.lm_head.weight, how).cpu(), eo))
        if j == 0:                                                  # prefill: hidden of every prompt row
            hn = calls["norm"][0]["out"]
            hn = (hn[0] if isinstance(hn, tuple) else hn).cuda()
            for how in tr.CHOICES["lm_head"]:
                full = tr.op_lm_head(hn, model.lm_head.weight, how)
                add("lm_head (trainer M = all prompt rows)", how, _eq(full[-1:].cpu(), eo))
    for v in out.values():
        for d in v.values():
            d["elem_eq"] = round(d["elem_eq"] / d["calls"], 6)
    return out


def F_linear(x, W):
    return torch.nn.functional.linear(x, W)


@torch.no_grad()
def layer_localize(model, cap, ch):
    """Run the trainer forward on the captured prompt and find the first engine tensor it does not match:
    per layer, the stream after attention (residual) and the MLP output."""
    calls = cap["calls"]
    ids = torch.tensor(cap["ids"][: cap["P"]])
    logits, hidden, hfin = tr.forward_matched(model, ids, ch, return_hidden=True)
    for i, (res, mlp) in enumerate(hidden):
        e_res = calls[f"layers.{i}.post_attention_layernorm"][0]["out"][1]
        e_mlp = calls[f"layers.{i}.mlp.down_proj"][0]["out"][0]
        fr, fm = _eq(res.cpu(), e_res), _eq(mlp.cpu(), e_mlp)
        if fr < 1.0:
            return {"first_layer": i, "tensor": "stream after attention", "elem_eq": fr}
        if fm < 1.0:
            return {"first_layer": i, "tensor": "mlp output", "elem_eq": fm}
    en = calls["norm"][0]["out"]
    fn = _eq(hfin.cpu(), en[0] if isinstance(en, tuple) else en)
    if fn < 1.0:
        return {"first_layer": "final", "tensor": "final norm", "elem_eq": fn}
    fl = _eq(logits[-1:].cpu(), calls["logits_processor"][0]["out"])
    if fl < 1.0:
        return {"first_layer": "final", "tensor": "logits (last prompt row)", "elem_eq": fl}
    return {"first_layer": None, "tensor": "all layers, final norm and logits equal"}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--workdir", default=os.path.join(HERE, "work"))
    ap.add_argument("--engines", default=None, help="comma list (default: all; quick: bi_eager_lmhead)")
    ap.add_argument("--old-python", default=os.path.expanduser("~/miniconda3/envs/sglang/bin/python"))
    ap.add_argument("--engines-only", action="store_true", help="run the engine processes and stop")
    ap.add_argument("--tput-dir", default=None, help="tput-<config>.pt files from tput.sh (default: workdir)")
    ap.add_argument("--save-trainer", default=None, help="save model.py stock/bi realized logprobs (B1 runs) here")
    ap.add_argument("--compare-trainer", default=None, help="a --save-trainer file from another torch version")
    ap.add_argument("--strip", default=None, help="write copies of the records without engine objects to this dir")
    ap.add_argument("--trainer-only", action="store_true", help="no engine runs; model.py stock/bi only")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    res = Results("cross_engine", quick=args.quick)
    log = os.path.join(args.workdir, "engines.log")
    names = (args.engines.split(",") if args.engines else
             ["bi_eager_lmhead"] if args.quick else list(ENGINES) + list(OLD_ENGINES))
    recs = {}
    t0 = time.time()
    for nm in names:
        path = os.path.join(args.workdir, f"engine-{nm}.pt")
        if not args.trainer_only:
            if nm in OLD_ENGINES:
                if not os.path.exists(args.old_python):
                    continue
                env_lib = ":".join(sorted(__import__("glob").glob(os.path.expanduser(
                    "~/miniconda3/envs/maia/lib/python3.11/site-packages/nvidia/*/lib"))))
                os.environ["LD_LIBRARY_PATH"] = env_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                path = run_engine(nm, OLD_ENGINES[nm], args.workdir, args.old_python, args.quick, log)
            else:
                path = run_engine(nm, ENGINES[nm], args.workdir, sys.executable, args.quick, log)
        if path and os.path.exists(path):
            recs[nm] = load_record(path)
        else:
            res.check(f"engine {nm} starts", False, level="other", impl=nm, expect="info")
    if args.strip:                                                  # for a torch-only environment
        os.makedirs(args.strip, exist_ok=True)
        for nm, r in recs.items():
            torch.save({k: v for k, v in r.items() if k != "capture"}, os.path.join(args.strip, f"engine-{nm}.pt"))
        return
    res.measure("engine runs wall time (0 when records were reused)", round(time.time() - t0, 1), "s")
    if args.engines_only:
        return

    # 1. engine self-consistency
    for nm, r in recs.items():
        runs = r["runs"]
        Bs = sorted(int(k[1:]) for k in runs if k[1:].isdigit())
        Bmax = max(Bs)
        bi = "--bi" in (ENGINES.get(nm) or [])
        if f"B{Bmax}_repeat" in runs:
            c = compare_runs(runs[f"B{Bmax}"], runs[f"B{Bmax}_repeat"])
            res.check(f"engine L1: batch {Bmax} twice", c["logprob_differ"] == 0 and c["token_seq_differ"] == 0,
                      level="L1", impl=nm, expect="info", detail=c)
        for B in Bs[1:]:
            c = compare_runs(runs["B1"], runs[f"B{B}"])
            res.check(f"engine L2: alone vs batch {B}", c["logprob_differ"] == 0 and c["token_seq_differ"] == 0,
                      level="L2", impl=nm, expect=L2_EXPECT.get(nm, "info"), detail=c)
        res.extra.setdefault("engines", {})[nm] = {"vllm": r["vllm"], "torch": r["torch"], "gpu": r["gpu"],
                                                   "attn_backend": r["attn_backend"], "init_s": round(r["init_s"], 1)}
        tf = os.path.join(args.tput_dir or args.workdir, f"tput-{nm}.pt")
        tp = load_record(tf).get("tput") if os.path.exists(tf) else None
        if tp:
            for B, d in tp.items():
                res.measure(f"vLLM output throughput B={B}", round(d["out_tok_per_s"], 1), "tok/s", impl=nm,
                            median_s=round(d["median_s"], 3), runs_s=[round(x, 3) for x in d["runs_s"]],
                            setup="128-token prompts, 128 new tokens, greedy, no logprobs; warm-up 1, median of 3; "
                                  "one config at a time on one GPU")
        elif "tput" in r:
            for B, d in r["tput"].items():
                res.measure(f"vLLM output throughput B={B}", round(d["out_tok_per_s"], 1), "tok/s", impl=nm,
                            median_s=round(d["median_s"], 3))

    if not recs:
        res.write(args.json)
        sys.exit(1)
    for a, b in (("default", "eager"), ("default", "bi"), ("bi", "bi_eager"), ("bi_eager", "bi_eager_lmhead"),
                 ("eager", "triton"), ("eager", "flashinfer"), ("eager", "flex"), ("bi_eager", "bi_triton_eager"),
                 ("eager", "v063_eager"), ("default", "v063_default"), ("v063_default", "v063_eager")):
        if a in recs and b in recs:
            c = compare_runs(recs[a]["runs"]["B1"], recs[b]["runs"]["B1"])
            res.extra.setdefault("engine_vs_engine_B1", {})[f"{a} vs {b}"] = c
    # 2. trainer vs engine
    model = tr.load_qwen(MODEL_DIR, ops="stock")
    T = Trainer(model)
    some = next(iter(recs.values()))
    probe = torch.tensor(some["runs"]["B1"][0]["ids"])
    ref_eq = tr.forward_matched(model, probe, {}).equal(model.prefill(probe[None].cuda(), ops="stock")[0])
    res.check("forward_matched(REFERENCE) == model.py stock prefill, bitwise", ref_eq, level="other",
              impl="trainer", expect="pass")
    for nm, r in recs.items():
        if "rope_cache" in r and tr.HAVE_VLLM:
            mine = tr.vllm_cos_sin_cache(model.cfg, "cuda")[:4096].cpu()
            res.check(f"trainer's copy of vLLM cos_sin_cache == engine's ({nm})", mine.equal(r["rope_cache"]),
                      level="other", impl=nm, expect="info")
            hf = model.ctx.tables("cuda")
            D = model.cfg.head_dim
            hft = torch.cat((hf["cos"][:4096, :D // 2], hf["sin"][:4096, :D // 2]), -1).cpu()
            diff = (hft.view(torch.int16) != r["rope_cache"].view(torch.int16))
            rows = diff.any(-1).nonzero()[:, 0].tolist()
            res.extra["rope_table_hf_vs_vllm"] = {"entries_differ": int(diff.sum()), "entries": diff.numel(),
                                                  "first_positions_differ": rows[:10], "positions_differ": len(rows)}
            break
    variants = ["model.py stock", "model.py bi"]
    if tr.HAVE_VLLM and not args.trainer_only:
        variants += [("matched:" + t, vkey(TARGETS[t])) for t in TARGETS]
    for nm, r in recs.items():
        for v in variants:
            vname, vv = (v, v) if isinstance(v, str) else v
            if not isinstance(v, str) and vname.split(":")[1] != nm and nm not in ("bi_eager", "bi_eager_lmhead", "eager", "default", "bi"):
                continue
            c = l4_compare(T, r["runs"], vv)
            exact = c["differ"] == 0
            expect = "info"
            if vname == "matched:" + nm and nm == "bi_eager_lmhead":
                expect = "pass"
            res.check(f"L4 trainer {vname} vs engine {nm}: realized-token logprobs, all batches", exact,
                      level="L4", impl=nm, expect=expect, detail=c)
            how = "vllm_bi" if "--bi" in ENGINES.get(nm, []) and tr.HAVE_VLLM else "torch"
            if r.get("kl") and r.setdefault("_kl_ok", kl_capture_ok(r["kl"], how)):
                k = kl_compare(T, r["kl"], vv)
                res.extra.setdefault("kl", {}).setdefault(nm, {})[vname] = k
            print(f"L4 {nm:18s} {vname:28s} {c['differ']}/{c['tokens']} differ, max {c['max_abs_dlogp']:.3g}", flush=True)
        res.extra.setdefault("kl_capture_reproduces_engine_logprobs", {})[nm] = bool(r.get("_kl_ok"))
        T.cache.clear()

    # trainer cost: one 512-token sequence, forward to logprobs (warm-up 2, synchronize, median of 5)
    pr = some["prompts"]
    flat = []
    while len(flat) < 512:                                          # prompt 10 onward, cyclically
        for p_ in pr[10 % len(pr):] + pr[:10 % len(pr)]:
            flat += p_
    ids512 = torch.tensor(flat[:512])

    def _time(fn):
        for _ in range(2):
            fn()
        ts = []
        for _ in range(5):
            torch.cuda.synchronize()
            a0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - a0)
        return sorted(ts)[2] * 1e3
    with torch.no_grad():
        res.measure("trainer forward+logprobs T=512: model.py stock", round(_time(
            lambda: model.logprobs(model.prefill(ids512[None].cuda(), ops="stock"))), 2), "ms", impl="trainer")
        res.measure("trainer forward+logprobs T=512: model.py bi", round(_time(
            lambda: model.logprobs(model.prefill(ids512[None].cuda(), ops="bi"))), 2), "ms", impl="trainer")
        if tr.HAVE_VLLM and not args.trainer_only:
            for t in TARGETS:
                ch = TARGETS[t]
                res.measure(f"trainer forward+logprobs T=512: matched:{t}", round(_time(
                    lambda: tr.logprobs(tr.forward_matched(model, ids512, ch), ch["logsoftmax"])), 2), "ms", impl="trainer")

    # trainer across torch versions: model.py's realized logprobs on the B1 sequences of one engine record
    ref_nm = "bi_eager_lmhead" if "bi_eager_lmhead" in recs else next(iter(recs))
    mine = {v: {i: T.realized(rec["ids"], v) for i, rec in recs[ref_nm]["runs"]["B1"].items()}
            for v in ("model.py stock", "model.py bi")}
    if args.save_trainer:
        torch.save({"torch": torch.__version__, "engine": ref_nm, "lp": mine}, args.save_trainer)
    if args.compare_trainer and os.path.exists(args.compare_trainer):
        o = torch.load(args.compare_trainer, weights_only=False)
        for v in mine:
            n = sum(x.numel() for x in mine[v].values())
            nd = sum(int((mine[v][i].view(torch.int32) != o["lp"][v][i].view(torch.int32)).sum()) for i in mine[v])
            res.check(f"trainer {v}: torch {torch.__version__} vs {o['torch']}, realized logprobs bitwise", nd == 0,
                      level="other", impl=v, expect="info", detail={"tokens": n, "differ": nd})
    T.cache.clear()

    # 3. op replay and stepwise alignment (needs vLLM kernels)
    if tr.HAVE_VLLM and not args.trainer_only:
        for nm in CAPTURE:
            if nm not in recs or "capture" not in recs[nm]:
                continue
            cap = recs[nm]["capture"]
            res.extra.setdefault("op_replay", {})[nm] = op_replay(model, cap)
            target = TARGETS[nm]
            steps, ch = [], {}
            seq = [("model.py stock (reference)", {})]
            for k, val in STEPS[nm]:
                ch = {**ch, k: val}
                seq.append((f"+ {k}={val}", dict(ch)))
            assert vkey(ch) == vkey(target), nm
            for label, c in seq:
                cmp_ = l4_compare(T, recs[nm]["runs"], vkey(c))
                loc = layer_localize(model, cap, c)
                steps.append({"step": label, **cmp_, "first_layer_diff": loc})
                print(f"step {nm:16s} {label:30s} {cmp_['differ']}/{cmp_['tokens']} B1 {cmp_['differ_B1']}/{cmp_['tokens_B1']}  {loc}", flush=True)
            loo = []
            for k in tr.CHOICES:
                if target.get(k, tr.REFERENCE[k]) == tr.REFERENCE[k]:
                    continue
                if k == "bias" and target["gemm"] == "vllm_bi":         # linear_batch_invariant always adds it after
                    continue
                c = {**target, k: tr.REFERENCE[k]}
                cmp_ = l4_compare(T, recs[nm]["runs"], vkey(c))
                loo.append({"revert": f"{k}={tr.REFERENCE[k]}", "differ": cmp_["differ"], "tokens": cmp_["tokens"],
                            "differ_B1": cmp_["differ_B1"],
                            "max_abs_dlogp": cmp_["max_abs_dlogp"]})
            res.extra.setdefault("steps", {})[nm] = steps
            res.extra.setdefault("leave_one_out", {})[nm] = loo
            T.cache.clear()
    res.write(args.json)
    sys.exit(res.exit_code())


if __name__ == "__main__":
    main()
