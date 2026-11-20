"""The whole model under tensor parallelism (L5): TP = 1, 2, 4, 8 (16 on two nodes), and
prefill/decode disaggregation with different TP degrees.

The model is examples/whole_model/model.py's dense Config() (8 layers, hidden 1024, 16/4 heads,
intermediate 3584, vocab 32000), sharded Megatron-style by tp_model.py. TP = 1 of each variant is the
reference; "stock" at TP = 1 is also checked against model.py's stock forward.

Sections (--only a,b,...):
  tp        trainer forward B x T: logits and realized log-probs at every degree against TP = 1;
            the vocab-parallel log_softmax against its one-GPU form; repeats (L1)
  modes     engine at TP = 2 and 4: decode in a changing batch, chunked prefill (in a changing batch
            and alone), against the TP = 1 trainer forward
  disagg    prefill at TP = a, KV cache handed over (re-sharded by heads, NCCL send/recv), decode at
            TP = b in a shrinking batch; against one-GPU prefill + decode and the TP = 1 trainer forward
  backward  "inv" only: gradients of a fixed weighted sum of realized log-probs with respect to the
            residual stream (every layer input, the final-norm input), TP = 2, 4 against TP = 1
  cost      prefill (B=4, T=512) and decode (B=16, context 512) per degree, inv against stock

Run with torchrun, one process per GPU (ranks 0..t-1 form TP group t; decode groups of the
disaggregation section use the last b ranks):
  torchrun --nproc_per_node=4 whole_model_tp.py --quick --json quick.json      (~2 min)
  torchrun --nproc_per_node=8 whole_model_tp.py --json results-a16.json
  python whole_model_tp.py --merge parts/*.json --json results-a16.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "whole_model"))
sys.path.insert(0, HERE)
from bi_check import bit_equal, bits, first_mismatch  # noqa: E402
from bi_report import Results  # noqa: E402
import model as wm  # noqa: E402
import whole_model as wmm  # noqa: E402
import tp_model as tpm  # noqa: E402
from tp_model import Grp, TPModel, gather_cols, heads_of  # noqa: E402

SECTIONS = ("tp", "modes", "disagg", "backward", "cost")
RANK = 0


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


def on(grp: Grp, fn):
    """Run fn on the members of grp, then a barrier on every rank. Returns fn's value (None elsewhere)."""
    out = fn() if grp.member else None
    torch.cuda.synchronize()
    dist.barrier()
    return out


def to_cpu(x):
    if torch.is_tensor(x):
        return x.cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    return x


def to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, (list, tuple)):
        return type(x)(to_cuda(v) for v in x)
    if isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    return x


def send_to_rank0(obj, src: int):
    """obj lives on global rank src; every rank calls. Returns it on rank 0 (None elsewhere)."""
    if src == 0:
        return obj if RANK == 0 else None
    box = [to_cpu(obj) if RANK == src else None]
    dist.broadcast_object_list(box, src=src)
    return to_cuda(box[0]) if RANK == 0 else None


def sha(t):
    return hashlib.sha1(bits(t).cpu().numpy().tobytes()).hexdigest()[:16]


def lp_one(variant, x):
    """The variant's log_softmax of full logits [M, V] on one GPU (rank 0)."""
    one = Grp([RANK])
    return tpm.lsm_stock(x, one) if variant == "stock" else tpm.lsm_inv(x, one, 16)


def expect_for(variant, differs: bool):
    """inv: the claim holds. stock: variance is expected where the degree differs."""
    if variant == "inv":
        return "pass"
    return "fail" if differs else "info"


IMPL = {"inv": "logical TP C=16 + fixed-order vocab log_softmax (bi ops)", "stock": "NCCL all-reduce (stock ops)"}


# ----------------------------------------------------------------------------- tp
def sec_tp(res, ref, G, degrees, quick):
    V = ref.cfg.vocab_size
    B, T = (2, 100) if quick else (4, 160)
    ids = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(1)).cuda()
    rl = lambda lp: wm.realized_logprob(lp, ids)  # noqa: E731

    def run(t, variant, reps=1):
        def f():
            m = TPModel(ref, G[t], variant)
            outs = []
            for _ in range(reps):
                with torch.no_grad():
                    flat = m.logits_local(ids).reshape(B * T, -1)
                    lp_loc = m.lsm_local(flat)
                    lg, lp = gather_cols(flat, m.grp), gather_cols(lp_loc, m.grp)
                if RANK == 0:
                    outs.append({"logits": lg.view(B, T, V), "logp": lp.view(B, T, V),
                                 "logp1": m.logprobs(lg).view(B, T, V)})
            del m
            return outs
        return on(G[t], f)

    for variant in ("inv", "stock"):
        impl = IMPL[variant]
        base = run(1, variant)
        if RANK == 0:
            base = base[0]
            with torch.no_grad():
                mlg = ref.prefill(ids, ops="bi" if variant == "inv" else "stock")
            mm = first_mismatch(base["logits"], mlg)
            if variant == "stock":
                res.check(f"stock TP=1 (sharding code, one rank) equals model.py's stock forward, logits B={B} T={T}",
                          mm is None, level="other", impl=impl, detail=mm)
            else:
                res.check(f"inv TP=1 against model.py's bi forward (logical TP C=16 is not a plain GEMM), logits B={B} T={T}",
                          mm is None, level="other", impl=impl, expect="info",
                          detail={"mismatched_logits": 0 if mm is None else mm["mismatched_elements"], "of": mlg.numel()})
            log(f"[tp] {variant} TP=1 vs model.py: {'equal' if mm is None else mm['mismatched_elements']} ")
            base["rlp"] = rl(base["logp"])
        for t in degrees[1:]:
            got = run(t, variant)
            if RANK != 0:
                continue
            got = got[0]
            st = wmm.merge_stats([wmm.compare({"logits": base["logits"][b], "rlp": base["rlp"][b]},
                                              [(0, got["logits"][b], rl(got["logp"])[b])], T) for b in range(B)])
            lpm = first_mismatch(base["logp"], got["logp"])
            wmm.record(res, f"trainer forward B={B} T={T}: logits and realized logprobs (vocab-parallel log_softmax), "
                            f"TP={t} vs TP=1", st, "L5", impl, expect_for(variant, True),
                       {"degree": t, "full_logp_mismatched": 0 if lpm is None else lpm["mismatched_elements"],
                        "full_logp_elems": got["logp"].numel()})
            same1 = bit_equal(got["logp"], got["logp1"])
            res.check(f"TP={t}: vocab-parallel log_softmax equals the one-GPU formula on the gathered logits",
                      same1, level="L5", impl=impl, expect="pass" if variant == "inv" else "info",
                      detail=first_mismatch(got["logp1"], got["logp"]))
            del got
        del base
        torch.cuda.empty_cache()
        # L1 at the largest degree
        t = degrees[-1]
        outs = run(t, variant, reps=3)
        if RANK == 0:
            ok = all(bit_equal(outs[0]["logits"], o["logits"]) and bit_equal(outs[0]["logp"], o["logp"]) for o in outs[1:])
            res.check(f"trainer forward TP={t}, 3 runs: logits and log-probs", ok, level="L1", impl=impl,
                      expect="pass" if variant == "inv" else "info")
            log(f"[tp] {variant} L1 at TP={t}: {ok}")
        del outs
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- modes
def sec_modes(res, ref, G, degrees, quick):
    V = ref.cfg.vocab_size
    T = 100 if quick else 160
    ids = torch.randint(0, V, (T,), generator=torch.Generator().manual_seed(2)).cuda()
    busy = (1, 2, 3, 8, 9) if quick else (1, 2, 3, 5, 8, 9, 15, 16, 17, 32, 33)
    for variant in ("inv",) if quick else ("inv", "stock"):
        impl = IMPL[variant]
        refd = on(G[1], lambda: wmm.reference(TPModel(ref, G[1], variant), None, ids) if RANK == 0 else None)
        modes = [("decode, one token per step, in a changing batch", [1] * T, busy),
                 ("chunked prefill, chunk 64, in a changing batch", wmm.chunk_plan(T, 64), busy),
                 ("chunked prefill, chunk 7, alone", wmm.chunk_plan(T, 7), None)]
        if variant == "stock":
            modes = modes[:1]
        for t in (2, 4):
            if t not in degrees:
                continue
            for name, plan, bz in modes:
                def f():
                    m = TPModel(ref, G[t], variant)
                    return wmm.run_mode(m, None, ids, plan, busy_sizes=bz, seed=5, poison=variant == "inv")
                got = on(G[t], f)
                if RANK == 0:
                    out, info = got
                    wmm.record(res, f"engine at TP={t}: {name} vs TP=1 trainer forward (T={T})", wmm.compare(refd, out, T),
                               "L5", impl, expect_for(variant, True), {"degree": t, "batch_sizes_seen": info["batch_sizes"],
                                                                       "engine_steps": info["steps"]})
                del got
                torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- disaggregation
def full_kv(cache, m):
    """Gather a TP group's KV caches into [layers, slots, cap, kv_heads, D] (on TP rank 0), and check
    that replicated heads hold the same bits on every rank that keeps them."""
    cfg, g = m.full_cfg, m.grp
    Gk = g.all_gather(cache.k) if g.t > 1 else cache.k[None]
    Gv = g.all_gather(cache.v) if g.t > 1 else cache.v[None]
    k = torch.empty(*cache.k.shape[:3], cfg.kv_heads, cfg.head_dim, device=cache.k.device, dtype=cache.k.dtype)
    v = torch.empty_like(k)
    repl_ok, done = True, set()
    for j in range(g.t):
        for p, h in enumerate(heads_of(cfg, g.t, j)[1]):
            if h in done:
                repl_ok &= bit_equal(k[:, :, :, h], Gk[j][:, :, :, p]) and bit_equal(v[:, :, :, h], Gv[j][:, :, :, p])
                continue
            k[:, :, :, h], v[:, :, :, h] = Gk[j][:, :, :, p], Gv[j][:, :, :, p]
            done.add(h)
    return (k, v, repl_ok) if g.idx == 0 else None


def transfer_kv(cfg, cache_a, ga: Grp, cache_b, gb: Grp, shape):
    """Hand the prefill instance's KV over to the decode instance, re-sharded by KV head: each head
    goes from the first prefill rank that holds it to every decode rank that needs it, with NCCL
    send/recv (or a local copy when both are the same process). Every rank calls. Returns bytes sent."""
    L, S, cap, D = shape
    ops, pend, sent = [], [], 0
    for h in range(cfg.kv_heads):
        sj = next(j for j in range(ga.t) if h in heads_of(cfg, ga.t, j)[1])
        src, sp = ga.ranks[sj], heads_of(cfg, ga.t, sj)[1].index(h)
        for j in range(gb.t):
            kb = heads_of(cfg, gb.t, j)[1]
            if h not in kb:
                continue
            dst, dp = gb.ranks[j], kb.index(h)
            if RANK == src:
                buf = torch.stack([cache_a.k[:, :, :, sp], cache_a.v[:, :, :, sp]]).contiguous()
                if dst == src:
                    cache_b.k[:, :, :, dp], cache_b.v[:, :, :, dp] = buf[0], buf[1]
                else:
                    ops.append(dist.P2POp(dist.isend, buf, dst))
                    sent += buf.numel() * buf.element_size()
            elif RANK == dst:
                rb = torch.empty(2, L, S, cap, D, device="cuda", dtype=torch.bfloat16)
                ops.append(dist.P2POp(dist.irecv, rb, src))
                pend.append((rb, dp))
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()
    for rb, dp in pend:
        cache_b.k[:, :, :, dp], cache_b.v[:, :, :, dp] = rb[0], rb[1]
    torch.cuda.synchronize()
    return sent


def run_disagg(ref, variant, ga: Grp, gb: Grp, reqs, P, Dn, cap):
    cfg = ref.cfg
    n = len(reqs)
    poison = variant == "inv"
    st = {}
    if ga.member:
        ma = TPModel(ref, ga, variant)
        st["ca"] = wm.KVCache(ma, n, cap, poison=poison)
        outs = ma.forward_cached(st["ca"], list(range(n)), [reqs[r][:P[r]] for r in range(n)])
        st["pre"] = outs if RANK == ga.ranks[0] else None
        st["kvA"] = full_kv(st["ca"], ma)
    if gb.member:
        mb = TPModel(ref, gb, variant)
        st["cb"] = wm.KVCache(mb, n, cap, poison=poison)
    xfer_ms = []
    for _ in range(2):                        # the second hand-over is timed without first-use setup
        dist.barrier()
        t0 = time.perf_counter()
        sent = transfer_kv(cfg, st.get("ca"), ga, st.get("cb"), gb, (cfg.layers, n, cap, cfg.head_dim))
        dist.barrier()
        xfer_ms.append((time.perf_counter() - t0) * 1e3)
    dec = None
    if gb.member:
        cb = st["cb"]
        cb.length = list(P)
        kvB = full_kv(cb, mb)
        dec = [[] for _ in range(n)]
        for s in range(max(Dn)):
            act = [r for r in range(n) if s < Dn[r]]
            outs = mb.forward_cached(cb, act, [reqs[r][P[r] + s:P[r] + s + 1] for r in act])
            for j, r in enumerate(act):
                dec[r].append(outs[j])
        dec = {"dec": [torch.cat(d) for d in dec], "kvB": kvB} if gb.idx == 0 else None
    st.pop("ca", None), st.pop("cb", None)
    sent_all = [None] * dist.get_world_size()
    dist.all_gather_object(sent_all, sent)
    pre = send_to_rank0(st.get("pre"), ga.ranks[0])
    kvA = send_to_rank0(st.get("kvA"), ga.ranks[0])
    dec = send_to_rank0(dec, gb.ranks[0])
    dist.barrier()
    if RANK != 0:
        return None
    logits = [torch.cat([pre[r], dec["dec"][r]]) for r in range(n)]
    rlp = [wm.realized_logprob(lp_one(variant, lg), reqs[r]) for r, lg in enumerate(logits)]
    return {"logits": logits, "rlp": rlp, "kvA": kvA, "kvB": dec["kvB"], "bytes_sent": sum(sent_all), "xfer_ms": xfer_ms}


def sec_disagg(res, ref, G, Gt, pairs, quick):
    V = ref.cfg.vocab_size
    P, Dn = ((37, 64), (24, 17)) if quick else ((37, 64, 100), (40, 25, 33))
    g = torch.Generator().manual_seed(3)
    reqs = [torch.randint(0, V, (p + d,), generator=g).cuda() for p, d in zip(P, Dn)]
    cap = max(p + d for p, d in zip(P, Dn)) + 8
    for variant in ("inv", "stock"):
        impl = IMPL[variant]
        base = run_disagg(ref, variant, G[1], G[1], reqs, P, Dn, cap)
        trainer = on(G[1], lambda: [wmm.reference(TPModel(ref, G[1], variant), None, r) for r in reqs] if RANK == 0 else None)
        for a, b in pairs:
            got = run_disagg(ref, variant, G[a], Gt[b], reqs, P, Dn, cap)
            if RANK != 0:
                continue
            tag = f"prefill TP={a} -> decode TP={b}"
            kA, vA, replA = got["kvA"]
            kB, vB, replB = got["kvB"]
            copy = bit_equal(kA, kB) and bit_equal(vA, vB) and replA and replB
            res.check(f"{tag}: the KV cache received by the decode ranks is bitwise the prefill ranks' cache "
                      f"(re-sharded by head, NCCL send/recv)", copy, level="L5", impl=impl,
                      detail={"sha_k_prefill": sha(kA), "sha_k_decode": sha(kB), "sha_v_prefill": sha(vA),
                              "sha_v_decode": sha(vB), "replicated_heads_equal": [replA, replB],
                              "bytes_sent": got["bytes_sent"], "transfer_ms_first_and_second": [round(x, 2) for x in got["xfer_ms"]]})
            k1, v1, _ = base["kvA"]
            kv_same = all(bit_equal(kA[:, r, :p], k1[:, r, :p]) and bit_equal(vA[:, r, :p], v1[:, r, :p])
                          for r, p in enumerate(P))
            res.check(f"{tag}: prefill KV at TP={a} equals one-GPU prefill KV", kv_same, level="L5", impl=impl,
                      expect="pass" if variant == "inv" else "info")
            sts = [wmm.compare({"logits": base["logits"][r], "rlp": base["rlp"][r]}, [(0, got["logits"][r], got["rlp"][r])],
                               P[r] + Dn[r]) for r in range(len(P))]
            wmm.record(res, f"{tag}: logits and realized logprobs of {len(P)} requests (prompts {list(P)}, decode {list(Dn)}, "
                            f"shrinking batch) vs one-GPU prefill + decode", wmm.merge_stats(sts), "L5", impl,
                       expect_for(variant, True), {"prefill_tp": a, "decode_tp": b})
            if variant == "inv":
                sts = [wmm.compare(trainer[r], [(0, got["logits"][r], got["rlp"][r])], P[r] + Dn[r]) for r in range(len(P))]
                wmm.record(res, f"{tag}: vs the TP=1 trainer forward of each request", wmm.merge_stats(sts), "L5", impl, "pass")
            log(f"[disagg] {variant} {tag}: copy {copy}, KV==1GPU {kv_same}, bytes {got['bytes_sent']}, {got['xfer_ms'][1]:.2f} ms")
            del got
        del base, trainer
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- backward
def sec_backward(res, ref, G, degrees, quick):
    cfg = ref.cfg
    V = cfg.vocab_size
    B, T = (1, 64) if quick else (2, 128)
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, V, (B, T), generator=g).cuda()
    wts = torch.randn(B, T - 1, generator=g).cuda()
    degs = [t for t in degrees if t <= cfg.kv_heads]

    def run(t, bwd_nccl=False):
        def f():
            m = TPModel(ref, G[t], "inv", bwd_nccl=bwd_nccl)
            taps = []
            with torch.enable_grad():
                loc = m.logits_local(ids, taps)
                Vl = loc.shape[-1]
                lp = m.lsm_local(loc.reshape(B * T, Vl))
                Gf = torch.zeros(B, T, V, device="cuda")
                Gf[:, :-1].scatter_(-1, ids[:, 1:, None], wts[..., None])
                lp.backward(Gf[..., m.vlo:m.vlo + Vl].reshape(B * T, Vl).contiguous())
            out = [x.grad.detach().clone() for x in taps] if RANK == 0 else None
            del taps, loc, lp
            return out
        return on(G[t], f)

    base = run(1)
    names = [f"layer{i} input" for i in range(cfg.layers)] + ["final norm input"]
    for t, nc in [(t, False) for t in degs[1:]] + [(t, True) for t in degs[1:]]:
        got = run(t, nc)
        if RANK != 0:
            continue
        if nc:
            bad = [i for i in range(len(base)) if not bit_equal(base[i], got[i])]
            res.check(f"backward B={B} T={T}: same, with the column-parallel input gradients summed by NCCL all-reduce "
                      f"(bf16 partials, Megatron's backward), TP={t} vs TP=1", not bad, level="L5",
                      impl="inv forward, NCCL all-reduce backward", expect="fail",
                      detail={"degree": t, "tensors_differing": [names[i] for i in bad]})
            log(f"[backward] TP={t} with NCCL backward: {len(bad)} of {len(base)} tensors differ")
            continue
        bad = {i: first_mismatch(base[i], got[i]) for i in range(len(base))}
        bad = {i: v for i, v in bad.items() if v}
        res.check(f"backward B={B} T={T}: d(weighted sum of realized logprobs)/d(residual stream) at {len(base)} points, "
                  f"TP={t} vs TP=1", not bad, level="L5", impl=IMPL["inv"] + ", logical-TP input gradients",
                  detail={"degree": t, "tensors_differing": [names[i] for i in bad],
                          "worst_fraction": max([v["mismatched_elements"] / v["of"] for v in bad.values()], default=0.0),
                          "grad_abs_max": float(base[0].float().abs().max())})
        log(f"[backward] TP={t}: {'equal' if not bad else str(len(bad)) + ' tensors differ'}")
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- cost
def sec_cost(res, ref, G, degrees, quick):
    V = ref.cfg.vocab_size
    Bp, Tp = (2, 256) if quick else (4, 512)
    Bd, ctx, steps = (8, 256, 10) if quick else (16, 512, 20)
    g = torch.Generator().manual_seed(6)
    ids = torch.randint(0, V, (Bp, Tp), generator=g).cuda()
    prompts = [torch.randint(0, V, (ctx,), generator=g).cuda() for _ in range(Bd)]
    toks = [torch.randint(0, V, (1,), generator=g).cuda() for _ in range(Bd)]
    rows = {}
    for t in degrees:
        for variant in ("stock", "inv"):
            def f():
                m = TPModel(ref, G[t], variant)
                with torch.no_grad():
                    ms_p = wmm.median_ms(lambda: m.lsm_local(m.logits_local(ids).reshape(Bp * Tp, -1)), warmup=2, runs=5)
                cache = wm.KVCache(m, Bd, ctx + 8)
                for s in range(0, Bd, 4):
                    sl = list(range(s, min(Bd, s + 4)))
                    m.forward_cached(cache, sl, [prompts[i] for i in sl])
                ts = []
                for i in range(steps + 3):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    m.forward_cached(cache, list(range(Bd)), toks)
                    torch.cuda.synchronize()
                    if i >= 3:
                        ts.append((time.perf_counter() - t0) * 1e3)
                    for s in range(Bd):
                        cache.truncate(s, ctx)
                del cache, m
                return ms_p, statistics.median(ts)
            r = on(G[t], f)
            if RANK == 0:
                rows[(t, variant)] = r
                ms_p, ms_d = r
                res.measure(f"prefill B={Bp} T={Tp} (forward + vocab-parallel log_softmax), TP={t}", round(Bp * Tp / ms_p * 1e3, 1),
                            "tokens/s", impl=variant, ms=round(ms_p, 3), degree=t)
                res.measure(f"decode B={Bd} context {ctx} (one step, logits gathered), TP={t}", round(Bd / ms_d * 1e3, 1),
                            "tokens/s", impl=variant, ms_per_step=round(ms_d, 3), degree=t)
            torch.cuda.empty_cache()
        if RANK == 0:
            (sp, sd), (ip, idd) = rows[(t, "stock")], rows[(t, "inv")]
            res.measure(f"prefill TP={t}: inv time / stock time", round(ip / sp, 3), "x", impl="inv", degree=t)
            res.measure(f"decode TP={t}: inv time / stock time", round(idd / sd, 3), "x", impl="inv", degree=t)
            log(f"[cost] TP={t}: prefill stock {sp:.1f} ms inv {ip:.1f} ms ({ip / sp:.2f}x); "
                f"decode stock {sd:.2f} ms inv {idd:.2f} ms ({idd / sd:.2f}x)")


# ----------------------------------------------------------------------------- main
def foreign_procs():
    """Compute processes on this node's GPUs that are not ours (a shared host: costs need none)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return len(out) - int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    except Exception as ex:  # noqa: BLE001
        return str(ex)


def merge(paths, out):
    res = Results("whole_model_tp", quick=False, argv=["--merge"] + list(paths))
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if d["extra"].get("world", 0) > 8:                    # the two-node run repeats some names
            for c in d["checks"]:
                c["name"] += f" [two nodes, world {d['extra']['world']}]"
            for m in d["measurements"]:
                m["run"] = f"two nodes, world {d['extra']['world']}"
        res.checks += d["checks"]
        res.measurements += d["measurements"]
        res.extra.setdefault("merged_from", []).append({"argv": d["argv"][1:], "counts": d["counts"],
                                                        "seconds": d["extra"].get("seconds"), "world": d["extra"].get("world")})
        for k in ("topology", "config", "nccl", "nccl_env", "logical_tp"):
            if k in d["extra"]:
                res.extra.setdefault(k + ("_2node" if d["extra"].get("world", 0) > 8 else ""), d["extra"][k])
    res.env = d["env"]
    res.write(out)
    return res


def main():
    global RANK
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="TP 1/2/4, reduced sizes, every expect='pass' check")
    ap.add_argument("--json")
    ap.add_argument("--only", default=",".join(SECTIONS))
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--degrees", help="comma-separated TP degrees (default: 1, 2, 4, ... up to the world size)")
    args = ap.parse_args()
    if args.merge:
        sys.exit(merge(args.merge, args.json).exit_code())
    sections = [s for s in args.only.split(",") if s]
    assert all(s in SECTIONS for s in sections), sections
    if args.quick:
        sections = [s for s in sections if s != "cost"]
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    RANK, world = dist.get_rank(), dist.get_world_size()
    t_start = time.time()
    degrees = [t for t in (1, 2, 4, 8, 16) if t <= world and (not args.quick or t <= 4)]
    if args.degrees:
        degrees = [int(x) for x in args.degrees.split(",")]
        assert degrees[0] == 1 and all(t <= world for t in degrees), degrees
    G = {1: Grp([0])}
    for t in degrees[1:]:
        G[t] = tpm.make_group(range(t))
    pairs = [p for p in ((1, 2), (2, 4), (4, 1), (8, 2)) if all(x in degrees for x in p)]
    if args.quick:
        pairs = pairs[:3]
    Gt = {}
    for b in sorted({b for _, b in pairs}):                   # decode instances: the last b ranks
        rk = list(range(world - b, world))
        Gt[b] = next((G[t] for t in G if G[t].ranks == rk), None) or tpm.make_group(rk)
    cfg = wm.Config()
    ref = wm.build(cfg, ops="bi", seed=0, device="cuda")
    ref.requires_grad_(False)
    res = Results("whole_model_tp", quick=args.quick)
    if RANK == 0:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
        res.extra.update(world=world, degrees=degrees, pairs=pairs, sections=sections,
                         config={k: getattr(cfg, k) for k in ("vocab_size", "hidden", "layers", "heads", "kv_heads", "intermediate")},
                         logical_tp={"C": 16, "Cv": 16, "chunk_tree": "pairwise", "row_parallel": "o_proj, down_proj"},
                         nccl=".".join(map(str, torch.cuda.nccl.version())),
                         nccl_env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                         topology=topo.replace("\x1b[4m", "").replace("\x1b[0m", ""))
        log(f"world {world}, degrees {degrees}, pairs {pairs}, sections {sections}")
    for s in sections:
        t0 = time.time()
        if RANK == 0:
            res.extra.setdefault("foreign_gpu_processes_at_start", {})[s] = foreign_procs()
        if s == "tp":
            sec_tp(res, ref, G, degrees, args.quick)
        elif s == "modes":
            sec_modes(res, ref, G, degrees, args.quick)
        elif s == "disagg":
            sec_disagg(res, ref, G, Gt, pairs, args.quick)
        elif s == "backward":
            sec_backward(res, ref, G, degrees, args.quick)
        elif s == "cost":
            sec_cost(res, ref, G, degrees, args.quick)
        log(f"== section {s}: {time.time() - t0:.1f} s")
    rc = 0
    if RANK == 0:
        res.extra["seconds"] = round(time.time() - t_start, 1)
        res.write(args.json)
        rc = res.exit_code()
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
