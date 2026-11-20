"""Whole-model invariance: per-token logits and realized log-probabilities, compared bitwise.

A sequence's logits must not depend on what else is in the batch (L2) or on how
the engine reached each position (L3). The model is examples/whole_model/model.py
(a Llama-architecture LM), run with its "stock" op set (cuBLAS, SDPA, torch
reductions) and its "bi" op set (tools/bi_kernels.py plus the kernels in model.py).

Sections (--only a,b,...):
  kernels   the pieces under the model: torch's row reduction against the fixed-tree
            row sum (RMSNorm variance), the GEMM at the model's shapes, the copied
            per-sequence attention against tools' attention_fixed_blocks
  spread    how peaked the random model's next-token distribution is
  l1        the same batch, repeated: trainer forward and engine decode
  l2        a sequence alone against co-batched, trainer forward: batch sizes,
            positions, fillers
  modes     the engine: decode with a KV cache in a batch whose membership changes,
            chunked prefill, prefix-cache hit, speculative k-token appends with
            rollback, preempt and recompute, right padding, packed varlen. Each
            mode alone against the trainer forward (L3) and busy against alone (L2)
  lengths   sequence lengths around the attention block size, trainer against decode
  backward  the gradient of one sequence's loss with respect to its own activations,
            alone against co-batched; parameter gradients run to run
  rounding  one rounding choice changed at a time (bi op set): how many logits move
  cost      tokens per second, prefill and decode, bi against stock
Variants (--variants dense,moe,wide): the default model, the MoE variant, the wide one.

Quantities for a failing comparison (what RL would see): the fraction of tokens whose
realized log-probability differs, the largest |delta logprob|, and the KL divergence
between the two next-token distributions.

Run (CUDA, Triton):
  python whole_model.py --quick --json out.json      every expect="pass" check, reduced sizes
  python whole_model.py --json results-a40.json      full mode
  python whole_model.py --only l2,modes --variants dense --json part.json
  python whole_model.py --merge part1.json part2.json --json results-a40.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, HERE)
from bi_check import bit_equal, bits, check_batch, first_mismatch  # noqa: E402
from bi_kernels import attention_fixed_blocks, attention_fixed_blocks_train, mm_one_config  # noqa: E402
from bi_report import Results  # noqa: E402
import model as wm  # noqa: E402

dev = "cuda"
BF = torch.bfloat16
SECTIONS = ("kernels", "spread", "l1", "l2", "modes", "lengths", "backward", "rounding", "cost")
SIZES_FULL = (2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65)
SIZES_QUICK = (2, 3, 8, 17, 33)


def log(*a):
    print(*a, flush=True)


def gpu_state() -> dict:
    try:
        util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.split()
        return {"util_pct_mem_mib": util, "compute_procs": len(apps)}
    except Exception as ex:  # noqa: BLE001
        return {"error": str(ex)}


def median_ms(fn, warmup=2, runs=5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


# ----------------------------------------------------------------------------- comparing runs
def reference(model, ops, ids):
    """The trainer forward of one sequence alone: logits [T, V] and realized log-probs [T-1]."""
    lg = model.prefill(ids[None], ops=ops)
    return {"logits": lg[0], "rlp": wm.realized_logprob(model.logprobs(lg), ids[None])[0]}


def compare(ref, entries, T) -> dict:
    """entries: (start, logits [n, V], realized logprob [m <= n]) pieces of one sequence, possibly
    overlapping (recomputed positions). Counts mismatching bits against the reference."""
    st = {"positions": 0, "logit_elems": 0, "logit_mism": 0, "pos_mism": 0, "lp_n": 0, "lp_mism": 0,
          "max_dlp": 0.0, "max_dlogit": 0.0, "kl_sum": 0.0, "kl_max": 0.0}
    covered = torch.zeros(T, dtype=torch.bool)
    for start, lg, rlp in entries:
        n = lg.shape[0]
        r = ref["logits"][start:start + n]
        covered[start:start + n] = True
        ne = (bits(r) != bits(lg)) if r.dtype == lg.dtype else (r.float() != lg.float())
        rows = ne.any(-1)
        st["positions"] += n
        st["logit_elems"] += ne.numel()
        st["logit_mism"] += int(ne.sum())
        st["pos_mism"] += int(rows.sum())
        if bool(rows.any()):
            a = torch.log_softmax(r[rows].double(), -1)
            b = torch.log_softmax(lg[rows].double(), -1)
            kl = (a.exp() * (a - b)).sum(-1)
            st["kl_sum"] += float(kl.sum())
            st["kl_max"] = max(st["kl_max"], float(kl.max()))
            st["max_dlogit"] = max(st["max_dlogit"], float((r[rows].float() - lg[rows].float()).abs().max()))
        m = rlp.numel()
        if m:
            rr = ref["rlp"][start:start + m]
            st["lp_n"] += m
            st["lp_mism"] += int((bits(rr) != bits(rlp)).sum())
            st["max_dlp"] = max(st["max_dlp"], float((rr.double() - rlp.double()).abs().max()))
    st["covered"] = bool(covered.all())
    return st


def merge_stats(stats: List[dict]) -> dict:
    out = {k: 0 for k in ("positions", "logit_elems", "logit_mism", "pos_mism", "lp_n", "lp_mism")}
    out.update(max_dlp=0.0, max_dlogit=0.0, kl_sum=0.0, kl_max=0.0, covered=True, runs=len(stats), runs_differing=0)
    for s in stats:
        for k in ("positions", "logit_elems", "logit_mism", "pos_mism", "lp_n", "lp_mism"):
            out[k] += s[k]
        out["kl_sum"] += s["kl_sum"]
        for k in ("max_dlp", "max_dlogit", "kl_max"):
            out[k] = max(out[k], s[k])
        out["covered"] = out["covered"] and s["covered"]
        out["runs_differing"] += int(s["logit_mism"] > 0 or s["lp_mism"] > 0)
    return out


def equal(st: dict) -> bool:
    return st["logit_mism"] == 0 and st["lp_mism"] == 0 and st["covered"]


def summary(st: dict) -> dict:
    """The numbers quoted in the case page."""
    return {"equal": equal(st), "runs": st.get("runs", 1), "runs_differing": st.get("runs_differing", int(not equal(st))),
            "frac_logits_differ": st["logit_mism"] / max(st["logit_elems"], 1),
            "frac_tokens_any_logit_differs": st["pos_mism"] / max(st["positions"], 1),
            "frac_tokens_logprob_differs": st["lp_mism"] / max(st["lp_n"], 1),
            "max_abs_dlogprob": st["max_dlp"], "max_abs_dlogit": st["max_dlogit"],
            "kl_mean": st["kl_sum"] / max(st["positions"], 1), "kl_max": st["kl_max"],
            "tokens": st["positions"], "covered": st["covered"]}


def record(res, name, st, level, impl, expect, extra=None):
    d = summary(st)
    if extra:
        d.update(extra)
    res.check(name, d["equal"], level=level, impl=impl, expect=expect, detail=d)
    tag = "equal" if d["equal"] else (f"differs: logprob of {d['frac_tokens_logprob_differs']:.1%} of tokens, "
                                      f"max|dlogp| {d['max_abs_dlogprob']:.3g}, KL mean {d['kl_mean']:.3g} max {d['kl_max']:.3g}")
    log(f"  [{impl}] {level} {name}: {tag}")
    return d["equal"]


# ----------------------------------------------------------------------------- data
def sample_sequences(model, n, T, seed, ops="bi"):
    """n sequences of T tokens sampled from the model itself through the KV cache (first token
    uniform): the realized tokens of a rollout."""
    V = model.cfg.vocab_size
    g = torch.Generator(device=dev).manual_seed(seed)
    cache = wm.KVCache(model, n, T)
    tok = torch.randint(0, V, (n,), generator=g, device=dev)
    seq = [tok]
    for _ in range(T - 1):
        outs = model.forward_cached(cache, list(range(n)), [tok[i:i + 1] for i in range(n)], ops=ops)
        tok = torch.multinomial(torch.softmax(torch.cat(outs).float(), -1), 1, generator=g).squeeze(1)
        seq.append(tok)
    return torch.stack(seq, 1)


def filler_ids(kind, n, T, V, gen, target):
    if kind == "const":
        return torch.full((n, T), 7, device=dev, dtype=torch.long)
    if kind == "copies":
        return target[None, :T].expand(n, T).clone()
    return torch.randint(0, V, (n, T), generator=gen, device=dev)


def prefill_in_batch(model, ops, ids, size, pos, kind, gen):
    """The target at row p of a trainer batch of `size` sequences. Log-softmax runs on the whole batch."""
    T = ids.numel()
    p = {"first": 0, "middle": size // 2, "last": size - 1}[pos]
    others = filler_ids(kind, size - 1, T, model.cfg.vocab_size, gen, ids)
    batch = torch.cat([others[:p], ids[None], others[p:]])
    lg = model.prefill(batch, ops=ops)
    rlp = wm.realized_logprob(model.logprobs(lg), batch)[p]
    return [(0, lg[p], rlp)]


# ----------------------------------------------------------------------------- the engine
@dataclass(eq=False)
class Req:
    ids: torch.Tensor                 # every token of the request, teacher-forced
    arrive: int                       # first engine step it takes part in
    plan: list                        # per active step: tokens fed, or (fed, kept) for a speculative append
    name: Optional[str] = None        # recorded when set
    prefix: Optional[tuple] = None    # (source Req, n): start from the source's first n cached positions
    slot: Optional[int] = None
    fed: int = 0
    step_i: int = 0
    done: bool = False
    out: list = field(default_factory=list)


def engine_run(model, ops, reqs: List[Req], slots: int, capacity: int, seed=0, poison=False, hold=()):
    """A serving loop with teacher-forced tokens. At each step every admitted, unfinished request
    feeds its next piece; rows are shuffled; finished requests free their slot (requests in `hold`
    keep theirs, to serve as prefix sources). Returns the step count; recorded requests carry
    (start, logits, realized logprob) entries in .out."""
    cache = wm.KVCache(model, slots, capacity, poison=poison)
    rng = random.Random(seed)
    free = list(range(slots))
    rng.shuffle(free)
    step, sizes = 0, []
    while not all(r.done for r in reqs):
        batch = []
        for r in reqs:
            if r.done or r.arrive > step:
                continue
            if r.slot is None:
                if r.prefix is not None and r.prefix[0].fed < r.prefix[1]:
                    continue                                    # its prefix is not cached yet
                r.slot = free.pop()
                if r.prefix is not None:
                    cache.copy_prefix(r.slot, r.prefix[0].slot, r.prefix[1])
                    r.fed = r.prefix[1]
            item = r.plan[r.step_i]
            fed, kept = item if isinstance(item, tuple) else (item, item)
            batch.append((r, fed, kept))
        if not batch:
            step += 1
            continue
        rng.shuffle(batch)
        outs = model.forward_cached(cache, [r.slot for r, _, _ in batch],
                                    [r.ids[r.fed:r.fed + fed] for r, fed, _ in batch], ops=ops)
        named = [i for i, (r, _, _) in enumerate(batch) if r.name]
        if named:
            lp = torch.split(model.logprobs(torch.cat(outs)), [o.shape[0] for o in outs])
        for i, (r, fed, kept) in enumerate(batch):
            if r.name:
                nxt = r.ids[r.fed + 1:r.fed + fed + 1]
                rlp = lp[i][:nxt.numel()].gather(1, nxt[:, None]).squeeze(1)
                r.out.append((r.fed, outs[i], rlp))
            r.fed += kept
            r.step_i += 1
            if kept < fed:                                      # rejected speculative tokens leave the cache
                cache.truncate(r.slot, r.fed)
            if r.step_i == len(r.plan):
                assert r.fed == r.ids.numel(), (r.fed, r.ids.numel())
                r.done = True
                if not any(r is h for h in hold):
                    cache.reset(r.slot)
                    free.append(r.slot)
        sizes.append(len(batch))
        step += 1
    return {"steps": step, "batch_sizes": sorted(set(sizes)), "max_rows": max(sizes)}


def chunk_plan(T, c):
    return [c] * (T // c) + ([T % c] if T % c else [])


def spec_plan(T, prompt):
    """A prompt chunk, then speculative appends of k = 1..8 tokens. Every third append keeps only
    part of its tokens: the rest are rolled back (cache.truncate) and fed again."""
    plan, fed, k, j = [prompt], prompt, 1, 0
    while fed < T:
        n = min(k, T - fed)
        keep = n if (j % 3 != 2 or n == 1) else max(1, n // 2)
        plan.append((n, keep))
        fed += keep
        k = k % 8 + 1
        j += 1
    return plan


def crowd(target_steps, sizes, V, gen_cpu, start=0, max_tokens=400):
    """Filler requests around a target that is active for `target_steps` steps from `start`. The
    number of rows per step walks up through `sizes` and back down, so requests join and finish
    while the target runs; no filler lives longer than 120 steps, so slots are reused. Fillers
    decode, prefill then decode, or prefill in chunks, on random or constant tokens."""
    def rnd(lo, hi):
        return int(torch.randint(lo, hi, (1,), generator=gen_cpu))
    prof = list(sizes) + list(reversed(sizes[:-1]))
    want = [prof[i * len(prof) // target_steps] - 1 for i in range(target_steps)]
    active, spans = [], []                                      # arrival steps; (arrive, leave)
    for t, n in enumerate(want):
        while len(active) < n:
            active.append(start + t)
        while len(active) > n:
            spans.append((active.pop(rnd(0, len(active))), start + t))
    spans += [(a, start + target_steps + rnd(0, 4)) for a in active]
    reqs = []
    for arrive, leave in spans:
        while arrive < leave:                                   # split long lives into back-to-back requests
            steps = min(leave - arrive, rnd(20, min(120, max_tokens) + 1))
            kind = rnd(0, 10)
            if kind < 6:
                plan = [1] * steps
            elif kind < 8:
                plan = [rnd(2, max(3, min(130, max_tokens - steps + 1)))] + [1] * (steps - 1)
            else:
                plan = [min((7, 16)[kind - 8], max(1, max_tokens // steps))] * steps
            n = sum(plan)
            assert n <= max_tokens
            ids = torch.full((n,), 7, device=dev) if kind == 5 else torch.randint(0, V, (n,), generator=gen_cpu).to(dev)
            reqs.append(Req(ids, arrive, plan))
            arrive += steps
    return reqs


def run_mode(model, ops, ids, plan, busy_sizes=None, seed=0, poison=False, capacity=None):
    """The target through the engine with feed plan `plan`, alone or inside a crowd."""
    T = ids.numel()
    tgt = Req(ids, 0, plan, name="target")
    reqs = [tgt]
    cap = max(capacity or T, 64)
    if busy_sizes:
        cap = max(cap, 400)
        reqs += crowd(len(plan), busy_sizes, model.cfg.vocab_size, torch.Generator().manual_seed(seed), max_tokens=400)
    slots = (max(busy_sizes) + 8) if busy_sizes else 2
    info = engine_run(model, ops, reqs, slots, cap, seed=seed, poison=poison)
    return tgt.out, info


def compare_runs(a, b) -> dict:
    """Two runs of the same feed plan, piece by piece (a is the reference)."""
    assert len(a) == len(b)
    sts = []
    for (s, lg, rl), (s2, lg2, rl2) in zip(a, b):
        assert s == s2 and lg.shape == lg2.shape
        sts.append(compare({"logits": lg, "rlp": rl}, [(0, lg2, rl2)], lg.shape[0]))
    st = merge_stats(sts)
    st.pop("runs"), st.pop("runs_differing")
    return st


# ----------------------------------------------------------------------------- sections
def sec_kernels(res, cfg, quick):
    """Kernel facts the model relies on. Stock ops: expect fail where variance is documented."""
    g = torch.Generator(device=dev).manual_seed(1)
    sizes = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 1024, 2048)
    if quick:
        sizes = (1, 2, 3, 4, 8, 9, 16, 17, 33, 65, 257)
    H = cfg.hidden
    rows = torch.randn(8 if quick else 16, H, generator=g, device=dev).to(BF)
    w = (1 + 0.1 * torch.randn(H, generator=g, device=dev)).to(BF)

    # 1. the RMSNorm variance: torch's reduction against the fixed-tree row sum
    def var_torch(x):
        return x.float().pow(2).mean(-1, keepdim=True)

    def var_bi(x):
        return wm.row_sum(x.float(), square=True) * (1.0 / H)

    def norm(var):
        return lambda x: (x.float() * torch.rsqrt(var(x) + 1e-6)).to(BF) * w
    rep = check_batch(var_torch, rows, sizes, name="torch mean(x^2), fp32 variance")
    fail_sizes = sorted({f["size"] for f in rep.failures})
    res.check(f"RMSNorm variance mean(x^2) in fp32, hidden {H}: a row alone vs in batches of {sizes[0]}..{sizes[-1]} rows",
              rep.passed, level="L2", impl="torch reduction", expect="fail",
              detail={"checked": rep.checked, "failures": len(rep.failures), "failing_sizes": fail_sizes})
    log(f"  torch variance fp32: {rep.summary()}")
    rep = check_batch(var_bi, rows, sizes, name="row_sum")
    res.check(f"RMSNorm variance by the fixed-tree row sum, hidden {H}: rows 1..{sizes[-1]}", rep.passed, level="L2",
              impl="bi row_sum", detail=rep.to_dict())
    log(f"  bi row_sum fp32: {rep.summary()}")
    rep = check_batch(norm(var_torch), rows, sizes, name="torch rmsnorm bf16")
    res.check(f"RMSNorm bf16 output from torch ops, hidden {H}, {rows.shape[0]} probe rows (the rounding to bf16 hides "
              f"most fp32 differences)", rep.passed, level="L2", impl="torch reduction", expect="info",
              detail={"checked": rep.checked, "failures": len(rep.failures),
                      "failing_sizes": sorted({f["size"] for f in rep.failures})})
    rep = check_batch(norm(var_bi), rows, sizes, name="bi rmsnorm bf16")
    res.check(f"RMSNorm bf16 output with the fixed-tree row sum, hidden {H}", rep.passed, level="L2", impl="bi row_sum",
              detail=rep.to_dict())
    # many rows: each alone (a decode step of one sequence) against all in one batch (a trainer batch)
    N = 2000 if quick else 20000
    big = torch.randn(N, H, generator=g, device=dev).to(BF)
    for name, var, expect in (("torch reduction", var_torch, "fail"), ("bi row_sum", var_bi, "pass")):
        vb, ob = var(big), norm(var)(big)
        va = torch.cat([var(big[i:i + 1]) for i in range(N)])
        oa = torch.cat([norm(var)(big[i:i + 1]) for i in range(N)])
        rows_var = int((bits(va) != bits(vb)).any(-1).sum())
        el = int((bits(oa) != bits(ob)).sum())
        rows_out = int((bits(oa) != bits(ob)).any(-1).sum())
        res.check(f"RMSNorm, hidden {H}: {N} rows each alone vs all in one batch (variance and bf16 output)",
                  rows_var == 0 and el == 0, level="L2", impl=name, expect=expect,
                  detail={"rows": N, "rows_variance_differs": rows_var, "rows_output_differs": rows_out,
                          "output_elements_differ": el, "output_elements": oa.numel()})
        log(f"  {name}: {N} rows alone vs batched: variance differs in {rows_var} rows, bf16 output in {el} of "
            f"{oa.numel()} elements ({rows_out} rows)")
    # cost of the row sum
    xb = torch.randn(16384, H, generator=g, device=dev).to(BF)
    x1 = xb[:1].contiguous()
    for label, x in (("16384 rows", xb), ("1 row", x1)):
        res.measure(f"RMSNorm {label} hidden {H}", round(median_ms(lambda: norm(var_torch)(x), runs=7), 4), "ms", impl="torch reduction")
        res.measure(f"RMSNorm {label} hidden {H}", round(median_ms(lambda: norm(var_bi)(x), runs=7), 4), "ms", impl="bi row_sum")

    # 2. log_softmax and the row sum at other widths
    lr = (torch.randn(4, cfg.vocab_size, generator=g, device=dev) * 4).to(BF)
    rep = check_batch(lambda x: torch.log_softmax(x.float(), -1), lr, [s for s in sizes if s <= 257], name="log_softmax")
    res.check(f"log_softmax fp32, vocab {cfg.vocab_size}: rows 1..257", rep.passed, level="L2", impl="torch", expect="info",
              detail=rep.to_dict())
    for K in (8, 100, 129, 1000) + (() if quick else (896, 4096)):
        xr = torch.randn(8, K, generator=g, device=dev)
        rep = check_batch(lambda x: wm.row_sum(x, square=True), xr, sizes, name="row_sum")
        res.check(f"fixed-tree row sum of squares, width {K}: rows 1..{sizes[-1]}", rep.passed, level="L2", impl="bi row_sum",
                  detail=rep.to_dict())

    # 3. the GEMM at this model's shapes
    c = cfg
    shapes = {(c.hidden, c.heads * c.head_dim), (c.hidden, c.kv_heads * c.head_dim), (c.hidden, c.intermediate),
              (c.intermediate, c.hidden), (c.hidden, c.vocab_size)}
    gsizes = [s for s in sizes if s <= (257 if quick else 2048)]
    for K, N in sorted(shapes):
        W = (torch.randn(K, N, generator=g, device=dev) * K ** -0.5).to(BF)
        rr = torch.randn(2, K, generator=g, device=dev).to(BF)
        sz = [s for s in gsizes if s * N <= 2 ** 27]
        rep = check_batch(lambda x: mm_one_config(x, W), rr, sz, name="mm_one_config")
        res.check(f"GEMM {K}->{N}: M = 1..{sz[-1]}", rep.passed, level="L2", impl="bi mm_one_config", detail=rep.to_dict())
        rep2 = check_batch(lambda x: x @ W, rr, sz, name="cublas")
        res.check(f"GEMM {K}->{N}: M = 1..{sz[-1]}", rep2.passed, level="L2", impl="torch matmul (cuBLAS)", expect="fail",
                  detail={"checked": rep2.checked, "failures": len(rep2.failures),
                          "failing_sizes": sorted({f["size"] for f in rep2.failures})})
        log(f"  GEMM {K}->{N}: bi {rep.summary()} | cuBLAS {rep2.summary()}")
        xx = torch.randn(300, K, generator=g, device=dev).to(BF)
        same = bit_equal(wm.mm_one_config_f32out(xx, W).to(BF), mm_one_config(xx, W))
        res.check(f"GEMM {K}->{N}: the fp32-output variant cast to bf16 equals mm_one_config", same, level="other",
                  impl="bi mm_one_config_f32out")

    # 4. the copied per-sequence attention against tools' attention_fixed_blocks
    Hq, Hk, D = c.heads, c.kv_heads, c.head_dim
    for T in ((65, 200) if quick else (1, 63, 64, 65, 200, 513)):
        q = torch.randn(3, Hq, T, D, generator=g, device=dev).to(BF)
        k = torch.randn(3, Hk, T, D, generator=g, device=dev).to(BF)
        v = torch.randn(3, Hk, T, D, generator=g, device=dev).to(BF)
        ref = attention_fixed_blocks(q, k, v, 0)
        qp = q.transpose(1, 2).reshape(3 * T, Hq, D).contiguous()
        kc, vc = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()        # [3, T, Hk, D]: a slot cache
        cu = torch.arange(4, device=dev, dtype=torch.int32) * T
        z = torch.zeros(3, device=dev, dtype=torch.int32)
        o1 = wm._serve_launch(qp, kc, vc, cu, z, T, wm.MODE_PAGED, block_table=torch.arange(3, device=dev, dtype=torch.int32).view(3, 1), page=T)
        o2 = wm._serve_launch(qp, kc.reshape(3 * T, Hk, D), vc.reshape(3 * T, Hk, D), cu, z, T, wm.MODE_VARLEN, cu_k=cu)
        want = ref.transpose(1, 2).reshape(3 * T, Hq, D)
        tr = attention_fixed_blocks_train(q, k, v)
        ok = bit_equal(o1, want) and bit_equal(o2, want) and bit_equal(tr, ref)
        res.check(f"attention T={T}, GQA {Hq}/{Hk}, D={D}: slot-cache and packed kernels (copied) and the training "
                  f"forward equal attention_fixed_blocks", ok, level="L3", impl="bi attention")
        log(f"  attention copies T={T}: {'equal' if ok else 'DIFFER'}")


def sec_spread(res, model, targets, impl):
    ids = targets[:, :min(targets.shape[1], 512)]
    lg = model.prefill(ids[:4])
    lp = model.logprobs(lg)
    p = lp.exp()
    ent = -(p * lp).sum(-1)
    rl = wm.realized_logprob(lp, ids[:4])
    out = {"logit_std": float(lg.float().std()), "logit_abs_max": float(lg.float().abs().max()),
           "entropy_mean_nats": float(ent.mean()), "top1_prob_mean": float(p.max(-1)[0].mean()),
           "top1_prob_median": float(p.max(-1)[0].median()),
           "realized_logprob_mean": float(rl.mean()), "realized_logprob_median": float(rl.median()),
           "realized_logprob_min": float(rl.min()), "uniform_logprob": -math.log(model.cfg.vocab_size)}
    for k, v in out.items():
        res.measure(f"spread: {k}", round(v, 4), "", impl=impl)
    log(f"  spread [{impl}]: {out}")


def sec_l1(res, model, ops, impl, targets, quick, expect):
    T = 100 if quick else 200
    ids = targets[:4, :T].contiguous()
    ref = model.prefill(ids, ops=ops)
    ok = all(bit_equal(ref, model.prefill(ids, ops=ops)) for _ in range(4))
    res.check(f"trainer forward B=4 T={T}, 5 runs", ok, level="L1", impl=impl, expect=expect)
    sizes = (1, 2, 3, 8) if quick else (1, 2, 3, 8, 9, 17, 33)
    plan = [1] * T
    runs = [run_mode(model, ops, ids[0], plan, busy_sizes=sizes, seed=5)[0] for _ in range(3)]
    ok2 = all(bit_equal(torch.cat([e[1] for e in runs[0]]), torch.cat([e[1] for e in r])) for r in runs[1:])
    res.check(f"engine decode T={T} in a changing batch (1..{sizes[-1]} rows), 3 runs", ok2, level="L1", impl=impl, expect=expect)
    log(f"  [{impl}] L1 trainer {'pass' if ok else 'FAIL'}, engine {'pass' if ok2 else 'FAIL'}")


def sec_l2(res, model, ops, impl, targets, quick, expect, small):
    """Trainer forward: the target alone against co-batched."""
    V = model.cfg.vocab_size
    gen = torch.Generator(device=dev).manual_seed(21)
    sizes = SIZES_QUICK if quick else SIZES_FULL
    plans = [(100 if quick else 200, sizes, ("first", "middle", "last"), ("random", "const") if quick else ("random", "const", "copies"))]
    if not quick:
        plans.append((65, (2, 3, 8, 17, 33, 65), ("first", "last"), ("random",)))
        if not small:
            plans.append((2048, (2, 3, 4), ("first", "last"), ("random",)))
    for T, szs, poss, kinds in plans:
        ids = targets[1 if T == 65 else 0, :T].contiguous()
        ref = reference(model, ops, ids)
        per_size, allst = {}, []
        for size in szs:
            sts = []
            for pos in poss:
                for kind in kinds:
                    sts.append(compare(ref, prefill_in_batch(model, ops, ids, size, pos, kind, gen), T))
            per_size[size] = summary(merge_stats(sts))
            allst += sts
        st = merge_stats(allst)
        first_fail = next((s for s in szs if not per_size[s]["equal"]), None)
        record(res, f"trainer forward, T={T}: alone vs co-batched, B in {list(szs)}, positions {list(poss)}, fillers {list(kinds)}",
               st, "L2", impl, expect,
               {"first_failing_batch": first_fail,
                "per_batch_size": {str(s): {k: per_size[s][k] for k in ("equal", "frac_logits_differ", "frac_tokens_logprob_differs",
                                                                         "max_abs_dlogprob", "kl_mean", "kl_max")} for s in szs}})
        del ref
    torch.cuda.empty_cache()


def sec_modes(res, model, ops, impl, targets, quick, expect, small):
    """The engine's forwards against the trainer forward of the same sequence."""
    V = model.cfg.vocab_size
    T = 100 if quick else 200
    ids = targets[0, :T].contiguous()
    ref = reference(model, ops, ids)
    poison = ops == "bi"
    busy = (1, 2, 3, 8, 9, 17) if quick else (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65)
    if small and not quick:
        busy = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33)
    modes = {"decode, one token per step": [1] * T}
    for c in ((7, 64) if quick else (7, 16, 64, 100)):
        modes[f"chunked prefill, chunk {c}"] = chunk_plan(T, c)
    modes["speculative appends k=1..8 with rollback, after a 40-token prompt"] = spec_plan(T, 40)
    alone, l3, l2 = {}, [], []
    for name, plan in modes.items():
        out, _ = run_mode(model, ops, ids, plan, poison=poison)
        alone[name] = out
        st = compare(ref, out, T)
        l3.append(st)
        record(res, f"{name}, alone vs trainer forward (T={T})", st, "L3", impl, expect)
    for name, plan in modes.items():
        sts, info = [], None
        for seed in ((3,) if quick else (3, 4)):
            out, info = run_mode(model, ops, ids, plan, busy_sizes=busy, seed=seed, poison=poison)
            sts.append(compare_runs(alone[name], out))
            l3.append(compare(ref, out, T))
        st = merge_stats(sts)
        l2.append(st)
        record(res, f"{name}: in a changing batch vs alone (T={T})", st, "L2", impl, expect,
               {"batch_sizes_seen": info["batch_sizes"], "engine_steps": info["steps"]})
    del alone
    torch.cuda.empty_cache()

    # prefix-cache hit: the prefix KV was computed inside another request's prefill, in another batch
    sts = []
    for n_pre, seed in (((64, 7),) if quick else ((64, 7), (150, 8), (1, 9))):
        g = torch.Generator().manual_seed(seed)
        src = Req(torch.cat([ids[:n_pre], torch.randint(0, V, (30,), generator=g).to(dev)]), 0, [n_pre] + [1] * 30, name="src")
        others = [Req(torch.randint(0, V, (n_pre + 9,), generator=g).to(dev), 0, [n_pre + 9]) for _ in range(4)]
        tgt = Req(ids, 2, chunk_plan(T - n_pre, 33), name="target", prefix=(src, n_pre))
        late = crowd(len(tgt.plan), (2, 3, 5), V, g, start=2, max_tokens=200)
        engine_run(model, ops, [src] + others + [tgt] + late, 16, max(T, 256), seed=seed, poison=poison, hold=(src,))
        pre = [(s, lg[:n_pre - s], rl[:max(0, n_pre - s - 1)]) for s, lg, rl in src.out if s < n_pre]
        # the logprob at the last prefix position scores the target's next token, not the source's
        sts.append(compare(ref, pre + tgt.out, T))
    st = merge_stats(sts)
    l3.append(st)
    record(res, f"prefix-cache hit (prefix computed in another request's batch, then the suffix) vs trainer forward (T={T})",
           st, "L3", impl, expect)

    # preempt and recompute from scratch
    sts = []
    for p, seed in (((57, 11),) if quick else ((57, 11), (129, 12))):
        g = torch.Generator().manual_seed(seed)
        first = Req(ids[:p], 1, [1] * p, name="before")
        again = Req(ids, p + 6, [p] + [1] * (T - p), name="after")
        late = crowd(p + 6 + len(again.plan), (3, 2, 5, 9, 4), V, g, start=0, max_tokens=250)
        engine_run(model, ops, [first, again] + late, 16, max(T, 256), seed=seed, poison=poison)
        sts.append(compare(ref, first.out + again.out, T))
    st = merge_stats(sts)
    l3.append(st)
    record(res, f"preempt after p tokens, recompute from scratch in a busy batch, continue vs trainer forward (T={T})",
           st, "L3", impl, expect)

    # right padding and packed varlen
    g = torch.Generator(device=dev).manual_seed(31)
    sts = []
    for Tpad, n, p in (((128, 3, 1),) if quick else ((T + 1, 2, 0), (256, 4, 3), (320, 3, 1))):
        batch = torch.randint(0, V, (n, Tpad), generator=g, device=dev)
        batch[p, :T] = ids
        batch[p, T:] = 0                                        # pad id
        lg = model.prefill(batch, ops=ops)
        rlp = wm.realized_logprob(model.logprobs(lg), batch)[p, :T - 1]
        sts.append(compare(ref, [(0, lg[p, :T], rlp)], T))
    st = merge_stats(sts)
    l3.append(st)
    record(res, f"right-padded to {'128' if quick else str(T + 1) + ', 256, 320'} in a trainer batch vs alone (T={T})", st, "L2", impl, expect)
    sts = []
    gc = torch.Generator().manual_seed(32)
    for layout in (((37, None, 130),) if quick else ((37, None, 130), (None, 1, 64, 65), (300, 200, None), (63, 64, None, 129, 7))):
        seqs = [ids if L is None else torch.randint(0, V, (L,), generator=gc).to(dev) for L in layout]
        outs = model.prefill_packed(seqs, ops=ops)
        i = layout.index(None)
        lpk = torch.split(model.logprobs(torch.cat(outs)), [o.shape[0] for o in outs])[i]
        sts.append(compare(ref, [(0, outs[i], lpk[:-1].gather(1, ids[1:, None]).squeeze(1))], T))
    st = merge_stats(sts)
    l3.append(st)
    record(res, f"packed varlen prefill among neighbours of other lengths vs trainer forward (T={T})", st, "L3", impl, expect)

    # what RL sees: a trainer batch against the engine's changing batch
    eng, info = run_mode(model, ops, ids, [1] * T, busy_sizes=busy, seed=17, poison=poison)
    pair = compare({"logits": torch.cat([e[1] for e in eng]), "rlp": torch.cat([e[2] for e in eng])},
                   prefill_in_batch(model, ops, ids, 16, "middle", "random", g), T)
    record(res, f"trainer (row 8 of a batch of 16) vs engine (decode in a changing batch): the pair RL compares (T={T})",
           pair, "L3", impl, expect, {"engine_batch_sizes": info["batch_sizes"]})

    if not quick and not small:                                 # the longest sequence: T = 2048
        Tl = 2048
        idl = targets[0, :Tl].contiguous()
        refl = reference(model, ops, idl)
        for name, plan, bz in (("decode, one token per step", [1] * Tl, (1, 2, 3, 4, 8, 9)),
                               ("chunked prefill, chunk 100", chunk_plan(Tl, 100), None),
                               ("chunked prefill, chunk 64", chunk_plan(Tl, 64), (2, 3, 5))):
            out, info = run_mode(model, ops, idl, plan, busy_sizes=bz, seed=41, poison=poison, capacity=Tl)
            record(res, f"{name}{', in a changing batch' if bz else ', alone'} vs trainer forward (T={Tl})",
                   compare(refl, out, Tl), "L3", impl, expect, {"batch_sizes_seen": info["batch_sizes"]})
            del out
        del refl
    torch.cuda.empty_cache()


def sec_lengths(res, model, ops, impl, targets, quick, expect):
    """Lengths around the 64-token attention block: trainer forward against decode and chunks."""
    lens = (63, 64, 65) if quick else (1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 193, 257)
    sts, bad = [], []
    for T in lens:
        ids = targets[2, :T].contiguous()
        ref = reference(model, ops, ids)
        for plan in ([1] * T, chunk_plan(T, 7), chunk_plan(T, 64)):
            st = compare(ref, run_mode(model, ops, ids, plan, poison=(ops == "bi"))[0], T)
            sts.append(st)
            if not equal(st):
                bad.append(T)
    record(res, f"lengths {list(lens)}: decode, chunk 7 and chunk 64 vs trainer forward", merge_stats(sts), "L3", impl, expect,
           {"failing_lengths": sorted(set(bad))})


# ---- backward
TAPS = ("embed_tokens", "input_layernorm", "self_attn.core", "self_attn", "post_attention_layernorm", "mlp", "norm", "lm_head")


def grads_of(model, ops, batch, p, all_rows):
    """Loss = -(sum of realized log-probs) of row p (or of every row). Returns d loss / d activation
    for the tapped module outputs, row p only, and the parameter gradients."""
    taps, hooks = {}, []

    def hook(name):
        def fn(mod, args, out):
            out.retain_grad()
            taps[name] = out
        return fn
    for name, mod in model.named_modules():
        if name.endswith(TAPS) and (".layers." not in name or name.split(".")[2] in ("0", str(model.cfg.layers - 1))):
            hooks.append(mod.register_forward_hook(hook(name)))
    for q in model.parameters():
        q.grad = None
    try:
        with torch.enable_grad():
            lg = model.prefill(batch, ops=ops)
            rl = wm.realized_logprob(model.logprobs(lg), batch)
            loss = -(rl.sum() if all_rows else rl[p].sum())
            loss.backward()
    finally:
        for h in hooks:
            h.remove()
    acts = {n: t.grad[p].detach().clone() for n, t in taps.items()}
    acts["logits(value)"] = lg[p].detach().clone()
    params = {n: q.grad.detach().clone() for n, q in model.named_parameters() if q.grad is not None}
    return acts, params


def sec_backward(res, model, ops, impl, targets, quick, expect, small):
    V = model.cfg.vocab_size
    T = 96 if quick else 200
    ids = targets[0, :T].contiguous()
    gen = torch.Generator(device=dev).manual_seed(51)
    for q in model.parameters():
        q.requires_grad_(True)
    try:
        ref, _ = grads_of(model, ops, ids[None], 0, False)
        sizes = (2, 3, 8) if (quick or small) else (2, 3, 8, 16)
        for all_rows in (False, True):
            bad, worst, checked = {}, 0.0, 0
            for size in sizes:
                for pos in ("first", "last"):
                    p = 0 if pos == "first" else size - 1
                    others = torch.randint(0, V, (size - 1, T), generator=gen, device=dev)
                    batch = torch.cat([others[:p], ids[None], others[p:]])
                    got, _ = grads_of(model, ops, batch, p, all_rows)
                    for n in ref:
                        checked += 1
                        mm = first_mismatch(ref[n], got[n])
                        if mm:
                            frac = mm["mismatched_elements"] / mm["of"]
                            bad.setdefault(n, []).append(size)
                            worst = max(worst, frac)
            what = "every sequence's loss summed" if all_rows else "the loss of that sequence only"
            res.check(f"backward, T={T}: d loss / d activations of a sequence, alone vs co-batched (B in {list(sizes)}, "
                      f"first/last), {what}; {len(ref)} tensors (activation gradients and logits)", not bad, level="L2",
                      impl=impl, expect=expect,
                      detail={"comparisons": checked, "tensors": sorted(ref), "tensors_differing": {k: sorted(set(v)) for k, v in bad.items()},
                              "worst_fraction_of_elements": worst})
            log(f"  [{impl}] backward L2 ({what}): {'equal' if not bad else 'differs in ' + str(len(bad)) + ' tensors, worst ' + format(worst, '.2%')}")
        # parameter gradients sum over the batch: run to run only
        B = 4
        batch = targets[:B, :T].contiguous()
        runs = [grads_of(model, ops, batch, 0, True)[1] for _ in range(3 if quick else 5)]
        diff = sorted({n for r in runs[1:] for n in runs[0] if not bit_equal(runs[0][n], r[n])})
        res.check(f"backward, B={B} T={T}: parameter gradients, {len(runs)} runs ({len(runs[0])} tensors)", not diff,
                  level="L1", impl=impl, expect="pass" if expect == "pass" else "info",
                  detail={"params": len(runs[0]), "params_differing": len(diff), "examples": diff[:6]})
        log(f"  [{impl}] backward L1 parameter gradients: {'equal' if not diff else str(len(diff)) + ' of ' + str(len(runs[0])) + ' tensors differ'}")
    finally:
        for q in model.parameters():
            q.requires_grad_(False)
            q.grad = None
    torch.cuda.empty_cache()


# ---- rounding points
ROUNDING = (("norm", "weight_then_cast", "RMSNorm: weight multiplies in fp32 before the cast"),
            ("norm", "all_dt", "RMSNorm: squares, mean, rsqrt and scale in bf16"),
            ("residual", "fp32_into_norm", "residual add in fp32 feeding the next norm unrounded (fused add+norm)"),
            ("residual", "fp32_stream", "residual stream kept in fp32"),
            ("rope", "fp32", "RoPE: q*cos + rot(q)*sin in fp32 with fp32 tables, one rounding"),
            ("rope", "cpu_table", "RoPE: cos/sin tables computed on the CPU"),
            ("rope", "inv_freq_dt", "RoPE: inv_freq rounded to bf16 (what model.to(bfloat16) does in transformers)"),
            ("lm_head", "fp32", "lm_head output kept in fp32 (no cast to bf16)"),
            ("logprob", "dt", "log_softmax in bf16"))


def sec_rounding(res, model, impl, targets, quick):
    T = 128 if quick else 256
    ids = targets[:2 if quick else 4, :T].contiguous()
    one = targets[0, :64 if quick else 100].contiguous()
    with model.using(ops="bi"):
        ref_lg = model.prefill(ids)
        ref_lp = model.logprobs(ref_lg)
        ref_rl = wm.realized_logprob(ref_lp, ids)
        rows = []
        for fld, val, label in ROUNDING:
            with model.using(**{fld: val}):
                lg = model.prefill(ids)
                lp = model.logprobs(lg)
                rl = wm.realized_logprob(lp, ids).float()
                # the variant is itself mode-invariant: decode equals its own trainer forward
                refv = reference(model, "bi", one)
                st = compare(refv, run_mode(model, "bi", one, [1] * one.numel(), poison=True)[0], one.numel())
            changed = int((lg.float() != ref_lg.float()).sum())
            a, b = ref_lp.double(), lp.double()
            kl = (a.exp() * (a - b)).sum(-1)
            row = {"choice": f"{fld}={val}", "what": label, "logits": lg.numel(), "logits_changed": changed,
                   "frac_logits_changed": changed / lg.numel(),
                   "frac_tokens_logprob_changed": float((rl != ref_rl).float().mean()),
                   "max_abs_dlogprob": float((rl.double() - ref_rl.double()).abs().max()),
                   "max_abs_dlogit": float((lg.float() - ref_lg.float()).abs().max()),
                   "kl_mean": float(kl.mean()), "kl_max": float(kl.max())}
            rows.append(row)
            res.check(f"rounding variant {fld}={val}: decode equals its own trainer forward", equal(st), level="L3", impl=impl)
            res.measure(f"rounding: logits changed by {fld}={val}", changed, "elements", impl=impl,
                        **{k: v for k, v in row.items() if k not in ("choice",)})
            log(f"  rounding {fld}={val}: {changed}/{lg.numel()} logits ({row['frac_logits_changed']:.1%}), logprob of "
                f"{row['frac_tokens_logprob_changed']:.1%} of tokens, max|dlogp| {row['max_abs_dlogprob']:.3g}, KL mean {row['kl_mean']:.3g}"
                f" | variant L3 {'pass' if equal(st) else 'FAIL'}")
    res.extra.setdefault("rounding_table", {})[impl] = rows


# ---- cost
def sec_cost(res, model, impl_prefix, quick, small):
    V = model.cfg.vocab_size
    g = torch.Generator(device=dev).manual_seed(61)
    Tp = 128 if quick else 512
    ctx = 128 if quick else 512
    batches = (1, 8) if quick else (1, 8, 64)
    state = {"before": gpu_state()}
    for B in batches:
        if small and B == 64:
            continue
        ids = torch.randint(0, V, (B, Tp), generator=g, device=dev)
        row = {}
        for ops in ("stock", "bi"):
            ms = median_ms(lambda: model.prefill(ids, ops=ops), warmup=2, runs=5)
            row[ops] = B * Tp / ms * 1e3
            res.measure(f"prefill B={B} T={Tp}", round(row[ops], 1), "tokens/s", impl=f"{ops}/{impl_prefix}", ms=round(ms, 3))
        res.measure(f"prefill B={B} T={Tp}: bi time / stock time", round(row["stock"] / row["bi"], 3), "x", impl=f"bi/{impl_prefix}")
        log(f"  cost prefill B={B} T={Tp}: stock {row['stock']:.0f} tok/s, bi {row['bi']:.0f} tok/s, bi/stock time {row['stock'] / row['bi']:.2f}x")
        torch.cuda.empty_cache()
        row = {}
        for ops in ("stock", "bi"):
            steps = 12 if quick else 24
            cache = wm.KVCache(model, B, ctx + steps + 8)
            for s in range(0, B, 16):                           # fill the context in pieces: the logits are large
                sl = list(range(s, min(B, s + 16)))
                model.forward_cached(cache, sl, [torch.randint(0, V, (ctx,), generator=g, device=dev) for _ in sl], ops=ops)
            toks = [torch.randint(0, V, (1,), generator=g, device=dev) for _ in range(B)]
            ts = []
            for i in range(steps + 4):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.forward_cached(cache, list(range(B)), toks, ops=ops)
                torch.cuda.synchronize()
                if i >= 4:
                    ts.append((time.perf_counter() - t0) * 1e3)
            ms = statistics.median(ts)
            row[ops] = B / ms * 1e3
            res.measure(f"decode B={B} context {ctx}", round(row[ops], 1), "tokens/s", impl=f"{ops}/{impl_prefix}", ms_per_step=round(ms, 3))
            del cache
        res.measure(f"decode B={B} context {ctx}: bi time / stock time", round(row["stock"] / row["bi"], 3), "x", impl=f"bi/{impl_prefix}")
        log(f"  cost decode B={B} ctx {ctx}: stock {row['stock']:.0f} tok/s, bi {row['bi']:.0f} tok/s, bi/stock time {row['stock'] / row['bi']:.2f}x")
        torch.cuda.empty_cache()
    state["after"] = gpu_state()
    res.extra.setdefault("gpu_state_cost", {})[impl_prefix] = state


# ----------------------------------------------------------------------------- main
def variant_cfg(name, quick):
    if name == "dense":
        return wm.Config(layers=4) if quick else wm.Config()
    if name == "moe":
        return wm.Config(moe=True, layers=2, moe_experts=16, moe_topk=4) if quick else wm.Config.moe_default()
    if name == "wide":
        return wm.Config.wide()
    if name == "qwen2":                                         # Qwen2.5-0.5B shapes (q/k/v bias, tied), 4 layers
        return wm.Config(**{**wm.Config.qwen2_0p5b().__dict__, "layers": 4, "max_positions": 2048})
    raise ValueError(name)


def run_variant(res, name, sections, quick, seed):
    cfg = variant_cfg(name, quick)
    small = name != "dense"                                     # the other variants run a reduced matrix
    t0 = time.time()
    model = wm.build(cfg, ops="bi", seed=seed)
    res.extra.setdefault("configs", {})[name] = {**cfg.__dict__, "parameters": sum(p.numel() for p in model.parameters())}
    log(f"== {name}: {cfg} ({res.extra['configs'][name]['parameters'] / 1e6:.0f}M parameters, built in {time.time() - t0:.1f}s)")
    need = 128 if quick else (2048 if (not small and {"l2", "modes"} & set(sections)) else 512)
    with torch.no_grad():
        t0 = time.time()
        targets = sample_sequences(model, 4, need, seed=100 + seed)
        res.extra.setdefault("targets", {})[name] = {
            "shape": list(targets.shape), "sampled_with": "bi", "seconds": round(time.time() - t0, 1),
            "sha1": hashlib.sha1(targets.cpu().numpy().tobytes()).hexdigest()[:16]}
        if "kernels" in sections and name != "moe":
            log("-- kernels")
            sec_kernels(res, cfg, quick)
        if "spread" in sections:
            sec_spread(res, model, targets, f"bi/{name}")
        for ops in ("bi", "stock"):
            impl = f"{ops}/{name}"
            exp_fwd = "pass" if ops == "bi" else "fail"
            if quick and ops == "stock":
                continue                                        # quick mode: every expect="pass" check
            if "l1" in sections:
                sec_l1(res, model, ops, impl, targets, quick, "pass" if ops == "bi" else "info")
            if "l2" in sections:
                log(f"-- l2 {impl}")
                sec_l2(res, model, ops, impl, targets, quick, exp_fwd, small)
            if "modes" in sections:
                log(f"-- modes {impl}")
                sec_modes(res, model, ops, impl, targets, quick, exp_fwd, small)
            if "lengths" in sections:
                sec_lengths(res, model, ops, impl, targets, quick, exp_fwd)
        if "rounding" in sections and name == "dense":
            log("-- rounding")
            sec_rounding(res, model, f"bi/{name}", targets, quick)
        if "cost" in sections:
            log("-- cost")
            sec_cost(res, model, name, quick, small)
    if "backward" in sections:
        log("-- backward")
        for ops in ("bi", "stock"):
            if quick and ops == "stock":
                continue
            sec_backward(res, model, ops, f"{ops}/{name}", targets, quick, "pass" if ops == "bi" else "fail", small)
    del model
    torch.cuda.empty_cache()


def merge(paths, out):
    res = Results("whole_model", quick=False, argv=["--merge"] + list(paths))
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        res.checks += d["checks"]
        res.measurements += d["measurements"]
        for k, v in d["extra"].items():
            if isinstance(v, dict) and isinstance(res.extra.get(k), dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict) and isinstance(res.extra[k].get(kk), dict):
                        res.extra[k][kk].update(vv)
                    else:
                        res.extra[k][kk] = vv
            else:
                res.extra[k] = v
        res.extra.setdefault("merged_from", []).append({"argv": d["argv"][1:], "counts": d["counts"]})
    res.write(out)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="every expect='pass' check at reduced sizes (about 2 minutes)")
    ap.add_argument("--json", help="write the results here (bi-results-v1)")
    ap.add_argument("--only", default=",".join(SECTIONS), help="comma-separated sections: " + ", ".join(SECTIONS))
    ap.add_argument("--variants", default=None, help="comma-separated: dense, moe, wide, qwen2 (default: all; quick: dense, moe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", nargs="+", help="merge result files written by separate runs")
    args = ap.parse_args()
    if args.merge:
        sys.exit(merge(args.merge, args.json).exit_code())
    sections = [s for s in args.only.split(",") if s]
    assert all(s in SECTIONS for s in sections), sections
    variants = (args.variants or ("dense,moe" if args.quick else "dense,moe,wide")).split(",")
    if args.quick:
        sections = [s for s in sections if s != "cost"]
    torch.manual_seed(0)
    res = Results("whole_model", quick=args.quick)
    res.extra["gpu_state_start"] = gpu_state()
    res.extra["sections"] = sections
    t0 = time.time()
    for v in variants:
        run_variant(res, v, sections, args.quick, args.seed)
    res.extra["seconds"] = round(time.time() - t0, 1)
    res.write(args.json)
    sys.exit(res.exit_code())


if __name__ == "__main__":
    main()
