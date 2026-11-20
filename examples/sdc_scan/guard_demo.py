"""In-job SDC checks from tools/sdc_scan.py: their cost, and that they catch an injected fault.

Part `guard` (one GPU):
  1. sdc_guard on the forward of a small transformer LM (tools/sdc_scan.TinyLM, bf16 autocast,
     d=1024, 8 layers, batch 8 x 512): a healthy recompute matches (no false alarm, 20 steps);
     a bit flipped in the recomputed logits raises SDCMismatch naming the output and the word.
  2. cost: one training step (forward, backward, AdamW) alone, and with sdc_guard on its
     forward; the guard's cost as a fraction of the step.
Part `ranks` (torch.distributed.run, 3 or more ranks, one GPU each):
  3. cross_rank_check: every rank computes rank 0's micro-batch; healthy ranks agree;
     a bit flipped on one rank's output names that rank; replica_check on the parameters.

  python guard_demo.py --part guard --json results-guard.json
  torchrun --nproc_per_node=4 guard_demo.py --part ranks --json results-ranks.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import torch  # noqa: E402
import sdc_scan as S  # noqa: E402
from bi_report import Results  # noqa: E402


def setup():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def model_and_batch(seed=1, d=1024, layers=8, B=8, T=512, vocab=32000):
    m = S.TinyLM(vocab=vocab, d=d, heads=8, layers=layers, T=T)
    g = S.Gen(seed)
    C = 1 << 24                                   # the generator serves at most 2^25 values per tensor
    flat = torch.cat([g.make(f"guard.flat.{i}", S.N(min(C, m.nparams - i), scale=2.0 ** -5))
                      for i in range(0, m.nparams, C)]).cuda()
    p = m.carve(flat)
    tok = g.make("guard.tok", S.I(B, T, hi=vocab)).cuda()
    tgt = g.make("guard.tgt", S.I(B, T, hi=vocab)).cuda()
    return m, p, tok, tgt


def timed(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def part_guard(res):
    m, p, tok, tgt = model_and_batch()
    opt = torch.optim.AdamW(list(p.values()), lr=1e-5, fused=True)
    impl = "sdc_guard"

    def fwd(tok):
        return m.logits(p, tok)

    # 1. no false alarm over 20 guarded forwards (weights change between steps)
    alarms = 0
    for _ in range(20):
        out, rep = S.sdc_guard(fwd, tok, on_mismatch="return")
        alarms += not rep["ok"]
        loss = torch.nn.functional.cross_entropy(out.float().view(-1, m.vocab), tgt.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    res.check("sdc_guard: 20 healthy training steps, recomputed forward equals the first (no false alarm)",
              alarms == 0, level="L1", impl=impl, detail={"alarms": alarms})

    # 1b. an injected fault in the recompute is caught and located
    calls = [0]

    def faulty(tok):
        out = fwd(tok)
        calls[0] += 1
        if calls[0] == 2:                       # the recompute
            w = out.view(-1).view(torch.int16)
            w[123457] ^= 1 << 3
        return out
    try:
        with torch.no_grad():
            S.sdc_guard(faulty, tok)
        caught = None
    except S.SDCMismatch as e:
        caught = e.report
    mm = (caught or {}).get("mismatches", [{}])[0]
    res.check("sdc_guard: a bit flipped in the recomputed logits raises SDCMismatch with the differing word",
              caught is not None and mm.get("words_differing") == 1 and mm["first"][0]["element"] == 123456,
              level="other", impl=impl, detail=caught)

    # 2. cost
    def step():
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(fwd(tok).float().view(-1, m.vocab), tgt.view(-1))
        loss.backward()
        opt.step()

    def guarded_step():
        opt.zero_grad(set_to_none=True)
        out = S.sdc_guard(fwd, tok)
        loss = torch.nn.functional.cross_entropy(out.float().view(-1, m.vocab), tgt.view(-1))
        loss.backward()
        opt.step()

    with torch.no_grad():
        f_ms = timed(lambda: fwd(tok))
        logits = fwd(tok)
        fp_ms = timed(lambda: S.fingerprint_tensor(logits))
    s_ms, g_ms = timed(step), timed(guarded_step)
    shape = f"d=1024, 8 layers, {m.nparams / 1e6:.0f}M params, batch 8x512, logits {logits.numel() * 2 / 2**20:.0f} MiB bf16"
    res.measure("training step", round(s_ms, 2), "ms", impl="baseline", shape=shape)
    res.measure("training step with sdc_guard on the forward", round(g_ms, 2), "ms", impl=impl, shape=shape)
    res.measure("forward alone (no grad)", round(f_ms, 2), "ms", impl="baseline", shape=shape)
    res.measure("fingerprint of the logits", round(fp_ms, 3), "ms", impl=impl, shape=shape)
    res.measure("guard cost / step", round(g_ms / s_ms - 1, 3), "fraction", impl=impl, shape=shape)
    res.extra["guard"] = {"shape": shape, "step_ms": s_ms, "guarded_step_ms": g_ms, "forward_ms": f_ms, "fingerprint_ms": fp_ms}
    print(f"step {s_ms:.1f} ms, guarded {g_ms:.1f} ms (+{(g_ms / s_ms - 1) * 100:.0f}%), forward {f_ms:.1f} ms, "
          f"fingerprint {fp_ms:.2f} ms")


def part_ranks(res):
    import torch.distributed as dist
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    m, p, _, _ = model_and_batch(d=512, layers=4, B=4, T=256, vocab=8000)
    g = S.Gen(100 + rank)                                    # every rank has its own micro-batch
    tok = g.make("rank.tok", S.I(4, 256, hi=8000)).cuda()

    def fwd(tok):
        return m.logits(p, tok)
    rep = S.cross_rank_check(fwd, tok)
    res.check(f"cross_rank_check: {world} ranks compute rank 0's micro-batch, all agree", rep["ok"], level="other",
              impl="cross_rank_check", detail=rep)
    bad_rank = world - 1

    def faulty(tok):
        out = fwd(tok)
        if dist.get_rank() == bad_rank:
            out.view(-1).view(torch.int16)[999] ^= 1 << 14
        return out
    rep2 = S.cross_rank_check(faulty, tok)
    res.check(f"cross_rank_check: a bit flipped on rank {bad_rank}'s output names rank {bad_rank}",
              not rep2["ok"] and rep2["outliers"] == [bad_rank] and not rep2["input_outliers"], level="other",
              impl="cross_rank_check", detail=rep2)
    rep3 = S.replica_check(list(p.values()))
    res.check(f"replica_check: {world} replicas hold the same parameters", rep3["ok"], level="other",
              impl="replica_check", detail=rep3)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        S.cross_rank_check(fwd, tok)
    torch.cuda.synchronize()
    res.measure("cross_rank_check (broadcast, forward, fingerprints, all-gather)",
                round((time.perf_counter() - t0) / 5 * 1e3, 2), "ms", impl="cross_rank_check", world=world)
    dist.destroy_process_group()
    return rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("guard", "ranks"), required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    setup()
    res = Results(f"sdc_scan_{args.part}", quick=args.quick)
    rank = 0
    if args.part == "guard":
        part_guard(res)
    else:
        rank = part_ranks(res)
    if rank == 0:
        res.write(args.json)
    sys.exit(res.exit_code())


if __name__ == "__main__":
    main()
