"""Two-node extension of moe_ep.py: 16 ranks (8 per node, node 0 = ranks 0-7).

L5 across nodes, L1 and L2 at EP = 16, and relative timing at EP = 8 (one node,
or 4+4 across nodes) against EP = 16. The layer, its six combines and its
kernels are moe_ep.py's; only the EP group changes.

Every layout splits the 16 ranks into EP groups that run at the same time:
  ep1       each GPU alone (16 groups of 1)
  ep2       pairs on one node          ep2_x   pairs (i, i+8), one rank per node
  ep4       four on one node
  ep8_node  one group per node         ep8_x   ranks 0-3 + 8-11, and 4-7 + 12-15
  ep16      all 16 ranks
Each group runs the same global batch under three shardings and compares, bitwise,
with the EP = 1 output of rank 0 (node 0, GPU 0), broadcast to every rank. Rank 0
also compares its EP = 1 output with the single-node run's EP = 1 dumps in
ref1node/<config>/ when they are present.

Launch on node 0 (arguments must not contain spaces):
  launch2.sh <workdir> moe_ep_2node.py --parts l5,l1 --json results-a16-2node.json
Regression suite (one launch): --parts l5,l1,l2 --tg4096-k 0 --l2-sizes 1,3,31,129, with
--ref1node pointing at the dump directory of the one-node run of the same suite.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moe_ep as M  # noqa: E402  (also puts tools/ on the path)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from bi_check import bits  # noqa: E402

H, E = M.H, 32
PER_NODE = 8
REF1NODE = os.path.join(HERE, "ref1node")       # --ref1node: EP = 1 dumps of a one-node run


def ref1node_path(cfg, v):
    """<dir>/<config>/<variant>__contig.pt, or the layout moe_ep.py --dump writes (<config>/ep1/)."""
    for sub in ((cfg,), (cfg, "ep1")):
        p = os.path.join(REF1NODE, *sub, f"{v}__contig.pt")
        if os.path.exists(p):
            return p
    return None


def layouts(n=16, node=PER_NODE):
    half = node // 2
    return {
        "ep1": [[i] for i in range(n)],
        "ep2": [[i, i + 1] for i in range(0, n, 2)],
        "ep2_x": [[i, i + node] for i in range(node)],
        "ep4": [list(range(i, i + 4)) for i in range(0, n, 4)],
        "ep8_node": [list(range(0, node)), list(range(node, n))],
        "ep8_x": [list(range(0, half)) + list(range(node, node + half)),
                  list(range(half, node)) + list(range(node + half, n))],
        "ep16": [list(range(n))],
    }


def make_groups(lay):
    """Every rank calls new_subgroups_by_enumeration for every layout, in the same order."""
    r, n = dist.get_rank(), dist.get_world_size()
    groups = {}
    for name, parts in lay.items():
        mine = next(p for p in parts if r in p)
        if len(parts) == 1 and len(parts[0]) == n:
            groups[name] = (None, mine)
        else:
            cur, _ = dist.new_subgroups_by_enumeration(parts)
            groups[name] = (cur, mine)
    return groups


def local_params(full, members, rank):
    """This rank's experts for an EP group: the same weights as moe_ep.make_params."""
    W, gr = len(members), members.index(rank)
    El = E // W
    return {"Wr": full["Wr"], **{k: full[k][gr * El:(gr + 1) * El] for k in ("W1", "W3", "W2")}}


def _cmp(go, gi, gk, ref, W):
    a, aid, ak = ref
    diff = bits(go) != bits(a)
    bad = diff.any(1)
    rec = {"el": int(diff.sum()), "tok": int(bad.sum()), "ids_equal": torch.equal(gi, aid),
           "drops_equal": torch.equal(gk, ak)}
    if rec["el"]:
        span = M._span(aid, E // W)
        rec["by_rank_span"] = {int(k): [int((bad & (span == k)).sum()), int((span == k).sum())]
                               for k in span.unique().tolist()}
        rec["tok_with_same_drops"] = int((bad & (gk == ak).all(1)).sum())
    return rec


def run_l5(full, lay, groups, K, Tg, variants, dev, cfg):
    r = dist.get_rank()
    Xg = M._randn((Tg, H), 7, 1.0).to(dev)
    # reference: EP = 1 on rank 0, broadcast to everyone
    g1, m1 = groups["ep1"]
    P1 = local_params(full, m1, r)
    REF, link = {}, {}
    for v in variants:
        out, ids, keep = M.moe_ep(Xg, P1, E, K, v, group=g1)
        ref = [out.clone(), ids.clone(), keep.to(torch.uint8)]
        for t in ref:
            dist.broadcast(t, 0)
        REF[v] = ref
        p = ref1node_path(cfg, v) if r == 0 else None
        if p:
            old = torch.load(p, weights_only=True)
            link[v] = int((bits(old["out"]) != bits(ref[0].cpu())).sum())
    recs = []
    for lname, (g, mem) in groups.items():
        W, gr = len(mem), mem.index(r)
        P = local_params(full, mem, r)
        for v in variants:
            for sname, parts in M.shardings(Tg, W).items():
                out, ids, keep = M.moe_ep(Xg[parts[gr].to(dev)], P, E, K, v, group=g)
                go, gi, gk = (M.gather_global(t, parts, Tg, group=g) for t in (out, ids, keep.to(torch.uint8)))
                if gr == 0:
                    recs.append({"layout": lname, "group": mem, "variant": v, "sharding": sname,
                                 **_cmp(go, gi, gk, REF[v], W)})
        dist.barrier()
    allrecs = [None] * dist.get_world_size()
    dist.all_gather_object(allrecs, recs)
    if r != 0:
        return None
    # compact: per layout and variant, mismatched elements per group for each sharding
    summ = {"vs_single_node_ep1_dump": link or "ref1node not present"}
    for rec in (x for lst in allrecs for x in lst):
        d = summ.setdefault(rec["layout"], {"groups": [], "variants": {}})
        if rec["group"] not in d["groups"]:
            d["groups"].append(rec["group"])
        e = d["variants"].setdefault(rec["variant"], {"el": {}, "tok": {}, "ids_equal": True})
        e["el"].setdefault(rec["sharding"], []).append(rec["el"])
        e["tok"].setdefault(rec["sharding"], []).append(rec["tok"])
        e["ids_equal"] &= rec["ids_equal"]
        if rec["el"] and "by_rank_span" in rec and rec["sharding"] == "contig" and "by_rank_span_contig" not in e:
            e["by_rank_span_contig"] = rec["by_rank_span"]
        if rec["el"]:
            e["max_tok_with_same_drops"] = max(e.get("max_tok_with_same_drops", 0), rec["tok_with_same_drops"])
    for lname, d in summ.items():
        if lname == "vs_single_node_ep1_dump":
            continue
        print(cfg, lname, {v: e["el"] for v, e in d["variants"].items()}, flush=True)
    print(cfg, "vs single-node EP=1 dump:", summ["vs_single_node_ep1_dump"], flush=True)
    return summ


def run_timing(full, lay, groups, K, variants, dev, tokens, iters=10, reps=2):
    """One group per layout runs while the other ranks wait: ep8_node on node 1, ep8_x as
    ranks 0-3 + 8-11, ep16 on all ranks."""
    r = dist.get_rank()
    active = {"ep8_node": lay["ep8_node"][1], "ep8_x": lay["ep8_x"][0], "ep16": lay["ep16"][0]}
    res = {}
    for lname, act in active.items():
        g, mem = groups[lname]
        for n in tokens:
            if r in act:
                P = local_params(full, mem, r)
                x = M.filler("random", n, 99 + r, dev)
                best = {v: float("inf") for v in variants}
                for _ in range(reps):
                    for v in variants:
                        for _ in range(3):
                            M.moe_ep(x, P, E, K, v, group=g)
                        torch.cuda.synchronize()
                        dist.barrier(group=g)
                        t0 = time.perf_counter()
                        for _ in range(iters):
                            M.moe_ep(x, P, E, K, v, group=g)
                        torch.cuda.synchronize()
                        t = torch.tensor([(time.perf_counter() - t0) / iters * 1e3], device=dev)
                        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=g)
                        best[v] = min(best[v], t.item())
                if mem.index(r) == 0:
                    res.setdefault(lname, {})[n] = best
                    print(f"timing {lname} tokens/rank={n}: "
                          + " ".join(f"{v}={best[v] / best['arrival_atomic']:.2f}" for v in variants), flush=True)
            dist.barrier()
    allres = [None] * dist.get_world_size()
    dist.all_gather_object(allres, res)
    if r != 0:
        return None
    out = {}
    for d in allres:
        for lname, byn in d.items():
            out.setdefault(lname, {}).update(byn)
    return out


def main():
    global REF1NODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="l5,l1")
    ap.add_argument("--ks", default="4,3", help="top-k values for L5 (and L1)")
    ap.add_argument("--tgs", default="1024", help="global batch sizes for L5")
    ap.add_argument("--tg4096-k", type=int, default=4, help="also run L5 at 4096 tokens for this top-k (0: off)")
    ap.add_argument("--l2-sizes", default="1,3,31,129,512")
    ap.add_argument("--timing-tokens", default="512,2048")
    ap.add_argument("--variants", default=",".join(M.VARIANTS))
    ap.add_argument("--json", default=os.path.join(HERE, "results-a16-2node.json"))
    ap.add_argument("--ref1node", default=REF1NODE, help="EP = 1 dumps of a one-node run, read by rank 0")
    args = ap.parse_args()
    REF1NODE = args.ref1node
    dist.init_process_group("nccl")
    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    r, n = dist.get_rank(), dist.get_world_size()
    assert n == 16, n
    hosts = [None] * n
    dist.all_gather_object(hosts, socket.gethostname())
    node_ok = len(set(hosts[:PER_NODE])) == 1 and len(set(hosts[PER_NODE:])) == 1 and hosts[0] != hosts[PER_NODE]
    variants = args.variants.split(",")
    parts = set(args.parts.split(","))
    ks = [int(k) for k in args.ks.split(",")]
    full = M.make_params(E, 0, 1, dev)                     # every expert on every rank; groups slice it
    lay = layouts()
    groups = make_groups(lay)
    res = {}
    if r == 0 and os.path.exists(args.json):
        res = json.load(open(args.json))
    res["meta"] = {"ranks": n, "per_node": PER_NODE, "ranks_0_7_one_node_8_15_another": node_ok,
                   "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                   "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"), "E": E, "H": H, "I": M.I,
                   "layouts": lay}
    if r == 0:
        print("hosts ok:", node_ok, flush=True)
    if "l5" in parts:
        runs = [(k, tg) for k in ks for tg in map(int, args.tgs.split(","))]
        if args.tg4096_k:
            runs.append((args.tg4096_k, 4096))
        for k, tg in runs:
            cfg = f"E{E}K{k}" + (f"T{tg}" if tg != 1024 else "")
            s = run_l5(full, lay, groups, k, tg, variants, dev, cfg)
            if r == 0:
                res.setdefault("L5", {})[cfg] = s
    g16, m16 = groups["ep16"]
    P16 = local_params(full, m16, r)
    if "l1" in parts:
        Xg = M._randn((1024, H), 7, 1.0).to(dev)
        for k in ks:
            out = M.run_l1(P16, E, k, variants, Xg, dev)
            if r == 0:
                res.setdefault("L1_ep16", {})[f"E{E}K{k}"] = out
    if "l2" in parts:
        probe = M._randn((2, H), 11, 1.0).to(dev)
        sizes = tuple(int(s) for s in args.l2_sizes.split(","))
        k = ks[0]
        out = M.run_l2(P16, E, k, variants, probe, dev, sizes=sizes)
        if r == 0:
            for v in out.values():
                v.pop("failures", None)
            res.setdefault("L2_ep16", {})[f"E{E}K{k}"] = {"tokens_per_rank": list(sizes), **out}
    if "timing" in parts:
        toks = [int(t) for t in args.timing_tokens.split(",")]
        out = run_timing(full, lay, groups, ks[0], variants, dev, toks)
        if r == 0:
            res.setdefault("timing_ms", {})[f"E{E}K{ks[0]}"] = out
    if r == 0:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
