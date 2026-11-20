"""An RL loop end to end with OpGuard's RL harness: online records, the staleness/numeric split,
the weight-sync check and offline localization, each through the harness CLI as an agent calls it.

    OPGUARD_ROOT=~/OpGuard PYTHONPATH=$OPGUARD_ROOT:$OPGUARD_ROOT/harness \
        python examples/rl_loop/rl_loop.py --quick --json examples/rl_loop/results-a16.json

The loop (rl_core.py): rollout process = model.py engine mode + bi_sampling, fused QKV;
trainer process = model.py prefill + AdamW; weights pushed every step. Scenarios:

  sync      staleness 0, bi op sets on both sides: weight_sync_check clean, numeric == 0
  async1/2  rollouts at policy version v trained at c = v+1 / v+2, a push in the middle of
            generation (mixed-version sequences); trainer replays at v from snapshots
  stock     fault a: rollout runs the stock op set while its flags say batch-invariant
  norm      fault b: rollout RMSNorm rounds after its weight (rounding point moved)
  sync_bug  fault c: one bit flipped in the fused qkv weight after repacking (v1), one stale
            row of down_proj (v2)
  version   fault d: the rollout labels batch 2 with version 2 but applies it after the batch

--quick: tiny config, 3 steps, scenarios sync, async1 and sync_bug (bitflip), one GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_core as C                                         # noqa: E402  (sets up sys.path)
import torch                                                # noqa: E402
import torch.multiprocessing as mp                          # noqa: E402
import opguard_bridge as ob                                 # noqa: E402
from bi_report import Results                               # noqa: E402
from opguard_harness.rl_records import RecordWriter        # noqa: E402

TRANSCRIPT: list = []


# ----------------------------------------------------------------------------- harness calls
class Agent:
    """Every fact comes from one `opguard-harness call` (a subprocess), logged trimmed."""

    def __init__(self, og, store, scenario):
        self.og, self.store, self.scenario = og, Path(store), scenario
        self.calls, self.seconds = 0, 0.0

    def __call__(self, tool, args, keep=None):
        t0 = time.perf_counter()
        out = ob.harness(tool, args, self.store, self.og)
        dt = time.perf_counter() - t0
        self.calls += 1
        self.seconds += dt
        TRANSCRIPT.append({"scenario": self.scenario, "tool": tool,
                           "args": {k: shorten(k, v) for k, v in args.items() if k != "artifact"},
                           "result": keep(out) if keep else out, "seconds": round(dt, 2)})
        return out


def shorten(key, value):
    """Transcript form of a tool argument: file names, not paths; id lists as their length."""
    if key == "records":
        return [Path(p).name for p in value]
    if key.endswith("token_ids") and isinstance(value, list):
        return f"<{len(value)} ids>"
    if key == "output" and value:
        return Path(value).name
    return value


def trim_decompose(d):
    toks = [t for t in d["tokens"] if t.get("numeric") and not t["numeric"]["zero"]][:2] or d["tokens"][:1]
    return {"summary": d["summary"], "versions_without_trainer_logprob": d["versions_without_trainer_logprob"],
            "example_tokens": [{k: t.get(k) for k in ("seq_id", "position", "policy_version", "mixed_version",
                                                       "staleness_zero_point", "numeric", "staleness")} for t in toks]}


def trim_wsc(d):
    bad = {side: [{k: p[k] for k in ("name", "rows", "unmatched_rows", "unmatched_count")}
                  for p in d[side] if p.get("status") != "all_rows_matched"] for side in ("trainer", "rollout")}
    return {"summary": d["summary"], "unmatched": bad}


# ----------------------------------------------------------------------------- the loop
def run_loop(run_dir: Path, cfg_name: str, steps: int, lag: int, devs, rollout_fault=None, sync_faults=None,
             defer_version=None, record=True, lr=2e-3, snap_every=1, log=print):
    sync_faults = sync_faults or {}
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = C.make_cfg(cfg_name)
    rec_roll, rec_tr = run_dir / "records-rollout.jsonl", run_dir / "records-trainer.jsonl"
    ctx = mp.get_context("spawn")
    here, there = ctx.Pipe()
    proc = ctx.Process(target=C.rollout_worker, args=(there, cfg_name, devs[1], rollout_fault,
                                                      str(rec_roll) if record else None))
    proc.start()
    torch.cuda.set_device(devs[0])
    tr = C.Trainer(cfg, devs[0], lr)
    rec = RecordWriter(rec_tr) if record else None
    snap = run_dir / "snap"
    snap.mkdir(exist_ok=True)
    states, tents, steps_out = {0: tr.state()}, {}, []

    def tentries(v):
        t0 = time.perf_counter()
        tents[v] = tr.entries() if record else None
        return time.perf_counter() - t0

    def snapshot(v):
        if not record or v % snap_every:
            return 0.0
        t0 = time.perf_counter()
        path = snap / f"v{v}.pt"
        torch.save(states[v], path)
        rec.snapshot(v, str(path))
        return time.perf_counter() - t0

    t_first = tentries(0) + snapshot(0)
    roll_version, deferred = None, False
    for k in range(steps):
        tm = {"fingerprint_trainer_s": t_first if k == 0 else 0.0}
        t_step = time.perf_counter()
        start = k if lag == 0 else max(0, k - lag)
        label_offset = 0
        if roll_version != start:
            fault = {"kind": "defer"} if defer_version == start else sync_faults.get(start)
            here.send(("load", states[start], start, fault))
            r = here.recv()
            tm.update(load_s=r["load_s"], fingerprint_rollout_s=r["entries_s"])
            if r.get("deferred"):
                label_offset, deferred = 1, True      # labels the sent version, still runs start - 1
            elif record:
                t0 = time.perf_counter()
                rec.weight_sync(start, tents[start], r["entries"])
                tm["record_trainer_s"] = time.perf_counter() - t0
            roll_version = start
        push = None
        if lag > 0 and k >= lag:
            j = start + 1
            push = (4, states[j], j, sync_faults.get(j))
        spec = C.batch_spec(k, cfg)
        here.send(("generate", k, spec, push, label_offset))
        g = here.recv()
        tm.update(gen_s=g["gen_s"], record_rollout_s=g["record_s"])
        pushed = push[2] if push else (start if deferred else None)
        if record and g["push_entries"] is not None and pushed is not None:
            t0 = time.perf_counter()
            rec.weight_sync(pushed, tents[pushed], g["push_entries"])
            tm["record_trainer_s"] = tm.get("record_trainer_s", 0.0) + time.perf_counter() - t0
        deferred = False
        if push:
            roll_version = push[2]
        seqs = g["seqs"]
        C.sync()
        t0 = time.perf_counter()
        lps = tr.logprobs(seqs, rec)
        C.sync()
        tm["trainer_logprob_s"] = time.perf_counter() - t0
        if record:
            t0 = time.perf_counter()
            rec.trainer_logprob(k, tr.version, [{"seq_id": s["seq_id"], "positions": s["positions"], "logprob": lp}
                                                for s, lp in zip(seqs, lps)],
                                engine_flags={"side": "trainer", "batch_invariant": True})
            tm["record_trainer_s"] = tm.get("record_trainer_s", 0.0) + time.perf_counter() - t0
        rewards = [C.reward(s) for s in seqs]
        t0 = time.perf_counter()
        if rec is not None:
            tr.forwards += 1
            rec.forward(counter=tr.forwards, rank=0, tokens=sum(len(s["prompt_ids"]) + len(s["token_ids"]) for s in seqs))
        loss = tr.train(seqs, rewards, [x["group"] for x in spec])
        C.sync()
        tm["train_s"] = time.perf_counter() - t0
        states[tr.version] = tr.state()
        for old in [v for v in states if v < tr.version - lag - 2]:
            del states[old]
        tm["fingerprint_trainer_s"] += tentries(tr.version)
        tm["snapshot_s"] = snapshot(tr.version)
        tm["step_s"] = time.perf_counter() - t_step
        versions = sorted({v for s in seqs for v in s["policy_version"]})
        mixed = sum(len(set(s["policy_version"])) > 1 for s in seqs)
        steps_out.append({"k": k, "reward": sum(rewards) / len(rewards), "loss": loss, "versions": versions,
                          "trainer_version": k, "mixed_sequences": mixed,
                          "tokens": sum(len(s["token_ids"]) for s in seqs), **tm})
        log(f"  step {k}: versions {versions} at c={k}, mixed {mixed}, reward {sum(rewards)/len(rewards):.3f}, "
            f"step {tm['step_s']*1e3:.0f} ms")
    here.send(("stop",))
    here.recv()
    proc.join()
    if rec is not None:
        rec.close()
    nbytes = sum(p.stat().st_size for p in (rec_roll, rec_tr) if p.exists())
    return {"dir": run_dir, "records": [str(rec_tr), str(rec_roll)], "steps": steps_out, "cfg": cfg_name,
            "record_bytes": nbytes, "lag": lag}


# ----------------------------------------------------------------------------- agent steps
def split_check(agent, run, replay_dev, cfg_name, rows_out):
    """Skill steps 1-2: weight sync for every version, then the split, with trainer replays."""
    records = run["records"]
    summ = agent("rl.records_summary", {"records": records},
                 keep=lambda d: {k: d[k] for k in ("by_type", "weight_sync_versions", "snapshot_versions",
                                                   "forwards_per_rank")})
    wsc = {}
    for v in summ["weight_sync_versions"]:
        d = agent("rl.weight_sync_check", {"records": records, "version": v}, keep=trim_wsc)
        wsc[v] = d["summary"]
    replays, replay_cost = [], []
    totals = {"tokens": 0, "mixed_version_tokens": 0, "numeric_zero": 0, "numeric_nonzero": 0, "numeric_missing": 0,
              "staleness_zero": 0, "staleness_nonzero": 0, "staleness_zero_points": 0, "mixed_sequences": 0}
    first = {}
    for st in run["steps"]:
        k = st["k"]
        d = agent("rl.decompose", {"records": records, "batch_id": k, "trainer_version": k, "policy_version_override": None}, keep=trim_decompose)
        first[k] = d["summary"]
        for v in d["versions_without_trainer_logprob"]:
            out = str(run["dir"] / f"replay-req-b{k}-v{v}.json")
            agent("rl.replay_request", {"records": records, "batch_id": k, "version": v, "side": "trainer",
                                        "composition": "original", "capture": None, "output": out},
                  keep=lambda r: {k2: r[k2] for k2 in ("weights_located", "reproduces", "executed", "note")})
            rp = run["dir"] / f"records-replay-b{k}-v{v}.jsonl"
            cost = C.run_replay(out, cfg_name, replay_dev, rp)
            replay_cost.append(cost["load_s"] + cost["run_s"])
            replays.append(str(rp))
        if d["versions_without_trainer_logprob"]:
            d = agent("rl.decompose", {"records": records + [r for r in replays if f"-b{k}-" in r], "batch_id": k,
                                       "trainer_version": k, "policy_version_override": None}, keep=trim_decompose)
        for key in totals:
            totals[key] += d["summary"].get(key, 0)
        rows_out.append({"k": k, "before_replay": first[k], "after_replay": d["summary"]})
    return {"summary": summ, "weight_sync": wsc, "totals": totals, "replays": replays,
            "replay_s": replay_cost}


def observed_rows(agent, records_path):
    """Each forward's observed (seq_id, position) rows, from the forward records (rl.forwards)."""
    got = agent("rl.forwards", {"records": [str(records_path)], "rank": 0, "counter_min": None, "counter_max": None},
                keep=lambda r: {"count": r["count"], "row_layout": sorted({f["row_layout"] for f in r["forwards"]})})
    return [f["rows"] for f in got["forwards"]]


def list_run(agent, artifact, run):
    """Every event of one run, in counter order, through find_events pages (200 events each)."""
    events, cursor = [], 0
    while cursor is not None:
        got = agent("find_events", {"artifact": artifact, "query": None, "run": run, "rank": None, "counter_min": None,
                                    "counter_max": None, "cursor": cursor, "limit": 200},
                    keep=lambda r: {"events": len(r["events"]), "match_count": r["match_count"]})
        events += got["events"]
        cursor = got["next_cursor"]
    return events


def localize(agent, run, k, version, cfg_name, dev, rollout_fault, workdir, validate=False):
    """Skill step 3: per-vector recapture of batch k on both sides, then a data-flow walk with
    vectors.compare from an agreeing anchor to the first paired candidate."""
    records = run["records"]
    workdir = Path(workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    d = agent("rl.decompose", {"records": records, "batch_id": k, "trainer_version": version, "policy_version_override": None},
              keep=trim_decompose)
    bad = [t for t in d["tokens"] if t.get("numeric") and not t["numeric"]["zero"] and not t["mixed_version"]]
    out = {"numeric_nonzero": len(bad), "tokens": d["summary"]["tokens"]}
    if bad:
        tok = min(bad, key=lambda t: (t["position"], t["seq_id"]))
        sid, pos = tok["seq_id"], tok["position"]
    else:
        sid, pos = d["tokens"][0]["seq_id"], d["tokens"][0]["position"]
    out["token"] = {"seq_id": sid, "position": pos, "scoring_row": [sid, pos - 1]}
    caps, logps = {}, {}
    for side in ("trainer", "rollout"):
        req = str(workdir / f"req-{side}.json")
        agent("rl.replay_request", {"records": records, "batch_id": k, "version": version, "side": side,
                                    "composition": "original", "capture": {"vector_fingerprints": True},
                                    "output": req},
              keep=lambda r: {k2: r[k2] for k2 in ("weights_located", "runnable_alone", "reproduces", "executed")} |
              {"capture_env": r["request"]["capture_env"], "capture_path": r["request"]["capture_path"]["assumes"]})
        caps[side] = workdir / side
        cost = C.run_replay(req, cfg_name, dev, workdir / f"records-{side}.jsonl",
                            rollout_fault=rollout_fault if side == "rollout" else None, capture_root=caps[side])
        logps[side] = cost["logprobs"]
        out[f"recapture_{side}_s"] = cost["load_s"] + cost["run_s"]
    # the recapture against the online records: does it reproduce the online rollout log-probs?
    roll_rec = [json.loads(l) for l in open(records[1]) if '"rollout_batch"' in l]
    online = next(r for r in roll_rec if r["batch_id"] == k)
    from opguard_harness.rl_records import decode_logprob
    on_bits = [b for s in online["sequences"] for (_, b, _) in decode_logprob(s["logprob"])]
    re_roll = [json.loads(l) for l in open(workdir / "records-rollout.jsonl") if '"rollout_batch"' in l][0]
    re_bits = [b for s in re_roll["sequences"] for (_, b, _) in decode_logprob(s["logprob"])]
    out["recapture_reproduces_online_rollout"] = on_bits == re_bits
    tr_bits = [int(x) for lp in logps["trainer"] for x in lp.view(torch.int32).tolist()]
    out["recapture_rollout_vs_trainer_equal_tokens"] = sum(a == (b & 0xFFFFFFFF) for a, b in
                                                           zip(re_bits, tr_bits))
    export = ob.export_pair(caps["trainer"], caps["rollout"], workdir / "compare", agent.og)
    artifact = str(export["reports"][0])
    out["artifact_bytes"] = Path(artifact).stat().st_size
    fd = agent("first_diff", {"artifact": artifact},
               keep=lambda r: {"boundaries": [{"run": b["run"], "first_observed_diff": (b["first_observed_diff"] or {}).get("name")}
                                              for b in r["boundaries"]],
                               "operations": [c["operations"] for c in r["call_sequences"]],
                               "functions_with_different_call_counts": [c["functions_with_different_call_counts"]
                                                                        for c in r["call_sequences"]],
                               "no_observed_diff_note": "no_observed_diff_note" in r})
    out["first_diff"] = {"observed_diffs": sum(b["first_observed_diff"] is not None for b in fd["boundaries"]),
                         "says_silence_is_not_evidence": "no_observed_diff_note" in fd,
                         "functions_with_different_call_counts": [c["functions_with_different_call_counts"]
                                                                  for c in fd["call_sequences"]]}
    runs = {b["run"]: list_run(agent, artifact, b["run"]) for b in fd["boundaries"]}
    # which run is which side: the rollout has the fused qkv_proj
    side_of = {r: ("rollout" if any("qkv_proj" in (e.get("fn") or e["name"]) for e in evs) else "trainer") for r, evs in runs.items()}
    by = {side_of[r]: {} for r in runs}
    for r, evs in runs.items():
        for e in sorted(evs, key=lambda e: e["counter"]):
            by[side_of[r]].setdefault((e.get("fn") or e["name"]).strip("<>"), []).append(e["event_ref"])
    # Row layouts are observed: the replays recorded each forward's (seq_id, position) rows.
    layout = observed_rows(agent, workdir / "records-rollout.jsonl")
    t_ids = observed_rows(agent, workdir / "records-trainer.jsonl")[0]          # padded [B, T]; padding rows null
    row = [sid, pos - 1]
    f = next(i for i, ids in enumerate(layout) if row in ids)
    seqs = sorted(online["sequences"], key=lambda s: s["seq_id"])
    t_lp_ids = [[s["seq_id"], p - 1] for s in seqs for p in s["positions"]]
    r_lp_ids = []                                   # the last row of each sequence in forward f
    for x in online["composition"][f]:
        r_lp_ids.append([i for i in layout[f] if i is not None and i[0] == x][-1])
    cfg = C.make_cfg(cfg_name)
    points = [("model.embed_tokens", "outputs", 0, "embedding")]
    for l in range(cfg.layers):
        p = f"model.layers.{l}"
        points += [(f"{p}.input_layernorm", "inputs", 0, "residual stream"),
                   (f"{p}.input_layernorm", "outputs", 0, f"{p}.input_layernorm (RMSNorm)"),
                   (f"{p}.self_attn.rope", "inputs", 0, f"{p}.self_attn.q_proj (trainer) / qkv_proj + split (rollout)"),
                   (f"{p}.self_attn.rope", "inputs", 1, f"{p}.self_attn.k_proj (trainer) / qkv_proj + split (rollout)"),
                   (f"{p}.self_attn.rope", "outputs", 0, f"{p}.self_attn.rope (q)"),
                   (f"{p}.self_attn.o_proj", "inputs", 0, f"{p} attention (trainer core / engine paged kernel)"),
                   (f"{p}.self_attn.o_proj", "outputs", 0, f"{p}.self_attn.o_proj"),
                   (f"{p}.post_attention_layernorm", "inputs", 0, f"{p} residual add (attention)"),
                   (f"{p}.post_attention_layernorm", "outputs", 0, f"{p}.post_attention_layernorm (RMSNorm)"),
                   (f"{p}.mlp.gate_proj", "outputs", 0, f"{p}.mlp.gate_proj"),
                   (f"{p}.mlp.up_proj", "outputs", 0, f"{p}.mlp.up_proj"),
                   (f"{p}.mlp.act", "outputs", 0, f"{p}.mlp.act (SiluMul)"),
                   (f"{p}.mlp.down_proj", "outputs", 0, f"{p}.mlp.down_proj")]
    points += [("model.norm", "inputs", 0, "residual add (last MLP)"), ("model.norm", "outputs", 0, "model.norm"),
               ("lm_head", "outputs", 0, "lm_head"), ("logprobs", "outputs", 0, "log_softmax")]
    walk, candidate, last_equal = [], None, None
    for name, kind, i, producer in points:
        lp = name == "logprobs"
        args = {"artifact": artifact, "left_ref": by["trainer"][name][0], "left_slot": [kind, i],
                "left_token_dims": 1 if lp else 2, "left_token_ids": t_lp_ids if lp else t_ids,
                "right_ref": by["rollout"][name][f], "right_slot": [kind, i],
                "right_token_dims": 1, "right_token_ids": r_lp_ids if lp else layout[f]}
        studied = row

        def keep(r, studied=studied):
            hit = next((x for x in r.get("rows", []) if x["token"] == studied), None)
            return {"status": r["status"], "unit": r.get("unit"), "summary": r.get("summary"), "studied_row": hit,
                    "widths": [r["left"]["vector_bytes"], r["right"]["vector_bytes"]]}
        res = agent("vectors.compare", args, keep=keep)
        hit = next((x for x in res.get("rows", []) if x["token"] == row), None)
        state = hit["state"] if hit else res["status"]
        walk.append({"point": f"{name}.{kind}[{i}]", "producer": producer, "studied_row": state,
                     "differing_vectors": hit.get("differing_vectors") if hit else None,
                     "forward_rows": res.get("summary")})
        if state == "equal":
            last_equal = walk[-1]
            if candidate is not None and "agreement_returns_at" not in candidate:
                candidate["agreement_returns_at"] = walk[-1]["point"]
        elif state == "differ" and candidate is None:
            candidate = {"op": producer, "observed_at": walk[-1]["point"], "input_agrees_at": last_equal and last_equal["point"],
                         "differing_vectors": walk[-1]["differing_vectors"],
                         "pair": "same input row on both sides, different output row of the same width"}
    out.update(walk=walk, candidate=candidate, forward_index=f, harness_calls=agent.calls)
    if validate:                                       # validate-hypothesis: undo the one suspected change
        req = str(workdir / "req-rollout.json")
        v = C.run_replay(req, cfg_name, dev, workdir / "records-validate.jsonl", rollout_fault=None)
        vb = [int(x) & 0xFFFFFFFF for lp in v["logprobs"] for x in lp.view(torch.int32).tolist()]
        out["validate"] = {"rollout_fault": None, "equal_to_trainer": sum(a == (b & 0xFFFFFFFF) for a, b in zip(vb, tr_bits)),
                           "tokens": len(vb)}
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--work", default=os.path.expanduser("~/bi-check-rl-loop/work"))
    ap.add_argument("--gpus", default="0,0", help="trainer,rollout cuda indices (after CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--only", default=None, help="comma list of scenarios")
    ap.add_argument("--transcript", default=None)
    args = ap.parse_args(argv)
    t_all = time.perf_counter()
    g = args.gpus.split(",")
    devs = (f"cuda:{g[0]}", f"cuda:{g[1]}")
    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    og = ob.locate_opguard(os.environ.get("OPGUARD_ROOT"), sys.executable)
    res = Results("rl_loop", quick=args.quick)
    steps = 3 if args.quick else 12
    fsteps = 3 if args.quick else 4
    cfg_name = "tiny"
    want = (args.only.split(",") if args.only else
            ["sync", "async1", "sync_bug"] if args.quick else
            ["sync", "async1", "async2", "stock", "norm", "sync_bug", "version", "cost"])
    extra = res.extra
    extra["devices"] = {"trainer": devs[0], "rollout": devs[1]}
    extra["opguard"] = {"root": og.get("root"), "capabilities_protocol": og.get("capabilities", {}).get("protocol_version")}
    rows = []
    nq = C.make_cfg(cfg_name).heads * C.make_cfg(cfg_name).head_dim

    for name in want:
        print(f"== {name}")
        agent = Agent(og, work / "harness", name)
        if name in ("sync", "async1", "async2"):
            lag = {"sync": 0, "async1": 1, "async2": 2}[name]
            run = run_loop(work / name, cfg_name, steps, lag, devs)
            sc = split_check(agent, run, devs[0], cfg_name, rows)
            t = sc["totals"]
            clean = all(v.get("trainer_unmatched_rows", 0) == 0 and v.get("rollout_unmatched_rows", 0) == 0
                        for v in sc["weight_sync"].values())
            res.check(f"{name}: rl.weight_sync_check clean at every pushed version (fused QKV on the rollout)",
                      clean and len(sc["weight_sync"]) > 0, level="L3", impl="rl.weight_sync_check",
                      detail={"versions": sorted(sc["weight_sync"]), "summary": sc["weight_sync"]})
            res.check(f"{name}: rl.decompose numeric exactly zero for every non-mixed token"
                      + (" (staleness-zero points)" if lag == 0 else " (after trainer replays from snapshots)"),
                      t["numeric_nonzero"] == 0 and t["numeric_missing"] == 0 and t["numeric_zero"] == t["tokens"] - t["mixed_version_tokens"],
                      level="L3", impl="rl.decompose", detail=t)
            if lag:
                res.check(f"{name}: mixed-version sequences flagged and not split", t["mixed_version_tokens"] > 0,
                          level="other", impl="rl.decompose", detail={"mixed_sequences": t["mixed_sequences"],
                                                                      "mixed_tokens": t["mixed_version_tokens"]})
                res.check(f"{name}: staleness carries the whole gap (nonzero staleness tokens)", True, level="other",
                          impl="rl.decompose", expect="info",
                          detail={"staleness_nonzero": t["staleness_nonzero"], "staleness_zero": t["staleness_zero"]})
                if sc["replay_s"]:
                    res.measure(f"{name}: trainer replay per batch (load snapshot + prefill + record)",
                                statistics.median(sc["replay_s"]) * 1e3, "ms", n=len(sc["replay_s"]))
            extra[name] = {"steps": run["steps"], "split": sc["totals"], "replays": len(sc["replays"]),
                           "harness_calls": agent.calls, "harness_s": round(agent.seconds, 1)}
            if name == "sync":
                extra["sync"]["record_bytes"] = run["record_bytes"]
                sync_run = run
        elif name in ("stock", "norm"):
            run = run_loop(work / name, cfg_name, fsteps, 0, devs, rollout_fault=name)
            sc = split_check(agent, run, devs[0], cfg_name, rows)
            t = sc["totals"]
            clean = all(v.get("trainer_unmatched_rows", 0) == 0 and v.get("rollout_unmatched_rows", 0) == 0
                        for v in sc["weight_sync"].values())
            res.check(f"{name}: fault detected: weight sync clean, numeric != 0 at staleness-zero points",
                      clean and t["numeric_nonzero"] > 0, level="other", impl="rl.decompose", detail=t)
            k = 1
            loc = localize(agent, run, k, k, cfg_name, devs[0], name, work / name / "localize",
                           validate=True)
            cand = loc["candidate"] or {}
            expected = {"norm": "model.layers.0.input_layernorm (RMSNorm)"}.get(name)
            res.check(f"{name}: localized by recapture + vectors.compare to a paired first candidate"
                      + (f" ({expected})" if expected else ""),
                      bool(cand) and (expected is None or cand.get("op") == expected) and loc["recapture_reproduces_online_rollout"],
                      level="other", impl="vectors.compare",
                      detail={k2: loc[k2] for k2 in ("token", "candidate", "recapture_reproduces_online_rollout",
                                                    "forward_index", "harness_calls")})
            res.check(f"{name}: validate: rollout replay without the fault equals the trainer",
                      loc["validate"]["equal_to_trainer"] == loc["validate"]["tokens"] if "validate" in loc else False,
                      level="other", impl="validate-hypothesis", detail=loc.get("validate"))
            extra[name] = {"split": t, "localize": loc, "steps": run["steps"]}
        elif name == "sync_bug":
            L = C.make_cfg(cfg_name).layers
            faults = {1: {"kind": "bitflip", "param": "model.layers.0.self_attn.qkv_proj.weight", "row": nq + 3, "col": 5}}
            if not args.quick:
                faults[2] = {"kind": "stale_row", "param": f"model.layers.{L - 1}.mlp.down_proj.weight", "row": 7}
            run = run_loop(work / name, cfg_name, fsteps, 0, devs, sync_faults=faults)
            records = run["records"]
            found = {}
            for v in faults:
                d = agent("rl.weight_sync_check", {"records": records, "version": v}, keep=trim_wsc)
                un = {side: [(p["name"], p["unmatched_rows"]) for p in d[side] if p.get("unmatched_count")]
                      for side in ("trainer", "rollout")}
                found[v] = un
            d0 = agent("rl.weight_sync_check", {"records": records, "version": 0}, keep=trim_wsc)
            ok1 = found[1] == {"trainer": [("model.layers.0.self_attn.k_proj.weight", [3])],
                               "rollout": [("model.layers.0.self_attn.qkv_proj.weight", [nq + 3])]}
            res.check("sync_bug: bit flip after repacking named (trainer k_proj row 3 = rollout qkv_proj row nq+3)",
                      ok1, level="other", impl="rl.weight_sync_check", detail=found[1])
            if 2 in faults:
                p = f"model.layers.{L - 1}.mlp.down_proj.weight"
                ok2 = found[2] == {"trainer": [(p, [7])], "rollout": [(p, [7])]}
                res.check("sync_bug: stale row named (down_proj row 7 on both sides)", ok2, level="other",
                          impl="rl.weight_sync_check", detail=found[2])
            res.check("sync_bug: clean version stays clean", d0["summary"].get("trainer_unmatched_rows", 1) == 0,
                      level="other", impl="rl.weight_sync_check", detail=d0["summary"])
            dec = {v: agent("rl.decompose", {"records": records, "batch_id": v, "trainer_version": v, "policy_version_override": None},
                            keep=trim_decompose)["summary"] for v in faults}
            extra[name] = {"weight_sync_unmatched": {str(v): x for v, x in found.items()},
                           "decompose_at_faulty_versions": {str(v): x for v, x in dec.items()}}
        elif name == "version":
            run = run_loop(work / name, cfg_name, fsteps, 0, devs, defer_version=2)
            sc = split_check(agent, run, devs[0], cfg_name, rows)
            per = {r["k"]: r["after_replay"] for r in rows[-fsteps:]}
            clean = all(v.get("trainer_unmatched_rows", 0) == 0 and v.get("rollout_unmatched_rows", 0) == 0
                        for v in sc["weight_sync"].values())
            res.check("version: weight sync clean, numeric != 0 only in batch 2 (all tokens staleness-zero points)",
                      clean and per[2]["numeric_nonzero"] == per[2]["tokens"] and
                      all(per[k]["numeric_nonzero"] == 0 for k in per if k != 2),
                      level="other", impl="rl.decompose", detail=per)
            loc = localize(agent, run, 2, 2, cfg_name, devs[0], None, work / name / "localize")
            res.check("version: recapture at the recorded version does not reproduce the online log-probs "
                      "(and matches the trainer exactly)",
                      (not loc["recapture_reproduces_online_rollout"]) and
                      loc["recapture_rollout_vs_trainer_equal_tokens"] == loc["tokens"] and loc["candidate"] is None,
                      level="other", impl="rl.replay_request + vectors.compare",
                      detail={k2: loc[k2] for k2 in ("recapture_reproduces_online_rollout",
                                                    "recapture_rollout_vs_trainer_equal_tokens", "tokens", "candidate")})
            # The batch's observed weights fingerprint against the recorded syncs.
            wc = agent("rl.decompose", {"records": run["records"], "batch_id": 2, "trainer_version": 2,
                                        "policy_version_override": None},
                       keep=lambda r: r["weights_check"])["weights_check"]
            others = [agent("rl.decompose", {"records": run["records"], "batch_id": k2, "trainer_version": k2,
                                             "policy_version_override": None},
                            keep=lambda r: r["weights_check"])["weights_check"]["first_labelled_version_matches"]
                      for k2 in per if k2 != 2]
            res.check("version: batch 2's recorded weights fingerprint matches the version-1 sync, not its label 2; "
                      "every other batch matches its label",
                      wc["first_labelled_version_matches"] is False and wc["rollout_sync_versions_matching"] == [1]
                      and all(others), level="other", impl="rl.decompose weights_check",
                      detail={"batch2": wc, "others_match": others})
            # hypothesis: the batch ran at version 1. Replay the trainer at 1; decompose under the override.
            out = str(run["dir"] / "replay-req-b2-v1.json")
            agent("rl.replay_request", {"records": run["records"], "batch_id": 2, "version": 1, "side": "trainer",
                                        "composition": "original", "capture": None, "output": out},
                  keep=lambda r: {k2: r[k2] for k2 in ("weights_located", "reproduces")})
            rp = run["dir"] / "records-replay-b2-v1.jsonl"
            C.run_replay(out, cfg_name, devs[0], rp)
            d = agent("rl.decompose", {"records": run["records"] + [str(rp)], "batch_id": 2, "trainer_version": 2,
                                       "policy_version_override": 1},
                      keep=lambda r: {**trim_decompose(r), "policy_version_source": r["policy_version_source"]})
            s = d["summary"]
            res.check("version: hypothesis 'batch 2 ran at version 1' (caller-supplied version) gives numeric zero "
                      "on every token", s["numeric_zero"] == s["tokens"] and d["policy_version_source"] == "caller_supplied",
                      level="other", impl="validate-hypothesis", detail=s)
            extra[name] = {"split_per_batch": per, "localize": loc, "hypothesis": s}
        elif name == "cost":
            for cfg_c, st in (("tiny", 6), ("default", 4)):
                on = run_loop(work / f"cost-{cfg_c}-on", cfg_c, st, 0, devs, snap_every=2)
                off = run_loop(work / f"cost-{cfg_c}-off", cfg_c, st, 0, devs, record=False)
                med = lambda run, key: statistics.median(s.get(key, 0.0) for s in run["steps"][1:]) * 1e3
                for key in ("step_s", "gen_s", "train_s", "fingerprint_trainer_s", "fingerprint_rollout_s",
                            "record_rollout_s", "record_trainer_s", "snapshot_s", "trainer_logprob_s"):
                    res.measure(f"cost {cfg_c}: {key[:-2]} per step, records on", med(on, key), "ms", config=cfg_c)
                res.measure(f"cost {cfg_c}: step, records off", med(off, "step_s"), "ms", config=cfg_c)
                res.measure(f"cost {cfg_c}: record bytes per step", on["record_bytes"] / st, "bytes", config=cfg_c)
                for d in (on["dir"], off["dir"]):
                    shutil.rmtree(d / "snap", ignore_errors=True)
        print(f"   harness calls {agent.calls}, {agent.seconds:.1f} s")

    if "sync" in want:
        st = extra["sync"]["steps"]
        for key in ("fingerprint_trainer_s", "fingerprint_rollout_s", "record_rollout_s", "record_trainer_s",
                    "snapshot_s", "step_s"):
            res.measure(f"sync tiny: {key[:-2]} per step", statistics.median(s.get(key, 0.0) for s in st[1:]) * 1e3,
                        "ms", config="tiny")
    extra["split_rows"] = rows
    extra["runtime_s"] = round(time.perf_counter() - t_all, 1)
    res.write(args.json)
    if args.transcript:
        with open(args.transcript, "w") as f:
            json.dump(TRANSCRIPT, f, indent=1, default=str)
    return res.exit_code()


if __name__ == "__main__":
    sys.exit(main())
