"""rl_core: the pieces of a small, real RL loop wired to OpGuard's RL harness records.

rollout   examples/whole_model/model.py in engine mode (slot KV cache, rows join and leave
          the batch), tools/bi_sampling.py (counter-keyed sampler), Q/K/V repacked into one
          fused qkv_proj weight. Runs in its own process (Rollout + rollout_worker).
trainer   model.py prefill (teacher forced, padded batch) + AdamW on fp32 master weights;
          the bf16 weights are what is pushed.
records   opguard_harness.rl_records.RecordWriter at the documented points: forward,
          rollout_batch (per-token version, logprob, composition), trainer_logprob,
          weight_sync (per-row fingerprints of both sides, rollout side after repacking),
          snapshot.
offline   run_replay(): executes a replay request built by rl.replay_request, optionally
          with per-vector capture (opguard_bridge.TraceWriter(vectors=True), which calls
          OpGuard's own fingerprint functions) at every module boundary.

Planted faults (Rollout.fault / load faults): "stock" (the rollout runs the stock op set
although its flags say batch-invariant), "norm" (RMSNorm multiplies by its weight in fp32,
then rounds), "bitflip" / "stale_row" (weight-sync bugs after repacking), "defer" (the
rollout labels a batch with the version it was sent but applies it only after the batch).
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
HB = HERE.parent.parent
for p in (HB / "examples" / "whole_model", HB / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch                                    # noqa: E402
import model as M                               # noqa: E402
import bi_sampling as S                         # noqa: E402
from opguard_harness.rl_records import RecordWriter, param_entries, weights_fingerprint   # noqa: E402

TARGET_MOD = 8                                  # reward: share of response tokens with id % 8 == 0
CAPACITY = 32


def make_cfg(name: str) -> M.Config:
    return M.Config.tiny() if name == "tiny" else M.Config()


def sync(dev=None):
    torch.cuda.synchronize(dev)


# ----------------------------------------------------------------------------- batches
def batch_spec(k: int, cfg: M.Config, groups: int = 4, n: int = 4, seed: int = 0) -> list:
    """Deterministic prompts for batch k: `groups` prompts, n samples each. Rows join the engine
    in three waves (engine steps 0, 2 and 5) and stop after 3..12 tokens."""
    g = torch.Generator().manual_seed(10007 * seed + k)
    B = groups * n
    waves = [0] * (B // 2) + [2] * (B // 4) + [5] * (B - B // 2 - B // 4)
    seqs = []
    for gi in range(groups):
        P = int(torch.randint(3, 9, (1,), generator=g))
        prompt = torch.randint(1, cfg.vocab_size, (P,), generator=g).tolist()
        for j in range(n):
            i = gi * n + j
            R = int(torch.randint(3, 13, (1,), generator=g))
            seqs.append({"seq_id": i, "group": gi, "prompt": prompt, "max_new": R, "admit": waves[i],
                         "seed": 1_000_003 * (k + 1) + i})
    return seqs


def token_logprobs(model, rows, ids):
    """The one log-probability formula both sides use: fp32 log_softmax of the bf16 logit rows."""
    return model.logprobs(rows).gather(1, ids[:, None])[:, 0]


# ----------------------------------------------------------------------------- rollout
def _fused_qkv(self, a, pos):
    """Attention.qkv on the rollout: one GEMM on the fused [q; k; v] weight, then split."""
    c = self.ctx.cfg
    D, nq, nk = c.head_dim, c.heads * c.head_dim, c.kv_heads * c.head_dim
    lead = a.shape[:-1]
    y = self.qkv_proj(a)
    q, k, v = y.split([nq, nk, nk], -1)
    q = q.contiguous().view(*lead, c.heads, D)
    k = k.contiguous().view(*lead, c.kv_heads, D)
    v = v.contiguous().view(*lead, c.kv_heads, D)
    q, k = self.rope(q, k, pos)
    return q, k, v


def fuse_qkv(m: M.WholeModel) -> None:
    c = m.cfg
    for blk in m.model.layers:
        att = blk.self_attn
        nq, nk = c.heads * c.head_dim, c.kv_heads * c.head_dim
        lin = M.Linear(m.ctx, c.hidden, nq + 2 * nk).to(device=att.q_proj.weight.device, dtype=att.q_proj.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(torch.cat([att.q_proj.weight, att.k_proj.weight, att.v_proj.weight]))
        del att.q_proj, att.k_proj, att.v_proj
        att.qkv_proj = lin
        att.qkv = types.MethodType(_fused_qkv, att)


class Rollout:
    """The rollout engine. `fault`: None | "stock" | "norm" (code faults, present in replays too)."""

    def __init__(self, cfg: M.Config, device, fault=None):
        self.cfg, self.device, self.fault = cfg, device, fault
        self.model = M.build(cfg, ops="bi", seed=12345, device=device)    # overwritten by the first push
        fuse_qkv(self.model)
        self.params = dict(self.model.named_parameters())
        self.version = None
        self.pending = None
        self.forwards = 0

    # what the engine claims about itself (the "stock" fault does not change the claim)
    engine_flags = {"batch_invariant": True, "cuda_graph": False, "attention_backend": "bi_serve_paged",
                    "sampler": "bi_sampling.sample(exp=torch)", "qkv": "fused"}

    def using(self):
        return self.model.using(ops="stock" if self.fault == "stock" else "bi",
                                norm="weight_then_cast" if self.fault == "norm" else "cast_then_weight")

    @torch.no_grad()
    def load(self, state: dict, version: int, fault=None) -> None:
        """Copy a trainer state (q/k/v separate) into the fused layout, then apply a sync fault."""
        c = self.cfg
        nq, nk = c.heads * c.head_dim, c.kv_heads * c.head_dim
        off = {"q_proj": 0, "k_proj": nq, "v_proj": nq + nk}
        stale = None
        if fault and fault.get("kind") == "stale_row":
            stale = (fault["param"], fault["row"])
        for name, t in state.items():
            parts = name.split(".")
            if parts[-2] in off:
                dst = self.params[".".join(parts[:-2] + ["qkv_proj", "weight"])]
                a = off[parts[-2]]
                dst[a:a + t.shape[0]].copy_(t, non_blocking=False)
            else:
                dst = self.params[name]
                if stale and name == stale[0]:
                    keep = dst[stale[1]].clone()
                    dst.copy_(t)
                    dst[stale[1]].copy_(keep)
                else:
                    dst.copy_(t)
        if fault and fault.get("kind") == "bitflip":
            w = self.params[fault["param"]]
            w.view(torch.int16)[fault["row"], fault["col"]] ^= 1        # lowest mantissa bit
        self.version = version

    def entries(self):
        return param_entries(self.params)

    @torch.no_grad()
    def generate(self, k: int, spec: list, push=None, label_offset: int = 0, rec=None, teacher=None,
                 composition=None):
        """Run one rollout batch. push = (engine_step, state, version, fault): weights replaced
        between two forwards, KV cache kept. teacher/composition: replay mode (feed recorded
        tokens in the recorded co-batching instead of sampling). Returns (seqs, composition, info)."""
        m, dev = self.model, self.device
        cache = M.KVCache(m, len(spec), CAPACITY)
        st = {s["seq_id"]: {"spec": s, "out": [], "lp": [], "ver": [], "pos": []} for s in spec}
        comp, info, step, f = [], {"push_entries": None, "forward_s": 0.0}, 0, 0
        # Observed: the weights this batch starts on (XOR of all row fingerprints; any packing agrees).
        info["weights_fingerprint"] = weights_fingerprint(self.entries()) if rec is not None else None
        with self.using():
            while True:
                if push is not None and step == push[0]:
                    self.load(push[1], push[2], push[3])
                    info["push_entries"] = self.entries()
                if composition is not None:
                    if f == len(composition):
                        break
                    active = [st[sid] for sid in composition[f]]
                else:
                    active = [x for x in st.values() if x["spec"]["admit"] <= step and len(x["out"]) < x["spec"]["max_new"]]
                    if not active:
                        if all(len(x["out"]) >= x["spec"]["max_new"] for x in st.values()):
                            break
                        step += 1
                        continue
                toks, pos_next, rows = [], [], []
                for x in active:
                    s = x["spec"]
                    if not x.get("started"):
                        toks.append(list(s["prompt"]))
                        rows += [[s["seq_id"], p] for p in range(len(s["prompt"]))]
                        x["started"] = True
                    else:
                        toks.append([x["out"][-1]])
                        rows.append([s["seq_id"], len(s["prompt"]) + len(x["out"]) - 1])
                    pos_next.append(len(s["prompt"]) + len(x["out"]))
                ntok = sum(len(t) for t in toks)
                if rec is not None:          # rows: the observed (seq_id, position) of each of the ntok rows
                    rec.forward(counter=self.forwards if rec_counter(rec) is None else rec_counter(rec), rank=0,
                                tokens=ntok, rows=rows)
                self.forwards += 1
                logits = m.forward_cached(cache, [x["spec"]["seq_id"] for x in active],
                                          [torch.tensor(t, device=dev) for t in toks])
                last = torch.stack([l[-1] for l in logits])
                if teacher is not None:
                    tok = torch.tensor([teacher[x["spec"]["seq_id"]][len(x["out"])] for x in active], device=dev)
                else:
                    tok = S.sample(last, [x["spec"]["seed"] for x in active], pos_next, temperature=1.0)
                lp = token_logprobs(m, last, tok)
                if getattr(self, "capture", None) is not None:
                    self.capture.record("logprobs", inputs=[last], outputs=[m.logprobs(last)], file="model.py", line=1)
                tok_l, lp_l = tok.tolist(), lp.cpu()
                for i, x in enumerate(active):
                    x["out"].append(tok_l[i])
                    x["lp"].append(lp_l[i])
                    x["ver"].append(self.version + label_offset)
                    x["pos"].append(pos_next[i])
                comp.append([x["spec"]["seq_id"] for x in active])
                step += 1
                f += 1
        if self.pending is not None:                    # "defer" fault: the push is applied only now
            state, version = self.pending
            self.pending = None
            self.load(state, version)
            info["push_entries"] = self.entries()
        seqs = [{"seq_id": sid, "token_ids": x["out"], "positions": x["pos"], "policy_version": x["ver"],
                 "logprob": torch.stack(x["lp"]), "prompt_ids": list(x["spec"]["prompt"])}
                for sid, x in st.items()]
        return seqs, comp, info


def rec_counter(rec):
    """Forward records in a capture carry the OpGuard counter of the forward's first record."""
    w = getattr(rec, "trace_writer", None)
    return None if w is None else w.counter + 1


def rollout_worker(conn, cfg_name: str, device: str, fault, rec_path):
    """The rollout process: a command loop over a Pipe. Weights arrive as CPU tensors."""
    torch.cuda.set_device(device)
    eng = Rollout(make_cfg(cfg_name), device, fault)
    rec = RecordWriter(rec_path) if rec_path else None
    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "stop":
            break
        if cmd == "load":
            _, state, version, lfault = msg
            t0 = time.perf_counter()
            if lfault and lfault.get("kind") == "defer":
                eng.pending = (state, version)
                conn.send({"entries": None, "load_s": time.perf_counter() - t0, "entries_s": 0.0, "deferred": True})
                continue
            eng.load(state, version, lfault)
            sync()
            t1 = time.perf_counter()
            ents = eng.entries() if rec is not None else None
            conn.send({"entries": ents, "load_s": t1 - t0, "entries_s": time.perf_counter() - t1})
        elif cmd == "generate":
            _, k, spec, push, label_offset = msg
            t0 = time.perf_counter()
            seqs, comp, info = eng.generate(k, spec, push=push, label_offset=label_offset, rec=rec)
            sync()
            t1 = time.perf_counter()
            if rec is not None:
                rec.rollout_batch(k, seqs, composition=comp, engine_flags=dict(eng.engine_flags),
                                  weights_fingerprint=info["weights_fingerprint"])
            t2 = time.perf_counter()
            conn.send({"seqs": [{**s, "logprob": s["logprob"].tolist()} for s in seqs], "composition": comp,
                       "push_entries": info["push_entries"], "gen_s": t1 - t0, "record_s": t2 - t1,
                       "version_after": eng.version})
    if rec is not None:
        rec.close()
    conn.send("bye")


# ----------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg: M.Config, device, lr: float):
        self.cfg, self.device = cfg, device
        self.model = M.build(cfg, ops="bi", seed=0, device=device)
        self.master = [p.detach().float().clone().requires_grad_(True) for p in self.model.parameters()]
        self.opt = torch.optim.AdamW(self.master, lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
        self.version = 0
        self.capture = None
        self.forwards = 0

    def state(self) -> dict:
        return {n: p.detach().to("cpu", copy=True) for n, p in self.model.named_parameters()}

    def entries(self):
        return param_entries(dict(self.model.named_parameters()))

    def load_state(self, state: dict) -> None:
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.copy_(state[n])

    def _rows(self, seqs):
        """Right-padded ids [B, T] of prompt + response; for each response token at position p,
        the row (b, p - 1) that scores it."""
        full = [list(s["prompt_ids"]) + list(s["token_ids"]) for s in seqs]
        T = max(len(f) for f in full)
        ids = torch.zeros(len(full), T, dtype=torch.long)
        for b, f in enumerate(full):
            ids[b, :len(f)] = torch.tensor(f)
        bi = [b for b, s in enumerate(seqs) for _ in s["positions"]]
        ti = [p - 1 for s in seqs for p in s["positions"]]
        return ids.to(self.device), torch.tensor(bi, device=self.device), torch.tensor(ti, device=self.device), full

    @torch.no_grad()
    def logprobs(self, seqs, rec=None):
        ids, bi, ti, full = self._rows(seqs)
        if rec is not None:
            self.forwards += 1
            c = rec_counter(rec)
            T = ids.shape[1]          # observed layout of the right-padded [B, T] batch; padding rows are null
            rows = [[s["seq_id"], p] if p < len(f) else None for s, f in zip(seqs, full) for p in range(T)]
            rec.forward(counter=self.forwards if c is None else c, rank=0, tokens=int(ids.numel()), rows=rows)
        logits = self.model(ids)
        rows = logits[bi, ti]
        tok = ids[bi, ti + 1]
        lp = token_logprobs(self.model, rows, tok)
        if self.capture is not None:
            self.capture.record("logprobs", inputs=[rows], outputs=[self.model.logprobs(rows)], file="model.py", line=1)
        out, a = [], 0
        for s in seqs:
            n = len(s["positions"])
            out.append(lp[a:a + n].cpu())
            a += n
        return out

    def train(self, seqs, rewards, groups):
        """One REINFORCE step with group-normalized advantages (GRPO without clipping)."""
        r = torch.tensor(rewards, dtype=torch.float64)
        adv = torch.zeros_like(r)
        for g in set(groups):
            idx = [i for i, x in enumerate(groups) if x == g]
            rg = r[idx]
            adv[idx] = (rg - rg.mean()) / (rg.std(unbiased=False) + 1e-6)
        ids, bi, ti, _ = self._rows(seqs)
        logits = self.model(ids)
        lp = token_logprobs(self.model, logits[bi, ti], ids[bi, ti + 1])
        a = torch.tensor([float(adv[b]) for b in bi.tolist()], device=self.device)
        loss = -(a * lp).sum() / lp.numel()
        loss.backward()
        for m, p in zip(self.master, self.model.parameters()):
            m.grad = None if p.grad is None else p.grad.float()
            p.grad = None
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for m, p in zip(self.master, self.model.parameters()):
                p.copy_(m)
        self.version += 1
        return float(loss)


def reward(seq) -> float:
    toks = seq["token_ids"]
    return sum(1 for t in toks if t % TARGET_MOD == 0) / len(toks)


# ----------------------------------------------------------------------------- capture
CAPTURED = (M.Linear, M.RMSNorm, M.RoPE, M.SiluMul, M.AttnCore, torch.nn.Embedding)


def attach_capture(model, writer):
    """Per-vector capture at every module boundary of model.py (OpGuard's fingerprint functions
    through opguard_bridge.TraceWriter(vectors=True))."""
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, CAPTURED):
            def hook(m, args, out, name=name):
                ins = [a for a in args if torch.is_tensor(a)]
                outs = list(out) if isinstance(out, tuple) else [out]
                writer.record(name, inputs=ins, outputs=outs, file="model.py", line=1)
            handles.append(mod.register_forward_hook(hook))
    return handles


# ----------------------------------------------------------------------------- offline replay
def read_ref(ref: str) -> dict:
    path, line = ref.rsplit(":", 1)
    with open(path) as f:
        for n, text in enumerate(f, 1):
            if n == int(line):
                return json.loads(text)
    raise ValueError(f"no line {ref}")


def run_replay(request_path, cfg_name: str, device, out_records, rollout_fault=None, capture_root=None,
               batch_label=None, force_version_weights=None):
    """Execute a replay request (schema opguard.rl.replay.v1) offline.

    trainer side: trainer_logprob at the request's version for the batch.
    rollout side: the engine re-run with the recorded co-batching, the recorded tokens fed back
    (teacher forced); writes a rollout_batch under batch_label.
    The request carries each sequence's prompt_ids (recorded on rollout_batch)."""
    import opguard_bridge as ob
    req = json.loads(Path(request_path).read_text())
    assert all(s.get("prompt_ids") is not None for s in req["sequences"]), "request without prompt_ids"
    cfg = make_cfg(cfg_name)
    t0 = time.perf_counter()
    state = torch.load(req["weights"]["location"], map_location="cpu", weights_only=True)
    t_load = time.perf_counter() - t0
    rec = RecordWriter(out_records)
    writer = ob.TraceWriter(capture_root, vectors=True) if capture_root else None
    rec.trace_writer = writer
    seqs = [{"seq_id": s["seq_id"], "token_ids": s["token_ids"], "positions": s["positions"],
             "prompt_ids": s["prompt_ids"]} for s in req["sequences"]]
    sync()
    t1 = time.perf_counter()
    if req["side"] == "trainer":
        tr = Trainer.__new__(Trainer)
        tr.cfg, tr.device, tr.capture, tr.forwards = cfg, device, writer, 0
        tr.model = M.build(cfg, ops="bi", seed=0, device=device)
        tr.load_state(state)
        hs = attach_capture(tr.model, writer) if writer else []
        lps = tr.logprobs(seqs, rec)
        for h in hs:
            h.remove()
        rec.trainer_logprob(req["batch_id"], req["version"],
                            [{"seq_id": s["seq_id"], "positions": s["positions"], "logprob": lp} for s, lp in zip(seqs, lps)],
                            engine_flags={"side": "trainer", "replay": True})
        result = lps
    else:
        eng = Rollout(cfg, device, rollout_fault)
        eng.capture = writer
        eng.load(state, req["version"])
        hs = attach_capture(eng.model, writer) if writer else []
        spec = [{"seq_id": s["seq_id"], "prompt": s["prompt_ids"], "max_new": len(s["token_ids"]), "admit": 0,
                 "seed": 0} for s in seqs]
        teacher = {s["seq_id"]: s["token_ids"] for s in seqs}
        out, comp, _ = eng.generate(-1, spec, rec=rec, teacher=teacher, composition=req["forward_composition"])
        for h in hs:
            h.remove()
        rec.rollout_batch(batch_label or f"{req['batch_id']}@rollout-replay", out, composition=comp,
                          engine_flags={**eng.engine_flags, "replay": True})
        result = [s["logprob"] for s in out]
    sync()
    t_run = time.perf_counter() - t1
    if writer:
        writer.close()
    rec.close()
    return {"load_s": t_load, "run_s": t_run, "logprobs": result}
