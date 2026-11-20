"""Checkpoint conversion check, on CPU: is a converted checkpoint a bit-exact copy?

A random Llama-shaped state dict (Hugging Face names, bf16) is converted by convert.py to
a Megatron-style layout (fused QKV interleaved per query group, fused gate/up per rank,
TP shards, vocab-parallel embedding and output layer) at TP 2 and TP 4, and back. Then:

  1. the round trip HF -> Megatron-style -> HF is bit-exact, tensor by tensor;
  2. tools/ckpt_compare.py, with mcore_map.json, matches every row of every logical
     parameter between the HF and the Megatron-style checkpoints, and between TP 2 and TP 4;
  3. a forward on fixed token ids at the target layout (the shards read the way the
     Megatron-style layers read them) gives the same logits, bit for bit, as the source
     weights run with the same reduction (row-parallel partial sums in rank order);
  4. planted faults are found and named by parameter and row: a transposed shard, a
     wrong QKV interleave, a bf16 -> fp16 -> bf16 cast, one flipped bit, a missing row;
  5. two faults a row multiset cannot see, by design, are caught by the forward check:
     two heads swapped inside q, and row-parallel shards joined in the wrong order (seen by
     the default row digests, not by XOR row fingerprints);
  6. XOR row fingerprints (--rows xor) name the same rows, except where two elements of a
     row change by the same bits in the same lane (recorded for the cast).

    python ckpt_conversion.py --quick --json results.json     # small model
    python ckpt_conversion.py --json results.json             # larger model
Writes bi-results-v1 (tools/bi_report.py). Needs torch; transformers is optional (an
extra info check that the plain forward equals LlamaForCausalLM's).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(HERE))
import ckpt_compare as cc  # noqa: E402
import convert as cv  # noqa: E402
from bi_report import Results  # noqa: E402

CONFIGS = {
    "quick": dict(vocab=512, hidden=128, intermediate=256, layers=2, heads=8, kv_heads=4, head_dim=16,
                  eps=1e-5, theta=10000.0),
    "full": dict(vocab=2048, hidden=512, intermediate=1024, layers=4, heads=16, kv_heads=4, head_dim=32,
                 eps=1e-5, theta=10000.0),
}
TPS = (2, 4)
MAP = HERE / "mcore_map.json"


# ============================================================================ the model
def make_state_dict(cfg: dict, seed: int = 0) -> dict:
    """Random weights with LlamaForCausalLM's names and shapes, bf16; norms near 1, not 1."""
    g = torch.Generator().manual_seed(seed)
    H, I, V, hd = cfg["hidden"], cfg["intermediate"], cfg["vocab"], cfg["head_dim"]
    nq, nkv = cfg["heads"] * hd, cfg["kv_heads"] * hd
    shapes = {"self_attn.q_proj.weight": (nq, H), "self_attn.k_proj.weight": (nkv, H),
              "self_attn.v_proj.weight": (nkv, H), "self_attn.o_proj.weight": (H, nq),
              "mlp.gate_proj.weight": (I, H), "mlp.up_proj.weight": (I, H), "mlp.down_proj.weight": (H, I)}
    sd = {}
    for name in cv.hf_names(cfg):
        if name.endswith("norm.weight"):
            t = 1.0 + 0.1 * torch.randn(H, generator=g)
        elif name in ("model.embed_tokens.weight", "lm_head.weight"):
            t = 0.02 * torch.randn(V, H, generator=g)
        else:
            t = 0.02 * torch.randn(*shapes[name.split(".", 3)[3]], generator=g)
        sd[name] = t.to(torch.bfloat16)
    return sd


def rms(x, w, eps):                                   # LlamaRMSNorm
    dt = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(dt)


def rope(B, T, hd, theta, dtype):                     # LlamaRotaryEmbedding, default type
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
    pos = torch.arange(T)[None].expand(B, T)
    freqs = (inv[None, :, None].float().expand(B, -1, 1) @ pos[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def attention(q, k, v, cos, sin, cfg):                # eager attention, causal
    B, T, _ = q.shape
    hd, G = cfg["head_dim"], cfg["kv_heads"]
    rep = cfg["heads"] // G
    q, k, v = (t.view(B, T, -1, hd).transpose(1, 2) for t in (q, k, v))
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    q = (q * c) + (rotate_half(q) * s)
    k = (k * c) + (rotate_half(k) * s)
    k = k[:, :, None].expand(B, G, rep, T, hd).reshape(B, G * rep, T, hd)
    v = v[:, :, None].expand(B, G, rep, T, hd).reshape(B, G * rep, T, hd)
    w = torch.matmul(q, k.transpose(2, 3)) * hd ** -0.5
    w = w + torch.full((T, T), torch.finfo(q.dtype).min, dtype=q.dtype).triu(1)
    w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(w, v).transpose(1, 2).contiguous().reshape(B, T, -1)


def forward(P, ids, cfg):
    """The Llama forward; P says how each layer reads its weights (one layout or the other)."""
    B, T = ids.shape
    h = P.embed(ids)
    cos, sin = rope(B, T, cfg["head_dim"], cfg["theta"], h.dtype)
    for i in range(cfg["layers"]):
        q, k, v = P.qkv(i, rms(h, P.norm(i, "attn"), cfg["eps"]))
        h = h + P.o(i, attention(q, k, v, cos, sin, cfg))
        gate, up = P.gate_up(i, rms(h, P.norm(i, "mlp"), cfg["eps"]))
        h = h + P.down(i, F.silu(gate) * up)
    return P.lm_head(rms(h, P.norm(None, "final"), cfg["eps"]))


def row_parallel(x, shards):
    """A row-parallel linear at TP = len(shards): partial GEMMs on the column blocks, summed in
    rank order in fp32, one cast. Used by both sides, so the reduction is the same."""
    acc = None
    for part, w in zip(x.split([w.shape[1] for w in shards], dim=-1), shards):
        y = F.linear(part, w).float()
        acc = y if acc is None else acc + y
    return acc.to(x.dtype)


class HFWeights:
    """The source: HF tensors; row-parallel layers reduced as at TP = tp."""

    def __init__(self, sd, cfg, tp=1):
        self.sd, self.cfg, self.tp = sd, cfg, tp

    def w(self, i, name):
        return self.sd[f"model.layers.{i}.{name}"]

    def embed(self, ids):
        return F.embedding(ids, self.sd["model.embed_tokens.weight"])

    def norm(self, i, kind):
        return (self.sd["model.norm.weight"] if kind == "final" else
                self.w(i, "input_layernorm.weight" if kind == "attn" else "post_attention_layernorm.weight"))

    def qkv(self, i, x):
        return tuple(F.linear(x, self.w(i, f"self_attn.{p}_proj.weight")) for p in "qkv")

    def o(self, i, a):
        return row_parallel(a, self.w(i, "self_attn.o_proj.weight").chunk(self.tp, 1))

    def gate_up(self, i, x):
        return F.linear(x, self.w(i, "mlp.gate_proj.weight")), F.linear(x, self.w(i, "mlp.up_proj.weight"))

    def down(self, i, m):
        return row_parallel(m, self.w(i, "mlp.down_proj.weight").chunk(self.tp, 1))

    def lm_head(self, h):
        return F.linear(h, self.sd["lm_head.weight"])


class MegatronWeights:
    """The target: Megatron-style shards, read the way its layers read them. Column-parallel
    outputs are computed from the rank shards reassembled per the layout (one GEMM per
    projection, the source's GEMM shape); row-parallel layers sum rank partials in rank order;
    the embedding is a vocab-parallel lookup; each norm is rank 0's copy."""

    def __init__(self, ranks, cfg):
        self.ranks, self.cfg = ranks, cfg
        self.tp = len(ranks)

    def shards(self, name):
        return [rk[name] for rk in self.ranks]

    def embed(self, ids):
        out, lo = None, 0
        for E in self.shards("embedding.word_embeddings.weight"):
            local = ids - lo
            mask = (local >= 0) & (local < E.shape[0])
            e = torch.where(mask[..., None], F.embedding(local.clamp(0, E.shape[0] - 1), E),
                            torch.zeros((), dtype=E.dtype))
            out = e if out is None else out + e
            lo += E.shape[0]
        return out

    def norm(self, i, kind):
        name = ("decoder.final_layernorm.weight" if kind == "final" else
                f"decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight" if kind == "attn" else
                f"decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight")
        return self.ranks[0][name]

    def qkv(self, i, x):
        G, hd, H = self.cfg["kv_heads"], self.cfg["head_dim"], self.cfg["hidden"]
        qpg = self.cfg["heads"] // G
        w = torch.cat(self.shards(f"decoder.layers.{i}.self_attention.linear_qkv.weight"), 0)
        q, k, v = w.view(G, (qpg + 2) * hd, H).split([qpg * hd, hd, hd], dim=1)
        return tuple(F.linear(x, t.reshape(-1, H)) for t in (q, k, v))

    def o(self, i, a):
        return row_parallel(a, self.shards(f"decoder.layers.{i}.self_attention.linear_proj.weight"))

    def gate_up(self, i, x):
        halves = [w.chunk(2, 0) for w in self.shards(f"decoder.layers.{i}.mlp.linear_fc1.weight")]
        return (F.linear(x, torch.cat([g for g, _ in halves], 0)),
                F.linear(x, torch.cat([u for _, u in halves], 0)))

    def down(self, i, m):
        return row_parallel(m, self.shards(f"decoder.layers.{i}.mlp.linear_fc2.weight"))

    def lm_head(self, h):
        return F.linear(h, torch.cat(self.shards("output_layer.weight"), 0))


def bits_equal(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


def differing(a, b) -> int:
    return int((a.view(torch.int16) != b.view(torch.int16)).sum()) if a.shape == b.shape else -1


def forward_check(ranks, hf, cfg, ids):
    """Target-layout forward vs the source forward with the target's reduction. A target that
    cannot even be read (a shape error) counts as a difference."""
    ref = forward(HFWeights(hf, cfg, len(ranks)), ids, cfg)
    try:
        got = forward(MegatronWeights(ranks, cfg), ids, cfg)
    except Exception as e:                            # noqa: BLE001
        return {"equal": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"equal": bits_equal(got, ref), "logits": got.numel(), "differing_logits": differing(got, ref),
            "max_abs_diff": float((got.float() - ref.float()).abs().max()) if got.shape == ref.shape else None}


# ============================================================================ helpers
def flat(ranks):
    """The Megatron-style ranks as ckpt_compare.load names them from disk."""
    return {f"mp_rank_{r:02d}/model.{k}": v for r, rk in enumerate(ranks) for k, v in rk.items()}


def copy_ranks(ranks):
    return [{k: v.clone() for k, v in rk.items()} for rk in ranks]


def map_vars(cfg):
    return {"i": cfg["layers"], "qrows": cfg["heads"] // cfg["kv_heads"] * cfg["head_dim"], "hd": cfg["head_dim"]}


def id_state_dict(hf):
    """Each HF row replaced by its global row id (int64), to follow rows through the converter."""
    ids, names, base = {}, {}, 0
    for name, t in hf.items():
        rows = t.numel() // t.shape[-1]
        ids[name] = (base + torch.arange(rows)).reshape(tuple(t.shape[:-1]) + (1,)).expand(t.shape).contiguous()
        for r in range(rows):
            names[base + r] = (name, r)
        base += rows
    return ids, names


def changed_rows(clean, faulty):
    """{"mp_rank_XX/model.<name>[row]"} of rows whose bits differ (same shapes)."""
    out = set()
    for key in clean:
        a, b = clean[key], faulty[key]
        if a.shape != b.shape:
            continue
        a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        for r in (a2.view(torch.int16) != b2.view(torch.int16)).any(dim=1).nonzero().reshape(-1).tolist():
            out.add(f"{key}[{r}]")
    return out


def hf_rows_of(physical, id_flat, id_names):
    out = set()
    for item in physical:
        key, row = item[:-1].rsplit("[", 1)
        t = id_flat[key]
        out.add(id_names[int(t.reshape(-1, t.shape[-1])[int(row), 0])])
    return out


def named_rows(rep):
    """What the tool reported: A rows as (param, row); B rows as sets of physical parts."""
    a_rows, b_parts, params = set(), [], set()
    for p in rep["params"]:
        if p["status"] in cc.OK_STATUS:
            continue
        params.add(p["name"])
        for r in p.get("a", {}).get("unmatched_rows", []):
            a_rows.add((p["name"], r))
        for o in p.get("b", {}).get("unmatched_origin", []):
            b_parts.append(set(o.split(" + ")))
    return params, a_rows, b_parts


def run_compare(a, b, spec, layouts, vars_a, vars_b, **kw):
    t0 = time.perf_counter()
    rep = cc.compare(a, b, spec, layouts, vars_a, vars_b, max_rows=1_000_000, **kw)
    return rep, time.perf_counter() - t0


def short(rep, lines=12):
    text = cc.format_report(rep, limit=3).splitlines()
    return text[:lines] + ([f"... {len(text) - lines} more lines"] if len(text) > lines else [])


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="small model")
    ap.add_argument("--json", help="write the bi-results-v1 file here")
    ap.add_argument("--keep", help="keep the checkpoints in this directory")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    cfg = CONFIGS["quick" if args.quick else "full"]
    res = Results("ckpt_conversion", quick=args.quick)
    res.extra["config"] = cfg
    res.extra["torch_threads"] = torch.get_num_threads()
    spec = cc.load_map(MAP)
    mv = map_vars(cfg)
    hf = make_state_dict(cfg)
    ids = torch.randint(cfg["vocab"], (2, 16), generator=torch.Generator().manual_seed(1))
    id_sd, id_names = id_state_dict(hf)
    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="ckpt-conversion-"))
    work.mkdir(parents=True, exist_ok=True)

    try:
        # ---- the model and the forward
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
            hcfg = LlamaConfig(vocab_size=cfg["vocab"], hidden_size=cfg["hidden"],
                               intermediate_size=cfg["intermediate"], num_hidden_layers=cfg["layers"],
                               num_attention_heads=cfg["heads"], num_key_value_heads=cfg["kv_heads"],
                               head_dim=cfg["head_dim"], rms_norm_eps=cfg["eps"], rope_theta=cfg["theta"],
                               max_position_embeddings=64, tie_word_embeddings=False, attn_implementation="eager")
            default = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)    # as from_pretrained: the RoPE inv_freq buffer stays
            try:                                       # fp32; .to(bfloat16) would round it to bf16
                model = LlamaForCausalLM(hcfg).eval()
            finally:
                torch.set_default_dtype(default)
            model.load_state_dict(hf, strict=True)
            with torch.no_grad():
                theirs = model(ids).logits
            mine = forward(HFWeights(hf, cfg), ids, cfg)
            res.check("the state dict loads strictly into LlamaForCausalLM and the plain forward equals its "
                      "logits", bits_equal(mine, theirs), level="other", impl="transformers", expect="info",
                      detail={"differing_logits": differing(mine, theirs)})
        except ImportError:
            res.extra["transformers"] = "not installed: LlamaForCausalLM check skipped"

        hf_dir = work / "hf"
        cv.save_hf(hf, hf_dir)
        hf_disk = cc.load(hf_dir)
        mcore_dirs = {}
        for tp in TPS:
            ranks = cv.hf_to_mcore(hf, cfg, tp)
            d = work / f"mcore_tp{tp}"
            cv.save_mcore(ranks, d)
            mcore_dirs[tp] = d
            back = cv.mcore_to_hf(cv.load_mcore(d), cfg)
            rep, sec = run_compare(hf_disk, back, None, None, None, None)
            res.check(f"round trip HF -> Megatron-style TP {tp} -> HF is bit-exact, tensor by tensor",
                      rep["summary"]["equal"] == len(hf) == rep["summary"]["params"], level="other",
                      impl="convert.py", detail={"tensors": len(hf), "summary": rep["summary"]})
            rep, sec = run_compare(hf_disk, cc.load(d), spec, ("hf", "mcore"), mv, {**mv, "r": tp})
            res.check(f"conversion to TP {tp}: every row of every logical parameter matched",
                      rep["all_rows_matched"] and not rep["unused"]["a"] and not rep["unused"]["b"],
                      level="other", impl="ckpt_compare.py",
                      detail={"summary": rep["summary"], "report": short(rep)})
            res.measure(f"ckpt_compare HF vs TP {tp}", round(sec, 3), "s", impl="ckpt_compare.py",
                        rows=sum(p["a"]["rows"] for p in rep["params"]))
            fc = forward_check(cv.load_mcore(d), hf, cfg, ids)
            res.check(f"forward at the target layout, TP {tp}, equals the source forward with the same "
                      f"reduction", fc["equal"], level="other", impl="forward check", detail=fc)
            ref1 = forward(HFWeights(hf, cfg, 1), ids, cfg)
            refk = forward(HFWeights(hf, cfg, tp), ids, cfg)
            res.check(f"the source forward reduced as at TP {tp} equals TP 1", bits_equal(ref1, refk),
                      level="L5", impl="row-parallel sum", expect="info",
                      detail={"differing_logits": differing(ref1, refk), "logits": ref1.numel(),
                              "max_abs_diff": float((ref1.float() - refk.float()).abs().max())})

        rep, sec = run_compare(cc.load(mcore_dirs[2]), cc.load(mcore_dirs[4]), spec, ("mcore", "mcore"),
                               {**mv, "r": 2}, {**mv, "r": 4})
        res.check("resharding TP 2 -> TP 4: every row matched", rep["all_rows_matched"]
                  and not rep["unused"]["a"] and not rep["unused"]["b"], level="other", impl="ckpt_compare.py",
                  detail={"summary": rep["summary"], "report": short(rep)})

        wide = flat(cv.hf_to_mcore(hf, cfg, 2, dtype=torch.float32))
        rep, sec = run_compare(hf_disk, wide, spec, ("hf", "mcore"), mv, {**mv, "r": 2})
        widened = rep["summary"]["dtype_changes"].get("widening (exact)", 0)
        res.check("a widening cast (bf16 -> fp32) is reported per parameter and its rows match exactly",
                  rep["all_rows_matched"] and widened == rep["summary"]["params"], level="other",
                  impl="ckpt_compare.py", detail={"summary": rep["summary"]})

        rep, sec = run_compare(hf_disk, cc.load(mcore_dirs[2]), None, None, None, None, scope="global")
        res.check("global scope without a map: row-parallel shards cannot match their full rows",
                  rep["all_rows_matched"], level="other", impl="ckpt_compare.py", expect="info",
                  detail={"summary": rep["summary"],
                          "unmatched_tensors_a": sorted(i["name"] for i in rep["a"] if i["unmatched_count"])[:8]})

        # ---- planted faults, at TP 2
        tp = 2
        clean = cv.hf_to_mcore(hf, cfg, tp)
        clean_flat = flat(clean)
        id_flat = flat(cv.hf_to_mcore(id_sd, cfg, tp))
        G, hd = cfg["kv_heads"], cfg["head_dim"]
        qpg = cfg["heads"] // G

        def transposed_shard(rk):
            w = rk[1]["decoder.layers.0.mlp.linear_fc1.weight"]
            rk[1]["decoder.layers.0.mlp.linear_fc1.weight"] = w.t().contiguous().view(w.shape)
            return rk

        def fp16_cast(rk):
            return [{k: v.to(torch.float16).to(torch.bfloat16) for k, v in r.items()} for r in rk]

        flip_row = qpg * hd + hd // 2                    # a k row of the first query group
        def bit_flip(rk):
            rk[0]["decoder.layers.1.self_attention.linear_qkv.weight"].view(torch.int16)[flip_row, 17] ^= 1 << 3
            return rk

        def missing_row(rk):
            rk[1]["output_layer.weight"] = rk[1]["output_layer.weight"][:-1].clone()
            return rk

        def head_swap(rk):
            w = rk[0]["decoder.layers.0.self_attention.linear_qkv.weight"]
            w[0:2 * hd] = torch.cat([w[hd:2 * hd], w[0:hd]]).clone()
            return rk

        def shard_order(rk):
            name = "decoder.layers.1.mlp.linear_fc2.weight"
            rk[0][name], rk[1][name] = rk[1][name], rk[0][name]
            return rk

        faults = {
            "transposed shard (rank 1 gate/up of layer 0 stored transposed)": transposed_shard(copy_ranks(clean)),
            "wrong QKV interleave (q, k, v stacked, not per query group)":
                cv.hf_to_mcore(hf, cfg, tp, qkv_layout="concat"),
            "bf16 -> fp16 -> bf16 cast of every tensor": fp16_cast(copy_ranks(clean)),
            "one flipped bit (layer 1 QKV, rank 0)": bit_flip(copy_ranks(clean)),
            "a missing row (last row of the rank 1 output layer)": missing_row(copy_ranks(clean)),
        }
        res.extra["faults"] = {}
        for fname, ranks in faults.items():
            fdir = work / ("fault_" + fname.split(" (")[0].replace(" ", "_").replace(">", "").replace("/", ""))
            cv.save_mcore(ranks, fdir)
            faulty = cc.load(fdir)
            rep, sec = run_compare(hf_disk, faulty, spec, ("hf", "mcore"), mv, {**mv, "r": tp})
            params, a_rows, b_parts = named_rows(rep)
            if fname.startswith("wrong QKV"):
                want = {f"model.layers.{i}.self_attn.{p}_proj.weight" for i in range(cfg["layers"]) for p in "qkv"}
                ok = params == want and all(any(".linear_qkv.weight[" in x for x in parts) for parts in b_parts)
                planted = {"params": sorted(want)}
            elif fname.startswith("a missing row"):
                dropped = {f"mp_rank_01/model.output_layer.weight[{clean[1]['output_layer.weight'].shape[0] - 1}]"}
                want_a = hf_rows_of(dropped, id_flat, id_names)
                ok = a_rows == want_a and not b_parts
                planted = {"physical": sorted(dropped), "hf_rows": sorted(want_a)}
            else:
                phys = changed_rows(clean_flat, faulty)
                want_a = hf_rows_of(phys, id_flat, id_names)
                covered = set().union(*b_parts) if b_parts else set()
                ok = (a_rows == want_a and phys <= covered and all(parts & phys for parts in b_parts))
                planted = {"physical_rows": len(phys), "hf_rows": len(want_a),
                           "examples": sorted(phys)[:3]}
            res.check(f"fault, {fname}: ckpt_compare names the parameters and rows", ok, level="other",
                      impl="ckpt_compare.py",
                      detail={"planted": planted, "params_named": sorted(params), "a_rows_named": len(a_rows),
                              "b_rows_named": len(b_parts), "seconds": round(sec, 3), "report": short(rep)})
            xrep, _ = run_compare(hf_disk, faulty, spec, ("hf", "mcore"), mv, {**mv, "r": tp}, rows="xor")
            xparams, xa_rows, _ = named_rows(xrep)
            # XOR keys cancel when two elements of one row change by the same bits in the same lane,
            # which a cast can do: recorded for the cast, claimed for the others.
            res.check(f"fault, {fname}: xor row fingerprints name the same rows",
                      (xparams, xa_rows) == (params, a_rows), level="other", impl="ckpt_compare.py --rows xor",
                      expect="info" if "fp16" in fname else "pass",
                      detail={"a_rows_named": len(xa_rows), "a_rows_missed": len(a_rows - xa_rows),
                              "missed_examples": sorted(a_rows - xa_rows)[:4]})
            fc = forward_check(cv.load_mcore(fdir), hf, cfg, ids)
            # A cast that only rounds the smallest weights may not move a single logit on a few
            # inputs: recorded, not claimed.
            res.check(f"fault, {fname}: the forward check differs", not fc["equal"], level="other",
                      impl="forward check", expect="info" if "fp16" in fname else "pass", detail=fc)
            res.extra["faults"][fname] = short(rep, 20)

        # ---- what a row multiset cannot see
        swapped = head_swap(copy_ranks(clean))
        rep, _ = run_compare(hf_disk, flat(swapped), spec, ("hf", "mcore"), mv, {**mv, "r": tp})
        res.check("two q heads swapped inside one parameter: the row multiset sees it",
                  not rep["all_rows_matched"], level="other", impl="ckpt_compare.py", expect="fail",
                  detail={"summary": rep["summary"]})
        fc = forward_check(swapped, hf, cfg, ids)
        res.check("two q heads swapped inside one parameter: the forward check sees it", not fc["equal"],
                  level="other", impl="forward check", detail=fc)

        shuffled = shard_order(copy_ranks(clean))
        for rows, expect in (("xor", "fail"), ("digest", "pass")):
            rep, _ = run_compare(hf_disk, flat(shuffled), spec, ("hf", "mcore"), mv, {**mv, "r": tp}, rows=rows)
            res.check(f"row-parallel shards joined in the wrong order: {rows} rows see it",
                      not rep["all_rows_matched"], level="other", impl=f"ckpt_compare.py --rows {rows}",
                      expect=expect, detail={"summary": rep["summary"], "report": short(rep, 6)})
        fc = forward_check(shuffled, hf, cfg, ids)
        res.check("row-parallel shards joined in the wrong order: the forward check sees it", not fc["equal"],
                  level="other", impl="forward check", detail=fc)

        # ---- the command line, as a user runs it
        argv = [str(hf_dir), str(mcore_dirs[2]), "--map", str(MAP), "--layouts", "hf", "mcore",
                "--set", f"i={cfg['layers']}", "--set", f"qrows={mv['qrows']}", "--set", f"hd={hd}", "--set", "b:r=2"]
        with contextlib.redirect_stdout(io.StringIO()) as out_clean:
            rc_clean = cc.main(argv)
        flipped = work / "fault_one_flipped_bit"
        with contextlib.redirect_stdout(io.StringIO()) as out_fault:
            rc_fault = cc.main([str(hf_dir), str(flipped)] + argv[2:])
        res.check("the command line exits 0 on the clean conversion and 1 on the flipped bit",
                  rc_clean == 0 and rc_fault == 1, level="other", impl="ckpt_compare.py",
                  detail={"clean": out_clean.getvalue().splitlines()[-1:],
                          "fault": out_fault.getvalue().splitlines()})
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    res.write(args.json)
    sys.exit(res.exit_code())


if __name__ == "__main__":
    main()
