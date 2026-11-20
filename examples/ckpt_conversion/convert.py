"""A Hugging Face Llama <-> Megatron-style checkpoint converter: the code under test.

Megatron-style layout (the names and packing of Megatron-core's GPT model), per TP rank r:

    embedding.word_embeddings.weight                      [V/tp, H]      vocab-parallel rows
    decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight   [H] replicated
    decoder.layers.{i}.self_attention.linear_qkv.weight   [(G/tp)*(q_per_group+2)*hd, H]
        for each of the rank's query groups g: the group's q heads, then its k head, then its v head
    decoder.layers.{i}.self_attention.linear_proj.weight  [H, nh*hd/tp]  row-parallel: columns
    decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight   [H]            replicated
    decoder.layers.{i}.mlp.linear_fc1.weight              [2*I/tp, H]    the rank's gate rows, then its up rows
    decoder.layers.{i}.mlp.linear_fc2.weight              [H, I/tp]      row-parallel: columns
    decoder.final_layernorm.weight                        [H]            replicated
    output_layer.weight                                   [V/tp, H]      vocab-parallel rows

saved as <dir>/mp_rank_{r:02d}/model_optim_rng.pt = {"model": state_dict, "checkpoint_version": 3.0}.

    ranks = hf_to_mcore(hf_state_dict, cfg, tp=2)          # list of per-rank state dicts
    save_mcore(ranks, out_dir); ranks = load_mcore(out_dir)
    hf = mcore_to_hf(ranks, cfg)

`qkv_layout="concat"` is a deliberate bug kept for the demo: it stacks all q, then all k,
then all v, and cuts that into TP shards, instead of interleaving per query group.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

HF_LAYER = "model.layers.{i}."
MC_LAYER = "decoder.layers.{i}."


def hf_names(cfg: dict) -> List[str]:
    names = ["model.embed_tokens.weight"]
    for i in range(cfg["layers"]):
        p = HF_LAYER.format(i=i)
        names += [p + n for n in ("input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                  "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                                  "post_attention_layernorm.weight", "mlp.gate_proj.weight",
                                  "mlp.up_proj.weight", "mlp.down_proj.weight")]
    return names + ["model.norm.weight", "lm_head.weight"]


def _check(cfg: dict, tp: int) -> None:
    for key, n in (("kv_heads", cfg["kv_heads"]), ("intermediate", cfg["intermediate"]), ("vocab", cfg["vocab"])):
        if n % tp:
            raise ValueError(f"{key}={n} is not divisible by tp={tp}")


def hf_to_mcore(hf: Dict[str, torch.Tensor], cfg: dict, tp: int, qkv_layout: str = "grouped",
                dtype=None) -> List[Dict[str, torch.Tensor]]:
    _check(cfg, tp)
    G, hd, H = cfg["kv_heads"], cfg["head_dim"], cfg["hidden"]
    qpg = cfg["heads"] // G
    ranks: List[Dict[str, torch.Tensor]] = [{} for _ in range(tp)]

    def put(r, name, t):
        t = t.contiguous().clone()
        ranks[r][name] = t if dtype is None else t.to(dtype)

    for r, (e, o) in enumerate(zip(hf["model.embed_tokens.weight"].chunk(tp, 0), hf["lm_head.weight"].chunk(tp, 0))):
        put(r, "embedding.word_embeddings.weight", e)
        put(r, "output_layer.weight", o)
        put(r, "decoder.final_layernorm.weight", hf["model.norm.weight"])
    for i in range(cfg["layers"]):
        h, m = HF_LAYER.format(i=i), MC_LAYER.format(i=i)
        q = hf[h + "self_attn.q_proj.weight"]
        k = hf[h + "self_attn.k_proj.weight"]
        v = hf[h + "self_attn.v_proj.weight"]
        if qkv_layout == "grouped":
            qkv = torch.cat([q.view(G, qpg * hd, H), k.view(G, hd, H), v.view(G, hd, H)], dim=1)
            qkv_shards = [s.reshape(-1, H) for s in qkv.chunk(tp, 0)]
        elif qkv_layout == "concat":                       # the planted bug
            qkv_shards = list(torch.cat([q, k, v], 0).chunk(tp, 0))
        else:
            raise ValueError(qkv_layout)
        gates = hf[h + "mlp.gate_proj.weight"].chunk(tp, 0)
        ups = hf[h + "mlp.up_proj.weight"].chunk(tp, 0)
        o_cols = hf[h + "self_attn.o_proj.weight"].chunk(tp, 1)
        d_cols = hf[h + "mlp.down_proj.weight"].chunk(tp, 1)
        for r in range(tp):
            put(r, m + "self_attention.linear_qkv.layer_norm_weight", hf[h + "input_layernorm.weight"])
            put(r, m + "self_attention.linear_qkv.weight", qkv_shards[r])
            put(r, m + "self_attention.linear_proj.weight", o_cols[r])
            put(r, m + "mlp.linear_fc1.layer_norm_weight", hf[h + "post_attention_layernorm.weight"])
            put(r, m + "mlp.linear_fc1.weight", torch.cat([gates[r], ups[r]], 0))
            put(r, m + "mlp.linear_fc2.weight", d_cols[r])
    return ranks


def mcore_to_hf(ranks: List[Dict[str, torch.Tensor]], cfg: dict, dtype=None) -> Dict[str, torch.Tensor]:
    tp = len(ranks)
    G, hd, H = cfg["kv_heads"], cfg["head_dim"], cfg["hidden"]
    qpg = cfg["heads"] // G
    out: Dict[str, torch.Tensor] = {}

    def put(name, t):
        t = t.contiguous().clone()
        out[name] = t if dtype is None else t.to(dtype)

    def cat(name, dim):
        return torch.cat([rk[name] for rk in ranks], dim)

    put("model.embed_tokens.weight", cat("embedding.word_embeddings.weight", 0))
    for i in range(cfg["layers"]):
        h, m = HF_LAYER.format(i=i), MC_LAYER.format(i=i)
        qkv = cat(m + "self_attention.linear_qkv.weight", 0).view(G, (qpg + 2) * hd, H)
        q, k, v = qkv.split([qpg * hd, hd, hd], dim=1)
        put(h + "input_layernorm.weight", ranks[0][m + "self_attention.linear_qkv.layer_norm_weight"])
        put(h + "self_attn.q_proj.weight", q.reshape(-1, H))
        put(h + "self_attn.k_proj.weight", k.reshape(-1, H))
        put(h + "self_attn.v_proj.weight", v.reshape(-1, H))
        put(h + "self_attn.o_proj.weight", cat(m + "self_attention.linear_proj.weight", 1))
        put(h + "post_attention_layernorm.weight", ranks[0][m + "mlp.linear_fc1.layer_norm_weight"])
        fc1 = [rk[m + "mlp.linear_fc1.weight"].chunk(2, 0) for rk in ranks]
        put(h + "mlp.gate_proj.weight", torch.cat([g for g, _ in fc1], 0))
        put(h + "mlp.up_proj.weight", torch.cat([u for _, u in fc1], 0))
        put(h + "mlp.down_proj.weight", cat(m + "mlp.linear_fc2.weight", 1))
    put("model.norm.weight", ranks[0]["decoder.final_layernorm.weight"])
    put("lm_head.weight", cat("output_layer.weight", 0))
    return out


def save_mcore(ranks: List[Dict[str, torch.Tensor]], out_dir) -> None:
    for r, sd in enumerate(ranks):
        d = Path(out_dir) / f"mp_rank_{r:02d}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"model": sd, "checkpoint_version": 3.0}, d / "model_optim_rng.pt")


def load_mcore(in_dir) -> List[Dict[str, torch.Tensor]]:
    dirs = sorted(Path(in_dir).glob("mp_rank_*"))
    return [torch.load(d / "model_optim_rng.pt", map_location="cpu", weights_only=True)["model"] for d in dirs]


def save_hf(sd: Dict[str, torch.Tensor], out_dir) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(sd, Path(out_dir) / "pytorch_model.bin")
