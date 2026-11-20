"""Trainer side: Qwen2.5-0.5B in examples/whole_model/model.py, plus a forward whose rounding points
can be moved one at a time toward vLLM 0.19.1's.

    model = load_qwen(dir)                         model.py WholeModel with the checkpoint's weights
    logits = model.prefill(ids[None], ops="bi")    the shared model, unchanged
    logits = forward_matched(model, ids, ch)       same weights, rounding points chosen by `ch`

`ch` holds one choice per rounding point (see CHOICES). REFERENCE reproduces model.py's stock op set
bit for bit (checked by cross_engine.py). The "vllm_*" choices call vLLM's own kernels from the
vllm-l4 environment (torch.ops._C, vLLM's Triton batch-invariant kernels, vLLM's FlashAttention);
they need `import vllm` and are unavailable in a torch-only environment.
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "whole_model"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import model as wm                                                  # noqa: E402

try:
    from vllm import _custom_ops as vops                            # noqa: F401  (loads torch.ops._C)
    from vllm.model_executor.layers import batch_invariant as vbi
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version
    HAVE_VLLM = True
except Exception:                                                   # noqa: BLE001
    HAVE_VLLM = False

# One entry per rounding point. The first value of each is model.py's (the reference).
CHOICES = {
    "norm": ("hf", "vllm_cuda", "vllm_bi"),                 # RMSNorm and the residual add before it
    "gemm": ("cublas_sep", "cublas_packed", "vllm_bi"),     # q/k/v, o, gate/up, down projections
    "bias": ("epilogue", "after_cast"),                     # q/k/v bias: in the fp32 sum, or added in bf16
    "rope_table": ("hf", "vllm"),                           # cos/sin: transformers' or vLLM's cos_sin_cache
    "rope": ("hf", "vllm_cuda"),                            # rotation formula / kernel
    "attn": ("sdpa", "bi", "fa2_split0", "fa2_split1"),     # FA2 = vLLM's paged FlashAttention, num_splits 0 / 1
    "act": ("torch", "vllm_cuda"),                          # silu(g) * u
    "lm_head": ("cublas", "vllm_bi"),
    "logsoftmax": ("torch", "vllm_bi"),
}
REFERENCE = {k: v[0] for k, v in CHOICES.items()}


def load_qwen(model_dir: str, ops: str = "stock", device: str = "cuda") -> wm.WholeModel:
    from safetensors.torch import load_file
    with open(os.path.join(model_dir, "config.json")) as f:
        hf = json.load(f)
    cfg = wm.config_from_hf(hf, dtype="bf16", max_positions=4096)
    model = wm.WholeModel(cfg, ops).to(device=device, dtype=torch.bfloat16).eval()
    wm.load_hf_state_dict(model, load_file(os.path.join(model_dir, "model.safetensors")))
    return model


def logprobs(logits: torch.Tensor, how: str = "torch") -> torch.Tensor:
    x = logits.float()
    if how == "vllm_bi":
        return vbi.log_softmax(x, dim=-1)
    return torch.log_softmax(x, -1)


# ----------------------------------------------------------------------------- vLLM-side tables
_VLLM_CACHE = {}


def vllm_cos_sin_cache(cfg, device) -> torch.Tensor:
    """vLLM 0.19.1 RotaryEmbedding._compute_cos_sin_cache (read from rotary_embedding/base.py): built on the
    model's device (vLLM constructs the model under the cuda device context), fp32, then cast to the dtype."""
    key = str(device)
    if key not in _VLLM_CACHE:
        D = cfg.head_dim
        with torch.device(device):
            inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.float) / D))
            t = torch.arange(cfg.max_positions, dtype=torch.float)
            fr = torch.einsum("i,j -> ij", t, inv)
            cache = torch.cat((fr.cos(), fr.sin()), dim=-1)
        _VLLM_CACHE[key] = cache.to(torch.bfloat16)
    return _VLLM_CACHE[key]


# ----------------------------------------------------------------------------- ops
def op_norm(x, residual, w, eps, how):
    """Returns (normed, new_residual). residual None: the first norm of the model."""
    if how == "hf":
        s = x if residual is None else x + residual                  # bf16 add (model.py R5/R8)
        xf = s.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return w * y.to(x.dtype), s
    if how == "vllm_cuda":
        if residual is None:
            out = torch.empty_like(x)
            torch.ops._C.rms_norm(out, x.contiguous(), w, eps)
            return out, x
        a, r = x.contiguous().clone(), residual.contiguous().clone()
        torch.ops._C.fused_add_rms_norm(a, r, w, eps)               # in place: a <- norm, r <- x + residual
        return a, r
    if how == "vllm_bi":
        s = x if residual is None else x + residual                  # layernorm.py fused_add_rms_norm under BI
        return vbi.rms_norm(s, w, eps), s
    raise ValueError(how)


def op_linear(x, W, b, gemm, bias):
    if gemm == "vllm_bi":
        y = vbi.matmul_persistent(x.reshape(-1, x.shape[-1]), W.t())
        y = y.view(*x.shape[:-1], W.shape[0])
        return y if b is None else y + b                             # linear_batch_invariant: bias after the cast
    if b is not None and bias == "after_cast":
        return F.linear(x, W) + b
    return F.linear(x, W, b)


def op_rope(q, k, pos, cfg, table, how, hf_tables):
    """q [T, Hq*D], k [T, Hk*D] -> rotated, same shapes."""
    D = cfg.head_dim
    if table == "vllm":
        cs = vllm_cos_sin_cache(cfg, q.device)
    else:                                                            # model.py's (transformers') table
        cs = torch.cat((hf_tables["cos"][:, : D // 2], hf_tables["sin"][:, : D // 2]), -1)
    if how == "vllm_cuda":
        q, k = q.contiguous().clone(), k.contiguous().clone()
        torch.ops._C.rotary_embedding(pos, q, k, D, cs, True)
        return q, k
    cos = torch.cat((cs[pos, : D // 2],) * 2, -1)[:, None, :]
    sin = torch.cat((cs[pos, D // 2:],) * 2, -1)[:, None, :]

    def rot(x):
        x = x.view(x.shape[0], -1, D)
        r = torch.cat((-x[..., D // 2:], x[..., : D // 2]), -1)
        return ((x * cos) + (r * sin)).reshape(x.shape[0], -1)
    return rot(q), rot(k)


def op_attn(q, k, v, cfg, how, ops_bi=None):
    """q [T, Hq*D], k, v [T, Hk*D], one causal sequence from position 0 -> [T, Hq*D]."""
    T, D, Hq, Hk = q.shape[0], cfg.head_dim, cfg.heads, cfg.kv_heads
    qh, kh, vh = q.view(T, Hq, D), k.view(T, Hk, D), v.view(T, Hk, D)
    scale = D ** -0.5
    if how == "sdpa":
        return wm.OPS["stock"].attn_full(qh[None], kh[None], vh[None], scale)[0].reshape(T, -1)
    if how == "bi":
        return wm.OPS["bi"].attn_full(qh[None], kh[None], vh[None], scale)[0].reshape(T, -1)
    if how in ("fa2_split0", "fa2_split1"):
        bs = 16
        nb = (T + bs - 1) // bs
        kc = torch.zeros(nb * bs, Hk, D, device=q.device, dtype=q.dtype)
        vc = torch.zeros_like(kc)
        kc[:T], vc[:T] = kh, vh
        kc, vc = kc.view(nb, bs, Hk, D), vc.view(nb, bs, Hk, D)
        dev = q.device
        out = torch.empty_like(qh)
        flash_attn_varlen_func(q=qh.contiguous(), k=kc, v=vc, out=out,
                               cu_seqlens_q=torch.tensor([0, T], device=dev, dtype=torch.int32), max_seqlen_q=T,
                               seqused_k=torch.tensor([T], device=dev, dtype=torch.int32), max_seqlen_k=T,
                               softmax_scale=scale, causal=True,
                               block_table=torch.arange(nb, device=dev, dtype=torch.int32)[None],
                               fa_version=get_flash_attn_version(),
                               num_splits=0 if how == "fa2_split0" else 1)
        return out.reshape(T, -1)
    raise ValueError(how)


def op_act(g, u, how):
    if how == "vllm_cuda":
        x = torch.cat((g, u), -1).contiguous()
        out = torch.empty_like(g)
        torch.ops._C.silu_and_mul(out, x)
        return out
    return F.silu(g) * u


def op_lm_head(h, E, how):
    if how == "vllm_bi":
        return vbi.matmul_persistent(h, E.t())
    return F.linear(h, E)


# ----------------------------------------------------------------------------- the forward
def _packed(layer):
    a, m = layer.self_attn, layer.mlp
    key = "_l4_packed"
    if not hasattr(layer, key):
        Wqkv = torch.cat((a.q_proj.weight, a.k_proj.weight, a.v_proj.weight), 0).contiguous()
        bqkv = torch.cat((a.q_proj.bias, a.k_proj.bias, a.v_proj.bias), 0).contiguous()
        Wgu = torch.cat((m.gate_proj.weight, m.up_proj.weight), 0).contiguous()
        object.__setattr__(layer, key, (Wqkv, bqkv, Wgu))
    return getattr(layer, key)


@torch.no_grad()
def forward_matched(model: wm.WholeModel, ids: torch.Tensor, ch: dict, return_hidden: bool = False):
    """ids [T] (one sequence from position 0) -> logits [T, V] (bf16 unless the lm_head says otherwise)."""
    ch = {**REFERENCE, **ch}
    cfg, bb = model.cfg, model.model
    T = ids.numel()
    dev = model.lm_head.weight.device
    ids = ids.to(dev)
    pos = torch.arange(T, device=dev)
    hf_tables = model.ctx.tables(dev)
    q_sz, kv_sz, I = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim, cfg.intermediate
    eps = cfg.rms_eps
    gemm, bias = ch["gemm"], ch["bias"]
    x = bb.embed_tokens.weight[ids]
    residual = None
    hidden = []
    for L in bb.layers:
        a, m = L.self_attn, L.mlp
        h, residual = op_norm(x, residual, L.input_layernorm.weight, eps, ch["norm"])
        if gemm == "cublas_sep":
            q = op_linear(h, a.q_proj.weight, a.q_proj.bias, gemm, bias)
            k = op_linear(h, a.k_proj.weight, a.k_proj.bias, gemm, bias)
            v = op_linear(h, a.v_proj.weight, a.v_proj.bias, gemm, bias)
        else:
            Wqkv, bqkv, _ = _packed(L)
            q, k, v = op_linear(h, Wqkv, bqkv, gemm, bias).split([q_sz, kv_sz, kv_sz], -1)
        q, k = op_rope(q, k, pos, cfg, ch["rope_table"], ch["rope"], hf_tables)
        o = op_attn(q, k, v.contiguous(), cfg, ch["attn"])
        x = op_linear(o, a.o_proj.weight, None, gemm, bias)
        h, residual = op_norm(x, residual, L.post_attention_layernorm.weight, eps, ch["norm"])
        if gemm == "cublas_sep":
            g = op_linear(h, m.gate_proj.weight, None, gemm, bias)
            u = op_linear(h, m.up_proj.weight, None, gemm, bias)
        else:
            _, _, Wgu = _packed(L)
            g, u = op_linear(h, Wgu, None, gemm, bias).split([I, I], -1)
        x = op_linear(op_act(g, u, ch["act"]), m.down_proj.weight, None, gemm, bias)
        if return_hidden:
            hidden.append((residual, x))
    h, _ = op_norm(x, residual, bb.norm.weight, eps, ch["norm"])
    logits = op_lm_head(h, model.lm_head.weight, ch["lm_head"])
    return (logits, hidden, h) if return_hidden else logits
