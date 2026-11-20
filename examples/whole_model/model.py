"""A Llama-architecture decoder-only LM in plain torch with a switchable op set.

The model follows Hugging Face transformers' LlamaForCausalLM (read from 4.49):
RMSNorm with eps, SwiGLU MLP (gate/up/down), grouped-query attention, RoPE with
theta 10000 on q and k, no biases, an untied lm_head. Parameter names and shapes
are HF's, so `model.state_dict()` loads into LlamaForCausalLM(hf_config(cfg))
with strict=True. A second variant replaces the MLP by the MoE layer of
examples/moe_combine/moe_bi.py (softmax router, top-k, fp32 fixed-order combine).

Rounding sequence (the reference; both op sets follow it and differ only in
the kernels marked KERNEL). dt is the model dtype, bf16 by default.

 R0  h = E[ids]                      gather, exact
 per layer
 R1  RMSNorm(h, g): xf = fp32(h)                         exact
                    s  = sum_k xf[k]^2                    fp32, order: KERNEL
                    var = s * fl32(1/H)                   fp32
                    y  = xf * rsqrt(var + eps)            fp32, three rounded ops
                    a  = dt(y) * g                        cast first; the weight multiplies in dt
 R2  q = a Wq^T, k = a Wk^T, v = a Wv^T                   GEMM: dt operands, fp32 sum (order: KERNEL), one cast to dt
     with qkv_bias: q = dt(sum + bq)                      bias added to the fp32 sum, one cast (cuBLAS addmm epilogue)
 R3  RoPE on q and k, per head, pairs (i, i + D/2), rot(x) = cat(-x[D/2:], x[:D/2]):
        table = dt(cos(pos * inv_freq)), dt(sin(...))     cos and sin in fp32 on the device, then cast
        q' = dt(dt(q * cos) + dt(rot(q) * sin))           three roundings in dt
 R4  o = softmax(q' k'^T / sqrt(D), causal) v             KERNEL (stock: SDPA on K/V expanded to the
                                                          query heads; bi: 64-key blocks from position
                                                          0, fp32 online softmax, P cast to dt before
                                                          P.V, one cast of the output to dt)
 R5  h = dt(h + o Wo^T)                                   GEMM as R2; the residual add rounds in dt
 R6  m = RMSNorm(h, g2)                                   as R1
 R7  u = (dt(silu(m Wg^T)) * (m Wu^T)) Wd^T               silu per element in fp32 then cast; product in dt
     MoE variant: p = softmax(fp32(m Wr)); (w, id) = topk(p); w = w / sum(w)   all fp32
                  u = dt(sum over top-k rank r of fp32(expert_id_r(m)) * w_r)  fp32, rank order
 R8  h = dt(h + u)
 final
 R9  logits = RMSNorm(h, gf) Wlm^T                        dt (GEMM as R2)
 R10 logp = log_softmax(fp32(logits))                     fp32

`Rounding` switches one of these choices at a time (norm cast order, residual
add dtype, RoPE formula, lm_head output dtype, log_softmax dtype).

Op sets
  "stock"  F.linear (cuBLAS), F.scaled_dot_product_attention, torch reductions.
  "bi"     tools/bi_kernels.py: mm_one_config, attention_fixed_blocks (and
           attention_fixed_blocks_train under autograd); plus, in this file, a
           fixed-tree row sum for the RMSNorm variance (torch's reduction
           changes its tree with the number of rows), the per-sequence-position
           attention of examples/attention_serving (copied) for the KV-cache
           and packed forwards, and an fp32-output GEMM for one rounding variant.

API
  cfg = Config()                     hidden 1024, 8 layers, 16/4 heads, vocab 32000
  Config.wide(), Config.moe_default(), Config.tiny()
  model = build(cfg, ops="bi", seed=0, device="cuda")
  logits = model.prefill(ids)                       trainer forward, ids [B, T] -> [B, T, V]
  logits = model.prefill(ids, ops="stock")          op set per call
  cache = KVCache(model, slots=64, capacity=2048)
  outs = model.forward_cached(cache, slots, tokens) engine forward: appends tokens[i] (1-D ids, any
                                                    length) to slot slots[i]; returns a list of [t_i, V]
  outs = model.prefill_packed(seqs)                 packed varlen prefill (cu_seqlens), no padding
  cache.reset(slot), cache.truncate(slot, n), cache.copy_prefix(dst, src, n)
  logp = model.logprobs(logits)                     fp32 log_softmax (R10)
  realized_logprob(logp, ids)                       logp[t, ids[t+1]]
  with model.using(ops="stock", norm="weight_then_cast"): ...
  save(model, path); model = load(path, ops="bi")
  hf_config(cfg) -> kwargs for transformers.LlamaConfig (Qwen2Config when cfg.qkv_bias);
  export_hf(model, dir); config_from_hf(hf_config); load_hf_state_dict(model, sd); from_hf(hf_model)
  Config(qkv_bias=True, tie_embeddings=True, rms_eps=..., rope_theta=...) covers the Qwen2
  architecture (Config.qwen2_0p5b() has the shapes of Qwen2.5-0.5B-Instruct).

The modules of the trainer forward take batch-first tensors, so tools/bi_audit.py
can replay each of them alone. Forward only unless autograd is enabled: under
autograd the bi op set runs GEMM, attention and the row sum through
autograd.Functions whose backward uses the same kernels.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))

try:
    import triton
    import triton.language as tl
    from bi_kernels import LOG2E, attention_fixed_blocks, attention_fixed_blocks_train, mm_one_config
    HAVE_TRITON = True
except ImportError:                                   # the stock op set still works
    HAVE_TRITON = False

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _note(name: str, opclass: str, inputs, output) -> None:
    """Tell tools/bi_audit.py about a kernel launch the torch dispatcher cannot see."""
    aud = sys.modules.get("bi_audit")
    if aud is not None and getattr(aud, "ACTIVE", None) is not None:
        aud.note(name, opclass, inputs, output)


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    vocab_size: int = 32000
    hidden: int = 1024
    layers: int = 8
    heads: int = 16
    kv_heads: int = 4
    intermediate: int = 3584
    max_positions: int = 2048
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    dtype: str = "bf16"                # "fp32" for the architecture check against HF
    qkv_bias: bool = False             # bias on q_proj, k_proj, v_proj (Qwen2); o_proj and the MLP have none
    tie_embeddings: bool = False       # lm_head.weight is embed_tokens.weight (Qwen2.5-0.5B)
    moe: bool = False                  # replace the MLP by the MoE layer of moe_bi.py
    moe_experts: int = 64
    moe_topk: int = 8
    moe_intermediate: int = 512
    logit_std: float = 4.5             # lm_head init: logits ~ N(0, logit_std^2) at init
    norm_weight_std: float = 0.1       # norm weights 1 + N(0, std^2): a weight of exactly 1 hides a rounding point

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @staticmethod
    def wide() -> "Config":            # Llama-3-8B layer shapes, 2 layers: large GEMMs
        return Config(hidden=4096, layers=2, heads=32, kv_heads=8, intermediate=14336)

    @staticmethod
    def moe_default() -> "Config":     # the sizes moe_bi.py was verified at: 64 experts, top-8, inner 512
        return Config(moe=True)

    @staticmethod
    def qwen2_0p5b() -> "Config":      # the shapes of Qwen2.5-0.5B-Instruct (Qwen2 architecture)
        return Config(vocab_size=151936, hidden=896, layers=24, heads=14, kv_heads=2, intermediate=4864,
                      max_positions=32768, rms_eps=1e-6, rope_theta=1000000.0, qkv_bias=True, tie_embeddings=True)

    @staticmethod
    def tiny() -> "Config":
        return Config(vocab_size=1000, hidden=256, layers=2, heads=4, kv_heads=2, intermediate=640,
                      max_positions=512, moe_experts=8, moe_topk=3, moe_intermediate=128)


@dataclass(frozen=True)
class Rounding:
    """One rounding choice per field. The defaults are the reference sequence (HF)."""
    norm: str = "cast_then_weight"     # "weight_then_cast": dt(y * g), g in fp32. "all_dt": squares, mean, rsqrt in dt
    residual: str = "dt"               # "fp32_into_norm": the next norm reads the unrounded fp32 sum (a fused
                                       # add+norm); the stream itself is stored in dt. "fp32_stream": never rounded
    rope: str = "hf"                   # "fp32": dt(q*cos + rot(q)*sin) in fp32 with fp32 tables, one rounding.
                                       # "cpu_table": the reference formula with cos/sin computed on the CPU.
                                       # "inv_freq_dt": inv_freq rounded to dt first, as transformers does
                                       # when a model is cast with .to(bfloat16) instead of from_pretrained
    lm_head: str = "dt"                # "fp32": the GEMM's fp32 sum is not rounded to dt
    logprob: str = "fp32"              # "dt": log_softmax on the dt logits


def hf_model_type(cfg: Config) -> str:
    """Which transformers class describes cfg: "llama" (no biases) or "qwen2" (bias on q, k, v)."""
    return "qwen2" if cfg.qkv_bias else "llama"


def hf_config(cfg: Config) -> dict:
    """kwargs for transformers.LlamaConfig, or Qwen2Config when cfg.qkv_bias, describing the same
    architecture (dense variant only)."""
    if cfg.moe:
        raise ValueError("the MoE variant has no LlamaConfig equivalent")
    kw = dict(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden, intermediate_size=cfg.intermediate,
              num_hidden_layers=cfg.layers, num_attention_heads=cfg.heads, num_key_value_heads=cfg.kv_heads,
              hidden_act="silu", max_position_embeddings=cfg.max_positions, rms_norm_eps=cfg.rms_eps,
              rope_theta=cfg.rope_theta, attention_dropout=0.0, tie_word_embeddings=cfg.tie_embeddings,
              use_cache=True, torch_dtype={"bf16": "bfloat16", "fp32": "float32"}[cfg.dtype])
    if cfg.qkv_bias:
        kw.update(use_sliding_window=False)
    else:
        kw.update(attention_bias=False, mlp_bias=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return kw


def config_from_hf(hf_cfg, **overrides) -> Config:
    """A Config from a transformers LlamaConfig or Qwen2Config (object or dict)."""
    d = hf_cfg if isinstance(hf_cfg, dict) else hf_cfg.to_dict()
    mt = d.get("model_type", "llama")
    if mt not in ("llama", "qwen2"):
        raise ValueError(f"unsupported model_type {mt!r}")
    if d.get("hidden_act", "silu") != "silu" or d.get("mlp_bias", False) or d.get("rope_scaling"):
        raise ValueError("only silu, no MLP bias and default RoPE are supported")
    if mt == "llama" and d.get("attention_bias", False):
        raise ValueError("LlamaConfig.attention_bias also puts a bias on o_proj, which model.py does not have")
    if mt == "qwen2" and d.get("use_sliding_window", False):
        raise ValueError("sliding-window attention is not supported")
    if d.get("head_dim") not in (None, d["hidden_size"] // d["num_attention_heads"]):
        raise ValueError("head_dim must be hidden_size / num_attention_heads")
    kw = dict(vocab_size=d["vocab_size"], hidden=d["hidden_size"], layers=d["num_hidden_layers"],
              heads=d["num_attention_heads"], kv_heads=d.get("num_key_value_heads") or d["num_attention_heads"],
              intermediate=d["intermediate_size"], max_positions=d["max_position_embeddings"],
              rms_eps=d["rms_norm_eps"], rope_theta=d.get("rope_theta", 10000.0), qkv_bias=mt == "qwen2",
              tie_embeddings=bool(d.get("tie_word_embeddings", mt == "qwen2")))
    kw.update(overrides)
    return Config(**kw)


# ----------------------------------------------------------------------------- extra kernels (Triton)
if HAVE_TRITON:
    @triton.jit(do_not_specialize=["M", "K"])
    def _row_sum_kernel(X, OUT, M, K, sxm, sxk, SQUARE: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
        """out[m] = sum_k x[m, k] (or x[m, k]^2), fp32. One tree per row, fixed by K alone: BK-wide blocks
        in ascending order, each summed by tl.sum inside the program. More rows only add programs."""
        pid = tl.program_id(0)
        rm = pid * BM + tl.arange(0, BM)
        rk = tl.arange(0, BK)
        acc = tl.zeros([BM], dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            x = tl.load(X + rm[:, None] * sxm + kk[None, :] * sxk,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            if SQUARE:
                x = x * x
            acc += tl.sum(x, 1)
        tl.store(OUT + rm, acc, mask=rm < M)

    def row_sum(x: torch.Tensor, square: bool = False, BM=16, BK=256) -> torch.Tensor:
        """x fp32 [M, K] -> fp32 [M, 1]. Batch-invariant row sum (of squares)."""
        M, K = x.shape
        out = torch.empty(M, 1, device=x.device, dtype=torch.float32)
        if M:
            _row_sum_kernel[(triton.cdiv(M, BM),)](x, out, M, K, x.stride(0), x.stride(1),
                                                   SQUARE=square, BM=BM, BK=BK, num_warps=4)
        _note("bi::row_sum", "rowwise", (x,), out)
        return out

    @triton.jit(do_not_specialize=["M"])
    def _mm_f32out_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """bi_kernels._mm_kernel with the fp32 accumulator stored as is (no cast to bf16)."""
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn,
                        mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    def mm_one_config_f32out(a: torch.Tensor, b: torch.Tensor, BM=64, BN=64, BK=32) -> torch.Tensor:
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty(M, N, device=a.device, dtype=torch.float32)
        if M:
            _mm_f32out_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
                a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
        _note("bi::mm_one_config_f32out", "gemm", (a, b), c)
        return c

    # Copied from examples/attention_serving/serving_kernels.py (_serve_fwd, _launch), unchanged in
    # arithmetic. Queries are packed [NQ, Hq, D]; sequence b owns rows cu_q[b]..cu_q[b+1]-1 and its
    # first query sits at absolute position pos0[b]. MODE 1 (paged) reads K/V through a block table;
    # the slot cache below is the case of one page per sequence. MODE 0 reads packed K/V.
    MODE_VARLEN, MODE_PAGED = 0, 1

    @triton.jit(do_not_specialize=["NQ"])
    def _serve_fwd(Q, K, V, O, LSE, BT, CUQ, CUK, POS0, scale_log2, Hq, NQ,
                   sq_t, sq_h, sk_p, sk_t, sk_h, sv_p, sv_t, sv_h, so_t, so_h, s_bt,
                   GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                   PAGE: tl.constexpr, MODE: tl.constexpr, STORE_LSE: tl.constexpr):
        pid_m, bh = tl.program_id(0), tl.program_id(1)
        b, h = bh // Hq, bh % Hq
        kh = h // GROUP
        q0 = tl.load(CUQ + b)
        q_len = tl.load(CUQ + b + 1) - q0
        p0 = tl.load(POS0 + b)                            # absolute position of the first query
        kv_len = p0 + q_len
        rm = pid_m * BM + tl.arange(0, BM)
        pos = p0 + rm
        rd = tl.arange(0, D)
        q = tl.load(Q + (q0 + rm[:, None]) * sq_t + h * sq_h + rd[None, :], mask=rm[:, None] < q_len, other=0.0)
        m_i = tl.full([BM], float("-inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, D], tl.float32)
        hi = tl.minimum(p0 + (pid_m + 1) * BM, kv_len)    # keys any row of this block may see
        hi = tl.where(pid_m * BM < q_len, hi, 0)          # programs past the sequence do nothing
        k0 = 0
        if MODE != 1:
            k0 = tl.load(CUK + b)                         # sequence start in the KV pack
        lo = 0
        if MODE == 2:
            lo = -(k0 % BN)
        for n0 in range(lo, hi, BN):                      # fixed blocks from absolute position 0
            rn = n0 + tl.arange(0, BN)
            kvm = rn < kv_len
            if MODE == 2:
                kvm = kvm & (rn >= 0)
            if MODE == 1:
                if PAGE >= BN:                            # the whole block sits in one page
                    pg = tl.load(BT + b * s_bt + n0 // PAGE).to(tl.int64)
                    slot = n0 % PAGE + tl.arange(0, BN)
                else:                                     # the block gathers from BN // PAGE pages
                    pg = tl.load(BT + b * s_bt + rn // PAGE, mask=kvm, other=0).to(tl.int64)
                    slot = rn % PAGE
                k_row = pg * sk_p + slot * sk_t
                v_row = pg * sv_p + slot * sv_t
            else:
                k_row = (k0 + rn) * sk_t
                v_row = (k0 + rn) * sv_t
            k = tl.load(K + k_row[:, None] + kh * sk_h + rd[None, :], mask=kvm[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale_log2
            s = tl.where((rn[None, :] <= pos[:, None]) & kvm[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + v_row[:, None] + kh * sv_h + rd[None, :], mask=kvm[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        o = acc / l_i[:, None]
        tl.store(O + (q0 + rm[:, None]) * so_t + h * so_h + rd[None, :], o.to(tl.bfloat16), mask=rm[:, None] < q_len)
        if STORE_LSE:
            tl.store(LSE + h * NQ + q0 + rm, m_i + tl.log2(l_i), mask=rm < q_len)

    def _serve_launch(q, k, v, cu_q, pos0, max_q, mode, block_table=None, cu_k=None, page=64, BN=64, BM=64):
        NQ, Hq, D = q.shape
        Hk = k.shape[-2]
        B = cu_q.numel() - 1
        o = torch.empty(NQ, Hq, D, device=q.device, dtype=q.dtype)
        if mode == MODE_PAGED:
            sk_p, sk_t, sk_h = k.stride(0), k.stride(1), k.stride(2)
            sv_p, sv_t, sv_h = v.stride(0), v.stride(1), v.stride(2)
            bt, s_bt, cuk = block_table, block_table.stride(0), cu_q
        else:
            sk_p, sk_t, sk_h = 0, k.stride(0), k.stride(1)
            sv_p, sv_t, sv_h = 0, v.stride(0), v.stride(1)
            bt, s_bt, cuk = cu_q, 0, cu_k
        if max_q > 0:
            _serve_fwd[(triton.cdiv(max_q, BM), B * Hq)](
                q, k, v, o, o, bt, cu_q, cuk, pos0, LOG2E / math.sqrt(D), Hq, NQ,
                q.stride(0), q.stride(1), sk_p, sk_t, sk_h, sv_p, sv_t, sv_h, o.stride(0), o.stride(1), s_bt,
                GROUP=Hq // Hk, BM=BM, BN=BN, D=D, PAGE=page, MODE=mode, STORE_LSE=False,
                num_warps=4, num_stages=2)
        _note("bi::attention_serve", "attention", (q, k, v), o)
        return o

    class _MM(torch.autograd.Function):
        """a [M, K] @ b [K, N] with the one-config GEMM. The backward uses the same kernel:
        dA = dC b^T keeps rows independent; dB = a^T dC sums the M rows in ascending order."""

        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return mm_one_config(a, b)

        @staticmethod
        def backward(ctx, g):
            a, b = ctx.saved_tensors
            ga = mm_one_config(g, b.t().contiguous()) if ctx.needs_input_grad[0] else None
            gb = mm_one_config(a.t().contiguous(), g) if ctx.needs_input_grad[1] else None
            return ga, gb

    class _RowSumSq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xf):
            ctx.save_for_backward(xf)
            return row_sum(xf, square=True)

        @staticmethod
        def backward(ctx, g):
            (xf,) = ctx.saved_tensors
            return g * (2 * xf)


# ----------------------------------------------------------------------------- op sets
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, Hk, T, D] -> [B, Hk * n_rep, T, D], as transformers' repeat_kv."""
    if n_rep == 1:
        return x
    b, hk, t, d = x.shape
    return x[:, :, None].expand(b, hk, n_rep, t, d).reshape(b, hk * n_rep, t, d)


def _i32(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.int32)


class StockOps:
    """What a plain torch model runs: cuBLAS GEMMs, SDPA, torch reductions."""
    name = "stock"

    def linear(self, x, weight, wt=None, f32_out=False):
        if f32_out:                                       # stand-in: fp32 operands, so no cast of the sum
            return F.linear(x.float(), weight.float())
        return F.linear(x, weight)

    def mm(self, a, b):                                   # a [M, K] @ b [K, N], for the MoE layer
        return a @ b

    def mean_sq(self, xf):
        return xf.pow(2).mean(-1, keepdim=True)

    def row_mean(self, xf):
        return xf.mean(-1, keepdim=True)

    def _sdpa(self, q, k, v, mask, causal, scale):
        """q [n, tq, Hq, D], k, v [n, tk, Hk, D] -> [n, tq, Hq, D]. As transformers' sdpa_attention_forward."""
        g = q.shape[2] // k.shape[2]
        Q = q.transpose(1, 2).contiguous()
        K = _repeat_kv(k.transpose(1, 2), g).contiguous()
        V = _repeat_kv(v.transpose(1, 2), g).contiguous()
        o = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=causal, scale=scale)
        return o.transpose(1, 2).contiguous()

    def attn_full(self, q, k, v, scale):
        return self._sdpa(q, k, v, None, q.shape[1] > 1, scale)

    def _padded(self, q, k_pad, v_pad, counts, pos0, kv_len, scale):
        """Ragged rows through one SDPA call: pad the queries, mask keys past each query's position."""
        n, tq, tk = len(counts), max(counts), k_pad.shape[1]
        NQ, Hq, D = q.shape
        dev = q.device
        cnt, p0 = torch.as_tensor(counts, device=dev), torch.as_tensor(pos0, device=dev)
        uniform = min(counts) == tq and min(pos0) == max(pos0) and min(kv_len) == tk
        if uniform:
            qp = q.view(n, tq, Hq, D)
            if tq == 1:                                   # one decode token per row: nothing to mask
                return self._sdpa(qp, k_pad, v_pad, None, False, scale).reshape(NQ, Hq, D)
            if pos0[0] == 0:                              # prefill
                return self._sdpa(qp, k_pad, v_pad, None, True, scale).reshape(NQ, Hq, D)
        valid = torch.arange(tq, device=dev)[None, :] < cnt[:, None]                     # [n, tq]
        qp = q.new_zeros(n, tq, Hq, D)
        qp[valid] = q
        key = torch.arange(tk, device=dev)
        lim = p0[:, None] + torch.arange(tq, device=dev)[None, :]                        # [n, tq]
        mask = (key[None, None, :] <= lim[:, :, None]) & valid[:, :, None]
        mask = mask | (~valid[:, :, None] & (key[None, None, :] == 0))                   # padded rows see key 0
        o = self._sdpa(qp, k_pad, v_pad, mask[:, None], False, scale)
        return o[valid]

    def attn_cached(self, q, kc, vc, slots, counts, pos0, scale):
        kv_len = [p + c for p, c in zip(pos0, counts)]
        idx = torch.as_tensor(slots, device=q.device)
        tk = max(kv_len)
        return self._padded(q, kc[idx, :tk], vc[idx, :tk], counts, pos0, kv_len, scale)

    def attn_packed(self, q, k, v, counts, scale):
        n, tk = len(counts), max(counts)
        dev = q.device
        valid = torch.arange(tk, device=dev)[None, :] < torch.as_tensor(counts, device=dev)[:, None]
        kp = k.new_zeros(n, tk, *k.shape[1:])
        vp = v.new_zeros(n, tk, *v.shape[1:])
        kp[valid], vp[valid] = k, v
        return self._padded(q, kp, vp, counts, [0] * n, list(counts), scale)


class BiOps:
    """Batch-invariant kernels: one configuration per op, whatever the batch."""
    name = "bi"

    def linear(self, x, weight, wt=None, f32_out=False):
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        grad = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
        if f32_out:
            y = mm_one_config_f32out(x2, wt)
        elif grad:
            y = _MM.apply(x2, weight.t().contiguous())
        else:
            y = mm_one_config(x2, wt)
        _note("bi::mm_one_config", "gemm", (x2, weight), y)
        return y.view(*x.shape[:-1], weight.shape[0])

    def mm(self, a, b):
        if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
            y = _MM.apply(a, b)
        else:
            y = mm_one_config(a, b)
        _note("bi::mm_one_config", "gemm", (a, b), y)
        return y

    def mean_sq(self, xf):
        K = xf.shape[-1]
        x2 = xf.reshape(-1, K)
        s = _RowSumSq.apply(x2) if (torch.is_grad_enabled() and xf.requires_grad) else row_sum(x2, square=True)
        return s.view(*xf.shape[:-1], 1) * (1.0 / K)

    def row_mean(self, xf):
        K = xf.shape[-1]
        return row_sum(xf.reshape(-1, K)).view(*xf.shape[:-1], 1) * (1.0 / K)

    def attn_full(self, q, k, v, scale):
        Q, K, V = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)        # [B, H, T, D] views
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
            o = attention_fixed_blocks_train(Q, K, V)
        else:
            o = attention_fixed_blocks(Q, K, V, 0)
        _note("bi::attention_fixed_blocks", "attention", (Q, K, V), o)
        return o.transpose(1, 2)

    def attn_cached(self, q, kc, vc, slots, counts, pos0, scale):
        dev = q.device
        cu = _i32([0] + list(torch.as_tensor(counts).cumsum(0).tolist()), dev)
        bt = _i32(slots, dev).view(-1, 1)
        return _serve_launch(q, kc, vc, cu, _i32(pos0, dev), max(counts), MODE_PAGED,
                             block_table=bt, page=kc.shape[1])

    def attn_packed(self, q, k, v, counts, scale):
        dev = q.device
        cu = _i32([0] + list(torch.as_tensor(counts).cumsum(0).tolist()), dev)
        return _serve_launch(q, k, v, cu, torch.zeros(len(counts), device=dev, dtype=torch.int32),
                             max(counts), MODE_VARLEN, cu_k=cu)


OPS = {"stock": StockOps(), "bi": BiOps()}


def _ops(name):
    if name == "bi" and not HAVE_TRITON:
        raise RuntimeError("the bi op set needs triton and tools/bi_kernels.py")
    return OPS[name]


class _Ctx:
    """Shared by every module of one model: the op set, the rounding choices, the RoPE tables."""

    def __init__(self, cfg: Config, ops: str):
        self.cfg, self.ops, self.r = cfg, _ops(ops), Rounding()
        self.dt = DTYPES[cfg.dtype]
        self._tables = {}

    def tables(self, device) -> dict:
        key = str(device)
        if key not in self._tables:
            cfg, D = self.cfg, self.cfg.head_dim
            # as transformers: inv_freq on the CPU, the angle and cos/sin on the model's device, in fp32
            inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.int64).float() / D))
            pos = torch.arange(cfg.max_positions, dtype=torch.int64)

            def table(dev):                               # reads `inv` when called
                fr = (inv.to(dev)[None, :, None].float() @ pos.to(dev)[None, None, :].float()).transpose(1, 2)[0]
                emb = torch.cat((fr, fr), -1)
                return emb.cos(), emb.sin()
            c, s = table(device)
            cc, sc = table("cpu")
            inv = inv.to(self.dt).float()                 # the "inv_freq_dt" variant: frequencies rounded to dt
            ci, si = table(device)
            self._tables[key] = {"cos32": c, "sin32": s, "cos": c.to(self.dt), "sin": s.to(self.dt),
                                 "cos_cpu": cc.to(device).to(self.dt), "sin_cpu": sc.to(device).to(self.dt),
                                 "cos_invdt": ci.to(self.dt), "sin_invdt": si.to(self.dt)}
        return self._tables[key]


class _M(nn.Module):
    """Base: holds the shared context without registering it as a submodule."""

    def __init__(self, ctx: _Ctx):
        super().__init__()
        object.__setattr__(self, "ctx", ctx)


# ----------------------------------------------------------------------------- modules
class Linear(_M):
    bi_class = "gemm"

    def __init__(self, ctx, k: int, n: int, bias: bool = False):
        super().__init__(ctx)
        self.weight = nn.Parameter(torch.empty(n, k))
        self.bias = nn.Parameter(torch.zeros(n)) if bias else None
        self._wt, self._wt_key = None, None

    def weight_t(self) -> torch.Tensor:
        """The weight as contiguous [K, N] for the one-config GEMM, refreshed when the weight changes."""
        w = self.weight
        key = (w._version, w.data_ptr(), w.dtype)
        if self._wt_key != key:
            self._wt, self._wt_key = w.detach().t().contiguous(), key
        return self._wt

    def forward(self, x, f32_out: bool = False):
        ops = self.ctx.ops
        if self.bias is None:
            return ops.linear(x, self.weight, self.weight_t() if ops.name == "bi" else None, f32_out)
        # With a bias (Qwen2's q/k/v): transformers calls F.linear(x, W, b), which runs addmm. cuBLAS
        # adds the bias to the fp32 sum in its epilogue and rounds once. The bi op set does the same:
        # fp32-output GEMM, fp32 bias add, one cast. Under autograd it adds the bias after the cast.
        if ops.name == "stock":
            return F.linear(x, self.weight, self.bias)
        if torch.is_grad_enabled() and (x.requires_grad or self.weight.requires_grad):
            return ops.linear(x, self.weight, self.weight_t(), False) + self.bias
        y = ops.linear(x, self.weight, self.weight_t(), True) + self.bias.float()
        return y if f32_out else y.to(self.ctx.dt)


class RMSNorm(_M):
    bi_class = "rowwise"

    def __init__(self, ctx, hidden: int):
        super().__init__(ctx)
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, x):
        ops, dt, eps, mode = self.ctx.ops, self.ctx.dt, self.ctx.cfg.rms_eps, self.ctx.r.norm
        if mode == "all_dt":                              # squares, mean, rsqrt and scale in dt
            x = x.to(dt)
            var = ops.row_mean((x * x).float()).to(dt)
            return x * torch.rsqrt(var + eps) * self.weight
        xf = x.float()
        y = xf * torch.rsqrt(ops.mean_sq(xf) + eps)
        if mode == "weight_then_cast":
            return (y * self.weight.float()).to(dt)
        return self.weight * y.to(dt)                     # reference: cast, then the weight


class RoPE(_M):
    bi_class = "elementwise"

    def forward(self, q, k, pos):
        """q [..., Hq, D], k [..., Hk, D], pos [...] absolute positions (int64)."""
        t, mode, dt = self.ctx.tables(q.device), self.ctx.r.rope, self.ctx.dt

        def rot(x):
            h = x.shape[-1] // 2
            return torch.cat((-x[..., h:], x[..., :h]), -1)
        if mode == "fp32":
            cos, sin = t["cos32"][pos].unsqueeze(-2), t["sin32"][pos].unsqueeze(-2)
            return ((q.float() * cos + rot(q).float() * sin).to(dt),
                    (k.float() * cos + rot(k).float() * sin).to(dt))
        sfx = {"hf": "", "cpu_table": "_cpu", "inv_freq_dt": "_invdt"}[mode]
        cos, sin = t["cos" + sfx][pos].unsqueeze(-2), t["sin" + sfx][pos].unsqueeze(-2)
        return (q * cos) + (rot(q) * sin), (k * cos) + (rot(k) * sin)


class AttnCore(_M):
    """Causal attention of full sequences: q [B, T, Hq, D], k, v [B, T, Hk, D] -> [B, T, Hq, D]."""
    bi_class = "attention"

    def forward(self, q, k, v):
        return self.ctx.ops.attn_full(q, k, v, self.ctx.cfg.head_dim ** -0.5)


class Attention(_M):
    def __init__(self, ctx):
        super().__init__(ctx)
        c = ctx.cfg
        self.q_proj = Linear(ctx, c.hidden, c.heads * c.head_dim, bias=c.qkv_bias)
        self.k_proj = Linear(ctx, c.hidden, c.kv_heads * c.head_dim, bias=c.qkv_bias)
        self.v_proj = Linear(ctx, c.hidden, c.kv_heads * c.head_dim, bias=c.qkv_bias)
        self.o_proj = Linear(ctx, c.heads * c.head_dim, c.hidden)
        self.rope = RoPE(ctx)
        self.core = AttnCore(ctx)

    def qkv(self, a, pos):
        c = self.ctx.cfg
        lead = a.shape[:-1]
        q = self.q_proj(a).view(*lead, c.heads, c.head_dim)
        k = self.k_proj(a).view(*lead, c.kv_heads, c.head_dim)
        v = self.v_proj(a).view(*lead, c.kv_heads, c.head_dim)
        q, k = self.rope(q, k, pos)
        return q, k, v

    def forward(self, a, pos):                            # a [B, T, H], pos [B, T]
        q, k, v = self.qkv(a, pos)
        o = self.core(q, k, v)
        return self.o_proj(o.reshape(*a.shape[:-1], -1))


class SiluMul(_M):
    bi_class = "elementwise"

    def forward(self, gate, up):
        return F.silu(gate) * up


class MLP(_M):
    def __init__(self, ctx):
        super().__init__(ctx)
        c = ctx.cfg
        self.gate_proj = Linear(ctx, c.hidden, c.intermediate)
        self.up_proj = Linear(ctx, c.hidden, c.intermediate)
        self.down_proj = Linear(ctx, c.intermediate, c.hidden)
        self.act = SiluMul(ctx)

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x), self.up_proj(x)))


_MOE_BI = None


def _moe_bi_module():
    global _MOE_BI
    if _MOE_BI is None:
        sys.path.insert(0, os.path.join(HERE, "..", "moe_combine"))
        import moe_bi
        _MOE_BI = moe_bi
    return _MOE_BI


class MoE(_M):
    """The layer of examples/moe_combine/moe_bi.py (_moe_bi: softmax router, top-k, renormalized
    weights, SwiGLU experts, per-token fp32 combine in top-k rank order). Both op sets run this code
    and differ only in the GEMM. moe_bi.py keeps its sizes in module constants; they are set here
    before each call, so one process should not run two MoE geometries on different threads."""
    bi_class = "moe"

    def __init__(self, ctx):
        super().__init__(ctx)
        c = ctx.cfg
        self.Wr = nn.Parameter(torch.empty(c.hidden, c.moe_experts))
        self.W1 = nn.Parameter(torch.empty(c.moe_experts, c.hidden, c.moe_intermediate))
        self.W3 = nn.Parameter(torch.empty(c.moe_experts, c.hidden, c.moe_intermediate))
        self.W2 = nn.Parameter(torch.empty(c.moe_experts, c.moe_intermediate, c.hidden))

    def forward(self, x):
        c, mb = self.ctx.cfg, _moe_bi_module()
        if self.ctx.dt != mb.BF:
            raise ValueError("the MoE variant is bf16 only")
        mb.H, mb.I, mb.E, mb.K = c.hidden, c.moe_intermediate, c.moe_experts, c.moe_topk
        P = {"Wr": self.Wr, "W1": self.W1, "W3": self.W3, "W2": self.W2}
        y = mb._moe_bi(x.reshape(-1, c.hidden), P, self.ctx.ops.mm)
        return y.view(x.shape)


class Block(_M):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.input_layernorm = RMSNorm(ctx, ctx.cfg.hidden)
        self.self_attn = Attention(ctx)
        self.post_attention_layernorm = RMSNorm(ctx, ctx.cfg.hidden)
        self.mlp = MoE(ctx) if ctx.cfg.moe else MLP(ctx)

    def add(self, s, o):
        """The residual add (R5, R8). `s` is the stream as the next norm reads it."""
        mode, dt = self.ctx.r.residual, self.ctx.dt
        if mode == "fp32_into_norm":                      # the stream is stored rounded; the norm sees the fp32 sum
            return s.to(dt).float() + o.float()
        if mode == "fp32_stream":
            return s.float() + o.float()
        return s + o

    def forward(self, s, pos):                            # trainer path: s [B, T, H], pos [B, T]
        s = self.add(s, self.self_attn(self.input_layernorm(s), pos))
        return self.add(s, self.mlp(self.post_attention_layernorm(s)))


class Backbone(_M):
    def __init__(self, ctx):
        super().__init__(ctx)
        c = ctx.cfg
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden)
        self.layers = nn.ModuleList([Block(ctx) for _ in range(c.layers)])
        self.norm = RMSNorm(ctx, c.hidden)

    def forward(self, ids):                               # ids [B, T] -> normed hidden [B, T, H]
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).expand(B, T)
        s = self.embed_tokens(ids)
        for blk in self.layers:
            s = blk(s, pos)
        return self.norm(s)


class KVCache:
    """Slot KV cache for the engine forward. Per layer, K and V of slot s live at [s, :length[s]]
    (K after RoPE, before the GQA expansion, as transformers caches them). A slot is one request.
    With poison=True unwritten entries hold NaN: the bi kernels never read them (loads are masked by
    each sequence's length). The stock path multiplies masked keys by a zero weight, so keep zeros there."""

    def __init__(self, model: "WholeModel", slots: int, capacity: int, poison: bool = False):
        c = model.cfg
        if capacity > c.max_positions:
            raise ValueError("capacity exceeds max_positions")
        dev = model.lm_head.weight.device
        shape = (c.layers, slots, capacity, c.kv_heads, c.head_dim)
        fill = float("nan") if poison else 0.0
        self.k = torch.full(shape, fill, device=dev, dtype=model.ctx.dt)
        self.v = torch.full(shape, fill, device=dev, dtype=model.ctx.dt)
        self.length = [0] * slots
        self.capacity = capacity

    def reset(self, slot: int) -> None:                   # a request finished or was preempted
        self.length[slot] = 0

    def truncate(self, slot: int, n: int) -> None:        # drop rejected speculative tokens
        assert 0 <= n <= self.length[slot]
        self.length[slot] = n

    def copy_prefix(self, dst: int, src: int, n: int, src_cache: Optional["KVCache"] = None) -> None:
        """A prefix-cache hit: slot dst starts from the first n cached positions of slot src."""
        sc = src_cache or self
        assert n <= sc.length[src]
        self.k[:, dst, :n] = sc.k[:, src, :n]
        self.v[:, dst, :n] = sc.v[:, src, :n]
        self.length[dst] = n


class WholeModel(_M):
    def __init__(self, cfg: Config, ops: str = "bi"):
        super().__init__(_Ctx(cfg, ops))
        self.cfg = cfg
        self.model = Backbone(self.ctx)
        self.lm_head = Linear(self.ctx, cfg.hidden, cfg.vocab_size)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    # ---- op set and rounding choices
    @property
    def ops(self) -> str:
        return self.ctx.ops.name

    def set_ops(self, ops: str) -> "WholeModel":
        self.ctx.ops = _ops(ops)
        return self

    @contextlib.contextmanager
    def using(self, ops: Optional[str] = None, **rounding):
        """Temporarily switch the op set and/or rounding choices: with model.using(ops="stock", rope="fp32")."""
        old = (self.ctx.ops, self.ctx.r)
        try:
            if ops is not None:
                self.ctx.ops = _ops(ops)
            if rounding:
                self.ctx.r = replace(self.ctx.r, **rounding)
            yield self
        finally:
            self.ctx.ops, self.ctx.r = old

    # ---- trainer forward
    def forward(self, ids):
        return self.lm_head(self.model(ids), f32_out=self.ctx.r.lm_head == "fp32")

    def prefill(self, ids, ops: Optional[str] = None):
        """Trainer forward: ids [B, T] -> logits [B, T, V], every position teacher-forced."""
        with self.using(ops):
            return self(ids)

    def logprobs(self, logits):
        """R10: fp32 log_softmax of the logits (or in dt, for that rounding variant)."""
        if self.ctx.r.logprob == "dt":
            return torch.log_softmax(logits.to(self.ctx.dt), -1)
        return torch.log_softmax(logits.float(), -1)

    # ---- engine forwards (packed rows)
    def _packed(self, ids, pos, attn):
        """ids, pos [NQ]; attn(layer, q [NQ, Hq, D], k, v [NQ, Hk, D]) -> o [NQ, Hq, D]."""
        bb = self.model
        s = bb.embed_tokens(ids)
        for i, blk in enumerate(bb.layers):
            q, k, v = blk.self_attn.qkv(blk.input_layernorm(s), pos)
            o = attn(i, q, k, v)
            s = blk.add(s, blk.self_attn.o_proj(o.reshape(o.shape[0], -1)))
            s = blk.add(s, blk.mlp(blk.post_attention_layernorm(s)))
        return self.lm_head(bb.norm(s), f32_out=self.ctx.r.lm_head == "fp32")

    @torch.no_grad()
    def forward_cached(self, cache: KVCache, slots: Sequence[int], tokens: Sequence[torch.Tensor],
                       ops: Optional[str] = None) -> List[torch.Tensor]:
        """Engine forward. Appends tokens[i] (1-D ids, length >= 1) to slot slots[i], whose cached
        length is its absolute start position, and returns the logits of the appended positions,
        one [t_i, V] tensor per row. Rows may differ in length and in position: decode steps,
        prefill chunks and speculative k-token appends mix in one call."""
        assert len(set(slots)) == len(slots) == len(tokens)
        counts = [int(t.numel()) for t in tokens]
        pos0 = [cache.length[s] for s in slots]
        assert all(c >= 1 for c in counts) and max(p + c for p, c in zip(pos0, counts)) <= cache.capacity
        dev = cache.k.device
        ids = torch.cat([t.reshape(-1) for t in tokens]).to(dev)
        cnt = torch.as_tensor(counts, device=dev)
        row = torch.repeat_interleave(torch.arange(len(slots), device=dev), cnt)
        first = torch.cumsum(cnt, 0) - cnt
        pos = torch.as_tensor(pos0, device=dev)[row] + (torch.arange(ids.numel(), device=dev) - first[row])
        row_slot = torch.as_tensor(list(slots), device=dev)[row]
        scale = self.cfg.head_dim ** -0.5
        with self.using(ops):
            o = self.ctx.ops

            def attn(i, q, k, v):
                cache.k[i][row_slot, pos] = k             # write first: the kernel reads one source
                cache.v[i][row_slot, pos] = v
                return o.attn_cached(q, cache.k[i], cache.v[i], list(slots), counts, pos0, scale)
            logits = self._packed(ids, pos, attn)
        for s, c in zip(slots, counts):
            cache.length[s] += c
        return list(torch.split(logits, counts))

    @torch.no_grad()
    def prefill_packed(self, seqs: Sequence[torch.Tensor], ops: Optional[str] = None) -> List[torch.Tensor]:
        """Packed varlen prefill (cu_seqlens): full sequences of different lengths, no padding and no
        cache. bi: K/V stay packed and each sequence's blocks start at its own position 0.
        stock: SDPA has no varlen path, so rows are padded inside the attention call and masked."""
        counts = [int(t.numel()) for t in seqs]
        dev = self.lm_head.weight.device
        ids = torch.cat([t.reshape(-1) for t in seqs]).to(dev)
        pos = torch.cat([torch.arange(c, device=dev) for c in counts])
        scale = self.cfg.head_dim ** -0.5
        with self.using(ops):
            o = self.ctx.ops
            logits = self._packed(ids, pos, lambda i, q, k, v: o.attn_packed(q, k, v, counts, scale))
        return list(torch.split(logits, counts))


def realized_logprob(logp: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """logp [..., T, V] (position t scores token t+1), ids [..., T] -> [..., T-1]: the log-probability
    the model gave to each token that actually came next."""
    return logp[..., :-1, :].gather(-1, ids[..., 1:].unsqueeze(-1)).squeeze(-1)


# ----------------------------------------------------------------------------- build, save, load
def init_weights(model: WholeModel, seed: int = 0) -> None:
    """Fixed-seed init on the CPU generator, the same on any device. Projections ~ N(0, 1/fan_in), so
    activations stay O(1) and attention scores have a spread near 1; embeddings ~ N(0, 1); norm
    weights 1 + N(0, norm_weight_std^2); lm_head ~ N(0, logit_std^2 / hidden), which is also the
    embedding when the two are tied; biases ~ N(0, 0.01)."""
    cfg, g = model.cfg, torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("embed_tokens.weight"):             # tied: it is also the lm_head
                w = torch.randn(p.shape, generator=g) * (cfg.logit_std / math.sqrt(cfg.hidden) if cfg.tie_embeddings else 1.0)
            elif name.endswith(".bias"):
                w = 0.1 * torch.randn(p.shape, generator=g)
            elif name == "lm_head.weight":
                w = torch.randn(p.shape, generator=g) * (cfg.logit_std / math.sqrt(cfg.hidden))
            elif p.dim() == 1:
                w = 1.0 + cfg.norm_weight_std * torch.randn(p.shape, generator=g)
            elif name.endswith(("Wr", "W1", "W3", "W2")):                  # MoE: [.., K, N], fan-in is K
                w = torch.randn(p.shape, generator=g) * p.shape[-2] ** -0.5
            else:                                                          # Linear weight [N, K]
                w = torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5
            p.copy_(w)


def build(cfg: Optional[Config] = None, ops: str = "bi", seed: int = 0, device: str = "cuda") -> WholeModel:
    """A randomly initialized model in eval mode, in cfg.dtype on `device`."""
    cfg = cfg or Config()
    model = WholeModel(cfg, ops)
    init_weights(model, seed)
    return model.to(device=device, dtype=DTYPES[cfg.dtype]).eval()


def save(model: WholeModel, path: str) -> None:
    torch.save({"cfg": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load(path: str, ops: str = "bi", device: str = "cuda") -> WholeModel:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**blob["cfg"])
    model = WholeModel(cfg, ops)
    model.load_state_dict(blob["state_dict"], strict=True)
    return model.to(device=device, dtype=DTYPES[cfg.dtype]).eval()


def load_hf_state_dict(model: WholeModel, state_dict: dict) -> None:
    """Fill the model from the state dict of a transformers LlamaForCausalLM or Qwen2ForCausalLM.
    Both use the names model.py uses (model.embed_tokens, model.layers.N.self_attn.{q,k,v,o}_proj,
    mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm, model.norm, lm_head), so the
    mapping is the identity; Qwen2 adds q/k/v biases and ties lm_head to the embedding. A tied
    checkpoint may omit lm_head.weight. Every parameter must be present with its shape; values are
    cast to the model's dtype."""
    own = dict(model.named_parameters())                        # a tied weight appears once
    sd = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
    if model.cfg.tie_embeddings:
        if "lm_head.weight" in sd and not torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"]):
            raise ValueError("cfg.tie_embeddings is set but lm_head.weight differs from the embedding")
        sd.pop("lm_head.weight", None)
    missing = sorted(set(own) - set(sd))
    extra = sorted(k for k in set(sd) - set(own) if not k.endswith(("inv_freq", "masked_bias", ".attn.bias")))
    if missing or extra:
        raise KeyError(f"state dict does not match: missing {missing[:5]}, unexpected {extra[:5]}")
    with torch.no_grad():
        for name, p in own.items():
            if sd[name].shape != p.shape:
                raise ValueError(f"{name}: shape {tuple(sd[name].shape)} != {tuple(p.shape)}")
            p.copy_(sd[name].to(device=p.device, dtype=p.dtype))


def from_hf(hf_model, ops: str = "bi", device: Optional[str] = None, **overrides) -> WholeModel:
    """A model with the architecture and weights of a loaded LlamaForCausalLM / Qwen2ForCausalLM."""
    p = next(hf_model.parameters())
    dtype = {torch.bfloat16: "bf16", torch.float32: "fp32"}[p.dtype]
    cfg = config_from_hf(hf_model.config, dtype=dtype, **overrides)
    model = WholeModel(cfg, ops).to(device=device or p.device, dtype=p.dtype).eval()
    load_hf_state_dict(model, hf_model.state_dict())
    return model


def export_hf(model: WholeModel, directory: str) -> None:
    """Write a LlamaForCausalLM (or Qwen2ForCausalLM) checkpoint that from_pretrained can load."""
    os.makedirs(directory, exist_ok=True)
    mt = hf_model_type(model.cfg)
    cfg = dict(hf_config(model.cfg), model_type=mt,
               architectures=[{"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM"}[mt]])
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if model.cfg.tie_embeddings:
        sd.pop("lm_head.weight")                                # tied: transformers rebuilds it
    try:
        from safetensors.torch import save_file
        save_file(sd, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})
    except ImportError:
        torch.save(sd, os.path.join(directory, "pytorch_model.bin"))
