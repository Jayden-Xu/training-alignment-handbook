"""A small decoder-only LM with bf16 activations: the model of the pretraining example.

Pre-norm blocks: RMSNorm, causal attention through torch SDPA, a GELU MLP.
Parameters stay fp32; the forward runs under autocast with a bf16 residual stream.

Two choices are constructor arguments, never different source lines, so that a
reference and a suspect run the same call sites:

    norm_mul  "act"   the norm rounds to the activation dtype before the weight and
                      multiplies there (what the reference stack does);
              "fp32"  it multiplies by the weight in fp32 and rounds once, after it
                      (what a fused norm kernel does).
    linear    "own"   Linear below calls F.linear from this file, so every GEMM is a
                      call site in model code;
              "nn"    torch.nn.Linear, whose GEMM is launched from torch's linear.py.

The code is written for capture by OpGuard as measured in the case page: one
operator per line where a boundary is wanted, and names longer than one letter
on those lines (its instrumenter matches call sites by the names around them).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """Bias-free linear layer that launches its GEMM from this file."""

    def __init__(self, fan_in: int, fan_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(fan_out, fan_in))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight)


def make_linear(kind: str, fan_in: int, fan_out: int) -> nn.Module:
    if kind == "own":
        return Linear(fan_in, fan_out)
    if kind == "nn":
        return nn.Linear(fan_in, fan_out, bias=False)
    raise ValueError("linear must be 'own' or 'nn'")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, norm_mul: str = "act", eps: float = 1e-6):
        super().__init__()
        if norm_mul not in ("act", "fp32"):
            raise ValueError("norm_mul must be 'act' or 'fp32'")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps, self.norm_mul = eps, norm_mul

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mul_dtype = hidden.dtype if self.norm_mul == "act" else torch.float32
        hidden32 = hidden.float()
        inv_rms = torch.rsqrt(hidden32.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = (hidden32 * inv_rms).to(mul_dtype)
        scaled = normed * self.weight.to(mul_dtype)
        return scaled.to(hidden.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, linear: str):
        super().__init__()
        self.heads = heads
        self.q_proj = make_linear(linear, dim, dim)
        self.k_proj = make_linear(linear, dim, dim)
        self.v_proj = make_linear(linear, dim, dim)
        self.o_proj = make_linear(linear, dim, dim)

    def heads_first(self, proj: torch.Tensor) -> torch.Tensor:
        batch, length, width = proj.shape
        return proj.view(batch, length, self.heads, width // self.heads).transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        query = self.heads_first(self.q_proj(hidden))
        key = self.heads_first(self.k_proj(hidden))
        value = self.heads_first(self.v_proj(hidden))
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        merged = attended.transpose(1, 2).reshape(batch, length, width)
        return self.o_proj(merged)


class MLP(nn.Module):
    def __init__(self, dim: int, linear: str):
        super().__init__()
        self.fc1 = make_linear(linear, dim, 4 * dim)
        self.fc2 = make_linear(linear, 4 * dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        widened = self.fc1(hidden)
        activated = F.gelu(widened)
        return self.fc2(activated)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, norm_mul: str, linear: str):
        super().__init__()
        self.norm1 = RMSNorm(dim, norm_mul)
        self.attn = Attention(dim, heads, linear)
        self.norm2 = RMSNorm(dim, norm_mul)
        self.mlp = MLP(dim, linear)
        self.drop = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(hidden))
        hidden = hidden + self.drop(attn_out)
        mlp_out = self.mlp(self.norm2(hidden))
        hidden = hidden + self.drop(mlp_out)
        return hidden


class TinyLM(nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int, heads: int, seq_len: int,
                 dropout: float = 0.1, norm_mul: str = "act", linear: str = "own"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, dropout, norm_mul, linear) for _ in range(layers))
        self.norm_f = RMSNorm(dim, norm_mul)
        self.head = make_linear(linear, dim, vocab)
        self.reset()

    def reset(self) -> None:
        """Weights as a trained checkpoint has them: norm weights are not exactly 1,
        so the place where a norm rounds matters from the first step."""
        for name, param in self.named_parameters():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name == "norm_f.weight":
                nn.init.normal_(param, mean=1.0, std=0.1)
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        pos_part = self.pos_emb(positions)
        embedded = self.tok_emb(tokens) + pos_part
        hidden = embedded.to(torch.bfloat16)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm_f(hidden)
        return self.head(hidden)
