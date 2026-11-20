"""A small decoder LM and its training step, with stock and fixed-order parts.

Model: token + position embedding, N x (RMSNorm, GQA causal attention, RMSNorm, SwiGLU),
RMSNorm, untied head. Weights are fp32; the forward runs under bf16 (or fp16) autocast.

Parts that can be swapped (Cfg):
  attn  "sdpa"   F.scaled_dot_product_attention on K/V expanded to the query heads
        "fixed"  tools/bi_kernels.attention_fixed_blocks_train (atomic-free backward)
  emb   "stock"  F.embedding
        "fixed"  ts_kernels.gather_rows_fixed (fixed-order backward)
  loss  "stock"  F.cross_entropy(mean)
        "fixed"  per-token losses reduced with ts_kernels.tree_mean_loss

Trainer: AdamW, warmup + cosine LambdaLR, optional GradScaler, optional clipping, a data
sampler that reshuffles every epoch from the CPU RNG, and a checkpoint that holds every piece
of state a bitwise resume needs.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from bi_kernels import attention_fixed_blocks_train
from ts_kernels import fingerprint, gather_rows_fixed, tree_mean_loss, tree_sum
from ts_util import BF, DEV, bits

STATE_PIECES = ("model", "optimizer", "scheduler", "scaler", "cpu_rng", "cuda_rng", "data")


@dataclass
class Cfg:
    vocab: int = 8192
    d_model: int = 1024
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 2048
    max_len: int = 1024
    dropout: float = 0.0
    attn: str = "sdpa"
    emb: str = "stock"
    loss: str = "stock"


@dataclass
class TrainCfg:
    seed: int = 1234
    data_seed: int = 99
    batch: int = 4
    seq: int = 1024
    steps: int = 20
    steps_per_epoch: int = 16
    lr: float = 3e-4
    warmup: int = 8
    weight_decay: float = 0.1
    amp: str = "bf16"               # "bf16" | "fp16" | "none"
    scaler: bool = False
    scaler_growth_interval: int = 2000
    scaler_init_scale: float = 65536.0
    clip: float = 1.0               # 0 disables clipping
    clip_path: str = "foreach"      # "foreach" | "single" | "fixed"
    opt: str = "default"            # "default" | "fused" | "foreach" | "single"
    accum: int = 1                  # micro-batches per optimizer step
    accum_reverse: bool = False     # process the micro-batches in reverse order


def preset(mode: str) -> dict:
    """Model parts for a mode. fix_l1 swaps only what fails run-to-run in default mode;
    fix_all also replaces every data-dependent reduction with a written order."""
    return {"default": {}, "strict": {}, "fix_l1": {"attn": "fixed"},
            "fix_all": {"attn": "fixed", "emb": "fixed", "loss": "fixed"}}[mode]


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        d, self.hq, self.hk = c.d_model, c.n_heads, c.n_kv_heads
        self.hd = d // c.n_heads
        self.attn_kind = c.attn
        self.norm1 = RMSNorm(d)                          # declared in forward order, so that
        self.wq = nn.Linear(d, self.hq * self.hd, bias=False)   # reversed(named_parameters())
        self.wk = nn.Linear(d, self.hk * self.hd, bias=False)   # is the backward order
        self.wv = nn.Linear(d, self.hk * self.hd, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.w1 = nn.Linear(d, c.d_ff, bias=False)
        self.w3 = nn.Linear(d, c.d_ff, bias=False)
        self.w2 = nn.Linear(c.d_ff, d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        h = self.norm1(x)
        q = self.wq(h).view(B, T, self.hq, self.hd).transpose(1, 2)
        k = self.wk(h).view(B, T, self.hk, self.hd).transpose(1, 2)
        v = self.wv(h).view(B, T, self.hk, self.hd).transpose(1, 2)
        if self.attn_kind == "fixed":
            assert q.dtype == BF, "the fixed-block attention kernel is bf16 only"
            o = attention_fixed_blocks_train(q, k, v)
        else:
            g = self.hq // self.hk
            ke = k[:, :, None].expand(B, self.hk, g, T, self.hd).reshape(B, self.hq, T, self.hd)
            ve = v[:, :, None].expand(B, self.hk, g, T, self.hd).reshape(B, self.hq, T, self.hd)
            o = F.scaled_dot_product_attention(q, ke, ve, is_causal=True)
        x = x + self.drop(self.wo(o.transpose(1, 2).reshape(B, T, C)))
        h = self.norm2(x)
        return x + self.drop(self.w2(F.silu(self.w1(h)) * self.w3(h)))


class LM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.tok = nn.Parameter(torch.empty(c.vocab, c.d_model))
        self.pos = nn.Parameter(torch.empty(c.max_len, c.d_model))
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).expand(B, T)
        if self.c.emb == "fixed":
            x = gather_rows_fixed(self.tok, idx) + gather_rows_fixed(self.pos, pos)
        else:
            x = F.embedding(idx, self.tok) + F.embedding(pos, self.pos)
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

    def loss(self, logits, y):
        B, T, V = logits.shape
        lf = logits.float().view(-1, V)
        if self.c.loss == "fixed":
            return tree_mean_loss(F.cross_entropy(lf, y.reshape(-1), reduction="none").view(B, T))
        return F.cross_entropy(lf, y.reshape(-1))

    def backward_order(self):
        """(name, parameter) in the order the backward pass reaches them: head first."""
        return list(reversed(list(self.named_parameters())))


def fixed_global_norm(grads) -> torch.Tensor:
    """sqrt of the sum of squares with a written order: each tensor flattened row-major and
    reduced with the fixed pairwise tree in fp32, then the tensors' sums reduced with the same
    tree in the order given."""
    per = torch.stack([tree_sum(g.reshape(-1).float().square()) for g in grads])
    return tree_sum(per).sqrt()


def clip_fixed_(grads, max_norm: float) -> torch.Tensor:
    """clip_grad_norm_ with the fixed-order norm and one multiply per tensor."""
    total = fixed_global_norm(grads)
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for g in grads:
        g.mul_(coef)
    return total


class Sampler:
    """Non-overlapping windows of a token corpus. Every epoch draws a new permutation of the
    windows from the global CPU RNG."""

    def __init__(self, corpus: torch.Tensor, batch: int, seq: int, steps_per_epoch: int):
        self.corpus, self.batch, self.seq = corpus, batch, seq
        self.n = batch * steps_per_epoch
        assert corpus.numel() >= self.n * seq + 1
        self.epoch, self.pos, self.perm = -1, 0, None

    def next(self):
        if self.perm is None or self.pos >= self.n:
            self.perm = torch.randperm(self.n)          # global CPU generator
            self.pos = 0
            self.epoch += 1
        ids = self.perm[self.pos:self.pos + self.batch].tolist()
        self.pos += self.batch
        x = torch.stack([self.corpus[i * self.seq:i * self.seq + self.seq] for i in ids])
        y = torch.stack([self.corpus[i * self.seq + 1:i * self.seq + self.seq + 1] for i in ids])
        return x, y

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos, "perm": self.perm}

    def load_state_dict(self, sd):
        self.epoch, self.pos, self.perm = sd["epoch"], sd["pos"], sd["perm"]


def make_corpus(seed: int, vocab: int, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, generator=g) ** 3 * vocab).long().clamp_(max=vocab - 1)


def loss_bits(loss: torch.Tensor) -> str:
    return f"{int(bits(loss.detach().float().reshape(1))[0].item()) & 0xFFFFFFFF:08x}"


class Trainer:
    def __init__(self, cfg: Cfg, tc: TrainCfg):
        self.cfg, self.tc = cfg, tc
        torch.manual_seed(tc.seed)                       # CPU and CUDA generators
        self.model = LM(cfg).to(DEV)
        okw = {"default": {}, "fused": {"fused": True}, "foreach": {"foreach": True, "fused": False},
               "single": {"foreach": False, "fused": False}}[tc.opt]
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=tc.lr, betas=(0.9, 0.95),
                                     weight_decay=tc.weight_decay, **okw)
        total, warm = max(tc.steps, 1), tc.warmup
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warm) * (0.55 + 0.45 * math.cos(math.pi * s / total)))
        self.scaler = torch.amp.GradScaler("cuda", enabled=tc.scaler, init_scale=tc.scaler_init_scale,
                                           growth_interval=tc.scaler_growth_interval)
        corpus = make_corpus(tc.data_seed, cfg.vocab, tc.batch * tc.accum * tc.steps_per_epoch * tc.seq + 1)
        self.sampler = Sampler(corpus, tc.batch, tc.seq, tc.steps_per_epoch * tc.accum)
        self.step_idx = 0
        self.amp_dtype = {"bf16": BF, "fp16": torch.float16, "none": None}[tc.amp]

    # ------------------------------------------------------------------ one step
    def forward_loss(self, x, y):
        with torch.autocast("cuda", dtype=self.amp_dtype or BF, enabled=self.amp_dtype is not None):
            return self.model.loss(self.model(x), y)

    def backward(self, record: bool = False):
        """Zero the gradients, then accumulate tc.accum micro-batches. Returns the summed
        (already divided) micro-batch losses, in processing order."""
        self.opt.zero_grad(set_to_none=True)
        loss = total = None
        micro = [self.sampler.next() for _ in range(self.tc.accum)]
        if self.tc.accum_reverse:
            micro.reverse()
        for x, y in micro:
            loss = self.forward_loss(x.to(DEV), y.to(DEV))
            scaled = loss / self.tc.accum if self.tc.accum > 1 else loss
            self.scaler.scale(scaled).backward()
            total = scaled.detach() if total is None else total + scaled.detach()
        return total

    def clip(self):
        if not self.tc.clip:
            return None
        params = [p for p in self.model.parameters() if p.grad is not None]
        if self.tc.clip_path == "fixed":
            return clip_fixed_([p.grad for _, p in self.model.backward_order() if p.grad is not None], self.tc.clip)
        return torch.nn.utils.clip_grad_norm_(params, self.tc.clip, foreach=self.tc.clip_path == "foreach")

    def step(self, record: bool = True) -> dict:
        loss = self.backward()
        self.scaler.unscale_(self.opt)                   # no-op when the scaler is disabled
        rec = {"step": self.step_idx}
        if record:
            rec["loss"] = loss_bits(loss)
            rec["grads"] = {n: fingerprint(p.grad) for n, p in self.model.backward_order()}
        self.clip()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        if record:
            rec["weights"] = {n: fingerprint(p) for n, p in self.model.backward_order()}
            rec["lr"] = self.sched.get_last_lr()[0]
            rec["scale"] = self.scaler.get_scale() if self.tc.scaler else None
        self.step_idx += 1
        return rec

    # ------------------------------------------------------------------ state
    def checkpoint(self) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "data": self.sampler.state_dict(), "step": self.step_idx}

    def restore(self, ck: dict, omit=()) -> None:
        """Load every piece of state except those named in `omit`. An omitted piece keeps the
        value a new process has after the same constructor ran."""
        unknown = set(omit) - set(STATE_PIECES) - {"optimizer_step"}
        assert not unknown, unknown
        if "model" not in omit:
            self.model.load_state_dict(ck["model"])
        if "optimizer" not in omit:
            self.opt.load_state_dict(ck["optimizer"])
            if "optimizer_step" in omit:                 # moments restored, step counter lost
                for st in self.opt.state.values():
                    st["step"] = torch.zeros_like(st["step"]) if torch.is_tensor(st["step"]) else 0
        if "scheduler" not in omit:
            self.sched.load_state_dict(ck["scheduler"])
        if "scaler" not in omit and self.tc.scaler:
            self.scaler.load_state_dict(ck["scaler"])
        if "data" not in omit:
            self.sampler.load_state_dict(ck["data"])
        if "cpu_rng" not in omit:
            torch.set_rng_state(ck["cpu_rng"])
        if "cuda_rng" not in omit:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        self.step_idx = ck["step"]


def cfg_dict(cfg: Cfg, tc: TrainCfg) -> dict:
    return {"model": asdict(cfg), "train": asdict(tc)}


def first_divergence(a: list, b: list) -> dict:
    """a, b: per-step records of two runs. The first step and tensor whose fingerprints differ,
    in the order loss, gradients (backward order), weights after the update."""
    for ra, rb in zip(a, b):
        assert ra["step"] == rb["step"]
        gd = [n for n in ra["grads"] if ra["grads"][n] != rb["grads"][n]]
        wd = [n for n in ra["weights"] if ra["weights"][n] != rb["weights"][n]]
        if ra["loss"] != rb["loss"] or gd or wd:
            kind, name = (("loss", "loss") if ra["loss"] != rb["loss"] else
                          ("grad", gd[0]) if gd else ("weight", wd[0]))
            return {"equal": False, "step": ra["step"], "kind": kind, "tensor": name,
                    "loss_differs": ra["loss"] != rb["loss"],
                    "grads_differ": len(gd), "weights_differ": len(wd), "tensors": len(ra["grads"]),
                    "grads_equal_here": [n for n in ra["grads"] if n not in gd][:60]}
    return {"equal": True, "steps_compared": min(len(a), len(b))}
