"""Run tools/bi_audit.py on model.py (both op sets) and on a transformers LlamaForCausalLM with
random weights (code this handbook did not write). Writes audit-*.json / audit-*.md.

Rows are one-token or short sequences so the 2-D kernels see small row counts (B x T rows).

Run (CUDA): python audit_model.py --out DIR [--quick]   (needs transformers for the HF part)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, HERE)
import bi_audit  # noqa: E402
import model as wm  # noqa: E402

dev = "cuda"


def rows_for(V, n, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, V, (T,), generator=g).to(dev) for _ in range(n)]


def logits_of(m, batch):
    out = m(batch, use_cache=False) if hasattr(m, "config") else m(batch)   # HF: no KV cache object
    return out.logits if hasattr(out, "logits") else out


def loss_fn(out, batch):
    lg = out.logits if hasattr(out, "logits") else out
    return -wm.realized_logprob(torch.log_softmax(lg.float(), -1), batch).sum()


def run(name, model, rows, out_dir, sizes, loss=True, tag=None):
    rep = bi_audit.audit(model, rows, lambda r: torch.stack(r), sizes=sizes, name=name,
                         forward=lambda m, b: logits_of(m, b), loss_fn=loss_fn if loss else None,
                         recipes=bi_audit.recipes_link(out_dir))
    rep = bi_audit._jsonable(rep)
    tag = tag or name.split()[0]
    with open(os.path.join(out_dir, f"audit-{tag}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(out_dir, f"audit-{tag}.md"), "w") as f:
        f.write(bi_audit.to_markdown(rep, leaf_only=True))
    d = rep["first_divergence"]
    print(f"{name}: {rep['summary']}; first divergence: {d['first']['module'] if d['first'] else 'none'}; "
          f"origins: {[o['module'] for o in d['origins']][:8]}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="stock,bi,hf")
    args = ap.parse_args()
    only = args.only.split(",")
    cfg = wm.Config(layers=2) if not args.quick else wm.Config(layers=1, vocab_size=4000)
    sizes = (2, 3, 4, 8, 9, 16, 17, 33) if not args.quick else (2, 3, 8)
    n, T = 34, 16
    for ops in ("stock", "bi"):
        if ops in only:
            m = wm.build(cfg, ops=ops, seed=0)
            m.ctx.tables(torch.device(dev))                     # build the RoPE tables outside the audit
            for p in m.parameters():
                p.requires_grad_(True)
            run(f"{ops} model.py dense, {cfg.layers} layers, rows of {T} tokens", m, rows_for(cfg.vocab_size, n, T), args.out, sizes)
            if not args.quick:                                  # one-token rows: the kernels see 1..33 rows
                run(f"{ops} model.py dense, {cfg.layers} layers, rows of 1 token", m, rows_for(cfg.vocab_size, n, 1),
                    args.out, sizes, tag=f"{ops}-1token")
            del m
            torch.cuda.empty_cache()
    if "hf" in only:
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(0)
        hcfg = LlamaConfig(**wm.hf_config(cfg))
        hcfg._attn_implementation = "sdpa"
        hf = LlamaForCausalLM(hcfg).to(dev, torch.bfloat16).eval()
        run(f"hf LlamaForCausalLM (transformers, random weights, sdpa), {cfg.layers} layers, rows of {T} tokens", hf,
            rows_for(cfg.vocab_size, n, T), args.out, sizes)


if __name__ == "__main__":
    main()
