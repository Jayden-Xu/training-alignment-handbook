"""Architecture check of model.py against Hugging Face transformers' LlamaForCausalLM.

This is a check of architectural equivalence, not a bitwise claim: the state dict
of model.py is loaded (strict) into a LlamaForCausalLM built from hf_config(cfg),
and the logits of the two are compared.

1. fp32        largest logit difference, stock op set, same batch (prefill)
2. fp32 cache  the same through the KV cache: model.forward_cached against HF with
               past_key_values, one token per step
3. bf16        how close the bf16 models are: stock against HF (same torch ops in the
               same order, so equal bits are possible), bi against HF (other kernels)
4. rope        model.py's cos/sin tables against HF's rotary_emb output, bitwise
6. qwen2       a random Qwen2ForCausalLM (bias on q/k/v, tied embeddings) loaded with from_hf()
5. export      export_hf() then from_pretrained(torch_dtype=bf16): the bf16 reference. A model
               cast with .to(bfloat16) instead rounds the RoPE inv_freq buffer to bf16; the
               difference is measured

Needs transformers (gpu-host: the maia env). No download: weights are random.

Run (CUDA):  python hf_check.py --json results-hf-a40.json [--quick]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, HERE)
from bi_check import bits, first_mismatch  # noqa: E402
from bi_report import Results  # noqa: E402
import model as wm  # noqa: E402

dev = "cuda"


def diff(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "ref_abs_max": float(b.abs().max()), "ref_std": float(b.std()),
            "elements": a.numel(), "bitwise_equal_elements": int((bits(a) == bits(b)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true", help="2 layers, vocab 4000")
    args = ap.parse_args()
    import transformers
    from transformers import LlamaConfig, LlamaForCausalLM

    res = Results("whole_model_hf", quick=args.quick)
    res.extra["transformers"] = transformers.__version__
    base = wm.Config(layers=2, vocab_size=4000) if args.quick else wm.Config()
    B, T = (2, 96) if args.quick else (4, 512)
    ids = torch.randint(0, base.vocab_size, (B, T), generator=torch.Generator().manual_seed(11)).to(dev)

    for dtype in ("fp32", "bf16"):
        cfg = wm.Config(**{**base.__dict__, "dtype": dtype})
        m = wm.build(cfg, ops="stock", seed=0)
        sd = m.state_dict()
        hf_cast = LlamaForCausalLM(LlamaConfig(**wm.hf_config(cfg))).to(dev, wm.DTYPES[dtype]).eval()
        same_keys = sorted(sd) == sorted(hf_cast.state_dict())
        hf_cast.load_state_dict(sd, strict=True)
        res.check(f"{dtype}: state dict keys equal HF's and load with strict=True", same_keys, level="other",
                  impl="model.py", expect="info", detail={"keys": len(sd)})
        if dtype == "fp32":
            hf = hf_cast
        else:
            # The reference bf16 model is the one from_pretrained(torch_dtype=bf16) gives: its RoPE
            # inv_freq buffer stays fp32. A model cast with .to(bfloat16) has inv_freq rounded to bf16.
            with tempfile.TemporaryDirectory() as tmp:
                wm.export_hf(m, tmp)
                files = sorted(os.listdir(tmp))
                hf = LlamaForCausalLM.from_pretrained(tmp, torch_dtype=torch.bfloat16).to(dev).eval()
            res.extra["export_files"] = files
            res.extra["inv_freq_dtype"] = {"from_pretrained": str(hf.model.rotary_emb.inv_freq.dtype),
                                           "cast_with_to": str(hf_cast.model.rotary_emb.inv_freq.dtype)}
            same_w = all(first_mismatch(sd[k], v) is None for k, v in hf.state_dict().items())
            res.check("export_hf() then from_pretrained(): every weight equals the state dict bitwise", same_w,
                      level="other", impl="model.py", expect="info", detail={"files": files})
        res.extra[f"hf_attn_implementation_{dtype}"] = hf.config._attn_implementation
        with torch.no_grad():
            ref = hf(ids).logits
            if dtype == "bf16":
                d = diff(hf_cast(ids).logits, ref)
                res.measure(f"bf16 HF cast with .to(bfloat16) against HF from_pretrained: max |logit diff| "
                            f"(inv_freq {res.extra['inv_freq_dtype']['cast_with_to']} against "
                            f"{res.extra['inv_freq_dtype']['from_pretrained']})", d["max_abs"], "logit", impl="hf", **d)
                print(f"bf16 HF .to(bf16) vs HF from_pretrained: {d}  inv_freq dtypes {res.extra['inv_freq_dtype']}")
            for ops in (("stock",) if dtype == "fp32" else ("stock", "bi")):
                d = diff(m.prefill(ids, ops=ops), ref)
                res.measure(f"{dtype} prefill B={B} T={T}: max |logit - HF logit|", d["max_abs"], "logit", impl=ops, **d)
                print(f"{dtype} {ops} prefill vs HF: {d}")
            # the KV-cache path: one token per step on both sides (first sequence)
            n = 48 if args.quick else 128
            cache = wm.KVCache(m, slots=1, capacity=n)
            mine = torch.cat([m.forward_cached(cache, [0], [ids[0, t:t + 1]], ops="stock")[0] for t in range(n)])
            past, outs = None, []
            for t in range(n):
                o = hf(ids[:1, t:t + 1], past_key_values=past, use_cache=True)
                past = o.past_key_values
                outs.append(o.logits[0])
            d = diff(mine, torch.cat(outs))
            res.measure(f"{dtype} cached decode {n} steps: max |logit - HF logit|", d["max_abs"], "logit", impl="stock", **d)
            print(f"{dtype} stock cached decode vs HF: {d}")
            # RoPE tables
            pos = torch.arange(T, device=dev)[None]
            tab = m.ctx.tables(torch.device(dev))
            for label, model_hf in (("HF", hf),) + ((("HF cast with .to(bfloat16)", hf_cast),) if dtype == "bf16" else ()):
                cos, sin = model_hf.model.rotary_emb(ref, pos)
                ne = int((bits(tab["cos"][:T]) != bits(cos[0])).sum() + (bits(tab["sin"][:T]) != bits(sin[0])).sum())
                res.check(f"{dtype}: RoPE cos/sin table equals the rotary_emb output of {label} bitwise (T={T})", ne == 0,
                          level="other", impl="model.py", expect="info", detail={"differing": ne, "of": 2 * cos[0].numel()})
                print(f"{dtype} rope table vs {label}: {ne} of {2 * cos[0].numel()} entries differ")
            cpu_ne = int((bits(tab["cos"]) != bits(tab["cos_cpu"])).sum() + (bits(tab["sin"]) != bits(tab["sin_cpu"])).sum())
            res.measure(f"{dtype}: RoPE table entries that differ between CPU and GPU cos/sin", cpu_ne, "elements",
                        impl="model.py", of=2 * tab["cos"].numel())
            print(f"{dtype} cpu vs gpu table entries differing: {cpu_ne} of {2 * tab['cos'].numel()}")
        del m, hf, hf_cast
        torch.cuda.empty_cache()
    # Qwen2 architecture (bias on q/k/v, tied embeddings, rope_theta 1e6): random Qwen2ForCausalLM -> model.py
    from transformers import Qwen2Config, Qwen2ForCausalLM
    qc = wm.Config.qwen2_0p5b()
    if args.quick:
        qc = wm.Config(**{**qc.__dict__, "layers": 2, "vocab_size": 8000})
    else:
        qc = wm.Config(**{**qc.__dict__, "layers": 4})          # Qwen2.5-0.5B shapes, 4 of 24 layers
    qids = torch.randint(0, qc.vocab_size, (2, 256), generator=torch.Generator().manual_seed(12)).to(dev)
    for dtype in ("fp32", "bf16"):
        torch.manual_seed(0)
        qcfg = Qwen2Config(**wm.hf_config(wm.Config(**{**qc.__dict__, "dtype": dtype})))
        hfq = Qwen2ForCausalLM(qcfg).to(dev, torch.float32)
        with torch.no_grad():                                   # HF inits biases to zero: make them visible
            for n_, p_ in hfq.named_parameters():
                if n_.endswith(".bias"):
                    p_.normal_(0, 0.1)
        hfq = hfq.to(dev, wm.DTYPES[dtype]).eval()
        if dtype == "bf16":                                     # keep inv_freq in fp32, as from_pretrained does
            hfq.model.rotary_emb.inv_freq = hfq.model.rotary_emb.original_inv_freq.to(dev).float()
        mq = wm.from_hf(hfq, ops="stock")
        ok_cfg = mq.cfg.qkv_bias and mq.cfg.tie_embeddings and mq.lm_head.weight is mq.model.embed_tokens.weight
        res.check(f"Qwen2 {dtype}: config_from_hf gives qkv_bias, tied embeddings, eps {mq.cfg.rms_eps}, theta {mq.cfg.rope_theta}; "
                  f"load_hf_state_dict fills every parameter", ok_cfg, level="other", impl="model.py", expect="info",
                  detail={"params": len(dict(mq.named_parameters()))})
        with torch.no_grad():
            ref = hfq(qids, use_cache=False).logits
            for ops in (("stock",) if dtype == "fp32" else ("stock", "bi")):
                d = diff(mq.prefill(qids, ops=ops), ref)
                res.measure(f"Qwen2 {dtype} prefill B=2 T=256 ({qc.layers} layers, vocab {qc.vocab_size}): max |logit - HF logit|",
                            d["max_abs"], "logit", impl=ops, **d)
                print(f"qwen2 {dtype} {ops} prefill vs HF: max {d['max_abs']:.3g}, equal elements {d['bitwise_equal_elements']}/{d['elements']}")
            n = 32
            cache = wm.KVCache(mq, slots=1, capacity=n)
            mine = torch.cat([mq.forward_cached(cache, [0], [qids[0, t:t + 1]], ops="stock")[0] for t in range(n)])
            d = diff(mine, ref[0, :n])
            res.measure(f"Qwen2 {dtype} cached decode {n} steps vs HF prefill: max |logit diff|", d["max_abs"], "logit", impl="stock", **d)
            print(f"qwen2 {dtype} stock decode vs HF prefill: max {d['max_abs']:.3g}")
        del hfq, mq
        torch.cuda.empty_cache()
    res.write(args.json)
    sys.exit(res.exit_code())


if __name__ == "__main__":
    main()
