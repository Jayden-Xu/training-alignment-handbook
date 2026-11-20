"""Engine side: run vLLM greedy on the fixed prompt set and record what the trainer must match.

One process per engine configuration (vLLM reads its settings at import and start-up):

    python engine_run.py --name bi_eager --bi --eager --out engine-bi_eager.pt

Records, per run (a batch size, or a repeat):
  ids (prompt + generated), prompt length, the realized-token logprob of every prompt position
  after the first (prompt_logprobs=0) and of every generated token (logprobs=0), as fp32 values.
Optional:
  --kl N        full bf16 logits of the generated positions of the first N prompts, alone (B=1),
                taken from the logits processor (for KL and for a logits-level comparison)
  --capture     eager only: inputs and outputs of every submodule of the decoder for one prompt
                (prefill and two decode steps), for op-by-op replay in the trainer
  --tput        output tokens/s at batch 1, 8, 64 (128-token prompts, 128 new tokens, no logprobs)
  --patch-lmhead  route the lm_head through vLLM's own batch-invariant matmul (on sm86 vLLM 0.19.1
                does not override aten::linear, so the lm_head stays on cuBLAS); a runtime patch in
                this process only

Works with vLLM 0.19.1 (V1 engine, in-process) and, for tokens and logprobs only, vLLM 0.6.3.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"))
    p.add_argument("--bi", action="store_true", help="VLLM_BATCH_INVARIANT=1")
    p.add_argument("--eager", action="store_true", help="enforce_eager=True")
    p.add_argument("--backend", default=None, help="attention backend name, e.g. TRITON_ATTN")
    p.add_argument("--nprompts", type=int, default=64)
    p.add_argument("--gen", type=int, default=48)
    p.add_argument("--batches", default="1,4,16,64")
    p.add_argument("--repeat", action="store_true", help="repeat the largest batch (L1)")
    p.add_argument("--kl", type=int, default=0)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--tput", action="store_true")
    p.add_argument("--patch-lmhead", action="store_true")
    p.add_argument("--gpu-mem", type=float, default=0.5)
    p.add_argument("--max-batched-tokens", type=int, default=None)
    return p.parse_args()


ARGS = parse()
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")      # model in this process: hooks work
if ARGS.bi:
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
if ARGS.backend:
    os.environ["VLLM_ATTENTION_BACKEND"] = ARGS.backend           # read by vLLM 0.6.3

import torch                                                       # noqa: E402
import vllm                                                        # noqa: E402
from vllm import LLM, SamplingParams                              # noqa: E402

from prompts import prompt_ids                                     # noqa: E402

V019 = not vllm.__version__.startswith("0.6")

if ARGS.patch_lmhead:
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe
    from vllm.model_executor.layers.batch_invariant import linear_batch_invariant

    def _apply(self, layer, x, bias=None):
        return linear_batch_invariant(x, layer.weight, bias)
    vpe.UnquantizedEmbeddingMethod.apply = _apply


def build_llm():
    kw = dict(model=ARGS.model, dtype="bfloat16", gpu_memory_utilization=ARGS.gpu_mem, max_model_len=2048,
              seed=0, enforce_eager=ARGS.eager, enable_prefix_caching=False)
    if ARGS.max_batched_tokens:
        kw["max_num_batched_tokens"] = ARGS.max_batched_tokens
    if V019 and ARGS.backend:
        kw["attention_backend"] = ARGS.backend
    return LLM(**kw)


def get_model(llm):
    if V019:
        return llm.apply_model(lambda m: m)[0]
    return llm.llm_engine.model_executor.driver_worker.model_runner.model


def _lp(d, tok):
    v = d[tok]
    return float(v.logprob if hasattr(v, "logprob") else v)


def run_batch(llm, prompts, idx, gen):
    sp = SamplingParams(temperature=0.0, max_tokens=gen, logprobs=0, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": prompts[i]} for i in idx], sp, use_tqdm=False)
    rec = {}
    for i, o in zip(idx, outs):
        P = list(o.prompt_token_ids)
        G = list(o.outputs[0].token_ids)
        plp = [_lp(o.prompt_logprobs[t], P[t]) for t in range(1, len(P))]
        glps = o.outputs[0].logprobs
        glp = [_lp(glps[t], G[t]) for t in range(len(G))]
        rec[i] = {"ids": P + G, "P": len(P), "plp": plp, "glp": glp}
    return rec


class Recorder:
    """Forward hooks that copy what a module received and returned, while switched on."""

    def __init__(self):
        self.on, self.data, self.handles = False, {}, []

    @staticmethod
    def _cp(x):
        if torch.is_tensor(x):
            return x.detach().to("cpu", copy=True)
        if isinstance(x, (tuple, list)):
            return type(x)(Recorder._cp(y) for y in x)
        if x is None or isinstance(x, (int, float, bool, str)):
            return x
        return f"<{type(x).__name__}>"                                # keep engine objects out of the record

    def attach(self, name, mod, inputs=True):
        def pre(m, args, kwargs):
            if self.on:
                self.data.setdefault(name, []).append({"in": self._cp(args), "kw": {k: self._cp(v) for k, v in kwargs.items()}})

        def post(m, args, kwargs, out):
            if self.on:
                self.data[name][-1]["out"] = self._cp(out)
        self.handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        self.handles.append(mod.register_forward_hook(post, with_kwargs=True))


def main():
    t0 = time.time()
    llm = build_llm()
    t_init = time.time() - t0
    tok = llm.get_tokenizer()
    prompts = prompt_ids(tok, ARGS.nprompts)
    res = {"name": ARGS.name, "vllm": vllm.__version__, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "args": vars(ARGS), "init_s": t_init,
           "env": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_", "CUBLAS"))},
           "prompts": prompts, "runs": {}}
    model = get_model(llm)
    try:
        res["attn_backend"] = type(model.model.layers[0].self_attn.attn.impl).__name__
    except Exception as e:                                          # noqa: BLE001
        res["attn_backend"] = f"unknown ({type(e).__name__})"
    try:
        res["rope_cache"] = model.model.layers[0].self_attn.rotary_emb.cos_sin_cache[:4096].detach().cpu()
    except Exception:                                               # noqa: BLE001
        pass
    n = len(prompts)
    batches = [int(b) for b in ARGS.batches.split(",")]
    for B in batches:
        rec = {}
        for s in range(0, n, B):
            rec.update(run_batch(llm, prompts, list(range(s, min(n, s + B))), ARGS.gen))
        res["runs"][f"B{B}"] = rec
        print(f"[{ARGS.name}] B={B} done {time.time() - t0:.0f}s", flush=True)
    if ARGS.repeat:
        B = max(batches)
        rec = {}
        for s in range(0, n, B):
            rec.update(run_batch(llm, prompts, list(range(s, min(n, s + B))), ARGS.gen))
        res["runs"][f"B{B}_repeat"] = rec

    if ARGS.kl:
        logits = []
        h = model.logits_processor.register_forward_hook(lambda m, a, o: logits.append(o.detach().to("cpu", copy=True)))
        kl = {}
        sp = SamplingParams(temperature=0.0, max_tokens=ARGS.gen, logprobs=0)
        for i in range(min(ARGS.kl, n)):
            logits.clear()
            o = llm.generate([{"prompt_token_ids": prompts[i]}], sp, use_tqdm=False)[0]
            G = list(o.outputs[0].token_ids)
            rows = torch.cat([x.reshape(-1, x.shape[-1]) for x in logits])[-len(G):]
            kl[i] = {"ids": prompts[i] + G, "P": len(prompts[i]), "logits": rows.to(torch.bfloat16)
                     if rows.dtype == torch.bfloat16 else rows,
                     "glp": [_lp(o.outputs[0].logprobs[t], G[t]) for t in range(len(G))]}
        h.remove()
        res["kl"] = kl

    if ARGS.capture:
        r = Recorder()
        r.attach("embed_tokens", model.model.embed_tokens)
        for i, L in enumerate(model.model.layers):
            for sub in ("input_layernorm", "self_attn.qkv_proj", "self_attn.rotary_emb", "self_attn.attn",
                        "self_attn.o_proj", "post_attention_layernorm", "mlp.gate_up_proj", "mlp.act_fn",
                        "mlp.down_proj"):
                m = L
                for part in sub.split("."):
                    m = getattr(m, part)
                r.attach(f"layers.{i}.{sub}", m)
        r.attach("norm", model.model.norm)
        r.attach("logits_processor", model.logits_processor)
        r.on = True
        i = 4                                                      # a 48-token prompt
        sp = SamplingParams(temperature=0.0, max_tokens=3, logprobs=0)
        o = llm.generate([{"prompt_token_ids": prompts[i]}], sp, use_tqdm=False)[0]
        r.on = False
        for hd in r.handles:
            hd.remove()
        res["capture"] = {"prompt": i, "ids": prompts[i] + list(o.outputs[0].token_ids), "P": len(prompts[i]),
                          "calls": r.data}

    if ARGS.tput:
        g = torch.Generator().manual_seed(1)
        tp = [torch.randint(100, 50000, (128,), generator=g).tolist() for _ in range(64)]
        sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
        out = {}
        for B in (1, 8, 64):
            inp = [{"prompt_token_ids": tp[j]} for j in range(B)]
            llm.generate(inp, sp, use_tqdm=False)                   # warm-up
            ts = []
            for _ in range(3):
                torch.cuda.synchronize()
                a = time.perf_counter()
                llm.generate(inp, sp, use_tqdm=False)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - a)
            med = statistics.median(ts)
            out[B] = {"median_s": med, "out_tok_per_s": B * 128 / med, "runs_s": ts}
            print(f"[{ARGS.name}] tput B={B}: {B * 128 / med:.0f} tok/s", flush=True)
        res["tput"] = out

    torch.save(res, ARGS.out)
    print(f"[{ARGS.name}] wrote {ARGS.out} in {time.time() - t0:.0f}s backend={res['attn_backend']}", flush=True)
    os._exit(0)                                                    # skip slow engine teardown


if __name__ == "__main__":
    main()
