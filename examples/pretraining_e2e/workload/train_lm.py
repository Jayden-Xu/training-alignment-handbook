"""Train the small LM for a few steps: the workload of the pretraining example.

One process is one run. The planted divergences are plain options, so a
reference and a suspect differ in one argument and run the same source lines:

    --norm-mul act|fp32     where the norms round (model.py)
    --linear own|nn         which Linear launches the GEMMs (model.py)
    --det strict|warn|off   torch.use_deterministic_algorithms(True, warn_only=False),
                            (True, warn_only=True), or (False)
    --resume CKPT --resume-skip PIECE
                            restore a checkpoint, leaving one piece of state out:
                            opt-step, sched, cpu-rng, cuda-rng or data-pos

It writes what the example compares afterwards: the final parameters
(--state-out, compared bit for bit), the per-step losses and a determinism
preflight report (--out).

    python train_lm.py --steps 4 --out run.json --state-out run.pt
    python train_lm.py --steps 2 --save ck.pt                      # phase 1
    python train_lm.py --steps 4 --resume ck.pt --out resumed.json  # phase 2
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from model import TinyLM  # noqa: E402

RESUME_PIECES = ("opt-step", "sched", "cpu-rng", "cuda-rng", "data-pos")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=4, help="train until this many steps are done")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--corpus-seqs", type=int, default=256, help="sequences in the synthetic corpus")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--adam", choices=("foreach", "fused", "single"), default="foreach")
    ap.add_argument("--norm-mul", choices=("act", "fp32"), default="act")
    ap.add_argument("--linear", choices=("own", "nn"), default="own")
    ap.add_argument("--det", choices=("strict", "warn", "off"), default="strict")
    ap.add_argument("--save", help="write a checkpoint here when training ends")
    ap.add_argument("--resume", help="restore this checkpoint before training")
    ap.add_argument("--resume-skip", choices=("none",) + RESUME_PIECES, default="none",
                    help="leave one piece of state out of the restore (a planted fault)")
    ap.add_argument("--out", help="write losses, settings and the preflight report here (JSON)")
    ap.add_argument("--state-out", help="write the final parameters here (.pt)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def set_determinism(mode: str) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if mode == "strict":
        torch.use_deterministic_algorithms(True, warn_only=False)
    elif mode == "warn":
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def make_corpus(seed: int, n_seq: int, length: int, vocab: int) -> torch.Tensor:
    """Sequences that walk through the vocabulary in steps of 1 to 4: learnable."""
    gen = torch.Generator().manual_seed(seed)
    start = torch.randint(0, vocab, (n_seq, 1), generator=gen)
    steps = torch.randint(1, 5, (n_seq, length), generator=gen)
    return (start + torch.cumsum(steps, 1)) % vocab


class Batches:
    """The loader as a stream of batches whose position can be saved and restored."""

    def __init__(self, loader: DataLoader):
        self.loader, self.gen = loader, loader.generator
        self.epoch_state, self.pos, self.it = None, 0, None

    def _start_epoch(self) -> None:
        self.epoch_state, self.pos = self.gen.get_state(), 0
        self.it = iter(self.loader)

    def next(self) -> torch.Tensor:
        if self.it is None:
            self._start_epoch()
        try:
            (batch,) = next(self.it)
        except StopIteration:
            self._start_epoch()
            (batch,) = next(self.it)
        self.pos += 1
        return batch

    def state(self) -> dict:
        if self.it is None:
            self._start_epoch()
        return {"epoch_state": self.epoch_state, "pos": self.pos}

    def restore(self, state: dict) -> None:
        self.gen.set_state(state["epoch_state"])
        self._start_epoch()
        for _ in range(state["pos"]):
            next(self.it)
        self.pos = state["pos"]


def grad_norms(model: torch.nn.Module) -> dict:
    """Per-parameter gradient norms, as a training loop logs them."""
    norms = {}
    for name, param in model.named_parameters():
        norms[name] = torch.linalg.vector_norm(param.grad)
    return norms


def save_checkpoint(path, step, model, opt, sched, batches, dev) -> None:
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "cpu_rng": torch.get_rng_state(),
                "py_rng": random.getstate(), "np_rng": np.random.get_state(), "cuda_rng": torch.cuda.get_rng_state(dev),
                "data": batches.state()}, path)


def restore_checkpoint(path, model, opt, sched, batches, dev, skip: str) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    if skip == "opt-step":          # moments come back, the step counter does not
        for state in opt.state.values():
            state["step"] = torch.zeros_like(state["step"])
    if skip != "sched":
        sched.load_state_dict(ck["sched"])
    if skip != "cpu-rng":
        torch.set_rng_state(ck["cpu_rng"])
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
    if skip != "cuda-rng":
        torch.cuda.set_rng_state(ck["cuda_rng"], dev)
    if skip != "data-pos":
        batches.restore(ck["data"])
    return int(ck["step"])


def preflight_report(**objects):
    """The determinism preflight of this handbook, when its tools are on the path."""
    try:
        import determinism_preflight
    except ImportError:
        return None
    return determinism_preflight.report(doctor=False, **objects)


def run(args) -> dict:
    """Train and return {'losses', 'state', 'preflight', ...}. `state` holds CPU tensors."""
    set_determinism(args.det)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    corpus = make_corpus(args.seed, args.corpus_seqs, args.seq_len + 1, args.vocab)
    loader = DataLoader(TensorDataset(corpus), batch_size=args.batch, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed + 1), num_workers=0)
    batches = Batches(loader)

    model = TinyLM(args.vocab, args.dim, args.layers, args.heads, args.seq_len,
                   args.dropout, args.norm_mul, args.linear).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
                            foreach=args.adam == "foreach", fused=args.adam == "fused")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    start = 0
    if args.resume:
        start = restore_checkpoint(args.resume, model, opt, sched, batches, dev, args.resume_skip)
    preflight = preflight_report(optimizer=opt, dataloader=loader, scheduler=sched, autocast=amp,
                                 model=model)

    losses, last_norms, t0 = [], {}, time.time()
    model.train()
    for step in range(start, args.steps):
        batch = batches.next().to(dev)
        tokens = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        with amp:
            logits = model(tokens)
        loss = F.cross_entropy(logits.float().view(-1, args.vocab), targets.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        last_norms = grad_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        losses.append(loss.detach())
        print(f"step {step + 1} done", flush=True)
    losses = [float(value) for value in losses]
    seconds = time.time() - t0
    for number, value in enumerate(losses, start + 1):
        print(f"step {number} loss {value:.6f}")

    if args.save:
        save_checkpoint(args.save, args.steps, model, opt, sched, batches, dev)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {"args": vars(args), "first_step": start + 1, "last_step": args.steps, "losses": losses,
            "loss_bits": [struct.pack("<f", x).hex() for x in losses],
            "param_names": [n for n, _ in model.named_parameters()],
            "grad_norms_last_step": {k: float(v) for k, v in last_norms.items()},
            "train_seconds": round(seconds, 3), "preflight": preflight, "state": state}


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run(args)
    state = result.pop("state")
    if args.state_out:
        torch.save(state, args.state_out)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
