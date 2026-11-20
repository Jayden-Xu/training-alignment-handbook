"""data_alignment: planted data-loader differences, each found and classified by
tools/data_fingerprint.py, and the fix that makes the fingerprints equal. CPU only.

A synthetic corpus (seeded, written to a JSONL file) is tokenized with a tiny
word-level tokenizer built in-process with the ``tokenizers`` library (no
download). Each case records two runs with ``data_fingerprint.Recorder`` at the
loader's exit and compares them with ``data_fingerprint.compare``.

Cases
 0. selftest        fingerprints against the documented format; the classifier on planted pairs
 1. identical       the same loader built twice; a map-style dataset at 0 and 2 workers: equal
 2. workers         IterableDataset sharded by sample across workers, 0 vs 2 workers: order
                    fix: shard whole batches across workers (batch_size=None): equal at 0, 2, 3 workers
 3. dp_size         DistributedSampler at DP 2 (4 per rank) vs DP 4 (2 per rank), same seed and
                    global batch: per rank, order across ranks; per step (ranks merged), order
                    inside the batch. fix: a contiguous-slice sampler: per step equal
 4. resume          restart after 4 steps without the sampler state: order across steps (a replay)
                    fix: save and restore the sampler position: equal
 5. shuffle_buffer  a shuffle buffer seeded from torch.initial_seed() inside a worker, and one extra
                    torch.randn before the loader is built: order across steps
                    fix: seed the buffer from (seed, epoch, worker), or give the loader a generator: equal
 6. packing         the same documents per step padded vs packed: packing; packed positions reset
                    vs not reset, and the EOS label masked: masks. fix: the same collate: equal
 7. tokenizer       lowercase normalizer; BOS from the post-processor; add_special_tokens=False at the
                    call: samples, and the header names the tokenizer (the call argument only when it
                    is recorded). fix: the same definition and call: equal
 8. opguard         (--opguard-root) the resume pair as OpGuard capture records: the harness's
                    first_diff names data.input_ids at the step the tool names; the fixed pair has none
 9. cost            ms per recorded batch, and per compared record

    python data_alignment.py --quick --json results-quick.json
    python data_alignment.py --opguard-root '~/OpGuard' --opguard-python '~/env/bin/python' --json results-cpu.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler, IterableDataset, RandomSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import data_fingerprint as dfp  # noqa: E402
import opguard_bridge as ob  # noqa: E402
from bi_report import Results  # noqa: E402

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


# ----------------------------------------------------------------------------- corpus and tokenizer
def make_corpus(path: Path, n_docs: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    syl = ["ka", "ri", "to", "me", "su", "no", "la", "vi", "po", "de", "an", "el", "or", "us", "im"]
    words = sorted({"".join(rng.choice(syl) for _ in range(rng.randint(1, 3))) for _ in range(400)})
    docs = []
    for _ in range(n_docs):
        ws = [rng.choice(words) for _ in range(rng.randint(3, 24))]
        ws = [w.capitalize() if (i == 0 or rng.random() < 0.15) else w for i, w in enumerate(ws)]
        docs.append(" ".join(ws) + " .")
    path.write_text("".join(json.dumps({"text": d}) + "\n" for d in docs))
    return docs


def make_tokenizer(docs, *, lowercase=False, bos=False):
    from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
    from tokenizers.models import WordLevel
    words = sorted({w for d in docs for w in d.split()} | {w.lower() for d in docs for w in d.split()})
    vocab = {s: i for i, s in enumerate(SPECIALS)}
    vocab.update({w: i + len(SPECIALS) for i, w in enumerate(words)})
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if lowercase:
        tok.normalizer = normalizers.Lowercase()
    single = ("[BOS] " if bos else "") + "$A [EOS]"
    tok.post_processor = processors.TemplateProcessing(single=single, special_tokens=[("[BOS]", BOS), ("[EOS]", EOS)])
    return tok


class Docs(Dataset):
    """Map-style: item i is document i's token ids."""

    def __init__(self, docs, tok, add_special_tokens=True):
        self.ids = [torch.tensor(tok.encode(d, add_special_tokens=add_special_tokens).ids, dtype=torch.int64)
                    for d in docs]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i]


def pad_collate(items, T=32, mask_eos_label=False):
    ids = torch.full((len(items), T), PAD, dtype=torch.int64)
    mask = torch.zeros_like(ids)
    for i, x in enumerate(items):
        ids[i, :len(x)], mask[i, :len(x)] = x[:T], 1
    labels = ids.masked_fill(mask == 0, -100)
    if mask_eos_label:
        labels = labels.masked_fill(ids == EOS, -100)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def pack_collate(items, T=32, reset_positions=True):
    rows, pos = [[]], [[]]
    for x in items:
        x = x.tolist()
        if rows[-1] and len(rows[-1]) + len(x) > T:
            rows.append([])
            pos.append([])
        start = 0 if reset_positions else len(rows[-1])
        rows[-1] += x
        pos[-1] += list(range(start, start + len(x)))
    used = torch.tensor([len(r) for r in rows])
    ids = torch.tensor([r + [PAD] * (T - len(r)) for r in rows], dtype=torch.int64)
    mask = (torch.arange(T)[None, :] < used[:, None]).long()
    positions = torch.tensor([p + [0] * (T - len(p)) for p in pos], dtype=torch.int64)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
            "labels": ids.masked_fill(mask == 0, -100)}


# ----------------------------------------------------------------------------- loaders
class BySample(IterableDataset):
    """The common worker split: worker w yields samples i with i % workers == w."""

    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        w, n = (info.id, info.num_workers) if info else (0, 1)
        for i in range(w, len(self.ds), n):
            yield self.ds[i]


class ByBatch(IterableDataset):
    """Fix: worker w builds whole batches b with b % workers == w; the loader takes
    workers in turn, so batch b arrives at step b for any worker count."""

    def __init__(self, ds, batch):
        self.ds, self.batch = ds, batch

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        w, n = (info.id, info.num_workers) if info else (0, 1)
        for b in range(w, len(self.ds) // self.batch, n):
            yield pad_collate([self.ds[i] for i in range(b * self.batch, (b + 1) * self.batch)])


class Buffered(IterableDataset):
    """A streaming shuffle buffer (size ``size``) over a document stream."""

    def __init__(self, ds, size, seed_fn):
        self.ds, self.size, self.seed_fn = ds, size, seed_fn

    def __iter__(self):
        rng = random.Random(self.seed_fn())
        buf = []
        for i in range(len(self.ds)):
            buf.append(self.ds[i])
            if len(buf) == self.size:
                yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf


class ContiguousSampler:
    """Fix for a DP-size change: step s takes global batch s of one seeded
    permutation; rank r takes the r-th contiguous slice of it. Ranks merged in
    rank order give the same global batch at every DP size."""

    def __init__(self, n, world, rank, global_batch, seed=0):
        self.n, self.world, self.rank, self.g, self.seed = n, world, rank, global_batch, seed

    def __iter__(self):
        perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed)).tolist()
        per = self.g // self.world
        for s in range(self.n // self.g):
            glob = perm[s * self.g:(s + 1) * self.g]
            yield glob[self.rank * per:(self.rank + 1) * per]

    def __len__(self):
        return self.n // self.g


class ResumableSampler:
    """A seeded random permutation with a position that is saved and restored."""

    def __init__(self, n, seed=0):
        self.n, self.seed, self.epoch, self.pos = n, seed, 0, 0

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, sd):
        self.seed, self.epoch, self.pos = sd["seed"], sd["epoch"], sd["pos"]

    def __iter__(self):
        perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        while self.pos < self.n:
            self.pos += 1
            yield perm[self.pos - 1]

    def __len__(self):
        return self.n


def record(loader, out, rank=0, steps=None, start=0, append=False, meta=None, opguard=None):
    """The two lines a training loop adds: a Recorder, and record(step, batch).
    ``steps`` stops after that many batches without fetching one more."""
    rec = dfp.Recorder(out, rank=rank, append=append, meta=meta, sep_id=EOS, opguard=opguard)
    batches = enumerate(loader, start=start)
    for step, batch in (batches if steps is None else itertools.islice(batches, steps)):
        rec.record(step, batch)
    rec.close()
    return out


def summary(res: dict) -> dict:
    f = res["first"]
    return {"equal": res["equal"], "records": res["records"], "classes": res["classes"],
            "meta_diff": [d["key"] for d in res["meta_diff"]],
            "first": None if f is None else {k: f[k] for k in ("step", "micro", "rank", "field", "class", "detail")}}


def cls(res):
    return (res["first"] or {}).get("class")


# ----------------------------------------------------------------------------- cases
def run(args):
    res = Results("data_alignment", quick=args.quick)
    work = Path(args.work).expanduser() if args.work else Path(tempfile.mkdtemp(prefix="data-align-"))
    marker = work / ".data-alignment-work"
    if work.exists() and any(work.iterdir()):
        if not marker.exists():
            raise SystemExit(f"--work {work} is not empty and was not made by this script")
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    marker.touch()
    n_docs = 64 if args.quick else 512
    corpus = work / "corpus.jsonl"
    docs = make_corpus(corpus, n_docs)
    tok = make_tokenizer(docs)
    ds = Docs(docs, tok)
    meta = {"source": dfp.source_fingerprint(corpus), "tokenizer": dfp.tokenizer_fingerprint(tok, add_special_tokens=True)}
    out = {}
    W = work

    def check(name, ok, *, level="L1", detail=None, expect="pass"):
        res.check(name, ok, level=level, impl="data_fingerprint", expect=expect, detail=detail)

    # 0. selftest
    st = dfp.selftest(verbose=False)
    check(f"selftest: {sum(c['passed'] for c in st)}/{len(st)} cases", all(c["passed"] for c in st),
          detail=[c for c in st if not c["passed"]])
    st_og = ob.selftest(verbose=False)
    check(f"opguard_bridge fingerprint selftest: {sum(c['passed'] for c in st_og)}/{len(st_og)}",
          all(c["passed"] for c in st_og))

    # 1. identical
    def mapped(workers, seed=0):
        return DataLoader(ds, batch_size=4, sampler=RandomSampler(ds, generator=torch.Generator().manual_seed(seed)),
                          num_workers=workers, collate_fn=pad_collate)
    a = record(mapped(0), W / "identical/a", meta=meta)
    b = record(mapped(0), W / "identical/b", meta=meta)
    c = record(mapped(2), W / "identical/c", meta=meta)
    r1, r2 = dfp.compare(a, b), dfp.compare(a, c)
    out["identical"] = {"rebuilt": summary(r1), "map_style_0_vs_2_workers": summary(r2)}
    check("identical: the same loader built twice is equal", r1["equal"], detail=summary(r1))
    check("identical: map-style dataset, 0 vs 2 workers, equal", r2["equal"], detail=summary(r2))

    # 2. workers
    a = record(DataLoader(BySample(ds), batch_size=4, num_workers=0, collate_fn=pad_collate), W / "workers/a", meta=meta)
    b = record(DataLoader(BySample(ds), batch_size=4, num_workers=2, collate_fn=pad_collate), W / "workers/b", meta=meta)
    r = dfp.compare(a, b)
    out["workers"] = {"planted": summary(r)}
    check("workers: sample-sharded IterableDataset, 0 vs 2 workers: detected, class order",
          not r["equal"] and cls(r) == "order", detail=summary(r))
    fixed = [record(DataLoader(ByBatch(ds, 4), batch_size=None, num_workers=w), W / f"workers/fix{w}", meta=meta)
             for w in (0, 2, 3)]
    rf = [dfp.compare(fixed[0], f) for f in fixed[1:]]
    rp = dfp.compare(a, fixed[0])
    out["workers"]["fix_0_vs_2"], out["workers"]["fix_0_vs_3"] = summary(rf[0]), summary(rf[1])
    out["workers"]["fix_equals_single_process"] = summary(rp)
    check("workers fix: batch-sharded IterableDataset, 0 vs 2 and 0 vs 3 workers, equal",
          all(x["equal"] for x in rf), detail=[summary(x) for x in rf])
    check("workers fix: equals the single-process order of the planted loader", rp["equal"], detail=summary(rp))

    # 3. dp_size
    G = 8

    def dist_run(world, tag, sampler):
        root = W / f"dp/{tag}{world}"
        for rank in range(world):
            if sampler == "distributed":
                s = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
                loader = DataLoader(ds, batch_size=G // world, sampler=s, collate_fn=pad_collate)
            else:
                loader = DataLoader(ds, batch_sampler=ContiguousSampler(len(ds), world, rank, G), collate_fn=pad_collate)
            record(loader, root, rank=rank, meta=meta)
        return root
    a, b = dist_run(2, "ds", "distributed"), dist_run(4, "ds", "distributed")
    per_rank, glob = dfp.compare(a, b), dfp.compare(a, b, global_=True)
    out["dp_size"] = {"distributed_per_rank": summary(per_rank), "distributed_global": summary(glob)}
    check("dp_size: DistributedSampler DP 2 vs DP 4, per rank: detected, class order (samples moved across ranks)",
          not per_rank["equal"] and cls(per_rank) == "order", level="L5", detail=summary(per_rank))
    check("dp_size: DistributedSampler DP 2 vs DP 4, per step (ranks merged): same samples, order inside the batch",
          cls(glob) == "order" and set(glob["classes"]) == {"order"}
          and (glob["first"]["detail"] or {}).get("scope") == "inside the batch", level="L5", detail=summary(glob))
    fa, fb = dist_run(2, "contig", "contiguous"), dist_run(4, "contig", "contiguous")
    fg, fr = dfp.compare(fa, fb, global_=True), dfp.compare(fa, fb)
    out["dp_size"]["contiguous_global"], out["dp_size"]["contiguous_per_rank"] = summary(fg), summary(fr)
    check("dp_size fix: contiguous-slice sampler, DP 2 vs DP 4, per step (ranks merged) equal",
          fg["equal"], level="L5", detail=summary(fg))
    check("dp_size fix: per rank still differs (a rank holds another slice at another DP size)",
          not fr["equal"], level="L5", expect="info", detail=summary(fr))

    # 4. resume
    steps, k = len(ds) // 4, 4

    def rnd_loader(sampler):
        return DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=pad_collate)
    a = record(rnd_loader(RandomSampler(ds, generator=torch.Generator().manual_seed(0))), W / "resume/a", meta=meta)
    b = W / "resume/b"
    record(rnd_loader(RandomSampler(ds, generator=torch.Generator().manual_seed(0))), b, steps=k, meta=meta)
    record(rnd_loader(RandomSampler(ds, generator=torch.Generator().manual_seed(0))), b, start=k,
           steps=steps - k, append=True, meta=meta)                          # restarted without the position
    r = dfp.compare(a, b)
    out["resume"] = {"planted": summary(r)}
    d = (r["first"] or {}).get("detail") or {}
    check("resume: restart without the sampler state: detected at the resume step, class order (replay)",
          cls(r) == "order" and r["first"]["step"] == k and d.get("suspect_repeats_its_own_steps") == [0],
          detail=summary(r))
    def rnd_workers(sampler):
        return DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=pad_collate, num_workers=2)
    fa = record(rnd_workers(ResumableSampler(len(ds))), W / "resume/fa", meta=meta)
    fb = W / "resume/fb"
    s1 = ResumableSampler(len(ds))
    record(rnd_workers(s1), fb, steps=k, meta=meta)
    ahead = s1.pos - 4 * k                              # indices the prefetching workers drew beyond training
    state = {"seed": s1.seed, "epoch": s1.epoch, "pos": 4 * k}                # consumed samples, from the loop
    s2 = ResumableSampler(len(ds))
    s2.load_state_dict(json.loads(json.dumps(state)))                         # through the checkpoint
    record(rnd_workers(s2), fb, start=k, steps=steps - k, append=True, meta=meta)
    rf = dfp.compare(fa, fb)
    out["resume"]["fix"] = summary(rf)
    out["resume"]["fix_state"] = state
    out["resume"]["sampler_position_ahead_with_2_workers"] = ahead
    check("resume fix: consumed samples saved with the checkpoint and restored (2 workers): equal",
          rf["equal"], detail=summary(rf))
    check(f"resume: the sampler's own position ran {ahead} samples ahead of training (2 workers prefetching)",
          ahead > 0, expect="info", detail={"ahead_samples": ahead, "consumed": 4 * k})

    # 5. shuffle_buffer
    def buffered(seed_fn, generator=None, extra_draw=False):
        torch.manual_seed(0)
        if extra_draw:
            torch.randn(3)                              # e.g. one more parameter initialized first
        return DataLoader(Buffered(ds, 16, seed_fn), batch_size=4, num_workers=1, collate_fn=pad_collate,
                          generator=generator)
    from_process = lambda: torch.initial_seed()  # noqa: E731

    def explicit(seed=0, epoch=0):
        info = torch.utils.data.get_worker_info()
        return (seed * 1_000_003 + epoch) * 1_009 + (info.id if info else 0)
    a = record(buffered(from_process), W / "buffer/a", meta=meta)
    b = record(buffered(from_process, extra_draw=True), W / "buffer/b", meta=meta)
    r = dfp.compare(a, b)
    out["shuffle_buffer"] = {"planted": summary(r)}
    check("shuffle buffer: seed from the process RNG, one extra draw: detected, class order",
          cls(r) == "order", detail=summary(r))
    fa = record(buffered(explicit), W / "buffer/fa", meta=meta)
    fb = record(buffered(explicit, extra_draw=True), W / "buffer/fb", meta=meta)
    ga = record(buffered(from_process, generator=torch.Generator().manual_seed(0)), W / "buffer/ga", meta=meta)
    gb = record(buffered(from_process, generator=torch.Generator().manual_seed(0), extra_draw=True),
                W / "buffer/gb", meta=meta)
    rf, rg = dfp.compare(fa, fb), dfp.compare(ga, gb)
    out["shuffle_buffer"]["fix_explicit_seed"], out["shuffle_buffer"]["fix_loader_generator"] = summary(rf), summary(rg)
    check("shuffle buffer fix: buffer seeded from (seed, epoch, worker): equal", rf["equal"], detail=summary(rf))
    check("shuffle buffer fix: loader given its own generator: equal", rg["equal"], detail=summary(rg))

    # 6. packing
    def seq(collate):
        return DataLoader(ds, batch_size=6, shuffle=False, collate_fn=collate, drop_last=True)
    a = record(seq(pad_collate), W / "pack/pad", meta=meta)
    b = record(seq(pack_collate), W / "pack/pack", meta=meta)
    c = record(seq(lambda x: pack_collate(x, reset_positions=False)), W / "pack/pack_noreset", meta=meta)
    e = record(seq(lambda x: pad_collate(x, mask_eos_label=True)), W / "pack/pad_eoslabel", meta=meta)
    f = record(seq(pack_collate), W / "pack/pack2", meta=meta)
    rp, rn, re_, rf = dfp.compare(a, b), dfp.compare(b, c), dfp.compare(a, e), dfp.compare(b, f)
    out["packing"] = {"pad_vs_pack": summary(rp), "positions_reset_vs_not": summary(rn),
                      "eos_label_masked": summary(re_), "fix_same_collate": summary(rf)}
    # A step whose documents each fill a row of their own has the same token rows in both
    # layouts; there only the field the packed side adds differs (class structure).
    check("packing: padded vs packed rows of the same documents: detected, class packing "
          "(structure where the rows coincide)",
          cls(rp) == "packing" and set(rp["classes"]) <= {"packing", "structure"}, detail=summary(rp))
    check("packing: positions reset per document vs not: detected, class masks (field position_ids)",
          cls(rn) == "masks" and rn["first"]["field"] == "position_ids", detail=summary(rn))
    check("packing: EOS label masked: detected, class masks (field labels)",
          cls(re_) == "masks" and re_["first"]["field"] == "labels", detail=summary(re_))
    check("packing fix: the same collate on both sides: equal", rf["equal"], detail=summary(rf))

    # 7. tokenizer
    def tok_run(tag, tk, add_special_tokens=True, record_call=True):
        m = {"source": meta["source"],
             "tokenizer": dfp.tokenizer_fingerprint(tk, **({"add_special_tokens": add_special_tokens} if record_call else {}))}
        return record(DataLoader(Docs(docs, tk, add_special_tokens), batch_size=4, shuffle=False, collate_fn=pad_collate),
                      W / f"tok/{tag}", meta=m)
    base = tok_run("base", tok)
    base_nocall = tok_run("base_nocall", tok, record_call=False)
    variants = {
        "lowercase normalizer": (tok_run("lower", make_tokenizer(docs, lowercase=True)), base, "samples", True),
        "BOS added by the post-processor": (tok_run("bos", make_tokenizer(docs, bos=True)), base, "samples", True),
        "add_special_tokens=False at the call, recorded": (tok_run("nospecial", tok, False), base, "samples", True),
        "add_special_tokens=False at the call, not recorded":
            (tok_run("nospecial_nocall", tok, False, record_call=False), base_nocall, "samples", False),
    }
    out["tokenizer"] = {}
    for name, (path, ref, want, header) in variants.items():
        r = dfp.compare(ref, path)
        out["tokenizer"][name] = summary(r)
        named = "tokenizer" in [m["key"] for m in r["meta_diff"]]
        check(f"tokenizer: {name}: batches differ, class {want}; header names the tokenizer: {header}",
              cls(r) == want and named == header, detail=summary(r))
    rf = dfp.compare(base, tok_run("base2", make_tokenizer(docs)))
    out["tokenizer"]["fix_same_definition_and_call"] = summary(rf)
    check("tokenizer fix: the same definition and call arguments (rebuilt): equal", rf["equal"], detail=summary(rf))

    # 8. opguard
    if args.opguard_root:
        try:
            og = ob.locate_opguard(str(Path(args.opguard_root).expanduser()),
                                   str(Path(args.opguard_python).expanduser()) if args.opguard_python else None)
        except ob.OpGuardError as exc:
            og = None
            out["opguard"] = {"error": str(exc)[:300]}
            check("opguard: OpGuard usable", False, expect="info")
        if og:
            ogw = W / "og"
            pairs = {}
            traces = {}
            for tag, (pa, pb) in {"planted": (W / "resume/a", W / "resume/b"),
                                  "fixed": (W / "resume/fa", W / "resume/fb")}.items():
                traces[tag] = (pa, ogw / f"{tag}_a", ogw / f"{tag}_b")
                dfp.export(pa, traces[tag][1])
                dfp.export(pb, traces[tag][2])
            # the same run recorded again with Recorder(opguard=...) while it runs
            record(rnd_workers(ResumableSampler(len(ds))), W / "resume/live", meta=meta, opguard=ogw / "live_b")
            traces["live_vs_export"] = (W / "resume/fa", ogw / "fixed_a", ogw / "live_b")
            for tag, (pa, ta, tb) in traces.items():
                t0 = time.perf_counter()
                exp = ob.export_pair(ta, tb, ogw / f"{tag}_export", og)
                facts = ob.first_divergence(exp["reports"][0], ogw, og, walk=False)
                pv = facts.get("pivot") or {}
                nf = len(dfp.load(pa)["records"][(0, 0, 0)]["fields"])
                pairs[tag] = {"diverged": facts["diverged"], "fn": pv.get("fn"), "counter": pv.get("counter"),
                              "step_from_counter": (pv["counter"] - 1) // nf if pv.get("counter") else None,
                              "diff_fields": pv.get("diff_fields"), "status_counts": facts["status_counts"],
                              "seconds": round(time.perf_counter() - t0, 1)}
            out["opguard"] = pairs
            tool_first = out["resume"]["planted"]["first"]
            p = pairs["planted"]
            check("opguard: harness first_diff on the planted resume pair names data.input_ids at the tool's step",
                  p["diverged"] and str(p["fn"]).strip("<>") == "data.input_ids"
                  and p["step_from_counter"] == tool_first["step"], detail=pairs)
            check("opguard: the fixed resume pair has no differing event", not pairs["fixed"]["diverged"], detail=pairs)
            check("opguard: Recorder(opguard=...) while running and export() afterwards: no differing event",
                  not pairs["live_vs_export"]["diverged"]
                  and pairs["live_vs_export"]["status_counts"].get("run1", {}).get("ok", 0) > 0, detail=pairs)

    # 9. cost
    B, T = (4, 512) if args.quick else (8, 4096)
    g = torch.Generator().manual_seed(0)
    batch = {"input_ids": torch.randint(4, 30000, (B, T), generator=g), "attention_mask": torch.ones(B, T, dtype=torch.int64),
             "position_ids": torch.arange(T).repeat(B, 1) % 512, "labels": torch.randint(4, 30000, (B, T), generator=g)}
    rec = dfp.Recorder(W / "cost", sep_id=EOS)
    rec.record(0, batch)
    reps = 5 if args.quick else 20
    t0 = time.perf_counter()
    for i in range(reps):
        rec.record(i + 1, batch)
    ms = (time.perf_counter() - t0) * 1000 / reps
    rec.close()
    res.measure(f"record one batch {B}x{T} int64, 4 fields, segments at every position reset", round(ms, 2), "ms",
                impl="data_fingerprint")
    t0 = time.perf_counter()
    rr = dfp.compare(W / "cost", W / "cost")
    res.measure("compare, per record", round((time.perf_counter() - t0) * 1000 / rr["records"]["equal"], 2), "ms",
                impl="data_fingerprint")
    res.measure("documents in the corpus", n_docs, "documents", impl="data_fingerprint")
    res.extra["cases"] = out
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--work", help="working directory, new or made by an earlier run (default: a new temporary "
                                   "one); removed at the end unless --keep")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--opguard-root", help="OpGuard checkout, for case 8")
    ap.add_argument("--opguard-python", help="interpreter that imports OpGuard (default: this one)")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    res = run(args)
    res.write(args.json)
    return res.exit_code()


if __name__ == "__main__":
    sys.exit(main())
