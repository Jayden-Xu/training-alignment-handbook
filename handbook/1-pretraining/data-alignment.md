# Data alignment

## Where we are

This continues step 2 of the [pre-training workflow](README.md). The
[preflight](determinism-and-preflight.md) has read back the settings of both
runs and only the declared difference is left. What is still open: whether
the two runs train on the same data. Two runs can only be compared bitwise if
every rank sees the same token bits at every step. If they do not, every
later comparison shows the data, not the cause you are looking for. This page
checks the batches themselves, where they leave the loader, before the pair is
captured ([Localize with OpGuard](localize-with-opguard.md)).

## The idea: same seed, different batches

A tiny example. Eight documents, A to H, a batch of 4, the same seed. Run 1
reads them in one process. Run 2 uses two worker processes, and each worker
takes every other document:

| Step | Run 1 (no workers) | Run 2 (two workers) |
|---|---|---|
| 0 | A B C D | A C E G |
| 1 | E F G H | B D F H |

Both runs see every document once in the epoch, so a check of the epoch's
contents passes. But step 0 sums the gradient of other samples, and the
weights part at the first update. The seed fixed the shuffle; it did not fix
which process delivered which sample.

The same happens in five places between the files and the model:

| Where | What changes | Tiny example |
|---|---|---|
| Order | which step a sample lands in | the worker split above; a shuffle buffer seeded from a process-wide random state |
| Partition | which rank a sample lands in | the same global batch split over 2 ranks or over 4 |
| Packing | how documents become rows | 6 documents in 6 padded rows, or packed into 3 rows |
| Tokenizer | which token ids a document becomes | a lowercasing normalizer; a begin-of-sequence token added |
| State | where a resumed run starts | a restart that does not restore the sampler's position replays the first steps |

None of these is a numerical effect. They change the input itself.

**Rule.** Fingerprint every batch where it leaves the loader, per step and
rank, and compare the two runs bitwise before looking at the model. The
loader's exit is the first boundary in call order, so a difference there is
the first divergence. This instances part 0's controlled experiment: the
input is part of what must be held equal.

A fingerprint of the whole batch is not enough to say what went wrong. Three
levels are kept: the whole field (did anything change), one per sample (did
the samples move or change), and one per document inside a row (are the same
documents packed differently). An XOR fingerprint does not see two tokens
swapped inside a row, so the per-sample value also keys every token by its
position.

## The step

### 2c. Pin the source and the tokenizer

**Goal:** both runs read the same files and turn them into the same token ids.

**How:** hash the data files and the tokenizer, including the arguments it is
called with, and store both in the header of the fingerprint records
(`source_fingerprint`, `tokenizer_fingerprint` in
[data_fingerprint.py](../../tools/data_fingerprint.py)). Tokenizing offline,
as Megatron-LM's preprocessing does, moves the tokenizer into the files.

**Success:** the two headers agree, or differ only in the declared difference.

### 2d. Fingerprint at the loader's exit

**Goal:** a record of what each rank consumed at each step.

**How:** two lines in the training loop, on every rank, on the batch as the
loader returns it:

```python
import data_fingerprint as dfp

rec = dfp.Recorder("fp/run_a", rank=dp_rank,
                   meta={"tokenizer": dfp.tokenizer_fingerprint(tok, add_special_tokens=True),
                         "source": dfp.source_fingerprint(data_files)})
for step, batch in enumerate(loader):
    rec.record(step, batch)
    ...
```

Every tensor field is recorded: token ids, attention mask, position ids,
labels or loss mask. With `opguard=` the recorder also writes the records in
OpGuard's capture layout, so OpGuard's `first_diff` sees the data boundary
before the model's ([Localize with OpGuard](localize-with-opguard.md)). A
resumed run passes `append=True`.

**Success:** one record per step and rank on both runs.

### 2e. Compare

**Goal:** the first (step, rank, field) whose bits differ.

**How:**

```bash
python tools/data_fingerprint.py compare fp/run_a fp/run_b
python tools/data_fingerprint.py compare fp/run_a fp/run_b --global
```

The comparison is exact: no tolerance. The second form concatenates the ranks
of each step first, for runs at different data-parallel sizes.

**Success:** no record differs. Then the data is aligned and the pair is ready
to capture ([Localize with OpGuard](localize-with-opguard.md)).

### 2f. Classify

**Goal:** the kind of difference, because each has its own fix.

**How:** the compare command names a class for the first difference and
counts the classes over all steps:

| Class | What it means | Typical cause |
|---|---|---|
| order | the same samples, in another order inside the batch, or at another step or rank | worker split of a streaming dataset; data-parallel size; a shuffle buffer's seed; a resume that replays (it repeats an earlier step of its own run) |
| samples | some documents exist on one side only | tokenizer or data source; a shuffle with another seed; the tail of a replayed resume |
| packing | the same documents per step, laid out in other rows | padding against packing; another row length |
| masks | the same token ids, other masks, positions or labels | position ids not reset per document; a loss mask change |

The header difference comes first: a changed tokenizer shows there, unless
only its call arguments changed and they were not recorded.

**Success:** one class and one named cause.

### 2g. Fix

**Goal:** the fingerprints equal, at every step and rank.

**How:** pick the fix by problem. The libraries named are the ones that solve
it by design; each holds only under its own conditions
([survey](../../examples/data_alignment/REPORT.md#existing-work)).

| Problem | Option | What it gives | Cost class | When to choose it |
|---|---|---|---|---|
| Worker count changes the order | a map-style dataset with the sampler in the main process; or each worker builds whole batches, taken in turn (Grain reads workers in strict round robin); MosaicML Streaming's order does not depend on workers | the same batch at the same step for any worker count | free | a streaming dataset split by sample across workers |
| Data-parallel size changes the partition | one global batch per step, each rank a contiguous slice (Megatron-LM's pretraining sampler, OLMo-core's global indices); MosaicML Streaming with a fixed canonical node count | the step's batch, ranks merged, identical at any size | free | comparing runs at different DP sizes |
| Resume does not restore the position | save the consumed samples or the loader state with the checkpoint (torchdata's StatefulDataLoader, Grain, Megatron-LM's consumed samples, Megatron-Energon, Streaming, OLMo-core) | the resumed run continues where it stopped | free | every resumed run |
| Shuffle buffer seeded from the process | seed it from seed, epoch and worker id, or give the loader its own generator; Hugging Face datasets takes seed and epoch | the buffer's order independent of other random draws | free | any streaming shuffle |
| Packing differs | the same collate, row length and packing rule on both sides | the same rows | free | packing on one side only, or a changed packer |
| Tokenizer differs | pin the tokenizer file and its call arguments, or tokenize offline | the same token ids | free | a tokenizer upgrade, a new special token |

**Success:** compare again: no record differs.

## What can go wrong

!!! warning "Per-rank comparison across data-parallel sizes"
    At another DP size a rank holds another slice of the batch, so a per-rank
    comparison always differs. You chase a difference that is by design.
    Compare with the ranks of each step merged.

!!! warning "The sampler's own position after prefetch"
    Workers prefetch, so the sampler has handed out more indices than training
    consumed. Restoring that position skips batches. Save the count of
    consumed samples kept by the training loop, or use a loader that tracks
    it (measured: [report](../../examples/data_alignment/REPORT.md#cases)).

!!! warning "A tokenizer fingerprint without its call arguments"
    The tokenizer's file does not hold call arguments such as adding special
    tokens. The header then agrees while the batches differ. Record the call
    arguments with the tokenizer.

!!! warning "A shuffle seed taken from the process"
    A buffer seeded from the process's random state moves when anything else
    draws a random number first, and only when workers are used. One extra
    initialized parameter changes the data. Seed it explicitly.

!!! warning "Equal over the epoch is not equal per step"
    A check of the epoch's contents passes when samples only moved between
    steps. The weights still differ from the first step. Compare per step.

!!! warning "Fingerprints taken after the batch changed"
    A batch recorded after a cast, a device move with padding, or the
    model's own masking mixes loader and model differences. You cannot tell
    which side caused it. Record the batch as the loader returns it.

## Summary

- Two runs are comparable only if every rank sees the same token bits at
  every step; a shared seed does not guarantee it.
- Order, partition, packing, tokenizer and resume state each change the
  batches without any numerical effect.
- Fingerprint every field at the loader's exit, per step and rank, and
  compare exactly; merge the ranks when the DP sizes differ.
- Read the class of the first difference, apply the matching fix, and compare
  again until no record differs.

Shown on CPU with ranks simulated in one process; not yet shown on a real
distributed job or with the surveyed libraries
([report](../../examples/data_alignment/REPORT.md#limits)).

## Go deeper

- [data_fingerprint.py](../../tools/data_fingerprint.py): the tool, its
  record format and the classifier.
- [Report: the survey of existing loaders, every planted case and its fix, the OpGuard check, the cost](../../examples/data_alignment/REPORT.md).
- [data_alignment.py](../../examples/data_alignment/data_alignment.py): the
  planted cases.
- [Determinism and preflight](determinism-and-preflight.md): the settings and
  state read back before this check.
- [Data parallel](data-parallel.md): a sum over the same samples in another
  order is bitwise equal only with an order-independent reduction.
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md).
