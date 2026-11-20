# Data alignment

## Where we are

This page continues step 2 of the [pre-training workflow](README.md). The
[preflight](determinism-and-preflight.md) has read back the settings of both
runs, and only the declared difference is left. What you do not know yet is
whether the two runs train on the same data. Two runs can only be compared bit
for bit if every rank sees the same token bits at every step. If they do not,
every later comparison shows you the data difference, not the cause you are
looking for. So this page checks the batches themselves, at the point where
they leave the loader, before the pair is captured
([Localize with OpGuard](localize-with-opguard.md)).

## The idea: the same seed can still give different batches

Here is a small example. There are eight documents, A to H, a batch size of 4
and the same seed in both runs. Run 1 reads the documents in one process. Run
2 uses two worker processes, and each worker takes every other document:

| Step | Run 1 (no workers) | Run 2 (two workers) |
|---|---|---|
| 0 | A B C D | A C E G |
| 1 | E F G H | B D F H |

Both runs see every document exactly once in the epoch, so a check of the
epoch's contents passes. But at step 0 each run sums the gradients of
different samples, so the weights already differ after the first update. The
seed fixed the shuffle; it did not fix which process delivered which sample.

The same thing can happen in five places between the files and the model:

| Where | What changes | A small example |
|---|---|---|
| Order | the step at which a sample arrives | the worker split above; a shuffle buffer seeded from the process-wide random state |
| Partition | the rank that receives a sample | the same global batch split over 2 ranks or over 4 |
| Packing | how documents are laid out in rows | 6 documents in 6 padded rows, or packed into 3 rows |
| Tokenizer | the token ids a document turns into | a normalizer that lowercases the text; an added begin-of-sequence token |
| State | the point where a resumed run starts | a restart that does not restore the sampler's position and so replays the first steps |

None of these is a numerical effect. They change the input itself.

**Rule.** Fingerprint every batch where it leaves the loader, for each step
and each rank, and compare the two runs bit for bit before you look at the
model. The loader's exit is the first boundary in call order, so a difference
there is the first divergence. This applies part 0's controlled experiment:
the input is part of what must be held equal.

A single fingerprint of the whole batch tells you that something changed, but
not what. So the records keep three levels: one fingerprint for each whole
field (did anything change?), one for each sample (did samples move, or did
their contents change?), and one for each document inside a row (are the same
documents packed differently?). An XOR fingerprint cannot see two tokens that
swapped places inside a row, so the per-sample value also mixes each token's
position into its fingerprint.

## The step

### 2c. Pin the data source and the tokenizer

**Goal:** both runs read the same files and turn them into the same token ids.

**How:** hash the data files and the tokenizer, together with the arguments
the tokenizer is called with, and store both hashes in the header of the
fingerprint records (`source_fingerprint` and `tokenizer_fingerprint` in
[data_fingerprint.py](../../tools/data_fingerprint.py)). If you tokenize
offline, as Megatron-LM's preprocessing does, the tokenizer's output is fixed
in the files themselves.

**Success:** the two headers agree, or differ only in the declared difference.

### 2d. Fingerprint each batch where it leaves the loader

**Goal:** a record of what each rank consumed at each step.

**How:** add two lines to the training loop, on every rank, and apply them to
the batch exactly as the loader returns it:

```python
import data_fingerprint as dfp

rec = dfp.Recorder("fp/run_a", rank=dp_rank,
                   meta={"tokenizer": dfp.tokenizer_fingerprint(tok, add_special_tokens=True),
                         "source": dfp.source_fingerprint(data_files)})
for step, batch in enumerate(loader):
    rec.record(step, batch)
    ...
```

Every tensor field is recorded: the token ids, the attention mask, the
position ids, and the labels or the loss mask. With `opguard=`, the recorder
also writes its records in OpGuard's capture layout, so OpGuard's `first_diff`
sees the data boundary before the model's boundaries
([Localize with OpGuard](localize-with-opguard.md)). A resumed run passes
`append=True`.

**Success:** both runs have one record for each step and each rank.

### 2e. Compare the two runs

**Goal:** find the first (step, rank, field) whose bits differ.

**How:**

```bash
python tools/data_fingerprint.py compare fp/run_a fp/run_b
python tools/data_fingerprint.py compare fp/run_a fp/run_b --global
```

The comparison is exact, with no tolerance. The second form first
concatenates the ranks of each step, which is what you need when the two runs
use different data-parallel sizes.

**Success:** no record differs. The data is then aligned, and the pair is
ready to capture ([Localize with OpGuard](localize-with-opguard.md)).

### 2f. Find out what kind of difference it is

**Goal:** name the kind of difference, because each kind has its own fix.

**How:** the compare command assigns a class to the first difference and
counts the classes over all steps:

| Class | What it means | Typical cause |
|---|---|---|
| order | the same samples, but in another order inside the batch, or at another step or rank | a streaming dataset split across workers; a different data-parallel size; the seed of a shuffle buffer; a resume that replays, repeating an earlier step of its own run |
| samples | some documents appear on one side only | the tokenizer or the data source; a shuffle with a different seed; the tail end of a resume that replays |
| packing | the same documents at each step, but laid out in different rows | padding on one side and packing on the other; a different row length |
| masks | the same token ids, but different masks, positions or labels | position ids that are not reset for each document; a changed loss mask |

A difference in the header is reported first. A changed tokenizer shows up
there, unless only its call arguments changed and they were not recorded.

**Success:** you have one class and one named cause.

### 2g. Fix the difference

**Goal:** the fingerprints are equal at every step and on every rank.

**How:** choose the fix by the problem. The libraries named in the table solve
the problem by design, but each one does so only under its own conditions
([survey](../../examples/data_alignment/REPORT.md#existing-work)).

| Problem | Option | What it gives | Cost class | When to choose it |
|---|---|---|---|---|
| The number of workers changes the order | use a map-style dataset with the sampler in the main process; or let each worker build whole batches and take them in turn (Grain reads its workers in strict round robin); MosaicML Streaming's order does not depend on the workers | the same batch at the same step, for any number of workers | free | a streaming dataset whose samples are split across workers |
| The data-parallel size changes the partition | build one global batch per step and give each rank a contiguous slice of it (Megatron-LM's pretraining sampler, OLMo-core's global indices); or use MosaicML Streaming with a fixed canonical node count | with the ranks merged, the batch of each step is identical at any size | free | comparing runs at different DP sizes |
| A resume does not restore the position | save the number of consumed samples, or the loader's state, with the checkpoint (torchdata's StatefulDataLoader, Grain, Megatron-LM's consumed samples, Megatron-Energon, Streaming, OLMo-core) | the resumed run continues where it stopped | free | every resumed run |
| The shuffle buffer is seeded from the process | seed it from the seed, the epoch and the worker id, or give the loader its own generator; Hugging Face datasets takes a seed and an epoch | an order that does not depend on other random draws | free | any streaming shuffle |
| The packing differs | use the same collate function, row length and packing rule on both sides | the same rows | free | packing on one side only, or a changed packer |
| The tokenizer differs | pin the tokenizer file and its call arguments, or tokenize offline | the same token ids | free | a tokenizer upgrade, or a new special token |

**Success:** compare again, and no record differs.

## What can go wrong

!!! warning "Comparing rank by rank across data-parallel sizes"
    At another DP size, each rank holds a different slice of the batch, so a
    rank-by-rank comparison always differs. You end up chasing a difference
    that exists by design. Compare with the ranks of each step merged.

!!! warning "Restoring the sampler's own position after prefetching"
    Workers prefetch, so the sampler has handed out more indices than
    training has consumed. Restoring that position skips batches. Save the
    count of consumed samples that the training loop keeps, or use a loader
    that tracks it (measured: [report](../../examples/data_alignment/REPORT.md#cases)).

!!! warning "A tokenizer fingerprint without the call arguments"
    The tokenizer's file does not contain call arguments such as whether to
    add special tokens. The headers then agree while the batches differ.
    Record the call arguments together with the tokenizer.

!!! warning "A shuffle seed taken from the process"
    A buffer seeded from the process's random state shifts whenever anything
    else draws a random number first, and this only happens when workers are
    used. A single extra initialized parameter then changes the data. Seed the
    buffer explicitly.

!!! warning "Equal over the epoch does not mean equal at each step"
    A check of the epoch's contents passes when samples have only moved
    between steps, yet the weights differ from the first step on. Compare step
    by step.

!!! warning "Fingerprints taken after the batch has changed"
    If you record a batch after a cast, after a move to the device that adds
    padding, or after the model's own masking, loader differences and model
    differences get mixed up, and you cannot tell which side caused the
    difference. Record the batch exactly as the loader returns it.

## Summary

- Two runs can be compared only if every rank sees the same token bits at
  every step, and a shared seed does not guarantee that.
- Order, partition, packing, the tokenizer and the resume state can each
  change the batches without any numerical effect.
- Fingerprint every field where the batch leaves the loader, for each step and
  rank, and compare exactly; merge the ranks when the DP sizes differ.
- Read the class of the first difference, apply the matching fix, and compare
  again until no record differs.

This was shown on CPU, with ranks simulated in one process. It has not yet
been shown on a real distributed job, or with the libraries in the survey
([report](../../examples/data_alignment/REPORT.md#limits)).

## Go deeper

- [data_fingerprint.py](../../tools/data_fingerprint.py) is the tool; it
  defines the record format and the classifier.
- [Report: the survey of existing loaders, every planted case and its fix, the OpGuard check, and the cost](../../examples/data_alignment/REPORT.md).
- [data_alignment.py](../../examples/data_alignment/data_alignment.py)
  contains the planted cases.
- [Determinism and preflight](determinism-and-preflight.md) covers the
  settings and the state that are read back before this check.
- [Data parallel](data-parallel.md) explains why a sum over the same samples
  in another order is bit-for-bit equal only with a reduction that does not
  depend on the order.
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md).
