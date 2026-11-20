# Data bitwise alignment: lab report

This is the full record behind the handbook page [Data alignment](../../handbook/1-pretraining/data-alignment.md). The page teaches the idea and the step; this report keeps the survey of existing loaders, the setup, every planted case with its numbers, the cost and the limits.

## Summary

- [tools/data_fingerprint.py](../../tools/data_fingerprint.py) records, per step
  and rank, fingerprints of every field of the batch at the loader's exit, and
  compares two runs with exact equality. Every planted difference (6 kinds, 11
  variants) was detected and classified as expected, and each fix made the
  fingerprints equal: 27 of 27 judged checks pass, CPU only
  ([results-cpu.json](results-cpu.json)).
- A DistributedSampler run at DP 2 and at DP 4 (same seed, same global batch)
  gives each rank different samples, but the same set of samples per step, in
  a different order. A sampler that hands each rank a contiguous slice of one
  global batch makes the step's batch, ranks concatenated, bit-identical across
  DP sizes.
- A restart without the sampler state replays the first steps; the tool names
  the resume step and the reference step it repeats. With 2 workers the
  sampler's own position ran 4 batches ahead of training: the position to save
  is the count of consumed samples, not the sampler's.
- Written as OpGuard capture records, the planted resume pair gives OpGuard's
  harness `first_diff` pivot `data.input_ids` at the same step the tool names;
  the fixed pair has no differing event.
- None of the seven loaders surveyed checks that two runs consumed the same
  bits; each guarantees order or resumption under conditions of its own.

## Existing work

What each library promises, read from its official documentation or source on
2026-11-19. "Not confirmed" marks a property the documentation does not state.
None of them compares the bits two runs actually consumed.

| Library (version read) | What it guarantees | What it does not | Source |
|---|---|---|---|
| MosaicML Streaming (PyPI latest 0.13.0, 2025-07-15; docs "stable") | "Elastic determinism": the sample order is the same for any number of GPUs, nodes or CPU workers, provided `num_canonical_nodes` stays the same and the global batch size is constant and divisible by every device count; all randomness derives from `shuffle_seed`; mid-epoch resumption through `StreamingDataLoader.state_dict()` | Tokenization, collation and packing are the user's; the shuffle algorithm and `num_canonical_nodes` change the order (same seed, different order); no check of the consumed batches | [elastic determinism](https://docs.mosaicml.com/projects/streaming/en/stable/distributed_training/elastic_determinism.html), [shuffling](https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/shuffling.html), [fast resumption](https://docs.mosaicml.com/projects/streaming/en/stable/distributed_training/fast_resumption.html), [PyPI](https://pypi.org/project/mosaicml-streaming/) |
| Google Grain (PyPI 0.2.19, 2026-11-13; docs "latest") | "Multiple runs of the same pipeline will produce the same output"; after preemption it resumes "as if it was never preempted"; the iterator state is small (per-worker last-seen indices); workers are read in strict round robin | A checkpoint is valid only with the same sampler and data source; whether the order is the same at another `worker_count`, or a checkpoint restores at another `worker_count`: not confirmed | [overview](https://google-grain.readthedocs.io/en/latest/), [DataLoader guide](https://google-grain.readthedocs.io/en/latest/tutorials/data_loader_tutorial.html), [behind the scenes](https://google-grain.readthedocs.io/en/latest/behind_the_scenes.html), [API](https://google-grain.readthedocs.io/en/latest/grain.data_loader.html), [PyPI](https://pypi.org/project/grain/) |
| torchdata `StatefulDataLoader` (docs main 0.12.0a0; latest release in the README table 0.11.0) | `state_dict()` / `load_state_dict()` for mid-epoch resume, worker state aggregated; the default samplers are patched to be resumable | `load_state_dict` asserts the same `num_workers`; an IterableDataset without its own `state_dict` is fast-forwarded by replaying; ranks are not handled (each rank saves its own) | [tutorial](https://meta-pytorch.org/data/main/stateful_dataloader_tutorial.html), [source](https://github.com/pytorch/data/blob/main/torchdata/stateful_dataloader/stateful_dataloader.py) |
| TorchTitan (v0.2.0–v0.3.0: `ParallelAwareDataloader`, a `StatefulDataLoader`; from v0.4.0, 2026-10-28: a Grain-based `GrainDataLoader`) | Per-DP-rank loader state in the checkpoint | v0.3.0 asserts the DP degree is unchanged ("dataloader resharding is not supported yet"); main raises on a changed effective DP degree, and its data README states that resume "requires unchanged code, config, source contents, tokenizer, and effective DP degree" | [v0.3.0 dataloader](https://github.com/pytorch/torchtitan/blob/v0.3.0/torchtitan/components/dataloader.py), [v0.4.0 loader](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/components/data/loader.py), [data README](https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/data/README.md) |
| Megatron-LM `GPTDataset` and samplers (main) | Document, sample and shuffle indices built from the seed, built once on one rank and cached under `--data-cache-path`, keyed by a hash made in `MegatronDataset.__init__`; `MegatronPretrainingSampler` resumes from `consumed_samples` and gives rank r the r-th contiguous slice of each global micro-batch; the random sampler seeds by epoch | What the cache hash covers: not confirmed; tokenization is offline, so a tokenizer change reaches the run only through the data files; the random sampler with data sharding gives each rank a bucket that depends on the DP size | [datasets readme](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/readme.md), [data_samplers.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/datasets/data_samplers.py) |
| NVIDIA Megatron-Energon (docs, 2025) | "Deterministic save and restore" of the loader state of all ranks (`save_state_rank` / `save_state_global`); the DP group is given in `WorkerConfig` | Restoring at another rank or worker count: not documented (not confirmed) | [save and restore](https://nvidia.github.io/Megatron-Energon/basic/save_restore.html), [parallelism](https://nvidia.github.io/Megatron-Energon/advanced/parallelism.html) |
| AI2 OLMo / OLMo-core (OLMo-core docs v2.6.0) | "Distributed, deterministic, stateful" loaders: one global shuffled index array from seed and epoch, saved as `.npy`; global batch i is a fixed slice of it, split across DP ranks by stride; the state holds a dataset fingerprint, batches and tokens processed, seed and epoch. `dump_training_batch.py` (OLMo-core) and `inspect_train_data.py` (OLMo) rebuild the batch of a given step from a checkpoint | Rebuilds a step's batch from the configuration; does not record what each rank consumed, nor compare two runs. The dataset fingerprint covers the dataset, not the per-step bits | [data_loader source](https://olmo-core.readthedocs.io/en/latest/_modules/olmo_core/data/data_loader.html), [dump_training_batch.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/dump_training_batch.py), [OLMo scripts](https://github.com/allenai/OLMo/tree/main/scripts) |
| Hugging Face `datasets` (PyPI 5.2.0; docs main) | `IterableDataset.shuffle(seed, buffer_size)` shuffles the shard order and uses a buffer; `set_epoch` makes the seed seed + epoch; `state_dict()` / `load_state_dict()` resume at the exact example | Resuming is exact "except if `.shuffle()` is used": the buffer's examples are lost; with a DataLoader each worker streams its own subset of shards, so the order depends on `num_workers`; `split_dataset_by_node` splits by shards when `num_shards % world_size == 0`, else by example, so the split rule changes with the world size | [stream](https://huggingface.co/docs/datasets/main/en/stream), [use with PyTorch](https://huggingface.co/docs/datasets/main/en/use_with_pytorch), [PyPI](https://pypi.org/project/datasets/) |

## How

### The tool

`Recorder(dir, rank)` and `record(step, batch)` are the two lines a training
loop adds. Each call writes one JSON line per batch to `dir/rank00000.jsonl`
(a header line first, with `meta`). Per field (each tensor of the batch,
dim 0 is the sample):

| Recorded | What it is | Why |
|---|---|---|
| `shape`, `dtype` | as given | |
| `xor` | `flat-u32-le-v1` fingerprint of the whole field, one region, from `opguard_bridge.xor_signature` | OpGuard's format |
| `rows` | one XOR per sample, `opguard_bridge.xor_vectors` on the sample's bytes, zero-padded to 32-bit words | OpGuard's per-row values; they XOR to `xor` |
| `keys` | one 64-bit key per sample: each 8-byte word mixed with its index (splitmix64), then XORed | XOR does not see a permutation inside a sample |
| `segments` (token field only) | one key and length per document | the same documents in another packing |

A document starts at the first unmasked token of a row, where `position_ids`
returns to 0, after a masked gap, and after `sep_id`. Masked tokens are left
out. The selftest shows the blind spot the keys close: swapping two tokens of
a row leaves `rows` equal and changes `keys`.

`compare(reference, suspect)` pairs records by (step, micro-batch, rank) and
compares them with exact equality; `global_=True` first concatenates the ranks
of each step in rank order. For a differing pair it names the first differing
field (the token field first) and a class:

| Class | Rule, in order |
|---|---|
| structure | the token field is equal (or missing), and fields or dtypes differ |
| masks | the token field is equal, other fields differ |
| order | the token field's sample keys are a permutation (inside the batch); or every document of both records is found in the other run (across steps or ranks) |
| packing | the document multiset of the step is equal, the rows are not |
| samples | some document exists on one side only |
| missing | a record on one side only |

Header `meta` (here: a source fingerprint of the corpus file and a tokenizer
fingerprint with its call arguments) is compared first, pooled over ranks.

### Setup

One Linux host, CPU only (an Intel Xeon Silver 4310; CUDA hidden), Python
3.10.13, torch 2.5.1, numpy 2.1.3, tokenizers 0.21.0, 4 torch threads.
OpGuard: a source checkout used through `opguard_bridge` with its own
interpreter (torch 2.5.1).

The corpus is 512 synthetic documents (3–24 words of a 400-word syllable
vocabulary, some capitalized, then "."), seeded and written to a JSONL file.
The tokenizer is built in-process with the `tokenizers` library: word level,
whitespace split, vocabulary of every word and its lowercase form, `[PAD]=0,
[UNK]=1, [BOS]=2, [EOS]=3`, post-processor `$A [EOS]`. Rows are 32 tokens,
right-padded with an attention mask; labels are the ids with -100 at padding.
Quick mode uses 64 documents.

## Cases

All numbers from [results-cpu.json](results-cpu.json). "Records" counts
(step, rank) pairs, or steps with ranks merged.

| # | Case | Planted | Tool's first difference | Records by class | Fix | After the fix |
|---|---|---|---|---|---|---|
| 1 | identical | none: the loader built twice; a map-style dataset at 0 vs 2 workers | none | 128 equal; 128 equal | — | — |
| 2 | workers | `IterableDataset` that gives sample i to worker i mod W; 0 vs 2 workers, batch 4 | step 0, `input_ids`, order across steps (the suspect's documents are at reference step 1) | order 128 | each worker builds whole batches b with b mod W = w; loader with `batch_size=None` | equal at 0, 2 and 3 workers (128 of 128), and equal to the planted loader at 0 workers |
| 3 | dp_size | `DistributedSampler`, seed 0, DP 2 × 4 vs DP 4 × 2 (global batch 8) | per rank: step 0 rank 0, order, across ranks of the same step; ranks merged: step 0, order inside the batch, permutation [0, 2, 4, 6, 1, 3, 5, 7] | per rank: order 128, missing 128 (ranks 2–3); merged: order 64 | contiguous-slice sampler: step s takes global batch s of one permutation, rank r its r-th slice | merged: equal 64 of 64; per rank still differs (info) |
| 4 | resume | stop after 4 steps, restart with a fresh `RandomSampler` (same seed, no position) | step 4, order across steps, step offset −4, repeats its own step 0 | equal 4, order 120, samples 4 (the reference's last 4 steps never come) | save consumed samples (4 × 4 = 16) with the checkpoint, restart the sampler there (2 workers) | equal 128 of 128 |
| 5 | shuffle_buffer | buffer of 16 seeded from `torch.initial_seed()` in the worker (1 worker); the suspect draws `torch.randn(3)` before building the loader | step 0, order across steps (documents from reference steps 1, 4, 5, 9) | order 128 | seed the buffer from (seed, epoch, worker); or give the loader its own generator | equal 128 of 128, both fixes |
| 6a | packing | same 6 documents per step: padded (6 rows) vs packed (rows of 32, positions reset) | step 0, packing: 6 documents, shapes [6, 32] vs [3, 32]; `position_ids` only in the suspect | packing 83, structure 2 | the same collate on both sides | equal 85 of 85 |
| 6b | positions | packed, positions reset per document vs not | step 0, `position_ids`, masks | masks 83, equal 2 | same | — |
| 6c | loss mask | EOS label set to -100 | step 0, `labels`, masks | masks 85 | same | — |
| 7a | tokenizer | lowercase normalizer added | step 0, samples; hint "same document lengths, different token ids"; header: tokenizer | samples 128 | the same definition and call arguments | equal 128 of 128 |
| 7b | tokenizer | `[BOS]` added by the post-processor | samples; hint "every document is 1 token longer"; header: tokenizer | samples 128 | same | — |
| 7c | tokenizer | `add_special_tokens=False` at the call, recorded in the header | samples; hint "1 token shorter"; header: tokenizer | samples 128 | same | — |
| 7d | tokenizer | the same, call arguments not recorded | samples; hint "1 token shorter"; header: no difference | samples 128 | record the call arguments | — |

Notes on the cases:

- **2.** The same documents arrive in the same epoch, so over the epoch the set
  is unchanged; per step it is not, and the trained weights differ from the
  first step. A map-style dataset is not affected: its sampler runs in the main
  process (case 1).
- **3.** The per-step set is the same because `DistributedSampler` gives rank
  r the indices r, r + W, r + 2W, … of one permutation: with W × per-rank batch
  fixed, step s covers positions [8s, 8s + 8) at any W. Only the order inside
  the global batch and the rank assignment change. The gradient then sums the
  same samples in another order, which is bitwise the same only with a
  reduction that does not depend on the order (see the
  [data-parallel page](../../handbook/1-pretraining/data-parallel.md)).
  Per rank, the comparison can never be equal across DP sizes: a rank holds a
  different slice.
- **4.** The class is order because every replayed document exists in the
  reference, at another step; `suspect_repeats_its_own_steps: [0]` shows it is
  a replay. The last 4 steps are classed samples: the reference's documents of
  those steps never reach the suspect. With 2 workers the sampler's own
  position after 4 steps was 16 samples ahead of the 16 consumed (prefetch of
  4 batches); restoring that position would skip 4 batches.
- **5.** With workers, the loader draws a base seed from the global torch RNG
  when it creates the iterator, and the worker's `torch.initial_seed()` is that
  base seed plus the worker id. Any extra draw before the loop (one more
  parameter initialized) changes it. With 0 workers the planted bug does not
  show: `torch.initial_seed()` is then the main process's seed.
- **6a.** In 2 of 85 steps every document filled a row of its own, so both
  layouts give the same `input_ids`; there the only difference is the
  `position_ids` field the packed side adds (class structure).
- **7d.** The tokenizer's definition does not hold call arguments, so a
  fingerprint of the definition alone passes; the batches still differ.

## OpGuard

`data_fingerprint.export(run, root)` writes one OpGuard boundary record per
field and step (`data.<field>`, meta step and micro-batch). Output 0 carries
the field's shape, dtype and whole-field fingerprint; output 1 is the sample
keys mixed with their row index, so a change of order also differs. The
planted and the fixed resume pairs went through OpGuard's backend `compare`
and the harness (`trace_summary`, `first_diff`, `inspect_event`,
`compare_events`) via `opguard_bridge.export_pair` and `first_divergence`:

| Pair | Events per run | OpGuard's first difference | Tool's first difference |
|---|---|---|---|
| planted resume | 384 (128 steps × 3 fields): 12 ok, 372 diff | `<data.input_ids>`, counter 13 = step 4 | step 4, `input_ids` |
| fixed resume | 384 ok | none | none |
| `Recorder(opguard=...)` while running vs `export()` afterwards, same run | 384 ok | none | — |

Each OpGuard comparison took about 3 s.

## Cost

| Measurement | Value |
|---|---|
| `record` of one batch 8 × 4096 int64, 4 fields, 8 documents per row | 7.3 ms |
| `compare`, per record | 0.16 ms |

The record cost is host work (the batch is on the CPU at the loader's exit);
it is not measured on device tensors or against a step time.

## Limits

- Equal fingerprints are equal bits up to a hash collision (64-bit keys,
  32-bit XOR values), not a proof. A differing fingerprint is proof.
- The classifier sees documents through `position_ids`, the attention mask and
  `sep_id`. Packed data with none of them is compared row by row, so a
  repacking is classed samples.
- Cross-step matching indexes every document of both runs in memory.
- Shown on one host, CPU, with an in-process synthetic corpus and tokenizer;
  ranks were simulated by building one loader per rank in one process. Not yet
  shown: a real distributed job, a streaming source, the libraries in the
  survey (none was run here; the `torchdata` on the host predates
  `StatefulDataLoader`).

## Reproduce

```bash
cd examples/data_alignment
python data_alignment.py --quick --json results-quick.json
python data_alignment.py --opguard-root '~/OpGuard' --opguard-python '~/opguard-env/bin/python' --json results-cpu.json
python ../../tools/data_fingerprint.py selftest
python ../../tools/data_fingerprint.py compare RUN_A RUN_B [--global] [--json out.json]
```

Without `--opguard-root` the OpGuard checks are left out. Quick mode took
about 6 s, full mode about 30 s with the three OpGuard comparisons.
