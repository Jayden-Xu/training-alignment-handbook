# An RL loop through OpGuard's RL harness: lab report

This is the full record behind the handbook page
[The RL loop](../../handbook/2-rl/rl-loop.md). The page teaches the step;
this report keeps the setup, every measurement and how the case was built.

## Summary

- A small RL loop on A16s, run through OpGuard's RL harness: with
  batch-invariant op sets on both sides, `numeric` is bit-zero on every
  non-mixed token, sync and async (lag 1–2), so staleness carries the whole gap.
- The weight-sync check survives a fused QKV repack and names the faulty row on
  both sides.
- Four planted faults were localized end to end (known-answer: toy engine,
  causes planted). A version-label fault shows only by its pattern.
- **Cost:** records and row fingerprints about 13 ms per step on the tiny
  model, plus snapshots.

## How

OpGuard's RL harness (records, `vectors.*`, `rl.*` actions, the rl-alignment
skill) had unit tests but had never run inside a training loop. This case runs a
small, real RL loop with it: online records, the split of each log-probability
gap into staleness and numeric error, the weight-sync check, and offline
localization of planted faults. Every fact below comes from one
`opguard-harness call` (a subprocess per call), in the order the skill gives.
### The loop

| Part | What runs |
|---|---|
| model | `examples/whole_model/model.py`, `Config.tiny()` (2 layers, hidden 256, 4 heads, 2 KV heads, vocab 1000), bf16, random init |
| rollout process | `forward_cached` with a slot KV cache; 16 sequences (4 prompts × 4 samples) join in three waves (engine steps 0, 2, 5) and stop after 3–12 tokens, so prefill and decode rows mix and the batch changes every step; `bi_sampling.sample` (counter keyed); Q/K/V repacked into one fused `qkv_proj` weight and run as one GEMM |
| trainer process | `model.py` prefill on the right-padded batch for old log-probs (no grad), then REINFORCE with group-normalized advantages; AdamW on fp32 master weights, bf16 copy pushed |
| push | every version, as CPU tensors through a pipe; the rollout repacks and fingerprints its own copy |
| sync | lag 0: batch k generated and trained at version k |
| async | lag 1 or 2: batch k starts at version k−L; version k−L+1 arrives after engine step 4 with the KV cache kept, so sequences running across it are mixed |
| reward | share of response tokens with id % 8 == 0 (recorded; this case makes no learning claim) |

Records, written with `RecordWriter` at the documented points: `forward` (each
engine and trainer forward), `rollout_batch` (per-token version and fp32
log-prob, composition = the seq_ids of each engine forward, engine flags),
`trainer_logprob` (version c = k), `weight_sync` (per-row fingerprints of the
trainer's q/k/v layout and of the rollout's fused layout, made in the rollout
process after repacking), `snapshot` (bf16 weights of every version).

## Evidence

### Setup, code and results

- Code: [examples/rl_loop/rl_core.py](rl_core.py) (engine,
  trainer, records, replay, capture), [rl_loop.py](rl_loop.py)
  (scenarios and the agent's tool calls), [run.sh](run.sh).
- Results: [results-a16.json](results-a16.json) (21 ok, 2 info);
  every harness call with trimmed output in
  [transcript-a16.json](transcript-a16.json) (283 calls).
- Setup: two NVIDIA A16 (sm86, 10 SMs), driver 535.309.01, torch 2.5.1+cu121,
  Triton 3.1.0. OpGuard: the working tree on top of commit 39601f8 with the
  uncommitted RL-harness changes (records, `vectors.*`, `rl.*`, and the fixes for
  the findings below), native XOR extension built.

### Results: bi op sets on both sides

| Scenario (12 steps each) | weight_sync_check | tokens | mixed-version tokens | numeric == 0 (non-mixed) | staleness ≠ 0 | trainer replays |
|---|---|---|---|---|---|---|
| sync (lag 0) | 12/12 versions, 0 unmatched rows (6,613 rows per side) | 1,419 | 0 | 1,419 / 1,419 at staleness-zero points | 0 | 0 |
| async, lag 1 | 12/12 clean | 1,419 | 914 (114 sequences) | 505 / 505 after replay | 67 | 10 |
| async, lag 2 | 11/11 clean | 1,419 | 852 (105 sequences) | 567 / 567 after replay | 426 | 20 |

- `numeric` is bit-zero for every non-mixed token, at staleness-zero points
  without replay and elsewhere after a trainer replay at the token's version.
  The whole gap of every non-mixed token is staleness.
- The fused QKV layout is not a difference: rows match as multisets.
- In async lag 2, 22 tokens with v ≠ c had staleness exactly zero: the update
  did not move that log-prob's bits.
- Replays were requested with `rl.replay_request` (side trainer, composition
  original), run offline from the `snapshot` record, and written as
  `trainer_logprob` records in separate files.

### Results: planted faults

| Fault | Step 1 weight_sync_check | Step 2 rl.decompose | Step 3 localization (studied token, first paired candidate) | Check of the cause |
|---|---|---|---|---|
| a. rollout runs the stock op set; its flags still say batch-invariant | clean | numeric ≠ 0 on 366 of 448 tokens, all staleness-zero points | token seq 4 pos 4 (decode row, forward 1): agreement through layer 0 attention, o_proj and post-attention norm; first differing row at `layers.0.mlp.gate_proj` output, input row equal on both sides (cuBLAS against the one-config GEMM) | replay of the rollout with the bi op set: 98/98 tokens equal the trainer |
| b. rollout RMSNorm multiplies by its weight in fp32, then rounds | clean | numeric ≠ 0 on 444 of 448 | token seq 4 pos 3 (prefill row, forward 0): norm input equal, `layers.0.input_layernorm` output differs | replay without the change: 98/98 equal |
| c1. one bit flipped in the fused qkv weight after repacking (v1) | names trainer `layers.0.self_attn.k_proj.weight` row 3 and rollout `layers.0.self_attn.qkv_proj.weight` row 259 (= 256 + 3) | numeric ≠ 0 on 79 of 98 | not needed: step 1 stops | — |
| c2. one stale row after the push (v2) | names `layers.1.mlp.down_proj.weight` row 7 on both sides | numeric ≠ 0 on 83 of 114 | not needed | — |
| d. rollout labels batch 2 with the version it was sent (2) but applies it after the batch | clean (the check runs after the weights are applied) | numeric ≠ 0 on all 114 tokens of batch 2, 0 in batches 0, 1, 3 | recapture at the recorded version 2 does not reproduce the online rollout log-probs, and equals the trainer on 114/114; the walk finds no differing row | the batch's weights fingerprint matches the version-1 sync; replay the trainer at 1 and decompose with `policy_version_override: 1` (caller-supplied): numeric 0 on 114/114, staleness ≠ 0 on 114 |

Localization used: `rl.replay_request` with capture `{vector_fingerprints:
true}` on both sides (composition original), a replay of each side with every
`model.py` module boundary captured by `opguard_bridge.TraceWriter(vectors=True)`
(OpGuard's own fingerprint functions), `opguard_bridge.export_pair` (OpGuard's
backend `compare`), then `first_diff`, `find_events`, `rl.forwards` and 31 `vectors.compare`
calls along the data flow, from the embedding output (the anchor) to the
log-probabilities.

### What the skill steps produced

1. **Setup (step 0).** `rl.records_summary` listed every record type and the
   weight-sync and snapshot versions. Engine flags are claims: in fault a the
   flags said batch-invariant while the stock op set ran.
2. **Weights (step 1).** Both sync faults were named to the parameter and row on
   each side, through the repack.
3. **Split (step 2).** `versions_without_trainer_logprob` said which replays to
   request. Mixed tokens came back with `numeric: null` and
   `not_split: mixed_version_sequence`.
4. **Localize (step 3).** The fused `qkv_proj` row (1,024 bytes) has no partner
   of its width on the trainer (512 bytes for q). Following its consumer through
   the split to the RoPE input gave rows of equal width (per head, 128 bytes)
   on both sides. In fault a and b the walk stopped at a paired candidate: same
   input row on both sides, different output row of the same width.
5. **Cause (step 4).** b is a rounding point. a is an algorithmic difference
   between two GEMMs (the stock GEMM agreed for q/k/v and o_proj on these
   shapes, not for gate_proj). d is bookkeeping: online numeric ≠ 0 that an
   offline replay of the same code at the recorded version does not reproduce.

### Findings about the harness and the skill

All nine are closed in OpGuard's harness, and this loop now uses the fixes in
place of its workarounds (the rerun is below in [After the fixes](#after-the-fixes)):

| # | Closed by |
|---|---|
| 1 | `rollout_batch` sequences take an optional `prompt_ids`; `rl.replay_request` copies it and says `runnable_alone` (prompts and weights all recorded). The replay reads prompts from the request. |
| 2 | `forward` records take optional observed `rows`: the `[seq_id, position]` of each row, `null` for padding. `rl.forwards` lists them, and they go straight into `vectors.compare` as `token_ids`. The inferred layout is gone. |
| 3 | `find_events`: a paged listing of events by a literal in the full `fn` or the name, run, rank and counter range, with io slots. It replaces paging `trace_neighbors`. |
| 4 | No matcher was added. `first_diff` now returns per-rank call counts and says when its silence is not evidence (the runs called functions a different number of times); the skill sends cross-shape comparison to `vectors.compare`. |
| 5 | `vectors.compare` reports `equal_zero_only` when agreement rests only on zero fingerprints; it is never `equal`. |
| 6 | `rollout_batch` takes an optional observed `weights_fingerprint` (XOR of all row fingerprints of the weights at batch start; any packing agrees); `rl.decompose` reports which weight-sync versions it matches. |
| 7 | `rl.decompose` takes `policy_version_override`; the result says `policy_version_source: caller_supplied` and keeps each token's recorded version. |
| 8 | Documented, and the request carries `capture_path: {assumes: opguard_tracer}`: `capture_env` reaches only OpGuard's tracer; a replay with its own writer computes vector fingerprints itself (as this one does). |
| 9 | The skill says it: compare a fused producer at its first consumer after the split. |

Original findings, as found:

| # | Finding | Where it showed |
|---|---|---|
| 1 | `rollout_batch` has no field for the prompt (context) tokens, and `rl.replay_request` copies only `seq_id`, `token_ids`, `positions`. A replay request cannot be run alone. Here each sequence carried an extra `prompt_ids` field (validation accepts it, the request drops it) and the replay read it from `source_records[0]`. | every replay |
| 2 | `forward` records give T, not which (seq_id, position) each row holds; `composition` gives the seq_ids per forward but not each row's token count. The engine's row layout was inferred from composition + prompt lengths (first appearance = whole prompt, then one token). Chunked prefill or speculative rows would make this ambiguous. | step 3.3 |
| 3 | No action lists events by name or counter. Event refs came from paging `trace_neighbors` (41 events per call): 22–25 calls per localization. The full module name is in `fn`; the event `name` is shortened (`0.input_layernorm#2`). | step 3 |
| 4 | `first_diff` reported no observed diff on either run in faults a, b and d: the exporter's order-based pairing cannot pair one trainer prefill with the 14–17 engine forwards of a batch. Only `vectors.compare` under declared token dims found the divergence. | step 3 |
| 5 | Zero row fingerprints from GQA: at position 0 attention returns v, so the two query heads of each KV head give equal halves and the 512-byte row XOR is 0 (8 such rows in forward 0). `vectors.compare` flags them (`zero_row`) but still labels them `equal`; a walk must treat them as no evidence. | o_proj input, fault b |
| 6 | No record ties a rollout batch to the weights it actually ran (for example a digest of the loaded rows per batch). A version-label fault passes `weight_sync_check` and shows only as numeric ≠ 0 that recapture does not reproduce. | fault d |
| 7 | `rl.decompose` takes each token's version from the record only. Testing "this batch ran at version v−1" needed a relabeled copy of the `rollout_batch` record. | fault d |
| 8 | `rl.replay_request` sets `capture_env: {OPGUARD_XOR_ROWS: 1}`; a replay workload that captures with its own boundary writer (as here) does not use the variable. | step 3.2 |
| 9 | The skill's pair rules worked as written for a fused producer only by comparing at its first consumer after the split; the skill could say so. | fault a, b |

### Cost

All values are the `measurements` of `results-a16.json`, one A16 per process.
Medians over every step after step 0 (step 0 includes kernel compilation).
The tiny column is the `sync` scenario (12 steps, a snapshot every version)
except where marked; the default column is the `cost` scenario (4 steps, a
snapshot every 2nd version).

| Per step | tiny (98–139 tokens) | default config (8 layers, hidden 1024, vocab 32,000) |
|---|---|---|
| step, records off | 157 ms (`cost` scenario) | 1,161 ms |
| step, records on | 220 ms | 1,720 ms |
| row fingerprints, trainer / rollout | 4.9 / 3.7 ms | 12.9 / 11.8 ms |
| record writes, rollout / trainer | 1.2 / 3.1 ms | 1.4 / 34.2 ms |
| snapshot of one version (torch.save) | 10.6 ms (every version) | 483 ms (every 2nd version) |
| record bytes | 90,467 B (`cost` scenario) | 1,639,824 B |

- Records and fingerprints without snapshots: about 13 ms (tiny) and 60 ms
  (default) per step, the sum of the measured parts. The snapshot dominates
  at the default size: sample versions, as the protocol says.
- The `cost` scenario's tiny records-on step was 615 ms, 469 ms of it in
  generation; the parts that records add sum to under 50 ms there. Do not read
  the tiny on/off step difference as the cost of records.
- Trainer replay per batch (load snapshot, prefill, write): 62 ms (lag 1, n =
  10), 59 ms (lag 2, n = 20), against 14 ms for the online trainer log-prob pass
  (median per step in the async1 and async2 scenarios).
- One localization (faults a, b and d): recapture 0.10–0.12 s (trainer) and
  0.36–0.62 s (rollout), export 257,585–265,974 B, 49 harness calls
  (`harness_calls`), 6.7–6.9 s of harness CLI time (summed from
  `transcript-a16.json`).
- Full run: 140.6 s on two GPUs (`extra.runtime_s`). The `--quick` run time
  (24 s on one GPU) is not in the saved results.

### What this shows

1. With the bi op sets on both sides, a real loop with a changing engine batch,
   a fused QKV repack and async staleness of 1 and 2 gives `numeric` bit-zero on
   every non-mixed token. Staleness then carries the whole gap and can be
   measured exactly, token by token.
2. The weight-sync check survives repacking and names the faulty row on both
   sides.
3. The skill localized both code faults to the first operator, and the
   validate step confirmed each cause. The localization needed a replay
   workload, a capture writer and an inferred row layout from the user.
4. A version-label fault is visible only by its pattern: weights clean,
   numeric ≠ 0 at staleness-zero points, recapture not reproducing.

### After the fixes

Rerun on two A16 (GPUs 4–5 of node1) with the fixes in OpGuard's harness and
this loop using them: **21 ok, 0 regressions, 2 info** (one check more than
before: the weights fingerprint), 283 harness calls (was 333), 141 s.

- **Same answers.** Faults a and b localize to the same candidates
  (`layers.0.mlp.gate_proj`, seq 4 pos 4; `layers.0.input_layernorm`, seq 4 pos 3)
  and the sync, async and weight-sync results are unchanged.
- **Row layouts are observed, not inferred** (finding 2): engine and trainer
  replays record `forward.rows`, read back with `rl.forwards` (2 calls per
  localization) and passed as `token_ids`.
- **Events by listing** (finding 3): 3 `find_events` calls per localization in
  place of 22–25 `trace_neighbors` calls; 49 harness calls per localization.
- **first_diff says why it is silent** (finding 4): 0 observed diffs in all three
  localizations, now with 28 trainer against 308–374 engine operations, 30
  functions called a different number of times, and the note that silence is not
  evidence.
- **Zero-only rows** (finding 5): 8 rows of each `o_proj` input in forward 0 come
  back `equal_zero_only` and are no longer counted as agreement.
- **Fault d from records alone** (findings 6–7): batch 2's recorded weights
  fingerprint matches the version-1 sync, not its label 2, while every other batch
  matches its label; the hypothesis replay is decomposed with
  `policy_version_override: 1` (labelled `caller_supplied`), numeric zero on
  every token, with no relabelled copy.
- **Replays from the request alone** (finding 1): every request carries
  `prompt_ids` and reports `runnable_alone: true`.

## Limits

- One small model, 2 layers, 12 steps, single rank on each side; no TP/SP/EP,
  no MoE, no CUDA graphs. The default-config run measured cost only.
- Known-answer walkthrough: the faults were planted and the walk's data-flow
  order was written from `model.py`, not discovered.
- The capture is at module boundaries (forward hooks), not OpGuard's own
  tracer; engine attention is not a module and shows only through o_proj's input.
- `perfetto/token_match.py` was not run. Original-composition against
  batch-invariant replays (batch-dependence class) were not compared.
- Rollout replays exist only for single-version batches; mixed-version
  batches were not recaptured.

## How it was built

What took more than one try:
- Export event names are shortened (`self_attn.q_proj#3`); the walk first keyed
  on `name` and found nothing. It now reads `fn`.
- The first transcript was 569 KB: every `vectors.compare` call carried its
  token-id lists. They are now logged as counts.
- The replay of the rollout side needed the prompts, which no record field
  holds (finding 1).
