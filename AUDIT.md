# Independent audit: handbook, skills, suite and the OpGuard RL harness

- **Date:** 2026-11-19.
- **Scope:**
  - The handbook working tree on top of `c2d39c2`.
  - The OpGuard working tree on top of `39601f8`, with its uncommitted RL-harness changes.
- **Method:**
  - Every number below was read from files or produced by commands run during this audit.
  - Nothing in either repo was edited apart from this file.
- **GPU re-run: not done.** node0, and gpu-host too, were unreachable from this Mac for the whole audit. Every SSH attempt, with and without the sandbox, failed with `Can't assign requested address`, and a ping to <node0-ip> got no reply. The route goes through VPN interface utun10, so the VPN was most likely down. Section 5 lists what is still to do.
- **Page names:** this audit predates the handbook's reorganization into parts. It cites the old case pages by file name and line (for example `rl-loop-e2e.md:134`); those pages now live under `handbook/1-pretraining/`, `handbook/2-rl/` and `handbook/3-kernels/` (`rl-loop-e2e.md` is `2-rl/rl-loop.md`, `kernel-cases.md` was split into `3-kernels/attention.md`, `gemm-and-quantized.md` and `linear-attention.md`), and the cited line numbers no longer apply.

## 1. Coverage

Column key:
- **Impl** = code exists.
- **Meas** = a committed results JSON exercises it.
- **Doc** = a chapter or case page covers it.
- **Skill** = a skill step tells an agent to use it.

### Pre-training

| Stage | What covers it | Impl | Meas | Doc | Skill | Gap |
|---|---|---|---|---|---|---|
| Reference choice | ch1 step 1 | n/a | yes, indirectly: the pretraining_e2e scenarios | yes | pipeline A2, pretraining step 1 | none |
| Determinism controls | `tools/determinism_preflight.py` (`report`/`findings`/`diff`) | yes | `pretraining_e2e` (warn_only finding; resume diff names the step counter) | ch1 step 2 | A1, step 2 | none |
| Capture, eager | `tools/opguard_bridge.py` (`run_pair`, `stage_checkouts`, `coverage`) | yes | `pretraining_e2e` results-a16 (25 pass) | case page | A3, step 3 | `torch.nn.Linear` GEMMs are not captured (documented) |
| Capture, compiled / CUDA graph | `tools/graph_probe.py` (`isolate=True`) | yes | `graph_probe` results-a40: 36 pass, covering 3 compile modes and graph replay | compile-graph-capture.md | A3' | case page measured on the A40; the quick mode is also in the A16 baseline (64 s). No compiled multi-GPU or engine-side capture |
| First divergence | `opguard_bridge.first_divergence` | yes | `pretraining_e2e` | ch1 step 4 | A4, step 4 | none |
| Localization | OpGuard skills `investigate-first-diff` / `trace-upstream` / `validate-hypothesis`, `suggest()` | yes (all three exist in OpGuard `harness/skills/`) | known-answer scenarios only | ch1 step 5, leads.md | A5, steps 5–6 | not exercised by an agent working blind |
| Fix | training-step and DP cases, batch-invariant-kernel skill | yes | yes | yes | A6, step 7 | none |
| Training step: backward, optimizer | `examples/training_step` | yes | results-a40 (146 pass) | training-step.md | A6 | A40 only |
| DP / FSDP | `examples/training_dp` | DDP real; **FSDP emulated** | 1 node and 2 nodes (166 / 181 pass) | training-dp-fsdp.md | A6, B4 | `torch.distributed.fsdp` never run (the probe was written but not launched); no DeepSpeed ZeRO; no bf16 or mixed-precision gradients. README, ch1 and ch3 call this "DP and FSDP" without saying "emulated" |
| Resume | training_step (9 omit-one variants), pretraining_e2e (resume scenario), training_dp (elastic resume d4→d8, d8→d16) | yes | yes | yes | pretraining step 2 (resume bullet); the pipeline names it only in its routing table | none |
| SDC | `tools/sdc_scan.py`, `sdc_guard`, `cross_rank_check` | yes | sdc_scan: a40 (26 pass), a16 fleet (109), guard (2), ranks (3) | sdc-scan.md, ch1 SDC | A0 | 2 figures unbacked (see section 2) |
| Regression | `tools/bi_suite.py`, `suite/manifest.json`, baselines, `.github/workflows` | yes | 3 quick baselines; comparison logic re-run locally (section 3) | ch4 | A8, B10, step 9 | CI workflow never run (it says so); full mode never run through the suite; 11 permanent "CHANGED" lines on A16 (section 2) |

### RL train–inference

| Stage | What covers it | Impl | Meas | Doc | Skill | Gap |
|---|---|---|---|---|---|---|
| Operator audit | `tools/bi_audit.py`, `examples/whole_model/audit/*.md` | yes | audits of stock, bi and hf models | whole-model.md | B1 | none |
| Per-op batch invariance | `bi_check.py`, `bi_kernels.py`, the kernel cases | yes | all `*_bi` examples | ch3, case pages | B2, batch-invariant-kernel skill | FP8 tensor-core, NVFP4 and newer attention libraries are untested (ch5) |
| Whole model, L1–L3 | `examples/whole_model` | yes | results-a40 (158 pass), HF check | whole-model.md | B3 | the Qwen2 cached-decode statement is incomplete (section 2) |
| L5 / TP / disaggregation | `examples/whole_model_tp`, collectives_tp, moe_ep | yes | results-a16 (42 pass), 2-node collectives and MoE | whole-model-tp.md, collectives, MoE EP | B4 | only up to 2 nodes over PCIe; no NVLink or IB (ch5) |
| Sampling | `tools/bi_sampling.py`, `examples/sampling` | yes | results-a40 (72 pass) | sampling.md | B5 | one stale number (section 2) |
| LoRA | `examples/lora` | yes | results-a40 (36 pass) | lora-multi-adapter.md | B6 | no LoRA inside a real engine |
| L4 against a real engine | `examples/cross_engine` (vLLM 0.19.1, Qwen2.5-0.5B, A16) | yes | `results-A16.json`: patched 0/15,231 and 0/76,155 differing | cross-engine-vllm.md | B6, engine-parity skill | one engine, one small model, single GPU. SGLang not run: the ring envs are broken. No engine TP, MoE, prefix cache or speculative decoding at L4 |
| Online records | OpGuard `rl_records.RecordWriter` | yes | rl_loop: 283 harness calls | RLAlignment.md, ch2 | B7, rl-alignment step 0 | **only a toy in-process engine was used** (rl_core.py). No recipe or hook for vLLM, SGLang, verl or OpenRLHF; records were never written from a real engine |
| Staleness / numeric split | `rl.decompose` | yes | rl_loop sync/async1/async2 (pass) | RLAlignment.md, ch2 | B8, step 2 | none |
| Weight sync | `rl.weight_sync_check` | yes | rl_loop (fused QKV, planted faults) | yes | B8, step 1 | none |
| Offline replay and localization | `rl.replay_request`, `rl.forwards`, `vectors.*`, `find_events` | yes | rl_loop: four planted faults | rl-loop-e2e.md | B9, rl-alignment step 3 | a known-answer walk whose data-flow order was written from `model.py`; capture used module hooks, not OpGuard's tracer; `token_match.py` not run; batch-invariant against original composition not compared; mixed-version batches not recaptured; single rank, no TP/EP/MoE, no CUDA graphs |
| Regression / CI | as in pre-training | yes | rl_loop and cross_engine are in the A16 baseline | ch4 | B10 | the same gaps as in pre-training |

## 2. Claims against evidence

About 120 headline claims across 19 case pages and 3 chapters were read against the JSON. Most match. The mismatches and unbacked figures follow. Every row was confirmed by reading the JSON; I spot-re-checked the rl-loop and sampling rows myself.

### Mismatches

| Page:line | Page says | JSON says |
|---|---|---|
| rl-loop-e2e.md:134–139 (Cost table) | tiny step with records off 190 ms; default config off/on 1,032 / 1,556 ms; fingerprints 4.6/2.7 and 12.0/10.6 ms; snapshot 474 ms; record size 87 KB | 157.5; 1160.9 / 1720.5; 4.9/3.65 and 12.9/11.8; 483; 90,467 B. The prose comes from an earlier run |
| rl-loop-e2e.md:144–145 | trainer replay 58 / 56 ms | 62.4 / 58.6 |
| rl-loop-e2e.md:147 | "66–71 harness calls" per localization | `harness_calls` = 49 (five localizations; 25/45/55 elsewhere) |
| rl-loop-e2e.md:148 | "Full run: 147 s" | `extra.runtime_s` = 140.6 |
| pretraining-opguard-e2e.md:131–132 | CPU times: torch 145–158 ms, OpGuard 370–378 ms | 143.3–161.3 and 371.7–516.6 ms |
| sampling.md:84–85 | the SGLang rule changed "6 of 512 rows" between two identical single-row calls | `single_row_call_twice` = 4 (`checks[181]`); 0 in every other config |
| kernel-cases.md:56 | FP8 per-tensor dynamic: "up to 963/1024" elements differ | 988 is the maximum among the recorded failures (only 50 of 164 are listed) |
| moe-combine.md:47 | row-chunk bmm "fail from 127" | `bi_chunk_bmm.L2_batch`: fails at [33, 64, 127, 128, 256, 512] |
| moe-combine.md:29 | router GEMM fails at "M = 31–64, 1024" | it fails at [31, 32, 33, 64, 1024]. A range implies 34–63 also fail |
| lora-multi-adapter.md:90 | failing group sizes "8, 15–17, 31–34, …" | group size 3 also fails (grouped_cublas and gathered_bmm) |
| whole-model.md:156 | Qwen2 fp32/bf16 stock gives "0 (bitwise equal)" | true for prefill only. Cached decode against HF prefill differs: 5.48e-6 in fp32, 0.031 in bf16 (`results-hf-a40.json`, info) |

### Unbacked: in the prose, not in any JSON

- **sdc-scan.md:112:** GPU-against-CPU differences of 26,540 and 29,772 out of 4.2M.
- **sdc-scan.md:198:** a `--quick` A40 run of 57 s. Every committed SDC file has `quick=false`.
- **whole-model.md:163:** "4.5e-6", from an earlier version.
- **01-pretraining.md:74:** the upper end of "242–345 ms".
- **attention-paged-varlen-compile.md:220:** 0.63 ms "in an earlier run".
- **lora-multi-adapter.md:226:** 2.0 against 1.2 ms. The page says this was not recorded.
- **kernel-cases.md:81 and linear-attention.md:194:** "2–3x trainer throughput", an external report with no citation marker next to it.

### Overstatement

- **02-rl-train-inference.md:133:** "Four planted faults were localized end to end". The case page itself says the walk's data-flow order was written from `model.py`, not discovered. The chapter should say "known-answer".
- **README.md:38, 01-pretraining.md:84, 03-batch-invariant-kernels.md:64:** "DP and FSDP" with no mention that FSDP was emulated.

### Required checks that failed

- **None.** Every `expect="pass"` check in every new-format results JSON and in all three `suite/baselines/*.json` files has `passed=true`:
  - A40 baseline: 443 required checks.
  - A16x8 baseline: 754.
  - A16 2-node baseline: 223.
- **Old-format files.** Ten result files predate the `expect` field: attention_backward, attention_serving, collectives_tp, kernel_cases, linear_attention, mla_sparse, moe_combine and moe_ep, plus `tools/demo-a40.json`. They have bare `passed` fields only, so for them this question can only be answered through the suite adapters, which turn them into required checks in the baselines. Those passed.
- **Documented failures that now pass.** In the A16x8 baseline, 11 `expect="fail"` checks have `passed=true`:
  - kernel_cases L2 sdpa;
  - 7 lora L3/L4 cuBLAS and gathered-bmm checks;
  - 2 mla_sparse L2 checks;
  - moe_combine L2 `torch.mm`.
- **The pages are honest about this.** Each page notes it in an "Other hardware" bullet, and the claims are A40 claims.
- **The suite is not.** I replayed the baseline through `bi_suite.py --from-report`, and every A16 run prints these 11 as `CHANGED`. That is permanent noise. Either make `expect` depend on the stack, or record these as `info` on the A16.

## 3. What ran locally (/usr/bin/python3, no torch)

| Command | Result |
|---|---|
| OpGuard `python3 harness/tests/test_rl_actions.py` | 21 tests OK |
| OpGuard `cd harness && python3 -m unittest discover -s tests` | 68 tests OK, 2 skipped: the real backend round trip needs an OpGuard install; the pinned demo artifact was not provided |
| OpGuard `python3 tests/test_xor_vectors.py` | 5 PASS, 3 SKIP (need torch/CUDA): the native kernel against the fallback, v1 capture for every dtype, and the workload through the tracer. **None of the GPU paths of the XOR-vector change were exercised here.** |
| Handbook `tools/bi_suite.py --list` | runs; lists 22 entries, all skipped on this Mac with correct reasons |
| `bi_suite.py --from-report`, a report built from the A40 baseline | 809 checks (443 required), 0 regressions, exit code 0 |
| The same report with one required check flipped to false | exactly 1 regression, exit code 1. The comparison logic works |
| The same with the A16 baseline | 0 regressions, 11 CHANGED (section 2) |
| OpGuard harness alias table | `rl.*` and `vectors.*` dotted aliases resolve to all 9 tools; `find_events` and `first_diff` exist |
| Relative links and anchors | **0 broken** in every `.md` of the handbook (430 link lines) and in OpGuard's changed docs: AgentTools, Capture, RLAlignment, harness Usage, and the rl-alignment SKILL. Anchors were checked with GitHub slug rules |

## 4. Can each skill be followed with the tools that exist?

**Method.** I grepped every file, function, flag and field the skills name:
- **Files:** every `tools/*.py` named.
- **Functions:** `run_pair`, `stage_checkouts`, `first_divergence`, `coverage`, `suggest`, `TraceWriter`, `xor_vectors`, `report`, `findings`, `diff`, `sdc_guard`, `cross_rank_check`, `bit_equal`, `check_batch`, `check_repeat`, `check_modes`, `Results.check`/`measure`/`write`/`exit_code`, `review_pair` (in OpGuard `reviews.py`), `RecordWriter.forward`/`rollout_batch`/`trainer_logprob`/`weight_sync`/`snapshot`, `param_entries`, `weights_fingerprint`, `read_records`, `validate_record`, `xor_vector_stats`.
- **Flags:** `--quick`, `--json`, `--golden`, `--record-golden`, `--record-baseline`, `--from-report`, `--vs-committed`, `--patch-lmhead`, `isolate=True`.
- **Fields:** `runnable_alone`, `capture_env`, `composition`, `token_ranks`, `rows`, `equal_zero_only`, `call_sequences`, `caller_supplied`, `mixed_version`.
- **OpGuard skills:** `investigate-first-diff`, `trace-upstream`, `validate-hypothesis`, and `evidence-contract.md`.

**All of them exist.** No skill step references a nonexistent file, action, flag or field.

### Steps that can be followed only in part

1. **rl-alignment step 0 / pipeline B7.** These assume records are written by the RL job. The only writer integration is the toy engine in `examples/rl_loop/rl_core.py`. For vLLM, SGLang, verl or OpenRLHF, an agent must invent where to call `rec.forward` and `rollout_batch`, and how to get `rows`. No guide or adapter covers this.
2. **rl-alignment step 3.2 (recapture with `OPGUARD_XOR_ROWS=1`).** This works only where OpGuard's tracer instruments the engine. The skill says a replay with its own boundary writer must compute vectors itself. Its only demonstration used module hooks, and no demonstration exists for a compiled or CUDA-graph engine, which is vLLM's default.
3. **Pipeline A5 / pretraining step 5.** These hand off to OpGuard skills, but no recorded run shows an agent using them without knowing the answer. The e2e cases are known-answer, with planted causes.
4. **Pipeline A8 / B10 on a stack without a baseline.** This is followable: `--record-baseline` exists. But the first run on any new stack has nothing to compare against, by design.
5. **engine-parity.** This is followable for vLLM on sm86. Its section 5 compiled-engine path points at OpGuard vectors, which have not been demonstrated on a compiled engine.

## 5. GPU re-run

**Not done: unreachable.**

| | Result |
|---|---|
| node0 SSH | 6 attempts, each `ssh: connect to host node0 port 22: Can't assign requested address` (exit 255) |
| gpu-host SSH | the same |
| Checksums | not compared |
| `lora_bi.py --quick` and a second quick mode | not run |

Local SHA-256 values, for when someone does the comparison:

| File | SHA-256 prefix |
|---|---|
| `examples/lora/lora_bi.py` | `7faa6f88…` |
| `examples/lora/lora_kernels.py` | `a9ff4fe5…` |
| `examples/sampling/sampling_bi.py` | `41d296a7…` |
| `tools/bi_sampling.py` | `2b7d1231…` |
| `tools/bi_check.py` | `98e07a90…` |
| `tools/bi_kernels.py` | `9ffb2991…` |
| `tools/bi_suite.py` | `99352291…` |
| `tools/bi_report.py` | `4c838a7c…` |

To do once the VPN is back:
1. Compare these checksums with `~/bi-check-suite`.
2. Run `flock /tmp/bi-check-16gpu.lock ~/miniconda3/envs/maia/bin/python examples/lora/lora_bi.py --quick --json /tmp/audit-lora.json`, and the same for `examples/sampling/sampling_bi.py --quick`.
3. Compare both with the A16 baseline through the suite (`bi_suite.py --only lora,sampling`).
4. Also confirm that `~/bi-check-opguard/OpGuard` on node1/19 matches this OpGuard working tree. The rl_loop case says it ran the uncommitted harness, and that could not be checked.

## 6. Honesty

- **Sales words:** none found in either repo's docs. The only hits are the OpGuard Usage heading "Execution and recovery guarantees", which states its limits, and "end to end" in two case titles.
- **"Untested" / "not run" items:** they are documented in the case pages' "Known limits" sections, ch4 "Limits" and ch5 "Untested". **There is no single open-items list.** Ch5 covers hardware only. The non-hardware open items are scattered across about 19 pages:
  - FSDP emulated;
  - SGLang not run;
  - `token_match` not run;
  - full suite mode not run;
  - CI not run;
  - no real-engine records.
  - The README "Status" section says only "Draft".
- **Stale prose:** see section 2. The rl-loop "Cost" section is the worst case, with a whole table from an earlier run.

## 7. Remaining gaps, by priority

### Fixable now, without new hardware or downloads

1. Correct the stale or wrong numbers in section 2. The rl-loop cost table and localization lines come first, then pretraining CPU times, sampling 6→4, kernel-cases 963, moe-combine 127 and 31–64, lora group size 3, and the whole-model Qwen2 decode row. Remove or source the unbacked figures.
2. Add one consolidated "Open items" section, in the README or a ch6. List every untested or not-run item from the case pages and chapters, not only hardware.
3. Write "FSDP-style (emulated)" wherever FSDP is named in a summary, and "known-answer" in ch2:133.
4. Make the 11 A16 `expect="fail"` checks stack-aware, or record them as `info` there, so an A16 suite run is not permanently `CHANGED`.
5. **Run on the existing A16/A40 machines once they are reachable** (sm86 hardware, no new purchase):
   - the real `torch.distributed.fsdp` probe that was already written;
   - full-mode suite runs and their baselines;
   - the CI workflow once on a self-hosted runner;
   - `token_match.py` on the rl_loop exports;
   - the batch-invariant against original-composition replay comparison;
   - mixed-version recapture;
   - the GPU-only tests of `test_xor_vectors.py`;
   - the GPU spot re-run of section 5.
6. Write a records-integration guide for one real engine: where `RecordWriter` hooks go in vLLM 0.19.1. Run the rl_loop scenarios with vLLM as the rollout side, using the Qwen2.5-0.5B already on node0. This is the largest gap between "harness exists" and "an agent automates RL alignment end to end".
7. Run a blind trial: an agent that is not told the planted cause follows `alignment-pipeline` → `rl-alignment` (and the pre-training path), with launches and harness calls recorded. Today every end-to-end case is known-answer.

### Needs hardware, software or downloads not available

- Hopper, Blackwell and Ada; FP8 tensor-core GEMM; NVFP4; FA3, FlashMLA and FlashInfer on new GPUs (deferred by scope).
- NVLink, NVSwitch, IB/RDMA, and more than two nodes.
- SGLang L4: the node0 envs lack libcudnn 9 and libnvJitLink, and fixing that requires installs. DeepSpeed ZeRO.
- A larger pretrained model or an MoE model in a real engine, for L4 with TP/EP, prefix caching or speculative decoding. That needs weight downloads.
- Other torch, Triton and NCCL versions, and CUDA 12.8+ for the handbook kernels.

## Resolved (2026-11-19, local follow-up, no GPU)

The findings above are kept as written. Each fix below was checked against the
results JSON named in section 2, read again for this follow-up.

### Section 2, mismatches (11)

- **rl-loop-e2e.md, Cost:** table and bullets rewritten from `results-a16.json` `measurements`, with the scenario each value comes from: records off 157 ms (tiny, `cost`) and 1,161 ms (default); on 220 / 1,720 ms; fingerprints 4.9/3.7 and 12.9/11.8 ms; record writes 1.2/3.1 and 1.4/34.2 ms; snapshot 10.6 / 483 ms; record bytes 90,467 / 1,639,824 B. Sum of record parts now about 13 / 60 ms (ch2 updated from 12 to 13 ms). Added: the `cost` scenario's tiny records-on step (615 ms, 469 ms of it generation) is not the cost of records.
- **rl-loop-e2e.md:** trainer replay 62 / 59 ms (online log-prob 14 ms, from the async step lists); 49 harness calls per localization, 6.7–6.9 s of CLI time (summed from `transcript-a16.json`); export 257,585–265,974 B; recapture 0.10–0.12 s; full run 140.6 s. The `--quick` 24 s is marked "not in the saved results".
- **pretraining-opguard-e2e.md:** CPU signature 143–161 ms (torch) and 372–517 ms (OpGuard NumPy); fp32 vectors 135 and 239 ms.
- **sampling.md:** 4 of 512 rows (`checks[181]`), 0 elsewhere.
- **kernel-cases.md:** FP8 per-tensor: 98 of 164 cases fail; 911–988 of 1024 elements differ in the 50 recorded failures.
- **moe-combine.md:** row-chunk bmm fails at 33, 64, 127, 128, 256, 512; router GEMM at 31, 32, 33, 64, 1024.
- **lora-multi-adapter.md:** group size 3 added (grouped cuBLAS, gathered bmm); the list is the union of `failing_group_sizes`.
- **whole-model.md:** Qwen2 row limited to prefill; a new row gives cached decode against HF prefill: 5.48e-6 (fp32), 0.031 (bf16).

### Section 2, unbacked figures (7)

- sdc-scan.md: the 26,540 / 29,772 sqrt counts are marked "measured during development; not in the saved results". The 57 s `--quick` row is replaced by the A40 quick suite baseline: 26 s, self-replay 14.7 s and self-test 8.6 s. README's "about a minute" became "about 30 s".
- whole-model.md 4.5e-6, 01-pretraining.md 345 ms (and the case page's copy), attention-paged-varlen-compile.md 0.63 ms, lora-multi-adapter.md 2.0 against 1.2 ms: each marked "not in the saved results". Ch1 now quotes 242 ms.
- kernel-cases.md and linear-attention.md "2–3x": cited to Wang et al. in references.md, marked "not measured here" in kernel-cases.

### Section 2, overstatement

- ch2:133 and the README rl-loop row now say "known-answer".
- FSDP: README case row, ch1, ch3, ch5, the alignment-pipeline skill, the kernel skill's operator-recipes and verification references, and the case's H1 now say "FSDP-style (emulated: flat gradient, reduce-scatter, sharded AdamW)".

### Section 2, the 11 A16 CHANGED lines

- **Why they stayed.** `compare()` in `tools/bi_suite.py` reported every passing `expect="fail"` check as CHANGED and never looked at the baseline for it. The stack-matched baseline was loaded, but the check ignored it.
- **Fix.** If the stack's baseline has the same check with `expect="fail"` and `passed=true`, the check now goes into a new information-only section, AS BASELINE. It counts as ok in the per-entry table. The summary line shows the count, and `--verbose` and `report.json` list the rows. If the check fails again on that stack, it is CHANGED again ("verdict differs from the baseline"). A stack with no baseline for the check still reports CHANGED.
- **Test.** The saved A16 report is not in the repo (`suite/runs/` is gitignored), so I rebuilt a report from the A16x8 baseline and replayed it with `--from-report`. Results:
  - Before the fix: 11 changed. After: 0 regressions, 0 changed, 11 as baseline, exit code 0.
  - One `expect="fail"` check flipped to passed: 1 CHANGED.
  - One required check flipped to failed: exit code 1.
  - The A40 report: 0 / 0 / 0.
- **Documentation.** `handbook/4-operations/regression.md` documents the new section and corrects its baseline note.

### Section 6 and section 7, item 2

- [handbook/06-open-items.md](handbook/open-items.md) lists every untested or open item in four groups: needs hardware, needs GPU time on existing machines, needs an engine/framework integration, and needs design work. It draws on the case pages, ch4, ch5 and this audit. README (chapter table and Status) and the alignment-pipeline skill link to it.

### Links

- All relative links and anchors were re-checked under GitHub slug rules: 456 links in the handbook's `.md` files, 0 broken.

### Not fixed (needs a GPU, an integration or design work)

- Everything in section 5, and section 7 items 5–7.
- Each of these is now an entry in `handbook/open-items.md`.
