# MoE under expert parallelism: lab report

This is the full record behind the handbook page [MoE](../../handbook/3-kernels/moe.md). The page teaches the step and what to do; this report keeps the setup, every table, every number, the limits and how the case was built.

## Summary
- **Expert parallelism:** a fixed-slot combine gives the same bits at EP 1–16
  across two 8x A16 nodes. A per-rank pre-reduce (`rank_partial`) is
  deterministic but not parallel-invariant from top-3 on.
- Top-2 hides combine-order bugs, and capacity-based dropping makes a token's
  output depend on its neighbours: test with top-k ≥ 3 and use dropless routing.
- **Cost:** the fixed-slot combine took 0.86–0.92 of the atomic path's time on
  one node, 0.95–0.98 across two.

## How

The fix in one line: an invariant router, a stable top-k, dropless routing,
expert GEMMs with one configuration for every M, and a per-token sum in a fixed
order (top-k rank, fp32) over rows placed in fixed slots, with no atomics.

## Under expert parallelism

The [single-GPU MoE case](../moe_combine/REPORT.md) made an MoE layer batch-invariant:
one-config expert GEMMs and a per-token fixed-order fp32 combine. Expert
parallelism (EP) moves experts onto different GPUs and adds two all-to-alls.
The question here: does the same layer give the same bits at EP = 1, 2, 4 and
8 (L5), and which combine designs keep L1 and L2 once tokens travel?

### Setup

- **Layer.** bf16; hidden 2048; SwiGLU experts with FFN 1408; softmax router,
  top-k weights renormalized in fp32 with a sum in fixed top-k rank order.
  Top-k is a stable sort: score descending, then expert id ascending. Dropless
  unless stated.
- **Configs.** E = 32 experts top-4 (main), E = 32 top-3, E = 16 top-2.
- **EP.** W ranks, E/W experts per rank, tokens sharded across ranks as in
  data/expert-parallel training.
  1. Router GEMM through `mm_one_config`.
  2. Each rank sorts its (token, expert) assignments by expert id (stable, so
     ties stay in (token, top-k rank) order).
  3. An `all_to_all_single` exchanges per-expert counts; a second one sends one
     row per assignment, with split sizes.
  4. The receiver regroups rows by expert; one `mm_one_config` call per GEMM per
     local expert.
  5. The outputs return by `all_to_all_single` in the order they were sent.
  6. Combine (the variants below).
- **Hardware and software.** One host with 8x NVIDIA A16 (sm86, 15 GB each),
  PCIe only, no NVLink. GPUs 0–3 share one PCIe switch, 4–7 another. torch
  2.5.1+cu121, Triton 3.1.0, NCCL 2.21.5. EP = W ran on GPUs 0..W−1; EP = 2 also
  ran on GPUs 0 and 4, across the switches.

### Where the variance comes from

A token's path under EP:

| Step | Arithmetic | Depends on others? |
|---|---|---|
| router GEMM, softmax, top-k | its own row | only if the GEMM is batch-variant |
| dispatch all-to-all | none | no: data movement |
| expert GEMMs | rows routed to the expert from every rank | yes: M_e changes with the whole global batch and its sharding, so the GEMM must be invariant in M |
| return all-to-all | none | no |
| combine | the sum of the token's top-k outputs | yes, if its order or grouping follows timing, rank layout or neighbours |

No collective reduces anything here. The all-to-alls only move rows, so NCCL's
algorithm and protocol cannot change bits (recipe 10's concern applies to TP
all-reduce, not to this dispatch). Three combine designs can still depend on more
than the token itself:
- **atomics:** the order of adds follows timing;
- **per-rank pre-reduction:** the grouping of the sum follows which experts share
  a rank, so it follows the EP degree;
- **capacity:** whether a token's assignment is kept depends on its neighbours.

### What was done

All variants share routing, dispatch and expert GEMMs. Only the combine differs.

| Variant | Combine |
|---|---|
| `arrival_atomic` | multiply each returned row by its bf16 weight, then one `index_add_` into a bf16 output (atomics), as the usual unpermute-and-add does |
| `arrival_det` | the same under `torch.use_deterministic_algorithms(True)` |
| **`fixed_slot`** (the recipe) | each returned row has a fixed slot (token, top-k rank). One Triton program per token reads its slots in rank order, accumulates weight × row in fp32, and rounds to bf16 once |
| `fixed_slot_torch` | the same sum written with torch ops, the formula of the single-GPU case; a cross-check of the Triton kernel |
| `rank_partial` | the expert rank sums each token's local experts (fp32, top-k rank order) and returns one fp32 row per (token, rank); the source sums those rows in rank order. Deterministic, like dispatchers that return one row per token per rank |
| `capacity` | `fixed_slot` with a per-expert capacity ⌈1.0 · T_local · k / E⌉ on the source rank; assignments past it (first come, first served in local token order) are dropped |

Notes on the fixed-slot kernel:
- The single-GPU case's combine is a torch-op sum, not a Triton kernel. The
  Triton kernel here is new. It reproduces that formula bitwise
  (`fixed_slot` vs `fixed_slot_torch`: 0 mismatches in every config), because it
  is compiled with `enable_fp_fusion=False` (product and add round separately,
  as in torch) and starts its accumulator at −0.0, so the first add is an exact
  identity.
- Dropped slots (capacity) are skipped, not added as zero.

**Tests.** Equality is bitwise through integer views (`tools/bi_check.py`).
- **L1:** the same 1024-token global batch, contiguously sharded, 5 runs.
- **L2:** a probe group of 2 tokens on rank 0. Reference: the group alone, other
  ranks empty. Then n co-batched tokens per rank, n ∈ {1, 2, 3, 8, 31, 64, 129,
  512}, placed on rank 0 only ("same"), on the other ranks only ("others"), or
  on all ranks ("both"); the probe first or last on rank 0; fillers random,
  ×1000, or zeros (zero rows tie every router score, so they exercise the
  tie-break). 48 checks at EP = 1, 144 at EP ≥ 2.
- **L5:** the same 1024 global tokens at EP = 1, 2, 4, 8 and EP = 2 across
  switches, each compared with EP = 1. Three shardings at every degree:
  contiguous, strided (token i on rank i mod W), and uneven (a shuffled order
  split in proportion 1 : 2 : … : W). Also the probe group alone. For E = 32
  top-4, again with 4096 global tokens.
- **Building blocks** at EP = 1: `check_batch` on the three GEMM shapes and on
  the router, sizes 1–2048, first/middle/last, three fillers.

## Results

Taken through the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md),
recipes 3, 8 and 10.
- Code: [examples/moe_ep/moe_ep.py](../../examples/moe_ep/moe_ep.py), driver
  [run_all.sh](../../examples/moe_ep/run_all.sh). GEMMs:
  `mm_one_config` from [tools/bi_kernels.py](../../tools/bi_kernels.py).
- Results: [results-a16.json](../../examples/moe_ep/results-a16.json).

## Results under expert parallelism

### Building blocks on the A16 (sm86)

`check_batch` at EP = 1: 2 rows, 23 batch sizes from 1 to 2048, first, middle
and last position, three fillers.

| GEMM (K→N) | cuBLAS | `mm_one_config` |
|---|---|---|
| 2048 → 32 (router, E = 32) | fails at M = 2–129 (every size tested in that range) | passes |
| 2048 → 16 (router, E = 16) | fails at M = 2–16, 64–257 and 2048 | passes |
| 2048 → 1408 (expert up and gate) | fails at every M from 2 | passes |
| 1408 → 2048 (expert down) | fails at every M from 2 except 1024 | passes |

The router (softmax over the GEMM output, stable sort, fp32 renormalization)
passed for weights and ids in all three configs. This repeats the A40 finding
on another GPU: cuBLAS is batch-variant at these shapes, and the one-config
GEMM is not.

### Invariance, E = 32, top-4

L1: 5 runs of 1024 tokens. L2: 48 checks at EP = 1, 144 at each EP ≥ 2. L5:
compared with EP = 1 contiguous; "1024" and "4096" are the global batch.

| Variant | L1 | L2 | L5, 1024 tokens | L5, 4096 tokens |
|---|---|---|---|---|
| `arrival_atomic` | **fail** at every EP: 3.7k–9.8k of 2.1M elements change per run | **fail**: 32/48, 67/144, 66/144, 66/144 (EP 1/2/4/8), from 3 tokens per rank, only when rank 0 held other tokens | fail: 3k–33k elements (59–155 tokens), including EP = 1 with the same token order, so this is L1 noise | fail (L1 noise) |
| `arrival_det` | pass | pass | pass | pass |
| **`fixed_slot`** | **pass** | **pass** | **pass**: EP 1/2/4/8, EP 2 across switches, 3 shardings, probe alone | **pass** |
| `fixed_slot_torch` | pass | pass | pass; bitwise equal to `fixed_slot` | pass |
| `rank_partial` | pass | pass | **fail**: 41, 53, 62 elements at EP 2, 4, 8 (of 2.1M); identical at every sharding and on both GPU placements | **fail**: 144, 204, 223 elements (of 8.4M) |
| `capacity` | pass | **fail**: 35/48, 67/144 at every EP ≥ 2; only with co-batched tokens on rank 0; every failure had a changed drop set | **fail**: 186k–570k elements (91–279 tokens), including EP = 1 when only the local order changes | fail |

Router ids were bitwise equal across every EP degree and sharding, in every
config.

### Invariance across top-k

| | top-2 (E = 16) | top-3 (E = 32) | top-4 (E = 32) |
|---|---|---|---|
| `arrival_atomic` L1 | **pass** at every EP | fail: 958–6048 elements per run | fail: 3673–9751 |
| `arrival_atomic` L2 | **pass** (0/48, 0/144) | fail from 8 tokens per rank | fail from 3 tokens per rank |
| `arrival_atomic` L5 | **pass** | fail | fail |
| `rank_partial` L5 (EP 2/4/8) | **pass** | fail: 27, 30, 31 elements | fail: 41, 53, 62 elements |
| `fixed_slot`, `fixed_slot_torch`, `arrival_det`: L1, L2, L5 | pass | pass | pass |
| `capacity` L2, L5 | fail | fail | fail |

### Where `rank_partial` fails

Tokens are grouped by how many ranks their top-4 experts sit on (E = 32,
4096-token batch). Each cell gives mismatched tokens / tokens in the group.

| EP | experts on 1 rank | on 2 ranks | on 3 ranks | on 4 ranks |
|---|---|---|---|---|
| 2 | 0 / 394 | 141 / 3702 | — | — |
| 4 | 0 / 30 | 54 / 1171 | 118 / 2447 | 26 / 448 |
| 8 | 0 / 2 | 16 / 217 | 100 / 1818 | 103 / 2059 |

A token whose experts all live on one rank gets one partial sum, which equals
the fixed-slot sum. Once its experts span ranks, its fp32 sum is grouped by
rank: ((a + b) + c) + d at EP = 1 can become (a + c) + (b + d) at EP = 2. Most of
those fp32 differences vanish in the final bf16 rounding. A few survive: about
4–6% of the tokens that span ranks, mostly in one element each (223 elements in
219 tokens at EP = 8).

### `capacity`: neighbours decide a token's fate

- L2: the probe changed only when co-batched tokens were on its own rank
  ("same" and "both"), at every size from 1 token per rank. Tokens on other
  ranks never changed it, because capacity here is computed from the local
  token count.
- Every L2 failure and every L5 mismatched token had a different set of dropped
  assignments. Tokens that kept the same assignments were bitwise equal.
- The mismatches are large: up to 4090 of the probe group's 4096 elements.

### Accuracy (sanity check, EP = 1)

The reference uses the same routing, fp32 expert GEMMs and an fp32 combine
(E = 32, top-4, 1024 tokens; output mean |y| = 0.256).

| | mean \|Δ\| | max \|Δ\| |
|---|---|---|
| `fixed_slot`, `fixed_slot_torch`, `rank_partial` | 1.08e-3 | 1.16e-2 |
| `arrival_atomic`, `arrival_det` (bf16 products, bf16 output) | 1.32e-3 | 1.75e-2 |
| `capacity` (drops; the reference is dropless) | 1.32e-2 | 1.47 |

### Time per layer (relative)

Each variant's time per layer relative to `arrival_atomic` at the same EP and
tokens per rank. E = 32, top-4. Each cell lists EP = 1 / 2 / 4 / 8. Best of 2
repetitions of 10 forwards, maximum over ranks. The time includes the router,
both all-to-alls, the host syncs for split sizes, a Python loop over local
experts, and the combine.

| tokens per rank | `fixed_slot` | `rank_partial` | `arrival_det` | `fixed_slot_torch` | `capacity` |
|---|---|---|---|---|---|
| 64 | 1.01 / 1.02 / 1.09 / 1.01 | 1.16 / 1.29 / 1.38 / 1.48 | 1.03 / 1.04 / 1.05 / 1.07 | 1.01 / 1.03 / 1.03 / 1.04 | 1.02 / 1.03 / 1.14 / 1.05 |
| 512 | 0.92 / 0.91 / 0.91 / 0.91 | 1.03 / 1.00 / 1.08 / 1.16 | 0.97 / 0.96 / 0.96 / 0.96 | 1.06 / 1.06 / 1.04 / 1.03 | 0.85 / 0.84 / 0.89 / 0.90 |
| 4096 | 0.86 / 0.88 / 0.89 / 0.91 | 0.87 / 0.90 / 0.99 / 1.09 | 0.93 / 0.94 / 0.95 / 0.96 | 1.05 / 1.04 / 1.03 / 1.02 | 0.93 / 0.95 / 0.96 / 0.97 |

- Two full runs gave the same ratios within 0.03 at 512 and 4096 tokens per
  rank, in all three configs. At 64 tokens per rank they moved by up to 0.15
  (for example `fixed_slot` at EP = 4: 0.94, then 1.09), so read that row as
  noise around 1.
- The top-2 and top-3 configs gave the same picture (in the JSON).
- EP scaling of `fixed_slot` at fixed tokens per rank, relative to EP = 1:
  - 64 tokens: 0.63 / 0.45 / 0.46 at EP 2/4/8;
  - 512 tokens: 0.98 / 1.08 / 1.18;
  - 4096 tokens: 1.24 / 1.50 / 1.89.

  With balanced routing each rank does about the same expert work at every
  degree. So the growth at 4096 tokens is likely the all-to-all over PCIe. The
  drop at 64 tokens is likely the per-expert launch loop: 32 experts on one GPU,
  4 per rank at EP = 8. Neither was timed separately.

## Two nodes (EP 16)

The same layer on two hosts: 16 ranks, EP up to 16.
- Code: [moe_ep_2node.py](../../examples/moe_ep/moe_ep_2node.py). It runs
  `moe_ep.py`'s layer, combines and kernels and changes only the EP group.
- Results: [results-a16-2node.json](../../examples/moe_ep/results-a16-2node.json).

**Setup.**
- **Hardware.** Two hosts with 8x A16 each: node 0 holds ranks 0–7, node 1
  ranks 8–15. Between them, NCCL 2.21.5 runs over one 25 GbE link with the
  socket transport (no RDMA, no InfiniBand). Inside a host, PCIe as before.
- **Launch.** One 16-rank launch. Each layout splits the ranks into EP groups.
  The groups run at the same time, each on the full global batch:

| Layout | EP | Groups |
|---|---|---|
| `ep1` | 1 | each of the 16 GPUs alone |
| `ep2` / `ep2_x` | 2 | pairs on one node / one rank on each node |
| `ep4` | 4 | four ranks on one node |
| `ep8_node` / `ep8_x` | 8 | one group per node / ranks 0–3 + 8–11 and 4–7 + 12–15 (4 + 4 across the nodes) |
| `ep16` | 16 | all ranks, both nodes |

- **L5.**
  - Every group ran three shardings (contiguous, strided, uneven). Its output
    was compared bitwise with rank 0's EP = 1 output, which was broadcast to
    all ranks.
  - Rank 0's EP = 1 output was also compared with the EP = 1 dump from the
    single-node runs.
  - Configs: E = 32 top-4 and top-3 on 1024 global tokens; top-4 also on 4096.
- **L1 and L2 at EP = 16** (top-4; L1 also top-3).
  - L1: 1024 tokens, 5 runs.
  - L2: the probe group on rank 0, with 1, 3, 31, 129 or 512 co-batched tokens
    per rank. Same, others or both; first or last; three fillers; 90 checks.

**L5.** Mismatched elements against rank 0's EP = 1 output. Each value was the
same at all three shardings (except `capacity`) and in every group of the
layout.

| Layout | `fixed_slot`, `fixed_slot_torch`, `arrival_det` | `rank_partial`: top-4 / top-3 / top-4 at 4096 tokens | `capacity`, top-4: contiguous / strided / uneven |
|---|---|---|---|
| `ep1` (16 GPUs) | **0** | 0 / 0 / 0 | 0 / 0 / 316,556 |
| `ep2` | **0** | 41 / 27 / 144 | 275,678 / 185,837 / 385,840 |
| `ep2_x` | **0** | 41 / 27 / 144 | the same as `ep2` |
| `ep4` | **0** | 53 / 30 / 204 | 394,039 / 328,628 / 477,682 |
| `ep8_node` | **0** | 62 / 31 / 223 | 563,756 / 494,044 / 569,622 |
| `ep8_x` | **0** | 62 / 31 / 223 | the same as `ep8_node` |
| `ep16` | **0** | 60 / 33 / 221 | 763,549 / 706,286 / 661,771 |

- Rank 0's EP = 1 output was bitwise equal to the single-node EP = 1 dump,
  for all five deterministic variants in all three configs. That dump came
  from GPU 0 of the other host.
- At EP = 2, 4 and 8, every `rank_partial` and `capacity` count equals the
  single-node table exactly.
- `arrival_atomic` differed in every comparison: 2.9k–69k elements at 1024
  tokens, top-4. That is its L1 failure.
- Router ids were equal everywhere.
- `rank_partial` at EP = 16 (top-4, 4096 tokens), mismatched / tokens by the
  number of ranks a token's experts span:
  - 2 ranks: 2 / 12;
  - 3 ranks: 36 / 757;
  - 4 ranks: 178 / 3327.

**L1 and L2 at EP = 16.**

| Variant | L1, top-4 / top-3 | L2, top-4 (90 checks) |
|---|---|---|
| `arrival_atomic` | **fail**, up to 23,069 / 12,247 elements per run | **fail** 43/90: "same" and "both" only, from 3 tokens per rank |
| `arrival_det`, `fixed_slot`, `fixed_slot_torch`, `rank_partial` | pass | pass |
| `capacity` | pass | **fail** 42/90: "same" and "both" only, from 1 token per rank; every failure had a changed drop set |

**Time per layer (relative).** E = 32, top-4.
- Left columns: each variant relative to `arrival_atomic` in the same layout
  and at the same tokens per rank.
- Only one group was timed per layout:
  - EP 8 on one node: node 1;
  - EP 8 split: ranks 0–3 + 8–11;
  - EP 16: all ranks.
- Two timing runs agreed within 0.01 (0.03 in the last column), with one
  exception: `arrival_det` at EP 8 split, 2048 tokens, read 1.28 and then 0.97.
  The table shows the second run.
- Last column: `fixed_slot`'s time relative to EP 8 on one node at the same
  tokens per rank.

| Layout | tokens per rank | `fixed_slot` | `rank_partial` | `arrival_det` | `fixed_slot_torch` | `capacity` | `fixed_slot` vs EP 8, one node |
|---|---|---|---|---|---|---|---|
| EP 8, one node | 512 | 0.92 | 1.18 | 0.96 | 1.03 | 0.91 | 1 |
| | 2048 | 0.91 | 1.09 | 0.96 | 1.03 | 0.94 | 1 |
| EP 8, 4 + 4 across nodes | 512 | 0.95 | 1.27 | 0.98 | 1.02 | 0.93 | 1.99 |
| | 2048 | 0.95 | 1.23 | 0.97 | 1.01 | 0.97 | 1.93 |
| EP 16, two nodes | 512 | 0.98 | 1.39 | 0.99 | 1.01 | 0.95 | 3.43 |
| | 2048 | 0.97 | 1.35 | 0.99 | 1.01 | 0.98 | 3.47 |

EP 16 runs twice the global tokens of EP 8 at the same tokens per rank. The
growth across nodes is likely the all-to-alls over the 25 GbE socket transport.
Communication was not timed separately. The combine's share shrinks
accordingly: `fixed_slot` took 0.97–0.98 of the atomic path at EP 16.
`rank_partial` grew to 1.35–1.39, likely because its extra metadata exchange
also crosses the network.

**What two nodes add.**
1. **Where the GPUs sit did not change a single bit.** `fixed_slot` matched
   EP = 1 at EP 16, at EP 8 split 4 + 4, and at EP 2 split 1 + 1, under three
   shardings, top-3 and top-4, and 1024 and 4096 tokens. EP = 1 also agreed on
   all 16 GPUs of the two hosts. For every deterministic variant, a layout
   inside one node and the same degree split across nodes gave identical
   counts.
2. **`rank_partial`'s error follows the degree, not the hardware.** At each
   degree its count was the same within a node and across nodes. It did not
   grow steadily with the degree: 62 at EP 8, then 60 at EP 16 (top-3: 31,
   then 33).
3. **Capacity behaves as on one node.** The same counts at each degree, and
   growth with the degree, likely because each rank's capacity comes from fewer
   local tokens.

## What the expert-parallel results show

1. **With invariant expert GEMMs and a fixed-slot combine, EP only moves data.**
   `fixed_slot` gave the same bits at EP = 1, 2, 4, 8 and 16:
   - on two GPU placements of EP = 2, within and across PCIe switches;
   - on two hosts: EP 16, EP 8 split 4 + 4, EP 2 split 1 + 1;
   - under three token-to-rank shardings;
   - at top-2, 3 and 4 (top-2 on one host only);
   - with 1024 and 4096 global tokens.

   It also passed L1 and L2 at every degree, EP 16 included, with co-batched
   tokens on the same rank, on other ranks, or both. The dispatch contains no
   reduction, so no NCCL setting was needed, over PCIe or over the network.
   This is L5 for the forward of this layer, on this setup.
2. **Deterministic is not parallel-invariant.** `rank_partial` was L1 and L2 at
   every degree and failed L5 from top-3 on. Its sum is grouped by which experts
   share a rank, and that follows the EP degree. Only tokens whose experts span
   ranks changed. The mismatches were rare (at most 223 of 8.4M elements) but
   not zero. They depend on the degree only: EP 8 inside one host and split
   across two gave the same count. Returning one row per (token, top-k rank)
   and summing in a fixed order removes the dependence; that is the recipe's "fixed slots before
   summing". Here `fixed_slot` was also faster than `rank_partial` at every point
   measured. That is likely because `rank_partial` needs an extra metadata
   exchange and a regrouping on the expert side, and at this scale the
   bandwidth it saves does not pay for them.
3. **Top-2 hides combine-order bugs.** Two terms added onto a zero give the same
   bits in either order: 0 + a is exactly a, and a + b = b + a. The atomic
   combine passed L1, L2 and L5 at top-2, and `rank_partial` passed L5 there.
   Both failed from top-3. A parity test on a top-2 model therefore cannot
   certify a combine for top-k ≥ 3. That likely includes top-2 plus a shared
   expert added in the same sum, which was not tested.
4. **Capacity makes a token's output depend on its neighbours.** With capacity
   computed per source rank, the tokens on the same rank decided which
   assignments survived. A different EP degree, a different sharding, or only a
   different local order changed which tokens were dropped. Every mismatch came
   from a changed drop set, so dropless routing (or a rule that depends only on
   the token) is required for parity.
5. **Deterministic `index_add_` passed everything here, for a reason the code
   does not state.** `arrival_det` sums in the order rows sit in the returned
   buffer. This dispatcher sorts them by (expert, token, top-k rank) at every
   degree. A dispatcher that orders rows by destination rank, or by token,
   would likely give a degree-dependent order. That was not tested. It also
   rounds each product to bf16 and adds into a bf16 output, which made it less
   accurate than the fp32 fixed-slot sum (mean |Δ| 1.32e-3 vs 1.08e-3).
6. **Cost.** The invariant layer was not slower than the common one here. The
   fixed-slot Triton combine took 0.86–0.92 of the time of the atomic
   `index_add_` path at 512 and 4096 tokens per rank, and about the same at 64.
   The torch-op version of the same sum took 1.02–1.06, likely because it writes
   a [T, k, H] fp32 buffer. Across two nodes the network dominates. There
   `fixed_slot` took 0.95–0.98 of the atomic path, while `rank_partial` rose to
   1.23–1.39.

## Limits

- **Forward only.** The backward under EP was not tested:
  - the expert weight gradients sum over all rows routed to an expert, from
    every rank;
  - the input gradient sums a token's k returned gradients, which is the
    fixed-slot question in reverse.
- **One or two hosts:** 8x A16 each. PCIe inside a host, one 25 GbE link with
  NCCL's socket transport between hosts. NCCL `all_to_all_single` only. Not
  tested:
  - NVLink;
  - RDMA or InfiniBand, and more than two nodes;
  - top-2 across nodes;
  - fused dispatch and combine kernels (e.g. DeepEP);
  - overlap of communication with compute;
  - per-rank token deduplication in dispatch;
  - CUDA graphs and `torch.compile`.
- **Not tested at all:**
  - a shared expert;
  - sigmoid or group-limited routers;
  - L4 against a trainer's MoE.
- **Capacity:** one rule only: per source rank, from the local token count,
  first come first served, factor 1.0. A capacity applied on the expert rank
  over arrivals from all ranks was not tested; it would likely make other
  ranks' tokens matter too. A token whose every slot is dropped gets −0.0 from
  the fixed-slot kernel.
- **Ties:** zero fillers exercised exact ties (all router scores equal), and
  the stable sort chose experts 0..k−1 every time. Near-ties between distinct
  values were not targeted.
- **Sizes:**
  - hidden 2048, FFN 1408;
  - E = 16 or 32, top-2 to 4;
  - L5 on 1024 and 4096 global tokens;
  - L2 up to 512 co-batched tokens per rank.
- **Untuned:**
  - `mm_one_config` uses 64 × 64 × 32 tiles;
  - the combine uses one program per token with 1024-wide chunks of the hidden
    dimension;
  - timings are relative only and include host syncs.

## How it was built

An agent wrote `moe_ep.py` and its harness in one pass. It reused the
single-GPU case's structure and `tools/`.
- **Fixed-slot kernel.** The single-GPU case's combine is a torch-op sum, so
  the per-token Triton kernel is new. It was checked bitwise against that sum.
- **Smoke runs (2).**
  1. The first stopped on a `torch.sort` argument error: `stable=True` needs
     `dim` as a keyword. It was fixed.
  2. The second ran EP = 2 L1 and EP = 1 L5. It confirmed that `fixed_slot`
     equals `fixed_slot_torch`, and that `rank_partial` equals `fixed_slot` at
     EP = 1.
- **Full run 1.** Top-4 (E = 32) and top-2 (E = 16), every test above. It found
  the top-2 passes of the atomic and per-rank combines.
- **Full run 2.** It added top-3, to test the commutativity explanation, and a
  4096-token L5 batch.
  - It reproduced run 1's mismatch counts for every deterministic variant
    exactly.
  - Timing ratios agreed within 0.03 at 512 and 4096 tokens per rank.
- **Two nodes.** A second agent pass added an optional EP-group argument to
  `moe_ep.py`; the default keeps the single-node behaviour. It also wrote
  `moe_ep_2node.py`. There were four 16-rank launches, about 1 minute each:
  1. L5 for top-4.
  2. L5 again for top-4, which reproduced the first launch exactly, plus top-3,
     4096 tokens, L1 and L2.
  3. Timing.
  4. A timing repeat, which showed the one outlier above.

  Nothing failed. The work tree was copied to the second host without the
  single-node dumps. Only the EP = 1 dumps needed for the link went along.

Human edits: none.
