# Collectives and tensor-parallel linear layers: lab report

This is the full record behind the handbook page [Collectives](../../handbook/3-kernels/collectives.md). The page teaches the step and what to do; this report keeps the setup, every table, every number, the limits and how the case was built.

## Summary

- Pinning Ring/Simple does not give L2: 15 different results across 46 bf16
  messages. `NCCL_ALGO=Tree` makes all-reduce batch-invariant on one and two
  nodes, at 1.0–1.3x (5.3x at 1 MB).
- Reduce-scatter is fixed per owner, which breaks sequence parallelism.
- **Fix without NCCL settings:** all-gather or all-to-all plus a local fp32 sum
  in rank order. The all-to-all form costs 1.04–1.20x from 1 MB on; across two
  nodes a two-level fixed order costs 1.06–1.8x.
- L5 needs a reduction that does not depend on the degree: a logical TP size
  gives one result at TP 1–8 (logical TP 16 at TP 1–16 on two nodes), at
  1.6–1.9x row-parallel at TP 8. Measured on 2x8 A16.

## How

A tensor-parallel (TP) linear layer adds partial products from several GPUs
with a collective. This case asks two questions, first on one 8-GPU node, then
across [two nodes](#two-nodes-16-ranks):
- Does a row's result depend on the batch it runs in (L2)?
- Does it depend on the TP degree (L5)?
### Where the variance comes from

A sum all-reduce adds each element over the ranks inside the library, one add
at a time, rounding to the dtype after every add. NCCL, not the caller,
chooses the order of those adds:
- **Ring.** The message is cut into chunks. Each chunk's sum starts at a
  different rank and walks the ring. Which chunk an element falls in depends on
  its offset and on the message size. In TP the message is the whole batch's
  output, so a row's order can change when the batch grows or when the row
  moves.
- **Tree, on one node:** one chain, 7→6→…→0, the same for every element.
- **Default selection:** NCCL picks algorithm and protocol per call from the
  message size. Here it always picked Ring: LL up to 128 KB, Simple from
  192 KB (the next size tested).
- **Ring reduce-scatter:** the output chunk that rank j keeps is summed
  starting at rank j+1. The order depends on which rank owns a row, not on the
  message size.
- **Two ranks:** x0 + x1 equals x1 + x0 exactly, so the order cannot matter.
- **TP degree.** A row-parallel layer rounds each rank's partial product to bf16,
  then sums the partials across ranks. Another degree gives other partials and
  another sum. L5 therefore cannot come from configuration alone.

The ring, chain and owner orders above were observed (see the order diagnostic
below), not taken from NCCL's source.

### What was done

**Collectives**, 8 ranks, sum, bf16 and fp32. Each rank holds a fixed slice:
4 rows of width 4096, and separately 512 elements. It is reduced alone, then
inside larger messages, at the first, middle and last position. Every result is
compared bitwise with the slice reduced alone.

| Test | Sizes |
|---|---|
| all-reduce, 4×4096 slice | messages of 4–32768 rows in bf16 (32 KB–256 MB), 4–16384 rows in fp32 (64 KB–256 MB) |
| all-reduce, 512-element slice | bf16 1 KB–2 MB, fp32 2 KB–4 MB |
| reduce-scatter | the slice at a fixed place in every rank's output chunk; chunks of 4–4096 rows in bf16 (256 KB–256 MB in total), 4–2048 in fp32 (512 KB–256 MB) |
| fallback A | all-gather the per-rank tensors, sum in rank order 0..7 in fp32, round once |
| fallback B | all-to-all, the same fp32 rank-order sum on each rank's piece, all-gather; its reduce-scatter half is the reduce-scatter fallback |
| order diagnostic | every result element is compared with the 16 sequential orders (start rank × ring direction), rounding after every add |
| L1 | 5 identical calls: all-reduce at 3 sizes, reduce-scatter at 2, per dtype |
| NCCL settings | default; Ring/Simple; Ring/LL; Ring/LL128; Tree; Tree/Simple; Tree/LL (one launch each) |

NCCL's algorithm and protocol per call were read from its TUNING log on rank 0
(`<bytes> Bytes -> Algo <a> proto <p>`). Channel counts were not logged.

**TP linear** y = x @ W, K = N = 4096, bf16 inputs.

| Scheme | Each rank computes | Then |
|---|---|---|
| row-parallel | `mm_one_config(x[:, K_r], W[K_r, :])`, bf16 output | NCCL all-reduce in bf16 |
| row-parallel + sequence parallel (SP) | the same | NCCL reduce-scatter over tokens (rows padded to a multiple of TP), all-gathered for the check |
| row-parallel + fallback | the same | fallback B |
| column-parallel (control) | `mm_one_config(x, W[:, N_r])` | all-gather (a copy) |
| logical TP 8 | its share of 8 fixed K chunks of 512, with the fp32-output GEMM | partials are moved, never reduced in the network; fp32 sum in a fixed order; one rounding to bf16 |

Logical TP 8 comes in three forms:
- `lt8_gather`: all-gather every partial, then sum in chunk order 0..7 on every rank.
- `lt8_a2a`: all-to-all by column block, sum in chunk order 0..7, all-gather the bf16 result.
- `lt8_tree`: a fixed pairwise tree, ((0+1)+(2+3))+((4+5)+(6+7)). A rank's
  chunks form one subtree, so it sums them locally and sends one fp32 partial.
  It finishes the tree after an all-to-all by column block, then all-gathers.

TP = 1 runs the same 8 chunk GEMMs and the same sum on one GPU. The fp32-output
GEMM, rounded to bf16, equals `mm_one_config` bitwise at M = 1, 65, 257 and
4096.

TP tests:
- **L2:** `check_batch` with 2 probe rows; M = 1, 2, 3, 4, 7, 8, 15, 16, 31, 32,
  63, 64, 65, 127, 128, 129, 255, 256, 257, 512, 1024, 2048, 4096; first,
  middle and last; random, zero and large fillers. That is 398 comparisons per
  scheme.
- **Groups:** TP 2 and 4 are ranks 0–1 and 0–3, inside one PCIe switch. TP 8
  spans both switches.
- **L1:** the same M = 4096 batch, 5 runs.
- **L5:** M = 1, 256 and 4096 at TP 1, 2, 4 and 8.
- **NCCL settings:** default, Ring/Simple and Tree.

## Evidence

### Setup, code and results

Taken through the [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md),
recipe 10.
- Code: [examples/collectives_tp/collectives_tp.py](../../examples/collectives_tp/collectives_tp.py),
  with drivers [run_all.sh](../../examples/collectives_tp/run_all.sh) (one node)
  and [run_2node.sh](../../examples/collectives_tp/run_2node.sh) (two nodes).
  The GEMM is `mm_one_config` from [tools/bi_kernels.py](../../tools/bi_kernels.py).
  An fp32-output copy of it lives in the script.
- Results: [results-a16.json](../../examples/collectives_tp/results-a16.json)
  (one node) and
  [results-a16-2node.json](../../examples/collectives_tp/results-a16-2node.json)
  (two nodes).
- Setup: one host, 8x NVIDIA A16 (sm86, 15 GB), PCIe only. GPUs 0–3 sit behind
  one PCIe switch and 4–7 behind another; the two connect through the host
  bridge. No NVLink. torch 2.5.1+cu121, Triton 3.1.0, NCCL 2.21.5 (bundled with
  torch).
- For the two-node section, a second identical host joins over 25 GbE: TCP
  sockets, no InfiniBand or RDMA.
- NCCL's own view on one node, from its INIT log:
  - 2 channels, both on the ring 0→1→…→7;
  - in both channels, a tree that is one chain rooted at rank 0;
  - P2P (CUMEM) inside a switch, shared memory across the two switches.

### Results

#### Collectives, 8 ranks

The table counts failing calls: calls where the slice's bits differ from the
slice reduced alone. In parentheses: how many distinct bit patterns the same
slice produced.

| NCCL setting | all-reduce used | all-reduce bf16, 4×4096 / 512 | all-reduce fp32, 4×4096 / 512 | reduce-scatter |
|---|---|---|---|---|
| default | Ring/LL ≤ 128 KB, Ring/Simple ≥ 192 KB | 43 of 46 fail (13) / 24 of 28 (11) | 42 of 43 (16) / 25 of 28 (11) | pass, Ring/Simple |
| Ring/Simple | Ring/Simple | 45 of 46 (15) / 12 of 28 (8) | 42 of 43 (16) / 14 of 28 (8) | pass |
| Ring/LL | Ring/LL | 43 of 46 (13) / 24 of 28 (11) | 42 of 43 (14) / 25 of 28 (11) | pass |
| Ring/LL128 | Ring/LL128 | 45 of 46 (26) / 8 of 28 (4) | 42 of 43 (27) / 10 of 28 (6) | pass |
| **Tree** | Tree/LL ≤ 128 KB, Tree/Simple ≥ 192 KB | **pass (1) / pass (1)** | **pass (1) / pass (1)** | pass; NCCL stayed on Ring/LL128 and Ring/Simple |
| Tree/Simple | Tree/Simple | pass (1) / pass (1) | pass (1) / pass (1) | pass; Ring/LL128 and Ring/Simple |
| Tree/LL | Tree/LL | pass (1) / pass (1) | pass (1) / pass (1) | pass; Ring/LL128 and Ring/Simple |
| fallback A and B | — | pass: 1 result per slice across all 7 settings | pass (1) | all-to-all form: pass |

- **Size of the differences.** Failing all-reduce calls changed up to 10011 of
  16384 elements in bf16 (max abs difference 0.125) and up to 9100 in fp32 (max
  abs difference 1.9e-6).
- **Order diagnostic.** Every element of every all-reduce and reduce-scatter
  result, in every setting, equalled one of the 16 sequential orders. No
  element was left unexplained.
  - Default setting, slice alone: its 16384 elements used 8 different start
    ranks.
  - Default setting, bf16 messages from 512 KB on: the slice used start rank 1
    at the first position, start rank 0 at the last, and a mix of 0 and 1 in
    the middle.
  - The pinned ring settings used other mixes of start ranks.
  - Tree: the chain 7→…→0 for every element, at every size and position.
- **Reduce-scatter.** The chunk kept by rank j was always summed from rank j+1,
  so it passed at every size. But the same slice kept by rank j ≠ 0 differed
  from the one kept by rank 0 in 9340–10001 of 16384 elements (fp32, largest
  size). The all-to-all fallback gave the same bits at every owner.
- **NCCL_ALGO=Tree does not reach reduce-scatter.** NCCL used Ring for it and
  also ignored the NCCL_PROTO pin. It picked LL128 and Simple, and LL128 is a
  protocol the default setting never chose.
- **Agreement.** In every all-reduce, all 8 ranks received identical bits. The
  fallbacks equalled an fp32 rank-order sum computed on one GPU.
- **L1** passed in every setting: 5 runs, every rank.

#### TP linear: L2 and L1

Each cell gives failing comparisons out of 398, with the number of mismatched
elements per failing row (of 4096).

| Scheme | TP | default | Ring/Simple | Tree |
|---|---|---|---|---|
| unsharded `mm_one_config` | 1 | pass | pass | pass |
| row-parallel | 2 | pass | pass | pass |
| row-parallel | 4 | fail 396 (1276–1781) | fail 378 (831–1725) | **pass** |
| row-parallel | 8 | fail 396 (2087–2466) | fail 354 (1158–2495) | **pass** |
| row-parallel + SP | 4 | fail 264 (1649–1746): middle and last only | the same | the same |
| row-parallel + SP | 8 | fail 264 (2325–2504): middle and last only | the same | the same |
| row-parallel + fallback | 4, 8 | pass | pass | pass |
| column-parallel | 8 | pass | pass | pass |
| `lt8_gather` | 1, 2, 8 | pass | pass | pass |
| `lt8_a2a` | 8 | pass | pass | pass |
| `lt8_tree` | 1, 2, 4, 8 | pass | pass | pass |

- **Default fails from M = 2, at every position.** The M = 1 reference is an
  8 KB message; M = 2 is already 16 KB. Under Ring/Simple a few cases passed,
  e.g. M = 2 at the first position.
- **SP passes only at the first position.** There the probe row stays in the
  chunk owned by rank 0. At the middle and last positions it moves to another
  owner. Tree does not help, because reduce-scatter stays on Ring.
- **L1** passed for row-parallel at TP 4 and 8, and for the three logical-TP
  forms at TP 8, under all three settings.

#### L5: the same rows at TP 1, 2, 4 and 8

The table gives elements that differ from the TP 1 result, at M = 4096 (16.8M
elements), default NCCL. M = 1 and 256 gave fractions within about 1 point of
these.

| Scheme | compared with | TP 2 | TP 4 | TP 8 |
|---|---|---|---|---|
| row-parallel | unsharded | 37.5% | 50.0% | 59.3% |
| row-parallel + fallback | unsharded | 37.5% | 40.2% | 41.4% |
| column-parallel | unsharded | 0 | 0 | 0 |
| `lt8_gather`, `lt8_a2a` | `lt8_gather` at TP 1 | 0 | 0 | 0 |
| `lt8_tree` | `lt8_tree` at TP 1 | 0 | 0 | 0 |

**Accuracy** against an fp64 reference, M = 256. The table gives relative
Frobenius error, with max absolute error in parentheses.

| | TP 1 | TP 2 | TP 4 | TP 8 |
|---|---|---|---|---|
| unsharded, column-parallel | 1.66e-3 (1.6e-2) | 1.66e-3 (1.6e-2) | 1.66e-3 (1.6e-2) | 1.66e-3 (1.6e-2) |
| row-parallel | — | 2.45e-3 (2.6e-2) | 3.15e-3 (2.9e-2) | 4.04e-3 (4.3e-2) |
| row-parallel + fallback | — | 2.45e-3 (2.6e-2) | 2.37e-3 (2.0e-2) | 2.35e-3 (2.1e-2) |
| logical TP 8, all forms | 1.66e-3 (1.6e-2) | 1.66e-3 (1.6e-2) | 1.66e-3 (1.6e-2) | 1.66e-3 (1.6e-2) |

- **Logical TP 8 is not the plain GEMM.** It differs from the unsharded
  `mm_one_config` in 0.25% of elements: 41728 of 16.8M with the sequential
  order, 41735 with the tree. It is another fixed arithmetic, with the same
  error to three digits.
- **Row-parallel at TP 4 and 8 also depends on the NCCL setting.** At M = 4096,
  8394165 and 9942833 elements differ from unsharded under Tree, against
  8393899 and 9944317 under Ring.
- **At TP 2, row-parallel and its fallback gave the same bits.** For two bf16
  values, the fp32 sum rounded once equals their bf16 sum.

#### Cost

Timing is wall-clock time over 5–20 calls, synchronized, after 2 warm-up
calls, with NCCL logging off. It is reported per call.

**Collectives, 8 ranks**, ms per call. "All-reduce" is NCCL's default choice;
"Tree" is the NCCL_ALGO=Tree launch. In parentheses: the ratio to all-reduce.

| Message (bf16) | all-reduce | Tree | reduce-scatter | fallback A: all-gather + sum | fallback B: all-to-all + sum + all-gather |
|---|---|---|---|---|---|
| 32 KB | 0.058 | 0.058 (1.0x) | 0.049 | 0.139 (2.4x) | 0.196 (3.4x) |
| 1 MB | 0.482 | 2.54 (5.3x) | 0.236 | 2.16 (4.5x) | 0.581 (1.20x) |
| 16 MB | 7.31 | 9.47 (1.29x) | 3.60 | 34.4 (4.7x) | 7.99 (1.09x) |
| 256 MB | 117 | 127 (1.09x) | 56.3 | 545 (4.7x) | 128 (1.10x) |

- **fp32** gave the same picture. From 1 MB on, fallback B cost 1.04–1.07x and
  fallback A 3.7–4.4x; Tree cost 4.8x at 1 MB.
- **Bandwidth.** At 256 MB, all-reduce reached 2.3 GB/s of algorithm bandwidth
  and reduce-scatter 4.8 GB/s.
- **Pinned Ring/Simple** cost 1.7x at 32 KB (0.099 ms), where the default used
  LL. From 1 MB on it matched the default.

**TP linear**, ms per call, default NCCL, inputs already sharded:

| M | unsharded, 1 GPU | row-parallel TP 2 / 4 / 8 | row + fallback B, TP 8 | column TP 8 | `lt8_tree` TP 1 / 2 / 4 / 8 | `lt8_a2a` TP 8 | `lt8_gather` TP 8 |
|---|---|---|---|---|---|---|---|
| 16 | 0.286 | 0.202 / 0.160 / 0.153 | 0.281 | 0.129 | 0.364 / 0.376 / 0.391 / 0.294 | 0.513 | 0.542 |
| 512 | 1.29 | 1.45 / 1.60 / 1.93 | 2.15 | 1.18 | 2.57 / 2.66 / 3.02 / 3.23 | 3.17 | 16.2 |
| 4096 | 15.3 | 13.8 / 13.8 / 16.3 | 17.9 | 9.97 | 26.9 / 24.1 / 24.8 / 26.6 | 26.6 | 129 |

- **Row-parallel under Tree, TP 8:** 0.177, 3.89 and 18.8 ms, i.e. 1.16x, 2.0x
  and 1.15x the default.
- **`lt8_tree` at TP 8** costs 1.9x, 1.7x and 1.6x row-parallel. At TP 1 it
  costs 1.3x, 2.0x and 1.8x the unsharded GEMM. Its time barely changes with TP.
- **`lt8_gather` at TP 8** costs 3.5x, 8.4x and 7.9x row-parallel. At M = 4096
  its TP 2 and 4 cost 70.5 and 97.0 ms, and `lt8_a2a` costs 44.6 and 37.1 ms.

### What this shows

1. **Pinning the ring algorithm and protocol does not give L2.** With
   Ring/Simple pinned, the slice gave 15 different results across 46 bf16
   messages. The order diagnostic shows why: each element is summed from a
   start rank set by its chunk, and chunk boundaries move with message size and
   offset. Recipe 10 asks to fix "the chunking" too. No setting tested here did
   that. In a TP layer this fails from M = 2.
2. **On this node, `NCCL_ALGO=Tree` makes all-reduce batch-invariant.** That is
   strategy 1, configure, in the skill. Every element took the same chain at
   every size, position and protocol, and row-parallel passed L2 at TP 4 and 8.
   - It costs 1.0–1.3x at 32 KB, 16 MB and 256 MB, but 5.3x at 1 MB (bf16),
     and 2.0x for the TP 8 layer at M = 512.
   - It holds because one node gives one chain in both channels. It still held
     on [two nodes](#two-nodes-16-ranks); three or more were not tested.
   - It does not cover reduce-scatter.
3. **Reduce-scatter is fixed per owner, which is not enough.** Each output
   chunk has one order, so the collective passes alone. But the order depends
   on which rank owns the row, and in sequence parallelism that depends on the
   row's position and the batch. Row-parallel with SP failed at the middle and
   last positions under every setting.
4. **The fallback works, and its all-to-all form is cheap.**
   - All-gather plus a local fp32 sum in rank order gave one result per slice
     across 7 NCCL settings and every size. It costs 3.7–4.7x all-reduce from
     1 MB on.
   - The all-to-all form gives the same bits for 1.04–1.20x from 1 MB on. It
     likely costs little because it moves about the same bytes as a ring.
   - Across [two nodes](#two-nodes-16-ranks) the flat forms cost 2.1–8.8x. A
     two-level fixed order brings this down to 1.06–1.8x.
   - Both forms change the arithmetic (fp32 accumulation, one rounding), so
     both engines must use the same one for L4.
5. **Two ranks are order-free.** Row-parallel at TP 2 passed L2 under every
   setting, because a two-operand sum commutes.
6. **L5 needs a reduction that does not depend on the degree.**
   - Row-parallel differs from the unsharded GEMM in 37.5%, 50% and 59% of
     elements at TP 2, 4 and 8. Its error grows with TP (1.66e-3 to 4.04e-3).
     With the fallback, it is batch-invariant but still TP-dependent.
   - Logical TP 8 gave one result at TP 1, 2, 4 and 8, in all three forms, with
     the unsharded GEMM's accuracy.
   - The tree form costs 1.6–1.9x row-parallel at TP 8. A trainer that must
     match it runs the same 8 chunks.
   - Across [two nodes](#two-nodes-16-ranks), logical TP 16 gave one result at
     TP 1–16. Logical TP 8 reaches TP 16 only by splitting N.

### Two nodes (16 ranks)

**Setup.**
- Two identical hosts: node 0 holds ranks 0–7, node 1 ranks 8–15. NCCL data
  crosses a 25 GbE link through NCCL's socket transport (NET/Socket, no RDMA).
- One torchrun launch per NCCL setting, through a lock shared with another job
  on the same 16 GPUs.
- The same script with `--xnode`:
  - Messages are capped at 64 MB. The link is slow: all-reduce reached
    1.5 GB/s of algorithm bandwidth at 64 MB.
  - The all-gather fallback is checked up to 16 MB in the sweep, since each
    rank receives 16 times the message.
  - Logical TP 16 (16 K-chunks of 256) replaces logical TP 8.
  - L2 uses random fillers only: 134 comparisons per scheme.

**NCCL's view**, from its INIT log:
- 2 channels, both on the ring 0→1→…→15, which crosses nodes at 7→8 and 15→0.
- Trees: each node is a chain into its first rank (7→…→0 and 15→…→8). Channel
  0 has root 0 with rank 8 as its remote child; channel 1 has root 8 with rank
  0 as its remote child.

**Additions.**
- **Order diagnostic.** It also tries two-level tree orders: each node's chain
  sum, combined at the root in each plausible order.
- **Two-level fallback.**
  1. Inside each node: all-to-all, then an fp32 sum over the node's 8 ranks in
     rank order.
  2. Across nodes: one fp32 exchange with the rank of the same local index on
     the other node, then node 0 + node 1, rounded once.
  3. Inside each node: all-gather.

  It is a fixed order like the flat fallback, but each rank sends only 1/8 of
  the message's elements, as fp32, across the link.
- **`lt16_treeh`.** `lt16_tree` routed the same way. A node's 8 chunks form one
  subtree of the pairwise tree, so it computes the same sums.
- **The 8-chunk scheme at TP 16**, two ways. Each chunk goes to 2 ranks:
  - `lt8_2d`: the two ranks split the chunk's N columns;
  - `lt8_splitk`: the two ranks split its K range and add their two fp32
    partials.

#### Collectives, 16 ranks

Same test as on one node, at these sizes:
- 4×4096 slice in messages of 4–8192 rows (bf16, 32 KB–64 MB) or 4–4096 rows
  (fp32, 64 KB–64 MB);
- 512-element slice in 1 KB–2 MB (bf16) or 2 KB–4 MB (fp32);
- reduce-scatter chunks of 4–256 rows (bf16, 512 KB–32 MB in total; fp32,
  1–64 MB).

| NCCL setting | all-reduce used | all-reduce bf16, 4×4096 / 512 | all-reduce fp32, 4×4096 / 512 | reduce-scatter |
|---|---|---|---|---|
| default | Tree/LL ≤ 4 KB, Ring/LL 8–256 KB, Ring/Simple ≥ 512 KB | 39 of 40 fail (16) / 18 of 28 (8) | 34 of 37 (16) / 21 of 28 (10) | pass, Ring/Simple |
| Ring/Simple | Ring/Simple | 39 of 40 (17) / 12 of 28 (8) | 36 of 37 (18) / 14 of 28 (10) | pass |
| Ring/LL | Ring/LL | 39 of 40 (16) / 23 of 28 (13) | 34 of 37 (14) / 24 of 28 (13) | pass |
| Ring/LL128 | Ring/LL128 | 39 of 40 (25) / 8 of 28 (5) | 36 of 37 (25) / 10 of 28 (6) | pass |
| **Tree** | Tree/LL ≤ 128 KB, Tree/Simple ≥ 192 KB | **pass (1) / pass (1)** | **pass (1) / pass (1)** | pass; NCCL stayed on Ring/LL128 and Ring/Simple |
| Tree/Simple, Tree/LL | the pinned pair | pass (1) / pass (1) | pass (1) / pass (1) | pass; Ring |
| fallbacks: flat all-gather, flat all-to-all, two-level | — | pass: one result per slice and form across all 7 settings | pass | flat all-to-all form: pass |

- **Size of the differences.** Failing all-reduce calls changed up to 11759 of
  16384 elements in bf16 (max abs difference 0.25) and up to 11844 in fp32 (max
  abs difference 3.8e-6).
- **Order diagnostic.** No element was left unexplained, in any setting.
  - Under defaults, the Tree/LL calls were explained by one tree order. The
    Ring calls needed 1 to 17 orders per slice, nearly all of them ring
    rotations.
- **Pinned Tree.** In every setting, every element of all 133 calls equalled
  C(7..0) + C(15..8): each node's chain sum, then one add of the two node sums.
  - The two channels have different roots. But with two nodes the top level
    adds only two operands, which commutes, so the channel does not matter.
  - With three or more nodes, NCCL's trees differ by more than a swap, so the
    order would likely depend on the channel again (inferred, not tested).
- **Reduce-scatter.** The chunk kept by rank j was always summed from rank j+1
  around the 16-rank ring. Owners 1–15 differed from owner 0 in 10694–11933 of
  16384 elements (fp32, 64 MB in total).
- **Agreement.** In every all-reduce, all 16 ranks received identical bits. L1
  passed in every setting.

#### TP linear, 16 ranks

TP 8 is ranks 0–7, on one node; TP 16 spans both nodes. The table gives
failing comparisons out of 134, with mismatched elements per failing row (of
4096) in parentheses.

| Scheme | TP | default | Tree |
|---|---|---|---|
| unsharded `mm_one_config` | 1 | pass | pass |
| row-parallel | 8 | fail 132 (2087–2466) | **pass** |
| row-parallel | 16 | fail 132 (2177–2953) | **pass** |
| row-parallel + SP | 16 | fail 88 (2638–2977): middle and last only | the same |
| row-parallel + flat fallback | 16 | pass | pass |
| row-parallel + two-level fallback | 16 | pass | pass |
| column-parallel | 16 | pass | pass |
| `lt16_tree` | 1, 8, 16 | pass | pass |
| `lt16_treeh` | 16 | pass | pass |

L1 (M = 4096, 5 runs) passed for row-parallel, `lt16_tree` and `lt16_treeh` at
TP 16 under both settings.

**L5** at M = 4096 (16.8M elements):

| Scheme | compared with | TP 2 | TP 4 | TP 8 | TP 16 |
|---|---|---|---|---|---|
| row-parallel, default NCCL | unsharded | 37.5% | 50.0% | 59.3% | 67.4% |
| row-parallel, Tree | unsharded | 37.5% | 50.0% | 59.3% | 63.3% |
| row-parallel + fallback, flat or two-level | unsharded | — | — | — | 42.0% |
| `lt16_tree` (and `lt16_treeh` at TP 16) | `lt16_tree` at TP 1 | 0 | 0 | 0 | 0 |
| `lt8_2d`: 8 chunks, N split | `lt8_tree` at TP 1 | — | — | — | 0 |
| `lt8_splitk`: 8 chunks, K split | `lt8_tree` at TP 1 | — | — | — | 5358 elements (0.03%) |

- **Splitting K breaks the bits; splitting N does not.** `lt8_splitk` differed
  in 0, 349 and 5358 elements at M = 1, 256 and 4096. For chunk 0 of 8 at
  M = 4096, splitting K in two changed 14.9M of 16.8M fp32 partial elements
  (89%); splitting N changed none.
- **The chunk count is part of the arithmetic.** Logical TP 16 and logical TP 8
  are different arithmetic. Each differs from the plain GEMM in 0.25% of
  elements, but in different elements: 2657 vs 2651 at M = 256.
- **The two fallbacks agreed in count.** The flat and two-level fallbacks gave
  the same mismatch counts at every M. For bf16 partials, an fp32 sum of 16
  terms is often exact, so they likely give the same bits (not compared
  directly).
- **Accuracy** at M = 256, relative Frobenius error:
  - row-parallel TP 16: 5.30e-3 (default), 4.43e-3 (Tree);
  - with a fallback: 2.35e-3;
  - every logical-TP form: 1.66e-3, the same as the unsharded GEMM.

#### Cost, 16 ranks

Timing used the default NCCL setting, except the Tree column. **Collectives**,
bf16, ms per call (ratio to all-reduce):

| Message | all-reduce | Tree | reduce-scatter | flat: all-gather + sum | flat: all-to-all + sum + all-gather | two-level |
|---|---|---|---|---|---|---|
| 1 MB | 1.46 | 3.00 (2.1x) | 0.51 | 5.49 (3.8x) | 3.07 (2.1x) | 1.55 (1.06x) |
| 16 MB | 11.1 | 11.8 (1.06x) | 5.55 | 95.6 (8.6x) | 31.8 (2.9x) | 20.2 (1.81x) |
| 64 MB | 44.1 | 41.7 (0.94x) | 22.0 | 387 (8.8x) | 189 (4.3x) | 79.2 (1.79x) |

Cross-node all-to-all timing was unstable. At 16 MB, the flat all-to-all
fallback took 102 and 164 ms in two quick launches, against 32 and 34 ms in the
two full launches.

**TP linear**, ms per call, inputs already sharded:

| M | unsharded | row TP 8 | row TP 16 | row + two-level TP 16 | row + flat TP 16 | column TP 16 | `lt16_tree` TP 1 / 8 / 16 | `lt16_treeh` TP 16 | `lt8_2d` TP 16 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.286 | 0.159 | 0.775 | 0.732 | 1.84 | 0.309 | 0.457 / 0.504 / 2.06 | 0.692 | 1.14 |
| 512 | 1.29 | 1.94 | 3.34 | 5.39 | 8.46 | 1.82 | 3.97 / 3.49 / 14.4 | 6.47 | 22.9 |
| 4096 | 15.3 | 16.4 | 22.3 | 41.0 | 65.7 | 11.9 | 39.4 / 28.4 / 111 | 49.6 | 193 |

- **Row-parallel TP 16 under Tree:** 0.524, 5.22 and 22.2 ms, i.e. 0.68x,
  1.57x and 1.00x the default.
- **`lt16_treeh`** costs 0.89x, 1.94x and 2.23x row-parallel at TP 16. The
  two-level row fallback costs 0.94x, 1.61x and 1.84x.
- **`lt8_2d`** all-gathers every fp32 partial, which is slow here. A two-level
  version, like `lt16_treeh`, was not written.
- **The link dominates.** At every M tested, the layer was slower at TP 16
  than at TP 8.

#### What the second node changes

1. **Ring still fails, pinned or not.** The mechanism is the same: the start
   rank follows the element's chunk. With 16 terms per sum, more elements
   differ: up to 72%.
2. **Pinned Tree is still batch-invariant across two nodes.** Every element
   took one two-level order at every size, position and protocol, and
   row-parallel passed L2 at TP 16.
   - It costs 2.1x all-reduce at 1 MB, 0.9–1.1x at 16 and 64 MB, and 1.57x for
     the TP 16 layer at M = 512.
   - It still does not reach reduce-scatter.
   - Three or more nodes were not tested, and likely break it (finding above).
3. **The default now mixes algorithms.** It used Tree up to 4 KB and Ring from
   8 KB, so a call's order depends on its size in a second way. Every TP 16
   message here (8 KB and up, from M = 1) falls in the Ring range.
4. **A fixed order still fixes L2. Across the link, its routing decides the
   cost.**
   - The flat fallbacks cost 2.1–8.8x all-reduce.
   - The two-level fixed order costs 1.06–1.8x and is just as invariant.
5. **L5 with a logical TP size reaches across nodes.**
   - Logical TP 16 gave one result at TP 1, 2, 4, 8 and 16.
   - A logical size C covers degrees that divide C by giving each rank whole
     chunks. It covers degrees above C by splitting each chunk's N columns,
     which is bitwise exact, never its K range.
   - Logical TP 8 and 16 differ from each other, so a deployment has to fix
     one C everywhere.
   - Cost at TP 16 with the two-level route: about 2x row-parallel from
     M = 512 on, and 0.89x at M = 16.

## Limits

- One node (8 A16s over PCIe) and two nodes over 25 GbE sockets, NCCL 2.21.5,
  2 channels on one ring order.
  - The Tree result rests on a chain per node and, with two nodes, a top level
    that adds two operands.
  - Not tested: three or more nodes, InfiniBand/RDMA, NVLink, NVLS and CollNet.
- NCCL's choice was logged at 8 and 16 ranks only, not for the 2- and 4-rank TP
  groups.
  Channel counts were not recorded or pinned (`NCCL_MIN_NCHANNELS` /
  `NCCL_MAX_NCHANNELS`).
- Only sum, bf16 and fp32, and in-place all-reduce.
- TP: K = N = 4096 only (8192 not run), one weight, `mm_one_config` partials
  only. Row-parallel with cuBLAS partials was not tested. Two probe rows, M up
  to 4096.
- Logical TP C covers degrees that divide C. Above C, the N-split was tested
  only at TP 16 with C = 8. Logical TP is invariant against itself, not against
  a plain GEMM (0.25% of elements differ), and C = 8 and C = 16 differ from each
  other.
- Timing is relative only: A16 over PCIe, untuned Triton GEMM. The logical-TP
  forms write fp32 partials, which inflates their cost at TP 1.
- Across nodes:
  - timing used 3–10 calls per point, and the all-to-all timings varied between
    launches;
  - messages were capped at 64 MB;
  - the TP part ran under default and Tree only;
  - L2 used random fillers only.
- Not tested:
  - gradient all-reduce (data parallel), backward;
  - `torch.compile`, CUDA graphs;
  - collectives overlapped with compute, or on several streams;
  - expert-parallel all-to-all;
  - other NCCL versions;
  - the SP fallback (the all-to-all reduce-scatter is implemented and passed
    alone, but not inside the layer).
- **Teardown error.** Two launches, a quick run and the Tree coll launch of
  run 2, printed an asynchronous NCCL "unhandled system error" on the final
  barrier. Rank 0 had already written its results, and the launch exited with
  status 0. This is likely a shutdown race on the shared-memory transport (not
  confirmed).

## How it was built

An agent wrote the script and the driver, with no human edits.
- **Probe.** A probe run on 8 GPUs established three things. NCCL 2.21.5 prints
  its per-call choice under `NCCL_DEBUG_SUBSYS=TUNING`. Tree and LL128 can be
  forced on this node. The topology is two identical rings and a chain.
- **Quick runs.** A quick run of each part worked on the first try.
- **Full run 1.** All 10 launches passed. A bug in the driver deleted the coll
  NCCL logs when the TP launch with the same tag started, so 3 settings lost
  their per-size algorithm table.
- **Changes.** The cleanup was fixed. Two schemes were added: `lt8_tree`, a
  cheaper degree-independent form, and row-parallel with SP, prompted by the
  Tree run.
- **Full run 2** produced these results. The merge was then rewritten to keep
  the JSON small and re-run on the same outputs, with no new GPU run.

**Two nodes**, the same agent, no human edits:
- **Topology** came from the launcher's smoke log.
- **Script changes.** The chunk count became a parameter, and the
  two-node tree orders, `lt8_2d`, `lt8_splitk` and `--xnode` were added.
- **First quick round.**
  - A one-node regression run caught a missing import, which was fixed.
  - A quick two-node launch showed the flat all-to-all fallback at 10x
    all-reduce at 16 MB. That prompted the two-level forms.
- **Second quick round** ran clean on both nodes.
- **One full run**: 9 launches, about 4 minutes of lock time, all exiting 0,
  with no NCCL errors.
- **Merge.** A compact merge was added, and a variable shadowed in it was
  fixed before the final merge.
