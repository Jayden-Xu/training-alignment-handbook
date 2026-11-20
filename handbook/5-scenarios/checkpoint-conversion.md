# Checkpoint conversion and resharding

## Where we are

Moving between frameworks, parallel layouts or number formats means converting
checkpoints: Hugging Face to a Megatron-style layout and back, separate q, k
and v projections to one fused QKV tensor, gate and up to one tensor, TP 2 to
TP 4, a change of pipeline stages, a cast. A conversion computes nothing; it
copies. So its claim is the strictest one in this handbook: every row of every
logical parameter arrives bit for bit. This page shows how to check that, and
what a copy check cannot see. It is step 2 of a
[framework migration](framework-migration.md), and the same check
[the RL loop](../2-rl/rl-loop.md) runs on every weight sync.

## The idea: rows survive repacking

A converter moves rows around. A row here is one last-dimension vector of a
weight, such as one output channel of a projection. Take one query group with
two q heads, one k head and one v head, and call their rows q0, q1, k0, v0:

| Layout | What is stored |
|---|---|
| Separate projections | `q_proj`: q0 q1 · `k_proj`: k0 · `v_proj`: v0 |
| Fused, per query group | `linear_qkv`: q0 q1 k0 v0 |
| Fused, wrong interleave | `linear_qkv`: q0 k0 q1 v0 |

Read back by the layout's rule (first two rows of each group are q), the wrong
interleave gives a "q" of q0 k0. Splitting across TP ranks moves whole rows to
other files; a row-parallel layer splits each row into pieces, one per rank.
None of this may change a single bit of any row.

So compare each logical parameter as a **multiset of exact rows**: build it on
both sides by the layout's rule, then match rows by their bytes, ignoring
order. Repacking is then not a difference. A row that changed, moved into
another parameter or vanished is left over, and the check names it: logical
parameter, row, and the physical tensor and row it came from. In the example,
k0 is left over in "q" and q1 is left over in "k".

What the multiset cannot see is a row in the right parameter at the wrong
place: two heads swapped inside q, or rows permuted for another rotary
convention. That is placement, and it needs a second check: a forward on fixed
inputs at the target layout, compared with the source.

The rule: **prove the copy row by row, then prove the placement with a
forward.** The general principle is to compare at the finest unit the
transform must preserve, in a form that ignores what the transform is allowed
to change.

## The step

1. **Write the map.**
   - Goal: for each logical parameter, how each layout stores it.
   - How: a `ckpt-map-v1` JSON file for
     [ckpt_compare.py](../../tools/ckpt_compare.py), one entry per logical
     parameter and layout: tensor names with `{i}` for the layer and `{r}`
     for the rank, `cat` over ranks, `part` for a fused tensor's groups,
     `slice` for padding, `replicated` for copies that must be bit-equal, and
     `"dim": -1` to join row-parallel pieces back into rows. The
     [example map](../../examples/ckpt_conversion/mcore_map.json) describes a
     Megatron-style layout:

     ```json
     "model.layers.{i}.self_attn.q_proj.weight": {
       "hf": "=",
       "mcore": {"cat": {"over": "r", "each": {
         "part": "mp_rank_{r:02d}/model.decoder.layers.{i}.self_attention.linear_qkv.weight",
         "pattern": ["{qrows}", "{hd}", "{hd}"], "take": 0}}}},
     "model.layers.{i}.self_attn.o_proj.weight": {
       "hf": "=",
       "mcore": {"cat": {"over": "r", "each":
         "mp_rank_{r:02d}/model.decoder.layers.{i}.self_attention.linear_proj.weight"}, "dim": -1}}
     ```
   - Success: with `--strict`, no tensor of the source is left unused by the
     map.
2. **Compare the rows.**
   - Goal: every row of every logical parameter matched.
   - How: run the tool on the two checkpoints, naming the map's layouts and
     the target's TP degree:

     ```bash
     python tools/ckpt_compare.py SRC_DIR DST_DIR --map map.json --layouts hf mcore --set b:r=2
     ```

     A single flipped bit in a converted QKV shard reads like this:

     ```
     model.layers.1.self_attn.k_proj.weight: rows_unmatched
       A 1 of 64 rows unmatched: row 8 <- model.layers.1.self_attn.k_proj.weight[8]
       B 1 of 64 rows unmatched: row 8 <- mp_rank_00/model.decoder.layers.1.self_attention.linear_qkv.weight[40]
     NOT all rows matched
     ```
   - Success: "all rows matched" and exit status 0. Every dtype change is
     reported per parameter: a widening (bf16 to fp32) is exact and its rows
     are compared after widening; a narrowing names each row that lost bits.
3. **Run a forward at the target layout.**
   - Goal: each row sits where the target's code reads it.
   - How: load the converted checkpoint into the target's model code at the
     target layout, run fixed token ids, and compare the logits with the
     source model run **with the same reduction**: a row-parallel layer at
     TP 2 adds two partial sums, which changes the bits against TP 1 even
     when every weight is right ([parallelism](../2-rl/parallelism.md)). The
     demo's [ckpt_conversion.py](../../examples/ckpt_conversion/ckpt_conversion.py)
     shows the pattern on CPU.
   - Success: the logits are bit-equal.
4. **Keep it.**
   - Goal: a converter change, or a new model, is checked the same way.
   - How: keep the conversion pair, with planted faults, as a
     [regression](../4-operations/regression.md) entry.
   - Success: the suite reports the clean pair equal and every planted fault
     named.

Which row key to use:

| Row key | What it sees | Blind to | Cost class | When to choose it |
|---|---|---|---|---|
| Digest of the row's bytes in order (`--rows digest`, the default) | any change to a row | nothing inside a row | small, on CPU, offline | conversion and resharding checks |
| XOR row fingerprint (`--rows xor`, OpGuard's format) | most changes; the values OpGuard records at a weight sync | words reordered inside a row; equal changes in one lane of a row, which a cast produces | free on the GPU | live weight-sync checks, or comparing against OpGuard records |

What to compare the target forward with:

| Reference | What it gives | Cost class | When to choose it |
|---|---|---|---|
| The source model with the target's reduction (same split, same order of partial sums) | bit equality for a correct placement | small | you can run the source's row-parallel layers split the same way |
| Both sides with a logical TP size ([parallelism](../2-rl/parallelism.md)) | bit equality at any degree | moderate (under 2x) | the target already uses parallelism-invariant layers |
| A round trip back to the source layout, then the source forward | bit equality, but it tests the reverse converter, not the target's reading | free | the target's model code is not available |
| Logits within a tolerance | catches gross misplacement only | free | last resort, never as the copy check |

Shown, on CPU, in a known-answer demonstration: a Llama-shaped checkpoint
converted to a Megatron-style layout at two TP degrees and back is bit-exact;
the tool matched every row of every logical parameter, also between the two
degrees; it named a transposed shard, a wrong QKV interleave, a bf16 to fp16
to bf16 cast, one flipped bit and a missing row by parameter and row; the
forward check was bit-equal for the correct conversion and saw two placement
faults the row check cannot see. It did not see the cast
([report](../../examples/ckpt_conversion/REPORT.md)). Not yet shown:
Megatron-core's own loader and distributed checkpoint format, pipeline
resharding, vocabulary padding, a real model's weights.

## What can go wrong

!!! warning "A forward check used as the copy check"
    A cast that rounds only the smallest weights can leave every logit
    unchanged on a few inputs (measured: see report). The conversion then
    passes with changed weights. Prove the copy with rows; use the forward for
    placement only.

!!! warning "XOR fingerprints for an offline conversion check"
    XOR cancels two equal changes in the same lane of a row, and a cast makes
    such pairs; it also ignores words reordered inside a row, such as
    row-parallel pieces joined in the wrong order. Changed rows then match.
    Use the row digest offline; keep XOR for live checks against OpGuard
    records.

!!! warning "A sharded forward compared with an unsharded one"
    A row-parallel layer at TP 2 sums two partials, so most logits differ from
    TP 1 for a perfect conversion. The check then fails for the wrong reason.
    Give both sides the same reduction.

!!! warning "Tensors the map does not mention"
    A logical parameter the map forgot is never compared, so a converter that
    drops it passes. Run with `--strict`, which fails on unused tensors, and
    read the unused list.

!!! warning "Padding and replicated copies"
    Zero padding rows match any zero row and prove nothing about placement;
    a stale copy of a replicated norm on one rank is invisible if only rank 0
    is read. Slice padding out in the map and declare copies as `replicated`.

## Summary

- A conversion copies, so every row of every logical parameter must arrive
  bit for bit.
- Compare each logical parameter as a multiset of exact rows: repacking is
  not a difference, and each leftover row is named with its origin.
- A multiset cannot see a row moved inside its parameter: check placement
  with a forward at the target layout, against the source with the same
  reduction.
- Use a digest of the row bytes offline; XOR fingerprints have blind spots
  inside a row.

## Go deeper

- [Report: the checkpoint conversion demo](../../examples/ckpt_conversion/REPORT.md):
  the layout, the map, each planted fault and what each check saw.
- Code: [ckpt_compare.py](../../tools/ckpt_compare.py),
  [convert.py](../../examples/ckpt_conversion/convert.py),
  [ckpt_conversion.py](../../examples/ckpt_conversion/ckpt_conversion.py).
- [RL loop](../2-rl/rl-loop.md): the weight-sync check, the same row matching
  on live weights.
- [Parallelism](../2-rl/parallelism.md): why a TP degree changes the bits of a
  forward, and the logical TP size that removes it.
