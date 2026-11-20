# Checkpoint conversion and resharding

## Where we are

Moving between frameworks, parallel layouts or number formats means converting
checkpoints: from Hugging Face to a Megatron-style layout and back, from
separate q, k and v projections to one fused QKV tensor, from separate gate and
up projections to one tensor, from TP 2 to TP 4, to a different number of
pipeline stages, or to another dtype. A conversion computes nothing; it only
copies. Its claim is therefore the strictest one in this handbook: every row of
every logical parameter must arrive bit for bit. This page shows how to check
that, and what a check of the copy cannot see. It is step 2 of a
[framework migration](framework-migration.md), and it is the same check that
[the RL loop](../2-rl/rl-loop.md) runs on every weight sync.

## The idea: each row keeps its values when a checkpoint is repacked

A converter moves rows around. A row here is one vector along the last
dimension of a weight, such as one output channel of a projection. Take one
query group with two q heads, one k head and one v head, and call their rows
q0, q1, k0 and v0:

| Layout | What is stored |
|---|---|
| Separate projections | `q_proj`: q0 q1 · `k_proj`: k0 · `v_proj`: v0 |
| Fused, grouped by query group | `linear_qkv`: q0 q1 k0 v0 |
| Fused, with the wrong interleave | `linear_qkv`: q0 k0 q1 v0 |

If you read the last layout back with the layout's rule (the first two rows of
each group are q), you get a "q" made of q0 and k0. Splitting across TP ranks
moves whole rows into other files, and a row-parallel layer splits each row
into pieces, one per rank. None of this may change a single bit of any row.

So compare each logical parameter as a **multiset of exact rows**. Build the
parameter on both sides with the layout's rule, then match rows by their
bytes and ignore their order. Repacking then does not count as a difference.
A row that changed, moved into another parameter or vanished is left over,
and the check names it: the logical parameter and row, and the physical tensor
and row it came from. In the example, k0 is left over in "q" and q1 is left
over in "k".

What the multiset cannot see is a row that is in the right parameter but at
the wrong place: two heads swapped inside q, or rows permuted for a different
rotary convention. That is a question of placement, and it needs a second
check: a forward on fixed inputs at the target layout, compared with the
source.

So the rule is: **prove the copy row by row, then prove the placement with a
forward.** The general principle is to compare at the finest unit that the
transform must preserve, in a form that ignores what the transform is allowed
to change.

## The step

1. **Write the map.**
   - Goal: for each logical parameter, describe how each layout stores it.
   - How: write a `ckpt-map-v1` JSON file for
     [ckpt_compare.py](../../tools/ckpt_compare.py), with one entry per
     logical parameter and layout. Tensor names use `{i}` for the layer and
     `{r}` for the rank; `cat` joins pieces over ranks, `part` picks a group
     out of a fused tensor, `slice` removes padding, `replicated` declares
     copies that must be bit-equal, and `"dim": -1` joins row-parallel pieces
     back into rows. The
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
   - Success: with `--strict`, the map uses every tensor of the source.
2. **Compare the rows.**
   - Goal: every row of every logical parameter has a match.
   - How: run the tool on the two checkpoints, naming the map's layouts and
     the target's TP degree:

     ```bash
     python tools/ckpt_compare.py SRC_DIR DST_DIR --map map.json --layouts hf mcore --set b:r=2
     ```

     A single flipped bit in a converted QKV shard is reported like this:

     ```
     model.layers.1.self_attn.k_proj.weight: rows_unmatched
       A 1 of 64 rows unmatched: row 8 <- model.layers.1.self_attn.k_proj.weight[8]
       B 1 of 64 rows unmatched: row 8 <- mp_rank_00/model.decoder.layers.1.self_attention.linear_qkv.weight[40]
     NOT all rows matched
     ```
   - Success: the tool prints "all rows matched" and exits with status 0. Every
     dtype change is reported per parameter. A widening (bf16 to fp32) is
     exact, and its rows are compared after widening; a narrowing names every
     row that lost bits.
3. **Run a forward at the target layout.**
   - Goal: confirm that each row sits where the target's code reads it.
   - How: load the converted checkpoint into the target's model code at the
     target layout, run it on fixed token ids, and compare the logits with the
     source model run **with the same reduction**. A row-parallel layer at
     TP 2 adds two partial sums, which changes the bits compared with TP 1
     even when every weight is right ([parallelism](../2-rl/parallelism.md)).
     The demo's [ckpt_conversion.py](../../examples/ckpt_conversion/ckpt_conversion.py)
     shows the pattern on CPU.
   - Success: the logits are bit-equal.
4. **Keep the check.**
   - Goal: a change to the converter, or a new model, is checked the same way.
   - How: keep the conversion pair, together with planted faults, as a
     [regression](../4-operations/regression.md) entry.
   - Success: the suite reports the clean pair as equal and names every
     planted fault.

Which row key to use:

| Row key | What it sees | What it is blind to | Cost class | When to choose it |
|---|---|---|---|---|
| A digest of the row's bytes in order (`--rows digest`, the default) | Any change to a row | Nothing inside a row | small, on CPU, offline | Checks of a conversion or a resharding |
| An XOR fingerprint of the row (`--rows xor`, OpGuard's format) | Most changes, and the values that OpGuard records at a weight sync | Words reordered inside a row, and equal changes in one lane of a row, which a cast can produce | free on the GPU | Live weight-sync checks, or comparisons against OpGuard records |

What to compare the target's forward with:

| Reference | What it gives | Cost class | When to choose it |
|---|---|---|---|
| The source model with the target's reduction (the same split and the same order of partial sums) | Bit equality when the placement is correct | small | You can run the source's row-parallel layers split the same way |
| Both sides with a logical TP size ([parallelism](../2-rl/parallelism.md)) | Bit equality at any degree | moderate (under 2x) | The target already uses layers that are invariant to parallelism |
| A round trip back to the source layout, followed by the source's forward | Bit equality, but it tests the reverse converter rather than how the target reads the weights | free | The target's model code is not available |
| Logits within a tolerance | Catches only gross misplacement | free | As a last resort, and never as the check of the copy |

What has been shown, on CPU, in a known-answer demonstration: a Llama-shaped
checkpoint converted to a Megatron-style layout at two TP degrees and back is
bit-exact. The tool matched every row of every logical parameter, also between
the two degrees. It named, by parameter and row, a transposed shard, a wrong
QKV interleave, a bf16 to fp16 to bf16 cast, one flipped bit and a missing
row. The forward check was bit-equal for the correct conversion and caught two
placement faults that the row check cannot see, but it did not catch the cast
([report](../../examples/ckpt_conversion/REPORT.md)). Not yet shown:
Megatron-core's own loader and its distributed checkpoint format, pipeline
resharding, vocabulary padding, and a real model's weights.

## What can go wrong

!!! warning "Using a forward check as the check of the copy"
    A cast that rounds only the smallest weights can leave every logit
    unchanged on a few inputs (measured: see report). The conversion then
    passes even though weights changed. Prove the copy with rows, and use the
    forward only for placement.

!!! warning "Using XOR fingerprints for an offline conversion check"
    XOR cancels two equal changes in the same lane of a row, and a cast
    produces such pairs. XOR also ignores words reordered inside a row, such
    as row-parallel pieces joined in the wrong order. Rows that changed then
    still match. Use the row digest offline, and keep XOR for live checks
    against OpGuard records.

!!! warning "Comparing a sharded forward with an unsharded one"
    A row-parallel layer at TP 2 sums two partial sums, so most logits differ
    from TP 1 even for a perfect conversion. The check then fails for the
    wrong reason. Give both sides the same reduction.

!!! warning "Tensors that the map does not mention"
    A logical parameter that the map forgot is never compared, so a converter
    that drops it still passes. Run with `--strict`, which fails on unused
    tensors, and read the list of unused tensors.

!!! warning "Padding and replicated copies"
    Rows of zero padding match any zero row and prove nothing about
    placement. A stale copy of a replicated norm on one rank goes unnoticed if
    only rank 0 is read. Slice the padding out in the map, and declare copies
    as `replicated`.

## Summary

- A conversion only copies, so every row of every logical parameter must
  arrive bit for bit.
- Compare each logical parameter as a multiset of exact rows. Repacking is
  then not a difference, and every row that is left over is named together
  with its origin.
- A multiset cannot see a row that moved inside its parameter, so check the
  placement with a forward at the target layout, against the source run with
  the same reduction.
- Use a digest of the row's bytes offline, because XOR fingerprints have blind
  spots inside a row.

## Go deeper

- [Report: the checkpoint conversion demo](../../examples/ckpt_conversion/REPORT.md):
  the layout, the map, each planted fault, and what each check saw.
- Code: [ckpt_compare.py](../../tools/ckpt_compare.py),
  [convert.py](../../examples/ckpt_conversion/convert.py),
  [ckpt_conversion.py](../../examples/ckpt_conversion/ckpt_conversion.py).
- [RL loop](../2-rl/rl-loop.md): the weight-sync check, which applies the same
  row matching to live weights.
- [Parallelism](../2-rl/parallelism.md): why the TP degree changes the bits of
  a forward, and how a logical TP size removes the change.
