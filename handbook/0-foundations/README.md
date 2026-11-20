# 0. Foundations

## Where we are

The handbook teaches three workflows: finding where two pre-training runs part
([part 1](../1-pretraining/README.md)), making a rollout engine and a trainer
agree ([part 2](../2-rl/README.md)), and making one operator invariant
([part 3](../3-kernels/README.md)). All three rest on the same few ideas, and
this part teaches them:

1. floating-point addition is not associative;
2. a reduction's order depends on the shape it runs at;
3. deterministic is not the same as invariant;
4. the five invariance levels, L1 to L5;
5. how to compare two runs: bitwise, at operator boundaries, with fingerprints.

Two runs are **aligned** when they are meant to compute the same model from the
same data, seeds and weights, and agree **bit for bit** at the points you
observe. Alignment asks "where is the first place they stop being identical?",
not "are they close?". That first place is where debugging starts.

## Floating-point addition is not associative

Take three numbers and add them in two orders. In ordinary Python (fp64):

```python
>>> (0.1 + 0.2) + 0.3
0.6000000000000001
>>> 0.1 + (0.2 + 0.3)
0.6
```

The same happens in fp32 with numbers a model meets every day. Let
a = 1e8, b = 1, c = −1e8:

| Order | Step by step (fp32) | Result |
|---|---|---|
| (a + b) + c | a + b = 100000001 is not representable; it rounds to 1e8. Then 1e8 − 1e8 | 0 |
| (a + c) + b | a + c = 0 exactly. Then 0 + 1 | 1 |

Every addition rounds its result to the nearest representable number, and
which roundings happen depends on which partial sums are formed. So:

**Rule.** The bits of a floating-point sum depend on the order of its additions
and on where it rounds. Same mathematics in another order is another number.

This is the general principle behind every page of the handbook: *the
reduction order is part of the result*. Every matrix multiply, norm, softmax,
attention, gradient sum and collective is a sum, so every one of them has an
order, and two runs agree only if their orders agree.

## A reduction's order depends on shape

A kernel does not choose its order from the mathematics. It chooses it from the
shape it is given, so the arithmetic changes with the batch. Take a dot product
of four terms, t = [1e8, 1, −1e8, 1], in fp32:

<figure class="fig">
<div class="fig-rows">
<div class="fig-group"><span class="fig-label">one program, left to right</span><div class="fig-chain"><span class="fig-node">1e8 + 1 = 1e8</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">1e8 − 1e8 = 0</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node is-key">0 + 1 = 1</span></div></div>
<div class="fig-group"><span class="fig-label">split-K in two halves</span><div class="fig-chain"><div class="fig-stack"><span class="fig-node">1e8 + 1 = 1e8</span><span class="fig-node">−1e8 + 1 = −1e8</span></div><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node is-key">1e8 + (−1e8) = 0</span></div></div>
<div class="fig-group"><span class="fig-label">atomics, one finishing order</span><div class="fig-chain"><span class="fig-node">1e8 − 1e8 = 0</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">0 + 1 = 1</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node is-key">1 + 1 = 2</span></div></div>
</div>
<figcaption>t = [1e8, 1, −1e8, 1] in fp32: the same four terms, three orders, three results.</figcaption>
</figure>

Three orders, three answers: 1, 0 and 2. Each is a correct rounding of some
order. Three mechanisms pick the order:

| Mechanism | What it is | What sets the order |
|---|---|---|
| split-K | The reduction dimension is cut into pieces summed by separate programs, then the partial sums are added | The library picks the number of pieces from the matrix shape, often from the number of rows |
| tiles | A program reduces its row in chunks of a tile width, then combines the chunks | The tile width and the number of chunks are chosen per shape; a kernel may switch to another kernel at a size threshold |
| atomics | Programs add their partial sums into one memory location as they finish | The finishing order, which changes from run to run |

**Rule.** If a kernel's order is a function of the shape, a row's result is a
function of the batch it sits in. If the order is a function of timing, the
result is a function of the run.

## Deterministic is not invariant

These are two different promises:

- **Deterministic** (run-to-run): the same inputs, at the same shapes, give the
  same bits on every run. Atomics break it.
- **Invariant** (here, batch-invariant): a row gives the same bits whatever else
  is computed with it: a batch of 1 or of 64, first or last in the batch.

A tiny example. A matrix multiply with no atomics is deterministic. At one row
the library picks split-K in two halves; at 64 rows it picks one program per
output. Run each twice: each repeats its own bits exactly. But row 0 computed
alone and row 0 computed inside the batch of 64 used the two different orders
above, so they can differ.

`torch.use_deterministic_algorithms(True)` buys the first promise, not the
second. A kernel can also be deterministic and still have an order that is a
detail of its implementation (for example, one that depends on the number of
indices it scatters), so two correct frameworks that are each deterministic
need not agree.

**Rule.** Determinism removes variation in time. Invariance removes variation
in context. Alignment between two different setups needs invariance.

## The five levels of invariance

Each level asks whether one thing may change without changing the bits:

| Level | Holds when | One-line example of a break |
|---|---|---|
| L1 run-to-run | The same input gives the same bits on every run | A backward that adds gradient rows with atomics gives a different weight gradient on the second run |
| L2 batch | A row's result does not depend on batch size, its position or the other rows | A GEMM picks split-K at one row and not at 64, so a sequence's logits change with the batch |
| L3 mode | Prefill, decode, chunked prefill and paged layouts give the same bits per position | Decode splits the KV cache across programs and prefill does not, so the same token gets two attention outputs |
| L4 cross-engine | The trainer and the inference engine give the same bits | The engine multiplies the norm weight in fp32 before casting to bf16; the trainer casts first |
| L5 parallelism | Different TP, EP, SP or DP degrees give the same bits | At TP 2 two partial sums are added across ranks; at TP 4, four, in another tree |

Each scenario needs a different subset:

| Scenario | Needs |
|---|---|
| Pre-training debugging (replay, config change, upgrade) | L1 on both runs, plus the same configuration on the controlled surfaces |
| Inference determinism | L1 + L2 (+ L3 with chunked prefill or prefix caching) |
| RL, train–inference parity | L1–L4 (+ L5 when the two sides shard differently) |
| Migration or upgrade equivalence | L1 + L4 between old and new stack (+ L5 if parallelism changes) |

Every level holds on one **stack**: one GPU type, driver and set of library
versions. Across GPU generations, bitwise agreement is generally not expected
([hardware coverage](../4-operations/hardware.md)). Verify the levels on each
stack; a bitwise comparison across stacks is a separate claim that must be
measured. The [alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md)
routes each scenario to its workflow.

## Controlled and residual nondeterminism

Remove only the **avoidable** variation, the kind that makes two correct runs
do different arithmetic:

- seeds and data order;
- library algorithm choice and numeric modes;
- collective reduction trees;
- incomplete checkpoint restore.

Leave the rest visible. A user or third-party kernel whose result changes with
scheduling (racy atomics, a missing barrier, a stream race) is a bug, not
noise. Controlling it would hide the fault you are hunting.

## How to compare

Comparing two runs is the step every workflow repeats. It has three choices:
the criterion, the place, and the summary.

**Criterion: bitwise.** A tolerance does not answer "where is the first
difference?". A threshold loose enough to hide harmless reduction-order noise
also hides a few corrupted elements, and on production bugs tolerances lost a
large share of the exact localizations (measured: see
[report](../../examples/pretraining_e2e/REPORT.md#overview-tolerances-against-bitwise-comparison)).
Control the avoidable variation instead, then compare bits.

**Place: model-level operator boundaries.** These are calls in model code that
start device work: a projection, a norm, an attention block, an expert combine.

- Whole steps or layers are too coarse to localize.
- Individual kernels change with every library version and fusion choice.

Model-level boundaries exist in every faithful implementation, so two stacks
can be compared there.

**Summary: fingerprints.** Dumping every tensor costs too much. A fingerprint
must catch any single-bit change, must not let small corruptions cancel, and
must not synchronize the device. A tiny example with a tensor of four values:

| | Values | Sum | XOR of the raw 32-bit words |
|---|---|---|---|
| reference | [1, 2, 3, 4] | 10 | w |
| one bit flipped in one value | [1, 2, 3, 4 + ε] | changes | changes |
| two errors that cancel | [1 + ε, 2, 3, 4 − ε] | 10, unchanged | changes |

A sum or a mean lets errors cancel; XOR over the raw bits does not lose a
single flip. OpGuard computes XOR fingerprints on the device, in two forms:

- **Regional** (default): XORs over equal byte ranges of the flattened tensor.
  Cheap, and shows which region differs.
- **Per vector** (opt-in): one XOR per last-dimension vector. XOR merges, so the
  vectors of a token combine into that token's row fingerprint whatever the
  layout: `[T, heads, dim]` and `[T, hidden]` give the same row. Rows become
  comparable across stacks that batch, pack and decode differently, and the
  differing head is named.

The options, from cheapest to most complete:

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Regional fingerprint at every boundary | Which boundary differs first, and which region | small per boundary: a few device kernels, no synchronization | Always, first |
| Per-vector fingerprint | Which row or head differs; comparable across layouts | small to moderate | Comparing stacks that batch or pack differently |
| Full tensor dump of one region | The values themselves, for diagnosis | large (2x or more), and storage | After a fingerprint names the place |
| Tolerance on values | "Close or not" | free | Only for checking accuracy, never to localize |

The tools are the [opguard_bridge](../../tools/opguard_bridge.py) (OpGuard's
fingerprint formats in plain torch) and OpGuard itself; part 1 uses both.
Success: two runs that should agree give equal fingerprints at every boundary,
and a planted one-bit change is found at the boundary where it was planted.

**Facts and judgments.** Keep the tools exact and generic, and put case
knowledge in skills. A tool computes fingerprints, records what it observed at
known points and answers equality questions. An agent or an engineer decides
which records correspond, whether a missing row was fused away, which
divergence is first, and what caused it, and writes each decision down with its
evidence. Built-in heuristics for these decisions broke on every new case in
our experience.

## What can go wrong

!!! warning "A tolerance hides the fault"
    A threshold that absorbs reduction-order noise also absorbs small
    corruptions. You localize the wrong place, or nothing. Control the avoidable
    nondeterminism and compare bitwise.

!!! warning "Deterministic mode taken for invariance"
    Deterministic algorithms make a run repeat itself; they do not make a row
    independent of its batch or of the other framework. Two deterministic
    setups can still disagree. Check the level you need (L2, L4, L5) directly.

!!! warning "Equal fingerprints taken as equal values"
    XOR is a fingerprint, not a hash: permutations and pairwise cancellations
    leave it unchanged, and 32-bit values collide at scale. Treat equality as
    evidence only with a matching width and a non-zero value, and recapture
    full tensors for a region you suspect.

!!! warning "Comparing across stacks as if it were one"
    The levels hold on one GPU type and one set of library versions. A bitwise
    difference across GPU generations is expected, not a bug. Verify each stack
    separately.

## Summary

- The reduction order is part of the result: three numbers in two orders can
  give two answers.
- Kernels pick their order from the shape (split-K, tiles) or from timing
  (atomics); so a row's bits can depend on its batch or on the run.
- Deterministic (L1) is not invariant (L2–L5); say which level a claim holds at.
- Compare bitwise, at model-level operator boundaries, with XOR fingerprints.
- Control the avoidable nondeterminism; leave residual nondeterminism visible,
  because it is the bug signal.

## Go deeper

- [Glossary](../glossary.md): every term used here.
- [Part 1](../1-pretraining/README.md): the comparison applied to two training
  runs.
- [Part 3](../3-kernels/README.md): how to write a kernel with a fixed order.
- [opguard_bridge.py](../../tools/opguard_bridge.py): the fingerprint formats.
- [Report: tolerances against bitwise comparison](../../examples/pretraining_e2e/REPORT.md#overview-tolerances-against-bitwise-comparison).
- [alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md).
