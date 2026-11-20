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

<figure class="fx" data-fig="two-orders" aria-label="Three numbers added in two orders">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-label">fp32</span><span class="fx-chip">a = 1e8</span><span class="fx-chip">b = 1</span><span class="fx-chip">c = −1e8</span></div>
<div class="fx-rows">
<div class="fx-row"><span class="fx-label fx-mono">(a + b) + c</span><div class="fx-chain is-tight is-grid"><span class="fx-swap" data-at="1"><span class="fx-node is-mono" data-at="1" data-to="2">a + b = 100000001</span><span class="fx-node is-mono is-warn" data-at="2">a + b → 1e8<span class="fx-note">rounded</span></span></span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono" data-at="3">1e8 + c = 0</span><span class="fx-arrow" data-at="4"></span><span class="fx-node is-mono is-key" data-at="4">0</span></div></div>
<div class="fx-row"><span class="fx-label fx-mono">(a + c) + b</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">a + c = 0<span class="fx-note">exact</span></span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono" data-at="3">0 + b = 1</span><span class="fx-arrow" data-at="4"></span><span class="fx-node is-mono is-key" data-at="4">1</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>Three fp32 numbers: a = 1e8, b = 1, c = −1e8. Two orders of adding them.</li>
<li>Each order forms its first partial sum.</li>
<li data-hold="1.5">a + b = 100000001 is not representable in fp32: it rounds to 1e8.</li>
<li>Each order adds its last number.</li>
<li data-hold="2">Same three numbers, two orders, two results: 0 and 1.</li>
</ol>
<figcaption>Figure: (a + b) + c and (a + c) + b in fp32. Which partial sums are formed decides which roundings happen.</figcaption>
</figure>

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

<figure class="fx" data-fig="three-orders" aria-label="One dot product reduced in three orders">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-label">t</span><span class="fx-chip">1e8</span><span class="fx-chip">1</span><span class="fx-chip">−1e8</span><span class="fx-chip">1</span></div>
<div class="fx-rows" style="--lw: 6.5rem; --cols: minmax(0, 1.2fr) auto minmax(0, 1fr) auto minmax(0, 1.2fr); --narrow: flex">
<div class="fx-row"><span class="fx-label">one program, left to right</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 + 1 = 1e8</span><span class="fx-arrow" data-at="2"></span><span class="fx-node is-mono" data-at="2">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono is-key" data-at="3" data-hot="4">0 + 1 = 1</span></div></div>
<div class="fx-row"><span class="fx-label">split-K in two halves</span><div class="fx-chain is-tight is-grid"><div class="fx-stack" data-at="1"><span class="fx-node is-mono" data-at="1">1e8 + 1 = 1e8</span><span class="fx-node is-mono" data-at="1">−1e8 + 1 = −1e8</span></div><span class="fx-arrow is-long" data-at="2" style="grid-column: 2 / 5"></span><span class="fx-node is-mono is-key" data-at="2" data-hot="4">1e8 + (−1e8) = 0</span></div></div>
<div class="fx-row"><span class="fx-label">atomics, one finishing order</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="2"></span><span class="fx-node is-mono" data-at="2">0 + 1 = 1</span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono is-key" data-at="3" data-hot="4">1 + 1 = 2</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>The same four terms, t = [1e8, 1, −1e8, 1], reduced three ways in fp32.</li>
<li>First adds. Split-K sums its two halves at once, on two programs.</li>
<li>Split-K adds its two partial sums and is done: 0.</li>
<li>The other two finish: left to right gives 1, this finishing order gives 2.</li>
<li data-hold="2">Three orders, three results: 1, 0 and 2. Each is a correct rounding of some order.</li>
</ol>
<figcaption>Figure: t = [1e8, 1, −1e8, 1] in fp32: the same four terms, three orders, three results.</figcaption>
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

<figure class="fx" data-fig="sum-xor" aria-label="A sum against an XOR fingerprint">
<div class="fx-stage">
<div class="fx-rows" style="--cols: minmax(0, 1.25fr) auto minmax(0, 1fr) minmax(0, 1fr); --cols-narrow: auto minmax(0, 1fr) minmax(0, 1fr)">
<div class="fx-row"><span class="fx-label">reference</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="1"><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span></div><span class="fx-arrow" data-at="1"></span><span class="fx-node" data-at="1">sum <span class="fx-mono">10</span></span><span class="fx-node" data-at="1">XOR of the raw 32-bit words <span class="fx-mono">w</span></span></div></div>
<div class="fx-row" data-at="2"><span class="fx-label">one bit flipped in one value</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="3"><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell is-bad">4 + ε</span></div><span class="fx-arrow" data-at="3"></span><span class="fx-node is-ok" data-at="3">sum <span class="fx-mono">changes</span><span class="fx-note">caught</span></span><span class="fx-node is-ok" data-at="3">XOR <span class="fx-mono">changes</span><span class="fx-note">caught</span></span></div></div>
<div class="fx-row" data-at="4"><span class="fx-label">two errors that cancel</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="5"><span class="fx-cell is-bad">1 + ε</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell is-bad">4 − ε</span></div><span class="fx-arrow" data-at="5"></span><span class="fx-node is-bad" data-at="5">sum <span class="fx-mono">10, unchanged</span><span class="fx-note">missed</span></span><span class="fx-node is-ok" data-at="5">XOR <span class="fx-mono">changes</span><span class="fx-note">caught</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>A tensor of four values: [1, 2, 3, 4].</li>
<li>Its sum is 10; the XOR of its raw 32-bit words is w.</li>
<li>Flip one bit in one value: 4 becomes 4 + ε.</li>
<li>Both summaries change: the flip is caught.</li>
<li>Now two errors that cancel: 1 + ε and 4 − ε.</li>
<li data-hold="2">The sum is 10 again and misses both; the XOR still changes.</li>
</ol>
<figcaption>Figure: a sum lets errors cancel; XOR over the raw bits does not lose a single flip.</figcaption>
</figure>

A sum or a mean lets errors cancel; XOR over the raw bits does not lose a
single flip. OpGuard computes XOR fingerprints on the device, in two forms:

- **Regional** (default): XORs over equal byte ranges of the flattened tensor.
  Cheap, and shows which region differs.
- **Per vector** (opt-in): one XOR per last-dimension vector. XOR merges, so the
  vectors of a token combine into that token's row fingerprint whatever the
  layout: `[T, heads, dim]` and `[T, hidden]` give the same row. Rows become
  comparable across stacks that batch, pack and decode differently, and the
  differing head is named.

<figure class="fx" data-fig="xor-fold" aria-label="Per-vector fingerprints folding into a row fingerprint">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label fx-mono">[T, heads, dim]</span><div class="fx-tree" style="--n:4">
<div class="fx-level"><span class="fx-cell">head 0</span><span class="fx-cell">head 1</span><span class="fx-cell">head 2</span><span class="fx-cell">head 3</span></div>
<div class="fx-level"><span class="fx-j" style="--k:1" data-at="1"><span class="fx-node is-mono">f0</span></span><span class="fx-j" style="--k:1" data-at="1"><span class="fx-node is-mono">f1</span></span><span class="fx-j" style="--k:1" data-at="1"><span class="fx-node is-mono">f2</span></span><span class="fx-j" style="--k:1" data-at="1"><span class="fx-node is-mono">f3</span></span></div>
<div class="fx-level"><span class="fx-j" style="--w:2" data-at="2"><span class="fx-node is-mono">f0 ⊕ f1</span></span><span class="fx-j" style="--w:2" data-at="2"><span class="fx-node is-mono">f2 ⊕ f3</span></span></div>
<div class="fx-level"><span class="fx-j" style="--w:4" data-at="3" data-hot="5"><span class="fx-node is-mono is-key">row r</span></span></div>
</div></div>
<div class="fx-row" data-at="4"><span class="fx-label fx-mono">[T, hidden]</span><div class="fx-tree" style="--n:4">
<div class="fx-level"><span class="fx-cell" style="--w:4">one vector of hidden values</span></div>
<div class="fx-level"><span class="fx-j" style="--w:4;--k:1" data-at="4" data-hot="5"><span class="fx-node is-mono is-ok">row r</span></span></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>One token's row, stored as [T, heads, dim]: four head vectors.</li>
<li>Per vector: one XOR for each last-dimension vector, f0 to f3.</li>
<li>XOR merges, so the vectors fold pairwise,</li>
<li>into the row's fingerprint r.</li>
<li>Stored as [T, hidden], the same row is one vector, and its XOR is the same r.</li>
<li data-hold="2">Rows compare across layouts; when they differ, the differing f names the head.</li>
</ol>
<figcaption>Figure: per-vector fingerprints fold into one row fingerprint whatever the layout.</figcaption>
</figure>

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
