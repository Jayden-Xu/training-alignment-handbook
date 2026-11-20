# 0. Foundations

## Where we are

The handbook teaches three workflows: finding where two pre-training runs stop
agreeing ([part 1](../1-pretraining/README.md)), making a rollout engine and a
trainer agree ([part 2](../2-rl/README.md)), and making one operator
batch-invariant ([part 3](../3-kernels/README.md)). All three rest on the same
few ideas, and this part teaches them:

1. floating-point addition is not associative;
2. a kernel picks the order of a sum from the shape of its input;
3. a deterministic kernel is not necessarily an invariant one;
4. invariance comes in five levels, L1 to L5;
5. two runs are compared bit for bit, at operator boundaries, using small
   summaries called fingerprints.

Two runs are **aligned** when they are meant to compute the same model from
the same data, seeds and weights, and they agree **bit for bit** at every
point you observe. Alignment does not ask whether the two runs are close. It
asks where they first stop being identical, because that place is where
debugging starts.

## Floating-point addition is not associative

Take three numbers and add them in two different orders. In ordinary Python,
which uses fp64:

```python
>>> (0.1 + 0.2) + 0.3
0.6000000000000001
>>> 0.1 + (0.2 + 0.3)
0.6
```

The same thing happens in fp32, with numbers a model meets every day. Let
a = 1e8, b = 1 and c = −1e8:

<figure class="fx" data-fig="two-orders" aria-label="Three numbers added in two orders">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-label">fp32</span><span class="fx-chip">a = 1e8</span><span class="fx-chip">b = 1</span><span class="fx-chip">c = −1e8</span></div>
<div class="fx-rows">
<div class="fx-row"><span class="fx-label fx-mono">(a + b) + c</span><div class="fx-chain is-tight is-grid"><span class="fx-swap" data-at="1"><span class="fx-node is-mono" data-at="1" data-to="2">a + b = 100000001</span><span class="fx-node is-mono is-warn" data-at="2">a + b → 1e8<span class="fx-note">rounded</span></span></span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono" data-at="3">1e8 + c = 0</span><span class="fx-arrow" data-at="4"></span><span class="fx-node is-mono is-key" data-at="4">0</span></div></div>
<div class="fx-row"><span class="fx-label fx-mono">(a + c) + b</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">a + c = 0<span class="fx-note">exact</span></span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono" data-at="3">0 + b = 1</span><span class="fx-arrow" data-at="4"></span><span class="fx-node is-mono is-key" data-at="4">1</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>Take three fp32 numbers, a = 1e8, b = 1 and c = −1e8, and add them in two orders.</li>
<li>Each order first forms its own partial sum.</li>
<li data-hold="1.5">fp32 cannot represent a + b = 100000001, so the sum rounds to 1e8.</li>
<li>Each order then adds the remaining number.</li>
<li data-hold="2">The same three numbers give 0 in one order and 1 in the other.</li>
</ol>
<figcaption>Figure: (a + b) + c and (a + c) + b in fp32. The partial sums an order forms decide which roundings happen.</figcaption>
</figure>

Every addition rounds its result to the nearest number the format can
represent, and which roundings happen depends on which partial sums the order
forms. So:

**Rule.** The bits of a floating-point sum depend on the order of its
additions and on where it rounds. The same mathematics, done in another order,
can give a different number.

This is the general principle behind every page of the handbook: *the
reduction order is part of the result*. Every matrix multiply, norm, softmax,
attention, gradient sum and collective is a sum, so each of them has an
order, and two runs agree only if their orders agree.

## A kernel picks its reduction order from the shape of its input {#a-reductions-order-depends-on-shape}

A kernel does not derive the order of its additions from the mathematics. It
picks the order from the shape of the input it is given, so the arithmetic
changes when the batch changes. Take a dot product of four terms,
t = [1e8, 1, −1e8, 1], in fp32:

<figure class="fx" data-fig="three-orders" aria-label="One dot product reduced in three orders">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-label">t</span><span class="fx-chip">1e8</span><span class="fx-chip">1</span><span class="fx-chip">−1e8</span><span class="fx-chip">1</span></div>
<div class="fx-rows" style="--lw: 6.5rem; --cols: minmax(0, 1.2fr) auto minmax(0, 1fr) auto minmax(0, 1.2fr); --narrow: flex">
<div class="fx-row"><span class="fx-label">one program, left to right</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 + 1 = 1e8</span><span class="fx-arrow" data-at="2"></span><span class="fx-node is-mono" data-at="2">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono is-key" data-at="3" data-hot="4">0 + 1 = 1</span></div></div>
<div class="fx-row"><span class="fx-label">split-K, two halves</span><div class="fx-chain is-tight is-grid"><div class="fx-stack" data-at="1"><span class="fx-node is-mono" data-at="1">1e8 + 1 = 1e8</span><span class="fx-node is-mono" data-at="1">−1e8 + 1 = −1e8</span></div><span class="fx-arrow is-long" data-at="2" style="grid-column: 2 / 5"></span><span class="fx-node is-mono is-key" data-at="2" data-hot="4">1e8 + (−1e8) = 0</span></div></div>
<div class="fx-row"><span class="fx-label">atomics, in one possible order</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="2"></span><span class="fx-node is-mono" data-at="2">0 + 1 = 1</span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono is-key" data-at="3" data-hot="4">1 + 1 = 2</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>Reduce the same four terms, t = [1e8, 1, −1e8, 1], in three ways in fp32.</li>
<li>Each method does its first additions. Split-K sums its two halves at the same time, on two programs.</li>
<li>Split-K then adds its two partial sums and finishes with 0.</li>
<li>The other two methods finish: left to right gives 1, and this order of atomic additions gives 2.</li>
<li data-hold="2">Three orders give three results, 1, 0 and 2, and each is the correct rounding of its own order.</li>
</ol>
<figcaption>Figure: the same four fp32 terms, t = [1e8, 1, −1e8, 1], summed in three orders give three results.</figcaption>
</figure>

So three orders give three answers, 1, 0 and 2, and each is a correct
rounding of its own order. Three mechanisms decide which order a kernel uses:

| Mechanism | What it is | What decides the order |
|---|---|---|
| split-K | The kernel cuts the reduction dimension into pieces, separate programs sum the pieces, and then the partial sums are added | The library picks the number of pieces from the matrix shape, often from the number of rows |
| tiles | A program reduces its row in chunks one tile wide, then combines the chunks | The tile width and the number of chunks are chosen for each shape, and above some size the library may switch to a different kernel altogether |
| atomics | Programs add their partial sums into one memory location as each one finishes | The order in which the programs finish, which changes from run to run |

**Rule.** If a kernel's order depends on the shape, a row's result depends on
the batch it sits in. If the order depends on timing, the result can change
from one run to the next.

## Deterministic is not the same as invariant {#deterministic-is-not-invariant}

The two words make two different promises:

- **Deterministic** (run-to-run): the same inputs, at the same shapes, give
  the same bits on every run. Atomics break this promise.
- **Invariant** (here, batch-invariant): a row gives the same bits no matter
  what else is computed with it, whether it sits in a batch of 1 or of 64,
  first or last.

Here is a small example. A matrix multiply without atomics is deterministic.
With one row, the library splits the reduction into two halves (split-K); with
64 rows, it gives each output its own program. Run each case twice, and each
repeats its own bits exactly. But row 0 computed alone and row 0 computed
inside the batch of 64 used the two different orders shown above, so their
bits can differ.

`torch.use_deterministic_algorithms(True)` gives you the first promise but not
the second. A kernel can also be deterministic and still use an order that is
a detail of its implementation (for example, an order that depends on how many
indices it scatters). So two correct frameworks that are each deterministic do
not necessarily agree with each other.

**Rule.** Determinism removes variation over time: the same job repeats
itself. Invariance removes variation with context: a row's result does not
depend on what it is computed with. To align two different setups, you need
invariance.

## The five levels of invariance

Each level asks whether one thing can change while the bits stay the same:

| Level | It holds when | An example of how it breaks |
|---|---|---|
| L1 run-to-run | The same input gives the same bits on every run | A backward pass that adds gradient rows with atomics gives a different weight gradient on the second run |
| L2 batch | A row's result does not depend on the batch size, on the row's position or on the other rows | A GEMM uses split-K for one row but not for 64, so a sequence's logits change with the batch |
| L3 mode | Prefill, decode, chunked prefill and a paged KV cache give the same bits at each position | Decode splits the KV cache across programs and prefill does not, so the same token gets two different attention outputs |
| L4 cross-engine | The trainer and the inference engine give the same bits | The engine multiplies by the norm weight in fp32 and then casts to bf16, while the trainer casts first |
| L5 parallelism | Different TP, EP, SP or DP degrees give the same bits | At TP 2 the ranks add two partial sums; at TP 4 they add four, in a different tree |

Each scenario needs a different set of levels:

| Scenario | Levels it needs |
|---|---|
| Debugging pre-training (a replay, a config change, an upgrade) | L1 on both runs, and the same settings on every surface you control |
| Deterministic inference | L1 and L2, plus L3 if you use chunked prefill or prefix caching |
| RL, where training and inference must match | L1–L4, plus L5 if the two sides shard the model differently |
| Showing that a migration or an upgrade changes nothing | L1 and L4 between the old and the new stack, plus L5 if the parallelism changes |

Each level holds only within one **stack**: one GPU type, one driver and one
set of library versions. Across GPU generations, bitwise agreement is
generally not expected ([hardware coverage](../4-operations/hardware.md)).
Verify the levels on each stack separately; a bitwise comparison across stacks
is a separate claim that needs its own measurement. The
[alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md) sends
each scenario to the workflow that handles it.

## Which nondeterminism to remove, and which to leave visible {#controlled-and-residual-nondeterminism}

Remove only the **avoidable** variation, the kind that makes two correct runs
do different arithmetic:

- seeds and data order;
- the algorithms and numeric modes that libraries choose;
- the reduction trees of collectives;
- a checkpoint that is restored only in part.

Removing these gives **controlled** nondeterminism. Leave the rest, the
**residual** nondeterminism, visible. When a user or third-party kernel gives
a result that changes with scheduling (racy atomics, a missing barrier, a race
between streams), that is a bug, not noise. If you controlled it away, you
would hide the very fault you are hunting.

## How to compare two runs {#how-to-compare}

Every workflow repeats one step: comparing two runs. Doing it well takes three
choices: what counts as equal, where to compare, and how to summarize each
tensor.

**Count only identical bits as equal.** A tolerance cannot tell you where the
first difference is. A threshold loose enough to hide harmless differences
from reduction order also hides a few corrupted elements. On real production
bugs, comparing with a tolerance lost a large share of the exact
localizations (measured: see
[report](../../examples/pretraining_e2e/REPORT.md#overview-tolerances-against-bitwise-comparison)).
Control the avoidable variation instead, and then compare bits.

**Compare at model-level operator boundaries.** An operator boundary is a call
in the model code that starts work on the device, such as a projection, a
norm, an attention block or an expert combine. This is the right level for two
reasons:

- Whole steps or whole layers are too coarse to show where a difference
  starts.
- Individual kernels are too fine, because they change with every library
  version and every fusion choice.

Model-level boundaries exist in every faithful implementation of a model, so
two stacks can be compared there.

**Summarize each tensor with a fingerprint.** Dumping every tensor costs too
much, so each tensor is reduced to a small summary, its fingerprint. A
fingerprint must catch any single-bit change, must not let small corruptions
cancel each other out, and must not make the device synchronize. Here is a
small example with a tensor of four values:

<figure class="fx" data-fig="sum-xor" aria-label="A sum against an XOR fingerprint">
<div class="fx-stage">
<div class="fx-rows" style="--cols: minmax(0, 1.25fr) auto minmax(0, 1fr) minmax(0, 1fr); --cols-narrow: auto minmax(0, 1fr) minmax(0, 1fr)">
<div class="fx-row"><span class="fx-label">reference</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="1"><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span></div><span class="fx-arrow" data-at="1"></span><span class="fx-node" data-at="1">sum <span class="fx-mono">10</span></span><span class="fx-node" data-at="1">XOR of the raw 32-bit words <span class="fx-mono">w</span></span></div></div>
<div class="fx-row" data-at="2"><span class="fx-label">one value with one bit flipped</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="3"><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell is-bad">4 + ε</span></div><span class="fx-arrow" data-at="3"></span><span class="fx-node is-ok" data-at="3">sum <span class="fx-mono">changes</span><span class="fx-note">caught</span></span><span class="fx-node is-ok" data-at="3">XOR <span class="fx-mono">changes</span><span class="fx-note">caught</span></span></div></div>
<div class="fx-row" data-at="4"><span class="fx-label">two errors that cancel out</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="5"><span class="fx-cell is-bad">1 + ε</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell is-bad">4 − ε</span></div><span class="fx-arrow" data-at="5"></span><span class="fx-node is-bad" data-at="5">sum <span class="fx-mono">still 10</span><span class="fx-note">missed</span></span><span class="fx-node is-ok" data-at="5">XOR <span class="fx-mono">changes</span><span class="fx-note">caught</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>Start with a tensor of four values, [1, 2, 3, 4].</li>
<li>Its sum is 10, and the XOR of its raw 32-bit words is some value w.</li>
<li>Flip one bit in one value, so that 4 becomes 4 + ε.</li>
<li>Both summaries change, so either one catches the flip.</li>
<li>Now make two errors that cancel out: 1 becomes 1 + ε and 4 becomes 4 − ε.</li>
<li data-hold="2">The sum is 10 again and misses both errors, but the XOR still changes.</li>
</ol>
<figcaption>Figure: errors can cancel out in a sum, but an XOR over the raw bits never loses a single flipped bit.</figcaption>
</figure>

Errors can cancel out in a sum or a mean, but an XOR over the raw bits never
loses a single flipped bit. OpGuard computes XOR fingerprints on the device,
in two forms:

- **Regional** fingerprints, the default, split the flattened tensor into
  equal byte ranges and XOR each range. They are cheap, and they show which
  region of the tensor differs.
- **Per-vector** fingerprints, which you turn on when you need them, compute
  one XOR for each vector along the last dimension. Because XORs can be
  combined, the vectors of one token fold into a single fingerprint for that
  token's row, whatever the layout: `[T, heads, dim]` and `[T, hidden]` give
  the same row fingerprint. This makes rows comparable between stacks that
  batch, pack and decode differently, and when a row differs, the per-vector
  values show which head differs.

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
<li>One token's row is stored as [T, heads, dim], which makes four head vectors.</li>
<li>The per-vector form computes one XOR for each of them, f0 to f3.</li>
<li>Because XORs combine, the vector fingerprints fold together in pairs,</li>
<li>until they form the row's fingerprint r.</li>
<li>If the same row is stored as [T, hidden], it is a single vector, and its XOR is the same r.</li>
<li data-hold="2">So rows can be compared across layouts, and when two rows differ, the f that differs names the head.</li>
</ol>
<figcaption>Figure: per-vector fingerprints fold into the same row fingerprint, whatever the layout.</figcaption>
</figure>

The options, from the cheapest to the most complete:

| Option | What it tells you | Cost class | When to choose it |
|---|---|---|---|
| A regional fingerprint at every boundary | Which boundary differs first, and which region of it | small at each boundary (a few device kernels and no synchronization) | Always, and first |
| A per-vector fingerprint | Which row or head differs, in a form that can be compared across layouts | small to moderate | When the two stacks batch or pack differently |
| A full dump of one region's tensors | The values themselves, to diagnose the difference | large (2x or more), plus the storage | After a fingerprint has named the place |
| A tolerance on the values | Whether the values are close | free | Only to check accuracy, never to localize a difference |

Two tools compute these fingerprints: [opguard_bridge](../../tools/opguard_bridge.py),
which implements OpGuard's fingerprint formats in plain PyTorch, and OpGuard
itself. Part 1 uses both. You know the comparison works when two runs that
should agree give equal fingerprints at every boundary, and a one-bit change
planted on purpose is found at the boundary where it was planted.

**Tools record facts; agents and engineers make judgments.** Keep the tools
exact and generic, and put knowledge about particular cases into skills. A
tool computes fingerprints, records what it saw at known points and answers
whether two things are equal. An agent or an engineer decides which records
correspond, whether a missing row was fused away, which divergence comes first
and what caused it, and writes down each decision with its evidence. In our
experience, heuristics built into a tool to make these decisions broke on
every new case.

## What can go wrong

!!! warning "A tolerance hides the fault"
    A threshold that absorbs the noise from reduction order also absorbs small
    corruptions. You then localize the wrong place, or nothing at all. Control
    the avoidable nondeterminism and compare bit for bit.

!!! warning "Mistaking deterministic mode for invariance"
    Deterministic algorithms make a run repeat itself. They do not make a
    row's result independent of its batch or of the other framework, so two
    deterministic setups can still disagree. Check the level you need (L2, L4
    or L5) directly.

!!! warning "Treating equal fingerprints as equal values"
    XOR is a fingerprint, not a hash. Reordering the words leaves it
    unchanged, and so do two changes that cancel each other; across many
    tensors, 32-bit values will also sometimes collide. Count equal
    fingerprints as evidence only when their widths match and the value is not
    zero, and recapture the full tensors of any region you suspect.

!!! warning "Comparing two stacks as if they were one"
    The levels hold on one GPU type and one set of library versions. A bitwise
    difference between GPU generations is expected, not a bug. Verify each
    stack separately.

## Summary

- The reduction order is part of the result: three numbers added in two
  orders can give two different answers.
- Kernels pick their order from the shape (split-K, tiles) or leave it to
  timing (atomics), so a row's bits can depend on its batch or on the run.
- Deterministic (L1) is not the same as invariant (L2–L5); always say which
  level a claim holds at.
- Compare bit for bit, at model-level operator boundaries, using XOR
  fingerprints.
- Control the avoidable nondeterminism, and leave the residual nondeterminism
  visible, because it is the signal of a bug.

## Go deeper

- The [glossary](../glossary.md) defines every term used here.
- [Part 1](../1-pretraining/README.md) applies this comparison to two training
  runs.
- [Part 3](../3-kernels/README.md) shows how to write a kernel whose order is
  fixed.
- [opguard_bridge.py](../../tools/opguard_bridge.py) defines the fingerprint
  formats.
- [Report: tolerances against bitwise comparison](../../examples/pretraining_e2e/REPORT.md#overview-tolerances-against-bitwise-comparison).
- [alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md).
