# MoE: making routing, the expert GEMMs and the combine invariant

## Where we are

Up to [linear attention](linear-attention.md), the length of every reduction
was fixed by the element itself: K for a GEMM, the position for attention. A
mixture-of-experts layer adds routing that depends on the data. Which tokens
an expert's GEMM sees changes with the batch, and each token's output is a
sum of k expert outputs that arrive in some order. This page relies on the
[GEMM page](gemm-and-quantized.md) for the experts and on the stable top-k
from [sparse attention](mla-and-sparse.md) for the router. It then adds
expert parallelism, where tokens travel between GPUs.

## The idea: a token's output is a sum of k terms, and you choose their order

A token's output is y = w1·e1 + w2·e2 + … + wk·ek, with one term for each
selected expert. Three things can make that sum depend on more than the token
itself:

- **The expert's batch.** Expert i runs a GEMM over the M_i tokens routed to
  it, and M_i depends on the other tokens in the batch. With a library GEMM,
  this token's e_i changes when they change.
- **The order of the adds.** A combine that adds each returned row with an
  atomic add sums the rows in the order they arrive, and that order follows
  timing.
- **Dropping.** A capacity limit keeps the first tokens that reach an expert,
  so whether this token's term exists at all depends on the other tokens.

The order matters only once there are three or more terms. To see why, round
every result to three significant digits:

| Terms | Grouping | Result |
|---|---|---|
| a = 1.00, b = 0.004 | a + b or b + a | 1.00 either way |
| a = 1.00, b = 0.004, c = 0.004 | (a + b) + c | 1.00 |
| a = 1.00, b = 0.004, c = 0.004 | a + (b + c) | 1.01 |

Two terms can be swapped without changing the result, so a top-2 model hides
every bug in the order of the combine.

**Under expert parallelism** the experts sit on different GPUs, and two
all-to-alls carry the rows out to the experts and back. The all-to-alls only
move data, so they cannot change any bits. But consider a combine that first
sums the terms on each GPU and then adds the partial sums from the GPUs. It
groups the sum by which experts share a GPU, so ((a + b) + c) + d on one GPU
can become (a + c) + (b + d) on two. The grouping then depends on the
parallel degree (L5).

**The rule for MoE.** Use an invariant router with a stable top-k, route
without dropping tokens, and run the expert GEMMs with one configuration for
every M. Then sum each token's k terms in a fixed order (the top-k rank) in
fp32, reading them from fixed slots, without atomics.

## The steps

### 1. Route each token the same way in every batch

- **Goal:** a token's experts and their weights depend only on the token.
- **How:** run the router GEMM with the one-configuration kernel, take the
  top-k with a stable sort by (score descending, expert id ascending),
  renormalize the weights in fp32 in top-k rank order, and set no capacity
  limit.
- **Success:** the router's expert ids and weights are bitwise equal across
  batches and parallel degrees
  ([report](../../examples/moe_ep/REPORT.md#under-expert-parallelism)).

### 2. Make the expert GEMMs invariant

- **Goal:** e_i does not change when M_i changes.
- **How:** call the one-configuration GEMM once per expert
  ([recipe 3](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)).
- **Success:** `check_batch` passes on every expert shape. This was shown on
  two GPU types ([report](../../examples/moe_combine/REPORT.md#1-locate-the-variance)).

### 3. Add each token's expert outputs in a fixed order

- **Goal:** the sum over a token's k terms always has the same order and the
  same precision.
- **How:** give each returned row a fixed slot, indexed by (token, top-k
  rank). One program per token reads its slots in rank order, accumulates them
  in fp32 and rounds once at the end. This is `fixed_slot` in
  [examples/moe_ep/moe_ep.py](../../examples/moe_ep/moe_ep.py); see
  [recipe 8](../../skills/batch-invariant-kernel/references/operator-recipes.md#8-moe-routing-dispatch-and-combine).
- **Success:** L1, L2 and L5 pass. This was shown on one GPU, and on up to 16
  GPUs across two nodes
  ([report](../../examples/moe_ep/REPORT.md#what-the-expert-parallel-results-show)).

| How the combine adds | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Atomic scatter-add | L1 fails from top-3 on | free | never, if you need parity |
| Deterministic scatter-add over rows sorted by expert | L1, L2; the order comes from the dispatcher's sort, not from the code | free | only if that sort is guaranteed |
| Reduce on each GPU first, then add the partial sums of the GPUs | L1, L2; L5 fails from top-3 on | moderate | not when the parallel degree can change |
| Fixed slots, with a per-token fp32 sum in rank order | L1, L2, L5 | free to moderate (no slower than atomics under expert parallelism) | the default fix |
| Call the library GEMM in a loop over fixed pieces of rows | L1, L2 | large | only as a correctness oracle |

### 4. Verify with at least three terms per token

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
with a top-k of at least 3. Put the other tokens of the batch on the same GPU,
on other GPUs and on both, try several ways of sharding the tokens across
GPUs, and test every expert-parallel degree you deploy, always comparing with
one GPU.

## What can go wrong

!!! warning "Top-2 hides bugs in the combine order"
    Two terms can be swapped without changing the result, so an atomic or
    per-GPU combine passes at top-2 and fails from top-3 on. A test at top-2
    cannot certify a combine for larger k.

!!! warning "A capacity limit makes a token depend on the other tokens"
    With a capacity limit, the tokens around this one decide which of its
    expert assignments survive. Every mismatch we measured came with a change
    in the set of dropped tokens, so route without dropping when you need
    parity.

!!! warning "Deterministic does not mean invariant to the parallel degree"
    Reducing on each GPU first gives the same bits on every run, but the bits
    change with the expert-parallel degree. Test L5 separately.

!!! warning "A deterministic scatter-add relies on an order nobody wrote down"
    It sums the rows in the order they sit in the buffer, and a different
    dispatcher can change that order. An explicit per-token sum states its
    order in the code.

**Not yet shown:** the backward pass under expert parallelism, a shared
expert inside the same sum, fused dispatch-and-combine kernels, and agreement
with a trainer's MoE (L4).

## Summary

- A token's output is a sum of k expert terms; fix the order (the top-k
  rank) and the precision (fp32) yourself.
- Each expert GEMM sees an M that depends on the batch, so use the
  one-configuration GEMM.
- Use no atomics, drop no tokens for capacity, and use a stable top-k.
- Under expert parallelism, return the rows to fixed slots; the all-to-alls
  themselves only move data.
- Test with a top-k of at least 3.

## Go deeper

- Reports: [MoE on one GPU](../../examples/moe_combine/REPORT.md) and
  [MoE under expert parallelism](../../examples/moe_ep/REPORT.md).
- Code: [examples/moe_combine/moe_bi.py](../../examples/moe_combine/moe_bi.py),
  [examples/moe_ep/moe_ep.py](../../examples/moe_ep/moe_ep.py).
- Recipes: [3, grouped GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm),
  [8, MoE routing and combine](../../skills/batch-invariant-kernel/references/operator-recipes.md#8-moe-routing-dispatch-and-combine).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
