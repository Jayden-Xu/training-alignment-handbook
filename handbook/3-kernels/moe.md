# MoE: routing, expert GEMMs and combine

## Where we are

Up to [linear attention](linear-attention.md), every reduction had a length
fixed by the element itself: K for a GEMM, the position for attention. A
mixture-of-experts layer adds routing that depends on the data. Which tokens
an expert GEMM sees changes with the batch, and each token's output is a sum of
k expert outputs that arrive in some order. This page builds on the
[GEMM page](gemm-and-quantized.md) for the experts and on the stable top-k from
[sparse attention](mla-and-sparse.md) for the router, then adds expert
parallelism, where tokens travel between GPUs.

## The idea: a token's output is a sum of k terms, and the order is ours to fix

A token's output is y = w1·e1 + w2·e2 + … + wk·ek, one term per selected
expert. Three things can make that sum depend on more than the token:

- **The expert's batch.** Expert i runs a GEMM over the M_i tokens routed to
  it, and M_i depends on the neighbours. With a library GEMM, the token's
  e_i changes with them.
- **The order of the adds.** A combine that adds each returned row with an
  atomic add sums in arrival order, which follows timing.
- **Dropping.** A capacity limit keeps the first tokens that reach an expert,
  so whether this token's term exists depends on its neighbours.

Order matters only from three terms on. Round to three significant digits:

| Terms | Grouping | Result |
|---|---|---|
| a = 1.00, b = 0.004 | a + b or b + a | 1.00 either way |
| a = 1.00, b = 0.004, c = 0.004 | (a + b) + c | 1.00 |
| a = 1.00, b = 0.004, c = 0.004 | a + (b + c) | 1.01 |

Two terms commute exactly, so a top-2 model hides every combine-order bug.

**Under expert parallelism** the experts sit on different GPUs and two
all-to-alls carry the rows out and back. They only move data, so they cannot
change bits. But a combine that first sums each GPU's terms and then adds the
per-GPU partials groups the sum by which experts share a GPU: ((a + b) + c) + d
on one GPU can become (a + c) + (b + d) on two. The grouping then depends on
the parallel degree (L5).

**The rule for MoE.** An invariant router with a stable top-k, dropless
routing, expert GEMMs with one configuration for every M, and a per-token sum
over the k terms in a fixed order (top-k rank) in fp32, read from fixed slots,
with no atomics.

## The steps

### 1. Route the same way for every batch

- **Goal:** a token's experts and weights depend only on the token.
- **How:** the router GEMM through the one-config kernel; top-k as a stable
  sort by (score descending, expert id ascending); weights renormalized in
  fp32 in top-k rank order; no capacity limit.
- **Success:** router ids and weights bitwise equal across batches and parallel
  degrees ([report](../../examples/moe_ep/REPORT.md#under-expert-parallelism)).

### 2. Make the expert GEMMs invariant

- **Goal:** e_i does not change with M_i.
- **How:** one one-config GEMM call per expert
  ([recipe 3](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)).
- **Success:** `check_batch` passes on every expert shape. Shown on two GPU
  types ([report](../../examples/moe_combine/REPORT.md#1-locate-the-variance)).

### 3. Combine in a fixed order

- **Goal:** the sum over a token's k terms has one order and one precision.
- **How:** give each returned row a fixed slot (token, top-k rank); one program
  per token reads its slots in rank order, accumulates in fp32 and rounds once.
  This is `fixed_slot` in [examples/moe_ep/moe_ep.py](../../examples/moe_ep/moe_ep.py);
  [recipe 8](../../skills/batch-invariant-kernel/references/operator-recipes.md#8-moe-routing-dispatch-and-combine).
- **Success:** L1, L2 and L5 pass. Shown on one GPU, and up to 16 GPUs on two
  nodes ([report](../../examples/moe_ep/REPORT.md#what-the-expert-parallel-results-show)).

| Combine | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Atomic scatter-add | fails L1 from top-3 | free | never for parity |
| Deterministic scatter-add over expert-sorted rows | L1, L2; order comes from the dispatcher's sort, not from the code | free | only if that sort is guaranteed |
| Per-GPU pre-reduce, then add per-GPU partials | L1, L2; fails L5 from top-3 | moderate | not when the parallel degree can differ |
| Fixed slots, per-token fp32 sum in rank order | L1, L2, L5 | free to moderate (no slower than atomics under expert parallelism) | the default fix |
| Library GEMM loop over fixed row pieces | L1, L2 | large | a correctness oracle only |

### 4. Verify with three or more terms

Run the [verification matrix](../../skills/batch-invariant-kernel/references/verification.md)
with top-k of at least 3, co-batched tokens on the same GPU, on other GPUs and
on both, several token-to-GPU shardings, and every expert-parallel degree you
deploy, compared with one GPU.

## What can go wrong

!!! warning "Top-2 hides combine-order bugs"
    Two terms commute exactly, so an atomic or per-GPU combine passes at top-2
    and fails from top-3. A top-2 test cannot certify a combine for larger k.

!!! warning "Capacity makes a token depend on its neighbours"
    With a capacity limit, the tokens next to this one decide which of its
    assignments survive. Every mismatch measured came with a changed drop set;
    use dropless routing for parity.

!!! warning "Deterministic is not parallel-invariant"
    A per-GPU pre-reduce gives the same bits on every run and changes with the
    expert-parallel degree. Test L5 separately.

!!! warning "A deterministic scatter-add inherits an undocumented order"
    It sums in the order rows sit in the buffer, which a different dispatcher
    can change. An explicit per-token sum states its order in the code.

**Not yet shown:** the backward under expert parallelism, a shared expert in
the same sum, fused dispatch-and-combine kernels, and agreement with a
trainer's MoE (L4).

## Summary

- A token's output sums k expert terms; fix the order (top-k rank) and the
  precision (fp32) yourself.
- Expert GEMMs see a batch-dependent M: use the one-config GEMM.
- No atomics, no capacity dropping, a stable top-k.
- Under expert parallelism return rows to fixed slots; the all-to-alls only
  move data.
- Test with top-k of at least 3.

## Go deeper

- Reports: [MoE on one GPU](../../examples/moe_combine/REPORT.md),
  [MoE under expert parallelism](../../examples/moe_ep/REPORT.md).
- Code: [examples/moe_combine/moe_bi.py](../../examples/moe_combine/moe_bi.py),
  [examples/moe_ep/moe_ep.py](../../examples/moe_ep/moe_ep.py).
- Recipes: [3, grouped GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm),
  [8, MoE routing and combine](../../skills/batch-invariant-kernel/references/operator-recipes.md#8-moe-routing-dispatch-and-combine).
- Skill: [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md).
