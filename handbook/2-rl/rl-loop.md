# The RL loop: records, the staleness split, offline localization

## Where we are

After [engine parity](engine-parity-vllm.md) the trainer and the engine agree
on a fixed prompt set. A running RL loop adds what that test did not have:
weights pushed at every version, asynchronous rollouts at older versions, a
batch that changes every step. This step keeps watch over the loop: record
cheaply online, split every gap into staleness and numeric error, and localize
offline only what is a real infrastructure fault. These are steps 7–9 of the
[RL workflow](README.md).

## The idea

The [workflow page](README.md)
splits a gap exactly:

```
gap       = logp_rollout(v) - logp_trainer(c)
numeric   = logp_rollout(v) - logp_trainer(v)   # same weights: an infrastructure fault, should be 0
staleness = logp_trainer(v) - logp_trainer(c)   # policy lag: corrected by IS / TIS / masking
```

Three things decide whether the split can be trusted, and they set the order
of the step:

1. **The weights must be the ones claimed.** If the engine holds a wrong row
   after a weight push, every comparison at that version is meaningless. Check
   the weights before any logprob.
2. **A token must belong to one version.** A sequence that was generating when
   new weights arrived, with its KV cache kept, has tokens from two policies.
   It cannot be replayed under either, so it is set aside.
3. **A divergence needs evidence from both sides.** When a nonzero numeric
   error is walked through the model, a row that has no partner on the other
   side is a gap (a step the other side fused away), not a divergence. A
   divergence is the same input row on both sides giving a different output
   row.

Localization walks the data flow, not the trace order. Start from a slot both
sides agree on, such as the embedding output, and follow each tensor to its
consumer until the first pair with equal input and different output.

<figure class="fx" data-fig="rl-triage" aria-label="Triage of a gap in a running RL loop">
<div class="fx-stage">
<div class="fx-spine">
<span class="fx-node" data-hot="0">records (online)</span><span></span>
<span class="fx-arrow is-down" data-hot="1" aria-hidden="true"></span><span></span>
<span class="fx-node is-decision" data-hot="1">weights match?</span><span class="fx-side" data-at="2"><span class="fx-arrow">no</span><span class="fx-node is-bad">fix the sync; stop</span></span>
<span class="fx-arrow is-down" data-hot="3">yes</span><span></span>
<span class="fx-node" data-hot="3 4">split each token: numeric, staleness</span><span class="fx-side"><span class="fx-swap is-block" data-at="3"><span class="fx-split" data-to="4"><span class="is-whole">gap 0.15</span></span><span class="fx-split" data-at="4" style="--a:1;--b:14"><span class="is-a">numeric 0.01</span><span class="is-b">staleness 0.14</span></span></span></span>
<span class="fx-arrow is-down" data-hot="6">numeric ≠ 0</span><span class="fx-side" data-at="5"><span class="fx-arrow">numeric = 0</span><span class="fx-node is-ok">infrastructure exonerated</span></span>
<span class="fx-node" data-at="6">recapture both sides, walk the data flow</span><span></span>
<span class="fx-arrow is-down" data-at="7" aria-hidden="true"></span><span></span>
<span class="fx-node is-key" data-at="7">classify the cause, test the smallest fix</span><span></span>
</div>
</div>
<ol class="fx-say">
<li>Records come from the running loop, with a fingerprint of the weights each batch used.</li>
<li>First: did the engine hold the weights the trainer had at that version?</li>
<li>If not, every comparison at that version is meaningless: fix the sync and stop.</li>
<li>If so, take each token's gap; here 0.15.</li>
<li data-hold="2">A replay at the rollout's own version splits it exactly: numeric 0.01 plus staleness 0.14.</li>
<li data-hold="1.5">Had numeric been 0, the gap would be policy lag alone: the infrastructure exonerated.</li>
<li data-hold="1.5">numeric ≠ 0: recapture both sides and walk the data flow, not the trace order.</li>
<li data-hold="2">Classify the cause and test the smallest fix.</li>
</ol>
<figcaption>Figure: check the weights first, split every gap at the rollout's version, and localize only a nonzero numeric error.</figcaption>
</figure>

## The step

OpGuard's RL harness implements every action below. Records are written with
`opguard_harness.rl_records.RecordWriter`; the actions (`rl.*` on record files,
`vectors.*` on captures) report facts and return `verdict: not_evaluated`
without applying a tolerance. The protocol and every field are in OpGuard's
`docs/RLAlignment.md`, and the OpGuard skill `rl-alignment`
(`harness/skills/rl-alignment`) runs the steps as an agent workflow.

1. **Record online.**
   - Goal: enough to split and replay any sampled batch later, cheaply.
   - How: `RecordWriter` records `rollout_batch` (token and prompt ids,
     positions, each token's policy version, rollout logprob bits, batch
     composition, a fingerprint of the weights the engine held), `forward`
     (each forward and its row layout), `weight_sync` (per-row weight
     fingerprints on both sides at every sync), `snapshot` (where version
     weights are kept) and, at staleness zero, `trainer_logprob`.
   - Success: the records summary lists every type, and snapshot versions are
     sampled so their cost stays bounded.
2. **Check the weights.**
   - Goal: the engine ran the weights the trainer pushed.
   - How: `rl.weight_sync_check` compares row fingerprints as multisets, so a
     different packing (for example a fused QKV) is not a difference.
   - Success: every compared version clean; otherwise the faulty parameter and
     row are named on each side.
3. **Split the gap.**
   - Goal: numeric and staleness per token.
   - How: `rl.decompose`, in exact arithmetic; it flags mixed-version tokens
     and says which recorded weight fingerprint each batch matches. For older
     versions `rl.replay_request` builds the replay (snapshot of version v, the
     same tokens, the original or a batch-invariant composition) for your
     workload to run offline.
   - Success: numeric zero on every non-mixed token, or the list of tokens
     where it is not.
4. **Localize offline.**
   - Goal: the first operator where numeric error appears.
   - How: recapture both sides with per-vector fingerprints (OpGuard
     `OPGUARD_XOR_ROWS=1`), list forwards with `rl.forwards`, query with
     `vectors.describe`, `vectors.lookup`, `vectors.coverage` and
     `vectors.compare`, starting from an agreed anchor and walking the data
     flow. Engines running compiled code or CUDA graphs are fingerprinted with
     [tools/graph_probe.py](../../tools/graph_probe.py)
     ([when the job runs compiled code](../1-pretraining/capture-compiled-code.md)).
   - Success: a paired candidate: same input row, different output row, on
     both sides.
5. **Classify and test the cause.**
   - Goal: a cause class, confirmed by an intervention.
   - How: rounding point or fusion order; batch dependence; an algorithmic
     choice; parallel layout; a different logprob formula; version bookkeeping
     or weight sync. Make the smallest change that makes numeric zero at that
     operator and replay.
   - Success: the replay with the change equals the other side.
6. **Keep it.**
   - Goal: the fix stays fixed.
   - How: add the check to the [regression](../4-operations/regression.md)
     suite with its baseline.
   - Success: a later version must reproduce the result.

| Option | What it gives | Cost class | When to choose it |
|---|---|---|---|
| Parity on, synchronous, short window | A gap of exactly 0 per token, or a failure to localize | the parity configuration's cost, for the window only | Before a long run, or to exonerate the infrastructure |
| Records and fingerprints every step, snapshots sampled | The split for any recorded batch, offline | small (a few percent) without snapshots; snapshots grow with model size | Every production run you may need to audit |
| Offline replay and recapture | Localization of one token's numeric error | seconds per localization, off the training path | Only for tokens with nonzero numeric error |

Shown: in a small loop with batch-invariant op sets on both sides, numeric
error was bit-zero on every non-mixed token, synchronous and asynchronous, so
staleness carried the whole gap; the weight check named a planted bad row
through a fused repack; planted code faults were localized to their operator
in a known-answer walk, whose data-flow order was written from the model code
rather than discovered ([report](../../examples/rl_loop/REPORT.md)).
Not yet shown: a large model, TP/EP on either side, CUDA graphs, a production
engine in the loop.

## What can go wrong

!!! warning "Engine flags are claims"
    A rollout's recorded flags can say batch-invariant while a stock op set
    runs. Trusting them sends the search to the wrong side. Treat flags as
    claims and let the numeric split decide.

!!! warning "A version-label fault looks like a numeric error"
    A batch labelled with one version may have run on another version's
    weights. Its numeric error is nonzero online, yet a replay at the recorded
    version equals the trainer and finds no differing row. Compare the batch's
    weight fingerprint with every sync version, then replay at the matching one.

!!! warning "Order-based pairing is silent across shapes"
    A trainer prefill and many engine decode forwards call functions a
    different number of times, so a diff that pairs events by order can report
    nothing. Silence there is not evidence. Compare per-vector fingerprints
    under declared token dimensions.

!!! warning "Zero fingerprints are not agreement"
    Some rows legitimately XOR to zero (repeated halves, for instance at
    position 0 under grouped-query attention). Counting them as equal stops a
    walk too early. Treat zero-only agreement as no evidence.

!!! warning "A fused producer has no partner"
    A fused projection's output row is wider than any single trainer row. A
    walk that requires equal widths finds no pair. Compare it at its first
    consumer after the split.

!!! warning "A replay reproduces only under a stated composition"
    A replayed rollout gives its original numbers only with the recorded batch
    composition, or with batch-invariant kernels. Without either, the replay
    differs for reasons unrelated to the fault. Say which one you used.

## Summary

- Record cheaply online; snapshot versions only as often as you can afford.
- Check the weights first, then split every gap; set mixed-version tokens
  aside.
- Numeric zero exonerates the infrastructure for those batches; staleness
  goes to the importance-sampling correction.
- Localize nonzero numeric error offline along the data flow, with evidence
  from both sides, and confirm the cause by an intervention.

## Go deeper

- [Lab report](../../examples/rl_loop/REPORT.md): the loop, every scenario,
  the planted faults, the harness findings and their fixes, cost.
- Code: [rl_core.py](../../examples/rl_loop/rl_core.py),
  [rl_loop.py](../../examples/rl_loop/rl_loop.py),
  [tools/opguard_bridge.py](../../tools/opguard_bridge.py),
  [tools/graph_probe.py](../../tools/graph_probe.py).
- A single-element localization against a real engine:
  [report](../../examples/cross_engine/REPORT.md#worked-example-one-element-at-q_norm).
- OpGuard: `docs/RLAlignment.md` and the `rl-alignment` skill.
- Next: [regression](../4-operations/regression.md).
