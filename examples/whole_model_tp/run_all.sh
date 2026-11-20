#!/bin/bash
# Full mode on one 8-GPU node, in pieces under the shared multi-GPU lock, then one merged file.
# Usage: bash run_all.sh [torchrun]   (from examples/whole_model_tp; writes parts/ and results-a16.json)
# Two nodes (TP 16): bash run_2node.sh on the first node. ONLY="disagg cost" bash run_all.sh reruns some parts.
set -e
TR=${1:-torchrun}
PY=$(dirname "$(command -v "$TR")")/python
# the node lock (handbook/4-operations/regression.md, "Machines and locks")
LOCK="flock ${NODE_LOCK:-/tmp/bi-check-16gpu.lock} timeout 1200"
mkdir -p parts
run() { tag=$1; shift; [ -n "$ONLY" ] && [[ " $ONLY " != *" $tag "* ]] && return; echo "== $tag"; $LOCK $TR --nproc_per_node=8 whole_model_tp.py "$@" --json parts/$tag.json > parts/$tag.log 2>&1 || echo "   (exit $?)"; grep -E "ok, .* regressions" parts/$tag.log | tail -1; }
run tp       --only tp
run modes    --only modes
run disagg   --only disagg,backward
run cost     --only cost
$PY whole_model_tp.py --merge parts/tp.json parts/modes.json parts/disagg.json parts/cost.json \
    $( [ -f parts/x16.json ] && echo parts/x16.json ) --json results-a16.json | tail -1
