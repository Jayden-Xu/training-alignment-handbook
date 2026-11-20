#!/bin/bash
# Full mode in pieces, each under the shared-GPU lock and under about 5 minutes, then one merged file.
# Usage: bash run_all.sh [python]   (from examples/whole_model; writes parts/ and results-a40.json)
set -e
PY=${1:-python}
LOCK="flock /tmp/bi-check-a40.lock"
mkdir -p parts
run() { tag=$1; shift; echo "== $tag"; $LOCK $PY whole_model.py "$@" --json parts/$tag.json > parts/$tag.log 2>&1 || echo "   (exit $?)"; tail -1 parts/$tag.log; }
run dense-kernels  --variants dense --only kernels,spread
run dense-l1l2     --variants dense --only l1,l2
run dense-modes    --variants dense --only modes
run dense-lengths  --variants dense --only lengths,rounding
run dense-backward --variants dense --only backward
run dense-cost     --variants dense --only cost
run moe-fwd        --variants moe --only spread,l1,l2,lengths
run moe-modes      --variants moe --only modes
run moe-rest       --variants moe --only backward,cost
run wide-kernels   --variants wide --only kernels,spread,l1,l2
run wide-modes     --variants wide --only modes,lengths
run wide-rest      --variants wide --only backward,cost
run qwen2          --variants qwen2 --only spread,l2,modes,lengths
PYTHONPATH= flock /tmp/bi-check-a40.lock ${AUDIT_PY:-$PY} audit_model.py --out . > parts/audit.log 2>&1 || echo "   audit exit $?"; grep first parts/audit.log
$PY whole_model.py --merge parts/*.json --json results-a40.json | tail -1
