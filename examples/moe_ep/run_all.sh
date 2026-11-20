#!/bin/bash
# Full matrix for moe_ep.py on one 8-GPU host:
#  - three layer configs (E, top-k) = (32, 4), (32, 3), (16, 2): EP = 1, 2, 4, 8
#    on GPUs 0..W-1 (L1, L2, L5 dumps, timing; GEMM checks at EP = 1), plus
#    EP = 2 on GPUs 0 and 4 (across PCIe switches; L1 and L5);
#  - (32, 4) again with a 4096-token global batch, L5 only;
#  - the bitwise comparison into results-a16.json.
# Usage: bash run_all.sh [DUMPDIR] [PYTHON]
#   QUICK=1    the regression suite's reduced matrix: the same configs, EP degrees and GPU
#              placements, with moe_ep.py --quick (fewer L2 and GEMM-check sizes), no timing and
#              no 4096-token batch. The comparison goes to DUMPDIR/results.json.
#   RESULTS=F  write the comparison to F instead.
set -u
cd "$(dirname "$0")"
D=${1:-dumps}
PY=${2:-python}
QUICK=${QUICK:-}
RESULTS=${RESULTS:-$([ -n "$QUICK" ] && echo "$D/results.json" || echo results-a16.json)}
run() {  # run W GPUS args...
  local W=$1 G=$2
  shift 2
  echo "=== EP=$W GPUs $G $*"
  CUDA_VISIBLE_DEVICES=$G timeout 3600 "$PY" -m torch.distributed.run --nproc_per_node=$W moe_ep.py "$@"
}
if [ -n "$QUICK" ]; then
  for cfg in "32 4" "32 3" "16 2"; do
    set -- $cfg
    E=$1; K=$2
    for W in 1 2 4 8; do
      run $W "$(seq -s, 0 $((W - 1)))" --E $E --K $K --dump "$D" --quick --parts gemm,l1,l2,l5
    done
    run 2 0,4 --E $E --K $K --dump "$D" --tag x04 --quick --parts l1,l5
  done
else
for cfg in "32 4" "32 3" "16 2"; do
  set -- $cfg
  E=$1; K=$2
  for W in 1 2 4 8; do
    run $W "$(seq -s, 0 $((W - 1)))" --E $E --K $K --dump "$D"
  done
  run 2 0,4 --E $E --K $K --dump "$D" --tag x04 --parts l1,l5
done
for W in 1 2 4 8; do
  run $W "$(seq -s, 0 $((W - 1)))" --E 32 --K 4 --tg 4096 --dump "$D" --parts l5
done
fi
echo "=== compare"
timeout 1800 "$PY" moe_ep.py --compare "$D" --json "$RESULTS"
