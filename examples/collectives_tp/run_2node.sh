#!/usr/bin/env bash
# Two-node runs (node0 = node 0, node1 = node 1; 16 ranks) of collectives_tp.py --xnode, each launch
# through ~/bi-check-xnode/launch2.sh, which takes the shared 16-GPU lock. Run on node0 from this
# directory; the same tree must exist on node1 at the same absolute path.
#   usage: bash run_2node.sh [outdir] ["tag/part tag/part ..."]     (default: every launch, then merge)
#   XARGS="--quick" adds arguments to every launch.
cd "$(dirname "$0")"
WD=$(pwd)
OUT=${1:-out2n}
STEPS=${2:-"default/coll ring-simple/coll ring-ll/coll ring-ll128/coll tree/coll tree-simple/coll tree-ll/coll default/tp tree/tp"}
L=~/bi-check-xnode/launch2.sh
PY=${PY:-python}
mkdir -p "$OUT"
ssh -o BatchMode=yes "${PEER:?set PEER to node 1}" "mkdir -p $WD/$OUT"

declare -A PIN=(
  [default]=""
  [ring-simple]="NCCL_ALGO=Ring NCCL_PROTO=Simple"
  [ring-ll]="NCCL_ALGO=Ring NCCL_PROTO=LL"
  [ring-ll128]="NCCL_ALGO=Ring NCCL_PROTO=LL128"
  [tree]="NCCL_ALGO=Tree"
  [tree-simple]="NCCL_ALGO=Tree NCCL_PROTO=Simple"
  [tree-ll]="NCCL_ALGO=Tree NCCL_PROTO=LL"
)

for step in $STEPS; do
  tag=${step%/*}; part=${step#*/}
  rm -f "$OUT/$tag-$part.json"
  [ "$part" = coll ] && rm -f "$OUT/$tag-coll.nccl."*.log
  echo "=== $tag $part ${PIN[$tag]} $(date +%T)"
  (
    for kv in ${PIN[$tag]}; do export "$kv"; done
    if [ "$part" = coll ]; then
      export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,COLL
      export NCCL_DEBUG_FILE="$WD/$OUT/$tag-coll.nccl.%h.%p.log"
      export TIMEOUT=${TIMEOUT_COLL:-420}
    else
      export TIMEOUT=${TIMEOUT_TP:-600}
    fi
    "$L" "$WD" collectives_tp.py --xnode --part "$part" --tag "$tag" --out "$OUT" ${XARGS:-}
  ) > "$OUT/$tag-$part.stdout" 2>&1
  echo "exit $? $(date +%T)"
  grep -vE '^\*|OMP_NUM_THREADS|^WARNING:__main__:$|^$|ProcessGroupNCCL' "$OUT/$tag-$part.stdout" | tail -n 30
  cp node1.log "$OUT/$tag-$part.node1.log" 2>/dev/null
  if [ "$part" = coll ]; then     # keep rank 0's NCCL log only
    for f in "$OUT/$tag-coll.nccl."*.log; do grep -q ' \[0\] NCCL INFO' "$f" || rm -f "$f"; done
  fi
done

"$PY" collectives_tp.py --merge "$OUT" --json "$OUT/results-a16-2node.json" --compact 30
