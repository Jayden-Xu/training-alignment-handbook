#!/usr/bin/env bash
# Every launch of collectives_tp.py on one 8-GPU node, then the merge into results-a16.json.
# NCCL reads NCCL_ALGO / NCCL_PROTO at init, so each setting is its own launch.
# The `coll` part logs NCCL's per-call algorithm choice (NCCL_DEBUG_SUBSYS=TUNING,COLL);
# the `tp` part, which also does the timing, runs without NCCL logging.
#   usage: PY=/path/to/python bash run_all.sh [outdir] [extra args, e.g. --quick]
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${1:-out}
shift || true
EXTRA=("$@")
mkdir -p "$OUT"
ABS=$(cd "$OUT" && pwd)

run() {  # tag part [NCCL env assignments...]
  local tag=$1 part=$2
  shift 2
  local dbg=()
  if [ "$part" = coll ]; then
    dbg=(NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,COLL "NCCL_DEBUG_FILE=$ABS/$tag-coll.nccl.%h.%p.log")
  fi
  rm -f "$ABS/$tag-$part.json"
  if [ "$part" = coll ]; then rm -f "$ABS/$tag-coll.nccl."*.log; fi
  echo "=== $tag $part $* $(date +%T)"
  env "$@" "${dbg[@]}" timeout 1800 "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
    collectives_tp.py --part "$part" --tag "$tag" --out "$OUT" "${EXTRA[@]}" > "$OUT/$tag-$part.stdout" 2>&1
  echo "exit $? $(date +%T)"
  grep -vE '^\*|OMP_NUM_THREADS|^WARNING:__main__:$|^$' "$OUT/$tag-$part.stdout" | tail -n 40
  if [ "$part" = coll ]; then   # keep rank 0's NCCL log only
    for f in "$ABS/$tag-coll.nccl."*.log; do grep -q ' \[0\] NCCL INFO' "$f" || rm -f "$f"; done
  fi
}

run default      coll
run ring-simple  coll NCCL_ALGO=Ring NCCL_PROTO=Simple
run ring-ll      coll NCCL_ALGO=Ring NCCL_PROTO=LL
run ring-ll128   coll NCCL_ALGO=Ring NCCL_PROTO=LL128
run tree         coll NCCL_ALGO=Tree
run tree-simple  coll NCCL_ALGO=Tree NCCL_PROTO=Simple
run tree-ll      coll NCCL_ALGO=Tree NCCL_PROTO=LL
run default      tp
run ring-simple  tp   NCCL_ALGO=Ring NCCL_PROTO=Simple
run tree         tp   NCCL_ALGO=Tree

"$PY" collectives_tp.py --merge "$OUT" --json "$OUT/results-a16.json"
