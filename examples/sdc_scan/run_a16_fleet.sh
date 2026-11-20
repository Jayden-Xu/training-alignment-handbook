#!/bin/bash
# Fleet scan of two 8x A16 hosts, driven from node0 (which can SSH to node1), under the
# shared 16-GPU lock. Each host runs self-replay, NCCL pairs and the self-test on its 8 GPUs
# at the same time; the parts are then judged together (fleet = devices of one group).
#   usage (on node0, from ~/bi-check-sdc): bash run_a16_fleet.sh [outdir] [extra sdc_scan args]
set -u
cd "$(dirname "$0")"
OUT=${1:-out}; shift || true
PY=${PY:-python}
WD=$(pwd)
PEER=${PEER:?set PEER to the second node}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30"
mkdir -p "$OUT"
exec 9>/tmp/bi-check-16gpu.lock
echo "waiting for /tmp/bi-check-16gpu.lock $(date +%T)"
flock 9
echo "lock held $(date +%T)"
t0=$(date +%s)
$SSH "$PEER" "mkdir -p $WD/$OUT && cd $WD && timeout 1500 $PY tools/sdc_scan.py --node node1 --part $OUT/node1.json $* > $OUT/node1.log 2>&1; echo node1 rc \$?" &
peer=$!
timeout 1500 $PY tools/sdc_scan.py --node node0 --part "$OUT/node0.json" "$@" > "$OUT/node0.log" 2>&1; echo "node0 rc $?"
wait $peer
for i in 1 2 3 4 5; do scp -q -o BatchMode=yes "$PEER:$WD/$OUT/node1.json" "$PEER:$WD/$OUT/node1.log" "$OUT/" && break; sleep 3; done
echo "scan wall $(( $(date +%s) - t0 )) s"
flock -u 9
$PY tools/sdc_scan.py --merge "$OUT/node0.json" "$OUT/node1.json" --cross-group-info --json "$OUT/results-a16-fleet.json" \
    --record-golden "$OUT/golden-a16.json" > "$OUT/merge.log" 2>&1; echo "merge rc $?"
tail -45 "$OUT/merge.log"
