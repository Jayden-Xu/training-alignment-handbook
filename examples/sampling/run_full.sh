#!/usr/bin/env bash
# Full run of sampling_bi.py, one section per GPU lock, then the merge into one result file.
#   usage: PY=/path/to/python LOCK=/tmp/bi-check-a40.lock bash run_full.sh [out.json]
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${1:-results.json}
TMP=$(mktemp -d)
RUN=()
[ -n "$LOCK" ] && RUN=(flock "$LOCK")
for s in rng ops ties nucleus unit chisq l2 engine cost; do
  echo "=== $s $(date +%T)"
  "${RUN[@]}" "$PY" sampling_bi.py --sections "$s" --json "$TMP/$s.json" | grep -E "^--|FAIL|CHG|regressions" 
done
"$PY" sampling_bi.py --merge "$TMP"/{rng,ops,ties,nucleus,unit,chisq,l2,engine,cost}.json --json "$OUT"
rc=$?
rm -rf "$TMP"
exit $rc
