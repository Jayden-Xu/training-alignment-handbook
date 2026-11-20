#!/bin/bash
# TP = 16 on two 8-GPU nodes (trainer forward against TP = 1 and 8, and cost), through the two-node
# launcher, which takes the 16-GPU lock. Run on the first node from examples/whole_model_tp of a copy
# at the same absolute path on both nodes:  bash run_2node.sh <second-node-ssh-host> [launch2.sh]
# Writes parts/x16.json; run_all.sh then merges it into results-a16.json.
set -e
NODE1=$1
LAUNCH=${2:-$HOME/bi-check-xnode/launch2.sh}
ROOT=$(cd ../.. && pwd)
FILES="tools/bi_check.py tools/bi_kernels.py tools/bi_report.py examples/whole_model/model.py examples/whole_model/whole_model.py
       examples/whole_model_tp/tp_model.py examples/whole_model_tp/whole_model_tp.py"
ssh -o BatchMode=yes "$NODE1" "mkdir -p $ROOT/examples/whole_model_tp/parts"
(cd "$ROOT" && rsync -az --relative $FILES "$NODE1:$ROOT/")
mkdir -p parts
# launch2.sh takes the node lock on both nodes (handbook/4-operations/regression.md, "Machines and locks")
TIMEOUT=1200 "$LAUNCH" "$(pwd)" whole_model_tp.py --degrees 1,8,16 --only tp,cost --json parts/x16.json
