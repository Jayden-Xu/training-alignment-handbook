#!/bin/bash
# Usage: examples/rl_loop/run.sh [--quick] [args...]   (from the handbook root, on a host with OpGuard)
# OPGUARD_ROOT: an OpGuard source checkout with its native XOR extension built.
set -e
OPGUARD_ROOT=${OPGUARD_ROOT:-$HOME/bi-check-rl-loop/OpGuard}
PY=${PY:-$HOME/miniforge3/envs/torchdbg/bin/python}
export OPGUARD_ROOT OPGUARD_VERBOSE=0 PYTHONPATH=$OPGUARD_ROOT:$OPGUARD_ROOT/harness
exec "$PY" "$(dirname "$0")/rl_loop.py" "$@"
