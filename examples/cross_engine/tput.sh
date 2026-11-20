#!/bin/bash
# Throughput of each engine configuration, one at a time on one GPU (nothing else of ours running).
# usage: CUDA_VISIBLE_DEVICES=N bash tput.sh OUTDIR
set -u
OUT=${1:?outdir}; mkdir -p "$OUT"
PY=~/miniconda3/envs/vllm-l4/bin/python
cd "$(dirname "$0")"
run() { name=$1; shift; $PY engine_run.py --name "$name" --out "$OUT/tput-$name.pt" --nprompts 1 --gen 1 --batches 1 --tput "$@" 2>&1 | grep "tput\|wrote"; }
run default
run eager --eager
run bi --bi --backend FLASH_ATTN
run bi_eager --bi --eager --backend FLASH_ATTN
run bi_eager_lmhead --bi --eager --backend FLASH_ATTN --patch-lmhead
