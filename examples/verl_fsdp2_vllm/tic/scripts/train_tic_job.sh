#!/usr/bin/env bash
set -euo pipefail
: "${CASE2_TIC_ROOT:?}" "${CASE2_JOB:?}" "${CASE2_PHASE:?}" "${CASE2_ARM:?}"
export CASE2_RUN_DIR="$CASE2_TIC_ROOT/results/$CASE2_JOB"
mkdir "$CASE2_RUN_DIR"
export CASE2_STATUS="$CASE2_RUN_DIR/STATUS"
export CASE2_TIC_RUN="$CASE2_RUN_DIR"
exec >"$CASE2_RUN_DIR/run.log" 2>&1
printf 'phase=%s\nstep=initializing\nmismatch=n/a\nKL=n/a\n' "$CASE2_PHASE" >"$CASE2_STATUS"
finish() {
    rc="$?"
    trap - EXIT
    {
        cat "$CASE2_STATUS"
        printf 'exit_code=%s\n' "$rc"
        if [[ "$rc" != 0 ]]; then
            summary="$(grep -E '(Error:|Exception:)' "$CASE2_RUN_DIR/run.log" | sed -n '$p' | cut -c1-400 || true)"
            printf 'error=%s\n' "${summary:-see run.log for error context}"
        fi
    } >"$CASE2_RUN_DIR/STATUS.finish"
    mv "$CASE2_RUN_DIR/STATUS.finish" "$CASE2_STATUS"
    printf '%s\n' "$rc" >"$CASE2_RUN_DIR/DONE.$CASE2_JOB"
    tmux wait-for -S "$CASE2_JOB"
    exit "$rc"
}
trap finish EXIT
bash "$CASE2_TIC_ROOT/scripts/train_tic.sh" "$CASE2_ARM" 1 "$@" --cfg job --resolve     >"$CASE2_RUN_DIR/resolved.yaml" 2>"$CASE2_RUN_DIR/config.stderr"
bash "$CASE2_TIC_ROOT/scripts/train_tic.sh" "$CASE2_ARM" 1 "$@"
