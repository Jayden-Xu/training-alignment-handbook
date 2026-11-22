#!/usr/bin/env bash
# c9_suite.sh: entry point of the case.
# Chain: c9_suite.sh -> train_tic_job.sh -> train_tic.sh -> verl trainer (verl.trainer.main_ppo);
# c9_results.py then verifies each arm and summarizes both.
# Arm: with CASE2_C9_ALIGNED_SUITE unset it runs the aligned arm, then the baseline arm; with
# CASE2_C9_ALIGNED_SUITE set to an earlier suite it checks that suite's archived aligned arm and
# runs only the baseline.
# Ports: the trainer's rendezvous port is chosen here automatically (a free port in 23000-23999)
# and handed to verl's worker group by tic_audit.py; nothing needs to be set by hand.
# Environment variables: see the header of train_tic.sh.
set -euo pipefail
: "${CASE2_TIC_ROOT:?}" "${CASE2_C9_SUITE:?}"
tic="$CASE2_TIC_ROOT"
suite="$CASE2_C9_SUITE"
run="$tic/results/$suite"
mkdir "$run"
exec >"$run/suite.log" 2>&1
source "${CASE2_ENV_SH:?set CASE2_ENV_SH to your env file}"
arm=aligned
if [[ -n "${CASE2_C9_ALIGNED_SUITE:-}" ]]; then arm=baseline; fi

finish() {
    local rc=$?
    trap - EXIT
    if ((rc)); then
        local arm_run="$tic/results/$suite-$arm"
        {
            printf 'phase=C9 failed\narm=%s\nexit_code=%s\n' "$arm" "$rc"
            if [[ -f "$arm_run/STATUS" ]]; then cat "$arm_run/STATUS"; fi
            if [[ -f "$arm_run/alignment-failure.json" ]]; then cat "$arm_run/alignment-failure.json"; fi
        } > "$run/STATUS"
        "$py" "$tic/scripts/c9_results.py" progress failed "$arm" || true
    fi
    printf '%s\n' "$rc" > "$run/DONE.$suite"
    tmux wait-for -S "$suite"
    exit "$rc"
}
trap finish EXIT

run_arm() {
    local job="$suite-$arm"
    local rendezvous_port
    rendezvous_port="$("$py" - <<'PORT'
import errno
import random
import socket

# Leave the kernel's ephemeral allocation range to Ray and other clients.
with open("/proc/sys/net/ipv4/ip_local_port_range") as config:
    ephemeral_low, ephemeral_high = map(int, config.read().split())
ports = [port for port in range(23000, 24000) if not ephemeral_low <= port <= ephemeral_high]
random.SystemRandom().shuffle(ports)
for port in ports:
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        try:
            listener.bind(("::", port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                continue
            raise
        print(port)
        break
else:
    raise SystemExit("No available C9 rendezvous port in 23000–23999")
PORT
)"
    printf '%s\n' "$rendezvous_port" > "$run/$arm-rendezvous-port"
    printf 'phase=C9 running\narm=%s\nmaster_port=%s\narm_status=%s\n' "$arm" "$rendezvous_port" "$tic/results/$job/STATUS" > "$run/STATUS"
    "$py" "$tic/scripts/c9_results.py" progress running "$arm"
    CASE2_JOB="$job" CASE2_ARM="$arm" CASE2_C9_ARM="$arm" CASE2_C9_MASTER_PORT="$rendezvous_port" \
        CASE2_PHASE="C9 full-model $arm" \
        bash "$tic/scripts/train_tic_job.sh"
    printf 'phase=C9 verifying\narm=%s\n' "$arm" > "$run/STATUS"
    "$py" "$tic/scripts/c9_results.py" verify "$arm"
}

if [[ "$arm" == aligned ]]; then
    run_arm
    arm=baseline
else
    "$py" "$tic/scripts/c9_results.py" archive
fi
run_arm
"$py" "$tic/scripts/c9_results.py" publish
printf 'phase=C9 complete\naligned_steps=10\nbaseline_steps=10\nweight_syncs_per_arm=11\nexit_code=0\n' > "$run/STATUS"
