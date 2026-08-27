#!/usr/bin/env bash
set -euo pipefail

root=/mnt/daiyang/vla
runtime=/var/tmp/daiyang_server_migration_supervisor
mkdir -p "$runtime"
exec >>"$runtime/supervisor.log" 2>&1
echo "SUPERVISOR_START $(date --iso-8601=seconds)"

priority_ready=/var/tmp/daiyang_vla_priority/REMOTE_VLA_READY
if [[ ! -f "$priority_ready" ]]; then
  echo "SUPERVISOR_WAIT_VLA_PRIORITY $(date --iso-8601=seconds)"
fi
while [[ ! -f "$priority_ready" ]]; do
  sleep 30
done
echo "SUPERVISOR_VLA_PRIORITY_READY $(date --iso-8601=seconds)"

# The six initial jobs were launched interactively.  Wait for them to release
# the source/network, then run an idempotent closure wave that creates markers.
while pgrep -f 'bash scripts/run_server_migration_part.sh [0-5]$' \
    >/dev/null; do
  sleep 60
done

run_transfer_wave() {
  local pass_name=$1
  local attempt=$2
  local failed=0
  local pids=()
  local parts=()
  for part in 0 1 2 3 4 5; do
    MIGRATION_PASS="$pass_name" \
      "$root/scripts/run_server_migration_part.sh" "$part" \
      >"$runtime/${pass_name}_${attempt}_part_${part}.out" 2>&1 &
    pids+=("$!")
    parts+=("$part")
  done
  for index in 0 1 2 3 4 5; do
    if ! wait "${pids[$index]}"; then
      echo "TRANSFER_FAIL pass=$pass_name attempt=$attempt part=${parts[$index]}"
      failed=1
    fi
  done
  return "$failed"
}

for attempt in 1 2 3 4 5; do
  if run_transfer_wave initial "$attempt"; then
    break
  fi
  if [[ $attempt == 5 ]]; then
    echo "SUPERVISOR_ABORT_INITIAL"
    exit 1
  fi
done

for closure_attempt in 1 2 3 4 5; do
  if ! run_transfer_wave final "$closure_attempt"; then
    continue
  fi
  if ! CODEX_REQUIRE_STABLE=1 "$root/scripts/sync_codex_state_to_target.sh" \
      >"$runtime/codex_final_${closure_attempt}.out" 2>&1; then
    echo "CODEX_FINAL_SYNC_FAIL closure_attempt=$closure_attempt"
    continue
  fi
  verify_failed=0
  verify_pids=()
  for part in 0 1 2 3 4 5; do
    "$root/scripts/verify_server_migration_part.sh" "$part" \
      >"$runtime/verify_${closure_attempt}_part_${part}.out" 2>&1 &
    verify_pids+=("$!")
  done
  for index in 0 1 2 3 4 5; do
    if ! wait "${verify_pids[$index]}"; then
      echo "VERIFY_FAIL closure_attempt=$closure_attempt part=$index"
      verify_failed=1
    fi
  done
  if [[ $verify_failed == 0 ]]; then
    touch "$runtime/ALL_CHECKSUM_VERIFICATIONS_PASS"
    echo "SUPERVISOR_TRANSFER_AND_VERIFY_PASS $(date --iso-8601=seconds)"
    exit 0
  fi
done

echo "SUPERVISOR_ABORT_VERIFICATION"
exit 1
