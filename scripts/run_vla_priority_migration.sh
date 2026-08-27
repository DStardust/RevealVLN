#!/usr/bin/env bash
set -euo pipefail

root=/mnt/daiyang/vla
runtime=/var/tmp/daiyang_vla_priority
logs="$root/artifacts/migration/vla_priority_logs"
key="$root/.secret/migration_ed25519_8_130_54_48"
remote=root@8.130.54.48
destination=/mnt/data_nas/deeprobotics/daiyang/vla
mkdir -p "$runtime" "$logs"
exec >>"$runtime/orchestrator.log" 2>&1
echo "VLA_PRIORITY_START $(date --iso-8601=seconds)"

if [[ ! -f "$key" || -L "$key" || $(stat -c '%a' "$key") != 600 ]]; then
  echo "migration credential is missing or unsafe" >&2
  exit 2
fi

du -x -B1 -s \
  --exclude=.disk_reserve \
  --exclude='artifacts/migration/transfer_logs' \
  --exclude='artifacts/migration/vla_priority_logs' \
  --exclude='migration_ed25519_8_130_54_48*' \
  "$root" | cut -f1 >"$runtime/source_bytes"

ssh_command="ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=yes"
$ssh_command "$remote" "set -eu
  mkdir -p '$destination'
  test -d '$destination'
  test ! -L '$destination'"

common=(
  -aHSx --numeric-ids --protect-args
  --partial --partial-dir=.rsync-partial
  --timeout=900
  --info=progress2,stats2 --outbuf=L
  --human-readable
  --log-file-format='%t %o %i %n%L %l'
  -e "$ssh_command"
)

sync_partition() {
  local part=$1
  local log="$logs/priority_part_${part}.rsync.log"
  case "$part" in
    0)
      rsync "${common[@]}" --log-file="$log" \
        "$root/third_party" "$remote:$destination/"
      ;;
    1)
      rsync "${common[@]}" --log-file="$log" \
        --exclude='/migration/transfer_logs/' \
        --exclude='/migration/vla_priority_logs/' \
        --exclude='/artifacts/migration/transfer_logs/' \
        --exclude='/artifacts/migration/vla_priority_logs/' \
        "$root/artifacts" "$remote:$destination/"
      ;;
    2)
      rsync "${common[@]}" --log-file="$log" \
        "$root/.envs" "$root/.tools" "$root/.cache" \
        "$remote:$destination/"
      ;;
    3)
      rsync "${common[@]}" --log-file="$log" \
        --exclude='/.disk_reserve/' \
        --exclude='/.envs/' \
        --exclude='/.tools/' \
        --exclude='/.cache/' \
        --exclude='/third_party/' \
        --exclude='/artifacts/' \
        --exclude='/.secret/migration_ed25519_8_130_54_48' \
        --exclude='/.secret/migration_ed25519_8_130_54_48.pub' \
        "$root/" "$remote:$destination/"
      ;;
  esac
  touch "$runtime/transfer_part_${part}.pass"
}

verify_common=(
  -aHSxnc --numeric-ids --protect-args
  --dry-run --itemize-changes
  --timeout=900
  -e "$ssh_command"
)

verify_partition() {
  local part=$1
  local output="$runtime/verify_part_${part}.changes"
  : >"$output"
  case "$part" in
    0)
      rsync "${verify_common[@]}" "$root/third_party" \
        "$remote:$destination/" >"$output"
      ;;
    1)
      rsync "${verify_common[@]}" --exclude='/migration/transfer_logs/' \
        --exclude='/migration/vla_priority_logs/' \
        --exclude='/artifacts/migration/transfer_logs/' \
        --exclude='/artifacts/migration/vla_priority_logs/' \
        "$root/artifacts" "$remote:$destination/" >"$output"
      ;;
    2)
      rsync "${verify_common[@]}" \
        "$root/.envs" "$root/.tools" "$root/.cache" \
        "$remote:$destination/" >"$output"
      ;;
    3)
      rsync "${verify_common[@]}" \
        --exclude='/.disk_reserve/' \
        --exclude='/.envs/' \
        --exclude='/.tools/' \
        --exclude='/.cache/' \
        --exclude='/third_party/' \
        --exclude='/artifacts/' \
        --exclude='/.secret/migration_ed25519_8_130_54_48' \
        --exclude='/.secret/migration_ed25519_8_130_54_48.pub' \
        "$root/" "$remote:$destination/" >"$output"
      ;;
  esac
  if [[ -s "$output" ]]; then
    echo "VLA priority verification drift in partition $part" >&2
    return 1
  fi
  touch "$runtime/verify_part_${part}.pass"
}

run_transfer_wave() {
  local attempt=$1
  local failed=0
  local pids=()
  rm -f "$runtime"/transfer_part_{0,1,2,3}.pass
  for part in 0 1 2 3; do
    sync_partition "$part" >"$runtime/transfer_${attempt}_part_${part}.out" 2>&1 &
    pids+=("$!")
  done
  for index in 0 1 2 3; do
    if ! wait "${pids[$index]}"; then
      failed=1
    fi
  done
  return "$failed"
}

run_verify_wave() {
  local attempt=$1
  local failed=0
  local pids=()
  rm -f "$runtime"/verify_part_{0,1,2,3}.pass
  for part in 0 1 2 3; do
    verify_partition "$part" >"$runtime/verify_${attempt}_part_${part}.out" 2>&1 &
    pids+=("$!")
  done
  for index in 0 1 2 3; do
    if ! wait "${pids[$index]}"; then
      failed=1
    fi
  done
  return "$failed"
}

for attempt in 1 2 3 4 5; do
  echo "VLA_PRIORITY_ATTEMPT_$attempt $(date --iso-8601=seconds)"
  if ! run_transfer_wave "$attempt"; then
    continue
  fi
  echo "VLA_PRIORITY_TRANSFER_PASS attempt=$attempt $(date --iso-8601=seconds)"
  if run_verify_wave "$attempt"; then
    touch "$runtime/VLA_WORKSET_READY"
    echo "VLA_PRIORITY_VERIFY_PASS attempt=$attempt $(date --iso-8601=seconds)"
    exit 0
  fi
done

echo "VLA_PRIORITY_ABORT $(date --iso-8601=seconds)" >&2
exit 1
