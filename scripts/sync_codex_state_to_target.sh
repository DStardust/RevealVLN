#!/usr/bin/env bash
set -euo pipefail

root=/mnt/daiyang/vla
runtime=/var/tmp/daiyang_codex_target_sync
key="$root/.secret/migration_ed25519_8_130_54_48"
remote=root@8.130.54.48
target=/mnt/data_nas/deeprobotics/daiyang/.codex
require_stable=${CODEX_REQUIRE_STABLE:-0}
mkdir -p "$runtime"

if [[ ! -f "$key" || -L "$key" || $(stat -c '%a' "$key") != 600 ]]; then
  echo "migration credential is missing or unsafe" >&2
  exit 2
fi
for source_dir in /mnt/daiyang/.codex /root/.codex /mnt/.codex; do
  if [[ ! -d "$source_dir" || -L "$source_dir" ]]; then
    echo "unsafe or missing Codex source: $source_dir" >&2
    exit 2
  fi
done

ssh_command="ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=yes"
$ssh_command "$remote" "set -eu
  mkdir -p '$target/migration_sources/mnt_daiyang_codex'
  mkdir -p '$target/migration_sources/root_codex'
  mkdir -p '$target/migration_sources/mnt_root_codex'
  test -d '$target'
  test ! -L '$target'"

common=(
  -aHSx --numeric-ids --protect-args
  --partial --partial-dir=.rsync-partial
  --timeout=900
  --exclude='/ipc/'
  --exclude='/thread-writer-locks/'
  --exclude='/.tmp/'
  --exclude='/tmp/'
  -e "$ssh_command"
)
verify=(
  "${common[@]}"
  -nc --dry-run --itemize-changes
)
conversation_filters=(
  --include='/sessions/'
  --include='/sessions/***'
  --include='/archived_sessions/'
  --include='/archived_sessions/***'
  --include='/history.jsonl'
  --exclude='*'
)

sync_all() {
  rsync "${common[@]}" /mnt/daiyang/.codex/ \
    "$remote:$target/migration_sources/mnt_daiyang_codex/"
  rsync "${common[@]}" /root/.codex/ \
    "$remote:$target/migration_sources/root_codex/"
  rsync "${common[@]}" /mnt/.codex/ \
    "$remote:$target/migration_sources/mnt_root_codex/"
  rsync "${common[@]}" \
    --exclude='/migration_sources/' \
    --exclude='/sessions/' \
    --exclude='/archived_sessions/' \
    --exclude='/history.jsonl' \
    /mnt/daiyang/.codex/ "$remote:$target/"
  rsync -a --protect-args -e "$ssh_command" \
    "$root/scripts/merge_codex_conversations.py" \
    "$remote:$target/migration_sources/merge_codex_conversations.py"
  $ssh_command "$remote" \
    "python3 '$target/migration_sources/merge_codex_conversations.py' '$target'"
}

verify_all() {
  : >"$runtime/changes"
  rsync "${verify[@]}" "${conversation_filters[@]}" \
    /mnt/daiyang/.codex/ \
    "$remote:$target/migration_sources/mnt_daiyang_codex/" >>"$runtime/changes"
  rsync "${verify[@]}" "${conversation_filters[@]}" /root/.codex/ \
    "$remote:$target/migration_sources/root_codex/" >>"$runtime/changes"
  rsync "${verify[@]}" "${conversation_filters[@]}" /mnt/.codex/ \
    "$remote:$target/migration_sources/mnt_root_codex/" >>"$runtime/changes"
  [[ ! -s "$runtime/changes" ]]
}

for attempt in 1 2 3 4 5; do
  sync_all
  if verify_all; then
    touch "$runtime/CODEX_TARGET_SYNC_PASS"
    echo "CODEX_TARGET_SYNC_PASS attempt=$attempt"
    exit 0
  fi
  if [[ $require_stable == 0 ]]; then
    touch "$runtime/CODEX_TARGET_SYNC_PASS"
    touch "$runtime/CODEX_TARGET_LIVE_DRIFT"
    echo "CODEX_TARGET_POINT_IN_TIME_PASS_WITH_LIVE_DRIFT attempt=$attempt"
    exit 0
  fi
done

echo "Codex sources did not reach a stable point-in-time snapshot" >&2
exit 1
