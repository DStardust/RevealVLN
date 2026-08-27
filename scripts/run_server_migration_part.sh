#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || ! $1 =~ ^[0-5]$ ]]; then
  echo "usage: $0 PART_INDEX (0..5)" >&2
  exit 2
fi
part=$1
pass_name=${MIGRATION_PASS:-initial}
if [[ $pass_name != initial && $pass_name != final ]]; then
  echo "MIGRATION_PASS must be initial or final" >&2
  exit 2
fi
root=/mnt/daiyang/vla
key="$root/.secret/migration_ed25519_8_130_54_48"
remote=root@8.130.54.48
destination=/mnt/data_nas/deeprobotics/daiyang
inventory="$root/artifacts/migration/SERVER_MIGRATION_SOURCE_INVENTORY.json"
logs="$root/artifacts/migration/transfer_logs"
mkdir -p "$logs"
log="$logs/${pass_name}_part_${part}.rsync.log"
marker="$logs/${pass_name}_part_${part}.pass"
ssh_command="ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=yes"

for path in "$key" "$inventory"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "unsafe or missing migration input: $path" >&2
    exit 2
  fi
done
if [[ $(stat -c '%a' "$key") != 600 ]]; then
  echo "migration key mode is not 600" >&2
  exit 2
fi

required=$("$root/.envs/etpr1/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["minimum_destination_free_bytes"])' \
  "$inventory")
$ssh_command "$remote" "set -eu
  test \"\$(id -u)\" = 0
  test -d '$destination'
  test ! -L '$destination'
  command -v rsync >/dev/null
  available=\$(df -B1 --output=avail '$destination' | tail -1)
  test \"\$available\" -ge '$required'
  mkdir -p '$destination/_source_mnt_top_level'
  mkdir -p '$destination/_source_root_state/root'"

common=(
  -aHSx --numeric-ids --protect-args
  --partial --partial-dir=.rsync-partial
  --timeout=900
  --info=progress2,stats2 --outbuf=L
  --human-readable
  --log-file="$log"
  --log-file-format='%t %o %i %n%L %l'
  -e "$ssh_command"
)

sync_mnt_parent() {
  local source=$1
  rsync "${common[@]}" "$source" \
    "$remote:$destination/_source_mnt_top_level/"
}

case "$part" in
  0)
    rsync "${common[@]}" \
      --exclude='/vla/.secret/migration_ed25519_8_130_54_48' \
      --exclude='/vla/.secret/migration_ed25519_8_130_54_48.pub' \
      --exclude='/vla/artifacts/migration/transfer_logs/' \
      /mnt/daiyang/ "$remote:$destination/"
    ;;
  1)
    sync_mnt_parent /mnt/nuscenes_sparse
    ;;
  2)
    sync_mnt_parent /mnt/nuscenes
    ;;
  3)
    sync_mnt_parent /mnt/data_hpda
    ;;
  4)
    for source in \
      /mnt/.codex \
      /mnt/FlashOCC \
      /mnt/SparseDrive \
      /mnt/claude-code \
      /mnt/e2e \
      /mnt/install_sparsedrive_env.sh \
      /mnt/mihomo \
      /mnt/pip-cache \
      /mnt/pip_cache \
      /mnt/sparsedrive_work \
      /mnt/tmp
    do
      sync_mnt_parent "$source"
    done
    ;;
  5)
    root_sources=(
      /root/.codex
      /root/.claude
      /root/.claude.json
      /root/.cache/claude
      /root/.cache/claude-cli-nodejs
      /root/.local/share/claude
      /root/.local/state/claude
      /root/.local/bin/claude
      /root/.vscode-server/extensions/anthropic.claude-code-2.1.243-linux-x64
      /root/.vscode-server/extensions/anthropic.claude-code-2.1.245-linux-x64
      /root/.vscode-server/data/CachedExtensionVSIXs/anthropic.claude-code-2.1.243-linux-x64
      /root/.vscode-server/data/CachedExtensionVSIXs/anthropic.claude-code-2.1.245-linux-x64
      /root/.antigravity-ide-server/extensions/anthropic.claude-code-2.1.219-linux-x64
      /root/.antigravity-ide-server/extensions/anthropic.claude-code-2.1.220-linux-x64
    )
    for source in "${root_sources[@]}"; do
      if [[ -e "$source" || -L "$source" ]]; then
        rsync "${common[@]}" -R "$source" \
          "$remote:$destination/_source_root_state/"
      fi
    done
    ;;
esac

touch "$marker"
echo "${pass_name^^}_PART_${part}_PASS"
