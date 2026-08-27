#!/usr/bin/env bash
set -euo pipefail

root=/mnt/daiyang/vla
key="$root/.secret/migration_ed25519_8_130_54_48"
remote=root@8.130.54.48
destination=/mnt/data_nas/deeprobotics/daiyang
inventory="$root/artifacts/migration/SERVER_MIGRATION_SOURCE_INVENTORY.json"
ssh_command="ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes"

if [[ ! -f "$key" || -L "$key" ]]; then
  echo "migration key is missing or unsafe" >&2
  exit 2
fi
if [[ $(stat -c '%a' "$key") != 600 ]]; then
  echo "migration key mode is not 600" >&2
  exit 2
fi
if ! command -v rsync >/dev/null; then
  echo "local rsync is unavailable" >&2
  exit 2
fi

if [[ ! -f "$inventory" || -L "$inventory" ]]; then
  echo "sealed source inventory is missing or unsafe" >&2
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
  --info=progress2,stats2
  --human-readable
  -e "$ssh_command"
)

# /mnt/daiyang maps directly to the new personal root.  Exclude only the
# ephemeral credential created for this transfer; all user reserve files are
# copied as ordinary sparse-aware data and are never modified on the source.
rsync "${common[@]}" \
  --exclude='/vla/.secret/migration_ed25519_8_130_54_48' \
  --exclude='/vla/.secret/migration_ed25519_8_130_54_48.pub' \
  /mnt/daiyang/ "$remote:$destination/"

for source in \
  /mnt/.codex \
  /mnt/FlashOCC \
  /mnt/SparseDrive \
  /mnt/claude-code \
  /mnt/data_hpda \
  /mnt/e2e \
  /mnt/install_sparsedrive_env.sh \
  /mnt/mihomo \
  /mnt/nuscenes \
  /mnt/nuscenes_sparse \
  /mnt/pip-cache \
  /mnt/pip_cache \
  /mnt/sparsedrive_work \
  /mnt/tmp
do
  rsync "${common[@]}" "$source" \
    "$remote:$destination/_source_mnt_top_level/"
done

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

echo "INITIAL_TRANSFER_PASS_FINAL_DELTA_AND_VERIFICATION_REQUIRED"
