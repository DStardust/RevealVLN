#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || ! $1 =~ ^[0-5]$ ]]; then
  echo "usage: $0 PART_INDEX (0..5)" >&2
  exit 2
fi
part=$1
root=/mnt/daiyang/vla
key="$root/.secret/migration_ed25519_8_130_54_48"
remote=root@8.130.54.48
destination=/mnt/data_nas/deeprobotics/daiyang
work=/var/tmp/daiyang_server_migration_verify
mkdir -p "$work"
changes="$work/part_${part}.changes"
errors="$work/part_${part}.errors"
marker="$work/part_${part}.pass"
ssh_command="ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=yes"

python3 -c 'import sys; open(sys.argv[1], "wb").close(); open(sys.argv[2], "wb").close()' \
  "$changes" "$errors"
common=(
  -aHSxnc --numeric-ids --protect-args
  --dry-run --itemize-changes
  --timeout=900
  --out-format='%i %n%L'
  -e "$ssh_command"
)

compare_mnt_parent() {
  local source=$1
  rsync "${common[@]}" "$source" \
    "$remote:$destination/_source_mnt_top_level/" \
    >>"$changes" 2>>"$errors"
}

case "$part" in
  0)
    rsync "${common[@]}" \
      --exclude='/vla/.secret/migration_ed25519_8_130_54_48' \
      --exclude='/vla/.secret/migration_ed25519_8_130_54_48.pub' \
      --exclude='/vla/artifacts/migration/transfer_logs/' \
      /mnt/daiyang/ "$remote:$destination/" \
      >>"$changes" 2>>"$errors"
    ;;
  1) compare_mnt_parent /mnt/nuscenes_sparse ;;
  2) compare_mnt_parent /mnt/nuscenes ;;
  3) compare_mnt_parent /mnt/data_hpda ;;
  4)
    for source in \
      /mnt/.codex /mnt/FlashOCC /mnt/SparseDrive /mnt/claude-code \
      /mnt/e2e /mnt/install_sparsedrive_env.sh /mnt/mihomo \
      /mnt/pip-cache /mnt/pip_cache /mnt/sparsedrive_work /mnt/tmp
    do
      compare_mnt_parent "$source"
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
          "$remote:$destination/_source_root_state/" \
          >>"$changes" 2>>"$errors"
      fi
    done
    ;;
esac

if [[ -s "$changes" || -s "$errors" ]]; then
  echo "VERIFY_PART_${part}_DRIFT_OR_ERROR" >&2
  exit 1
fi
touch "$marker"
echo "VERIFY_PART_${part}_PASS"
