#!/usr/bin/env bash
set -euo pipefail

root=/mnt/daiyang/vla
runtime=/var/tmp/daiyang_server_migration_supervisor
verify_runtime=/var/tmp/daiyang_server_migration_verify
key="$root/.secret/migration_ed25519_8_130_54_48"
remote=root@8.130.54.48
destination=/mnt/data_nas/deeprobotics/daiyang
ssh_options=(
  -i "$key"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o StrictHostKeyChecking=yes
)

if [[ ! -f "$key" || -L "$key" || $(stat -c '%a' "$key") != 600 ]]; then
  echo "migration credential is missing or unsafe" >&2
  exit 2
fi

active_state=$(systemctl show daiyang-server-migration \
  -p ActiveState --value 2>/dev/null || true)
sub_state=$(systemctl show daiyang-server-migration \
  -p SubState --value 2>/dev/null || true)
result=$(systemctl show daiyang-server-migration \
  -p Result --value 2>/dev/null || true)
running_parts=$(pgrep -fc 'run_server_migration_part\.sh [0-5]$' || true)
running_rsync=$(pgrep -fc 'rsync .*8\.130\.54\.48' || true)

echo "service_active_state=${active_state:-not-found}"
echo "service_sub_state=${sub_state:-not-found}"
echo "service_result=${result:-unknown}"
echo "running_part_jobs=$running_parts"
echo "running_remote_rsync_processes=$running_rsync"

if [[ -e "$runtime/ALL_CHECKSUM_VERIFICATIONS_PASS" ]]; then
  echo "checksum_closure=PASS"
else
  echo "checksum_closure=PENDING"
fi
if [[ -e /var/tmp/daiyang_vla_priority/VLA_WORKSET_READY ]]; then
  echo "vla_file_closure=PASS"
else
  echo "vla_file_closure=PENDING"
fi
if [[ -e /var/tmp/daiyang_vla_priority/REMOTE_VLA_READY ]]; then
  echo "vla_remote_ready=PASS"
else
  echo "vla_remote_ready=PENDING"
fi

for pass_name in initial final; do
  count=0
  for part in 0 1 2 3 4 5; do
    [[ -f "$root/artifacts/migration/transfer_logs/${pass_name}_part_${part}.pass" ]] \
      && count=$((count + 1))
  done
  echo "${pass_name}_transfer_markers=$count/6"
done

verified=0
for part in 0 1 2 3 4 5; do
  [[ -f "$verify_runtime/part_${part}.pass" ]] && verified=$((verified + 1))
done
echo "checksum_verify_markers=$verified/6"

ssh "${ssh_options[@]}" "$remote" \
  "set -eu
   test -d '$destination'
   test ! -L '$destination'
   test ! -e '$destination/_source_mnt_top_level/zrh'
   test ! -e '$destination/_source_mnt_top_level/wyh'
   test ! -e '$destination/vla/.secret/migration_ed25519_8_130_54_48'
   available=\$(df -B1 --output=avail '$destination' | tail -1)
   echo remote_available_bytes=\$available
   echo remote_boundary=PASS"

if [[ -f "$runtime/supervisor.log" ]]; then
  echo "supervisor_last_line=$(tail -n 1 "$runtime/supervisor.log")"
fi
