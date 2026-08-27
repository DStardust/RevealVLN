#!/usr/bin/env bash

set -euo pipefail

workspace=/mnt/data_nas/deeprobotics/daiyang/vla
cd "$workspace"
source scripts/activate_remote_vla.sh >/dev/null

geometry_dir=artifacts/phase1/rxr_train_expansion/geometry
frontend_dir=artifacts/phase1/rxr_train_expansion/causal_frontend
log_dir="$frontend_dir/logs"
mkdir -p "$log_dir"
exec >>"$log_dir/systemd_frontend_filter.log" 2>&1

echo "FRONTEND_TAKEOVER_STARTED $(date --iso-8601=seconds)"
while [[ ! -f "$geometry_dir/SYSTEMD_GEOMETRY_CONTROLLER_DONE" ]]; do
  sleep 15
done

process_ids=()
shard_indices=()
failed=0
for shard_index in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$shard_index" \
    python scripts/run_rxr_expansion_frontend_shard.py \
      --shard-index "$shard_index" --shard-count 8 \
      >"$log_dir/shard_$(printf '%02d' "$shard_index").log" 2>&1 &
  process_ids+=("$!")
  shard_indices+=("$shard_index")
done
echo "FRONTEND_SHARDS_STARTED 8 $(date --iso-8601=seconds)"

for position in "${!process_ids[@]}"; do
  if ! wait "${process_ids[$position]}"; then
    echo "FRONTEND_SHARD_FAILED ${shard_indices[$position]}"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "FRONTEND_TAKEOVER_FAILED $(date --iso-8601=seconds)"
  exit 1
fi
printf 'PASS %s\n' "$(date --iso-8601=seconds)" \
  >"$frontend_dir/SYSTEMD_FRONTEND_SHARDS_DONE"
echo "FRONTEND_TAKEOVER_PASS $(date --iso-8601=seconds)"
