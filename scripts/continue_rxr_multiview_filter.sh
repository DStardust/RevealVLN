#!/usr/bin/env bash

set -uo pipefail

workspace=/mnt/data_nas/deeprobotics/daiyang/vla
cd "$workspace"
source scripts/activate_remote_vla.sh >/dev/null

log_dir=artifacts/phase1/rxr_train_expansion/branch_factory/logs
mkdir -p "$log_dir"
exec >>"$log_dir/systemd_takeover.log" 2>&1

echo "TAKEOVER_STARTED $(date --iso-8601=seconds)"
while pgrep -f '[r]un_rxr_multiview_branch_factory.py.*--execute' >/dev/null; do
  sleep 15
done

run_pass() {
  local pass_name=$1
  local shard_index
  local process_id
  local failed=0
  local -a process_ids=()
  local -a shard_indices=()

  echo "PASS_STARTED $pass_name $(date --iso-8601=seconds)"
  for shard_index in $(seq 0 27); do
    python scripts/run_rxr_multiview_branch_factory.py \
      --shard-index "$shard_index" --execute \
      >"$log_dir/${pass_name}_shard_$(printf '%02d' "$shard_index").log" 2>&1 &
    process_ids+=("$!")
    shard_indices+=("$shard_index")
  done
  for process_id in "${!process_ids[@]}"; do
    if ! wait "${process_ids[$process_id]}"; then
      echo "PASS_SHARD_FAILED $pass_name ${shard_indices[$process_id]}"
      failed=1
    fi
  done
  echo "PASS_FINISHED $pass_name failed=$failed $(date --iso-8601=seconds)"
  return "$failed"
}

for pass_number in 1 2; do
  python scripts/repair_rxr_multiview_branch_factory.py \
    >"$log_dir/takeover_repair_${pass_number}.log" 2>&1 || true
  run_pass "takeover_pass_${pass_number}" || true
done

python scripts/repair_rxr_multiview_branch_factory.py \
  >"$log_dir/takeover_repair_final.log" 2>&1 || true
run_pass "takeover_manifest_refresh" || true

if python - <<'PY'
import json
from pathlib import Path

run_dir = Path("artifacts/phase1/rxr_train_expansion/branch_factory/runs")
for index in range(28):
    path = run_dir / f"shard_{index:02d}.json"
    if not path.is_file() or json.loads(path.read_text()).get("status") != "PASS":
        raise SystemExit(1)
PY
then
  python scripts/aggregate_rxr_multiview_branch_factory.py \
    >"$log_dir/takeover_aggregate.log" 2>&1
  printf 'PASS %s\n' "$(date --iso-8601=seconds)" \
    >artifacts/phase1/rxr_train_expansion/branch_factory/SYSTEMD_AUTOFILTER_DONE
  echo "TAKEOVER_PASS $(date --iso-8601=seconds)"
else
  printf 'PENDING_INVALID_RESULTS %s\n' "$(date --iso-8601=seconds)" \
    >artifacts/phase1/rxr_train_expansion/branch_factory/SYSTEMD_AUTOFILTER_PENDING
  echo "TAKEOVER_PENDING_INVALID_RESULTS $(date --iso-8601=seconds)"
fi
