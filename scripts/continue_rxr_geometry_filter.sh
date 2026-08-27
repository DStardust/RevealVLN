#!/usr/bin/env bash

set -euo pipefail

workspace=/mnt/data_nas/deeprobotics/daiyang/vla
cd "$workspace"
source scripts/activate_remote_vla.sh >/dev/null

branch_dir=artifacts/phase1/rxr_train_expansion/branch_factory
geometry_dir=artifacts/phase1/rxr_train_expansion/geometry
mkdir -p "$geometry_dir"
exec >>"$geometry_dir/systemd_geometry_filter.log" 2>&1

echo "GEOMETRY_TAKEOVER_STARTED $(date --iso-8601=seconds)"
while [[ ! -f "$branch_dir/SYSTEMD_AUTOFILTER_DONE" ]]; do
  if [[ -f "$branch_dir/SYSTEMD_AUTOFILTER_PENDING" ]]; then
    echo "GEOMETRY_TAKEOVER_BLOCKED_BY_INVALID_MLLM $(date --iso-8601=seconds)"
    exit 1
  fi
  sleep 15
done

python scripts/run_rxr_expansion_directed_geometry.py
CR5_CONTROLLER_GPU=0 python scripts/run_rxr_expansion_controller_gate.py
printf 'PASS %s\n' "$(date --iso-8601=seconds)" \
  >"$geometry_dir/SYSTEMD_GEOMETRY_CONTROLLER_DONE"
echo "GEOMETRY_TAKEOVER_PASS $(date --iso-8601=seconds)"
