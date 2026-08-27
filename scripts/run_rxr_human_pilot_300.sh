#!/usr/bin/env bash

set -euo pipefail

workspace=/mnt/data_nas/deeprobotics/daiyang/vla
cd "$workspace"
source scripts/activate_remote_vla.sh >/dev/null

output=artifacts/phase1/rxr_train_expansion/human_pilot_300
mkdir -p "$output"
exec >>"$output/build.log" 2>&1

echo "BUILD_STARTED $(date --iso-8601=seconds)"
python scripts/build_rxr_human_pilot_300.py
python scripts/validate_rxr_human_pilot_300_package.py
printf 'COMPLETE %s\n' "$(date --iso-8601=seconds)" >"$output/SYSTEMD_HUMAN_PILOT_300_DONE"
echo "BUILD_COMPLETE $(date --iso-8601=seconds)"
