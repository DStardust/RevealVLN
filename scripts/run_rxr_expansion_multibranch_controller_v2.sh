#!/usr/bin/env bash
set -uo pipefail

cd /mnt/daiyang/vla
source scripts/activate_remote_vla.sh

dir="artifacts/phase1/rxr_train_expansion/multibranch_v2"
log="${dir}/RXR_MULTIBRANCH_CONTROLLER_V2.log"
marker="${dir}/RXR_MULTIBRANCH_CONTROLLER_V2_DONE"
exec >>"${log}" 2>&1
echo "MULTIBRANCH_CONTROLLER_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
CR5_CONTROLLER_GPU=0 python scripts/run_rxr_expansion_multibranch_controller_v2.py
status=$?
echo "MULTIBRANCH_CONTROLLER_END $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${status}"
printf 'exit=%s\n' "${status}" >"${marker}.part"
mv "${marker}.part" "${marker}"
exit "${status}"
