#!/usr/bin/env bash
set -uo pipefail

cd /mnt/daiyang/vla
source scripts/activate_remote_vla.sh

tx_dir="artifacts/phase1/rxr_train_expansion/tx_gate"
log_path="${tx_dir}/RXR_EXPANSION_TX_BACKGROUND.log"
marker_path="${tx_dir}/RXR_EXPANSION_TX_SYSTEMD_DONE"
mkdir -p "${tx_dir}"
rm -f "${marker_path}"

exec >>"${log_path}" 2>&1
echo "TX_BACKGROUND_START $(date -u +%Y-%m-%dT%H:%M:%SZ) gpus=0,1"
python scripts/run_rxr_expansion_tx_gate.py --gpus 0,1
status=$?
echo "TX_BACKGROUND_END $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${status}"
printf 'exit=%s\n' "${status}" >"${marker_path}.part"
mv "${marker_path}.part" "${marker_path}"
exit "${status}"
