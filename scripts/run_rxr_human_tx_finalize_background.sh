#!/usr/bin/env bash
set -uo pipefail

cd /mnt/daiyang/vla
source scripts/activate_remote_vla.sh

tx_dir="artifacts/phase1/rxr_train_expansion/tx_gate"
tx_gate="${tx_dir}/RXR_EXPANSION_TX_GATE.json"
tx_done="${tx_dir}/RXR_EXPANSION_TX_SYSTEMD_DONE"
log="${tx_dir}/RXR_HUMAN_TX_FINALIZE_BACKGROUND.log"
done_marker="${tx_dir}/RXR_HUMAN_TX_FINALIZE_DONE"
exec >>"${log}" 2>&1

echo "HUMAN_TX_FINALIZE_WAIT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for _ in $(seq 1 720); do
  if [[ -f "${tx_gate}" ]]; then
    break
  fi
  if [[ -f "${tx_done}" ]] && ! grep -qx 'exit=0' "${tx_done}"; then
    echo "T_X background failed: $(cat "${tx_done}")"
    exit 1
  fi
  sleep 30
done
if [[ ! -f "${tx_gate}" ]]; then
  echo "Timed out waiting six hours for ${tx_gate}"
  exit 2
fi

python scripts/validate_rxr_human_pilot_300_labels.py
label_status=$?
if [[ "${label_status}" -ne 0 ]]; then
  echo "Human label validation failed with ${label_status}"
  exit "${label_status}"
fi
python scripts/finalize_rxr_human_tx_join.py
join_status=$?
echo "HUMAN_TX_FINALIZE_END $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${join_status}"
printf 'exit=%s\n' "${join_status}" >"${done_marker}.part"
mv "${done_marker}.part" "${done_marker}"
exit "${join_status}"
