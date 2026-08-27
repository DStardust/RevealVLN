#!/usr/bin/env bash
set -euo pipefail

cd /mnt/daiyang/vla
source scripts/activate_remote_vla.sh

python scripts/validate_rxr_human_pilot_300_labels.py

tx_gate="artifacts/phase1/rxr_train_expansion/tx_gate/RXR_EXPANSION_TX_GATE.json"
if [[ ! -f "${tx_gate}" ]]; then
  echo "Human labels are valid, but the background T_X gate is still running."
  echo "Monitor: systemctl status revealnav-rxr-expansion-tx-v1.service"
  echo "Log: tail -f artifacts/phase1/rxr_train_expansion/tx_gate/RXR_EXPANSION_TX_BACKGROUND.log"
  exit 3
fi

python scripts/finalize_rxr_human_tx_join.py
