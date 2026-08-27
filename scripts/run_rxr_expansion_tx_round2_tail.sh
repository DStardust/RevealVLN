#!/usr/bin/env bash
set -uo pipefail

cd /mnt/daiyang/vla
source scripts/activate_remote_vla.sh

log="artifacts/phase1/rxr_train_expansion/tx_gate/RXR_EXPANSION_TX_ROUND2_TAIL.log"
exec >>"${log}" 2>&1
echo "TX_ROUND2_TAIL_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python scripts/run_rxr_expansion_tx_round2_tail.py
status=$?
echo "TX_ROUND2_TAIL_END $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${status}"
exit "${status}"
