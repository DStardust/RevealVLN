#!/usr/bin/env bash
set -euo pipefail

cd /mnt/daiyang/vla
source scripts/activate_remote_vla.sh >/dev/null
V2=artifacts/phase1/rxr_train_expansion/multibranch_v2
LOG="$V2/RXR_MULTIBRANCH_FEATURE_V2.log"
for _ in $(seq 1 2880); do
  if [[ -f "$V2/RXR_MULTIBRANCH_TX_V2_DONE" ]]; then
    printf 'MULTIBRANCH_FEATURE_START %s\n' "$(date -u +%FT%TZ)" >> "$LOG"
    python scripts/run_rxr_multibranch_feature_v2.py \
        --gpus 0,1,2,3,4,5,6,7 >> "$LOG" 2>&1
    printf 'MULTIBRANCH_FEATURE_END %s exit=0\n' "$(date -u +%FT%TZ)" >> "$LOG"
    printf 'PASS\n' > "$V2/RXR_MULTIBRANCH_FEATURE_V2_DONE"
    exit 0
  fi
  sleep 30
done
printf 'timeout waiting for multi-branch T_X\n' >> "$LOG"
exit 1
