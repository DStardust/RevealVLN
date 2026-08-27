#!/usr/bin/env bash
set -euo pipefail

cd /mnt/daiyang/vla
source scripts/activate_remote_vla.sh >/dev/null
V2=artifacts/phase1/rxr_train_expansion/multibranch_v2
LOG="$V2/RXR_MULTIBRANCH_TRAINING_INDEX_V2.log"
for _ in $(seq 1 1440); do
  if [[ -f "$V2/RXR_MULTIBRANCH_LANGUAGE_V2_DONE" ]]; then
    python scripts/build_rxr_multibranch_training_index_v2.py >> "$LOG" 2>&1
    printf 'PASS\n' > "$V2/RXR_MULTIBRANCH_TRAINING_INDEX_V2_DONE"
    exit 0
  fi
  starts=$(grep -c 'MULTIBRANCH_LANGUAGE_START' \
      "$V2/RXR_MULTIBRANCH_LANGUAGE_V2.log" 2>/dev/null || true)
  ends=$(grep -c 'MULTIBRANCH_LANGUAGE_END' \
      "$V2/RXR_MULTIBRANCH_LANGUAGE_V2.log" 2>/dev/null || true)
  last_end=$(grep 'MULTIBRANCH_LANGUAGE_END' \
      "$V2/RXR_MULTIBRANCH_LANGUAGE_V2.log" 2>/dev/null | tail -n 1 || true)
  if [[ "$starts" -gt 0 && "$starts" -eq "$ends" && \
      "$last_end" != *"exit=0"* ]]; then
    printf 'language gate failed; refusing index assembly\n' >> "$LOG"
    exit 1
  fi
  sleep 30
done
printf 'timeout waiting for full-set language gate\n' >> "$LOG"
exit 1
