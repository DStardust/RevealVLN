#!/usr/bin/env bash
set -euo pipefail

cd /mnt/daiyang/vla
source scripts/activate_remote_vla.sh >/dev/null
LOG=artifacts/phase1/rxr_train_expansion/multibranch_v2/RXR_MULTIBRANCH_LANGUAGE_V2.log
DONE=artifacts/phase1/rxr_train_expansion/multibranch_v2/RXR_MULTIBRANCH_LANGUAGE_V2_DONE
printf 'MULTIBRANCH_LANGUAGE_START %s\n' "$(date -u +%FT%TZ)" >> "$LOG"
if CR5_CAUSAL_MEDIA_GPU=7 \
    python scripts/build_rxr_multibranch_causal_prefix_media_v2.py \
    >> "$LOG" 2>&1 && \
    python scripts/run_rxr_multibranch_causal_prefix_language_v2.py \
    --execute --workers 12 >> "$LOG" 2>&1; then
  printf 'MULTIBRANCH_LANGUAGE_END %s exit=0\n' "$(date -u +%FT%TZ)" >> "$LOG"
  printf 'PASS\n' > "$DONE"
else
  status=$?
  printf 'MULTIBRANCH_LANGUAGE_END %s exit=%s\n' \
    "$(date -u +%FT%TZ)" "$status" >> "$LOG"
  exit "$status"
fi
