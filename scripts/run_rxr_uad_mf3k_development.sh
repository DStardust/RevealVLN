#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data_nas/deeprobotics/daiyang/vla
PY=.envs/etpr1/bin/python
TRAIN=scripts/train_rxr_uad_policy_top2_mf3k.py
LOG=artifacts/training/mf3k_policy_top2_v1/DEVELOPMENT_RUN.log

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "MF3K_DEVELOPMENT_START $(date --iso-8601=seconds)"

running=0
index=0
failed=0
for hidden in 64 128; do
  for bound in 1.0 2.0; do
    for seed in 20260826 20260827 20260828; do
      gpu=$((index % 2))
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$TRAIN" \
        --hidden "$hidden" --bound "$bound" --seed "$seed" \
        --device cuda > "artifacts/training/mf3k_policy_top2_v1/train_h${hidden}_b${bound}_s${seed}.log" 2>&1 &
      running=$((running + 1))
      index=$((index + 1))
      if (( running == 2 )); then
        wait -n || failed=1
        running=$((running - 1))
      fi
    done
  done
done
while (( running > 0 )); do
  wait -n || failed=1
  running=$((running - 1))
done
if (( failed != 0 )); then
  echo "MF3K_TRAINING_FAILED $(date --iso-8601=seconds)"
  exit 1
fi
"$PY" scripts/select_rxr_uad_policy_top2_mf3k.py
echo "MF3K_DEVELOPMENT_COMPLETE $(date --iso-8601=seconds)"
