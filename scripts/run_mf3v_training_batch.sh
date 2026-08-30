#!/usr/bin/env bash
set -u
cd /mnt/data_nas/deeprobotics/daiyang/vla
mkdir -p artifacts/training/mf3v_horizon_ranker_v1
running=0
slot=0
for fold in 0 1 2 final; do
  for seed in 20260826 20260827 20260828; do
    while [ "$running" -ge 6 ]; do
      wait -n || true
      running=$((running - 1))
    done
    gpu=$slot
    slot=$(((slot + 1) % 6))
    (
      CUDA_VISIBLE_DEVICES="$gpu" .envs/etpr1/bin/python \
        scripts/train_rxr_uad_horizon_mf3v.py \
        --fold "$fold" --seed "$seed" --device cuda \
        > "artifacts/training/mf3v_horizon_ranker_v1/train_${fold}_${seed}.stdout" \
        2> "artifacts/training/mf3v_horizon_ranker_v1/train_${fold}_${seed}.stderr"
      rc=$?
      printf 'fold=%s seed=%s gpu=%s rc=%s\n' "$fold" "$seed" "$gpu" "$rc" \
        >> artifacts/training/mf3v_horizon_ranker_v1/MF3V_TRAINING_LAUNCH.log
    ) &
    running=$((running + 1))
  done
done
wait
