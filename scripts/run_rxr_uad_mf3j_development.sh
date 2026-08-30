#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data_nas/deeprobotics/daiyang/vla
OUT=artifacts/training/mf3j_switch_utility_v1
mkdir -p "$OUT"

run() {
  local gpu=$1 hidden=$2 seed=$3
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
    .envs/etpr1/bin/python scripts/train_rxr_uad_switch_utility_mf3j.py \
    --hidden "$hidden" --seed "$seed" --device cuda \
    >"$OUT/hidden_${hidden}_seed_${seed}.log" 2>&1
}

run 0 64 20260826 & a=$!
run 0 64 20260827 & b=$!
run 0 64 20260828 & c=$!
run 1 128 20260826 & d=$!
run 1 128 20260827 & e=$!
run 1 128 20260828 & f=$!
wait "$a" "$b" "$c" "$d" "$e" "$f"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
  .envs/etpr1/bin/python scripts/select_rxr_uad_switch_rule_mf3j.py
