#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.envs/etpr1/bin/python
$PY scripts/train_rxr_uad_top2_utility_mf3n.py --seal

pids=()
slot=0
for hidden in 32 64; do
  for seed in 20260826 20260827 20260828; do
    result="artifacts/training/mf3n_top2_utility_v1/hidden_${hidden}/seed_${seed}/RESULT.json"
    if [[ ! -f "$result" ]]; then
      gpu=$((slot % 2))
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_rxr_uad_top2_utility_mf3n.py \
        --hidden "$hidden" --seed "$seed" --device cuda:0 &
      pids+=("$!")
      slot=$((slot + 1))
    fi
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
$PY scripts/select_rxr_uad_top2_utility_mf3n.py
