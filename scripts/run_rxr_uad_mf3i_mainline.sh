#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data_nas/deeprobotics/daiyang/vla

DATA=artifacts/phase1/mf3i_policy_token_uad/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json
TRAIN=artifacts/training/mf3i_policy_token_uad_v1
GATE=artifacts/evaluation/mf3i_contextual_uad_shadow_gate_v1/MF3I_UAD_SHADOW_GATE.json
METRICS=artifacts/evaluation/mf3i_uad_rxr_val_seen_v1
LOG=artifacts/evaluation/mf3i_mainline.log

mkdir -p artifacts/evaluation "$TRAIN"
exec > >(tee -a "$LOG") 2>&1

while [[ ! -f "$DATA" ]]; do
  sleep 30
done

jq -e '
  .status == "PASS" and
  .counts == {calibration:112,diagnostic:112,fit:519,shadow:56} and
  all(.records[];
    .observation_frontend == "frozen_etp_r1_policy_fusion_token" and
    .candidate_feature_dim == 1536)
' "$DATA" >/dev/null

PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/train_rxr_uad_contextual_mf3i.py --seal

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
  .envs/etpr1/bin/python scripts/train_rxr_uad_contextual_mf3i.py \
  --seed 20260826 --device cuda >"$TRAIN/seed_20260826.log" 2>&1 &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
  .envs/etpr1/bin/python scripts/train_rxr_uad_contextual_mf3i.py \
  --seed 20260827 --device cuda >"$TRAIN/seed_20260827.log" 2>&1 &
pid_b=$!
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
  .envs/etpr1/bin/python scripts/train_rxr_uad_contextual_mf3i.py \
  --seed 20260828 --device cuda >"$TRAIN/seed_20260828.log" 2>&1 &
pid_c=$!
wait "$pid_a"
wait "$pid_b"
wait "$pid_c"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
  .envs/etpr1/bin/python scripts/evaluate_rxr_uad_contextual_shadow_gate_mf3i.py
jq -e '.status == "SHADOW_GATE_PASS" and .task_metric_run_authorized == true' \
  "$GATE" >/dev/null

PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py seal
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py execute --preflight \
  --gpus 0,1 --workers-per-gpu 2
PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py verify --preflight
jq -e '.status == "PREFLIGHT_PASS"' \
  "$METRICS/MF3I_RXR_VAL_SEEN_PREFLIGHT.json" >/dev/null

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py execute \
  --gpus 0,1 --workers-per-gpu 4
PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py verify
