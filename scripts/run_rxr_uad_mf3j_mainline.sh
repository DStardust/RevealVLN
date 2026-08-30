#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data_nas/deeprobotics/daiyang/vla
SELECTION=artifacts/evaluation/mf3j_switch_utility_development_v1/MF3J_DEVELOPMENT_SELECTION.json
GATE=artifacts/evaluation/mf3j_switch_utility_shadow_gate_v1/MF3J_SHADOW_GATE.json
METRICS=artifacts/evaluation/mf3j_uad_rxr_val_seen_v1
LOG=artifacts/evaluation/mf3j_mainline.log
exec > >(tee -a "$LOG") 2>&1

while [[ ! -f "$SELECTION" ]]; do
  sleep 30
done
jq -e '.status == "DEVELOPMENT_PASS" and .rank14_payload_read == false' \
  "$SELECTION" >/dev/null

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
  .envs/etpr1/bin/python scripts/evaluate_rxr_uad_switch_shadow_gate_mf3j.py
jq -e '.status == "SHADOW_GATE_PASS" and .task_metric_run_authorized == true' \
  "$GATE" >/dev/null

PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py seal
PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py execute --preflight \
  --gpus 0,1 --workers-per-gpu 2
PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py verify --preflight
jq -e '.status == "PREFLIGHT_PASS"' \
  "$METRICS/MF3J_RXR_VAL_SEEN_PREFLIGHT.json" >/dev/null

PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py execute \
  --gpus 0,1 --workers-per-gpu 4
PYTHONNOUSERSITE=1 .envs/etpr1/bin/python \
  scripts/run_rxr_uad_paired_metrics_mf3.py verify
