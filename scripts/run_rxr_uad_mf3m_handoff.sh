#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data_nas/deeprobotics/daiyang/vla
PY=.envs/etpr1/bin/python
SELECTION=artifacts/evaluation/mf3m_robust_top2_development_v1/MF3M_DEVELOPMENT_SELECTION.json
DATA_DIR=artifacts/phase1/mf3m_robust_top2_rank23/dataset_v1
MANIFEST="$DATA_DIR/MF3B_ONLINE_DATA_MANIFEST.json"
OLD_MANIFEST=artifacts/phase1/mf3k_policy_top2_rank17/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json
LOG=artifacts/evaluation/mf3m_robust_top2_handoff.log

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
jq -e '.status == "DEVELOPMENT_PASS" and .ranks18_23_payload_read == false' \
  "$SELECTION" >/dev/null

echo "MF3M_FRESH_COLLECTION_START $(date --iso-8601=seconds)"
"$PY" scripts/run_rxr_uad_online_dataset_mf3.py \
  --output-dir "$DATA_DIR" \
  --gpus 0,1 --workers-per-gpu 4 --episode-rank-split 9,2,6,6 \
  --reuse-manifest "$OLD_MANIFEST" --policy-fusion-features \
  --retry-failures
jq -e '.status == "PASS"' "$MANIFEST" >/dev/null

echo "MF3M_FRESH_SHADOW_START $(date --iso-8601=seconds)"
set +e
CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/evaluate_rxr_uad_robust_top2_shadow_mf3m.py
gate_code=$?
set -e
if (( gate_code != 0 )); then
  echo "MF3M_HANDOFF_STOP_SHADOW_FAIL $(date --iso-8601=seconds)"
  exit "$gate_code"
fi

echo "MF3M_VAL_SEEN_PREFLIGHT_START $(date --iso-8601=seconds)"
"$PY" scripts/run_rxr_uad_paired_metrics_mf3.py seal
"$PY" scripts/run_rxr_uad_paired_metrics_mf3.py execute \
  --preflight --gpus 0,1 --workers-per-gpu 1
"$PY" scripts/run_rxr_uad_paired_metrics_mf3.py verify --preflight

echo "MF3M_VAL_SEEN_FULL_START $(date --iso-8601=seconds)"
"$PY" scripts/run_rxr_uad_paired_metrics_mf3.py execute \
  --gpus 0,1 --workers-per-gpu 4
set +e
"$PY" scripts/run_rxr_uad_paired_metrics_mf3.py verify
metric_code=$?
set -e
echo "MF3M_HANDOFF_COMPLETE metric_code=$metric_code $(date --iso-8601=seconds)"
exit "$metric_code"
