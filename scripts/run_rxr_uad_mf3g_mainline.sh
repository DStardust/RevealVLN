#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data_nas/deeprobotics/daiyang/vla
PYTHON="$ROOT/.envs/etpr1/bin/python"
DATA="$ROOT/artifacts/phase1/mf3g_uad_online_expanded/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json"
LOG="$ROOT/artifacts/runtime/MF3G_MAINLINE_PIPELINE.log"

cd "$ROOT"
mkdir -p artifacts/runtime
printf '%s waiting_for_dataset\n' "$(date --iso-8601=seconds)" >> "$LOG"
while [[ ! -f "$DATA" ]]; do
  sleep 60
done
"$PYTHON" - "$DATA" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
if value.get("status") != "PASS":
    raise SystemExit("MF3G dataset completed without PASS")
PY

printf '%s training_started\n' "$(date --iso-8601=seconds)" >> "$LOG"
"$PYTHON" scripts/train_rxr_uad_residual_mf3g.py --seal >> "$LOG" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/train_rxr_uad_residual_mf3g.py \
  --seed 20260826 --device cuda >> "$LOG" 2>&1 &
PID_26=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/train_rxr_uad_residual_mf3g.py \
  --seed 20260827 --device cuda >> "$LOG" 2>&1 &
PID_27=$!
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/train_rxr_uad_residual_mf3g.py \
  --seed 20260828 --device cuda >> "$LOG" 2>&1 &
PID_28=$!
TRAINING_STATUS=0
wait "$PID_26" || TRAINING_STATUS=1
wait "$PID_27" || TRAINING_STATUS=1
wait "$PID_28" || TRAINING_STATUS=1
if [[ "$TRAINING_STATUS" -ne 0 ]]; then
  printf '%s training_failed\n' "$(date --iso-8601=seconds)" >> "$LOG"
  exit 1
fi

printf '%s shadow_gate_started\n' "$(date --iso-8601=seconds)" >> "$LOG"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  scripts/evaluate_rxr_uad_residual_shadow_gate_mf3g.py >> "$LOG" 2>&1

printf '%s paired_metrics_started\n' "$(date --iso-8601=seconds)" >> "$LOG"
"$PYTHON" scripts/run_rxr_uad_paired_metrics_mf3.py seal >> "$LOG" 2>&1
"$PYTHON" scripts/run_rxr_uad_paired_metrics_mf3.py execute \
  --preflight --gpus 0,1 --workers-per-gpu 2 >> "$LOG" 2>&1
"$PYTHON" scripts/run_rxr_uad_paired_metrics_mf3.py verify \
  --preflight >> "$LOG" 2>&1
"$PYTHON" scripts/run_rxr_uad_paired_metrics_mf3.py execute \
  --gpus 0,1 --workers-per-gpu 3 >> "$LOG" 2>&1
"$PYTHON" scripts/run_rxr_uad_paired_metrics_mf3.py verify >> "$LOG" 2>&1
printf '%s complete\n' "$(date --iso-8601=seconds)" >> "$LOG"
