#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.envs/etpr1/bin/python
DATA=artifacts/phase1/mf3n_top2_utility_rank29/dataset_v1

scripts/run_rxr_uad_mf3n_development.sh
while [[ ! -f "$DATA/MF3B_ONLINE_DATA_MANIFEST.json" ]]; do
  sleep 60
done
$PY scripts/evaluate_rxr_uad_top2_utility_shadow_mf3n.py
