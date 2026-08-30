#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data_nas/deeprobotics/daiyang/vla"
PYTHON="$ROOT/.envs/etpr1/bin/python"
RUNNER="$ROOT/scripts/run_rxr_primary_screen_v5_22.py"

cd "$ROOT"
"$PYTHON" "$RUNNER" execute --gpus 0,1
"$PYTHON" "$RUNNER" verify
