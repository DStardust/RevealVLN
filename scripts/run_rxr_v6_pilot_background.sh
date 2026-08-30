#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data_nas/deeprobotics/daiyang/vla
exec .envs/etpr1/bin/python scripts/run_rxr_v6_counterfactual_pipeline.py all \
  --cohort pilot_v6_0 --episodes 60 --gpus 2,3,4,5,6,7
