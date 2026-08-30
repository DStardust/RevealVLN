#!/usr/bin/env bash
set -u

cd /mnt/data_nas/deeprobotics/daiyang/vla
progress=artifacts/phase1/mf3b_uad_online/dataset_v1/MF3B_ONLINE_DATA_PROGRESS.json
manifest=artifacts/phase1/mf3b_uad_online/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json
log=artifacts/training/mf3b_uad_online_v1/pipeline.log
mkdir -p artifacts/training/mf3b_uad_online_v1

while true; do
    status=$(.envs/etpr1/bin/python -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' \
        "$progress" 2>/dev/null || true)
    if [ "$status" = "COMPLETE" ] || [ "$status" = "FAIL" ]; then
        break
    fi
    sleep 30
done

manifest_status=$(.envs/etpr1/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' \
    "$manifest" 2>/dev/null || true)
if [ "$manifest_status" != "PASS" ]; then
    echo "ONLINE_DATA_FAIL" >> "$log"
    exit 2
fi

.envs/etpr1/bin/python scripts/train_rxr_uad_mf3.py --seal >> "$log" 2>&1 || exit $?
seeds=(20260826 20260827 20260828)
gpus=(0 1 2)
pids=()
for index in 0 1 2; do
    seed=${seeds[$index]}
    gpu=${gpus[$index]}
    CUDA_VISIBLE_DEVICES=$gpu .envs/etpr1/bin/python \
        scripts/train_rxr_uad_mf3.py --seed "$seed" --device cuda \
        > "artifacts/training/mf3b_uad_online_v1/seed_${seed}.stdout" \
        2> "artifacts/training/mf3b_uad_online_v1/seed_${seed}.stderr" &
    pids+=("$!")
done
training_status=0
for pid in "${pids[@]}"; do
    wait "$pid" || training_status=1
done
if [ "$training_status" -ne 0 ]; then
    echo "TRAINING_FAIL" >> "$log"
    exit 3
fi

if CUDA_VISIBLE_DEVICES=0 .envs/etpr1/bin/python \
    scripts/evaluate_rxr_uad_shadow_gate_mf3.py >> "$log" 2>&1; then
    echo "SHADOW_GATE_PASS" >> "$log"
else
    echo "SHADOW_GATE_FAIL" >> "$log"
    exit 4
fi

.envs/etpr1/bin/python scripts/run_rxr_uad_paired_metrics_mf3.py seal \
    >> "$log" 2>&1 || exit $?
.envs/etpr1/bin/python scripts/run_rxr_uad_paired_metrics_mf3.py execute \
    --preflight --gpus 0,1 >> "$log" 2>&1 || exit $?
.envs/etpr1/bin/python scripts/run_rxr_uad_paired_metrics_mf3.py verify \
    --preflight >> "$log" 2>&1 || exit $?
.envs/etpr1/bin/python scripts/run_rxr_uad_paired_metrics_mf3.py execute \
    --gpus 0,1,2,3,4,5,6,7 >> "$log" 2>&1 || exit $?
if .envs/etpr1/bin/python scripts/run_rxr_uad_paired_metrics_mf3.py verify \
    >> "$log" 2>&1; then
    echo "RXR_VAL_SEEN_TASK_METRIC_GATE_PASS" >> "$log"
else
    echo "RXR_VAL_SEEN_TASK_METRIC_GATE_FAIL" >> "$log"
    exit 5
fi
