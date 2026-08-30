#!/usr/bin/env bash
# Resume-safe, outcome-blind MF3ZL pipeline supervisor.
# It never changes a sealed selection and never invokes a public split.
set -u

ROOT="/mnt/data_nas/deeprobotics/daiyang/vla"
PYTHON="$ROOT/.envs/etpr1/bin/python"
COLLECT="$ROOT/scripts/collect_mf3zl_exact_replay.py"
TRAIN="$ROOT/scripts/train_mf3zl_rcsp.py"
LOG_DIR="$ROOT/artifacts/training/mf3zl_rcsp_v1/logs"
LOG="$LOG_DIR/pipeline_supervisor.log"
mkdir -p "$LOG_DIR"
exec >>"$LOG" 2>&1

timestamp() { date -Is; }
note() { printf '%s %s\n' "$(timestamp)" "$*"; }

native_status() {
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("/mnt/data_nas/deeprobotics/daiyang/vla/artifacts/training/mf3zl_rcsp_v1/MF3ZL_NATIVE_SHADOW_PROGRESS.json")
if not p.is_file():
    print("MISSING")
else:
    try:
        print(json.loads(p.read_text()).get("status", "INVALID"))
    except Exception:
        print("INVALID")
PY
}

target_status() {
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("/mnt/data_nas/deeprobotics/daiyang/vla/artifacts/training/mf3zl_rcsp_v1/MF3ZL_TARGETED_SWITCH_PROGRESS.json")
if not p.is_file():
    print("MISSING")
else:
    try:
        print(json.loads(p.read_text()).get("status", "INVALID"))
    except Exception:
        print("INVALID")
PY
}

cd "$ROOT" || exit 1
note "MF3ZL supervisor started; sealed protocol is immutable"

native_attempts=0
while :; do
  state="$(native_status)"
  note "native state=$state"
  if [ "$state" = "COMPLETE" ]; then
    break
  fi
  if [ "$state" = "RUNNING" ]; then
    sleep 30
    continue
  fi
  native_attempts=$((native_attempts + 1))
  if [ "$native_attempts" -gt 3 ]; then
    note "native retry ceiling reached; stopping fail-closed"
    exit 2
  fi
  note "starting/resuming native shadow with explicit retry"
  "$PYTHON" "$COLLECT" run-native-shadow --gpus 0 1 --workers-per-gpu 8 --retry-failed
  rc=$?
  note "native invocation rc=$rc"
  if [ "$rc" -eq 0 ]; then
    break
  fi
  sleep 10
done

target_attempts=0
while :; do
  state="$(target_status)"
  note "target state=$state"
  if [ "$state" = "COMPLETE" ]; then
    break
  fi
  if [ "$state" = "RUNNING" ]; then
    sleep 30
    continue
  fi
  target_attempts=$((target_attempts + 1))
  if [ "$target_attempts" -gt 3 ]; then
    note "target retry ceiling reached; stopping fail-closed"
    exit 2
  fi
  note "starting/resuming targeted switches with explicit retry"
  "$PYTHON" "$COLLECT" run-targeted-switches --gpus 0 1 --workers-per-gpu 8 --retry-failed
  rc=$?
  note "target invocation rc=$rc"
  if [ "$rc" -eq 0 ]; then
    break
  fi
  sleep 10
done

note "assembling exact paired returns"
"$PYTHON" "$COLLECT" assemble
rc=$?
note "assemble rc=$rc"
if [ "$rc" -ne 0 ]; then
  note "assembly failed; no training was started"
  exit "$rc"
fi

note "auditing data support"
"$PYTHON" "$COLLECT" audit
rc=$?
note "data audit rc=$rc"
if [ "$rc" -ne 0 ]; then
  note "data support gate failed; RCSP training remains unauthorized"
  exit "$rc"
fi

note "starting sealed train-development fit and controls"
"$PYTHON" "$TRAIN" fit
rc=$?
note "train-development rc=$rc"
exit "$rc"
