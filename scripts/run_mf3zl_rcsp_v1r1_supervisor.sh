#!/usr/bin/env bash
# Resume-safe v1r1 expansion supervisor. It never invokes a public split.
set -u

ROOT="/mnt/data_nas/deeprobotics/daiyang/vla"
PYTHON="$ROOT/.envs/etpr1/bin/python"
COLLECT="$ROOT/scripts/collect_mf3zl_rcsp_v1r1.py"
LOG_DIR="$ROOT/artifacts/training/mf3zl_rcsp_v1r1/logs"
LOG="$LOG_DIR/expansion_supervisor.log"
mkdir -p "$LOG_DIR"
exec >>"$LOG" 2>&1

note() { printf '%s %s\n' "$(date -Is)" "$*"; }
status_of() {
  local path="$1"
  "$PYTHON" - "$path" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
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
note "MF3ZL v1r1 supervisor started; sealed protocol is immutable"

attempt=0
while :; do
  state="$(status_of "$ROOT/artifacts/training/mf3zl_rcsp_v1r1/MF3ZL_R2R_VARIANT_NATIVE_PROGRESS.json")"
  note "native state=$state"
  [ "$state" = "COMPLETE" ] && break
  attempt=$((attempt + 1))
  [ "$attempt" -le 3 ] || { note "native retry ceiling reached"; exit 2; }
  "$PYTHON" "$COLLECT" run-native-shadow --gpus 0 1 --workers-per-gpu 4 --retry-failed
  rc=$?; note "native rc=$rc"
  [ "$rc" -eq 0 ] || sleep 10
done

attempt=0
while :; do
  state="$(status_of "$ROOT/artifacts/training/mf3zl_rcsp_v1r1/MF3ZL_R2R_VARIANT_TARGET_PROGRESS.json")"
  note "target state=$state"
  [ "$state" = "COMPLETE" ] && break
  attempt=$((attempt + 1))
  [ "$attempt" -le 3 ] || { note "target retry ceiling reached"; exit 2; }
  "$PYTHON" "$COLLECT" run-targeted-switches --gpus 0 1 --workers-per-gpu 4 --retry-failed
  rc=$?; note "target rc=$rc"
  [ "$rc" -eq 0 ] || sleep 10
done

note "assembling v1r1 exact pairs"
"$PYTHON" "$COLLECT" assemble || { note "assembly failed"; exit 2; }
note "auditing combined support"
"$PYTHON" "$COLLECT" audit
rc=$?; note "audit rc=$rc"
exit "$rc"
