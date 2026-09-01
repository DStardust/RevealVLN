#!/usr/bin/env bash
# Detached continuation for the sealed MF3ZP v2 pipeline.
# It never changes protocol files and stops at the first failed gate.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.envs/etpr1/bin/python"
OUT="$ROOT/artifacts/training/mf3zp_qwen_uad_reference_v2"
LOG="$OUT/logs"
COLLECT_PID="${1:-}"
mkdir -p "$LOG"
exec > >(tee -a "$LOG/pipeline.stdout.log") 2> >(tee -a "$LOG/pipeline.stderr.log" >&2)

echo "PIPELINE_START $(date -Is)"
if [[ -n "$COLLECT_PID" ]]; then
  while [[ ! -f "$OUT/MF3ZP_OBSERVATION_COLLECTION_MANIFEST.json" ]]; do
    if ! kill -0 "$COLLECT_PID" 2>/dev/null; then
      echo "COLLECTOR_EXITED_WITHOUT_MANIFEST pid=$COLLECT_PID"
      exit 20
    fi
    sleep 30
  done
fi

if ! "$PY" - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/training/mf3zp_qwen_uad_reference_v2/MF3ZP_OBSERVATION_COLLECTION_MANIFEST.json")
v = json.loads(p.read_text())
raise SystemExit(0 if v.get("status") == "PASS" else 21)
PY
then
  echo "COLLECTION_GATE_FAIL"
  exit 21
fi

echo "RUN_OBSERVATION_AUDIT $(date -Is)"
if ! "$PY" scripts/audit_mf3zp_v2_observations.py > "$LOG/observation_audit.stdout.log" 2> "$LOG/observation_audit.stderr.log"; then
  echo "OBSERVATION_AUDIT_FAIL"
  exit 22
fi

echo "RUN_ASSEMBLE $(date -Is)"
if ! "$PY" scripts/run_mf3zp_qwen_reference_v2.py assemble > "$LOG/assemble.stdout.log" 2> "$LOG/assemble.stderr.log"; then
  echo "ASSEMBLE_FAIL"
  exit 23
fi

echo "RUN_QWEN_ANNOTATE $(date -Is)"
if ! "$PY" scripts/run_mf3zp_qwen_reference_v2.py annotate --max-workers 8 > "$LOG/annotate.stdout.log" 2> "$LOG/annotate.stderr.log"; then
  echo "ANNOTATE_FAIL"
  exit 24
fi

echo "RUN_EXPLORATORY_SCOUT $(date -Is)"
if ! "$PY" scripts/run_mf3zp_qwen_reference_v2.py scout > "$LOG/scout.stdout.log" 2> "$LOG/scout.stderr.log"; then
  echo "SCOUT_FAIL"
  exit 25
fi

echo "PIPELINE_END $(date -Is)"
