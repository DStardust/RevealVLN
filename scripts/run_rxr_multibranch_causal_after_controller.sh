#!/usr/bin/env bash
set -uo pipefail

cd /mnt/daiyang/vla
source scripts/activate_remote_vla.sh

dir="artifacts/phase1/rxr_train_expansion/multibranch_v2"
controller="${dir}/RXR_MULTIBRANCH_CONTROLLER_EXECUTION_V2.json"
controller_done="${dir}/RXR_MULTIBRANCH_CONTROLLER_V2_DONE"
log="${dir}/RXR_MULTIBRANCH_CAUSAL_V2.log"
marker="${dir}/RXR_MULTIBRANCH_CAUSAL_V2_DONE"
exec >>"${log}" 2>&1
echo "MULTIBRANCH_CAUSAL_WAIT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for _ in $(seq 1 720); do
  [[ -f "${controller}" ]] && break
  if [[ -f "${controller_done}" ]] && ! grep -qx 'exit=0' "${controller_done}"; then
    echo "controller failed: $(cat "${controller_done}")"
    exit 1
  fi
  sleep 30
done
if [[ ! -f "${controller}" ]]; then
  echo "timed out waiting for multibranch controller"
  exit 2
fi
python scripts/analyze_rxr_expansion_multibranch_causal_v2.py
status=$?
echo "MULTIBRANCH_CAUSAL_END $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${status}"
printf 'exit=%s\n' "${status}" >"${marker}.part"
mv "${marker}.part" "${marker}"
exit "${status}"
