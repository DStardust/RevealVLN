#!/usr/bin/env bash

set -euo pipefail

workspace=/mnt/data_nas/deeprobotics/daiyang/vla
cd "$workspace"
source scripts/activate_remote_vla.sh >/dev/null

causal_dir=artifacts/phase1/rxr_train_expansion/causal_frontend
log_dir="$causal_dir/logs"
mkdir -p "$log_dir"
exec >>"$log_dir/systemd_causal_filter.log" 2>&1

echo "CAUSAL_TAKEOVER_STARTED $(date --iso-8601=seconds)"
while [[ ! -f "$causal_dir/SYSTEMD_FRONTEND_SHARDS_DONE" ]]; do
  sleep 15
done

if [[ ! -f "$causal_dir/RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json" ]]; then
  python scripts/analyze_rxr_expansion_causal_candidates.py
fi
if [[ ! -f "$causal_dir/RXR_EXPANSION_CAUSAL_PREFIX_MEDIA_MANIFEST.json" ]]; then
  CR5_CAUSAL_MEDIA_GPU=0 python scripts/build_rxr_expansion_causal_prefix_media.py
fi
python scripts/run_rxr_expansion_causal_prefix_language.py --execute --workers 28
python scripts/finalize_rxr_expansion_auto_filter.py || true
printf 'COMPLETE %s\n' "$(date --iso-8601=seconds)" \
  >"$causal_dir/SYSTEMD_CAUSAL_FILTER_DONE"
echo "CAUSAL_TAKEOVER_COMPLETE $(date --iso-8601=seconds)"
