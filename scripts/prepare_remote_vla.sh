#!/usr/bin/env bash
set -euo pipefail

root=/mnt/daiyang/vla
priority=/var/tmp/daiyang_vla_priority
runtime=/var/tmp/daiyang_remote_vla_bootstrap
evidence="$root/artifacts/migration/remote_vla_bootstrap"
key="$root/.secret/migration_ed25519_8_130_54_48"
remote=root@8.130.54.48
destination=/mnt/data_nas/deeprobotics/daiyang/vla
mkdir -p "$runtime" "$evidence"
exec >>"$runtime/orchestrator.log" 2>&1
echo "REMOTE_VLA_BOOTSTRAP_WAIT $(date --iso-8601=seconds)"

while [[ ! -f "$priority/VLA_WORKSET_READY" ]]; do
  sleep 30
done
echo "REMOTE_VLA_BOOTSTRAP_START $(date --iso-8601=seconds)"

"$root/scripts/sync_codex_state_to_target.sh"

ssh_options=(
  -i "$key"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=10
  -o StrictHostKeyChecking=yes
)

rsync -a --protect-args \
  -e "ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes" \
  "$root/scripts/remote_vla_smoke.py" \
  "$root/scripts/activate_remote_vla.sh" \
  "$remote:$destination/scripts/"

ssh "${ssh_options[@]}" "$remote" 'bash -s' >"$runtime/remote_bootstrap.log" 2>&1 <<'REMOTE'
set -euo pipefail
workspace=/mnt/data_nas/deeprobotics/daiyang/vla
compat=/mnt/daiyang/vla
overlay="$workspace/.remote_runtime"

test -d "$workspace"
test ! -L "$workspace"
if [[ ! -e /mnt/daiyang ]]; then
  mkdir /mnt/daiyang
fi
test -d /mnt/daiyang
if [[ -L "$compat" ]]; then
  test "$(readlink -f "$compat")" = "$workspace"
elif [[ ! -e "$compat" ]]; then
  ln -s "$workspace" "$compat"
else
  echo "unsafe existing compatibility path: $compat" >&2
  exit 2
fi

test ! -e "$overlay"
mkdir "$overlay"
rsync -aH "$workspace/.envs/etpr1/" "$overlay/etpr1/"
rsync -aH \
  --exclude='/build/' \
  --exclude='*.so' \
  --exclude='/.git/' \
  "$workspace/third_party/habitat-sim/" "$overlay/habitat-sim/"

grep -rlI '^#!/mnt/daiyang/vla/.envs/etpr1/bin/python' "$overlay/etpr1/bin" \
  | xargs -r sed -i "1s|^#!/mnt/daiyang/vla/.envs/etpr1/bin/python|#!$overlay/etpr1/bin/python|"

python="$overlay/etpr1/bin/python"
test -x "$python"
"$python" -m pip uninstall -y magnum

export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
cd "$overlay/habitat-sim"
"$python" setup.py build_ext --inplace --parallel 8 --headless --no-update-submodules

export PYTHONPATH="$workspace:$overlay/habitat-sim:$workspace/third_party/ETP-R1"
export CUDA_VISIBLE_DEVICES=0
cd "$compat"
"$python" -m pip check
"$python" tests/test_toporeveal.py -v
"$python" scripts/remote_vla_smoke.py

rpm -q gcc gcc-c++ cmake ninja-build git mesa-libEGL-devel mesa-libGL-devel
touch "$workspace/artifacts/migration/REMOTE_VLA_READY"
REMOTE

rsync -a --protect-args \
  -e "ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes" \
  "$remote:$destination/artifacts/migration/REMOTE_VLA_SMOKE.json" \
  "$remote:$destination/artifacts/migration/REMOTE_VLA_READY" \
  "$evidence/"
cp "$runtime/remote_bootstrap.log" "$evidence/REMOTE_VLA_BOOTSTRAP.log"
touch "$priority/REMOTE_VLA_READY"
echo "REMOTE_VLA_BOOTSTRAP_PASS $(date --iso-8601=seconds)"
