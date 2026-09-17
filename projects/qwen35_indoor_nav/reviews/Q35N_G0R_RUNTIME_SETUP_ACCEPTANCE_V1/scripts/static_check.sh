#!/usr/bin/env bash
set -euo pipefail

line=/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav
out="$line/reviews/Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1"
envdir="$line/.envs/q35n_habitat_v017_g0r"
build="$line/runtime/q35n_habitat_v017_g0r"
cache="$line/.cache/q35n_habitat_v017_g0r"
py="$envdir/bin/python3.10"

for p in "$out" "$envdir" "$build" "$cache"; do
  resolved=$(readlink -f "$p")
  case "$resolved" in
    "$line"/*) ;;
    *) echo "FAIL path_outside_line $p -> $resolved"; exit 1 ;;
  esac
done
echo "PASS all_target_paths_inside_line"

external_links=0
for base in "$out" "$envdir" "$build" "$cache"; do
  while IFS= read -r -d '' link; do
    target=$(readlink -f "$link" 2>/dev/null || true)
    case "$target" in "$line"/*|'') ;; *) external_links=$((external_links + 1)); echo "FAIL external_link $link -> $target";; esac
  done < <(find "$base" -type l -print0)
done
test "$external_links" -eq 0
echo "PASS no_external_symlink_targets"

while IFS= read -r -d '' json; do
  "$py" -m json.tool "$json" >/dev/null
done < <(find "$out" -type f -name '*.json' -print0)
"$py" - "$out" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
for path in out.rglob("*.jsonl"):
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            json.loads(line)

result = json.loads((out / "result.json").read_text())
assert result["runtime_pass"] is False
assert result["renderer_pass"] is False
assert result["scientific_pass"] is False
assert result["navigation_gain"] is None
assert result["new_training_runs"] == 0
assert result["qwen_load_count"] == 0
assert result["family_generation_count"] == 0
assert result["simulator_smoke_run_count"] == 0
assert result["action_count"] == 0

obs = json.loads((out / "OBSERVATION_MANIFEST.json").read_text())
assert obs["observation_files"] == [] and obs["actions_executed"] == 0

restore = json.loads((out / "GPU_RESTORE_RESULT.json").read_text())
assert restore["placeholder_release_performed"] is False
assert restore["own_gpu_contexts_remaining"] == 0

scene = json.loads((out / "SCENE_ASSET_FINGERPRINT.json").read_text())
assert all(len(item["sha256"]) == 64 for item in scene["files"])
print("PASS json_and_result_invariants")
PY

test "$(wc -l < "$out/SOURCE_SNAPSHOT_MANIFEST.tsv")" -eq 24
test "$(find "$build/src/habitat-sim" -type f | wc -l)" -eq 17940
echo "PASS source_snapshot_cardinality"

sha256sum -c - <<EOF
52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85  $build/src/habitat-sim/LICENSE
cbb2e22b60d98806fa3bf51db7df8e79660510b010c5d3db613a7be47c76d866  $build/src/habitat-sim/requirements.txt
a2ecb5db1bda83fad54c4538f087d2ef966f8109d156f548d465e5ff829e9d92  $build/src/habitat-sim/setup.py
10d0db98ee6f9297a58cad5c8edefcc9f133ca4da8f9fb0de048dbe4a6aafa2a  $envdir/bin/python3.10
ab0f715afe9f1a0d77ca6c91e6ff41f589b7d649618c5c1cd3776e13ccd075b1  /mnt/data_nas/deeprobotics/daiyang/vla/third_party/ETP-R1/data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb
0f36abb98ee3545d7e1f5103882269206feaaea05dcb54014333290b1d2b00c8  /mnt/data_nas/deeprobotics/daiyang/vla/third_party/ETP-R1/data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.navmesh
bdb833931a9196d2c72df8ead2842d9a161e8dc7d42da1536b10b461a9b14d8a  /mnt/data_nas/deeprobotics/daiyang/vla/third_party/ETP-R1/data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.house
b7b522bd2d0aa59ee3aec15e509f0b22ea63115b8b66301b50a3a3a0be0a7a26  /mnt/data_nas/deeprobotics/daiyang/vla/third_party/ETP-R1/data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy_semantic.ply
EOF
echo "PASS source_python_scene_hashes"

(cd "$line/reviews/Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1" && sha256sum -c SHA256SUMS >/dev/null)
(cd "$line/reviews/Q35N_P2R1_SPEC_CORRECTIONS_V1" && sha256sum -c SHA256SUMS >/dev/null)
echo "PASS p2_and_p2r1_sha256_13_of_13_each"

test -z "$(find "$cache/wheelhouse" -maxdepth 1 -type f -print -quit)"
test -z "$(find "$out/observations" -maxdepth 1 -type f -print -quit)"
test "$(find "$envdir/lib/python3.10/site-packages" -maxdepth 1 -type d -name 'habitat*' | wc -l)" -eq 0
echo "PASS hard_stop_left_no_wheels_observations_or_habitat_install"

if find "$out" "$envdir" "$build" "$cache" -type f -printf '%f\n' | grep -Eiq '(qwen|model[-_]?weights|checkpoint|family[-_]?record)'; then
  echo "FAIL forbidden_artifact_name_detected"
  exit 1
fi
echo "PASS no_qwen_weight_checkpoint_or_family_artifact_names"

echo "STATIC_CHECK_PASS"
