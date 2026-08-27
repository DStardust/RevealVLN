#!/usr/bin/env python3
"""Fail-closed final regression for MF2-CR2 Phase-0C machine evidence.

This gate validates engineering and evidence integrity only.  It deliberately
requires the private language packet to remain entirely unreviewed and cannot
turn the pending human subgate into a pass.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

from PIL import Image


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/runtime/phase0_correctness"
PACKET_DIR = ROOT / "artifacts/phase0/phase0c_language_review_35"
OUT = BASE / "PHASE0C_FINAL_REGRESSION.json"
PYTHON = ROOT / ".envs/etpr1/bin/python"
UV = ROOT / ".tools/uv/uv"
MIN_FREE_BYTES = 8 * 1024 ** 3
RESERVE_BYTES = 10_737_418_240

EXPECTED = {
    "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    "artifacts/phase0/evidence_current.json":
        "430f73ec5752783aa553c2c2f4fe4128247a93fa29bee0ae454c97f5547d9ce1",
    "artifacts/runtime/PHASE0_REVEAL_CLOSURE_MAIN_ACCEPTANCE.json":
        "36c2fb2bce69b8ebc337e0d2192c731c52dba33d8f0b5fe781ffa2a53783b435",
    "artifacts/runtime/ETPR1_RUNTIME_POST_FREEZE.txt":
        "c5cfea518dc455748afd3fff978dfbc64b4eed06bd477f4122fcdee8e103cee6",
    "METHOD_FREEZE_2_CORRECTNESS_REVISION_1.md":
        "de1cc32a890153d9962047841ff2dbc469c130f4cfb68de53c4ba5f9fb90262b",
    "METHOD_FREEZE_2_CORRECTNESS_REVISION_2.md":
        "3026e4696803ec6e7278831cb1f781a93a588f9fe09db73ddd67869e7c6e314b",
    "revealnav_cr1/__init__.py":
        "8626fe9b2b40513ca8089bf5eb56ebf8ca079465c32438cdaa4d4108b7f72b24",
    "revealnav_cr1/causal_frontend.py":
        "1b3523e2ac8e3f85522558d26414346a78d7ef0b9889991614e9f23f6d043e77",
    "artifacts/runtime/phase0_correctness/PHASE0C_ORACLE_LOWLEVEL_PROBE.json":
        "b2e94b8310dc14d9ae0fa024ae1fb67633fd77bbab41cbb4cdc9939d229e27ac",
    "artifacts/runtime/phase0_correctness/PHASE0C_COST_FRONTIER_WITNESS.json":
        "9b59ea9b7b9995aeb604b00587dd79a3af863ea51b16ddfe805b3f719f1a16d1",
    "artifacts/runtime/phase0_correctness/PHASE0C_COST_FRONTIER_ADJUDICATION.json":
        "43481d408358322a826f9769e269b38115ba0cacb794d2de377aaae4b6b12551",
    "artifacts/runtime/phase0_correctness/PHASE0C_COST_FRONTIER_REPLAY.json":
        "cfa53fd23b12505283265dbf0d0021d2415ca44667b3047ab15c078b1d41013d",
    "artifacts/runtime/phase0_correctness/CAUSAL_FRONTEND_CONTRACT_GATE.json":
        "5a64238c1cf66dcf5aedb01b2ab63575575164a4d2eccaea92407a1d2bbd75d8",
    "artifacts/runtime/phase0_correctness/CAUSAL_FRONTEND_MODEL_INTEGRATION.json":
        "3b5cf7d50cc5fab8a1e241f6a1e6416144866f74d059471490677a3901492ab9",
    "artifacts/runtime/phase0_correctness/PHYSICAL_INSPECT_ACQUISITION_GATE.json":
        "d76362431b05a962b0569915f82d45db1fe05e014afe878c34d3f8c5e8f0d93a",
    "artifacts/runtime/phase0_correctness/CAUSAL_FRONTEND_MULTIVIEW_INTEGRATION.json":
        "33354897e7db3fe0b5e88e727fbb817a5c279c3db5cdaba8bc4939f90cfd394b",
    "artifacts/runtime/phase0_correctness/ORACLE_SEMANTIC_BRANCH_TRACK_AUDIT.json":
        "e4b570dc9cdbe317d28b57507f1f74b9a16f92c8350810beb6b0f4dacd9df6a4",
    "artifacts/runtime/phase0_correctness/AUTOMATIC_SEMANTIC_CANDIDATE_GATE.json":
        "13797692e69847392b572f17f0559f36b685ec84b10051fc14c9f26c13ad2f7b",
    "artifacts/runtime/phase0_correctness/AUTOMATIC_SEMANTIC_MULTIPLICITY_ADJUDICATION.json":
        "e2dfba0b25f7df3cfcc4082567d95d897860595a1b6e0bf46bbe81846f696d3a",
    "artifacts/phase0/phase0c_language_review_35/PHASE0C_LANGUAGE_REVIEW_35.json":
        "b97f546d454d09a57c21153adc55bc02c30a4c694b07cd925091fac0b07a6784",
    "artifacts/phase0/phase0c_language_review_35/PHASE0C_LANGUAGE_REVIEW_35.csv":
        "b5edda64df6d32382afc00151a1307e7e28883234b2d3918c9a89150922bba9c",
    "artifacts/phase0/phase0c_language_review_35/REVIEW_GUIDE.md":
        "bf1abc474b157d2b5bafaa0c0a9f713a7665fef5a58e3a685d7babdde859640d",
    "artifacts/phase0/phase0c_language_review_35/PRIVATE_DO_NOT_DISTRIBUTE.txt":
        "231f32f882078576dc229a4dd2f9a24b00249a79e2abef0d9ef2e32ef6d3c985",
}

REPOS = {
    "habitat-sim": ("856d4b08c1a2632626bf0d205bf46471a99502b7", True),
    "habitat-lab": ("d6ed1c0a0e786f16f261de2beafe347f4186d0d8", True),
    "ETP-R1": ("a94b5c8fe20d1631e9e150c430a925543eb1cba9", False),
}
ETP_DIFF_SHA = \
    "5a207d26b5b582e8c810f5c95ecfe52897caecf69bb689874ad3222d21bfc521"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def load(relative: str):
    return json.loads((ROOT / relative).read_text())


def run(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, timeout=900,
                          **kwargs)


def all_true(mapping) -> bool:
    return isinstance(mapping, dict) and mapping and all(
        value is True for value in mapping.values())


def main() -> int:
    checks = []

    def record(name, passed, observed):
        checks.append({"name": name, "pass": bool(passed),
                       "observed": observed})
        print(("PASS " if passed else "FAIL ") + name, flush=True)

    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        observed = sha256_file(path) if path.is_file() else None
        record("pinned_sha:" + relative, observed == expected, observed)

    causal_contract = load(
        "artifacts/runtime/phase0_correctness/CAUSAL_FRONTEND_CONTRACT_GATE.json")
    causal_model = load(
        "artifacts/runtime/phase0_correctness/CAUSAL_FRONTEND_MODEL_INTEGRATION.json")
    physical = load(
        "artifacts/runtime/phase0_correctness/PHYSICAL_INSPECT_ACQUISITION_GATE.json")
    multiview = load(
        "artifacts/runtime/phase0_correctness/CAUSAL_FRONTEND_MULTIVIEW_INTEGRATION.json")
    record("causal_contract_6_of_6", causal_contract.get("status") == "PASS"
           and len(causal_contract.get("checks", {})) == 6
           and all_true(causal_contract["checks"]), causal_contract.get("decision"))
    record("real_model_hidden_view_noninterference_8_of_8",
           causal_model.get("status") == "PASS"
           and len(causal_model.get("checks", {})) == 8
           and all_true(causal_model["checks"])
           and all(run_item.get("network_attempts") == 0
                   for run_item in causal_model.get("runs", [])),
           causal_model.get("decision"))
    record("physical_inspect_9_of_9", physical.get("status") == "PASS"
           and len(physical.get("checks", {})) == 9
           and all_true(physical["checks"])
           and all(value is False for value in
                   physical.get("boundaries", {}).values()),
           physical.get("measurements"))
    record("post_inspect_multiview_8_of_8",
           multiview.get("status") == "PASS"
           and len(multiview.get("checks", {})) == 8
           and all_true(multiview["checks"])
           and multiview.get("counts") == {
               "model_records": 75, "action_records": 25,
               "graph_records": 25}, multiview.get("decision"))

    lowlevel = load(
        "artifacts/runtime/phase0_correctness/PHASE0C_ORACLE_LOWLEVEL_PROBE.json")
    cost = load(
        "artifacts/runtime/phase0_correctness/PHASE0C_COST_FRONTIER_ADJUDICATION.json")
    replay = load(
        "artifacts/runtime/phase0_correctness/PHASE0C_COST_FRONTIER_REPLAY.json")
    record("oracle_event_floor_104_events_32_scenes",
           lowlevel.get("status") == "PASS"
           and lowlevel["counts"].get("provisional_k3_events") == 104
           and lowlevel["counts"].get("scenes_with_event") == 32,
           lowlevel.get("counts"))
    record("cost_frontier_94_of_104_and_reentry_retained",
           cost.get("status") == "PASS"
           and cost["gates"].get("gate4_pass") is True
           and cost["gates"].get(
               "frozen_events_unique_at_least_two_budgets") == 94
           and cost["gates"].get("frozen_fraction") > 0.9
           and sum(cost["counts"]["frontier_status_by_controller_budget"]
                   ["frozen_shortest_path_compat"][budget].get(
                       "UNIQUE_LAST_SAFE_WITH_REENTRY", 0)
                   for budget in ("1.5", "2.0", "3.0", "4.0")) > 0,
           cost.get("gates"))
    record("cost_determinism_three_exact_replays",
           replay.get("status") == "PASS"
           and len(replay.get("results", [])) == 3
           and all(item.get("exact_canonical_match") is True
                   for item in replay.get("results", [])), replay.get("decision"))

    oracle_semantic = load(
        "artifacts/runtime/phase0_correctness/ORACLE_SEMANTIC_BRANCH_TRACK_AUDIT.json")
    auto_raw = load(
        "artifacts/runtime/phase0_correctness/AUTOMATIC_SEMANTIC_CANDIDATE_GATE.json")
    auto = load(
        "artifacts/runtime/phase0_correctness/AUTOMATIC_SEMANTIC_MULTIPLICITY_ADJUDICATION.json")
    record("oracle_machine_semantic_90_zero_ambiguity",
           oracle_semantic.get("status") == "PASS"
           and oracle_semantic["counts"].get("machine_geometric_admitted") == 90
           and oracle_semantic["counts"].get(
               "semantic_ambiguity_among_admitted") == 0,
           oracle_semantic.get("counts"))
    record("raw_automatic_failure_preserved",
           auto_raw.get("status") == "FAIL"
           and auto_raw["counts"].get("automatic_tracked_k3") == 11,
           auto_raw.get("decision"))
    record("many_to_one_adjudication_38_events_23_scenes",
           auto.get("status") == "PASS"
           and auto["counts"].get("tracked_k3") == 38
           and auto["counts"].get("tracked_scenes") == 23
           and auto["counts"]["status_counts"].get(
               "TRACKED_K3_WITHIN_REGION_MULTIPLICITY") == 27
           and auto["gates"].get("full_gate6_pass") is False,
           auto.get("counts"))
    shard_paths = sorted(BASE.glob("automatic_semantic_shards/shard_*.json"))
    shard_network = [json.loads(path.read_text()).get("network_attempts")
                     for path in shard_paths]
    record("automatic_workers_network_zero", len(shard_paths) == 7
           and shard_network == [0] * 7, shard_network)

    packet_path = PACKET_DIR / "PHASE0C_LANGUAGE_REVIEW_35.json"
    packet = json.loads(packet_path.read_text())
    rows = packet.get("rows", [])
    human_fields = packet.get("human_fields", [])
    tracked_ids = {item["provisional_event_id"] for item in auto["events"]
                   if item["adjudicated_status"].startswith("TRACKED_K3")}
    cost_ids = {item["provisional_event_id"] for item in cost["events"]
                if item["controllers"]["frozen_shortest_path_compat"]
                ["passes_two_budget_gate"]}
    eligible = tracked_ids & cost_ids
    row_ids = [row.get("event_id") for row in rows]
    record("review_selection_exact_fixed_intersection",
           len(eligible) == 35 and len(set(row_ids)) == 35
           and set(row_ids) == eligible
           and len({row.get("scene_id") for row in rows}) == 22,
           {"rows": len(row_ids), "scenes": len({row.get("scene_id")
                                                  for row in rows})})
    pending_ok = (
        packet.get("status") == "PASS_PENDING_HUMAN_REVIEW"
        and len(rows) == 35
        and packet.get("reviewed_true_count") == 0
        and packet.get("human_fields_prefilled") is False
        and packet.get("all_rows_pending") is True
        and len(human_fields) == 11
        and all(row.get("reviewed") is False
                and row.get("annotation_status") == "PENDING_HUMAN_REVIEW"
                and all(row.get(field) is None for field in human_fields)
                for row in rows))
    record("human_fields_all_null_and_unreviewed", pending_ok,
           {"rows": len(rows), "human_fields": human_fields})

    manifest = packet.get("media_manifest", [])
    actual_jpegs = sorted((PACKET_DIR / "private_media").glob("*.jpg"))
    manifest_paths = {item.get("path") for item in manifest}
    row_media = {path for row in rows for path in
                 row.get("private_media", []) +
                 [row.get("private_contact_sheet")]}
    media_ok, decoded = True, 0
    for item in manifest:
        path = ROOT / item["path"]
        try:
            resolved = path.resolve(strict=True)
            if (not path.is_file() or path.is_symlink()
                    or PACKET_DIR.resolve() not in resolved.parents
                    or path.stat().st_size != item["bytes"]
                    or sha256_file(path) != item["sha256"]):
                media_ok = False
                continue
            with Image.open(path) as image:
                image.load()
                expected_size = ((1120, 224) if path.name.endswith(
                    "_contact.jpg") else (224, 224))
                extrema = image.convert("RGB").getextrema()
                if image.size != expected_size or not all(
                        high - low > 10 for low, high in extrema):
                    media_ok = False
                decoded += 1
        except Exception:
            media_ok = False
    actual_rel = {str(path.relative_to(ROOT)) for path in actual_jpegs}
    record("private_media_175_hash_size_decode_exact",
           media_ok and decoded == 175 and len(manifest) == 175
           and manifest_paths == actual_rel == row_media
           and sum(item["bytes"] for item in manifest) ==
                   packet.get("media_total_bytes"),
           {"manifest": len(manifest), "decoded": decoded,
            "actual": len(actual_jpegs)})

    with (PACKET_DIR / "PHASE0C_LANGUAGE_REVIEW_35.csv").open(
            newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_human = [field for field in human_fields
                 if field not in {"reviewer_id", "review_timestamp"}]
    csv_ok = (len(csv_rows) == 35
              and [row["event_id"] for row in csv_rows] == row_ids
              and all(row["reviewed"] == "False"
                      and row["reviewer_id"] == ""
                      and row["review_timestamp"] == ""
                      and all(row[field] == "" for field in csv_human)
                      for row in csv_rows))
    record("csv_35_rows_all_human_cells_blank", csv_ok, len(csv_rows))

    checkpoint_map = causal_model.get("checkpoints", {})
    checkpoint_results = {}
    for name, metadata in checkpoint_map.items():
        path = ROOT / metadata["path"]
        checkpoint_results[name] = (path.is_file() and not path.is_symlink()
                                    and sha256_file(path) == metadata["sha256"])
    record("loaded_checkpoint_provenance_still_exact",
           len(checkpoint_results) == 3 and all(checkpoint_results.values()),
           checkpoint_results)

    env = dict(os.environ,
               VIRTUAL_ENV=str(ROOT / ".envs/etpr1"),
               PYTHONNOUSERSITE="1")
    proc = run([str(PYTHON), "-m", "unittest", "discover", "-s", "tests",
                "-v"], cwd=ROOT, env=dict(env, PYTHONPATH=str(ROOT)))
    record("toporeveal_unit_tests", proc.returncode == 0,
           (proc.stderr.strip().splitlines() or [""])[-1])
    proc = run([str(UV), "pip", "check"], cwd=ROOT, env=env)
    record("uv_pip_check", proc.returncode == 0,
           (proc.stdout + proc.stderr).strip()[-300:])
    proc = run([str(UV), "pip", "freeze"], cwd=ROOT, env=env)
    freeze_sha = hashlib.sha256(proc.stdout.encode()).hexdigest()
    record("environment_freeze_exact", proc.returncode == 0
           and freeze_sha == EXPECTED[
               "artifacts/runtime/ETPR1_RUNTIME_POST_FREEZE.txt"], freeze_sha)

    for name, (expected_head, clean) in REPOS.items():
        repo = ROOT / "third_party" / name
        head_proc = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
        status_proc = run(["git", "-C", str(repo), "status", "--porcelain=v1"])
        head = head_proc.stdout.strip()
        if clean:
            passed = (head == expected_head and not status_proc.stdout)
        else:
            diff_proc = subprocess.run(["git", "-C", str(repo), "diff"],
                                       capture_output=True, timeout=180)
            lines = status_proc.stdout.splitlines()
            tracked = [line for line in lines if not line.startswith("??")]
            untracked = [line[3:] for line in lines if line.startswith("??")]
            passed = (head == expected_head and len(tracked) == 14
                      and untracked == ["etpr1_compat.py"]
                      and hashlib.sha256(diff_proc.stdout).hexdigest()
                      == ETP_DIFF_SHA)
        record("repo_state:" + name, passed,
               {"head": head, "status_lines": len(status_proc.stdout.splitlines())})

    reserves = []
    for index in range(9, 28):
        path = ROOT / ".disk_reserve" / ("reserve_10G_%02d.bin" % index)
        reserves.append(path.is_file() and not path.is_symlink()
                        and path.stat().st_size == RESERVE_BYTES)
    record("reserves_09_27_unchanged", all(reserves),
           {"ok": sum(reserves), "total": 19})

    parts = glob.glob(str(ROOT / "**/*.part"), recursive=True)
    artifact_symlinks = [str(path) for path in
                         (ROOT / "artifacts").rglob("*") if path.is_symlink()]
    record("no_part_or_artifact_symlink", not parts and not artifact_symlinks,
           {"parts": len(parts), "artifact_symlinks": len(artifact_symlinks)})

    scripts = [
        "adjudicate_cost_frontier.py", "replay_cost_frontier_sample.py",
        "test_causal_frontend_contract.py", "causal_frontend_model_worker.py",
        "run_causal_frontend_integration.py",
        "physical_inspect_acquisition_gate.py",
        "run_causal_multiview_integration.py",
        "audit_oracle_semantic_branch_tracks.py",
        "automatic_semantic_candidate_worker.py",
        "run_automatic_semantic_candidate_gate.py",
        "adjudicate_automatic_semantic_multiplicity.py",
        "build_phase0c_language_review_packet.py",
        "accept_phase0c_correctness_v2.py",
        "phase0c_final_regression.py",
    ]
    compile_proc = run([str(PYTHON), "-m", "py_compile",
                        str(ROOT / "revealnav_cr1/__init__.py"),
                        str(ROOT / "revealnav_cr1/causal_frontend.py")] +
                       [str(ROOT / "scripts" / item) for item in scripts])
    record("new_sources_compile", compile_proc.returncode == 0,
           (compile_proc.stderr or "%d files" % (len(scripts) + 2)).strip())

    batch_names = tuple(scripts[:-1])
    live = []
    # Ignore this verifier's launcher ancestry.  The parent shell necessarily
    # contains the verifier command line and is not a surviving batch worker.
    ancestry = {os.getpid()}
    ancestor = os.getpid()
    while ancestor > 1:
        try:
            stat = (Path("/proc") / str(ancestor) / "stat").read_text()
            ancestor = int(stat[stat.rfind(")") + 2:].split()[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError,
                ValueError, IndexError):
            break
        ancestry.add(ancestor)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestry:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(name in command for name in batch_names):
            live.append({"pid": int(entry.name), "command": command[:300]})
    record("no_live_phase0c_workers", not live, live)

    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    record("disk_free_at_least_8gib", free >= MIN_FREE_BYTES, free)

    passed = all(item["pass"] for item in checks)
    output = {
        "gate": "mf2_cr2_phase0c_final_machine_regression",
        "revision": "phase0c-final-regression/2",
        "status": "PASS" if passed else "FAIL",
        "decision": ("MACHINE_EVIDENCE_PASS_PENDING_HUMAN_LANGUAGE_REVIEW"
                     if passed else "MACHINE_EVIDENCE_NO_GO"),
        "checks_passed": sum(item["pass"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "explicit_boundaries": {
            "canonical_frozen_spec_modified": False,
            "phase0_protocol_modified": False,
            "official_checkpoints_loaded_for_engineering_integration": True,
            "official_checkpoints_modified": False,
            "training_performed": False,
            "feature_generation_authorized": False,
            "human_review_performed": False,
            "human_fields_fabricated": False,
            "val_unseen_or_test_used": False,
            "network_used": False,
            "reserve_released": False,
            "private_review_media_distribution_authorized": False,
        },
        "remaining_blocker": "35 fixed candidates require complete, signed, "
                             "fail-closed human language/branch review",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"], "decision": output["decision"],
        "checks": "%d/%d" % (output["checks_passed"],
                                output["checks_total"]),
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
