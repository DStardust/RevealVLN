#!/usr/bin/env python3
"""V2 outcome-blind activation screen over all remaining val_seen episodes."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_v5_3_activation_screen as base  # noqa: E402


OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_3_activation_screen_v2"
PROTOCOL = OUT / "R2R_V5_3_ACTIVATION_SCREEN_PROTOCOL_V2.json"
RESULT = OUT / "R2R_V5_3_ACTIVATION_SCREEN_RESULT_V2.json"
CUMULATIVE = OUT / "R2R_V5_3_ACTIVATION_SCREEN_CUMULATIVE_RESULT_V2.json"
V1_PROTOCOL = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_3_activation_screen/"
    "R2R_V5_3_ACTIVATION_SCREEN_PROTOCOL.json"
)
V1_RESULT = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_3_activation_screen/"
    "R2R_V5_3_ACTIVATION_SCREEN_RESULT.json"
)
SALT = "revealnav-r2r-v5.3-outcome-blind-full-val-seen-extension/2"
ACTIVE_COHORT_LIMIT = 24


def extension_selection() -> list[dict]:
    v1 = json.loads(V1_PROTOCOL.read_text())
    excluded = {row["episode_id"] for row in v1["selection"]}
    with gzip.open(base.DATASET, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    grouped = defaultdict(list)
    for row in episodes:
        episode_id = str(row["episode_id"])
        if episode_id not in excluded:
            grouped[Path(row["scene_id"]).stem].append(row)
    ranked = []
    for scene, rows in grouped.items():
        rows.sort(key=lambda row: hashlib.sha256(
            f"{SALT}|{scene}|{row['episode_id']}".encode()
        ).hexdigest())
        for scene_round, row in enumerate(rows):
            digest = hashlib.sha256(
                f"{SALT}|round={scene_round}|{scene}|{row['episode_id']}".encode()
            ).hexdigest()
            ranked.append({
                "episode_id": str(row["episode_id"]),
                "scene_id": scene,
                "trajectory_id": row.get("trajectory_id"),
                "scene_round": scene_round,
                "screen_rank": digest,
            })
    selected = sorted(ranked, key=lambda row: (row["scene_round"], row["screen_rank"]))
    if len(selected) != 672 or len({row["episode_id"] for row in selected}) != 672:
        raise RuntimeError("V2 must contain every one of the 672 remaining episodes")
    return selected


def protocol_value() -> dict:
    v1_protocol = json.loads(V1_PROTOCOL.read_text())
    v1_result = json.loads(V1_RESULT.read_text())
    if not (
        v1_protocol.get("status") == "SEALED_BEFORE_OUTCOME_BLIND_ACTIVATION_SCREEN"
        and v1_result.get("status") == "ACTIVATION_SCREEN_PASS"
        and v1_result.get("screened_episodes") == 106
        and v1_result.get("selection_used_task_metrics") is False
    ):
        raise RuntimeError("completed V1 outcome-blind screen is required")
    selected = extension_selection()
    return {
        "schema_version": "revealnav-r2r-v5.3-activation-screen-protocol/2",
        "status": "SEALED_BEFORE_FULL_VAL_SEEN_ACTIVATION_EXTENSION",
        "scope": "all 672 R2R val_seen episodes absent from V1; development only",
        "selection_salt": SALT,
        "selection": selected,
        "screen_seed": base.SCREEN_SEED,
        "runs": len(selected),
        "cumulative_inventory_after_completion": 778,
        "selection_contract": {
            "worker_executes_no_controller_action": True,
            "worker_summary_contains_no_task_metrics": True,
            "verifier_must_not_open_etp_metric_files": True,
            "eligible": "activation_count > 0 under strict OPV > 0.025",
            "active_cohort_order": "V1 order then V2 scene-round/hash order",
            "active_cohort_limit": ACTIVE_COHORT_LIMIT,
            "navigation_outcomes_never_used_for_selection": True,
            "complete_val_seen_inventory_not_cherry_picked": True,
        },
        "sources": {
            str(Path(__file__).resolve().relative_to(ROOT)): base.sha256_file(Path(__file__).resolve()),
            str(base.WORKER.relative_to(ROOT)): base.sha256_file(base.WORKER),
            str(base.PILOT.relative_to(ROOT)): base.sha256_file(base.PILOT),
            str(base.FUSION.relative_to(ROOT)): base.sha256_file(base.FUSION),
            str(base.DATASET.relative_to(ROOT)): base.sha256_file(base.DATASET),
            str(base.CALIBRATION.relative_to(ROOT)): base.sha256_file(base.CALIBRATION),
            str(V1_PROTOCOL.relative_to(ROOT)): base.sha256_file(V1_PROTOCOL),
            str(V1_RESULT.relative_to(ROOT)): base.sha256_file(V1_RESULT),
        },
        "test_or_test_challenge_allowed": False,
        "paper_result": False,
    }


def configure_base() -> None:
    base.OUT = OUT
    base.PROTOCOL = PROTOCOL
    base.RESULT = RESULT
    base.ACTIVE_COHORT_LIMIT = ACTIVE_COHORT_LIMIT
    base.selection = extension_selection
    base.protocol_value = protocol_value


def verify_cumulative() -> int:
    code = base.verify()
    v1 = json.loads(V1_RESULT.read_text())
    v2 = json.loads(RESULT.read_text())
    active = list(v1["active_cohort"])
    observed = {}
    for path in sorted((OUT / "full/runs").glob("*/RUN_SUMMARY.json")):
        row = json.loads(path.read_text())
        observed[str(row["episode_id"])] = row
    for selected in protocol_value()["selection"]:
        summary = observed[selected["episode_id"]]
        if summary["controller"]["activation_count"] > 0:
            active.append({
                **selected,
                "activation_count": summary["controller"]["activation_count"],
                "maximum_preservation_gain": summary["controller"]["maximum_preservation_gain"],
            })
    value = {
        "schema_version": "revealnav-r2r-v5.3-activation-screen-cumulative/2",
        "status": (
            "FULL_VAL_SEEN_ACTIVATION_SCREEN_PASS"
            if code == 0 and v2["status"] == "ACTIVATION_SCREEN_PASS"
            else "FULL_VAL_SEEN_ACTIVATION_SCREEN_FAIL"
        ),
        "screened_episodes": v1["screened_episodes"] + v2["screened_episodes"],
        "active_episodes": len(active),
        "active_scenes": len({row["scene_id"] for row in active}),
        "active_rate": len(active) / 778,
        "active_cohort": active[:ACTIVE_COHORT_LIMIT],
        "active_cohort_limit": ACTIVE_COHORT_LIMIT,
        "active_cohort_ready": len(active) >= ACTIVE_COHORT_LIMIT,
        "selection_used_task_metrics": False,
        "result_contains_task_metrics": False,
        "v1_result_sha256": base.sha256_file(V1_RESULT),
        "v2_result_sha256": base.sha256_file(RESULT),
        "paper_result": False,
    }
    base.atomic_json(CUMULATIVE, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "FULL_VAL_SEEN_ACTIVATION_SCREEN_PASS" else 1


def main() -> int:
    configure_base()
    parser = base.argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "run", "resume", "verify"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain distinct device indices")
    if args.mode == "seal":
        return base.seal()
    if args.mode == "run":
        return base.execute(gpus, False)
    if args.mode == "resume":
        return base.execute(gpus, True)
    return verify_cumulative()


if __name__ == "__main__":
    raise SystemExit(main())
