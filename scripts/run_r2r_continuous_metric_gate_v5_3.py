#!/usr/bin/env python3
"""Seal, run, and verify the corrected V5.3 paired R2R controller gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/r2r_continuous_controller_worker_v5_3.py"
V52_WORKER = ROOT / "scripts/r2r_continuous_controller_worker_v5_2.py"
V5_PILOT = ROOT / "scripts/r2r_action_enabled_pilot_worker_v5.py"
FUSION = ROOT / "revealnav_mf2r4/fusion.py"
INTEGRATED = ROOT / "revealnav_mf2r4/integrated_controller.py"
CALIBRATION = ROOT / (
    "artifacts/evaluation/mf2_ecog_opp_shared_calibration_v3/"
    "RXR_ECOG_OPP_SHARED_CALIBRATION_RESULT_V3.json"
)
ACTIVE_SCREEN = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_3_activation_screen_v2/"
    "R2R_V5_3_ACTIVATION_SCREEN_PARTIAL_RESULT_V2.json"
)
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "R2R_VLNCE_v1-3_preprocessed_xlmr"
)
SEEDS = (20260826, 20260827, 20260828)
METRICS = (
    "success", "spl", "ndtw", "sdtw", "distance_to_goal",
    "path_length", "steps_taken", "oracle_success",
)
HIGHER_IS_BETTER = {"success", "spl", "ndtw", "sdtw", "oracle_success"}
COHORTS = {
    "seen-dev": {
        "split": "val_seen", "episodes": 20,
        "out": "mf2_r2r_continuous_metric_v5_3_seen_dev",
        "salt": "revealnav-v5.3-seen-development-20260827",
    },
    "unseen-confirm": {
        "split": "val_unseen", "episodes_per_scene": 2,
        "out": "mf2_r2r_continuous_metric_v5_3_unseen_confirm",
        "salt": "revealnav-v5.3-fresh-unseen-confirmation-20260827",
    },
    "seen-active-dev": {
        "split": "val_seen",
        "out": "mf2_r2r_continuous_metric_v5_3_seen_active_dev",
        "salt": "locked-by-outcome-blind-v5.3-partial-activation-screen",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def dataset_episodes(split: str) -> tuple[Path, list[dict]]:
    path = DATASET / split / f"{split}.json.gz"
    with gzip.open(path, "rt") as stream:
        rows = json.load(stream)["episodes"]
    return path, rows


def scene_id(row: dict) -> str:
    return Path(row["scene_id"]).stem


def rank(salt: str, row: dict) -> str:
    return hashlib.sha256(
        f"{salt}|{scene_id(row)}|{row['episode_id']}".encode()
    ).hexdigest()


def prior_unseen_protocols() -> list[Path]:
    paths = []
    for path in sorted((ROOT / "artifacts/evaluation").glob(
        "mf2_r2r_*/R2R_*PROTOCOL*.json"
    )):
        if "v5_3" in str(path):
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("selection") and "val_unseen" in json.dumps(value.get("scope", "")):
            paths.append(path)
        elif value.get("selection") and any(
            row.get("split") == "val_unseen" for row in value["selection"]
        ):
            paths.append(path)
    return paths


def select(cohort: str) -> tuple[list[dict], dict[str, str]]:
    config = COHORTS[cohort]
    if cohort == "seen-active-dev":
        value = json.loads(ACTIVE_SCREEN.read_text())
        if not (
            value.get("status")
            == "PARTIAL_SCREEN_ENGINEERING_PASS_ACTIVE_COHORT_READY"
            and value.get("active_cohort_size") == 24
            and value.get("selection_used_task_metrics") is False
            and value.get("result_contains_task_metrics") is False
            and value.get("paper_result") is False
        ):
            raise RuntimeError("outcome-blind active development cohort is invalid")
        selection = [{
            "episode_id": str(row["episode_id"]),
            "scene_id": row["scene_id"],
            "trajectory_id": row.get("trajectory_id"),
        } for row in value["active_cohort"]]
        if len(selection) != len({row["episode_id"] for row in selection}):
            raise RuntimeError("active development cohort contains duplicates")
        dataset_path, dataset_rows = dataset_episodes(config["split"])
        inventory = {
            str(row["episode_id"]): scene_id(row) for row in dataset_rows
        }
        if any(
            inventory.get(row["episode_id"]) != row["scene_id"]
            for row in selection
        ):
            raise RuntimeError("active cohort metadata differs from R2R val_seen")
        return selection, {
            str(ACTIVE_SCREEN.relative_to(ROOT)): sha256_file(ACTIVE_SCREEN),
            str(dataset_path.relative_to(ROOT)): sha256_file(dataset_path),
        }
    path, rows = dataset_episodes(config["split"])
    sources = {str(path.relative_to(ROOT)): sha256_file(path)}
    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(scene_id(row), []).append(row)
    excluded: set[str] = set()
    if cohort == "unseen-confirm":
        for protocol in prior_unseen_protocols():
            value = json.loads(protocol.read_text())
            excluded.update(str(row["episode_id"]) for row in value["selection"])
            sources[str(protocol.relative_to(ROOT))] = sha256_file(protocol)
        chosen = []
        for scene in sorted(by_scene):
            eligible = [
                row for row in by_scene[scene]
                if str(row["episode_id"]) not in excluded
            ]
            chosen.extend(sorted(
                eligible, key=lambda row: rank(config["salt"], row)
            )[:config["episodes_per_scene"]])
    else:
        ranked_scenes = sorted(
            by_scene,
            key=lambda value: hashlib.sha256(
                f"{config['salt']}|{value}".encode()
            ).hexdigest(),
        )[:config["episodes"]]
        chosen = [
            min(by_scene[scene], key=lambda row: rank(config["salt"], row))
            for scene in ranked_scenes
        ]
    selection = sorted(({
        "episode_id": str(row["episode_id"]),
        "scene_id": scene_id(row),
        "trajectory_id": row.get("trajectory_id"),
    } for row in chosen), key=lambda row: (row["scene_id"], row["episode_id"]))
    if len({row["episode_id"] for row in selection}) != len(selection):
        raise RuntimeError("selection contains duplicate episodes")
    if cohort == "unseen-confirm" and any(
        row["episode_id"] in excluded for row in selection
    ):
        raise RuntimeError("fresh unseen cohort overlaps a prior protocol")
    return selection, sources


def locations(cohort: str) -> tuple[Path, Path, Path]:
    out = ROOT / "artifacts/evaluation" / COHORTS[cohort]["out"]
    return (
        out,
        out / "R2R_CONTINUOUS_METRIC_PROTOCOL_V5_3.json",
        out / "R2R_CONTINUOUS_METRIC_RESULT_V5_3.json",
    )


def protocol_value(cohort: str) -> dict:
    selection, selection_sources = select(cohort)
    calibration = json.loads(CALIBRATION.read_text())
    config = calibration.get("selected_shared_config", {})
    if not (
        calibration.get("status") == "CONTROLLER_SHARED_CALIBRATION_ENGINEERING_PASS"
        and calibration.get("gold_payload_read") is False
        and config.get("opv_threshold") == 0.025
        and config.get("persistence_k") == 3
        and config.get("wrong_commitment_weight") == 5
    ):
        raise RuntimeError("frozen pre-outcome shared calibration is invalid")
    split = COHORTS[cohort]["split"]
    return {
        "schema_version": "revealnav-r2r-continuous-metric-protocol/5.3",
        "status": "SEALED_BEFORE_V5_3_PAIRED_RUNS",
        "cohort": cohort, "split": split,
        "scope": f"R2R-CE {split} engineering cohort; no test payload",
        "selection": selection,
        "selection_salt": COHORTS[cohort]["salt"],
        "selection_uses_only_episode_and_scene_metadata": True,
        "prior_diagnostic_episode_overlap": 0 if cohort == "unseen-confirm" else None,
        "seeds": list(SEEDS),
        "runs": {
            "baseline": len(selection),
            "revealnav": len(selection) * len(SEEDS),
            "total": len(selection) * (1 + len(SEEDS)),
        },
        "correctness_revision": {
            "kind": "versioned_correctness_revision_not_new_module",
            "opv_gate": "preservation_gain > 0.025 (strict)",
            "opv_threshold_selected_before_r2r_val_unseen_outcome": True,
            "persistent_option_states": [
                "untried", "active", "exhausted", "committed",
            ],
            "returned_branch_cannot_be_reselected": True,
        },
        "paired_design": {
            "baseline": "one deterministic frozen ETP-R1 run per episode",
            "treatment": "three fixed model-seed overlay runs per episode",
            "task_seed": 100, "base_policy_sampling": False,
            "unit": "episode paired to the identical ETP-R1 baseline",
            "uncertainty": "10000 deterministic episode bootstrap replicates",
        },
        "predeclared_interpretation": {
            "directional_positive": "mean SPL>0, nDTW>0, Success>=0",
            "statistically_positive": "bootstrap lower bounds SPL,nDTW>0",
            "scientific_failure_is_preserved": True,
        },
        "metrics": list(METRICS),
        "sources": {
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            str(V52_WORKER.relative_to(ROOT)): sha256_file(V52_WORKER),
            str(V5_PILOT.relative_to(ROOT)): sha256_file(V5_PILOT),
            str(FUSION.relative_to(ROOT)): sha256_file(FUSION),
            str(INTEGRATED.relative_to(ROOT)): sha256_file(INTEGRATED),
            str(CALIBRATION.relative_to(ROOT)): sha256_file(CALIBRATION),
            **selection_sources,
        },
        "paper_result": False,
        "test_or_test_challenge_allowed": False,
    }


def seal(cohort: str) -> int:
    _, protocol, _ = locations(cohort)
    value = protocol_value(cohort)
    if protocol.exists() and json.loads(protocol.read_text()) != value:
        raise RuntimeError("sealed V5.3 protocol drift")
    if not protocol.exists():
        atomic_json(protocol, value)
    print(json.dumps({
        "status": value["status"], "cohort": cohort,
        "runs": value["runs"], "protocol_sha256": sha256_file(protocol),
    }, indent=2))
    return 0


def job_name(mode: str, episode_id: str, seed: int | None) -> str:
    return (
        f"baseline_ep_{episode_id}" if mode == "baseline"
        else f"revealnav_seed_{seed}_ep_{episode_id}"
    )


def jobs(selection: list[dict]):
    yield from (("baseline", None, row) for row in selection)
    yield from (
        ("revealnav", seed, row) for seed in SEEDS for row in selection
    )


def launch(out: Path, split: str, mode: str, seed, episode, gpu: int) -> dict:
    name = job_name(mode, episode["episode_id"], seed)
    run_dir = out / "full/runs" / name
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    logs = out / "full/logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / f"{name}.stdout.log").open("w")
    stderr = (logs / f"{name}.stderr.log").open("w")
    command = [
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--episode-id", episode["episode_id"], "--mode", mode,
        "--split", split, "--run-dir", str(run_dir),
    ]
    if seed is not None:
        command.extend(("--seed", str(seed)))
    process = subprocess.Popen(command, cwd=ROOT, env={
        **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }, stdout=stdout, stderr=stderr)
    return {
        "name": name, "mode": mode, "seed": seed,
        "episode_id": episode["episode_id"], "gpu": gpu,
        "process": process, "streams": (stdout, stderr),
    }


def execute(cohort: str, gpus: tuple[int, ...], resume: bool) -> int:
    out, protocol_path, _ = locations(cohort)
    protocol = protocol_value(cohort)
    if not protocol_path.is_file() or json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("V5.3 protocol must be sealed before execution")
    completed = []
    completed_keys = set()
    run_root = out / "full"
    if run_root.exists() and not resume:
        raise RuntimeError(f"refusing to overwrite {run_root}")
    run_root.mkdir(parents=True, exist_ok=resume)
    if resume:
        for run_dir in sorted((run_root / "runs").glob("*")):
            path = run_dir / "RUN_SUMMARY.json"
            if path.is_file() and json.loads(path.read_text()).get("status") == "PASS":
                row = json.loads(path.read_text())
                key = (row["mode"], row.get("seed"), row["episode_id"])
                completed_keys.add(key)
                completed.append({"name": run_dir.name, "returncode": 0, "recovered": True})
            else:
                destination = run_root / "interrupted" / run_dir.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(run_dir, destination)
    queue = [
        row for row in jobs(protocol["selection"])
        if (row[0], row[1], row[2]["episode_id"]) not in completed_keys
    ]
    free = list(gpus)
    active = []
    while queue or active:
        while queue and free:
            mode, seed, episode = queue.pop(0)
            active.append(launch(
                out, protocol["split"], mode, seed, episode, free.pop(0)
            ))
        time.sleep(0.5)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({
                key: item[key] for key in (
                    "name", "mode", "seed", "episode_id", "gpu"
                )
            } | {"returncode": code})
            active.remove(item)
            free.append(item["gpu"])
            free.sort()
            print(json.dumps(completed[-1]), flush=True)
    failures = [row for row in completed if row["returncode"] != 0]
    atomic_json(run_root / "RUN_STATUS.json", {
        "status": "PASS" if not failures else "FAIL",
        "completed": completed, "failures": failures,
    })
    return 0 if not failures else 1


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def valid_chain(rows: list[dict]) -> bool:
    previous = "0" * 64
    for row in rows:
        if row.get("previous_hash") != previous:
            return False
        value = dict(row)
        claimed = value.pop("record_hash", None)
        digest = hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if digest != claimed:
            return False
        previous = claimed
    return True


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def verify(cohort: str) -> int:
    out, protocol_path, result_path = locations(cohort)
    protocol = protocol_value(cohort)
    if json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("sealed V5.3 protocol drift")
    observed = {}
    for path in sorted((out / "full/runs").glob("*/RUN_SUMMARY.json")):
        row = json.loads(path.read_text())
        key = row["mode"], row.get("seed"), row["episode_id"]
        if key in observed:
            raise RuntimeError("duplicate V5.3 run")
        observed[key] = row
    expected = {
        (mode, seed, row["episode_id"])
        for mode, seed, row in jobs(protocol["selection"])
    }
    reveal = [row for key, row in observed.items() if key[0] == "revealnav"]
    traces = [
        load_jsonl(out / "full/runs" / job_name(key[0], key[2], key[1]) / "controller_trace.jsonl")
        for key in observed if key[0] == "revealnav"
    ]
    activity = {
        name: sum(row["controller"][name] for row in reveal)
        for name in (
            "checkpointed_excursions", "continue_decisions", "backtrack_decisions",
            "successful_returns", "failed_returns", "threshold_suppressions",
            "ledger_suppressions",
        )
    }
    engineering = {
        "all_runs_complete": set(observed) == expected and all(
            row.get("status") == "PASS" for row in observed.values()
        ),
        "all_metrics_finite": all(
            row.get("metrics") is not None and all(
                math.isfinite(float(row["metrics"][name])) for name in METRICS
            ) for row in observed.values()
        ),
        "strict_checkpoints": all(row["controller"]["strict_load"] for row in reveal),
        "valid_controller_hash_chains": all(valid_chain(rows) for rows in traces),
        "opv_threshold_exact": all(row["opv_threshold"] == 0.025 for row in reveal),
        "excursions_have_post_decisions": (
            activity["checkpointed_excursions"]
            == activity["continue_decisions"] + activity["backtrack_decisions"]
        ),
        "all_requested_returns_succeeded": (
            activity["backtrack_decisions"] == activity["successful_returns"]
            and activity["failed_returns"] == 0
        ),
        "baseline_has_no_controller": all(
            row.get("controller") is None for key, row in observed.items()
            if key[0] == "baseline"
        ),
        "no_test_payload": True,
    }
    deltas = {name: {} for name in METRICS}
    paired = []
    for episode in protocol["selection"]:
        episode_id = episode["episode_id"]
        baseline = observed[("baseline", None, episode_id)]["metrics"]
        for seed in SEEDS:
            treatment = observed[("revealnav", seed, episode_id)]["metrics"]
            row = {"episode_id": episode_id, "seed": seed, "delta": {}}
            for name in METRICS:
                raw = float(treatment[name]) - float(baseline[name])
                benefit = raw if name in HIGHER_IS_BETTER else -raw
                deltas[name][(episode_id, seed)] = benefit
                row["delta"][name] = benefit
            row["interventions"] = observed[("revealnav", seed, episode_id)]["controller"]
            paired.append(row)
    episodes = [row["episode_id"] for row in protocol["selection"]]
    per_episode = {
        name: {
            episode_id: sum(deltas[name][(episode_id, seed)] for seed in SEEDS) / len(SEEDS)
            for episode_id in episodes
        } for name in METRICS
    }
    rng = random.Random(20260827)
    boot = {name: [] for name in METRICS}
    for _ in range(10000):
        sample = [rng.choice(episodes) for _ in episodes]
        for name in METRICS:
            boot[name].append(sum(per_episode[name][ep] for ep in sample) / len(sample))
    aggregate = {
        name: {
            "mean": sum(values.values()) / len(values),
            "median": quantile(list(values.values()), 0.5),
            "minimum": min(values.values()), "maximum": max(values.values()),
            "episode_bootstrap_95pct": [
                quantile(boot[name], 0.025), quantile(boot[name], 0.975),
            ],
        } for name, values in per_episode.items()
    }
    directional = (
        aggregate["spl"]["mean"] > 0 and aggregate["ndtw"]["mean"] > 0
        and aggregate["success"]["mean"] >= 0
    )
    statistical = (
        aggregate["spl"]["episode_bootstrap_95pct"][0] > 0
        and aggregate["ndtw"]["episode_bootstrap_95pct"][0] > 0
    )
    if activity["checkpointed_excursions"] == 0:
        outcome = "INACTIVE_NO_INTERVENTIONS"
    elif statistical:
        outcome = "STATISTICALLY_POSITIVE"
    elif directional:
        outcome = "DIRECTIONALLY_POSITIVE_INCONCLUSIVE"
    elif aggregate["spl"]["mean"] == aggregate["ndtw"]["mean"] == aggregate["success"]["mean"] == 0:
        outcome = "NO_MEASURED_EFFECT"
    else:
        outcome = "NEGATIVE_OR_MIXED"
    passed = all(engineering.values())
    result = {
        "schema_version": "revealnav-r2r-continuous-metric-result/5.3",
        "status": f"R2R_V5_3_ENGINEERING_{'PASS' if passed else 'FAIL'}_{outcome}",
        "cohort": cohort, "scientific_outcome": outcome,
        "engineering_gates": engineering,
        "predeclared_scientific_gates": {
            "directional_positive": directional,
            "statistically_positive": statistical,
        },
        "policy_activity": activity,
        "benefit_deltas_treatment_minus_baseline": aggregate,
        "paired_runs": paired,
        "protocol_sha256": sha256_file(protocol_path),
        "test_or_test_challenge_accessed": False, "paper_result": False,
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "run", "resume", "verify"))
    parser.add_argument("--cohort", choices=tuple(COHORTS), required=True)
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain distinct device indices")
    if args.mode == "seal":
        return seal(args.cohort)
    if args.mode == "run":
        return execute(args.cohort, gpus, False)
    if args.mode == "resume":
        return execute(args.cohort, gpus, True)
    return verify(args.cohort)


if __name__ == "__main__":
    raise SystemExit(main())
