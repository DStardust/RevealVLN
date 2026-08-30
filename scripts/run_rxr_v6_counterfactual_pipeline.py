#!/usr/bin/env python3
"""Seal, run, resume, and assemble RxR-train V6 paired rollouts."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
WORKER = ROOT / "scripts/rxr_v6_counterfactual_worker.py"
PIPELINE = ROOT / "scripts/run_rxr_v6_counterfactual_pipeline.py"
TRAINER = ROOT / "scripts/train_rxr_v6_relative_advantage.py"
DESIGN = ROOT / "artifacts/design/MF2_POLICY_RELATIVE_REVERSIBLE_ADVANTAGE_V6.md"
MODEL = ROOT / "revealnav_mf2r6/model.py"
MAX_EVENTS_PER_EPISODE = 1
CANDIDATE_POLICY = (
    "V5.16 native-first retained alternative with the frozen-pass "
    "source-balanced RxR Q V5.1 proposal head"
)
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
BASE = ROOT / "artifacts/phase1/rxr_v6"
SEEDS = (20260826, 20260827, 20260828)
METRICS = (
    "ndtw", "sdtw", "spl", "success", "distance_to_goal",
    "path_length", "steps_taken", "oracle_success",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(part, path)


def paths(cohort: str) -> dict[str, Path]:
    root = BASE / cohort
    return {
        "root": root,
        "protocol": root / "RXR_V6_PAIR_PROTOCOL.json",
        "selection": root / "RXR_V6_EPISODE_SELECTION.json",
        "progress": root / "RXR_V6_PAIR_PROGRESS.json",
        "runs": root / "runs",
        "targets": root / "targets",
        "manifest": root / "RXR_V6_PAIRED_DATASET_MANIFEST.json",
        "arrays": root / "RXR_V6_PAIRED_DATASET.npz",
    }


def load_english_episodes() -> list[dict]:
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    rows = []
    for episode in episodes:
        language = episode.get("instruction", {}).get("language")
        if language not in ("en-US", "en-IN"):
            continue
        scene = Path(episode["scene_id"]).parts[-2]
        rows.append({
            "episode_id": str(episode["episode_id"]),
            "trajectory_id": str(episode.get("trajectory_id")),
            "scene_id": scene,
            "language": language,
        })
    if not rows or len({row["episode_id"] for row in rows}) != len(rows):
        raise RuntimeError("RxR English train episode identity drift")
    return rows


def select_episodes(cohort: str, count: int) -> list[dict]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in load_english_episodes():
        by_scene[row["scene_id"]].append(row)
    for scene, rows in by_scene.items():
        rows.sort(key=lambda row: stable_hash({
            "cohort": cohort, "scene": scene, "episode": row["episode_id"]
        }))
    scene_order = sorted(by_scene, key=lambda scene: stable_hash({
        "cohort": cohort, "scene": scene,
    }))
    selected = []
    depth = 0
    while len(selected) < count:
        added = False
        for scene in scene_order:
            if depth < len(by_scene[scene]):
                row = dict(by_scene[scene][depth])
                row["seed"] = SEEDS[
                    int(stable_hash(row), 16) % len(SEEDS)
                ]
                row["scene_fold"] = int(stable_hash({
                    "v6_scene_fold": scene,
                }), 16) % 5
                selected.append(row)
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise RuntimeError("requested V6 cohort exceeds English RxR train")
        depth += 1
    return selected


def seal(cohort: str, count: int) -> dict:
    layout = paths(cohort)
    selection = select_episodes(cohort, count)
    selection_value = {
        "schema_version": "revealnav-rxr-v6-episode-selection/1",
        "cohort": cohort,
        "split": "train",
        "selection_rule": "scene-balanced deterministic hash round-robin",
        "episodes": selection,
        "episode_count": len(selection),
        "scene_count": len({row["scene_id"] for row in selection}),
        "fold_counts": dict(sorted(Counter(
            str(row["scene_fold"]) for row in selection
        ).items())),
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    protocol = {
        "schema_version": "revealnav-rxr-v6-pair-protocol/1",
        "status": "SEALED_BEFORE_V6_PAIR_COLLECTION",
        "cohort": cohort,
        "split": "RxR-train-English-only",
        "episode_count": count,
        "seeds": list(SEEDS),
        "candidate_policy": CANDIDATE_POLICY,
        "maximum_events_per_episode": MAX_EVENTS_PER_EPISODE,
        "pair": (
            "deterministic replay; action prefix through native outbound and "
            "post-native causal feature content must be identical"
        ),
        "macro": (
            "native outbound, verified physical return, exact topology restore, "
            "retained alternative commit, then unchanged frozen ETP continuation"
        ),
        "target": {
            "delta": "0.50*delta_ndtw+0.25*delta_sdtw+0.25*delta_spl",
            "positive_means": "macro better than native continuation",
        },
        "scene_partition": "sha256({v6_scene_fold: scene_id}) mod 5",
        "online_scalars": [
            "navigation_step/max_len", "remaining_steps/max_len",
            "causal_online_return_path_length_m/10",
        ],
        "forbidden_online_inputs": [
            "goal", "reference_path", "future_frame", "task_metric",
            "counterfactual_outcome", "val_seen", "val_unseen", "test",
        ],
        "sources": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size, "sha256": sha256_file(path)
            }
            for path in (DESIGN, WORKER, PIPELINE, TRAINER, MODEL, DATASET)
        },
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    if layout["protocol"].exists() and json.loads(
        layout["protocol"].read_text()
    ) != protocol:
        raise RuntimeError("sealed V6 pair protocol drift")
    if layout["selection"].exists() and json.loads(
        layout["selection"].read_text()
    ) != selection_value:
        raise RuntimeError("sealed V6 episode selection drift")
    if not layout["protocol"].exists():
        atomic_json(layout["protocol"], protocol)
    if not layout["selection"].exists():
        atomic_json(layout["selection"], selection_value)
    return {"protocol": protocol, "selection": selection_value}


def run_worker(command: list[str], gpu: int, stdout: Path, stderr: Path) -> dict:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONNOUSERSITE": "1",
    })
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w") as out, stderr.open("w") as err:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, stdout=out, stderr=err,
            text=True, check=False,
        )
    return {"returncode": completed.returncode, "gpu": gpu}


def valid_summary(path: Path, mode: str) -> bool:
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    return (
        value.get("status") == "PASS"
        and value.get("mode") == mode
        and value.get("split") == "train"
        and value.get("metrics") is not None
        and not value.get("unseen_or_test_read")
    )


def rejected_macro_summary(path: Path) -> bool:
    """Accept only evidenced post-target transaction rejection, never a label."""
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    status = value.get("status")
    if status == "FAIL" and value.get("error") != (
        "RuntimeError: V6 macro transaction did not complete"
    ):
        return False
    return (
        status in ("REJECTED_UNEXECUTABLE_MACRO", "FAIL")
        and value.get("mode") == "macro"
        and value.get("split") == "train"
        and value.get("metrics") is not None
        and value.get("target_reached") is True
        and value.get("target_return_scheduled") is True
        and value.get("target_alternative_committed") is False
        and not value.get("unseen_or_test_read")
    )


def progress(cohort: str, stage: str, selected: int, completed: int,
             active: dict, failures: list[dict], rejections: list[dict]) -> None:
    atomic_json(paths(cohort)["progress"], {
        "schema_version": "revealnav-rxr-v6-pair-progress/1",
        "cohort": cohort, "stage": stage, "selected": selected,
        "completed": completed, "remaining": selected - completed,
        "active": active, "failures": failures, "rejections": rejections,
    })


def parallel_jobs(cohort: str, stage: str, jobs: list[dict],
                  gpus: tuple[int, ...]) -> None:
    active: dict[str, dict] = {}
    failures: list[dict] = []
    rejections: list[dict] = []
    completed = 0
    progress(cohort, stage, len(jobs), completed, active, failures, rejections)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        running = {}
        iterator = iter(jobs)
        for slot, gpu in enumerate(gpus):
            try:
                job = next(iterator)
            except StopIteration:
                break
            future = pool.submit(
                run_worker, job["command"], gpu, job["stdout"], job["stderr"]
            )
            running[future] = (slot, gpu, job)
            active[str(slot)] = {"gpu": gpu, "id": job["id"]}
        progress(cohort, stage, len(jobs), completed, active, failures, rejections)
        while running:
            done, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                slot, gpu, job = running.pop(future)
                result = future.result()
                rejected = (
                    job["mode"] == "macro"
                    and rejected_macro_summary(job["summary"])
                )
                if not result["returncode"] and rejected:
                    rejections.append({
                        "id": job["id"],
                        "reason": "unexecutable_macro_transaction",
                    })
                elif result["returncode"] or not valid_summary(
                    job["summary"], job["mode"]
                ):
                    failures.append({
                        "id": job["id"], "returncode": result["returncode"],
                        "stderr": str(job["stderr"].relative_to(ROOT)),
                    })
                completed += 1
                active.pop(str(slot), None)
                try:
                    next_job = next(iterator)
                except StopIteration:
                    next_job = None
                if next_job is not None:
                    future = pool.submit(
                        run_worker, next_job["command"], gpu,
                        next_job["stdout"], next_job["stderr"],
                    )
                    running[future] = (slot, gpu, next_job)
                    active[str(slot)] = {"gpu": gpu, "id": next_job["id"]}
                progress(
                    cohort, stage, len(jobs), completed, active, failures,
                    rejections,
                )
    if failures:
        raise RuntimeError(f"{stage} has {len(failures)} failed workers")


def collect_shadows(cohort: str, gpus: tuple[int, ...]) -> None:
    layout = paths(cohort)
    selection = json.loads(layout["selection"].read_text())["episodes"]
    jobs = []
    for row in selection:
        run_dir = layout["runs"] / f"shadow_ep{row['episode_id']}_s{row['seed']}"
        summary = run_dir / "RUN_SUMMARY.json"
        if valid_summary(summary, "shadow"):
            continue
        jobs.append({
            "id": f"shadow:{row['episode_id']}:{row['seed']}",
            "mode": "shadow", "summary": summary,
            "stdout": layout["root"] / "logs" / f"shadow_ep{row['episode_id']}.out",
            "stderr": layout["root"] / "logs" / f"shadow_ep{row['episode_id']}.err",
            "command": [
                str(PYTHON), str(WORKER), "--episode-id", row["episode_id"],
                "--seed", str(row["seed"]), "--mode", "shadow",
                "--run-dir", str(run_dir),
            ],
        })
    parallel_jobs(cohort, "shadow", jobs, gpus)


def selected_events(cohort: str, maximum: int | None) -> list[dict]:
    layout = paths(cohort)
    selection = json.loads(layout["selection"].read_text())["episodes"]
    rows = []
    for episode in selection:
        run_dir = layout["runs"] / (
            f"shadow_ep{episode['episode_id']}_s{episode['seed']}"
        )
        summary = json.loads((run_dir / "RUN_SUMMARY.json").read_text())
        # The versioned protocol fixes this cap before collection. All events
        # from one episode remain in one scene fold.
        for event in summary["candidate_events"][:MAX_EVENTS_PER_EPISODE]:
            rows.append({
                **event,
                "shadow_run_dir": str(run_dir.relative_to(ROOT)),
            })
    rows.sort(key=lambda row: stable_hash({
        "v6_pair": row["event_id"],
    }))
    return rows if maximum is None else rows[:maximum]


def collect_macros(
    cohort: str, gpus: tuple[int, ...], maximum: int | None,
) -> None:
    layout = paths(cohort)
    events = selected_events(cohort, maximum)
    jobs = []
    for event in events:
        target = {
            key: event[key] for key in (
                "event_index", "checkpoint_id", "native_branch_id",
                "alternative_branch_id", "causal_state_sha256",
            )
        }
        target_path = layout["targets"] / f"{event['event_id']}.json"
        if target_path.exists() and json.loads(target_path.read_text()) != target:
            raise RuntimeError("V6 target drift")
        if not target_path.exists():
            atomic_json(target_path, target)
        run_dir = layout["runs"] / f"macro_{event['event_id']}"
        summary = run_dir / "RUN_SUMMARY.json"
        if valid_summary(summary, "macro") or rejected_macro_summary(summary):
            continue
        jobs.append({
            "id": f"macro:{event['event_id']}",
            "mode": "macro", "summary": summary,
            "stdout": layout["root"] / "logs" / f"macro_{event['event_id']}.out",
            "stderr": layout["root"] / "logs" / f"macro_{event['event_id']}.err",
            "command": [
                str(PYTHON), str(WORKER), "--episode-id", event["episode_id"],
                "--seed", str(event["controller_seed"]), "--mode", "macro",
                "--target", str(target_path), "--run-dir", str(run_dir),
            ],
        })
    parallel_jobs(cohort, "macro", jobs, gpus)


def task_utility(metrics: dict) -> float:
    return (
        0.50 * float(metrics["ndtw"])
        + 0.25 * float(metrics["sdtw"])
        + 0.25 * float(metrics["spl"])
    )


def assemble(cohort: str, maximum: int | None) -> dict:
    layout = paths(cohort)
    events = selected_events(cohort, maximum)
    arrays: dict[str, list[np.ndarray | float]] = defaultdict(list)
    records = []
    rejections = []
    for event in events:
        shadow_dir = ROOT / event["shadow_run_dir"]
        macro_dir = layout["runs"] / f"macro_{event['event_id']}"
        shadow = json.loads((shadow_dir / "RUN_SUMMARY.json").read_text())
        macro = json.loads((macro_dir / "RUN_SUMMARY.json").read_text())
        prefix = int(event["prefix_action_count"])
        shadow_actions = (shadow_dir / "base_trace.jsonl").read_text().splitlines()
        macro_actions = (macro_dir / "base_trace.jsonl").read_text().splitlines()
        causal_match = (
            len(macro.get("candidate_events", [])) > event["event_index"]
            and macro["candidate_events"][event["event_index"]][
                "causal_state_sha256"
            ] == event["causal_state_sha256"]
        )
        if shadow["status"] != "PASS" or not causal_match:
            raise RuntimeError("invalid V6 paired worker evidence")
        if shadow_actions[:prefix] != macro_actions[:prefix]:
            raise RuntimeError("V6 deterministic same-state action prefix drift")
        if rejected_macro_summary(macro_dir / "RUN_SUMMARY.json"):
            rejections.append({
                "event_id": event["event_id"],
                "episode_id": event["episode_id"],
                "reason": "unexecutable_macro_transaction",
                "included_as_label": False,
            })
            continue
        if not (
            macro["status"] == "PASS"
            and macro["target_alternative_committed"] is True
        ):
            raise RuntimeError("invalid V6 paired worker evidence")
        feature_path = (ROOT / event["feature_path"]).resolve()
        if (
            ROOT not in feature_path.parents or feature_path.is_symlink()
            or feature_path.stat().st_size != event["feature_bytes"]
            or sha256_file(feature_path) != event["feature_sha256"]
        ):
            raise RuntimeError("V6 causal feature provenance drift")
        with np.load(feature_path, allow_pickle=False) as feature:
            values = {key: feature[key] for key in feature.files}
        for key in (
            "instruction", "post_observation", "temporal_history",
            "checkpoint", "native", "alternative", "scalars",
        ):
            arrays[key].append(values[key])
        native_metrics = shadow["metrics"]
        macro_metrics = macro["metrics"]
        if any(
            not math.isfinite(float(metrics[key]))
            for metrics in (native_metrics, macro_metrics) for key in METRICS
        ):
            raise RuntimeError("non-finite V6 paired task metric")
        metric_delta = {
            key: float(macro_metrics[key]) - float(native_metrics[key])
            for key in METRICS
        }
        delta = task_utility(macro_metrics) - task_utility(native_metrics)
        arrays["target"].append(delta)
        records.append({
            "row_index": len(records),
            **{key: event[key] for key in (
                "event_id", "episode_id", "trajectory_id", "scene_id",
                "language", "controller_seed", "post_navigation_step",
                "checkpoint_id", "native_branch_id", "alternative_branch_id",
                "causal_state_sha256", "online_return_path_length_m",
            )},
            "scene_fold": int(stable_hash({
                "v6_scene_fold": event["scene_id"],
            }), 16) % 5,
            "paired_prefix_action_count": prefix,
            "paired_prefix_exact": True,
            "native_metrics": {key: float(native_metrics[key]) for key in METRICS},
            "macro_metrics": {key: float(macro_metrics[key]) for key in METRICS},
            "metric_delta_macro_minus_native": metric_delta,
            "relative_advantage": delta,
            "macro_better": delta > 0.0,
            "feature_path": event["feature_path"],
            "feature_sha256": event["feature_sha256"],
        })
    if not records:
        raise RuntimeError("no complete V6 counterfactual pairs")
    tensor_arrays = {
        key: np.asarray(value, dtype=(
            np.float16 if key not in ("scalars", "target") else np.float32
        ))
        for key, value in arrays.items()
    }
    atomic_npz(layout["arrays"], tensor_arrays)
    fold_counts = Counter(str(row["scene_fold"]) for row in records)
    manifest = {
        "schema_version": "revealnav-rxr-v6-paired-dataset/1",
        "status": "RXR_V6_PAIRED_DATASET_READY",
        "cohort": cohort,
        "records": records,
        "rejections": rejections,
        "metadata": {
            "attempted_pairs": len(events),
            "pairs": len(records),
            "rejected_unexecutable_pairs": len(rejections),
            "episodes": len({row["episode_id"] for row in records}),
            "scenes": len({row["scene_id"] for row in records}),
            "positive_pairs": sum(row["macro_better"] for row in records),
            "negative_or_tied_pairs": sum(
                not row["macro_better"] for row in records
            ),
            "fold_counts": dict(sorted(fold_counts.items())),
            "mean_relative_advantage": float(np.mean(tensor_arrays["target"])),
            "same_state_pairs": len(records),
            "future_information_used_for_online_input": 0,
            "unseen_or_test_read": False,
            "paper_result": False,
        },
        "arrays": {
            "path": str(layout["arrays"].relative_to(ROOT)),
            "bytes": layout["arrays"].stat().st_size,
            "sha256": sha256_file(layout["arrays"]),
        },
        "protocol": {
            "path": str(layout["protocol"].relative_to(ROOT)),
            "sha256": sha256_file(layout["protocol"]),
        },
        "correctness_revision": {
            "name": "V6.2.1 transaction-rejection semantics",
            "design": str(DESIGN.relative_to(ROOT)),
            "design_sha256": sha256_file(DESIGN),
            "assembler_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(layout["manifest"], manifest)
    return manifest


def parse_gpus(value: str) -> tuple[int, ...]:
    gpus = tuple(int(item) for item in value.split(","))
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus must contain non-negative indices")
    return gpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "seal", "shadows", "macros", "assemble", "all",
    ))
    parser.add_argument("--cohort", default="pilot_v6_0")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    args = parser.parse_args()
    layout = paths(args.cohort)
    if args.command in ("seal", "all"):
        value = seal(args.cohort, args.episodes)
        print(json.dumps({
            "status": value["protocol"]["status"],
            "episodes": value["selection"]["episode_count"],
            "scenes": value["selection"]["scene_count"],
            "protocol": str(layout["protocol"].relative_to(ROOT)),
        }, sort_keys=True))
    elif not layout["protocol"].is_file() or not layout["selection"].is_file():
        raise RuntimeError("seal V6 protocol and selection before execution")
    gpus = parse_gpus(args.gpus)
    if args.command in ("shadows", "all"):
        collect_shadows(args.cohort, gpus)
    if args.command in ("macros", "all"):
        collect_macros(args.cohort, gpus, args.max_pairs)
    if args.command in ("assemble", "all"):
        value = assemble(args.cohort, args.max_pairs)
        print(json.dumps(value["metadata"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
