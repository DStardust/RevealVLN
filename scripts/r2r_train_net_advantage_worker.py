#!/usr/bin/env python3
"""Collect causal R2R-train branch features above a frozen ETP-R1 policy."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ETPR1 = ROOT / "third_party/ETP-R1"
HABITAT_LAB = ROOT / "third_party/habitat-lab"
HABITAT_SIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / ".remote_runtime/habitat-sim"
)).resolve()
DATASET = ETPR1 / (
    "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz"
)
R2R_CHECKPOINT = ETPR1 / (
    "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)
JOINT_PRETRAINED = ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt"
)
for path in reversed((ROOT, ROOT / "scripts", ETPR1, HABITAT_LAB, HABITAT_SIM)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_native_preservation_worker_v5_5 as v55  # noqa: E402
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims, sha256_file,
)


install_runtime_shims()


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


def scene_id(scene_path: str) -> str:
    parts = Path(scene_path).parts
    if len(parts) < 2 or parts[-1] != f"{parts[-2]}.glb":
        raise RuntimeError("unexpected R2R scene path")
    return parts[-2]


def episode_metadata(episode_id: str) -> dict:
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    rows = [row for row in episodes if str(row["episode_id"]) == episode_id]
    if len(rows) != 1:
        raise RuntimeError("R2R train episode identity is not unique")
    row = rows[0]
    return {
        "episode_id": episode_id,
        "trajectory_id": str(row["trajectory_id"]),
        "scene_id": scene_id(row["scene_id"]),
    }


class TrainNetAdvantageFeatureController:
    """Observe every aligned branch set without changing the native action."""

    def __init__(
        self, seed: int, trace_path: Path, event_dir: Path, metadata: dict,
    ) -> None:
        self.seed = seed
        self.trace_path = trace_path
        self.event_dir = event_dir
        self.metadata = metadata
        self.feature_events: list[dict] = []
        self.instruction = None
        self.latest_history = None
        self.global_rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        self.global_current: dict[str, torch.Tensor] = {}
        self.step = 0
        self.pending_return_action = None

    def record_language(self, embeddings, mask) -> None:
        self.instruction = (
            (embeddings * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_panorama(self, embeddings, mask) -> None:
        self.latest_history = (
            (embeddings * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def _features(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ) -> dict[str, torch.Tensor]:
        if len(pilot._CURRENT_IDS) != 1 or len(pilot._LOCAL_FRONTIERS) != 1:
            raise RuntimeError("ETP graph identity hook is unavailable")
        self.global_current = {}
        for index, branch_id in enumerate(gmap_vp_ids[0]):
            if (
                index == 0 or branch_id is None
                or not bool(gmap_masks[0, index])
                or bool(gmap_visited_masks[0, index])
            ):
                continue
            self.global_current[str(branch_id)] = gmap_img_fts[0, index].detach()
        local_ids = set(self.global_current).intersection(pilot._LOCAL_FRONTIERS[0])
        return {
            branch: self.global_current[branch]
            for branch in sorted(local_ids)
        }

    def _capture(self, controls: tuple[str, ...], native: str) -> None:
        if not 2 <= len(controls) <= 4 or native not in controls:
            raise RuntimeError("aligned candidate width or native identity drift")
        graph = pilot._TRAINER.gmaps[0]
        checkpoint_id = pilot._CURRENT_IDS[0]
        if checkpoint_id not in graph.node_pos:
            raise RuntimeError("checkpoint position is absent from ETP graph")
        missing = [branch for branch in controls if branch not in graph.ghost_aug_pos]
        if missing:
            raise RuntimeError(f"candidate positions absent from ETP graph: {missing}")

        steps = len(self.rows)
        history = torch.stack([row[0] for row in self.rows]).detach().cpu().float()
        candidates = torch.zeros(steps, len(controls), 768, dtype=torch.float32)
        mask = torch.zeros(steps, len(controls), dtype=torch.bool)
        for time_index, (_, features) in enumerate(self.rows):
            for branch_index, branch in enumerate(controls):
                if branch in features:
                    candidates[time_index, branch_index] = features[branch].detach().cpu().float()
                    mask[time_index, branch_index] = True
        if not bool(mask[-1].all()):
            raise RuntimeError("current aligned controls lack causal embeddings")

        event_id = (
            f"r2r_ep{self.metadata['episode_id']}_s{self.step:02d}_"
            f"e{len(self.feature_events):02d}"
        )
        feature_path = self.event_dir / f"{event_id}.npz"
        atomic_npz(feature_path, {
            "instruction_embedding": self.instruction.detach().cpu().half().numpy(),
            "history_embeddings": history.half().numpy(),
            "candidate_embeddings": candidates.half().numpy(),
            "candidate_mask": mask.numpy(),
        })
        row = {
            "event_id": event_id,
            "episode_id": self.metadata["episode_id"],
            "trajectory_id": self.metadata["trajectory_id"],
            "scene_id": self.metadata["scene_id"],
            "navigation_step": self.step,
            "checkpoint_id": checkpoint_id,
            "checkpoint_position": [float(v) for v in graph.node_pos[checkpoint_id]],
            "candidate_branch_ids": list(controls),
            "candidate_positions": {
                branch: [float(v) for v in graph.ghost_aug_pos[branch]]
                for branch in controls
            },
            "native_branch_id": native,
            "feature_path": str(feature_path.relative_to(ROOT)),
            "feature_bytes": feature_path.stat().st_size,
            "feature_sha256": sha256_file(feature_path),
            "causal_prefix_only": True,
            "offline_target_truth_read": False,
        }
        self.feature_events.append(row)
        with self.trace_path.open("a") as stream:
            stream.write(json.dumps({
                "event_id": event_id,
                "step": self.step,
                "candidate_branch_ids": list(controls),
                "native_branch_id": native,
                "feature_sha256": row["feature_sha256"],
            }, sort_keys=True) + "\n")

    def record_navigation(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
        native_branch,
    ):
        if self.instruction is None or self.latest_history is None:
            return None
        current = self._features(
            gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks
        )
        self.global_rows.append((
            self.latest_history.detach(), dict(self.global_current)
        ))
        controls = tuple(current)
        if 2 <= len(controls) <= 4 and native_branch in current:
            self.rows = [
                (history, {
                    branch: candidates[branch]
                    for branch in controls if branch in candidates
                })
                for history, candidates in self.global_rows
            ]
            self._capture(controls, native_branch)
        self.step += 1
        return None

    def finalize_episode(self) -> None:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside project")
    metadata = episode_metadata(str(args.episode_id))
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    controller_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)

    state = TrainNetAdvantageFeatureController(
        args.seed, controller_trace, run_dir / "events", metadata
    )
    v55._CONTROLLER = state
    v55.install_native_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", f"train_net_advantage_{args.episode_id}",
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", "train", "TASK_CONFIG.DATASET.SPLIT", "train",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED), "IL.back_algo", "control",
        "INFERENCE.SPLIT", "train", "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]", "TORCH_GPU_ID", "0",
        "VIDEO_OPTION", "[]", "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]
    summary = {
        "schema_version": "revealnav-r2r-train-net-advantage-worker/1",
        "status": "RUNNING",
        **metadata,
        "seed": args.seed,
        "split": "train",
        "mode": "causal_shadow_collection",
        "task_metric_payload_read": False,
        "ground_truth_payload_read": False,
        "native_action_overridden": False,
        "argv": argv,
    }
    sys.argv = argv
    import run

    started = time.monotonic()
    try:
        run.main()
        state.finalize_episode()
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        pilot.close_envs()
        summary.update({
            "wall_time_s": round(time.monotonic() - started, 3),
            "peak_rss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_children_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "feature_event_count": len(state.feature_events),
            "feature_events": state.feature_events,
            "base_trace_sha256": sha256_file(base_trace),
            "controller_trace_sha256": sha256_file(controller_trace),
            "paper_result": False,
            "unseen_or_test_read": False,
        })
        atomic_json(run_dir / "RUN_SUMMARY.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "episode_id": metadata["episode_id"],
        "events": summary["feature_event_count"],
        "wall_time_s": summary["wall_time_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
