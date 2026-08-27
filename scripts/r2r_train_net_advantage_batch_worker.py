#!/usr/bin/env python3
"""Collect causal R2R-train features for several episodes per ETP load."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_train_net_advantage_worker as single  # noqa: E402


class BatchFeatureCollector:
    def __init__(self, metadata: dict[str, dict], runs_root: Path) -> None:
        self.metadata = metadata
        self.runs_root = runs_root
        self.trainer = None
        self.current_episode_ids: tuple[str, ...] = ()
        self.current_ids: tuple[str, ...] = ()
        self.local_frontiers: tuple[dict[str, int], ...] = ()
        self.states = {
            episode_id: {
                "instruction": None,
                "latest_history": None,
                "global_rows": [],
                "step": 0,
                "events": [],
            }
            for episode_id in metadata
        }

    def episode_ids(self) -> tuple[str, ...]:
        if self.trainer is None or self.trainer.envs is None:
            raise RuntimeError("batch collector lacks the active trainer")
        return tuple(
            str(episode.episode_id)
            for episode in self.trainer.envs.current_episodes()
        )

    def record_language(self, embeddings, mask) -> None:
        episode_ids = self.episode_ids()
        if len(episode_ids) != embeddings.shape[0]:
            raise RuntimeError("language batch identity drift")
        for index, episode_id in enumerate(episode_ids):
            state = self.states[episode_id]
            if state["instruction"] is not None:
                raise RuntimeError("episode language was encoded more than once")
            state["instruction"] = (
                (embeddings[index] * mask[index].unsqueeze(-1)).sum(0)
                / mask[index].sum().clamp_min(1)
            ).detach()

    def record_panorama(self, embeddings, mask) -> None:
        episode_ids = self.episode_ids()
        if len(episode_ids) != embeddings.shape[0]:
            raise RuntimeError("panorama batch identity drift")
        for index, episode_id in enumerate(episode_ids):
            self.states[episode_id]["latest_history"] = (
                (embeddings[index] * mask[index].unsqueeze(-1)).sum(0)
                / mask[index].sum().clamp_min(1)
            ).detach()

    def publish_graph(self, trainer, current_ids) -> None:
        self.trainer = trainer
        self.current_episode_ids = self.episode_ids()
        self.current_ids = tuple(str(value) for value in current_ids)
        self.local_frontiers = tuple(
            {
                str(ghost_id): int(graph.ghost_embeds[ghost_id][1])
                for ghost_id, fronts in graph.ghost_fronts.items()
                if current_id in fronts
            }
            for graph, current_id in zip(trainer.gmaps, current_ids)
        )
        size = len(self.current_episode_ids)
        if not (
            size == len(self.current_ids) == len(self.local_frontiers)
            == len(trainer.gmaps)
        ):
            raise RuntimeError("graph batch identity drift")

    def capture(
        self,
        episode_id: str,
        state: dict,
        controls: tuple[str, ...],
        native: str,
        graph,
        checkpoint_id: str,
        rows,
    ) -> None:
        if checkpoint_id not in graph.node_pos:
            raise RuntimeError("checkpoint position is absent from ETP graph")
        missing = [branch for branch in controls if branch not in graph.ghost_aug_pos]
        if missing:
            raise RuntimeError(f"candidate positions absent from ETP graph: {missing}")
        history = torch.stack([row[0] for row in rows]).detach().cpu().float()
        candidates = torch.zeros(
            len(rows), len(controls), 768, dtype=torch.float32
        )
        mask = torch.zeros(len(rows), len(controls), dtype=torch.bool)
        for time_index, (_, features) in enumerate(rows):
            for branch_index, branch in enumerate(controls):
                if branch in features:
                    candidates[time_index, branch_index] = (
                        features[branch].detach().cpu().float()
                    )
                    mask[time_index, branch_index] = True
        if not bool(mask[-1].all()):
            raise RuntimeError("current aligned controls lack causal embeddings")

        event_id = (
            f"r2r_ep{episode_id}_s{state['step']:02d}_"
            f"e{len(state['events']):02d}"
        )
        feature_path = self.runs_root / f"ep_{episode_id}" / "events" / f"{event_id}.npz"
        single.atomic_npz(feature_path, {
            "instruction_embedding": state["instruction"].detach().cpu().half().numpy(),
            "history_embeddings": history.half().numpy(),
            "candidate_embeddings": candidates.half().numpy(),
            "candidate_mask": mask.numpy(),
        })
        metadata = self.metadata[episode_id]
        event = {
            "event_id": event_id,
            **metadata,
            "navigation_step": state["step"],
            "checkpoint_id": checkpoint_id,
            "checkpoint_position": [float(value) for value in graph.node_pos[checkpoint_id]],
            "candidate_branch_ids": list(controls),
            "candidate_positions": {
                branch: [float(value) for value in graph.ghost_aug_pos[branch]]
                for branch in controls
            },
            "native_branch_id": native,
            "feature_path": str(feature_path.relative_to(ROOT)),
            "feature_bytes": feature_path.stat().st_size,
            "feature_sha256": single.sha256_file(feature_path),
            "causal_prefix_only": True,
            "offline_target_truth_read": False,
        }
        state["events"].append(event)

    def record_navigation(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
        global_logits,
    ) -> None:
        size = len(self.current_episode_ids)
        if not (
            size == len(gmap_vp_ids) == gmap_img_fts.shape[0]
            == global_logits.shape[0]
        ):
            raise RuntimeError("navigation batch identity drift")
        native_indices = torch.argmax(global_logits, dim=-1).detach().cpu().tolist()
        for environment_index, episode_id in enumerate(self.current_episode_ids):
            state = self.states[episode_id]
            if state["instruction"] is None or state["latest_history"] is None:
                raise RuntimeError("navigation arrived before causal features")
            global_current = {}
            for index, branch_id in enumerate(gmap_vp_ids[environment_index]):
                if (
                    index == 0 or branch_id is None
                    or not bool(gmap_masks[environment_index, index])
                    or bool(gmap_visited_masks[environment_index, index])
                ):
                    continue
                global_current[str(branch_id)] = (
                    gmap_img_fts[environment_index, index].detach()
                )
            local_ids = set(global_current).intersection(
                self.local_frontiers[environment_index]
            )
            current = {
                branch: global_current[branch] for branch in sorted(local_ids)
            }
            state["global_rows"].append((
                state["latest_history"].detach(), dict(global_current)
            ))
            ids = gmap_vp_ids[environment_index]
            native_index = int(native_indices[environment_index])
            native = (
                str(ids[native_index])
                if 0 < native_index < len(ids) and ids[native_index] is not None
                else None
            )
            controls = tuple(current)
            if 2 <= len(controls) <= 4 and native in current:
                rows = [
                    (history, {
                        branch: features[branch]
                        for branch in controls if branch in features
                    })
                    for history, features in state["global_rows"]
                ]
                self.capture(
                    episode_id, state, controls, native,
                    self.trainer.gmaps[environment_index],
                    self.current_ids[environment_index], rows,
                )
            state["step"] += 1


def install_hooks(collector: BatchFeatureCollector) -> None:
    from vlnce_baselines.models.R1Policy import ETP
    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer

    original_rollout = RLTrainer.rollout

    def rollout_wrapped(self, *args, **kwargs):
        collector.trainer = self
        return original_rollout(self, *args, **kwargs)

    RLTrainer.rollout = rollout_wrapped
    original_gmap = RLTrainer._nav_gmap_variable

    def gmap_wrapped(self, cur_vp, cur_pos, cur_ori, task_type):
        result = original_gmap(self, cur_vp, cur_pos, cur_ori, task_type)
        collector.publish_graph(self, cur_vp)
        return result

    RLTrainer._nav_gmap_variable = gmap_wrapped
    original_forward = ETP.forward

    def forward_wrapped(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        mode = kwargs.get("mode", args[0] if args else None)
        if mode == "language":
            collector.record_language(result, kwargs["txt_masks"])
        elif mode == "panorama":
            collector.record_panorama(result[0], result[1])
        elif mode == "navigation":
            collector.record_navigation(
                kwargs["gmap_vp_ids"], kwargs["gmap_img_fts"],
                kwargs["gmap_masks"], kwargs["gmap_visited_masks"],
                result["global_logits"],
            )
        return result

    ETP.forward = forward_wrapped


def write_summaries(
    collector: BatchFeatureCollector, batch_id: str, wall_time_s: float,
) -> None:
    for episode_id, state in collector.states.items():
        run_dir = collector.runs_root / f"ep_{episode_id}"
        summary = {
            "schema_version": "revealnav-r2r-train-net-advantage-batch-worker/1",
            "status": "PASS",
            **collector.metadata[episode_id],
            "seed": 20260826,
            "split": "train",
            "mode": "causal_shadow_collection",
            "batch_id": batch_id,
            "batch_size": len(collector.states),
            "task_metric_payload_read": False,
            "ground_truth_payload_read": False,
            "native_action_overridden": False,
            "wall_time_s": wall_time_s,
            "peak_rss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_children_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "feature_event_count": len(state["events"]),
            "feature_events": state["events"],
            "paper_result": False,
            "unseen_or_test_read": False,
        }
        single.atomic_json(run_dir / "RUN_SUMMARY.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-ids", required=True)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--batch-dir", required=True, type=Path)
    args = parser.parse_args()
    episode_ids = tuple(value for value in args.episode_ids.split(",") if value)
    if not episode_ids or len(episode_ids) != len(set(episode_ids)):
        raise SystemExit("episode ids must be non-empty and unique")
    runs_root = args.runs_root.resolve()
    batch_dir = args.batch_dir.resolve()
    if ROOT not in runs_root.parents or ROOT not in batch_dir.parents:
        raise SystemExit("batch paths must remain inside the project")
    if batch_dir.exists():
        raise SystemExit("batch directory must be new")
    batch_dir.mkdir(parents=True)
    metadata = {
        episode_id: single.episode_metadata(episode_id)
        for episode_id in episode_ids
    }
    for episode_id in episode_ids:
        if (runs_root / f"ep_{episode_id}").exists():
            raise SystemExit("episode run directory must be absent")
        (runs_root / f"ep_{episode_id}").mkdir(parents=True)
    collector = BatchFeatureCollector(metadata, runs_root)
    install_hooks(collector)

    os.chdir(single.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = batch_dir / "etp_output"
    batch_id = batch_dir.name
    episode_literal = "[" + ",".join(repr(value) for value in episode_ids) + "]"
    count = len(episode_ids)
    argv = [
        "run.py", "--exp_name", f"train_net_advantage_{batch_id}",
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", "train", "TASK_CONFIG.DATASET.SPLIT", "train",
        "EVAL.EPISODE_ID", episode_literal, "EVAL.EPISODE_COUNT", str(count),
        "EVAL.CKPT_PATH_DIR", str(single.R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(single.JOINT_PRETRAINED), "IL.back_algo", "control",
        "INFERENCE.SPLIT", "train", "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", str(count),
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]", "TORCH_GPU_ID", "0",
        "VIDEO_OPTION", "[]", "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]
    sys.argv = argv
    import run

    started = time.monotonic()
    run.main()
    wall_time_s = round(time.monotonic() - started, 3)
    missing = [
        episode_id for episode_id, state in collector.states.items()
        if state["instruction"] is None or state["step"] < 1
    ]
    if missing:
        raise RuntimeError(f"batch episodes were not evaluated: {missing}")
    write_summaries(collector, batch_id, wall_time_s)
    single.atomic_json(batch_dir / "BATCH_SUMMARY.json", {
        "schema_version": "revealnav-r2r-train-net-advantage-batch/1",
        "status": "PASS",
        "batch_id": batch_id,
        "episode_ids": list(episode_ids),
        "episodes": count,
        "feature_events": sum(
            len(state["events"]) for state in collector.states.values()
        ),
        "wall_time_s": wall_time_s,
        "task_metric_payload_read": False,
        "unseen_or_test_read": False,
    })
    print(json.dumps({
        "status": "PASS", "batch_id": batch_id,
        "episodes": count,
        "events": sum(len(state["events"]) for state in collector.states.values()),
        "wall_time_s": wall_time_s,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
