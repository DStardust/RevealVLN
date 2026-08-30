#!/usr/bin/env python3
"""Collect or execute one RxR-train V6 post-native counterfactual."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ETPR1 = ROOT / "third_party/ETP-R1"
DATASET = ETPR1 / (
    "data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
RXR_CHECKPOINT = ETPR1 / (
    "data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth"
)
JOINT_PRETRAINED = ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt"
)
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_native_first_deferred_switch_worker_v5_16 as v516  # noqa: E402
import r2r_native_preservation_worker_v5_5 as v55  # noqa: E402
from revealnav_mf2r3 import OptionStatus  # noqa: E402
from rxr_primary_controller_worker_v5_22 import install_expanded_q  # noqa: E402
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims,
    sha256_file,
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


def stable_array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def scene_id(scene_path: str) -> str:
    parts = Path(scene_path).parts
    if len(parts) < 2 or parts[-1] != f"{parts[-2]}.glb":
        raise RuntimeError("unexpected RxR scene path")
    return parts[-2]


def episode_metadata(episode_id: str) -> dict:
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    rows = [row for row in episodes if str(row["episode_id"]) == episode_id]
    if len(rows) != 1:
        raise RuntimeError("RxR train episode identity is not unique")
    row = rows[0]
    language = row.get("instruction", {}).get("language")
    if language not in ("en-US", "en-IN"):
        raise RuntimeError("V6 primary collector accepts English RxR only")
    return {
        "episode_id": episode_id,
        "trajectory_id": str(row.get("trajectory_id")),
        "scene_id": scene_id(row["scene_id"]),
        "language": language,
    }


class V6CounterfactualController(v516.NativeFirstDeferredSwitchController):
    """Observe all trials; force exactly one return only in macro mode."""

    def __init__(
        self, seed: int, trace_path: Path, event_dir: Path, metadata: dict,
        mode: str, target: dict | None,
    ) -> None:
        super().__init__(seed, "revealnav", torch.device("cuda:0"), trace_path)
        if mode not in ("shadow", "macro"):
            raise ValueError("unknown V6 counterfactual mode")
        if (mode == "macro") != (target is not None):
            raise ValueError("macro mode requires one target; shadow forbids it")
        self.v6_mode = mode
        self.target = target
        self.event_dir = event_dir
        self.metadata = metadata
        self.candidate_events: list[dict] = []
        self.target_reached = False
        self.target_return_scheduled = False
        self.target_alternative_committed = False

    def _return_path_length(self) -> float | None:
        graph = pilot._TRAINER.gmaps[0]
        current = pilot._CURRENT_IDS[0]
        if (
            current not in graph.shortest_path
            or self.checkpoint_id not in graph.shortest_path[current]
        ):
            return None
        nodes = graph.shortest_path[current][self.checkpoint_id]
        points = [np.asarray(graph.node_pos[node], dtype=float) for node in nodes]
        value = sum(
            float(np.linalg.norm(right - left))
            for left, right in zip(points, points[1:])
        )
        return value if math.isfinite(value) else None

    def _causal_arrays(self, current: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        alternative = self.retained_alternative
        if alternative is None or alternative not in self.checkpoint_candidates:
            raise RuntimeError("V6 trial lacks its retained alternative embedding")
        history = torch.stack(
            [*self.pre_histories, self.latest_history.detach()]
        )
        local = (
            torch.stack(list(current.values())).mean(0)
            if current else torch.zeros(768, device=self.device)
        )
        max_len = float(pilot._TRAINER.max_len)
        return_distance = self._return_path_length()
        if return_distance is None:
            raise RuntimeError("V6 candidate has no executable online return path")
        return {
            "instruction": self.instruction.detach().cpu().half().numpy(),
            "post_observation": local.detach().cpu().half().numpy(),
            "temporal_history": history.mean(0).detach().cpu().half().numpy(),
            "checkpoint": self.checkpoint_embedding.detach().cpu().half().numpy(),
            "native": self.selected_embedding.detach().cpu().half().numpy(),
            "alternative": self.checkpoint_candidates[
                alternative
            ].detach().cpu().half().numpy(),
            "scalars": np.asarray([
                self.step / max_len,
                max(0.0, max_len - self.step) / max_len,
                return_distance / 10.0,
            ], dtype=np.float32),
        }

    def _event(self, current: dict[str, torch.Tensor]) -> dict:
        arrays = self._causal_arrays(current)
        index = len(self.candidate_events)
        event_id = (
            f"rxr_ep{self.metadata['episode_id']}_seed{self.seed}_"
            f"post{self.step:03d}_e{index:02d}"
        )
        feature_path = self.event_dir / f"{event_id}.npz"
        if self.v6_mode == "shadow":
            atomic_npz(feature_path, arrays)
        value = {
            "event_id": event_id,
            "event_index": index,
            **self.metadata,
            "controller_seed": self.seed,
            "post_navigation_step": self.step,
            "prefix_action_count": self.step,
            "checkpoint_id": self.checkpoint_id,
            "native_branch_id": self.selected_branch,
            "alternative_branch_id": self.retained_alternative,
            "alternative_source": self.retained_alternative_source,
            "candidate_branch_ids": sorted(self.checkpoint_candidates),
            "causal_state_sha256": stable_array_hash(arrays),
            "causal_prefix_only": True,
            "online_return_path_length_m": round(
                float(arrays["scalars"][2]) * 10.0, 6
            ),
            "feature_path": (
                str(feature_path.relative_to(ROOT))
                if self.v6_mode == "shadow" else None
            ),
            "feature_bytes": (
                feature_path.stat().st_size
                if self.v6_mode == "shadow" else None
            ),
            "feature_sha256": (
                sha256_file(feature_path)
                if self.v6_mode == "shadow" else None
            ),
        }
        self.candidate_events.append(value)
        return value

    def _matches_target(self, event: dict) -> bool:
        if self.target is None:
            return False
        keys = (
            "event_index", "checkpoint_id", "native_branch_id",
            "alternative_branch_id", "causal_state_sha256",
        )
        return all(event.get(key) == self.target.get(key) for key in keys)

    def _continue_native(self, event: dict) -> None:
        self.post_policy_action = "continue"
        self.record(
            "v6_relative_advantage_shadow",
            event_index=event["event_index"],
            event_id=event["event_id"],
            causal_state_sha256=event["causal_state_sha256"],
            native_branch=event["native_branch_id"],
            alternative_branch=event["alternative_branch_id"],
            executed_return=False,
        )
        self.continue_decisions += 1
        self.executor.continue_excursion()
        if self.ledger.status(
            self.checkpoint_id, self.selected_branch
        ) is OptionStatus.ACTIVE:
            self.ledger.resolve_continue(self.checkpoint_id, self.selected_branch)
        self.retained_alternative = None
        self.retained_alternative_source = None
        self.trial_preservation_gain = None
        self.checkpoint_graph_snapshot = None
        self.checkpoint_graph_signature = None
        self._reset_search()

    def _post_decision(self, current) -> None:
        event = self._event(current)
        target_index = None if self.target is None else self.target["event_index"]
        if self.v6_mode == "macro" and event["event_index"] == target_index:
            if not self._matches_target(event):
                raise RuntimeError("V6 replay target state or identity drift")
            self.target_reached = True
            self.post_policy_action = "backtrack"
            self.record(
                "v6_relative_advantage_intervention",
                event_index=event["event_index"],
                event_id=event["event_id"],
                causal_state_sha256=event["causal_state_sha256"],
                native_branch=event["native_branch_id"],
                alternative_branch=event["alternative_branch_id"],
                executed_return=True,
                gate_mode="forced_train_counterfactual_only",
            )
            self.backtrack_decisions += 1
            self.target_return_scheduled = self._schedule_return()
            if not self.target_return_scheduled:
                raise RuntimeError("V6 target return could not be scheduled")
            return
        if (
            self.v6_mode == "macro"
            and target_index is not None
            and event["event_index"] > target_index
            and not self.target_reached
        ):
            raise RuntimeError("V6 replay passed its target without matching")
        self._continue_native(event)

    def _consume_pending_alternative(self):
        branch, consumed = super()._consume_pending_alternative()
        if consumed and self.target_reached and branch is not None:
            self.target_alternative_committed = True
        return branch, consumed


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("shadow", "macro"), required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.seed not in (20260826, 20260827, 20260828):
        raise SystemExit("seed is outside the locked V6 set")
    if (args.mode == "macro") != (args.target is not None):
        raise SystemExit("macro requires --target; shadow forbids it")
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside project")
    target = None
    if args.target is not None:
        target_path = args.target.resolve()
        if ROOT not in target_path.parents or target_path.is_symlink():
            raise SystemExit("unsafe V6 target path")
        target = json.loads(target_path.read_text())
    metadata = episode_metadata(str(args.episode_id))
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)

    state = V6CounterfactualController(
        args.seed, controller_trace, run_dir / "events", metadata,
        args.mode, target,
    )
    q_checkpoint = install_expanded_q(state, args.seed)
    v55._CONTROLLER = state
    v516.v512._install_v5_12_hooks()
    v55.install_native_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    name = f"rxr_v6_{args.mode}_{args.seed}_{args.episode_id}"
    argv = [
        "run.py", "--exp_name", name,
        "--run-type", "eval", "--exp-config", "run_rxr/iter_train.yaml",
        "EVAL.SPLIT", "train", "TASK_CONFIG.DATASET.SPLIT", "train",
        "EVAL.LANGUAGES", "['en-US','en-IN']",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']",
        "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(RXR_CHECKPOINT),
        "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED),
        "IL.back_algo", "control",
        "IL.RECOLLECT_TRAINER.gt_file",
        "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        "INFERENCE.SPLIT", "train", "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]",
        "TORCH_GPU_ID", "0", "VIDEO_OPTION", "[]",
        "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]
    summary = {
        "schema_version": "revealnav-rxr-v6-counterfactual-worker/1",
        "status": "RUNNING", **metadata, "seed": args.seed,
        "mode": args.mode, "split": "train",
        "target": target,
        "expanded_q_checkpoint": q_checkpoint,
        "task_metrics_used_for_training_target_only": True,
        "future_information_used_for_online_input": False,
        "unseen_or_test_read": False,
        "argv": argv,
    }
    sys.argv = argv
    import run

    started = time.monotonic()
    try:
        run.main()
        state.finalize_episode()
        if args.mode == "macro":
            if not (state.target_reached and state.target_return_scheduled):
                raise RuntimeError("V6 macro transaction did not reach its target")
            if not state.target_alternative_committed:
                summary["status"] = "REJECTED_UNEXECUTABLE_MACRO"
                summary["rejection_reason"] = (
                    "return_or_alternative_commit_not_executable"
                )
            else:
                summary["status"] = "PASS"
        else:
            summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        state.finalize_episode()
        pilot.close_envs()
        summary.update({
            "wall_time_s": round(time.monotonic() - started, 3),
            "peak_rss_self_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "peak_rss_children_kib": resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_maxrss,
            "candidate_event_count": len(state.candidate_events),
            "candidate_events": state.candidate_events,
            "target_reached": state.target_reached,
            "target_return_scheduled": state.target_return_scheduled,
            "target_alternative_committed": state.target_alternative_committed,
            "base_trace_sha256": sha256_file(base_trace),
            "controller_trace_sha256": sha256_file(controller_trace),
            "paper_result": False,
        })
        stats = list(output.rglob("stats_ep_ckpt_1320_train_r0_w1.json"))
        summary["metrics"] = None
        if len(stats) == 1:
            payload = json.loads(stats[0].read_text())
            summary["metrics"] = payload.get(str(args.episode_id))
            summary["metrics_path"] = str(stats[0].relative_to(ROOT))
        atomic_json(run_dir / "RUN_SUMMARY.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "episode_id": args.episode_id,
        "mode": args.mode,
        "candidate_events": summary["candidate_event_count"],
        "target_committed": summary["target_alternative_committed"],
        "wall_time_s": summary["wall_time_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
