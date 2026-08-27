#!/usr/bin/env python3
"""Collect V5.6 policy-induced proposal pairs on R2R train only."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ETPR1 = ROOT / "third_party/ETP-R1"
HABITAT_LAB = ROOT / "third_party/habitat-lab"
HABITAT_SIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / ".remote_runtime/habitat-sim"
)).resolve()
for path in reversed((ROOT, ROOT / "scripts", ETPR1, HABITAT_LAB, HABITAT_SIM)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_full_opp_worker_v5_6 as v56  # noqa: E402
import r2r_native_preservation_worker_v5_5 as v55  # noqa: E402
import r2r_train_net_advantage_worker as base  # noqa: E402
from rxr_unseen_controller_worker import install_runtime_shims  # noqa: E402


install_runtime_shims()


class V56PolicyProposalCollector(v56.FullOPPActionController):
    """Observe scoreable V5.6 proposals while preserving the ETP trajectory."""

    def __init__(
        self, seed: int, trace_path: Path, event_dir: Path, metadata: dict,
    ) -> None:
        super().__init__(seed, "shadow", torch.device("cuda:0"), trace_path)
        self.event_dir = event_dir
        self.metadata = metadata
        self.feature_events: list[dict] = []
        self.missing_causal_inputs = 0
        self._proposal_current: dict[str, torch.Tensor] = {}
        self._proposal_persistent: tuple[str, ...] = ()

    def _evaluate(self, current, persistent):
        value = super()._evaluate(current, persistent)
        self._proposal_current = dict(current)
        self._proposal_persistent = tuple(persistent)
        return value

    def record(self, event: str, **values) -> None:
        super().record(event, **values)
        if event != "opp_initial_decision":
            return
        action = values["opp_action"]
        proposed = values["selected_branch"]
        native = values["native_base_branch"]
        if action not in ("commit", "explore") or proposed == native:
            return
        if (
            native not in self._proposal_current
            or proposed not in self._proposal_current
            or proposed not in self._proposal_persistent
        ):
            self.missing_causal_inputs += 1
            return
        self._capture(native, proposed, action)

    def _capture(self, native: str, proposed: str, action: str) -> None:
        graph = pilot._TRAINER.gmaps[0]
        checkpoint_id = pilot._CURRENT_IDS[0]
        controls = (native, proposed)
        if (
            checkpoint_id not in graph.node_pos
            or any(branch not in graph.ghost_aug_pos for branch in controls)
        ):
            self.missing_causal_inputs += 1
            return
        steps = len(self.rows)
        if steps < 1:
            raise RuntimeError("V5.6 proposal lacks causal history")
        history = torch.stack([row[0] for row in self.rows]).detach().cpu().float()
        candidates = torch.zeros(steps, 2, 768, dtype=torch.float32)
        mask = torch.zeros(steps, 2, dtype=torch.bool)
        for time_index, (_, features) in enumerate(self.rows):
            for branch_index, branch in enumerate(controls):
                if branch in features:
                    candidates[time_index, branch_index] = (
                        features[branch].detach().cpu().float()
                    )
                    mask[time_index, branch_index] = True
        if not bool(mask[-1].all()):
            raise RuntimeError("current policy proposal embeddings are incomplete")
        event_id = (
            f"r2r_ep{self.metadata['episode_id']}_seed{self.seed}_"
            f"s{self.step:02d}_p{len(self.feature_events):02d}"
        )
        feature_path = self.event_dir / f"{event_id}.npz"
        base.atomic_npz(feature_path, {
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
            "controller_seed": self.seed,
            "navigation_step": self.step,
            "proposal_action": action,
            "proposed_branch_id": proposed,
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
            "feature_sha256": base.sha256_file(feature_path),
            "causal_prefix_only": True,
            "offline_target_truth_read": False,
            "policy_induced": True,
        }
        self.feature_events.append(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.seed not in (20260826, 20260827, 20260828):
        raise SystemExit("seed is outside the locked V5.6 triplet")
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside project")
    metadata = base.episode_metadata(str(args.episode_id))
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    state = V56PolicyProposalCollector(
        args.seed, controller_trace, run_dir / "events", metadata
    )
    v55._CONTROLLER = state
    v55.install_native_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name",
        f"v5_15_policy_proposal_{args.seed}_{args.episode_id}",
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", "train", "TASK_CONFIG.DATASET.SPLIT", "train",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(base.R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(base.JOINT_PRETRAINED), "IL.back_algo", "control",
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
        "schema_version": "revealnav-r2r-v5.15-policy-proposal-worker/1",
        "status": "RUNNING", **metadata, "controller_seed": args.seed,
        "split": "train", "source_policy": "V5.6 shadow proposals",
        "task_metric_payload_read": False, "ground_truth_payload_read": False,
        "native_action_overridden": False, "unseen_or_test_read": False,
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
            "missing_causal_inputs": state.missing_causal_inputs,
            "opp_commit_decisions": state.commit_decisions,
            "opp_explore_decisions": state.explore_decisions,
            "base_trace_sha256": base.sha256_file(base_trace),
            "controller_trace_sha256": base.sha256_file(controller_trace),
            "paper_result": False,
        })
        base.atomic_json(run_dir / "RUN_SUMMARY.json", summary)
    print(json.dumps({
        "status": summary["status"], "episode_id": metadata["episode_id"],
        "seed": args.seed, "proposal_events": summary["feature_event_count"],
        "missing_causal_inputs": summary["missing_causal_inputs"],
        "wall_time_s": summary["wall_time_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
