#!/usr/bin/env python3
"""Screen one R2R val_seen episode for V5.3 activation without acting."""

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
from revealnav_mf2r4 import BranchMacroAction, ReeQFusionController  # noqa: E402
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims, sha256_file,
)


OPV_THRESHOLD = 0.025
SCREEN_SEED = 20260826
install_runtime_shims()


class RecordingFusionController(ReeQFusionController):
    def __init__(self) -> None:
        super().__init__(3, 5.0, OPV_THRESHOLD)
        self.last_decision = None

    def decide(self, *args, **kwargs):
        self.last_decision = super().decide(*args, **kwargs)
        return self.last_decision


class ActivationShadowController(pilot.ActionEnabledPilotController):
    """Observe locked controller proposals while ETP-R1 keeps every action."""

    def __init__(self, device: torch.device, trace_path: Path) -> None:
        super().__init__(SCREEN_SEED, "natural", device, trace_path)
        self.fusion = RecordingFusionController()
        self.activations = 0
        self.threshold_suppressions = 0
        self.maximum_gain = None

    def _reset_proposal(self) -> None:
        self.phase = "seeking_excursion"
        self.rows = []
        self.pre_histories = None
        self.selected_embedding = None
        self.checkpoint_embedding = None
        self.checkpoint_id = None
        self.checkpoint_position = None
        self.selected_branch = None
        self.executor = None

    def record_navigation(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ) -> None:
        if self.instruction is None or self.latest_history is None:
            return
        current, persistent = self._features(
            gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks
        )
        self.fusion.last_decision = None
        selected = self._initial_decision(current, persistent)
        decision = self.fusion.last_decision
        if selected is not None:
            if (
                decision is None
                or decision.action is not BranchMacroAction.CHECKPOINTED_EXCURSION
                or decision.preservation_gain is None
                or decision.preservation_gain <= OPV_THRESHOLD
            ):
                raise RuntimeError("shadow activation violates frozen OPV gate")
            self.activations += 1
            self.maximum_gain = max(
                decision.preservation_gain,
                self.maximum_gain if self.maximum_gain is not None else float("-inf"),
            )
            self.record(
                "shadow_activation", checkpoint_id=self.checkpoint_id,
                branch_id=selected,
                preservation_gain=round(float(decision.preservation_gain), 8),
                opv_threshold=OPV_THRESHOLD,
                shadow_only_not_executed=True,
            )
            self._reset_proposal()
        elif (
            decision is not None
            and decision.reason == "checkpoint_value_not_above_frozen_opv_threshold"
        ):
            self.threshold_suppressions += 1
        self.step += 1


_CONTROLLER = None


def controller():
    return _CONTROLLER


def install_shadow_hooks() -> None:
    from vlnce_baselines.models.R1Policy import ETP
    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer

    original_gmap = RLTrainer._nav_gmap_variable

    def gmap_wrapped(self, cur_vp, cur_pos, cur_ori, task_type):
        result = original_gmap(self, cur_vp, cur_pos, cur_ori, task_type)
        pilot._TRAINER = self
        pilot._CURRENT_IDS = tuple(cur_vp)
        pilot._LOCAL_FRONTIERS = tuple(
            {
                ghost_id: int(graph.ghost_embeds[ghost_id][1])
                for ghost_id, fronts in graph.ghost_fronts.items()
                if current_id in fronts
            }
            for graph, current_id in zip(self.gmaps, cur_vp)
        )
        return result

    RLTrainer._nav_gmap_variable = gmap_wrapped
    original_forward = ETP.forward

    def forward_wrapped(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        state = controller()
        mode = kwargs.get("mode", args[0] if args else None)
        if mode == "language":
            state.record_language(result, kwargs["txt_masks"])
        elif mode == "panorama":
            state.record_panorama(result[0], result[1])
        elif mode == "navigation":
            state.record_navigation(
                kwargs["gmap_vp_ids"], kwargs["gmap_img_fts"],
                kwargs["gmap_masks"], kwargs["gmap_visited_masks"],
            )
        return result

    ETP.forward = forward_wrapped


def main() -> None:
    global _CONTROLLER
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside project")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "activation_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    _CONTROLLER = ActivationShadowController(
        torch.device("cuda:0"), controller_trace
    )
    install_shadow_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", f"v5_3_activation_{args.episode_id}",
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", "val_seen", "TASK_CONFIG.DATASET.SPLIT", "val_seen",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED), "IL.back_algo", "control",
        "INFERENCE.SPLIT", "val_seen", "TASK_CONFIG.DATASET.SUFFIX", "''",
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
        "schema_version": "revealnav-r2r-v5.3-activation-shadow-worker/1",
        "status": "RUNNING", "episode_id": args.episode_id,
        "split": "val_seen", "screen_seed": SCREEN_SEED,
        "opv_threshold": OPV_THRESHOLD,
        "shadow_actions_executed": 0,
        "task_metric_payload_read": False,
        "argv": argv,
    }
    sys.argv = argv
    import run

    started = time.monotonic()
    try:
        run.main()
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        state = controller()
        summary.update({
            "wall_time_s": round(time.monotonic() - started, 3),
            "peak_rss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_children_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "controller": {
                "strict_load": True, "activation_count": state.activations,
                "threshold_suppressions": state.threshold_suppressions,
                "maximum_preservation_gain": state.maximum_gain,
                "final_record_hash": state.previous_hash,
            },
            "base_trace_sha256": sha256_file(base_trace),
            "activation_trace_sha256": sha256_file(controller_trace),
        })
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
