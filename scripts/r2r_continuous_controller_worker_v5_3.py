#!/usr/bin/env python3
"""Run one R2R episode with the thresholded, persistent V5.3 overlay."""

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
import r2r_continuous_controller_worker_v5_2 as v52  # noqa: E402
from revealnav_mf2r4 import (  # noqa: E402
    BranchMacroAction, PersistentExcursionLedger, ReeQFusionController,
)
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims, sha256_file,
)


OPV_THRESHOLD = 0.025
install_runtime_shims()


class RecordingFusionController(ReeQFusionController):
    def __init__(self) -> None:
        super().__init__(3, 5.0, OPV_THRESHOLD)
        self.last_decision = None

    def decide(self, *args, **kwargs):
        self.last_decision = super().decide(*args, **kwargs)
        return self.last_decision


class PersistentContinuousController(v52.ContinuousController):
    """V5.2 motion integration with the frozen OPV gate and ECOG ledger."""

    def __init__(self, seed: int, device: torch.device, trace_path: Path) -> None:
        super().__init__(seed, device, trace_path)
        self.fusion = RecordingFusionController()
        self.ledger = PersistentExcursionLedger(OPV_THRESHOLD)
        self.threshold_suppressions = 0
        self.ledger_suppressions = 0

    def _features(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ):
        current, persistent = super()._features(
            gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks
        )
        if len(pilot._CURRENT_IDS) != 1:
            raise RuntimeError("checkpoint identity hook is unavailable")
        checkpoint_id = pilot._CURRENT_IDS[0]
        available = self.ledger.untried(checkpoint_id, persistent)
        self.ledger_suppressions += len(persistent) - len(available)
        return current, available

    def _initial_decision(self, current, persistent):
        self.fusion.last_decision = None
        selected = super()._initial_decision(current, persistent)
        decision = self.fusion.last_decision
        if selected is None:
            if (
                decision is not None
                and decision.reason
                == "checkpoint_value_not_above_frozen_opv_threshold"
            ):
                self.threshold_suppressions += 1
            return None
        if (
            decision is None
            or decision.action is not BranchMacroAction.CHECKPOINTED_EXCURSION
            or decision.branch_id != selected
            or not self.ledger.authorize(self.checkpoint_id, decision)
        ):
            raise RuntimeError("ECOG ledger rejected an executed excursion")
        self.record(
            "ecog_authorized", checkpoint_id=self.checkpoint_id,
            branch_id=selected,
            preservation_gain=round(float(decision.preservation_gain), 8),
            opv_threshold=OPV_THRESHOLD,
        )
        return selected

    def _post_decision(self, current) -> None:
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        super()._post_decision(current)
        if self.post_policy_action == "continue":
            self.ledger.resolve_continue(checkpoint_id, branch_id)

    def complete_pending_return(self) -> None:
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        super().complete_pending_return()
        if self.return_intervention_success:
            self.ledger.resolve_return(checkpoint_id, branch_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--mode", choices=("baseline", "revealnav"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if (args.mode == "revealnav") != (args.seed is not None):
        raise SystemExit("revealnav requires a seed; baseline forbids one")
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside project")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    state = None
    if args.mode == "revealnav":
        state = PersistentContinuousController(
            args.seed, torch.device("cuda:0"), controller_trace
        )
        v52._CONTROLLER = state
        v52.install_continuous_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    name = f"continuous_v5_3_{args.mode}_{args.seed}_{args.episode_id}"
    argv = [
        "run.py", "--exp_name", name,
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", args.split, "TASK_CONFIG.DATASET.SPLIT", args.split,
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED), "IL.back_algo", "control",
        "INFERENCE.SPLIT", args.split, "TASK_CONFIG.DATASET.SUFFIX", "''",
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
        "schema_version": "revealnav-r2r-continuous-controller-worker/5.3",
        "status": "RUNNING", "episode_id": args.episode_id,
        "seed": args.seed, "mode": args.mode, "split": args.split,
        "opv_threshold": OPV_THRESHOLD, "argv": argv,
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
        summary["wall_time_s"] = round(time.monotonic() - started, 3)
        summary["peak_rss_self_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary["peak_rss_children_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        summary["controller"] = None if state is None else {
            "strict_load": True,
            "checkpointed_excursions": state.checkpointed_excursions,
            "continue_decisions": state.continue_decisions,
            "backtrack_decisions": state.backtrack_decisions,
            "successful_returns": state.successful_returns,
            "failed_returns": state.failed_returns,
            "threshold_suppressions": state.threshold_suppressions,
            "ledger_suppressions": state.ledger_suppressions,
            "ledger_counts": state.ledger.counts(),
            "final_record_hash": state.previous_hash,
            "checkpoint_triplet": {
                "ree": state.pair["ree"], "q": state.pair["q"],
                "post": state.post_row,
            },
        }
        summary["base_trace_sha256"] = sha256_file(base_trace)
        summary["controller_trace_sha256"] = (
            sha256_file(controller_trace) if controller_trace.is_file() else None
        )
        stats = list(output.rglob(f"stats_ep_ckpt_270_{args.split}_r0_w1.json"))
        summary["metrics"] = None
        if len(stats) == 1:
            payload = json.loads(stats[0].read_text())
            summary["metrics"] = payload.get(str(args.episode_id))
            summary["metrics_path"] = str(stats[0].relative_to(ROOT))
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
