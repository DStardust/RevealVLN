#!/usr/bin/env python3
"""Run one R2R episode with the complete frozen OPP action order."""

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
import r2r_continuous_controller_worker_v5_4 as v54  # noqa: E402
import r2r_native_preservation_worker_v5_5 as v55  # noqa: E402
from revealnav_mf2r4 import (  # noqa: E402
    BranchMacroAction, StateConditionedReturnExecutor,
)
from revealnav_mf2r3 import OptionStatus  # noqa: E402
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims, sha256_file,
)


install_runtime_shims()


class FullOPPActionController(v54.FullOPPContinuousController):
    def __init__(
        self, seed: int, mode: str, device: torch.device, trace_path: Path,
    ) -> None:
        super().__init__(seed, mode, device, trace_path)
        self.commit_decisions = 0
        self.effective_commit_interventions = 0
        self.explore_decisions = 0
        self.inspect_delegations = 0
        self.follow_delegations = 0
        self.unresolved_decisions = 0

    def _evaluate(self, current, persistent):
        self.rows.append((self.latest_history, current))
        ordered = tuple(dict.fromkeys(
            branch_id for _, values in self.rows for branch_id in values
        ))
        if len(persistent) < 2:
            return None
        index = {branch_id: value for value, branch_id in enumerate(ordered)}
        steps = len(self.rows)
        history = torch.stack([row[0] for row in self.rows]).unsqueeze(0)
        candidates = torch.zeros(
            1, steps, len(ordered), 768, device=self.device
        )
        mask = torch.zeros(
            1, steps, len(ordered), dtype=torch.bool, device=self.device
        )
        for time_index, (_, values) in enumerate(self.rows):
            for branch_id, embedding in values.items():
                candidates[0, time_index, index[branch_id]] = embedding
                mask[0, time_index, index[branch_id]] = True
        with torch.no_grad():
            q = self.q_model(
                history, candidates, mask, self.instruction.unsqueeze(0),
                torch.tensor([steps - 1], device=self.device),
            )
            budgets = torch.tensor(
                [1.5, 2.0, 3.0, 4.0], device=self.device
            ).view(1, 1, 4).expand(1, steps, 4)
            ree = self.ree_model(
                history, candidates, mask, budgets,
                self.instruction.unsqueeze(0),
            )
        indices = [index[branch_id] for branch_id in persistent]
        probabilities = torch.softmax(
            ree.target_logits[0, steps - 1, indices], dim=-1
        )
        commit = [float(q.commit_cost[0, value]) for value in indices]
        excursion = [float(q.excursion_cost[0, value]) for value in indices]
        macro = self.fusion.decide(
            persistent, probabilities.cpu().tolist(), commit, excursion,
            v54.FROZEN_CONFIG["persistence_k"],
        )
        fused_commit = [
            cost + v54.FROZEN_CONFIG["wrong_commitment_weight"] * (1.0 - prob)
            for cost, prob in zip(commit, probabilities.cpu().tolist())
        ]
        commit_cost, commit_branch = min(zip(fused_commit, persistent))
        belief = self._event_belief(ree, steps - 1, indices)
        action, reason = self.event_gate.initial_action(
            belief["p_discriminable"], belief["evidence"],
            belief["maximum_target_probability"], belief["reveal_hazard"],
            belief["expiry_hazard"],
            macro.action is BranchMacroAction.CHECKPOINTED_EXCURSION,
        )
        return {
            "action": action, "reason": reason, "macro": macro,
            "belief": belief, "probabilities": probabilities,
            "commit_branch": commit_branch, "commit_cost": commit_cost,
        }

    def _initial_decision(self, current, persistent, native_branch):
        value = self._evaluate(current, persistent)
        if value is None:
            return None
        action = value["action"]
        macro = value["macro"]
        selected = (
            value["commit_branch"] if action == "commit"
            else macro.branch_id if action == "explore" else None
        )
        self.record(
            "opp_initial_decision", opp_action=action,
            opp_reason=value["reason"], selected_branch=selected,
            native_base_branch=native_branch,
            action_differs_from_base=(
                selected is not None and selected != native_branch
            ),
            macro_action=macro.action.value, macro_branch=macro.branch_id,
            preservation_gain=(
                None if macro.preservation_gain is None
                else round(float(macro.preservation_gain), 8)
            ),
            **{
                key: round(number, 8)
                for key, number in value["belief"].items()
            },
        )
        if action == "inspect":
            self.inspect_delegations += 1
            return None
        if action == "follow":
            self.follow_delegations += 1
            return None
        if action == "unresolved":
            self.unresolved_decisions += 1
            return None
        if action == "commit":
            self.commit_decisions += 1
            self.effective_commit_interventions += int(selected != native_branch)
            self._reset_search()
            if self.mode == "shadow":
                return None
            self.record(
                "commit_executed", branch_id=selected,
                fused_commit_cost=round(float(value["commit_cost"]), 8),
                base_action_overridden=selected != native_branch,
            )
            return selected
        if action != "explore":
            raise RuntimeError("unknown frozen OPP action")
        if (
            macro.action is not BranchMacroAction.CHECKPOINTED_EXCURSION
            or macro.preservation_gain is None
            or macro.preservation_gain <= v54.FROZEN_CONFIG["opv_threshold"]
        ):
            raise RuntimeError("OPP EXPLORE lacks a valid OPV proposal")
        self.explore_decisions += 1
        if self.mode == "shadow":
            self._reset_search()
            return None
        checkpoint_id = pilot._CURRENT_IDS[0]
        if not self.ledger.authorize(checkpoint_id, macro):
            raise RuntimeError("ECOG ledger rejected OPP EXPLORE")
        self.phase = "outbound_in_flight"
        self.pre_histories = [row[0].detach() for row in self.rows]
        self.selected_embedding = current[selected].detach()
        self.checkpoint_embedding = self.latest_history.detach()
        self.selected_branch = selected
        self.checkpoint_id = checkpoint_id
        graph = pilot._TRAINER.gmaps[0]
        self.checkpoint_position = np.asarray(
            graph.node_pos[checkpoint_id], dtype=float
        ).copy()
        self.executor = StateConditionedReturnExecutor(
            checkpoint_id, "ETP-R1:frozen-control", persistent
        )
        self.executor.start_excursion(selected)
        self.checkpoint_candidates = {
            branch_id: current[branch_id].detach() for branch_id in persistent
        }
        self.checkpointed_excursions += 1
        self.record(
            "explore_executed", checkpoint_id=checkpoint_id,
            branch_id=selected,
            preservation_gain=round(float(macro.preservation_gain), 8),
        )
        return selected

    def _post_decision(self, current) -> None:
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        super()._post_decision(current)
        if (
            self.phase == "seeking_excursion"
            and self.ledger.status(checkpoint_id, branch_id) is OptionStatus.ACTIVE
        ):
            self.ledger.resolve_return(checkpoint_id, branch_id)

    def complete_pending_return(self) -> None:
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        v52.ContinuousController.complete_pending_return(self)
        self.ledger.resolve_return(checkpoint_id, branch_id)

    def record_navigation(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
        native_branch,
    ):
        if self.instruction is None or self.latest_history is None:
            return None
        current, persistent = self._features(
            gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks
        )
        selected = None
        if self.phase == "seeking_excursion":
            selected = self._initial_decision(current, persistent, native_branch)
        elif self.phase == "outbound_in_flight":
            self.outbound_action_executed = True
            self._post_decision(current)
        elif self.phase != "return_pending":
            raise RuntimeError(f"unknown V5.6 phase: {self.phase}")
        self.step += 1
        return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--mode", choices=("shadow", "revealnav"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside project")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    state = FullOPPActionController(
        args.seed, args.mode, torch.device("cuda:0"), controller_trace
    )
    v55._CONTROLLER = state
    v55.install_native_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", f"full_opp_v5_6_{args.mode}_{args.seed}_{args.episode_id}",
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", args.split, "TASK_CONFIG.DATASET.SPLIT", args.split,
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED), "IL.back_algo", "control",
        "INFERENCE.SPLIT", args.split, "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]", "TORCH_GPU_ID", "0",
        "VIDEO_OPTION", "[]", "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]
    summary = {
        "schema_version": "revealnav-r2r-full-opp-worker/5.6",
        "status": "RUNNING", "episode_id": args.episode_id,
        "seed": args.seed, "mode": args.mode, "split": args.split,
        "frozen_opp_config": v54.FROZEN_CONFIG,
        "task_metric_payload_read": args.mode == "revealnav", "argv": argv,
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
        state.finalize_episode()
        summary["wall_time_s"] = round(time.monotonic() - started, 3)
        summary["peak_rss_self_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary["peak_rss_children_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        summary["controller"] = {
            "strict_load": True,
            "commit_decisions": state.commit_decisions,
            "effective_commit_interventions": state.effective_commit_interventions,
            "explore_decisions": state.explore_decisions,
            "inspect_delegations": state.inspect_delegations,
            "follow_delegations": state.follow_delegations,
            "unresolved_decisions": state.unresolved_decisions,
            "checkpointed_excursions": state.checkpointed_excursions,
            "continue_decisions": state.continue_decisions,
            "backtrack_decisions": state.backtrack_decisions,
            "successful_returns": state.successful_returns,
            "failed_returns": state.failed_returns,
            "terminal_unresolved_excursions": state.terminal_unresolved_excursions,
            "ledger_counts": state.ledger.counts(),
            "final_record_hash": state.previous_hash,
            "checkpoint_triplet": {
                "ree": state.pair["ree"], "q": state.pair["q"],
                "post": state.post_row,
            },
        }
        summary["base_trace_sha256"] = sha256_file(base_trace)
        summary["controller_trace_sha256"] = sha256_file(controller_trace)
        summary["metrics"] = None
        if args.mode == "revealnav":
            stats = list(output.rglob(f"stats_ep_ckpt_270_{args.split}_r0_w1.json"))
            if len(stats) == 1:
                payload = json.loads(stats[0].read_text())
                summary["metrics"] = payload.get(str(args.episode_id))
                summary["metrics_path"] = str(stats[0].relative_to(ROOT))
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
