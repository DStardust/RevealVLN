#!/usr/bin/env python3
"""Run one R2R episode with native-action ECOG preservation."""

from __future__ import annotations

import argparse
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
from revealnav_mf2r4 import (  # noqa: E402
    BranchMacroAction, PostExcursionAction, StateConditionedReturnExecutor,
)
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims, sha256_file,
)


install_runtime_shims()
_CONTROLLER = None


class NativePreservationController(v54.FullOPPContinuousController):
    """Save options around the base action; never replace the outbound action."""

    def __init__(self, seed: int, device: torch.device, trace_path: Path) -> None:
        super().__init__(seed, "revealnav", device, trace_path)
        self.native_checkpoint_count = 0
        self.native_not_persistent = 0
        self.ree_closed_return_vetoes = 0

    def _checkpoint_proposal(self, current, persistent):
        self.rows.append((self.latest_history, current))
        ordered = tuple(dict.fromkeys(
            branch_id for _, values in self.rows for branch_id in values
        ))
        if len(persistent) < 2:
            return None, None, None
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
        decision = self.fusion.decide(
            persistent, probabilities.cpu().tolist(),
            [float(q.commit_cost[0, value]) for value in indices],
            [float(q.excursion_cost[0, value]) for value in indices],
            v54.FROZEN_CONFIG["persistence_k"],
        )
        belief = self._event_belief(ree, steps - 1, indices)
        return decision, belief, probabilities

    def _initial_native_decision(
        self, current, persistent, native_branch: str | None,
    ) -> None:
        self.fusion.last_decision = None
        decision, belief, probabilities = self._checkpoint_proposal(
            current, persistent
        )
        if decision is None:
            return
        if decision.action is not BranchMacroAction.CHECKPOINTED_EXCURSION:
            if decision.reason == "checkpoint_value_not_above_frozen_opv_threshold":
                self.threshold_suppressions += 1
            return
        self.opp_checkpoint_acceptances += 1
        if native_branch not in persistent:
            self.native_not_persistent += 1
            self.record(
                "native_checkpoint_deferred",
                reason="base_action_not_in_persistent_local_branch_set",
                native_branch=native_branch,
                q_selected_branch=decision.branch_id,
                preservation_gain=round(float(decision.preservation_gain), 8),
                **{key: round(value, 8) for key, value in belief.items()},
            )
            return
        checkpoint_id = pilot._CURRENT_IDS[0]
        if not self.ledger.authorize_branch(
            checkpoint_id, native_branch, decision.preservation_gain
        ):
            raise RuntimeError("ECOG ledger rejected a native active branch")
        self.phase = "outbound_in_flight"
        self.pre_histories = [row[0].detach() for row in self.rows]
        self.selected_embedding = current[native_branch].detach()
        self.checkpoint_embedding = self.latest_history.detach()
        self.selected_branch = native_branch
        self.checkpoint_id = checkpoint_id
        graph = pilot._TRAINER.gmaps[0]
        self.checkpoint_position = np.asarray(
            graph.node_pos[checkpoint_id], dtype=float
        ).copy()
        self.executor = StateConditionedReturnExecutor(
            checkpoint_id, "ETP-R1:frozen-control", persistent
        )
        self.executor.start_excursion(native_branch)
        self.checkpoint_candidates = {
            branch_id: current[branch_id].detach() for branch_id in persistent
        }
        self.native_checkpoint_count += 1
        self.checkpointed_excursions += 1
        native_probability = float(
            probabilities[list(persistent).index(native_branch)].cpu()
        )
        self.record(
            "native_checkpoint_created", checkpoint_id=checkpoint_id,
            native_branch=native_branch, q_selected_branch=decision.branch_id,
            native_branch_probability=round(native_probability, 8),
            preservation_gain=round(float(decision.preservation_gain), 8),
            base_action_overridden=False,
            **{key: round(value, 8) for key, value in belief.items()},
        )

    def _post_decision(self, current) -> None:
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        history = torch.stack(
            [*self.pre_histories, self.latest_history.detach()]
        ).unsqueeze(0)
        local = (
            torch.stack(list(current.values())).mean(0)
            if current else torch.zeros(768, device=self.device)
        )
        with torch.no_grad():
            output = self.post_model(
                history, torch.tensor([history.shape[1]], device=self.device),
                self.instruction.unsqueeze(0),
                self.selected_embedding.unsqueeze(0),
                self.checkpoint_embedding.unsqueeze(0), local.unsqueeze(0),
                torch.tensor([1.0], device=self.device),
            )
        continue_cost = float(output.continue_cost[0])
        backtrack_cost = float(output.backtrack_cost[0])
        raw_backtrack = backtrack_cost < continue_cost
        if raw_backtrack:
            self.raw_post_backtracks += 1
        else:
            self.raw_post_continues += 1
        belief = self._post_ree_belief(current)
        ree_closed, reason = self.event_gate.post_excursion_decision(
            belief["p_discriminable"], belief["evidence"],
            belief["selected_target_probability"],
        )
        execute_backtrack = raw_backtrack and not ree_closed
        if raw_backtrack and ree_closed:
            self.ree_closed_return_vetoes += 1
        action = (
            PostExcursionAction.BACKTRACK
            if execute_backtrack else PostExcursionAction.CONTINUE
        )
        self.post_policy_action = action.value
        self.record(
            "post_decision", policy_action=action.value,
            raw_post_q_action=("backtrack" if raw_backtrack else "continue"),
            ree_closed_selected_branch=ree_closed, ree_reason=reason,
            predicted_continue_cost=round(continue_cost, 8),
            predicted_backtrack_cost=round(backtrack_cost, 8),
            executed_return=execute_backtrack,
            forced_stress_return=False,
            **{key: round(value, 8) for key, value in belief.items()},
        )
        if not execute_backtrack:
            self.continue_decisions += 1
            self.executor.continue_excursion()
            self.ledger.resolve_continue(checkpoint_id, branch_id)
            self._reset_search()
            return
        self.backtrack_decisions += 1
        if not self._schedule_return():
            self.ledger.resolve_return(checkpoint_id, branch_id)

    def complete_pending_return(self) -> None:
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        v52.ContinuousController.complete_pending_return(self)
        self.ledger.resolve_return(checkpoint_id, branch_id)

    def record_navigation(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
        native_branch: str | None,
    ) -> None:
        if self.instruction is None or self.latest_history is None:
            return
        current, persistent = self._features(
            gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks
        )
        if self.phase == "seeking_excursion":
            self._initial_native_decision(current, persistent, native_branch)
        elif self.phase == "outbound_in_flight":
            self.outbound_action_executed = True
            self._post_decision(current)
        elif self.phase != "return_pending":
            raise RuntimeError(f"unknown V5.5 phase: {self.phase}")
        self.step += 1


def controller():
    return _CONTROLLER


def install_native_hooks() -> None:
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
        env_type = type(self.envs)
        if not getattr(env_type.step, "_revealvln_native_return", False):
            original_step = env_type.step

            def step_wrapped(envs_self, actions):
                state = controller()
                pending = None if state is None else state.pending_return_action
                if pending is not None:
                    if len(actions) != 1:
                        raise RuntimeError("native return requires one environment")
                    actions = [pending]
                outputs = original_step(envs_self, actions)
                if pending is not None:
                    state.complete_pending_return()
                return outputs

            step_wrapped._revealvln_native_return = True
            env_type.step = step_wrapped
        return result

    RLTrainer._nav_gmap_variable = gmap_wrapped
    original_forward = ETP.forward

    def forward_wrapped(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        mode = kwargs.get("mode", args[0] if args else None)
        state = controller()
        if state is None:
            return result
        if mode == "language":
            state.record_language(result, kwargs["txt_masks"])
        elif mode == "panorama":
            state.record_panorama(result[0], result[1])
        elif mode == "navigation":
            chosen_index = int(torch.argmax(result["global_logits"][0]))
            ids = kwargs["gmap_vp_ids"][0]
            native_branch = (
                str(ids[chosen_index])
                if 0 < chosen_index < len(ids) and ids[chosen_index] is not None
                else None
            )
            branch_id = state.record_navigation(
                kwargs["gmap_vp_ids"], kwargs["gmap_img_fts"],
                kwargs["gmap_masks"], kwargs["gmap_visited_masks"],
                native_branch,
            )
            if branch_id is not None or state.pending_return_action is not None:
                changed = dict(result)
                logits = result["global_logits"].clone()
                logits[0].fill_(-1e9)
                index = (
                    0 if state.pending_return_action is not None
                    else kwargs["gmap_vp_ids"][0].index(branch_id)
                )
                logits[0, index] = 1e9
                changed["global_logits"] = logits
                return changed
        return result

    ETP.forward = forward_wrapped


def validate_native_actions(state, base_trace: Path) -> dict:
    actions = [
        json.loads(line) for line in base_trace.read_text().splitlines() if line
    ]
    events = [
        row for row in state.events if row["event"] == "native_checkpoint_created"
    ]
    checks = []
    for event in events:
        step = event["step"]
        observed = actions[step].get("ghost_vp") if step < len(actions) else None
        checks.append({
            "step": step, "declared_native_branch": event["native_branch"],
            "executed_ghost_vp": observed,
            "equal": observed == event["native_branch"],
        })
    if len(checks) != state.native_checkpoint_count or not all(
        row["equal"] for row in checks
    ):
        raise RuntimeError("native outbound action equality check failed")
    return {"checks": len(checks), "all_equal": True, "rows": checks}


def main() -> None:
    global _CONTROLLER
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
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
    _CONTROLLER = NativePreservationController(
        args.seed, torch.device("cuda:0"), controller_trace
    )
    install_native_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name",
        f"native_v5_5_{args.seed}_{args.episode_id}",
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
        "schema_version": "revealnav-r2r-native-preservation-worker/5.5",
        "status": "RUNNING", "episode_id": args.episode_id,
        "seed": args.seed, "mode": "revealnav", "split": args.split,
        "frozen_opp_config": v54.FROZEN_CONFIG,
        "outbound_action_policy": "unchanged_deterministic_etp_r1_argmax",
        "argv": argv,
    }
    sys.argv = argv
    import run

    started = time.monotonic()
    try:
        run.main()
        _CONTROLLER.finalize_episode()
        summary["native_action_validation"] = validate_native_actions(
            _CONTROLLER, base_trace
        )
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _CONTROLLER.finalize_episode()
        state = _CONTROLLER
        summary["wall_time_s"] = round(time.monotonic() - started, 3)
        summary["peak_rss_self_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        summary["peak_rss_children_kib"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        summary["controller"] = {
            "strict_load": True,
            "native_checkpoint_count": state.native_checkpoint_count,
            "native_not_persistent": state.native_not_persistent,
            "continue_decisions": state.continue_decisions,
            "backtrack_decisions": state.backtrack_decisions,
            "raw_post_continues": state.raw_post_continues,
            "raw_post_backtracks": state.raw_post_backtracks,
            "ree_closed_return_vetoes": state.ree_closed_return_vetoes,
            "successful_returns": state.successful_returns,
            "failed_returns": state.failed_returns,
            "terminal_unresolved_excursions": state.terminal_unresolved_excursions,
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
        summary["controller_trace_sha256"] = sha256_file(controller_trace)
        stats = list(output.rglob(
            f"stats_ep_ckpt_270_{args.split}_r0_w1.json"
        ))
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
