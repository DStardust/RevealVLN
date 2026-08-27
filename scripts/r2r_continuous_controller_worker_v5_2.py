#!/usr/bin/env python3
"""Run one paired R2R episode with the conservative RevealNav overlay."""

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
from revealnav_mf2r4 import PostExcursionAction  # noqa: E402
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims, sha256_file,
)


install_runtime_shims()

_CONTROLLER = None


class ContinuousController(pilot.ActionEnabledPilotController):
    """V5 policy heads with return actions synchronized to the ETP loop."""

    def __init__(self, seed: int, device: torch.device, trace_path: Path) -> None:
        super().__init__(seed, "natural", device, trace_path)
        self.pending_return_action = None
        self.return_departure_id = None
        self.return_departure_stop_score = None
        self.return_departure_had_stop_score = False
        self.checkpointed_excursions = 0
        self.continue_decisions = 0
        self.backtrack_decisions = 0
        self.successful_returns = 0
        self.failed_returns = 0

    def _reset_search(self) -> None:
        self.phase = "seeking_excursion"
        self.rows = []
        self.pre_histories = None
        self.selected_embedding = None
        self.checkpoint_embedding = None
        self.checkpoint_id = None
        self.checkpoint_position = None
        self.selected_branch = None
        self.executor = None

    def _schedule_return(self) -> bool:
        graph = pilot._TRAINER.gmaps[0]
        current_id = pilot._CURRENT_IDS[0]
        if (
            current_id not in graph.shortest_path
            or self.checkpoint_id not in graph.shortest_path[current_id]
        ):
            self.failed_returns += 1
            self.executor.report_return(False)
            self.record(
                "return_complete", success=False,
                reason="checkpoint_absent_from_online_graph",
            )
            self._reset_search()
            return False
        path = [
            (node, graph.node_pos[node])
            for node in graph.shortest_path[current_id][self.checkpoint_id]
        ][1:]
        tryout = bool(
            pilot._TRAINER.config.IL.tryout
            and not pilot._TRAINER.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING
        )
        self.return_departure_id = current_id
        self.return_departure_had_stop_score = current_id in graph.node_stop_scores
        self.return_departure_stop_score = graph.node_stop_scores.get(current_id)
        self.pending_return_action = {
            "action": {
                "act": 4,
                "cur_vp": current_id,
                "front_vp": self.checkpoint_id,
                "front_pos": graph.node_pos[self.checkpoint_id],
                "ghost_vp": self.selected_branch,
                "ghost_pos": graph.node_pos[self.checkpoint_id],
                "back_path": path,
                "tryout": tryout,
            },
            "vis_info": None,
        }
        self.return_intervention_attempted = True
        self.executor.request_backtrack()
        self.phase = "return_pending"
        self.record(
            "return_scheduled", checkpoint_id=self.checkpoint_id,
            departure_id=current_id, graph_path_nodes=len(path),
            controller_ref="ETP-R1:frozen-control",
        )
        return True

    def _post_decision(self, current) -> None:
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
        action = (
            PostExcursionAction.CONTINUE
            if continue_cost <= backtrack_cost else PostExcursionAction.BACKTRACK
        )
        self.post_policy_action = action.value
        self.record(
            "post_decision", policy_action=action.value,
            predicted_continue_cost=round(continue_cost, 8),
            predicted_backtrack_cost=round(backtrack_cost, 8),
            executed_return=action is PostExcursionAction.BACKTRACK,
            forced_stress_return=False,
        )
        if action is PostExcursionAction.CONTINUE:
            self.continue_decisions += 1
            self.executor.continue_excursion()
            self._reset_search()
        else:
            self.backtrack_decisions += 1
            self._schedule_return()

    def record_navigation(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ):
        if self.instruction is None or self.latest_history is None:
            return None
        current, persistent = self._features(
            gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks
        )
        selected = None
        if self.phase == "seeking_excursion":
            selected = self._initial_decision(current, persistent)
            if selected is not None:
                self.checkpointed_excursions += 1
        elif self.phase == "outbound_in_flight":
            self.outbound_action_executed = True
            self._post_decision(current)
        elif self.phase != "return_pending":
            raise RuntimeError(f"unknown continuous controller phase: {self.phase}")
        self.step += 1
        return selected

    def complete_pending_return(self) -> None:
        graph = pilot._TRAINER.gmaps[0]
        departure = self.return_departure_id
        if self.return_departure_had_stop_score:
            graph.node_stop_scores[departure] = self.return_departure_stop_score
        else:
            graph.node_stop_scores.pop(departure, None)
        position = np.asarray(pilot._TRAINER.get_pos_ori()[0][0], dtype=float)
        error = float(np.linalg.norm(position - self.checkpoint_position))
        success = math.isfinite(error) and error <= 0.5
        self.return_intervention_success = success
        self.executor.report_return(success)
        if success:
            self.successful_returns += 1
        else:
            self.failed_returns += 1
        self.record(
            "return_complete", success=success,
            final_checkpoint_error_m=round(error, 6),
            controller_ref="ETP-R1:frozen-control",
        )
        self.pending_return_action = None
        self.return_departure_id = None
        self._reset_search()


def controller():
    return _CONTROLLER


def install_continuous_hooks() -> None:
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
        if not getattr(env_type.step, "_revealvln_continuous_return", False):
            original_step = env_type.step

            def step_wrapped(envs_self, actions):
                state = controller()
                pending = None if state is None else state.pending_return_action
                if pending is not None:
                    if len(actions) != 1:
                        raise RuntimeError("continuous return requires one environment")
                    actions = [pending]
                outputs = original_step(envs_self, actions)
                if pending is not None:
                    state.complete_pending_return()
                return outputs

            step_wrapped._revealvln_continuous_return = True
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
            branch_id = state.record_navigation(
                kwargs["gmap_vp_ids"], kwargs["gmap_img_fts"],
                kwargs["gmap_masks"], kwargs["gmap_visited_masks"],
            )
            if branch_id is not None or state.pending_return_action is not None:
                index = (
                    0 if state.pending_return_action is not None
                    else kwargs["gmap_vp_ids"][0].index(branch_id)
                )
                changed = dict(result)
                logits = result["global_logits"].clone()
                logits[0].fill_(-1e9)
                logits[0, index] = 1e9
                changed["global_logits"] = logits
                return changed
        return result

    ETP.forward = forward_wrapped


def main() -> None:
    global _CONTROLLER
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--mode", choices=("baseline", "revealnav"), required=True)
    parser.add_argument("--seed", type=int)
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
    if args.mode == "revealnav":
        _CONTROLLER = ContinuousController(
            args.seed, torch.device("cuda:0"), controller_trace
        )
        install_continuous_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    name = f"continuous_v5_2_{args.mode}_{args.seed}_{args.episode_id}"
    argv = [
        "run.py", "--exp_name", name,
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", "val_unseen", "TASK_CONFIG.DATASET.SPLIT", "val_unseen",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED), "IL.back_algo", "control",
        "INFERENCE.SPLIT", "val_unseen", "TASK_CONFIG.DATASET.SUFFIX", "''",
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
        "schema_version": "revealnav-r2r-continuous-controller-worker/5.2",
        "status": "RUNNING", "episode_id": args.episode_id,
        "seed": args.seed, "mode": args.mode, "split": "val_unseen",
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
        summary["wall_time_s"] = round(time.monotonic() - started, 3)
        summary["peak_rss_self_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        summary["peak_rss_children_kib"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        state = controller()
        summary["controller"] = None if state is None else {
            "strict_load": True,
            "checkpointed_excursions": state.checkpointed_excursions,
            "continue_decisions": state.continue_decisions,
            "backtrack_decisions": state.backtrack_decisions,
            "successful_returns": state.successful_returns,
            "failed_returns": state.failed_returns,
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
        stats = list(output.rglob(
            "stats_ep_ckpt_270_val_unseen_r0_w1.json"
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
