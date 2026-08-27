#!/usr/bin/env python3
"""Execute one real ETP outbound and optional frozen return intervention."""

from __future__ import annotations

import argparse
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
HABITAT_LAB = ROOT / "third_party/habitat-lab"
HABITAT_SIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / ".remote_runtime/habitat-sim"
)).resolve()
FUSION_LOCK = ROOT / "locks/REE_Q_FUSION_CONTROLLER_V4_4.json"
POST_LOCK = ROOT / "locks/POST_EXCURSION_INTEGRATED_CONTROLLER_V4_9.json"
R2R_CHECKPOINT = ETPR1 / (
    "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)
JOINT_PRETRAINED = ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt"
)
for path in reversed((ROOT, ROOT / "scripts", ETPR1, HABITAT_LAB, HABITAT_SIM)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r3 import RelationalRevealExpiryHeads  # noqa: E402
from revealnav_mf2r4 import (  # noqa: E402
    BranchExcursionQHead, BranchMacroAction, ExecutorPhase,
    PostExcursionAction, PostExcursionQHead, ReeQFusionController,
    StateConditionedReturnExecutor,
)
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims, sha256_file,
)


install_runtime_shims()


class PilotComplete(RuntimeError):
    pass


_TRAINER = None
_CURRENT_IDS: tuple[str, ...] = ()
_LOCAL_FRONTIERS: tuple[dict[str, int], ...] = ()
_CONTROLLER = None


def stable_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


class ActionEnabledPilotController:
    def __init__(
        self, seed: int, mode: str, device: torch.device, trace_path: Path,
    ) -> None:
        if mode not in ("natural", "forced_negative"):
            raise ValueError("unknown online intervention mode")
        fusion = json.loads(FUSION_LOCK.read_text())
        post = json.loads(POST_LOCK.read_text())
        pairs = [row for row in fusion["checkpoint_pairs"] if row["seed"] == seed]
        posts = [row for row in post["post_excursion_checkpoints"]
                 if row["seed"] == seed]
        if len(pairs) != 1 or len(posts) != 1:
            raise RuntimeError("checkpoint triplet absent from lock")
        pair = pairs[0]
        post_row = posts[0]
        payloads = {}
        for name, row in (("ree", pair["ree"]), ("q", pair["q"]),
                          ("post", post_row)):
            path = ROOT / row["path"]
            if (
                path.is_symlink() or not path.is_file()
                or path.stat().st_size != row["bytes"]
                or sha256_file(path) != row["sha256"]
            ):
                raise RuntimeError(f"locked {name} checkpoint drift")
            payloads[name] = torch.load(
                path, map_location="cpu", weights_only=True
            )
        self.q_model = BranchExcursionQHead(768, 96, 128.0)
        self.q_model.load_state_dict(payloads["q"]["model_state_dict"], strict=True)
        self.ree_model = RelationalRevealExpiryHeads(768, 128, 4)
        self.ree_model.load_state_dict(
            payloads["ree"]["model_state_dict"], strict=True
        )
        self.post_model = PostExcursionQHead(768, 96, 5.0)
        self.post_model.load_state_dict(
            payloads["post"]["model_state_dict"], strict=True
        )
        for model in (self.q_model, self.ree_model, self.post_model):
            model.to(device).eval()
        self.seed = seed
        self.mode = mode
        self.device = device
        self.pair = pair
        self.post_row = post_row
        self.trace_path = trace_path
        self.trace_path.write_text("")
        self.fusion = ReeQFusionController(3, 5.0)
        self.instruction = None
        self.latest_history = None
        self.rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        self.phase = "seeking_excursion"
        self.pre_histories = None
        self.selected_embedding = None
        self.checkpoint_embedding = None
        self.checkpoint_id = None
        self.checkpoint_position = None
        self.selected_branch = None
        self.executor = None
        self.previous_hash = "0" * 64
        self.events = []
        self.outbound_action_executed = False
        self.post_policy_action = None
        self.return_intervention_attempted = False
        self.return_intervention_success = None
        self.step = 0

    def record(self, event: str, **values) -> None:
        row = {
            "event": event, "step": self.step, "mode": self.mode,
            **values, "previous_hash": self.previous_hash,
        }
        row["record_hash"] = stable_hash(row)
        self.previous_hash = row["record_hash"]
        self.events.append(row)
        with self.trace_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

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

    def _features(self, gmap_vp_ids, gmap_img_fts, gmap_masks,
                  gmap_visited_masks):
        if len(_LOCAL_FRONTIERS) != 1:
            raise RuntimeError("local frontier hook did not publish one environment")
        frontier = _LOCAL_FRONTIERS[0]
        current = {}
        for index, branch_id in enumerate(gmap_vp_ids[0]):
            if (
                index == 0 or branch_id is None
                or not bool(gmap_masks[0, index])
                or bool(gmap_visited_masks[0, index])
                or str(branch_id) not in frontier
            ):
                continue
            current[str(branch_id)] = gmap_img_fts[0, index].detach()
        persistent = tuple(sorted(
            branch_id for branch_id in current
            if frontier[branch_id] >= self.fusion.persistence_k
        ))
        return current, persistent

    def _initial_decision(self, current, persistent):
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
        persistent_indices = [index[branch_id] for branch_id in persistent]
        probabilities = torch.softmax(
            ree.target_logits[0, steps - 1, persistent_indices], -1
        )
        commit = [float(q.commit_cost[0, value]) for value in persistent_indices]
        excursion = [
            float(q.excursion_cost[0, value]) for value in persistent_indices
        ]
        natural = self.fusion.decide(
            persistent, probabilities.cpu().tolist(), commit, excursion, 3
        )
        if natural.action is not BranchMacroAction.CHECKPOINTED_EXCURSION:
            return None
        selected = natural.branch_id
        if self.mode == "forced_negative":
            ranked = sorted(
                zip(probabilities.cpu().tolist(), persistent),
                key=lambda item: (item[0], item[1]),
            )
            selected = next(
                branch_id for _, branch_id in ranked
                if branch_id != natural.branch_id
            )
        self.phase = "outbound_in_flight"
        self.pre_histories = [row[0].detach() for row in self.rows]
        self.selected_embedding = current[selected].detach()
        self.checkpoint_embedding = self.latest_history.detach()
        self.selected_branch = selected
        self.checkpoint_id = _CURRENT_IDS[0]
        graph = _TRAINER.gmaps[0]
        self.checkpoint_position = np.asarray(
            graph.node_pos[self.checkpoint_id], dtype=float
        ).copy()
        self.executor = StateConditionedReturnExecutor(
            self.checkpoint_id, "ETP-R1:frozen-control", persistent
        )
        self.executor.start_excursion(selected)
        self.record(
            "outbound_selected", natural_action=natural.action.value,
            natural_branch=natural.branch_id, executed_branch=selected,
            persistent_branch_count=len(persistent),
            executed_branch_probability=round(float(
                probabilities[list(persistent).index(selected)]
            ), 8),
            intervention=self.mode == "forced_negative",
        )
        return selected

    def _return_to_checkpoint(self, policy_action: str) -> None:
        graph = _TRAINER.gmaps[0]
        current_id = _CURRENT_IDS[0]
        self.return_intervention_attempted = True
        tryout = bool(
            _TRAINER.config.IL.tryout
            and not _TRAINER.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING
        )
        if (
            current_id not in graph.shortest_path
            or self.checkpoint_id not in graph.shortest_path[current_id]
        ):
            self.return_intervention_success = False
            self.executor.report_return(False)
            self.record(
                "return_complete", policy_action=policy_action,
                success=False, reason="checkpoint_absent_from_online_graph",
            )
            return
        path = [
            (node, graph.node_pos[node])
            for node in graph.shortest_path[current_id][self.checkpoint_id]
        ][1:]
        action = [{
            "action": {
                "act": 4, "cur_vp": current_id,
                "front_vp": self.checkpoint_id,
                "front_pos": graph.node_pos[self.checkpoint_id],
                "ghost_vp": self.selected_branch,
                "ghost_pos": graph.node_pos[self.checkpoint_id],
                "back_path": path, "tryout": tryout,
            },
            "vis_info": None,
        }]
        _TRAINER.envs.step(action)
        position = np.asarray(_TRAINER.get_pos_ori()[0][0], dtype=float)
        error = float(np.linalg.norm(position - self.checkpoint_position))
        success = math.isfinite(error) and error <= 0.5
        self.return_intervention_success = success
        self.executor.report_return(success)
        self.record(
            "return_complete", policy_action=policy_action,
            success=success, final_checkpoint_error_m=round(error, 6),
            graph_path_nodes=len(path), controller_ref="ETP-R1:frozen-control",
        )

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
        policy_action = (
            PostExcursionAction.CONTINUE
            if continue_cost <= backtrack_cost else PostExcursionAction.BACKTRACK
        )
        self.post_policy_action = policy_action.value
        execute_return = (
            policy_action is PostExcursionAction.BACKTRACK
            or self.mode == "forced_negative"
        )
        self.record(
            "post_decision", policy_action=policy_action.value,
            predicted_continue_cost=round(continue_cost, 8),
            predicted_backtrack_cost=round(backtrack_cost, 8),
            executed_return=execute_return,
            forced_stress_return=(
                self.mode == "forced_negative"
                and policy_action is not PostExcursionAction.BACKTRACK
            ),
        )
        if execute_return:
            self.executor.request_backtrack()
            self._return_to_checkpoint(policy_action.value)
        else:
            self.executor.continue_excursion()
        self.phase = "complete"
        raise PilotComplete("action-enabled pilot complete")

    def record_navigation(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ):
        if self.instruction is None or self.latest_history is None:
            return None
        current, persistent = self._features(
            gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks
        )
        if self.phase == "seeking_excursion":
            result = self._initial_decision(current, persistent)
            self.step += 1
            return result
        if self.phase == "outbound_in_flight":
            self.outbound_action_executed = True
            self._post_decision(current)
        self.step += 1
        return None


def controller():
    return _CONTROLLER


def install_online_hooks() -> None:
    from vlnce_baselines.models.R1Policy import ETP
    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer

    original_gmap = RLTrainer._nav_gmap_variable

    def gmap_wrapped(self, cur_vp, cur_pos, cur_ori, task_type):
        global _TRAINER, _CURRENT_IDS, _LOCAL_FRONTIERS
        result = original_gmap(self, cur_vp, cur_pos, cur_ori, task_type)
        _TRAINER = self
        _CURRENT_IDS = tuple(cur_vp)
        _LOCAL_FRONTIERS = tuple(
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
            if branch_id is not None:
                index = kwargs["gmap_vp_ids"][0].index(branch_id)
                changed = dict(result)
                logits = result["global_logits"].clone()
                logits[0].fill_(-1e9)
                logits[0, index] = 1e9
                changed["global_logits"] = logits
                return changed
        return result

    ETP.forward = forward_wrapped


def close_envs() -> None:
    if _TRAINER is not None and getattr(_TRAINER, "envs", None) is not None:
        try:
            _TRAINER.envs.close()
        except Exception:
            pass


def main() -> None:
    global _CONTROLLER
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--mode", choices=("natural", "forced_negative"), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside project")
    run_dir.mkdir(parents=True)
    trace_path = run_dir / "action_trace.jsonl"
    _CONTROLLER = ActionEnabledPilotController(
        args.seed, args.mode, torch.device("cuda:0"), trace_path
    )
    install_online_hooks()
    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    name = f"action_v5_{args.mode}_{args.seed}_{args.episode_id}"
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
        "schema_version": "revealnav-r2r-action-enabled-worker/5",
        "status": "RUNNING", "episode_id": args.episode_id,
        "seed": args.seed, "mode": args.mode, "split": "val_unseen",
        "argv": argv,
    }
    sys.argv = argv
    import run

    started = time.monotonic()
    try:
        run.main()
        summary["status"] = "FAIL_NO_INTERVENTION"
    except PilotComplete:
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        close_envs()
        state = _CONTROLLER
        summary.update({
            "wall_time_s": round(time.monotonic() - started, 3),
            "peak_rss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "strict_checkpoint_load": True,
            "outbound_action_executed": state.outbound_action_executed,
            "post_policy_action": state.post_policy_action,
            "return_intervention_attempted": state.return_intervention_attempted,
            "return_intervention_success": state.return_intervention_success,
            "executor_phase": (
                None if state.executor is None else state.executor.phase.value
            ),
            "events": len(state.events),
            "final_record_hash": state.previous_hash,
            "trace_sha256": sha256_file(trace_path),
            "checkpoints": {
                "ree": state.pair["ree"], "q": state.pair["q"],
                "post": state.post_row,
            },
            "paper_result": False,
        })
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
