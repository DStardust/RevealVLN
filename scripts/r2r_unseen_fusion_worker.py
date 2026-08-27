#!/usr/bin/env python3
"""Run one R2R val_unseen episode with the locked REE+Q shadow controller."""

from __future__ import annotations

import argparse
import hashlib
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
LOCK = ROOT / "locks/REE_Q_FUSION_CONTROLLER_V4_4.json"
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
    BranchExcursionQHead, BranchMacroAction, ReeQFusionController,
)
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims, sha256_file,
)


install_runtime_shims()

_LOCAL_FRONTIERS: tuple[dict[str, int], ...] = ()


def install_local_frontier_hook() -> None:
    """Expose only ghosts directly observed from the current graph node."""

    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer

    original = RLTrainer._nav_gmap_variable
    if getattr(original, "_revealvln_local_frontier", False):
        return

    def wrapped(self, cur_vp, cur_pos, cur_ori, task_type):
        global _LOCAL_FRONTIERS
        result = original(self, cur_vp, cur_pos, cur_ori, task_type)
        _LOCAL_FRONTIERS = tuple(
            {
                ghost_id: int(graph.ghost_embeds[ghost_id][1])
                for ghost_id, front_ids in graph.ghost_fronts.items()
                if current_id in front_ids
            }
            for graph, current_id in zip(self.gmaps, cur_vp)
        )
        return result

    wrapped._revealvln_local_frontier = True
    RLTrainer._nav_gmap_variable = wrapped


install_local_frontier_hook()


class FusedShadowController:
    def __init__(self, seed: int, device: torch.device, output: Path) -> None:
        lock = json.loads(LOCK.read_text())
        matches = [row for row in lock["checkpoint_pairs"] if row["seed"] == seed]
        if len(matches) != 1:
            raise RuntimeError("seed is absent from the fusion checkpoint lock")
        pair = matches[0]
        payloads = {}
        for name in ("ree", "q"):
            frozen = pair[name]
            path = ROOT / frozen["path"]
            if (
                path.is_symlink() or not path.is_file()
                or path.stat().st_size != frozen["bytes"]
                or sha256_file(path) != frozen["sha256"]
            ):
                raise RuntimeError(f"locked {name} checkpoint provenance drift")
            payloads[name] = torch.load(
                path, map_location="cpu", weights_only=True
            )
        self.q_model = BranchExcursionQHead(768, 96, 128.0)
        self.q_model.load_state_dict(
            payloads["q"]["model_state_dict"], strict=True
        )
        self.ree_model = RelationalRevealExpiryHeads(768, 128, 4)
        self.ree_model.load_state_dict(
            payloads["ree"]["model_state_dict"], strict=True
        )
        self.q_model.to(device).eval()
        self.ree_model.to(device).eval()
        self.seed = seed
        self.pair = pair
        self.device = device
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("")
        self.controller = ReeQFusionController(3, 5.0)
        self.instruction: torch.Tensor | None = None
        self.latest_history: torch.Tensor | None = None
        self.rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        self.branch_observations: dict[str, int] = {}
        self.saved_signatures: set[tuple[str, ...]] = set()
        self.previous_hash = "0" * 64
        self.step = 0
        self.decision_rows = 0
        self.checkpoint_rows = 0

    def record_language(self, embeddings, mask) -> None:
        pooled = (embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        self.instruction = pooled[0].detach()

    def record_panorama(self, embeddings, mask) -> None:
        pooled = (embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        self.latest_history = pooled[0].detach()

    def record_navigation(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ) -> None:
        if self.instruction is None or self.latest_history is None:
            return
        if len(_LOCAL_FRONTIERS) != 1:
            raise RuntimeError("local frontier hook did not publish one environment")
        local_frontier = _LOCAL_FRONTIERS[0]
        current: dict[str, torch.Tensor] = {}
        for index, branch_id in enumerate(gmap_vp_ids[0]):
            if (
                index == 0 or branch_id is None
                or not bool(gmap_masks[0, index])
                or bool(gmap_visited_masks[0, index])
                or str(branch_id) not in local_frontier
            ):
                continue
            current[str(branch_id)] = gmap_img_fts[0, index].detach()
        current_ids = set(current)
        self.branch_observations = {
            branch_id: local_frontier[branch_id] for branch_id in current_ids
        }
        signature = tuple(sorted(current_ids))
        persistent = tuple(sorted(
            branch_id for branch_id in current_ids
            if self.branch_observations[branch_id] >= self.controller.persistence_k
        ))
        if not signature:
            self.step += 1
            return
        self.rows.append((self.latest_history, current))
        ordered_ids = tuple(dict.fromkeys(
            branch_id for _, candidates in self.rows for branch_id in candidates
        ))
        if len(ordered_ids) < 2:
            self.step += 1
            return
        branch_index = {
            branch_id: index for index, branch_id in enumerate(ordered_ids)
        }
        steps = len(self.rows)
        history = torch.stack([row[0] for row in self.rows]).unsqueeze(0)
        candidates = torch.zeros(
            1, steps, len(ordered_ids), 768, device=self.device
        )
        mask = torch.zeros(
            1, steps, len(ordered_ids), dtype=torch.bool, device=self.device
        )
        for time_index, (_, values) in enumerate(self.rows):
            for branch_id, value in values.items():
                index = branch_index[branch_id]
                candidates[0, time_index, index] = value
                mask[0, time_index, index] = True
        decision_index = torch.tensor([steps - 1], device=self.device)
        with torch.no_grad():
            q_output = self.q_model(
                history, candidates, mask, self.instruction.unsqueeze(0),
                decision_index,
            )
            budgets = torch.tensor(
                [1.5, 2.0, 3.0, 4.0], device=self.device
            ).view(1, 1, 4).expand(1, steps, 4)
            ree_output = self.ree_model(
                history, candidates, mask, budgets,
                self.instruction.unsqueeze(0),
            )
        current_order = tuple(sorted(current_ids))
        current_indices = [branch_index[branch_id] for branch_id in current_order]
        probabilities = torch.softmax(
            ree_output.target_logits[0, steps - 1, current_indices], dim=-1
        )
        probability_by_id = {
            branch_id: float(probability)
            for branch_id, probability in zip(current_order, probabilities)
        }
        persistent_indices = [branch_index[branch_id] for branch_id in persistent]
        target_probabilities = [probability_by_id[branch_id] for branch_id in persistent]
        commit_costs = [
            float(q_output.commit_cost[0, index]) for index in persistent_indices
        ]
        excursion_costs = [
            float(q_output.excursion_cost[0, index]) for index in persistent_indices
        ]
        decision = self.controller.decide(
            persistent, target_probabilities, commit_costs, excursion_costs,
            self.controller.persistence_k,
        )
        raw_decision = self.controller.macro.decide(
            persistent, commit_costs, excursion_costs,
            self.controller.persistence_k,
        )
        eligible = decision.action is not BranchMacroAction.DEFER
        checkpoint_created = (
            eligible
            and decision.action is BranchMacroAction.CHECKPOINTED_EXCURSION
            and persistent not in self.saved_signatures
        )
        if eligible:
            self.decision_rows += 1
        if checkpoint_created:
            self.saved_signatures.add(persistent)
            self.checkpoint_rows += 1
        row = {
            "step": self.step,
            "candidate_count": len(signature),
            "persistent_candidate_count": len(persistent),
            "minimum_persistent_observations": (
                min(self.branch_observations[branch_id] for branch_id in persistent)
                if persistent else 0
            ),
            "decision_eligible": eligible,
            "macro_action": decision.action.value,
            "branch_id": decision.branch_id,
            "raw_q_macro_action": raw_decision.action.value,
            "raw_q_branch_id": raw_decision.branch_id,
            "selected_target_probability": (
                None if decision.branch_id is None
                else round(probability_by_id[decision.branch_id], 8)
            ),
            "predicted_fused_cost": (
                None if decision.predicted_cost is None
                else round(decision.predicted_cost, 8)
            ),
            "checkpoint_created": checkpoint_created,
            "shadow_only_not_executed": True,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        row["record_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        self.previous_hash = row["record_hash"]
        with self.output.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.step += 1


_CONTROLLER: FusedShadowController | None = None


def controller() -> FusedShadowController | None:
    global _CONTROLLER
    if str(os.getpid()) != os.environ.get("REVEALVLN_CONTROLLER_MAIN_PID"):
        return None
    output = os.environ.get("REVEALVLN_CONTROLLER_TRACE")
    seed = os.environ.get("REVEALVLN_CONTROLLER_SEED")
    if not output or not seed:
        return None
    if _CONTROLLER is None:
        _CONTROLLER = FusedShadowController(
            int(seed), torch.device("cuda:0"), Path(output)
        )
    return _CONTROLLER


def install_forward_controller() -> None:
    from vlnce_baselines.models.R1Policy import ETP

    original_forward = ETP.forward
    if getattr(original_forward, "_revealvln_ree_q_shadow", False):
        return

    def wrapped(self, *args, **kwargs):
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
            state.record_navigation(
                kwargs["gmap_vp_ids"], kwargs["gmap_img_fts"],
                kwargs["gmap_masks"], kwargs["gmap_visited_masks"],
            )
        return result

    wrapped._revealvln_ree_q_shadow = True
    ETP.forward = wrapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be a new directory inside the project")
    run_dir.mkdir(parents=True)
    trace_path = run_dir / "base_trace.jsonl"
    controller_path = run_dir / "fusion_controller.jsonl"
    trace_path.write_text("")
    os.environ.update({
        "REVEALVLN_BASE_TRACE": str(trace_path),
        "REVEALVLN_CONTROLLER_MAIN_PID": str(os.getpid()),
        "REVEALVLN_CONTROLLER_TRACE": str(controller_path),
        "REVEALVLN_CONTROLLER_SEED": str(args.seed),
    })
    install_forward_controller()
    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output_root = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", args.exp_name,
        "--run-type", "eval", "--exp-config", "run_r2r/iter_train.yaml",
        "EVAL.SPLIT", "val_unseen",
        "TASK_CONFIG.DATASET.SPLIT", "val_unseen",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']",
        "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(R2R_CHECKPOINT),
        "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED),
        "IL.back_algo", "control",
        "INFERENCE.SPLIT", "val_unseen",
        "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]",
        "TORCH_GPU_ID", "0", "VIDEO_OPTION", "[]",
        "TENSORBOARD_DIR", str(output_root / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output_root / "checkpoints"),
        "RESULTS_DIR", str(output_root / "results"),
    ]
    summary = {
        "schema_version": "revealnav-r2r-unseen-fusion-worker/1",
        "status": "RUNNING",
        "episode_id": args.episode_id,
        "seed": args.seed,
        "split": "val_unseen",
        "controller_mode": "locked_ree_q_fusion_shadow_only",
        "shadow_actions_executed": 0,
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
            "ree_checkpoint": state.pair["ree"],
            "q_checkpoint": state.pair["q"],
            "strict_load": True,
            "formula": "q + 5.0 * (1 - p_target)",
            "rows": len(state.rows),
            "decision_rows": state.decision_rows,
            "checkpoint_rows": state.checkpoint_rows,
            "final_record_hash": state.previous_hash,
        }
        summary["base_trace_sha256"] = sha256_file(trace_path)
        summary["controller_trace_sha256"] = (
            sha256_file(controller_path) if controller_path.is_file() else None
        )
        stats = list(output_root.rglob(
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
