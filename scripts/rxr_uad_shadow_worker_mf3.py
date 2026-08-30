#!/usr/bin/env python3
"""Run one RxR-train episode with action-preserving UAD shadow probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ETPR1 = ROOT / "third_party/ETP-R1"
HABITAT_LAB = ROOT / "third_party/habitat-lab"
HABITAT_SIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / ".remote_runtime/habitat-sim"
)).resolve()
RXR_CHECKPOINT = ETPR1 / (
    "data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth"
)
JOINT_PRETRAINED = ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/"
    "model_step_367500.pt"
)
UAD_ROOT = ROOT / (
    "artifacts/evaluation/mf2_scale_relational_v2_count_stable"
)
UAD_SEEDS = (20260826, 20260827, 20260828)
for path in reversed((ROOT, ETPR1, HABITAT_LAB, HABITAT_SIM)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r2 import RelationalRevealOptionHeads  # noqa: E402
from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE,
    classify_shadow_outcome,
    current_local_action_indices,
)
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims,
    sha256_file,
)


# Habitat VectorEnv uses forkserver and re-imports this module in the child.
install_runtime_shims()


def _ghost_costs_to_current_teacher_subgoal(self, ghost_real_positions):
    """Label-only geodesic costs after ETP's native RxR teacher update."""

    subgoal = self.prev_sub_goal_pos
    return {
        str(identity): min(
            float(self._env.sim.geodesic_distance(position, subgoal))
            for position in positions
        )
        for identity, positions in ghost_real_positions.items()
        if positions
    }


# VectorEnv uses forkserver and re-imports this module in the environment child.
# Install the read-only label method at module import time so it is available in
# that child without modifying frozen Habitat/ETP-R1 sources.
from vlnce_baselines.common.environments import VLNCEDaggerEnv  # noqa: E402

VLNCEDaggerEnv.mf3b_ghost_costs_to_current_teacher_subgoal = (
    _ghost_costs_to_current_teacher_subgoal
)

_CONTROLLER = None
_TRAINER = None
_LOCAL_ACTION_IDS: tuple[set[str], ...] = ()
_NO_VP_LEFT: tuple[bool, ...] = ()


def stable_hash(value: dict) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_native_trace(rows: list[dict], trace_rows: list[dict]) -> dict:
    """Verify that every shadow decision matches the executed base action."""

    checks = []
    for row in rows:
        step = row["step"]
        if step >= len(trace_rows):
            raise RuntimeError("base action trace is shorter than shadow trace")
        executed = trace_rows[step]
        if row["native_action_index"] == 0:
            equal = int(executed["act"]) == 0
        else:
            equal = (
                int(executed["act"]) == 4
                and str(executed.get("ghost_vp"))
                == row["native_action_id"]
            )
        checks.append(equal)
    if not all(checks):
        raise RuntimeError("UAD shadow changed or misreported a native action")
    return {"checked_decisions": len(checks), "all_equal": True}


class UADShadowController:
    """Probe accepted count-stable UAD models without changing ETP output."""

    def __init__(
        self,
        device: torch.device,
        trace_path: Path,
        *,
        collect_feature_path: Path | None = None,
        contextual_gmap_features: bool = False,
        policy_fusion_features: bool = False,
    ) -> None:
        self.device = device
        self.trace_path = trace_path
        self.trace_path.write_text("")
        self.models = []
        self.checkpoints = []
        self.collect_feature_path = collect_feature_path
        self.contextual_gmap_features = contextual_gmap_features
        self.policy_fusion_features = policy_fusion_features
        for seed in (() if collect_feature_path is not None else UAD_SEEDS):
            checkpoint = UAD_ROOT / (
                f"seed_{seed}/relational_ree_count_stable.pt"
            )
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            if not (
                payload.get("schema_version")
                == "revealnav-mf2-scale-model-checkpoint/2"
                and payload.get("model_name") == "relational_ree_count_stable"
                and payload.get("seed") == seed
                and payload.get("candidate_count_encoding") == "saturating"
            ):
                raise RuntimeError("count-stable UAD checkpoint schema drift")
            model = RelationalRevealOptionHeads(
                768, int(payload["hidden_dim"]), 4,
                candidate_count_encoding="saturating",
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to(device).eval()
            self.models.append((seed, model))
            self.checkpoints.append({
                "seed": seed,
                "path": str(checkpoint.relative_to(ROOT)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "strict_load": True,
                "engineering_initialization_only": True,
            })
        self.instruction = None
        self.latest_history = None
        self.rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        self.supervision: list[dict] = []
        self.decisions: list[dict] = []
        self.pending_navigation = None
        self.previous_hash = "0" * 64
        self.step = 0

    def record_language(self, embeddings: torch.Tensor, mask: torch.Tensor) -> None:
        self.instruction = (
            (embeddings * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_panorama(self, embeddings: torch.Tensor, mask: torch.Tensor) -> None:
        self.latest_history = (
            (embeddings * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_navigation(
        self,
        gmap_vp_ids,
        gmap_img_fts: torch.Tensor,
        gmap_masks: torch.Tensor,
        gmap_visited_masks: torch.Tensor,
        global_logits: torch.Tensor,
        teacher_action: int,
        teacher_costs: dict[str, float] | None = None,
    ) -> None:
        global _TRAINER
        if self.instruction is None or self.latest_history is None:
            return
        if not (
            len(gmap_vp_ids) == 1
            and len(_LOCAL_ACTION_IDS) == 1
            and len(_NO_VP_LEFT) == 1
        ):
            raise RuntimeError("MF3B shadow requires exactly one environment")
        ids = gmap_vp_ids[0]
        action_mask = [bool(value) for value in gmap_masks[0]]
        visited_mask = [bool(value) for value in gmap_visited_masks[0]]
        local_action_ids = _LOCAL_ACTION_IDS[0]
        current_global_indices = current_local_action_indices(
            ids, action_mask, visited_mask, local_action_ids,
        )
        current = {}
        for index in range(1, len(ids)):
            identity = None if ids[index] is None else str(ids[index])
            if (
                identity is None or not action_mask[index]
                or visited_mask[index] or identity not in local_action_ids
            ):
                continue
            current[identity] = gmap_img_fts[0, index].detach()
        if current:
            self.rows.append((self.latest_history, current))
            self._record_supervision(
                ids, current, global_logits[0], teacher_action, teacher_costs
            )

        if bool(_NO_VP_LEFT[0]) or self.step >= int(_TRAINER.max_len) - 1:
            native_action = 0
        else:
            native_action = int(torch.argmax(global_logits[0]))
        if (
            self.collect_feature_path is None
            and len(current_global_indices) >= 2 and self.rows
        ):
            self._record_decision(
                ids, current_global_indices, native_action, teacher_action,
            )
        self.step += 1

    def prepare_navigation(self, *values) -> None:
        if self.pending_navigation is not None:
            raise RuntimeError("previous UAD shadow teacher label is missing")
        self.pending_navigation = values

    def complete_navigation(
        self, teacher_action: int, teacher_costs: dict[str, float] | None = None
    ) -> None:
        if self.pending_navigation is None:
            raise RuntimeError("UAD shadow teacher label has no policy output")
        values = self.pending_navigation
        self.pending_navigation = None
        self.record_navigation(*values, teacher_action, teacher_costs)

    def _record_supervision(
        self,
        ids,
        current: dict[str, torch.Tensor],
        global_logits: torch.Tensor,
        teacher_action: int,
        teacher_costs: dict[str, float] | None,
    ) -> None:
        if self.collect_feature_path is None:
            return
        teacher_id = (
            str(ids[teacher_action])
            if 0 < teacher_action < len(ids) and ids[teacher_action] is not None
            else None
        )
        target_in_set = float(teacher_id in current)
        separation = -1.0
        if teacher_id in current and teacher_costs is not None:
            costs = [
                (float(teacher_costs[identity]), identity)
                for identity in current if identity in teacher_costs
            ]
            costs.sort()
            if len(costs) >= 2 and costs[0][1] == teacher_id:
                # Half a metre is two Habitat low-level forward steps.  It is
                # fixed as a label-stability resolution, not metric-tuned.
                separation = float(costs[1][0] - costs[0][0] >= 0.5)
        self.supervision.append({
            "target_id": teacher_id if target_in_set else None,
            "target_in_set": target_in_set,
            "separation": separation,
            "native_id": (
                None if int(torch.argmax(global_logits)) == 0
                else str(ids[int(torch.argmax(global_logits))])
            ),
            "native_scores": {
                identity: float(global_logits[ids.index(identity)].detach())
                for identity in current
            },
            "outside_score": max(
                float(global_logits[index].detach())
                for index in range(len(ids))
                if index == 0 or str(ids[index]) not in current
            ),
        })

    def write_feature_shard(self) -> dict | None:
        """Write one exact-online-front-end, label-only training episode."""

        if self.collect_feature_path is None:
            return None
        if len(self.rows) != len(self.supervision):
            raise RuntimeError("MF3B feature/supervision length drift")
        if not self.rows:
            raise RuntimeError("MF3B episode contains no current-local candidates")
        ordered = tuple(dict.fromkeys(
            identity for _, values in self.rows for identity in values
        ))
        tensor_candidate_count = max(2, len(ordered))
        branch_index = {identity: index for index, identity in enumerate(ordered)}
        steps = len(self.rows)
        history = torch.stack([value for value, _ in self.rows]).cpu().float()
        feature_dims = {
            int(embedding.shape[-1])
            for _, values in self.rows for embedding in values.values()
        }
        if len(feature_dims) != 1:
            raise RuntimeError("MF3B candidate feature dimension drift")
        candidate_feature_dim = feature_dims.pop()
        candidates = torch.zeros(
            steps, tensor_candidate_count, candidate_feature_dim
        )
        mask = torch.zeros(steps, tensor_candidate_count, dtype=torch.bool)
        target_index = torch.full((steps,), -1, dtype=torch.long)
        native_index = torch.full((steps,), -1, dtype=torch.long)
        native_scores = torch.full((steps, tensor_candidate_count), -torch.inf)
        outside_score = torch.empty(steps)
        for step, ((_, values), label) in enumerate(zip(
            self.rows, self.supervision
        )):
            for identity, embedding in values.items():
                index = branch_index[identity]
                candidates[step, index] = embedding.detach().cpu().float()
                mask[step, index] = True
            if label["target_id"] is not None:
                target_index[step] = branch_index[label["target_id"]]
            if label["native_id"] in values:
                native_index[step] = branch_index[label["native_id"]]
            for identity, score in label["native_scores"].items():
                native_scores[step, branch_index[identity]] = score
            outside_score[step] = label["outside_score"]
        arrays = {
            "instruction_embedding": self.instruction.detach().cpu().float().numpy(),
            "history_embeddings": history.numpy(),
            "candidate_embeddings": candidates.numpy(),
            "candidate_mask": mask.numpy(),
            "target_index": target_index.numpy(),
            "native_index": native_index.numpy(),
            "native_scores": native_scores.numpy(),
            "outside_score": outside_score.numpy(),
            "target_in_set": np.asarray([
                row["target_in_set"] for row in self.supervision
            ], dtype=np.float32),
            "separation": np.asarray([
                row["separation"] for row in self.supervision
            ], dtype=np.float32),
            # These semantic/timing targets are deliberately not invented by
            # a geometry teacher.  Existing audited Reveal Events supervise
            # them in mixed-source training; -1 masks them for this shard.
            "evidence_complete": np.full(steps, -1.0, dtype=np.float32),
            "reveal_hazard": np.full(steps, -1.0, dtype=np.float32),
            "expiry_hazard": np.full(steps, -1.0, dtype=np.float32),
        }
        part = self.collect_feature_path.with_name(
            self.collect_feature_path.name + ".part"
        )
        with part.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(part, self.collect_feature_path)
        return {
            "path": str(self.collect_feature_path.relative_to(ROOT)),
            "bytes": self.collect_feature_path.stat().st_size,
            "sha256": sha256_file(self.collect_feature_path),
            "steps": steps,
            "candidate_count": tensor_candidate_count,
            "observed_candidate_count": len(ordered),
            "candidate_feature_dim": candidate_feature_dim,
            "target_in_set_positive": int(arrays["target_in_set"].sum()),
            "separation_labeled": int((arrays["separation"] >= 0).sum()),
            "separation_positive": int((arrays["separation"] == 1).sum()),
            "observation_frontend": (
                "frozen_etp_r1_policy_fusion_token"
                if self.policy_fusion_features
                else "frozen_etp_r1_contextual_gmap_token"
                if self.contextual_gmap_features
                else "frozen_etp_r1_12_view_graphmap"
            ),
            "future_teacher_used_as_online_input": False,
        }

    def _record_decision(
        self,
        ids,
        current_global_indices: tuple[int, ...],
        native_action: int,
        teacher_action: int,
    ) -> None:
        ordered = tuple(dict.fromkeys(
            identity for _, values in self.rows for identity in values
        ))
        branch_index = {identity: index for index, identity in enumerate(ordered)}
        steps = len(self.rows)
        history = torch.stack([value for value, _ in self.rows]).unsqueeze(0)
        candidates = torch.zeros(
            1, steps, len(ordered), 768, device=self.device
        )
        candidate_mask = torch.zeros(
            1, steps, len(ordered), dtype=torch.bool, device=self.device
        )
        for time_index, (_, values) in enumerate(self.rows):
            for identity, embedding in values.items():
                candidate_index = branch_index[identity]
                candidates[0, time_index, candidate_index] = embedding
                candidate_mask[0, time_index, candidate_index] = True
        current_ids = tuple(str(ids[index]) for index in current_global_indices)
        current_candidate_indices = [branch_index[value] for value in current_ids]
        budgets = torch.tensor(
            [1.5, 2.0, 3.0, 4.0], device=self.device
        ).view(1, 1, 4).expand(1, steps, 4)
        members = []
        with torch.no_grad():
            for seed, model in self.models:
                output = model(
                    history, candidates, candidate_mask, budgets,
                    self.instruction.unsqueeze(0),
                )
                scores = output.target_logits[
                    0, steps - 1, current_candidate_indices
                ]
                probability = torch.softmax(scores, dim=-1)
                choice = int(torch.argmax(probability))
                uad_action = current_global_indices[choice]
                target_in_set = torch.sigmoid(
                    output.target_in_set_logit[0, steps - 1]
                )
                separation = torch.sigmoid(
                    output.separation_logit[0, steps - 1]
                )
                evidence = torch.sigmoid(
                    output.evidence_logit[0, steps - 1]
                )
                p_decisive = target_in_set * separation * evidence
                top_two = torch.topk(probability, 2).values
                members.append({
                    "seed": seed,
                    "uad_action_index": uad_action,
                    "uad_action_id": str(ids[uad_action]),
                    "current_candidate_probabilities": [
                        round(float(value), 8) for value in probability
                    ],
                    "p_target_in_set": round(float(target_in_set), 8),
                    "p_separation": round(float(separation), 8),
                    "p_evidence": round(float(evidence), 8),
                    "p_decisive": round(float(p_decisive), 8),
                    "target_margin": round(float(top_two[0] - top_two[1]), 8),
                    "outcome": classify_shadow_outcome(
                        native_action, uad_action, teacher_action,
                        current_global_indices,
                    ),
                })
        row = {
            "schema_version": "revealnav-mf3b-uad-shadow-decision/1",
            "step": self.step,
            "native_action_index": native_action,
            "native_action_id": (
                None if native_action == 0 else str(ids[native_action])
            ),
            "teacher_action_index_label_only": teacher_action,
            "teacher_action_id_label_only": (
                None if teacher_action <= 0 else str(ids[teacher_action])
            ),
            "current_local_action_indices": list(current_global_indices),
            "current_local_action_ids": list(current_ids),
            "native_action_in_current_set": (
                native_action in current_global_indices
            ),
            "members": members,
            "shadow_only_actions_changed": 0,
            "observation_frontend": "frozen_etp_r1_12_view_panorama",
            "teacher_used_as_online_input": False,
            "previous_hash": self.previous_hash,
        }
        row.update(MF3B_SCOPE)
        row["record_hash"] = stable_hash(row)
        self.previous_hash = row["record_hash"]
        self.decisions.append(row)
        with self.trace_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def outcome_counts(self) -> dict[str, dict[str, int]]:
        result = {}
        for seed in UAD_SEEDS:
            counts = Counter(
                member["outcome"]
                for row in self.decisions for member in row["members"]
                if member["seed"] == seed
            )
            result[str(seed)] = dict(sorted(counts.items()))
        return result


def controller() -> UADShadowController | None:
    return _CONTROLLER


def install_shadow_hooks() -> None:
    """Install read-only ETP hooks; navigation output is returned unchanged."""

    from vlnce_baselines.models.R1Policy import ETP
    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer

    original_gmap = RLTrainer._nav_gmap_variable
    original_rollout = RLTrainer.rollout
    original_teacher = RLTrainer._teacher_action_new

    def rollout_wrapped(self, mode, *args, **kwargs):
        # The stock eval path omits candidate real positions and therefore
        # cannot compute its own teacher action. Enable that existing label
        # path only after environment construction. Worker-side video remains
        # disabled because VLNCEDaggerEnv captured VIDEO_OPTION=[] at init.
        was_frozen = self.config.is_frozen()
        self.config.defrost()
        previous_video_option = self.config.VIDEO_OPTION
        self.config.VIDEO_OPTION = ["mf3b_label_only"]
        if was_frozen:
            self.config.freeze()
        try:
            return original_rollout(self, mode, *args, **kwargs)
        finally:
            self.config.defrost()
            self.config.VIDEO_OPTION = previous_video_option
            if was_frozen:
                self.config.freeze()

    def teacher_wrapped(self, *args, **kwargs):
        random_state = random.getstate()
        try:
            result = original_teacher(self, *args, **kwargs)
        finally:
            random.setstate(random_state)
        state = controller()
        if state is not None:
            if result.shape != (1,):
                raise RuntimeError("MF3B shadow requires one teacher action")
            teacher_costs = None
            if state.collect_feature_path is not None and int(result[0]) > 0:
                if len(self.gmaps) != 1:
                    raise RuntimeError("MF3B collection requires one graph")
                teacher_costs = self.envs.call_at(
                    0,
                    "mf3b_ghost_costs_to_current_teacher_subgoal",
                    {"ghost_real_positions": {
                        str(identity): [
                            [float(value) for value in position]
                            for position in positions
                        ]
                        for identity, positions in self.gmaps[
                            0
                        ].ghost_real_pos.items()
                    }},
                )
            state.complete_navigation(int(result[0]), teacher_costs)
        return result

    RLTrainer.rollout = rollout_wrapped
    RLTrainer._teacher_action_new = teacher_wrapped

    def gmap_wrapped(self, cur_vp, cur_pos, cur_ori, task_type):
        global _TRAINER, _LOCAL_ACTION_IDS, _NO_VP_LEFT
        result = original_gmap(self, cur_vp, cur_pos, cur_ori, task_type)
        _TRAINER = self
        _NO_VP_LEFT = tuple(bool(value) for value in result["no_vp_left"])
        _LOCAL_ACTION_IDS = tuple(
            {
                str(ghost_id) for ghost_id, fronts in graph.ghost_fronts.items()
                if current_id in fronts
            }
            for graph, current_id in zip(self.gmaps, cur_vp)
        )
        return result

    RLTrainer._nav_gmap_variable = gmap_wrapped
    original_forward = ETP.forward

    def forward_wrapped(self, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else None)
        state = controller()
        captured_policy_input = []
        handle = None
        if (
            state is not None and mode == "navigation"
            and state.policy_fusion_features
        ):
            def capture_policy_input(_module, values):
                if len(values) != 1 or captured_policy_input:
                    raise RuntimeError("MF3 policy fusion hook cardinality drift")
                captured_policy_input.append(values[0])

            handle = self.vln_bert.global_sap_head.register_forward_pre_hook(
                capture_policy_input
            )
        try:
            result = original_forward(self, *args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
        if state is None:
            return result
        if mode == "language":
            state.record_language(result, kwargs["txt_masks"])
        elif mode == "panorama":
            state.record_panorama(result[0], result[1])
        elif mode == "navigation":
            if state.policy_fusion_features and len(captured_policy_input) != 1:
                raise RuntimeError("MF3 policy fusion feature was not captured")
            gmap_features = (
                captured_policy_input[0]
                if state.policy_fusion_features and len(captured_policy_input) == 1
                else result["gmap_embeds"]
                if state.contextual_gmap_features
                else kwargs["gmap_img_fts"]
            )
            if gmap_features.shape[:2] != kwargs["gmap_img_fts"].shape[:2]:
                raise RuntimeError("MF3 contextual graph feature shape drift")
            state.prepare_navigation(
                kwargs["gmap_vp_ids"], gmap_features,
                kwargs["gmap_masks"], kwargs["gmap_visited_masks"],
                result["global_logits"],
            )
        return result

    ETP.forward = forward_wrapped


def main() -> None:
    global _CONTROLLER
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--collect-feature", action="store_true")
    parser.add_argument("--contextual-gmap-features", action="store_true")
    parser.add_argument("--policy-fusion-features", action="store_true")
    args = parser.parse_args()
    if args.contextual_gmap_features and args.policy_fusion_features:
        raise SystemExit("select at most one contextual feature source")
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and inside the project")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    shadow_trace = run_dir / "uad_shadow.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    feature_path = run_dir / "online_feature.npz" if args.collect_feature else None
    _CONTROLLER = UADShadowController(
        torch.device("cuda:0"), shadow_trace,
        collect_feature_path=feature_path,
        contextual_gmap_features=args.contextual_gmap_features,
        policy_fusion_features=args.policy_fusion_features,
    )
    install_shadow_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name", f"mf3b_uad_shadow_{args.episode_id}",
        "--run-type", "eval", "--exp-config", "run_rxr/iter_train.yaml",
        "EVAL.SPLIT", "train", "EVAL.LANGUAGES", "['en-US','en-IN']",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']",
        "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(RXR_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED),
        "IL.back_algo", "control",
        "IL.RECOLLECT_TRAINER.gt_file",
        "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        "TASK_CONFIG.DATASET.SPLIT", "train",
        "INFERENCE.SPLIT", "train", "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]",
        "TORCH_GPU_ID", "0", "VIDEO_OPTION", "[]",
        "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]
    summary = {
        "schema_version": "revealnav-mf3b-uad-shadow-run/1",
        "status": "RUNNING",
        "episode_id": args.episode_id,
        "split": "train",
        "observation_frontend": "frozen_etp_r1_12_view_panorama",
        "teacher_used_as_online_input": False,
        "actions_changed": 0,
        "checkpoints": _CONTROLLER.checkpoints,
        "argv": argv,
        **MF3B_SCOPE,
    }
    sys.argv = argv
    import run

    started = time.monotonic()
    try:
        run.main()
        summary["online_feature"] = _CONTROLLER.write_feature_shard()
        trace_rows = [
            json.loads(line) for line in base_trace.read_text().splitlines()
            if line
        ]
        summary["native_action_verification"] = verify_native_trace(
            _CONTROLLER.decisions, trace_rows
        )
        summary["status"] = "SHADOW_PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["wall_time_s"] = round(time.monotonic() - started, 3)
        summary["decision_rows"] = len(_CONTROLLER.decisions)
        summary["outcome_counts"] = _CONTROLLER.outcome_counts()
        summary["final_record_hash"] = _CONTROLLER.previous_hash
        summary["base_trace_sha256"] = sha256_file(base_trace)
        summary["shadow_trace_sha256"] = sha256_file(shadow_trace)
        summary["peak_rss_self_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        summary["peak_rss_children_kib"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
