#!/usr/bin/env python3
"""Run one paired RxR val_seen episode with the MF3 UAD residual adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ETPR1 = ROOT / "third_party/ETP-R1"
RXR_CHECKPOINT = ETPR1 / (
    "data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth"
)
JOINT_PRETRAINED = ETPR1 / (
    "pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/"
    "model_step_367500.pt"
)
TRAIN = ROOT / "artifacts/training/mf3b_uad_online_v1"
GATE = ROOT / (
    "artifacts/evaluation/mf3b_uad_shadow_gate_v1/"
    "MF3B_UAD_SHADOW_GATE.json"
)
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE,
    NativeConditionedUAD,
    PairwiseSwitchUtility,
    PolicyAnchoredTop2UAD,
    StructuredUADHeads,
    current_local_action_indices,
    fuse_current_candidate_logits,
    median_native_conditioned_outputs,
    median_mad_lower_confidence,
    native_alternative_posterior_gain,
    native_residual_logits,
    pairwise_expected_utility,
    top2_conditional_advantage,
    top2_posterior_advantage,
    top2_switch_indices,
)
from revealnav_mf3.action_aligned import (  # noqa: E402
    ActionAlignedReturnGate,
    action_aligned_features,
    hierarchical_proposal_tier,
    residual_with_uncertainty_source,
)
from rxr_unseen_controller_worker import (  # noqa: E402
    install_runtime_shims,
    sha256_file,
)


SEEDS = (20260826, 20260827, 20260828)
MF3G_TRAIN = ROOT / "artifacts/training/mf3g_uad_residual_v1"
MF3G_GATE = ROOT / (
    "artifacts/evaluation/mf3g_uad_residual_shadow_gate_v1/"
    "MF3G_UAD_SHADOW_GATE.json"
)
MF3H_GATE = ROOT / (
    "artifacts/evaluation/mf3h_uad_consensus_shadow_gate_v1/"
    "MF3H_UAD_SHADOW_GATE.json"
)
MF3I_TRAIN = ROOT / "artifacts/training/mf3i_policy_token_uad_v1"
MF3I_GATE = ROOT / (
    "artifacts/evaluation/mf3i_contextual_uad_shadow_gate_v1/"
    "MF3I_UAD_SHADOW_GATE.json"
)
MF3J_TRAIN = ROOT / "artifacts/training/mf3j_switch_utility_v1"
MF3J_GATE = ROOT / (
    "artifacts/evaluation/mf3j_switch_utility_shadow_gate_v1/"
    "MF3J_SHADOW_GATE.json"
)
MF3K_TRAIN = ROOT / "artifacts/training/mf3k_policy_top2_v1"
MF3K_GATE = ROOT / (
    "artifacts/evaluation/mf3k_policy_top2_shadow_gate_v1/"
    "MF3K_SHADOW_GATE.json"
)
MF3L_TRAIN = ROOT / "artifacts/training/mf3l_conditional_top2_v1"
MF3L_GATE = ROOT / (
    "artifacts/evaluation/mf3l_conditional_top2_shadow_gate_v1/"
    "MF3L_SHADOW_GATE.json"
)
MF3M_GATE = ROOT / (
    "artifacts/evaluation/mf3m_robust_top2_shadow_gate_v1/"
    "MF3M_SHADOW_GATE.json"
)
MF3P_TRAIN = ROOT / "artifacts/training/mf3p_rescue_harm_v1"
MF3P_GATE = ROOT / (
    "artifacts/evaluation/mf3p_rescue_harm_shadow_gate_v1/"
    "MF3P_SHADOW_GATE.json"
)
MF3Q_TRAIN = ROOT / "artifacts/training/mf3q_crossfit_v1"
MF3S_GATE = ROOT / (
    "artifacts/evaluation/mf3s_policy_risk_shadow_gate_v1/"
    "MF3S_SHADOW_GATE.json"
)
MF3T_TRAIN = ROOT / "artifacts/training/mf3t_coverage_ranker_v2"
MF3T_GATE = ROOT / (
    "artifacts/evaluation/mf3t_coverage_shadow_gate_v2/"
    "MF3T_SHADOW_GATE.json"
)
MF3V_TRAIN = ROOT / "artifacts/training/mf3v_horizon_ranker_v1"
MF3U_GATE = ROOT / (
    "artifacts/evaluation/mf3u_policy_anchor_shadow_gate_v1/"
    "MF3U_SHADOW_GATE.json"
)
MF3V_GATE = ROOT / (
    "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/"
    "MF3V_SHADOW_GATE.json"
)
MF3Y_TRAIN = MF3V_TRAIN
MF3Y_GATE = ROOT / (
    "artifacts/evaluation/mf3y_consensus_tail_shadow_gate_v1/"
    "MF3Y_SHADOW_GATE.json"
)
MF3Z_TRAIN = MF3V_TRAIN
MF3Z_GATE = ROOT / (
    "artifacts/evaluation/mf3z_adaptive_tail_shadow_gate_v1/"
    "MF3Z_SHADOW_GATE.json"
)
MF3ZA_TRAIN = MF3V_TRAIN
MF3ZA_GATE = ROOT / (
    "artifacts/evaluation/mf3za_consensus_band_shadow_gate_v1/"
    "MF3ZA_SHADOW_GATE.json"
)
MF3ZB_TRAIN = MF3V_TRAIN
MF3ZB_GATE = ROOT / (
    "artifacts/evaluation/mf3zb_temporal_maturity_shadow_gate_v1/"
    "MF3ZB_SHADOW_GATE.json"
)
MF3ZC_TRAIN = MF3V_TRAIN
MF3ZC_GATE = ROOT / (
    "artifacts/evaluation/mf3zc_calibrated_dissent_shadow_gate_v1/"
    "MF3ZC_SHADOW_GATE.json"
)
MF3ZE_TRAIN = MF3V_TRAIN
MF3ZE_GATE = ROOT / (
    "artifacts/training/mf3ze_action_aligned_return_gate_v1/"
    "MF3ZE_CROSSFIT_GATE.json"
)
MF3ZE_MODEL = ROOT / (
    "artifacts/training/mf3ze_action_aligned_return_gate_v1/"
    "MF3ZE_GATE_MODELS.npz"
)
MF3ZF_COLLECTION_GATE = ROOT / (
    "artifacts/training/mf3zf_expanded_collection_v1/"
    "MF3ZF_COLLECTION_GATE.json"
)
MF3ZF_GATE = ROOT / (
    "artifacts/training/mf3zf_action_aligned_return_gate_v1/"
    "MF3ZF_CROSSFIT_GATE.json"
)
MF3ZF_MODEL = ROOT / (
    "artifacts/training/mf3zf_action_aligned_return_gate_v1/"
    "MF3ZF_GATE_MODELS.npz"
)
MF3ZG_GATE = ROOT / (
    "artifacts/training/mf3zg_hierarchical_core_preserving_gate_v1/"
    "MF3ZG_SHADOW_GATE.json"
)
MF3ZH_GATE = ROOT / (
    "artifacts/training/mf3zh_uncertainty_floor_residual_gate_v1/"
    "MF3ZH_SHADOW_GATE.json"
)
install_runtime_shims()
_CONTROLLER = None
_TRAINER = None
_LOCAL_ACTION_IDS: tuple[set[str], ...] = ()
_NO_VP_LEFT: tuple[bool, ...] = ()


def stable_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


class UADResidualController:
    def __init__(self, seed: int, device: torch.device, trace: Path) -> None:
        if seed not in SEEDS:
            raise RuntimeError("unsealed MF3 UAD seed")
        gate = json.loads(GATE.read_text())
        if not (
            gate.get("status") == "SHADOW_GATE_PASS"
            and gate.get("task_metric_run_authorized") is True
        ):
            raise RuntimeError("MF3 UAD shadow gate does not authorize metrics")
        selected = gate["selected"]
        self.parameters = {
            "alpha": float(selected["alpha"]),
            "decisive_threshold": float(selected["decisive_threshold"]),
            "margin_threshold": float(selected["margin_threshold"]),
        }
        checkpoint = TRAIN / f"seed_{seed}/uad_mf3.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3b-uad-checkpoint/1"
            and payload.get("seed") == seed
        ):
            raise RuntimeError("MF3 UAD checkpoint schema drift")
        self.model = StructuredUADHeads(768, int(payload["hidden_dim"]))
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(device).eval()
        self.checkpoint = {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "strict_load": True,
        }
        self.device = device
        self.trace = trace
        self.trace.write_text("")
        self.instruction = None
        self.latest_history = None
        self.rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        self.records = []
        self.previous_hash = "0" * 64
        self.step = 0

    def record_language(self, embedding, mask) -> None:
        self.instruction = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_panorama(self, embedding, mask) -> None:
        self.latest_history = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def navigation(self, kwargs: dict, result: dict) -> dict:
        global _TRAINER
        if self.instruction is None or self.latest_history is None:
            return result
        ids = kwargs["gmap_vp_ids"][0]
        action_mask = [bool(value) for value in kwargs["gmap_masks"][0]]
        visited = [bool(value) for value in kwargs["gmap_visited_masks"][0]]
        current_global = current_local_action_indices(
            ids, action_mask, visited, _LOCAL_ACTION_IDS[0]
        )
        current = {
            str(ids[index]): kwargs["gmap_img_fts"][0, index].detach()
            for index in current_global
        }
        if current:
            self.rows.append((self.latest_history, current))
        native_logits = result["global_logits"]
        native_argmax = int(torch.argmax(native_logits[0]))
        forced_stop = (
            bool(_NO_VP_LEFT[0]) or self.step >= int(_TRAINER.max_len) - 1
        )
        native = 0 if forced_stop else native_argmax
        fused = native_logits
        authorized = False
        decisive = margin = None
        if (
            current and len(current_global) >= 2
            and not forced_stop
        ):
            ordered = tuple(dict.fromkeys(
                identity for _, values in self.rows for identity in values
            ))
            branch_index = {identity: index for index, identity in enumerate(ordered)}
            steps = len(self.rows)
            history = torch.stack([value for value, _ in self.rows]).unsqueeze(0)
            candidates = torch.zeros(
                1, steps, len(ordered), 768, device=self.device
            )
            mask = torch.zeros(
                1, steps, len(ordered), dtype=torch.bool, device=self.device
            )
            for time_index, (_, values) in enumerate(self.rows):
                for identity, value in values.items():
                    index = branch_index[identity]
                    candidates[0, time_index, index] = value
                    mask[0, time_index, index] = True
            with torch.no_grad():
                output = self.model(
                    history, candidates, mask, self.instruction.unsqueeze(0)
                )
            local_indices = torch.tensor(
                [current_global], dtype=torch.long, device=self.device
            )
            target_scores = torch.stack([
                output.target_logits[0, steps - 1, branch_index[str(ids[index])]]
                for index in current_global
            ]).unsqueeze(0)
            decisive_tensor = output.uad_probabilities[0, steps - 1, 2].view(1)
            fusion = fuse_current_candidate_logits(
                native_logits, local_indices, target_scores,
                torch.ones_like(local_indices, dtype=torch.bool),
                decisive_tensor, **self.parameters,
            )
            fused = fusion.logits
            authorized = bool(fusion.authorized[0])
            decisive = float(decisive_tensor[0])
            margin = float(torch.topk(target_scores[0], 2).values.diff().abs()[0])
        adapted = 0 if forced_stop else int(torch.argmax(fused[0]))
        row = {
            "schema_version": "revealnav-mf3b-uad-policy-decision/1",
            "step": self.step,
            "native_action_index": native,
            "adapted_action_index": adapted,
            "native_action_id": None if native == 0 else str(ids[native]),
            "adapted_action_id": None if adapted == 0 else str(ids[adapted]),
            "current_local_action_ids": [str(ids[index]) for index in current_global],
            "authorized": authorized,
            "action_changed": adapted != native,
            "p_decisive": decisive,
            "target_margin": margin,
            "parameters": self.parameters,
            "previous_hash": self.previous_hash,
            **MF3B_SCOPE,
        }
        row["record_hash"] = stable_hash(row)
        self.previous_hash = row["record_hash"]
        self.records.append(row)
        with self.trace.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.step += 1
        if fused is native_logits:
            return result
        updated = dict(result)
        updated["global_logits"] = fused
        return updated


class MF3GResidualController:
    def __init__(
        self, seed: int | None, device: torch.device, trace: Path,
        *, gate_path: Path = MF3G_GATE, require_unanimous: bool = False,
        revision: str = "mf3g",
    ) -> None:
        if seed is not None and seed not in SEEDS:
            raise RuntimeError("unsealed MF3G UAD seed")
        gate = json.loads(gate_path.read_text())
        if not (
            gate.get("status") == "SHADOW_GATE_PASS"
            and gate.get("task_metric_run_authorized") is True
        ):
            raise RuntimeError("MF3G UAD shadow gate does not authorize metrics")
        selected = gate["selected"]
        if revision == "mf3i":
            self.parameters = {
                "posterior_gain_threshold": float(
                    selected["posterior_gain_threshold"]
                )
            }
        else:
            self.parameters = {
                "error_threshold": float(selected["error_threshold"]),
                "fused_advantage_threshold": float(
                    selected["fused_advantage_threshold"]
                ),
            }
        self.revision = revision
        self.policy_fusion_features = revision == "mf3i"
        self.require_unanimous = require_unanimous
        self.models = []
        checkpoints = []
        for member_seed in (SEEDS if seed is None else (seed,)):
            checkpoint = (
                MF3I_TRAIN / f"seed_{member_seed}/uad_contextual_mf3i.pt"
                if revision == "mf3i"
                else MF3G_TRAIN / f"seed_{member_seed}/uad_residual_mf3g.pt"
            )
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            expected_schema = (
                "revealnav-mf3i-uad-checkpoint/1"
                if revision == "mf3i"
                else "revealnav-mf3g-uad-checkpoint/1"
            )
            expected_bound = 1.0 if revision == "mf3i" else 2.0
            if not (
                payload.get("schema_version") == expected_schema
                and payload.get("seed") == member_seed
                and float(payload.get("correction_bound")) == expected_bound
                and (
                    revision != "mf3i"
                    or (
                        payload.get("feature")
                        == "frozen_etp_r1_policy_fusion_token"
                        and int(payload.get("candidate_feature_dim")) == 1536
                    )
                )
            ):
                raise RuntimeError("MF3G UAD checkpoint schema drift")
            model = NativeConditionedUAD(
                768,
                int(payload["hidden_dim"]),
                candidate_feature_dim=(1536 if revision == "mf3i" else 768),
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            self.models.append(model.to(device).eval())
            checkpoints.append({
                "path": str(checkpoint.relative_to(ROOT)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "strict_load": True,
            })
        self.correction_bound = 1.0 if revision == "mf3i" else 2.0
        self.checkpoint = checkpoints[0] if seed is not None else checkpoints
        self.device = device
        self.trace = trace
        self.trace.write_text("")
        self.instruction = None
        self.latest_history = None
        self.rows = []
        self.records = []
        self.previous_hash = "0" * 64
        self.step = 0

    def record_language(self, embedding, mask) -> None:
        self.instruction = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_panorama(self, embedding, mask) -> None:
        self.latest_history = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def navigation(self, kwargs: dict, result: dict) -> dict:
        global _TRAINER
        if self.instruction is None or self.latest_history is None:
            return result
        ids = kwargs["gmap_vp_ids"][0]
        action_mask = [bool(value) for value in kwargs["gmap_masks"][0]]
        visited = [bool(value) for value in kwargs["gmap_visited_masks"][0]]
        current_global = current_local_action_indices(
            ids, action_mask, visited, _LOCAL_ACTION_IDS[0]
        )
        candidate_features = (
            kwargs["_mf3_policy_fusion_features"]
            if self.policy_fusion_features
            else kwargs["gmap_img_fts"]
        )
        if candidate_features.shape[:2] != kwargs["gmap_img_fts"].shape[:2]:
            raise RuntimeError("MF3 policy fusion feature shape drift")
        current = {
            str(ids[index]): candidate_features[0, index].detach()
            for index in current_global
        }
        native_logits = result["global_logits"]
        native_argmax = int(torch.argmax(native_logits[0]))
        forced_stop = (
            bool(_NO_VP_LEFT[0]) or self.step >= int(_TRAINER.max_len) - 1
        )
        native = 0 if forced_stop else native_argmax
        native_id = (
            str(ids[native]) if native in current_global else None
        )
        if current:
            self.rows.append({
                "history": self.latest_history,
                "candidates": current,
                "scores": {
                    str(ids[index]): native_logits[0, index].detach()
                    for index in current_global
                },
                "native_id": native_id,
            })

        fused_global = native_logits
        authorized = False
        error_probability = fused_advantage = None
        member_adapted = []
        member_posterior_gains = []
        adapted = native
        if current and len(current_global) >= 2 and native_id is not None and not forced_stop:
            ordered = tuple(dict.fromkeys(
                identity for row in self.rows for identity in row["candidates"]
            ))
            branch_index = {identity: index for index, identity in enumerate(ordered)}
            steps = len(self.rows)
            history = torch.stack([
                row["history"] for row in self.rows
            ]).unsqueeze(0)
            candidates = torch.zeros(
                1, steps, len(ordered), 768, device=self.device
            )
            mask = torch.zeros(
                1, steps, len(ordered), dtype=torch.bool, device=self.device
            )
            scores = torch.full(
                (1, steps, len(ordered)), -torch.inf, device=self.device
            )
            native_indices = torch.full(
                (1, steps), -1, dtype=torch.long, device=self.device
            )
            for time_index, row in enumerate(self.rows):
                for identity, value in row["candidates"].items():
                    index = branch_index[identity]
                    candidates[0, time_index, index] = value
                    mask[0, time_index, index] = True
                    scores[0, time_index, index] = row["scores"][identity]
                if row["native_id"] is not None:
                    native_indices[0, time_index] = branch_index[row["native_id"]]
            with torch.no_grad():
                member_outputs = tuple(
                    model(
                        history, candidates, mask,
                        self.instruction.unsqueeze(0), scores, native_indices,
                    )
                    for model in self.models
                )
                output = median_native_conditioned_outputs(member_outputs)
                fused, _ = native_residual_logits(
                    output, scores, mask,
                    correction_bound=self.correction_bound,
                )
            time_index = steps - 1
            candidate_adapted = int(torch.argmax(fused[0, time_index]))
            candidate_native = int(native_indices[0, time_index])
            member_adapted = []
            for member_output in member_outputs:
                member_fused, _ = native_residual_logits(
                    member_output, scores, mask,
                    correction_bound=self.correction_bound,
                )
                member_adapted.append(int(torch.argmax(
                    member_fused[0, time_index]
                )))
                step_output = type(member_output)(
                    native_error_logit=member_output.native_error_logit[
                        :, time_index:time_index + 1
                    ],
                    alternative_logits=member_output.alternative_logits[
                        :, time_index:time_index + 1
                    ],
                )
                member_index = torch.tensor(
                    [[member_adapted[-1]]], dtype=torch.long,
                    device=self.device,
                )
                member_posterior_gains.append(float(
                    native_alternative_posterior_gain(
                        step_output, member_index
                    )[0, 0]
                ))
            unanimous = (
                len(set(member_adapted)) == 1
                and member_adapted[0] != candidate_native
            )
            error_probability = float(torch.sigmoid(
                output.native_error_logit[0, time_index]
            ))
            fused_advantage = max(
                0.0, float(
                    fused[0, time_index, candidate_adapted]
                    - fused[0, time_index, candidate_native]
                )
            )
            authorized = (
                candidate_adapted != candidate_native
                and (not self.require_unanimous or unanimous)
            )
            if self.revision == "mf3i":
                authorized = authorized and min(member_posterior_gains) > (
                    self.parameters["posterior_gain_threshold"]
                )
            else:
                authorized = (
                    authorized
                    and error_probability >= self.parameters["error_threshold"]
                    and fused_advantage
                    >= self.parameters["fused_advantage_threshold"]
                )
            if authorized:
                fused_global = native_logits.clone()
                for index in current_global:
                    identity = str(ids[index])
                    fused_global[0, index] = fused[
                        0, time_index, branch_index[identity]
                    ]
                adapted = int(torch.argmax(fused_global[0]))
                expected = branch_index[str(ids[adapted])]
                if expected != candidate_adapted:
                    raise RuntimeError("MF3G local/global fused action drift")
        row = {
            "schema_version": f"revealnav-{self.revision}-uad-policy-decision/1",
            "step": self.step,
            "native_action_index": native,
            "adapted_action_index": adapted,
            "native_action_id": None if native == 0 else str(ids[native]),
            "adapted_action_id": None if adapted == 0 else str(ids[adapted]),
            "current_local_action_ids": [str(ids[index]) for index in current_global],
            "authorized": authorized, "action_changed": adapted != native,
            "native_error_probability": error_probability,
            "fused_advantage": fused_advantage,
            "correction_bound": self.correction_bound,
            "member_candidate_indices": member_adapted,
            "member_posterior_gains": member_posterior_gains,
            "unanimous_required": self.require_unanimous,
            "parameters": self.parameters,
            "previous_hash": self.previous_hash,
            **MF3B_SCOPE,
        }
        row["record_hash"] = stable_hash(row)
        self.previous_hash = row["record_hash"]
        self.records.append(row)
        with self.trace.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.step += 1
        if fused_global is native_logits:
            return result
        updated = dict(result)
        updated["global_logits"] = fused_global
        return updated


class MF3JSwitchController:
    def __init__(self, device: torch.device, trace: Path) -> None:
        gate = json.loads(MF3J_GATE.read_text())
        if not (
            gate.get("status") == "SHADOW_GATE_PASS"
            and gate.get("task_metric_run_authorized") is True
        ):
            raise RuntimeError("MF3J shadow gate does not authorize metrics")
        hidden = int(gate["selected_architecture"]["hidden_dim"])
        rule = gate["selected_rule"]
        self.parameters = {
            "agreement": str(rule["agreement"]),
            "threshold": float(rule["threshold"]),
        }
        self.policy_fusion_features = True
        self.models = []
        checkpoints = []
        for seed in SEEDS:
            checkpoint = MF3J_TRAIN / (
                f"hidden_{hidden}/seed_{seed}/switch_utility_mf3j.pt"
            )
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            if not (
                payload.get("schema_version") == "revealnav-mf3j-checkpoint/1"
                and payload.get("hidden_dim") == hidden
                and payload.get("seed") == seed
                and payload.get("candidate_feature_dim") == 1536
            ):
                raise RuntimeError("MF3J checkpoint schema drift")
            model = PairwiseSwitchUtility(768, 1536, hidden)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            self.models.append(model.to(device).eval())
            checkpoints.append({
                "path": str(checkpoint.relative_to(ROOT)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "strict_load": True,
            })
        self.checkpoint = checkpoints
        self.device = device
        self.trace = trace
        self.trace.write_text("")
        self.instruction = None
        self.latest_history = None
        self.rows = []
        self.records = []
        self.previous_hash = "0" * 64
        self.step = 0

    def record_language(self, embedding, mask) -> None:
        self.instruction = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_panorama(self, embedding, mask) -> None:
        self.latest_history = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def navigation(self, kwargs: dict, result: dict) -> dict:
        global _TRAINER
        if self.instruction is None or self.latest_history is None:
            return result
        ids = kwargs["gmap_vp_ids"][0]
        action_mask = [bool(value) for value in kwargs["gmap_masks"][0]]
        visited = [bool(value) for value in kwargs["gmap_visited_masks"][0]]
        current_global = current_local_action_indices(
            ids, action_mask, visited, _LOCAL_ACTION_IDS[0]
        )
        policy_features = kwargs["_mf3_policy_fusion_features"]
        if policy_features.shape[:2] != kwargs["gmap_img_fts"].shape[:2]:
            raise RuntimeError("MF3J policy-token shape drift")
        current = {
            str(ids[index]): policy_features[0, index].detach()
            for index in current_global
        }
        native_logits = result["global_logits"]
        native_argmax = int(torch.argmax(native_logits[0]))
        forced_stop = (
            bool(_NO_VP_LEFT[0]) or self.step >= int(_TRAINER.max_len) - 1
        )
        native = 0 if forced_stop else native_argmax
        native_id = str(ids[native]) if native in current_global else None
        if current:
            self.rows.append({
                "history": self.latest_history,
                "candidates": current,
                "scores": {
                    str(ids[index]): native_logits[0, index].detach()
                    for index in current_global
                },
                "native_id": native_id,
            })

        adapted = native
        authorized = False
        minimum_utility = median_utility = None
        member_candidates = []
        fused = native_logits
        if (
            current and len(current_global) >= 2 and native_id is not None
            and not forced_stop
        ):
            ordered = tuple(dict.fromkeys(
                identity for row in self.rows for identity in row["candidates"]
            ))
            branch_index = {identity: index for index, identity in enumerate(ordered)}
            steps = len(self.rows)
            history = torch.stack([
                row["history"] for row in self.rows
            ]).unsqueeze(0)
            candidates = torch.zeros(
                1, steps, len(ordered), 1536, device=self.device
            )
            mask = torch.zeros(
                1, steps, len(ordered), dtype=torch.bool, device=self.device
            )
            scores = torch.full(
                (1, steps, len(ordered)), -torch.inf, device=self.device
            )
            native_indices = torch.full(
                (1, steps), -1, dtype=torch.long, device=self.device
            )
            for time_index, row in enumerate(self.rows):
                for identity, value in row["candidates"].items():
                    index = branch_index[identity]
                    candidates[0, time_index, index] = value
                    mask[0, time_index, index] = True
                    scores[0, time_index, index] = row["scores"][identity]
                if row["native_id"] is not None:
                    native_indices[0, time_index] = branch_index[row["native_id"]]
            with torch.no_grad():
                utilities = tuple(pairwise_expected_utility(model(
                    history, candidates, mask, self.instruction.unsqueeze(0),
                    scores, native_indices,
                )) for model in self.models)
            time_index = steps - 1
            candidate_native = int(native_indices[0, time_index])
            valid = mask[0, time_index].clone()
            valid[candidate_native] = False
            member_candidates = [int(
                value[0, time_index].masked_fill(~valid, -torch.inf).argmax()
            ) for value in utilities]
            median_scores = torch.stack([
                value[0, time_index] for value in utilities
            ]).median(0).values.masked_fill(~valid, -torch.inf)
            candidate_adapted = int(median_scores.argmax())
            member_values = [float(
                value[0, time_index, candidate_adapted]
            ) for value in utilities]
            minimum_utility = min(member_values)
            median_utility = sorted(member_values)[1]
            unanimous = (
                len(set(member_candidates)) == 1
                and member_candidates[0] == candidate_adapted
            )
            authorized = (
                minimum_utility > self.parameters["threshold"]
                and (
                    self.parameters["agreement"] == "median" or unanimous
                )
            )
            if authorized:
                adapted_id = ordered[candidate_adapted]
                adapted = ids.index(adapted_id)
                fused = native_logits.clone()
                fused[0, adapted] = native_logits[0, native] + 1e-4
                if int(torch.argmax(fused[0])) != adapted:
                    raise RuntimeError("MF3J declared/global action drift")
        row = {
            "schema_version": "revealnav-mf3j-policy-decision/1",
            "step": self.step,
            "native_action_index": native,
            "adapted_action_index": adapted,
            "native_action_id": None if native == 0 else str(ids[native]),
            "adapted_action_id": None if adapted == 0 else str(ids[adapted]),
            "current_local_action_ids": [str(ids[index]) for index in current_global],
            "authorized": authorized,
            "action_changed": adapted != native,
            "minimum_expected_utility": minimum_utility,
            "median_expected_utility": median_utility,
            "member_candidate_indices": member_candidates,
            "parameters": self.parameters,
            "previous_hash": self.previous_hash,
            **MF3B_SCOPE,
        }
        row["record_hash"] = stable_hash(row)
        self.previous_hash = row["record_hash"]
        self.records.append(row)
        with self.trace.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.step += 1
        if fused is native_logits:
            return result
        updated = dict(result)
        updated["global_logits"] = fused
        return updated


class MF3KTop2Controller:
    """Conservative policy-anchored switch to the frozen runner-up only."""

    def __init__(
        self, device: torch.device, trace: Path, *, revision: str = "mf3k"
    ) -> None:
        if revision not in ("mf3k", "mf3l", "mf3m", "mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"):
            raise ValueError("unsupported policy-anchored Top-2 revision")
        self.revision = revision
        gate_path = (
            MF3ZH_GATE if revision == "mf3zh" else
            MF3ZG_GATE if revision == "mf3zg" else
            MF3ZF_COLLECTION_GATE if revision == "mf3zf" else
            MF3V_GATE if revision == "mf3ze" else
            MF3ZC_GATE if revision == "mf3zc" else
            MF3ZB_GATE if revision == "mf3zb" else
            MF3ZA_GATE if revision == "mf3za" else
            MF3Z_GATE if revision == "mf3z" else
            MF3Y_GATE if revision == "mf3y" else
            MF3V_GATE if revision == "mf3v" else
            MF3U_GATE if revision == "mf3u" else
            MF3T_GATE if revision == "mf3t"
            else MF3S_GATE if revision == "mf3s"
            else MF3P_GATE if revision == "mf3p"
            else MF3M_GATE if revision == "mf3m"
            else MF3L_GATE if revision == "mf3l"
            else MF3K_GATE
        )
        train_root = (
            MF3V_TRAIN if revision in ("mf3zf", "mf3zg", "mf3zh") else
            MF3ZE_TRAIN if revision == "mf3ze" else
            MF3ZC_TRAIN if revision == "mf3zc" else
            MF3ZB_TRAIN if revision == "mf3zb" else
            MF3ZA_TRAIN if revision == "mf3za" else
            MF3Z_TRAIN if revision == "mf3z" else
            MF3Y_TRAIN if revision == "mf3y" else
            MF3V_TRAIN if revision == "mf3v" else
            MF3T_TRAIN if revision in ("mf3t", "mf3u") else
            MF3Q_TRAIN if revision == "mf3s" else
            MF3P_TRAIN if revision == "mf3p" else
            MF3L_TRAIN if revision in ("mf3l", "mf3m") else MF3K_TRAIN
        )
        gate = json.loads(gate_path.read_text())
        collection_only = (
            revision == "mf3zf"
            and os.environ.get("REVEALNAV_MF3ZF_COLLECTION_ONLY") == "1"
        )
        self.collection_only = collection_only
        valid_gate = (
            gate.get("status") == "TRAIN_RETURN_COLLECTION_AUTHORIZED"
            and gate.get("task_metric_run_authorized") is False
            and gate.get("unseen_or_test_read") is False
        ) if revision == "mf3zf" else (
            gate.get("status") == "SHADOW_GATE_PASS"
            and gate.get("task_metric_run_authorized") is True
        )
        if not valid_gate:
            raise RuntimeError(f"{revision.upper()} shadow gate does not authorize metrics")
        rule = gate["selected_rule"]
        architecture = gate.get("selected_architecture")
        # MF3T seals its hidden size in the selected coverage-ranking rule
        # rather than duplicating an architecture object in the gate.
        if architecture is None and revision in ("mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"):
            architecture = {"hidden_dim": int(rule["hidden_dim"])}
        if architecture is None:
            raise RuntimeError(f"{revision.upper()} gate is missing architecture")
        hidden = int(architecture["hidden_dim"])
        bound = float(architecture.get("correction_bound", 0.0))
        self.parameters = (
            {
                "mad_weight": float(rule["mad_weight"]),
                "policy_risk_beta": float(rule["policy_risk_beta"]),
                "score_threshold": float(rule["final_training_threshold"]),
                "persistence_steps": int(rule["persistence_steps"]),
                **({"score_upper_threshold": float(rule["score_upper_threshold"])}
                   if revision in ("mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh") else {}),
                **({
                    "consensus_mad_threshold": float(rule["consensus_mad_threshold"]),
                    "consensus_margin_threshold": float(rule["consensus_margin_threshold"]),
                } if revision == "mf3y" else {}),
                **({
                    "consensus_mad_threshold": float(rule["consensus_mad_threshold"]),
                    "consensus_relative_margin_threshold": float(rule["consensus_relative_margin_threshold"]),
                } if revision == "mf3z" else {}),
                **({
                    "consensus_mad_floor_threshold": float(rule["consensus_mad_floor_threshold"]),
                    "consensus_mad_threshold": float(rule["consensus_mad_threshold"]),
                    "consensus_relative_margin_threshold": float(rule["consensus_relative_margin_threshold"]),
                } if revision == "mf3za" else {}),
                **({
                    "minimum_decision_step": int(rule["minimum_decision_step"]),
                } if revision == "mf3zb" else {}),
                **({
                    "cold_start_steps": int(rule["cold_start_steps"]),
                    "cold_start_floor_ratio_threshold": float(
                        rule["cold_start_floor_ratio_threshold"]
                    ),
                    "cold_start_relative_mad_threshold": float(
                        rule["cold_start_relative_mad_threshold"]
                    ),
                } if revision == "mf3zc" else {}),
            }
            if revision in ("mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh")
            else {
                "mad_weight": float(rule["mad_weight"]),
                "policy_risk_beta": float(rule["policy_risk_beta"]),
                "score_threshold": float(rule["final_training_threshold"]),
                "persistence_steps": int(rule["persistence_steps"]),
            }
            if revision == "mf3s"
            else {
                "mad_weight": float(rule["mad_weight"]),
                "logit_threshold": float(rule["logit_threshold"]),
                "persistence_steps": int(rule["persistence_steps"]),
            }
            if revision == "mf3p"
            else {
                "mad_weight": float(rule["mad_weight"]),
                "robust_advantage_threshold": float(
                    rule["robust_advantage_threshold"]
                ),
            }
            if revision == "mf3m"
            else {"conditional_advantage_threshold": float(
                rule["conditional_advantage_threshold"]
            )}
            if revision == "mf3l"
            else {
                "utility_threshold": float(rule["utility_threshold"]),
                "native_margin_max": float(rule["native_margin_max"]),
            }
        )
        self.policy_fusion_features = True
        self.models = []
        checkpoints = []
        architecture_name = (
            "fold_final" if revision in ("mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh") else
            f"hidden_{hidden}/fold_final" if revision in ("mf3t", "mf3u") else
            "fold_final" if revision == "mf3s" else
            f"hidden_{hidden}" if revision == "mf3p" else
            f"hidden_{hidden}_bound_{str(bound).replace('.', 'p')}"
        )
        for seed in SEEDS:
            checkpoint = train_root / architecture_name / f"seed_{seed}" / (
                "horizon_ranker_mf3v.pt" if revision in ("mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh") else
                "coverage_ranker_mf3t.pt" if revision in ("mf3t", "mf3u")
                else "crossfit_mf3q.pt" if revision == "mf3s"
                else "rescue_harm_mf3p.pt" if revision == "mf3p"
                else "conditional_top2_mf3l.pt" if revision in ("mf3l", "mf3m")
                else "policy_top2_mf3k.pt"
            )
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            if not (
                payload.get("schema_version")
                == ("revealnav-mf3v-checkpoint/1" if revision in ("mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh")
                    else "revealnav-mf3t-checkpoint/2" if revision in ("mf3t", "mf3u")
                    else "revealnav-mf3q-checkpoint/1" if revision == "mf3s"
                    else "revealnav-mf3p-checkpoint/1" if revision == "mf3p"
                    else "revealnav-mf3l-checkpoint/1" if revision == "mf3m"
                    else f"revealnav-{revision}-checkpoint/1")
                and payload.get("hidden_dim") == hidden
                and (
                    revision in ("mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh")
                    or float(payload.get("correction_bound")) == bound
                )
                and payload.get("seed") == seed
                and (
                    revision not in ("mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh")
                    or (
                        payload.get("fold") == "final"
                        and payload.get("optimizer_steps") == (800 if revision in ("mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh") else 200)
                    )
                )
                and (
                    revision in ("mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh")
                    or payload.get("candidate_feature_dim") == 1536
                )
            ):
                raise RuntimeError(f"{revision.upper()} checkpoint schema drift")
            model = (
                PairwiseSwitchUtility(768, 1536, hidden)
                if revision in ("mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh")
                else PolicyAnchoredTop2UAD(768, 1536, hidden, bound)
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            self.models.append(model.to(device).eval())
            checkpoints.append({
                "path": str(checkpoint.relative_to(ROOT)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "strict_load": True,
            })
        self.checkpoint = checkpoints
        self.device = device
        self.trace = trace
        self.trace.write_text("")
        self.instruction = None
        self.latest_history = None
        self.rows = []
        self.records = []
        self.previous_hash = "0" * 64
        self.step = 0
        self.persistence_count = 0
        self.intervened = False
        self.return_gate = None
        self.core_return_gate = None
        self.expansion_return_gate = None
        if self.revision == "mf3ze" or (
            self.revision == "mf3zf" and not collection_only
        ):
            safety_path = MF3ZF_GATE if self.revision == "mf3zf" else MF3ZE_GATE
            model_path = MF3ZF_MODEL if self.revision == "mf3zf" else MF3ZE_MODEL
            safety = json.loads(safety_path.read_text())
            model_evidence = safety.get("model", {})
            if not (
                safety.get("status") == "SHADOW_GATE_PASS"
                and safety.get("task_metric_run_authorized") is True
                and safety.get("controls", {}).get("unseen_or_test_read") is False
                and model_evidence.get("path") == str(model_path.relative_to(ROOT))
                and model_evidence.get("bytes") == model_path.stat().st_size
                and model_evidence.get("sha256") == sha256_file(model_path)
            ):
                raise RuntimeError(
                    f"{self.revision.upper()} action-aligned safety gate drift"
                )
            self.return_gate = ActionAlignedReturnGate(
                model_path, safety["selected_rule"]
            )
            self.parameters.update({
                "return_threshold": float(
                    safety["selected_rule"]["return_threshold"]
                ),
                "harm_probability_threshold": float(
                    safety["selected_rule"]["harm_probability_threshold"]
                ),
            })
            self.checkpoint.append({
                "path": str(model_path.relative_to(ROOT)),
                "bytes": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
                "strict_load": True,
            })
        if self.revision in ("mf3zg", "mf3zh"):
            hierarchy = gate.get("hierarchy", {})
            if not (
                hierarchy.get("rejected_expansion_consumes_core_opportunity")
                is False
                and hierarchy.get("maximum_executed_switches_per_episode") == 1
                and float(hierarchy["expansion_score_threshold"])
                == self.parameters["score_threshold"]
                and float(hierarchy["score_upper_threshold"])
                == self.parameters["score_upper_threshold"]
                and float(hierarchy["core_score_threshold"])
                > self.parameters["score_threshold"]
            ):
                raise RuntimeError("MF3ZG hierarchy drift")
            for tier, safety_path, model_path, rule in (
                ("core", MF3ZE_GATE, MF3ZE_MODEL, None),
                ("expansion", MF3ZF_GATE, MF3ZF_MODEL, {
                    "return_threshold": float(
                        hierarchy["expansion_return_threshold"]
                    ),
                    "harm_probability_threshold": float(
                        hierarchy["expansion_harm_probability_threshold"]
                    ),
                }),
            ):
                safety = json.loads(safety_path.read_text())
                model_evidence = safety.get("model", {})
                if not (
                    safety.get("status") == "SHADOW_GATE_PASS"
                    and safety.get("task_metric_run_authorized") is True
                    and safety.get("controls", {}).get("unseen_or_test_read") is False
                    and model_evidence.get("path")
                    == str(model_path.relative_to(ROOT))
                    and model_evidence.get("bytes") == model_path.stat().st_size
                    and model_evidence.get("sha256") == sha256_file(model_path)
                ):
                    raise RuntimeError(f"MF3ZG {tier} safety gate drift")
                loaded = ActionAlignedReturnGate(
                    model_path, safety["selected_rule"] if rule is None else rule
                )
                if tier == "core":
                    self.core_return_gate = loaded
                else:
                    self.expansion_return_gate = loaded
                self.checkpoint.append({
                    "path": str(model_path.relative_to(ROOT)),
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                    "strict_load": True,
                    "tier": tier,
                })
            self.parameters.update({
                "core_score_threshold": float(
                    hierarchy["core_score_threshold"]
                ),
                "expansion_return_threshold": float(
                    hierarchy["expansion_return_threshold"]
                ),
                "expansion_harm_probability_threshold": float(
                    hierarchy["expansion_harm_probability_threshold"]
                ),
            })
            if self.revision == "mf3zh":
                composition = gate.get("composition", {})
                if not (
                    composition.get("learned_residual_priority") is True
                    and composition.get(
                        "uncertainty_actions_consume_learned_budget"
                    ) is False
                    and composition.get(
                        "uncertainty_floor_retains_original_multi_step_behavior"
                    ) is True
                ):
                    raise RuntimeError("MF3ZH composition drift")
                self.parameters["uncertainty_native_margin_max"] = float(
                    gate["exact_budget_control"]["native_margin_max"]
                )
        self.proposal_evaluated = False
        self.core_proposal_evaluated = False
        self.expansion_proposal_evaluated = False

    def write_intervention_feature(
        self, native_feature: torch.Tensor, alternative_feature: torch.Tensor,
    ) -> None:
        raw = os.environ.get("REVEALNAV_MF3_INTERVENTION_FEATURE")
        if raw is None:
            return
        path = Path(raw).resolve()
        if ROOT not in path.parents or path.exists() or path.is_symlink():
            raise RuntimeError("unsafe or duplicate MF3 intervention feature path")
        part = path.with_name(path.name + ".part")
        if part.exists() or part.is_symlink():
            raise RuntimeError("stale MF3 intervention feature partial")
        arrays = {
            "instruction": self.instruction.detach().cpu().float().numpy(),
            "checkpoint": self.latest_history.detach().cpu().float().numpy(),
            "native": native_feature.detach().cpu().float().numpy(),
            "alternative": alternative_feature.detach().cpu().float().numpy(),
        }
        if not all(value.shape == (768,) and np.isfinite(value).all()
                   for value in arrays.values()):
            raise RuntimeError("MF3 intervention feature shape or value drift")
        with part.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(part, path)

    def record_language(self, embedding, mask) -> None:
        self.instruction = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_panorama(self, embedding, mask) -> None:
        self.latest_history = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def navigation(self, kwargs: dict, result: dict) -> dict:
        global _TRAINER
        if self.instruction is None or self.latest_history is None:
            return result
        ids = kwargs["gmap_vp_ids"][0]
        action_mask = [bool(value) for value in kwargs["gmap_masks"][0]]
        visited = [bool(value) for value in kwargs["gmap_visited_masks"][0]]
        current_global = current_local_action_indices(
            ids, action_mask, visited, _LOCAL_ACTION_IDS[0]
        )
        policy_features = kwargs["_mf3_policy_fusion_features"]
        if policy_features.shape[:2] != kwargs["gmap_img_fts"].shape[:2]:
            raise RuntimeError(f"{self.revision.upper()} policy-token shape drift")
        current = {
            str(ids[index]): policy_features[0, index].detach()
            for index in current_global
        }
        native_logits = result["global_logits"]
        native_argmax = int(torch.argmax(native_logits[0]))
        forced_stop = (
            bool(_NO_VP_LEFT[0]) or self.step >= int(_TRAINER.max_len) - 1
        )
        native = 0 if forced_stop else native_argmax
        native_id = str(ids[native]) if native in current_global else None
        if current:
            self.rows.append({
                "history": self.latest_history,
                "candidates": current,
                "scores": {
                    str(ids[index]): native_logits[0, index].detach()
                    for index in current_global
                },
                "native_id": native_id,
            })

        adapted = native
        authorized = False
        minimum_advantage = median_advantage = robust_advantage = native_margin = None
        ensemble_mad = None
        cold_start_floor_ratio = cold_start_relative_mad = None
        policy_risk_adjusted_score = None
        return_gate_evidence = None
        decision_source = None
        fused = native_logits
        runner_local = None
        if (
            current and len(current_global) >= 2 and native_id is not None
            and not forced_stop
        ):
            ordered = tuple(dict.fromkeys(
                identity for row in self.rows for identity in row["candidates"]
            ))
            branch_index = {identity: index for index, identity in enumerate(ordered)}
            steps = len(self.rows)
            history = torch.stack([
                row["history"] for row in self.rows
            ]).unsqueeze(0)
            candidates = torch.zeros(
                1, steps, len(ordered), 1536, device=self.device
            )
            mask = torch.zeros(
                1, steps, len(ordered), dtype=torch.bool, device=self.device
            )
            scores = torch.full(
                (1, steps, len(ordered)), -torch.inf, device=self.device
            )
            native_indices = torch.full(
                (1, steps), -1, dtype=torch.long, device=self.device
            )
            for time_index, row in enumerate(self.rows):
                for identity, value in row["candidates"].items():
                    index = branch_index[identity]
                    candidates[0, time_index, index] = value
                    mask[0, time_index, index] = True
                    scores[0, time_index, index] = row["scores"][identity]
                if row["native_id"] is not None:
                    native_indices[0, time_index] = branch_index[row["native_id"]]
            with torch.no_grad():
                member = []
                for model in self.models:
                    output = model(
                        history, candidates, mask,
                        self.instruction.unsqueeze(0), scores, native_indices,
                    )
                    if self.revision in ("mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"):
                        runner, valid = top2_switch_indices(
                            scores, mask, native_indices
                        )
                        pair_logits = output.outcome_logits.gather(
                            2,
                            runner[..., None, None].expand(
                                *runner.shape, 1,
                                output.outcome_logits.shape[-1],
                            ),
                        ).squeeze(2)
                        value = pair_logits[..., 1] - pair_logits[..., 2]
                        member.append((value, runner, valid))
                    else:
                        advantage_function = (
                            top2_conditional_advantage
                            if self.revision in ("mf3l", "mf3m")
                            else top2_posterior_advantage
                        )
                        member.append(advantage_function(
                            output, scores, mask, native_indices,
                        ))
                member = tuple(member)
            time_index = steps - 1
            runners = [int(value[1][0, time_index]) for value in member]
            if len(set(runners)) != 1:
                raise RuntimeError(
                    f"{self.revision.upper()} frozen runner-up disagreement"
                )
            runner_local = runners[0]
            values = [float(value[0][0, time_index]) for value in member]
            minimum_advantage = min(values)
            median_advantage = statistics.median(values)
            if self.revision in ("mf3m", "mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"):
                robust_advantage = float(median_mad_lower_confidence(
                    torch.tensor(values, device=self.device),
                    mad_weight=self.parameters["mad_weight"],
                ))
            ensemble_mad = statistics.median(
                abs(value - median_advantage) for value in values
            )
            consensus_denominator = max(abs(median_advantage), 1e-6)
            cold_start_floor_ratio = minimum_advantage / consensus_denominator
            cold_start_relative_mad = ensemble_mad / consensus_denominator
            candidate_native = int(native_indices[0, time_index])
            native_margin = float(
                scores[0, time_index, candidate_native]
                - scores[0, time_index, runner_local]
            )
            if self.revision in ("mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"):
                policy_risk_adjusted_score = (
                    robust_advantage
                    - self.parameters["policy_risk_beta"]
                    * math.log1p(max(0.0, native_margin))
                )
                if self.revision in ("mf3u", "mf3v", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"):
                    upper_ok = policy_risk_adjusted_score <= self.parameters["score_upper_threshold"]
                elif self.revision == "mf3y":
                    upper_ok = (
                        policy_risk_adjusted_score <= self.parameters["score_upper_threshold"]
                        or (
                            ensemble_mad <= self.parameters["consensus_mad_threshold"]
                            and native_margin <= self.parameters["consensus_margin_threshold"]
                        )
                    )
                elif self.revision == "mf3z":
                    upper_ok = (
                        policy_risk_adjusted_score <= self.parameters["score_upper_threshold"]
                        or (
                            ensemble_mad <= self.parameters["consensus_mad_threshold"]
                            and native_margin / max(policy_risk_adjusted_score, 1e-6)
                            <= self.parameters["consensus_relative_margin_threshold"]
                        )
                    )
                elif self.revision == "mf3za":
                    upper_ok = (
                        policy_risk_adjusted_score <= self.parameters["score_upper_threshold"]
                        or (
                            self.parameters["consensus_mad_floor_threshold"]
                            <= ensemble_mad
                            <= self.parameters["consensus_mad_threshold"]
                            and native_margin / max(policy_risk_adjusted_score, 1e-6)
                            <= self.parameters["consensus_relative_margin_threshold"]
                        )
                    )
                else:
                    upper_ok = True
                cold_start_ok = True
                if self.revision == "mf3zc" and self.step < self.parameters["cold_start_steps"]:
                    cold_start_ok = (
                        cold_start_floor_ratio
                        <= self.parameters["cold_start_floor_ratio_threshold"]
                        and cold_start_relative_mad
                        <= self.parameters["cold_start_relative_mad_threshold"]
                    )
                if self.revision in ("mf3zg", "mf3zh"):
                    proposal_tier = hierarchical_proposal_tier(
                        policy_risk_adjusted_score,
                        self.parameters["score_threshold"],
                        self.parameters["core_score_threshold"],
                        self.parameters["score_upper_threshold"],
                        core_evaluated=self.core_proposal_evaluated,
                        expansion_evaluated=self.expansion_proposal_evaluated,
                        intervened=self.intervened,
                    )
                    authorized = proposal_tier is not None
                    self.persistence_count = int(authorized)
                    if proposal_tier == "core":
                        self.core_proposal_evaluated = True
                    elif proposal_tier == "expansion":
                        self.expansion_proposal_evaluated = True
                    self.proposal_evaluated = (
                        self.core_proposal_evaluated
                        or self.expansion_proposal_evaluated
                    )
                else:
                    above = (
                        not self.intervened
                        and (
                            self.revision not in ("mf3ze", "mf3zf")
                            or not self.proposal_evaluated
                        )
                        and (
                            self.revision != "mf3zb"
                            or self.step >= self.parameters["minimum_decision_step"]
                        )
                        and cold_start_ok
                        and policy_risk_adjusted_score
                        > self.parameters["score_threshold"]
                        and upper_ok
                    )
                    self.persistence_count = (
                        self.persistence_count + 1 if above else 0
                    )
                    authorized = (
                        above and self.persistence_count
                        >= self.parameters["persistence_steps"]
                    )
                    if authorized and self.revision in ("mf3ze", "mf3zf"):
                        if self.revision == "mf3zf" and self.collection_only:
                            return_gate_evidence = {
                                "authorized": True,
                                "collection_only": True,
                            }
                        else:
                            self.proposal_evaluated = True
                if (
                    authorized
                    and self.revision in ("mf3ze", "mf3zf", "mf3zg", "mf3zh")
                    and not (self.revision == "mf3zf" and self.collection_only)
                ):
                    gate_decision = {
                        "step": self.step,
                        "policy_risk_adjusted_score": policy_risk_adjusted_score,
                        "native_margin": native_margin,
                        "minimum_top2_advantage": minimum_advantage,
                        "median_top2_advantage": median_advantage,
                        "robust_top2_advantage": robust_advantage,
                        "ensemble_mad": ensemble_mad,
                        "cold_start_floor_ratio": cold_start_floor_ratio,
                        "cold_start_relative_mad": cold_start_relative_mad,
                        "current_local_action_ids": [
                            str(ids[index]) for index in current_global
                        ],
                    }
                    features = action_aligned_features(
                        gate_decision,
                        self.instruction.detach().cpu().float().numpy(),
                        self.latest_history.detach().cpu().float().numpy(),
                        kwargs["gmap_img_fts"][0, native]
                        .detach().cpu().float().numpy(),
                        kwargs["gmap_img_fts"][0, ids.index(ordered[runner_local])]
                        .detach().cpu().float().numpy(),
                    )
                    if self.revision in ("mf3zg", "mf3zh"):
                        return_gate_evidence = (
                            self.core_return_gate
                            if proposal_tier == "core"
                            else self.expansion_return_gate
                        ).evaluate(features)
                        return_gate_evidence["tier"] = proposal_tier
                    else:
                        return_gate_evidence = self.return_gate.evaluate(features)
                    authorized = bool(return_gate_evidence["authorized"])
                    if authorized:
                        decision_source = "learned_residual"
            elif self.revision == "mf3p":
                above = (
                    not self.intervened
                    and robust_advantage > self.parameters["logit_threshold"]
                )
                self.persistence_count = self.persistence_count + 1 if above else 0
                authorized = (
                    above and self.persistence_count
                    >= self.parameters["persistence_steps"]
                )
            else:
                authorized = (
                robust_advantage > self.parameters["robust_advantage_threshold"]
                if self.revision == "mf3m"
                else minimum_advantage
                > self.parameters["conditional_advantage_threshold"]
                if self.revision == "mf3l"
                else (
                    minimum_advantage > self.parameters["utility_threshold"]
                    and native_margin <= self.parameters["native_margin_max"]
                )
                )
            if self.revision == "mf3zh":
                decision_source = residual_with_uncertainty_source(
                    authorized, native_margin,
                    self.parameters["uncertainty_native_margin_max"],
                )
                authorized = decision_source is not None
            if authorized:
                if self.revision == "mf3zh":
                    if decision_source == "learned_residual":
                        self.intervened = True
                elif self.revision in ("mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg"):
                    self.intervened = True
                adapted_id = ordered[runner_local]
                adapted = ids.index(adapted_id)
                self.write_intervention_feature(
                    kwargs["gmap_img_fts"][0, native],
                    kwargs["gmap_img_fts"][0, adapted],
                )
                fused = native_logits.clone()
                fused[0, adapted] = native_logits[0, native] + 1e-4
                if int(torch.argmax(fused[0])) != adapted:
                    raise RuntimeError(
                        f"{self.revision.upper()} declared/global action drift"
                    )
        elif self.revision in ("mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"):
            self.persistence_count = 0
        row = {
            "schema_version": f"revealnav-{self.revision}-policy-decision/1",
            "step": self.step,
            "native_action_index": native,
            "adapted_action_index": adapted,
            "native_action_id": None if native == 0 else str(ids[native]),
            "adapted_action_id": None if adapted == 0 else str(ids[adapted]),
            "current_local_action_ids": [str(ids[index]) for index in current_global],
            "authorized": authorized,
            "action_changed": adapted != native,
            "minimum_top2_advantage": minimum_advantage,
            "median_top2_advantage": median_advantage,
            "ensemble_mad": ensemble_mad,
            "cold_start_floor_ratio": cold_start_floor_ratio,
            "cold_start_relative_mad": cold_start_relative_mad,
            "robust_top2_advantage": robust_advantage,
            "policy_risk_adjusted_score": policy_risk_adjusted_score,
            "native_margin": native_margin,
            "runner_local_index": runner_local,
            "return_gate": return_gate_evidence,
            "decision_source": decision_source,
            "proposal_evaluated": self.proposal_evaluated,
            "proposal_evaluated_tiers": ({
                "core": self.core_proposal_evaluated,
                "expansion": self.expansion_proposal_evaluated,
            } if self.revision in ("mf3zg", "mf3zh") else None),
            "parameters": self.parameters,
            "episode_intervention_already_used": self.intervened,
            "persistence_count": self.persistence_count,
            "previous_hash": self.previous_hash,
            **MF3B_SCOPE,
        }
        row["record_hash"] = stable_hash(row)
        self.previous_hash = row["record_hash"]
        self.records.append(row)
        with self.trace.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.step += 1
        if fused is native_logits:
            return result
        updated = dict(result)
        updated["global_logits"] = fused
        return updated


class UncertaintyOnlyController:
    def __init__(self, trace: Path, gate_path: Path = MF3G_GATE) -> None:
        gate = json.loads(gate_path.read_text())
        if not (
            gate.get("status") == "SHADOW_GATE_PASS"
            and gate.get("task_metric_run_authorized") is True
        ):
            raise RuntimeError("MF3G shadow gate does not authorize uncertainty control")
        # The frozen MF3V floor is multi-step.  A collection-only caller may
        # request a one-shot intervention explicitly; the default deployment
        # behavior remains byte-for-byte unchanged.
        gate_one_shot = "uncertainty_rule" in gate
        self.one_shot = (
            os.environ.get("REVEALNAV_MF3_UNCERTAINTY_ONE_SHOT") == "1"
            or gate_one_shot
        )
        if gate_one_shot:
            source = gate["uncertainty_rule"]
        elif "uncertainty_calibration_budget_match" in gate:
            source = gate["uncertainty_calibration_budget_match"]
        elif gate.get("schema_version") in (
            "revealnav-mf3t-shadow-gate/1",
            "revealnav-mf3u-shadow-gate/1",
            "revealnav-mf3v-shadow-gate/1",
            "revealnav-mf3y-shadow-gate/1",
            "revealnav-mf3z-shadow-gate/1",
            "revealnav-mf3za-shadow-gate/1",
            "revealnav-mf3zb-shadow-gate/1",
            "revealnav-mf3zc-shadow-gate/1",
        ):
            # MF3T's exact-budget control is sealed directly in the shadow
            # gate.  It is the same native-margin policy, with no learned
            # parameters or val_seen information.
            source = gate["exact_budget_control"]
        else:
            raise RuntimeError("shadow gate is missing uncertainty control")
        self.parameters = {"native_margin_max": float(source["native_margin_max"])}
        self.checkpoint = None
        self.trace = trace
        self.trace.write_text("")
        self.records = []
        self.previous_hash = "0" * 64
        self.step = 0
        self.intervened = False
        feature_value = os.environ.get("REVEALNAV_MF3_UNCERTAINTY_FEATURE")
        self.feature_path = Path(feature_value).resolve() if feature_value else None
        if self.feature_path is not None:
            if ROOT not in self.feature_path.parents or self.feature_path.is_symlink():
                raise RuntimeError("uncertainty feature path must be project-local")
            if self.feature_path.exists():
                raise RuntimeError("refusing to overwrite uncertainty feature")
        self.instruction = None
        self.latest_history = None

    def record_language(self, embedding, mask) -> None:
        self.instruction = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_panorama(self, embedding, mask) -> None:
        self.latest_history = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def _write_feature(
        self, native_embedding: torch.Tensor,
        alternative_embedding: torch.Tensor,
    ) -> None:
        if self.feature_path is None:
            return
        if self.instruction is None or self.latest_history is None:
            raise RuntimeError("uncertainty feature lacks language/history state")
        values = {
            "instruction": self.instruction.detach().cpu().float().numpy(),
            "checkpoint": self.latest_history.detach().cpu().float().numpy(),
            "native": native_embedding.detach().cpu().float().numpy(),
            "alternative": alternative_embedding.detach().cpu().float().numpy(),
        }
        if any(value.shape != (768,) or not np.isfinite(value).all()
               for value in values.values()):
            raise RuntimeError("uncertainty action-aligned feature drift")
        self.feature_path.parent.mkdir(parents=True, exist_ok=True)
        part = self.feature_path.with_name(self.feature_path.name + ".part")
        with part.open("wb") as stream:
            np.savez(stream, **values)
        os.replace(part, self.feature_path)

    def navigation(self, kwargs: dict, result: dict) -> dict:
        global _TRAINER
        ids = kwargs["gmap_vp_ids"][0]
        action_mask = [bool(value) for value in kwargs["gmap_masks"][0]]
        visited = [bool(value) for value in kwargs["gmap_visited_masks"][0]]
        current_global = current_local_action_indices(
            ids, action_mask, visited, _LOCAL_ACTION_IDS[0]
        )
        native_logits = result["global_logits"]
        native_argmax = int(torch.argmax(native_logits[0]))
        forced_stop = (
            bool(_NO_VP_LEFT[0]) or self.step >= int(_TRAINER.max_len) - 1
        )
        native = 0 if forced_stop else native_argmax
        adapted = native
        margin = None
        authorized = False
        fused = native_logits
        if len(current_global) >= 2 and native in current_global and not forced_stop:
            indices = torch.tensor(current_global, device=native_logits.device)
            values = native_logits[0, indices]
            order = torch.argsort(values, descending=True)
            margin = float(values[order[0]] - values[order[1]])
            authorized = (
                not self.intervened
                and margin <= self.parameters["native_margin_max"]
            )
            if authorized:
                if self.one_shot:
                    self.intervened = True
                adapted = int(indices[order[1]])
                self._write_feature(
                    kwargs["gmap_img_fts"][0, native],
                    kwargs["gmap_img_fts"][0, adapted],
                )
                fused = native_logits.clone()
                fused[0, adapted] = native_logits[0, native] + 1e-4
        row = {
            "schema_version": "revealnav-mf3g-uncertainty-policy-decision/1",
            "step": self.step,
            "native_action_index": native,
            "adapted_action_index": adapted,
            "native_action_id": None if native == 0 else str(ids[native]),
            "adapted_action_id": None if adapted == 0 else str(ids[adapted]),
            "current_local_action_ids": [str(ids[index]) for index in current_global],
            "authorized": authorized, "action_changed": adapted != native,
            "native_margin": margin, "parameters": self.parameters,
            "one_shot": self.one_shot,
            "previous_hash": self.previous_hash,
            **MF3B_SCOPE,
        }
        row["record_hash"] = stable_hash(row)
        self.previous_hash = row["record_hash"]
        self.records.append(row)
        with self.trace.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.step += 1
        if fused is native_logits:
            return result
        updated = dict(result)
        updated["global_logits"] = fused
        return updated


def install_hooks() -> None:
    from vlnce_baselines.models.R1Policy import ETP
    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer

    original_gmap = RLTrainer._nav_gmap_variable

    def gmap(self, cur_vp, cur_pos, cur_ori, task_type):
        global _TRAINER, _LOCAL_ACTION_IDS, _NO_VP_LEFT
        value = original_gmap(self, cur_vp, cur_pos, cur_ori, task_type)
        _TRAINER = self
        _NO_VP_LEFT = tuple(bool(row) for row in value["no_vp_left"])
        _LOCAL_ACTION_IDS = tuple(
            {
                str(ghost) for ghost, fronts in graph.ghost_fronts.items()
                if current in fronts
            }
            for graph, current in zip(self.gmaps, cur_vp)
        )
        return value

    RLTrainer._nav_gmap_variable = gmap
    original_forward = ETP.forward

    def forward(self, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else None)
        captured_policy_input = []
        handle = None
        if (
            mode == "navigation"
            and getattr(_CONTROLLER, "policy_fusion_features", False)
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
        if mode == "language":
            _CONTROLLER.record_language(result, kwargs["txt_masks"])
        elif mode == "panorama":
            _CONTROLLER.record_panorama(result[0], result[1])
        elif mode == "navigation":
            if getattr(_CONTROLLER, "policy_fusion_features", False):
                if len(captured_policy_input) != 1:
                    raise RuntimeError("MF3 policy fusion feature was not captured")
                kwargs = {
                    **kwargs,
                    "_mf3_policy_fusion_features": captured_policy_input[0],
                }
            result = _CONTROLLER.navigation(kwargs, result)
        return result

    ETP.forward = forward


def verify_execution(records: list[dict], actions: list[dict]) -> dict:
    if len(actions) < len(records):
        raise RuntimeError("executed action trace is shorter than UAD decisions")
    matches = []
    for row in records:
        action = actions[row["step"]]
        if row["adapted_action_index"] == 0:
            matches.append(int(action["act"]) == 0)
        else:
            matches.append(
                int(action["act"]) == 4
                and str(action.get("ghost_vp")) == row["adapted_action_id"]
            )
    if not all(matches):
        raise RuntimeError("UAD declared action differs from execution")
    return {"checked": len(matches), "all_equal": True}


def main() -> None:
    global _CONTROLLER
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument(
        "--mode", choices=(
            "baseline", "uad", "ensemble", "uncertainty"
        ), required=True
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--revision", choices=(
            "mf3b", "mf3g", "mf3h", "mf3i", "mf3j", "mf3k", "mf3l", "mf3m", "mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"
        ), default="mf3b"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val_seen"), default="val_seen")
    args = parser.parse_args()
    if (args.mode == "uad") != (args.seed in SEEDS):
        raise SystemExit("UAD requires one sealed seed; controls forbid seed")
    if args.mode in ("ensemble", "uncertainty") and args.revision not in (
        "mf3g", "mf3h", "mf3i", "mf3j", "mf3k", "mf3l", "mf3m", "mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"
    ):
        raise SystemExit("ensemble/uncertainty controls require MF3G or MF3H")
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents or run_dir.exists():
        raise SystemExit("run directory must be new and project-local")
    run_dir.mkdir(parents=True)
    base_trace = run_dir / "base_trace.jsonl"
    controller_trace = run_dir / "controller_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)
    if args.mode in ("uad", "ensemble", "uncertainty"):
        controller_class = (
            MF3GResidualController if args.revision == "mf3g"
            else UADResidualController
        )
        _CONTROLLER = (
            UncertaintyOnlyController(
                controller_trace,
                (
                    MF3V_GATE if args.revision in ("mf3ze", "mf3zf", "mf3zg", "mf3zh")
                    else MF3ZC_GATE if args.revision == "mf3zc"
                    else MF3U_GATE if args.revision == "mf3u"
                    else MF3V_GATE if args.revision == "mf3v"
                    else MF3ZB_GATE if args.revision == "mf3zb"
                    else MF3ZA_GATE if args.revision == "mf3za"
                    else MF3Z_GATE if args.revision == "mf3z"
                    else MF3Y_GATE if args.revision == "mf3y"
                    else MF3T_GATE if args.revision == "mf3t"
                    else MF3S_GATE if args.revision == "mf3s"
                    else MF3P_GATE if args.revision == "mf3p"
                    else MF3M_GATE if args.revision == "mf3m"
                    else MF3L_GATE if args.revision == "mf3l"
                    else MF3K_GATE if args.revision == "mf3k"
                    else MF3J_GATE if args.revision == "mf3j"
                    else MF3I_GATE if args.revision == "mf3i"
                    else MF3H_GATE if args.revision == "mf3h"
                    else MF3G_GATE
                ),
            )
            if args.mode == "uncertainty"
            else MF3JSwitchController(
                torch.device("cuda:0"), controller_trace
            ) if args.mode == "ensemble" and args.revision == "mf3j"
            else MF3KTop2Controller(
                torch.device("cuda:0"), controller_trace,
                revision=args.revision,
            ) if args.mode == "ensemble" and args.revision in ("mf3k", "mf3l", "mf3m", "mf3p", "mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh")
            else MF3GResidualController(
                None, torch.device("cuda:0"), controller_trace,
                gate_path=(
                    MF3I_GATE if args.revision == "mf3i"
                    else MF3H_GATE if args.revision == "mf3h"
                    else MF3G_GATE
                ),
                require_unanimous=args.revision in ("mf3h", "mf3i"),
                revision=args.revision,
            ) if args.mode == "ensemble"
            else controller_class(
                args.seed, torch.device("cuda:0"), controller_trace
            )
        )
        install_hooks()

    os.chdir(ETPR1)
    from etpr1_compat import configure_project_cache_env
    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = [
        "run.py", "--exp_name",
        f"{args.revision}_{args.mode}_{args.seed}_{args.episode_id}",
        "--run-type", "eval", "--exp-config", "run_rxr/iter_train.yaml",
        "EVAL.SPLIT", args.split, "TASK_CONFIG.DATASET.SPLIT", args.split,
        "EVAL.LANGUAGES", "['en-US','en-IN']",
        "EVAL.EPISODE_ID", f"['{args.episode_id}']", "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(RXR_CHECKPOINT), "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED), "IL.back_algo", "control",
        "IL.RECOLLECT_TRAINER.gt_file",
        "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
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
        "schema_version": f"revealnav-{args.revision}-rxr-worker/1",
        "status": "RUNNING", "episode_id": args.episode_id,
        "mode": args.mode, "seed": args.seed, "revision": args.revision,
        "split": args.split,
        "public_unseen_accessed": False,
        "checkpoint": None if _CONTROLLER is None else _CONTROLLER.checkpoint,
        **MF3B_SCOPE,
    }
    sys.argv = argv
    import run
    started = time.monotonic()
    try:
        run.main()
        if _CONTROLLER is not None:
            actions = [json.loads(line) for line in base_trace.read_text().splitlines() if line]
            summary["executed_action_validation"] = verify_execution(
                _CONTROLLER.records, actions
            )
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["wall_time_s"] = round(time.monotonic() - started, 3)
        summary["peak_rss_self_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary["controller"] = None if _CONTROLLER is None else {
            "decisions": len(_CONTROLLER.records),
            "authorized": sum(row["authorized"] for row in _CONTROLLER.records),
            "actions_changed": sum(row["action_changed"] for row in _CONTROLLER.records),
            "parameters": _CONTROLLER.parameters,
            "final_record_hash": _CONTROLLER.previous_hash,
        }
        stats = list(output.rglob(f"stats_ep_ckpt_1320_{args.split}_r0_w1.json"))
        summary["metrics"] = None
        if len(stats) == 1:
            summary["metrics"] = json.loads(stats[0].read_text()).get(str(args.episode_id))
        (run_dir / "RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
