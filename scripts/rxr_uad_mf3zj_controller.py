#!/usr/bin/env python3
"""MF3ZJ: MF3ZG-first policy with one transferred fallback opportunity."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import rxr_uad_controller_worker_mf3 as base
from revealnav_mf3.uncertainty_gate import (
    CounterfactualTransferGate,
    transfer_action_features,
)


ROOT = base.ROOT
GATE = ROOT / (
    "artifacts/training/mf3zj_counterfactual_transfer_gate_v1/"
    "MF3ZJ_CROSSFIT_GATE.json"
)
MODEL = ROOT / (
    "artifacts/training/mf3zj_counterfactual_transfer_gate_v1/"
    "MF3ZJ_TRANSFER_GATE_MODELS.npz"
)


class TransferredFallbackController:
    """Evaluate only the first eligible native-margin runner-up proposal."""

    def __init__(self, trace: Path) -> None:
        gate = json.loads(GATE.read_text())
        model = gate.get("model", {})
        if not (
            gate.get("status") == "SHADOW_GATE_PASS"
            and gate.get("task_metric_run_authorized") is True
            and gate.get("controls", {}).get("unseen_or_test_read") is False
            and model.get("path") == str(MODEL.relative_to(ROOT))
            and model.get("bytes") == MODEL.stat().st_size
            and model.get("sha256") == base.sha256_file(MODEL)
        ):
            raise RuntimeError("MF3ZJ transfer gate does not authorize metrics")
        source = json.loads(base.MF3V_GATE.read_text())
        if not (
            source.get("status") == "SHADOW_GATE_PASS"
            and source.get("public_unseen_authorized") is False
        ):
            raise RuntimeError("MF3ZJ native-margin source drift")
        self.parameters = {
            "native_margin_max": float(
                source["exact_budget_control"]["native_margin_max"]
            ),
            "return_threshold": float(
                gate["selected_rule"]["return_threshold"]
            ),
            "harm_probability_threshold": float(
                gate["selected_rule"]["harm_probability_threshold"]
            ),
        }
        self.gate = CounterfactualTransferGate(MODEL, gate["selected_rule"])
        self.checkpoint = {
            "path": str(MODEL.relative_to(ROOT)),
            "bytes": MODEL.stat().st_size,
            "sha256": base.sha256_file(MODEL),
            "strict_load": True,
        }
        self.trace = trace
        self.trace.write_text("")
        self.records: list[dict] = []
        self.previous_hash = "0" * 64
        self.step = 0
        self.proposal_evaluated = False
        self.intervened = False
        self.instruction: torch.Tensor | None = None
        self.latest_history: torch.Tensor | None = None

    def record_language(self, embedding: torch.Tensor, mask: torch.Tensor) -> None:
        self.instruction = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_panorama(self, embedding: torch.Tensor, mask: torch.Tensor) -> None:
        self.latest_history = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def navigation(self, kwargs: dict, result: dict) -> dict:
        ids = kwargs["gmap_vp_ids"][0]
        action_mask = [bool(value) for value in kwargs["gmap_masks"][0]]
        visited = [bool(value) for value in kwargs["gmap_visited_masks"][0]]
        current_global = base.current_local_action_indices(
            ids, action_mask, visited, base._LOCAL_ACTION_IDS[0]
        )
        native_logits = result["global_logits"]
        native_argmax = int(torch.argmax(native_logits[0]))
        forced_stop = bool(base._NO_VP_LEFT[0]) or (
            self.step >= int(base._TRAINER.max_len) - 1
        )
        native = 0 if forced_stop else native_argmax
        adapted = native
        margin = None
        gate_evidence = None
        evaluated_now = False
        authorized = False
        fused = native_logits
        if (
            len(current_global) >= 2
            and native in current_global
            and not forced_stop
            and not self.intervened
            and not self.proposal_evaluated
            and self.instruction is not None
            and self.latest_history is not None
        ):
            indices = torch.tensor(current_global, device=native_logits.device)
            values = native_logits[0, indices]
            order = torch.argsort(values, descending=True)
            if int(indices[order[0]]) == native:
                runner = int(indices[order[1]])
                margin = float(values[order[0]] - values[order[1]])
                if not math.isfinite(margin) or margin < 0:
                    raise RuntimeError("MF3ZJ native margin drift")
                if margin <= self.parameters["native_margin_max"]:
                    evaluated_now = True
                    self.proposal_evaluated = True
                    decision = {
                        "step": self.step,
                        "native_margin": margin,
                        "current_local_action_ids": [
                            str(ids[index]) for index in current_global
                        ],
                    }
                    features = transfer_action_features(
                        decision,
                        self.instruction.detach().cpu().float().numpy(),
                        self.latest_history.detach().cpu().float().numpy(),
                        kwargs["gmap_img_fts"][0, native]
                        .detach().cpu().float().numpy(),
                        kwargs["gmap_img_fts"][0, runner]
                        .detach().cpu().float().numpy(),
                    )
                    gate_evidence = self.gate.evaluate(features)
                    authorized = bool(gate_evidence["authorized"])
                    if authorized:
                        self.intervened = True
                        adapted = runner
                        fused = native_logits.clone()
                        fused[0, adapted] = native_logits[0, native] + 1e-4
                        if int(torch.argmax(fused[0])) != adapted:
                            raise RuntimeError("MF3ZJ declared/global action drift")
        row = {
            "schema_version": "revealnav-mf3zj-fallback-decision/1",
            "step": self.step,
            "native_action_index": native,
            "adapted_action_index": adapted,
            "native_action_id": None if native == 0 else str(ids[native]),
            "adapted_action_id": None if adapted == 0 else str(ids[adapted]),
            "current_local_action_ids": [
                str(ids[index]) for index in current_global
            ],
            "authorized": authorized,
            "action_changed": adapted != native,
            "native_margin": margin,
            "return_gate": gate_evidence,
            "proposal_evaluated_this_step": evaluated_now,
            "proposal_evaluated": self.proposal_evaluated,
            "episode_intervention_already_used": self.intervened,
            "parameters": self.parameters,
            "previous_hash": self.previous_hash,
            "public_unseen_authorized": base.MF3B_SCOPE[
                "public_unseen_authorized"
            ],
            "method_scope": "counterfactual_transfer_fallback",
        }
        row["record_hash"] = base.stable_hash(row)
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


class MF3ZJController:
    """Run MF3ZG first and enforce one switch across both proposal sources."""

    policy_fusion_features = True

    def __init__(self, device: torch.device, trace: Path, run_dir: Path) -> None:
        self.learned = base.MF3KTop2Controller(
            device, run_dir / "mf3zg_internal_trace.jsonl", revision="mf3zg"
        )
        self.fallback = TransferredFallbackController(
            run_dir / "fallback_internal_trace.jsonl"
        )
        self.trace = trace
        self.trace.write_text("")
        self.records: list[dict] = []
        self.previous_hash = "0" * 64
        self.checkpoint = [*self.learned.checkpoint, self.fallback.checkpoint]
        self.parameters = {
            "learned": self.learned.parameters,
            "fallback": self.fallback.parameters,
            "priority": "mf3zg_then_counterfactual_transfer_fallback",
            "maximum_executed_switches_per_episode": 1,
        }
        self.switch_used = False
        self.step = 0

    def record_language(self, embedding: torch.Tensor, mask: torch.Tensor) -> None:
        self.learned.record_language(embedding, mask)
        self.fallback.record_language(embedding, mask)

    def record_panorama(self, embedding: torch.Tensor, mask: torch.Tensor) -> None:
        self.learned.record_panorama(embedding, mask)
        self.fallback.record_panorama(embedding, mask)

    def navigation(self, kwargs: dict, result: dict) -> dict:
        if self.switch_used:
            self.learned.intervened = True
            self.fallback.intervened = True
        decision_step = self.learned.step
        learned_result = self.learned.navigation(kwargs, result)
        learned_row = self.learned.records[-1]
        learned_changed = bool(learned_row["action_changed"])
        fallback_row = None
        chosen_result = learned_result
        chosen = learned_row
        source = "learned_residual" if learned_changed else None
        if learned_changed:
            if self.switch_used:
                raise RuntimeError("MF3ZJ exceeded its global switch budget")
            self.switch_used = True
            self.fallback.intervened = True
        elif not self.switch_used:
            self.fallback.step = decision_step
            fallback_result = self.fallback.navigation(kwargs, result)
            fallback_row = self.fallback.records[-1]
            chosen = fallback_row
            if fallback_row["action_changed"]:
                self.switch_used = True
                self.learned.intervened = True
                chosen_result = fallback_result
                source = "counterfactual_fallback"
            self.fallback.step = self.learned.step
        else:
            self.fallback.step = self.learned.step
        if sum(bool(row["action_changed"]) for row in self.records) + bool(
            chosen["action_changed"]
        ) > 1:
            raise RuntimeError("MF3ZJ trace contains more than one switch")
        row = dict(chosen)
        row["schema_version"] = "revealnav-mf3zj-policy-decision/1"
        row["decision_source"] = source
        row["learned_action_changed"] = learned_changed
        row["learned_authorized"] = bool(learned_row["authorized"])
        row["fallback_considered"] = fallback_row is not None
        row["fallback_proposal_evaluated"] = self.fallback.proposal_evaluated
        row["global_switch_used"] = self.switch_used
        row["previous_hash"] = self.previous_hash
        row["record_hash"] = base.stable_hash(row)
        self.previous_hash = row["record_hash"]
        self.records.append(row)
        with self.trace.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.step += 1
        return chosen_result
