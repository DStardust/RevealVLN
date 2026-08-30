#!/usr/bin/env python3
"""MF3ZI controller: MF3ZG primary plus a gated one-shot uncertainty residual."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import rxr_uad_controller_worker_mf3 as base
from revealnav_mf3.uncertainty_gate import (
    UncertaintyReturnGate,
    uncertainty_action_features,
)


ROOT = base.ROOT
MF3V_GATE = base.MF3V_GATE
GATE = ROOT / "artifacts/training/mf3zi_uncertainty_return_gate_v1/MF3ZI_CROSSFIT_GATE.json"
MODEL = ROOT / "artifacts/training/mf3zi_uncertainty_return_gate_v1/MF3ZI_GATE_MODELS.npz"


class GatedUncertaintyController:
    """A one-shot native-margin candidate guarded by an exact-return model."""

    def __init__(self, trace: Path) -> None:
        gate = json.loads(GATE.read_text())
        if not (
            gate.get("status") == "SHADOW_GATE_PASS"
            and gate.get("task_metric_run_authorized") is True
            and gate.get("controls", {}).get("unseen_or_test_read") is False
            and gate.get("model", {}).get("path") == str(MODEL.relative_to(ROOT))
            and gate.get("model", {}).get("bytes") == MODEL.stat().st_size
            and gate.get("model", {}).get("sha256") == base.sha256_file(MODEL)
        ):
            raise RuntimeError("MF3ZI uncertainty gate does not authorize metrics")
        mf3v = json.loads(MF3V_GATE.read_text())
        if mf3v.get("public_unseen_authorized") is not False:
            raise RuntimeError("MF3ZI MF3V uncertainty source is not train-only")
        source = mf3v.get("exact_budget_control", {})
        self.parameters = {
            "native_margin_max": float(source["native_margin_max"]),
            "return_threshold": float(gate["selected_rule"]["return_threshold"]),
            "harm_probability_threshold": float(
                gate["selected_rule"]["harm_probability_threshold"]
            ),
        }
        self.gate = UncertaintyReturnGate(MODEL, gate["selected_rule"])
        self.checkpoint = {
            "path": str(MODEL.relative_to(ROOT)),
            "bytes": MODEL.stat().st_size,
            "sha256": base.sha256_file(MODEL),
            "strict_load": True,
        }
        self.trace = trace
        self.trace.write_text("")
        self.records = []
        self.previous_hash = "0" * 64
        self.step = 0
        self.intervened = False
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
        authorized = False
        fused = native_logits
        if (
            len(current_global) >= 2 and native in current_global
            and not forced_stop and not self.intervened
            and self.instruction is not None and self.latest_history is not None
        ):
            indices = torch.tensor(current_global, device=native_logits.device)
            values = native_logits[0, indices]
            order = torch.argsort(values, descending=True)
            # The controller is explicitly native-vs-runner-up; fail closed if
            # the native index is not the current-set maximum.
            if int(indices[order[0]]) == native:
                runner = int(indices[order[1]])
                margin = float(values[order[0]] - values[order[1]])
                if margin <= self.parameters["native_margin_max"]:
                    decision = {
                        "step": self.step,
                        "native_margin": margin,
                        "current_local_action_ids": [
                            str(ids[index]) for index in current_global
                        ],
                    }
                    features = uncertainty_action_features(
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
                            raise RuntimeError("MF3ZI declared/global action drift")
        row = {
            "schema_version": "revealnav-mf3zi-uncertainty-decision/1",
            "step": self.step,
            "native_action_index": native,
            "adapted_action_index": adapted,
            "native_action_id": None if native == 0 else str(ids[native]),
            "adapted_action_id": None if adapted == 0 else str(ids[adapted]),
            "current_local_action_ids": [str(ids[index]) for index in current_global],
            "authorized": authorized,
            "action_changed": adapted != native,
            "native_margin": margin,
            "return_gate": gate_evidence,
            "one_shot": True,
            "episode_intervention_already_used": self.intervened,
            "parameters": self.parameters,
            "previous_hash": self.previous_hash,
            "public_unseen_authorized": base.MF3B_SCOPE["public_unseen_authorized"],
            "method_scope": "mf3zi_uncertainty_residual",
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


class MF3ZIController:
    """Compose MF3ZG and the gated uncertainty residual without changing either."""

    policy_fusion_features = True

    def __init__(self, device: torch.device, trace: Path, run_dir: Path) -> None:
        self.learned = base.MF3KTop2Controller(
            device, run_dir / "mf3zg_internal_trace.jsonl", revision="mf3zg"
        )
        self.uncertainty = GatedUncertaintyController(
            run_dir / "uncertainty_internal_trace.jsonl"
        )
        self.trace = trace
        self.trace.write_text("")
        self.records = []
        self.previous_hash = "0" * 64
        self.checkpoint = [
            *self.learned.checkpoint,
            self.uncertainty.checkpoint,
        ]
        self.parameters = {
            "learned": self.learned.parameters,
            "uncertainty": self.uncertainty.parameters,
            "priority": "mf3zg_then_gated_one_shot_uncertainty",
        }
        self.step = 0

    def record_language(self, embedding, mask) -> None:
        self.learned.record_language(embedding, mask)
        self.uncertainty.record_language(embedding, mask)

    def record_panorama(self, embedding, mask) -> None:
        self.learned.record_panorama(embedding, mask)
        self.uncertainty.record_panorama(embedding, mask)

    def navigation(self, kwargs: dict, result: dict) -> dict:
        decision_step = self.learned.step
        learned_result = self.learned.navigation(kwargs, result)
        learned_row = self.learned.records[-1]
        chosen_result = learned_result
        chosen = learned_row
        source = "learned_residual" if learned_row["action_changed"] else None
        if not learned_row["action_changed"]:
            # The learned controller increments its step at the end of
            # navigation; align the fallback's decision index explicitly.
            self.uncertainty.step = decision_step
            fallback_result = self.uncertainty.navigation(kwargs, result)
            fallback_row = self.uncertainty.records[-1]
            if fallback_row["action_changed"]:
                chosen_result = fallback_result
                chosen = fallback_row
                source = "uncertainty_gate"
            self.uncertainty.step = self.learned.step
        else:
            self.uncertainty.step = self.learned.step
        row = dict(chosen)
        row["schema_version"] = "revealnav-mf3zi-policy-decision/1"
        row["decision_source"] = source
        row["learned_action_changed"] = bool(learned_row["action_changed"])
        row["learned_authorized"] = bool(learned_row["authorized"])
        row["uncertainty_considered"] = not bool(learned_row["action_changed"])
        row["previous_hash"] = self.previous_hash
        row["record_hash"] = base.stable_hash(row)
        self.previous_hash = row["record_hash"]
        self.records.append(row)
        with self.trace.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.step += 1
        if chosen_result is result:
            return result
        return chosen_result
