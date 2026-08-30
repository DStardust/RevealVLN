"""Exact native-shadow and one-switch replay mechanics for MF3ZL data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np
import torch

from revealnav_mf3.shadow import validate_action_identity


@dataclass(frozen=True)
class ProposalEventIdentity:
    dataset: str
    episode_id: str
    scene_id: str
    step: int
    tier: str
    native_action_id: str
    runner_action_id: str

    def __post_init__(self) -> None:
        if self.dataset not in {"RxR", "R2R"}:
            raise ValueError("exact-replay dataset drift")
        if not self.episode_id or len(self.scene_id) != 11 or self.step < 0:
            raise ValueError("invalid exact-replay episode identity")
        if self.tier not in {"core", "expansion"}:
            raise ValueError("invalid exact-replay proposal tier")
        if (
            not self.native_action_id
            or not self.runner_action_id
            or self.native_action_id == self.runner_action_id
        ):
            raise ValueError("invalid exact-replay action identity")


class _AlwaysAbstain:
    def evaluate(self, _features: np.ndarray) -> dict:
        return {"authorized": False, "collection_observer": True}


def validate_collection_scope(
    *, dataset: str, split: str, scene_id: str,
    allowed_scenes: set[str], consumed_scenes: set[str],
) -> None:
    if dataset not in {"RxR", "R2R"} or split != "train":
        raise ValueError("exact replay is restricted to RxR/R2R train")
    if scene_id not in allowed_scenes:
        raise ValueError("exact replay scene is outside the sealed population")
    if scene_id in consumed_scenes:
        raise ValueError("consumed confirmation scene entered exact replay")


def validate_shadow_event(record: dict) -> ProposalEventIdentity:
    if record.get("mode") != "native_shadow" or record.get("action_changed") is not False:
        raise ValueError("native shadow changed an action")
    identity = ProposalEventIdentity(**record["event_identity"])
    if (
        record.get("native_action_id") != identity.native_action_id
        or record.get("adapted_action_id") != identity.native_action_id
    ):
        raise ValueError("native shadow action identity drift")
    return identity


def validate_forced_switch(
    records: list[dict], target: ProposalEventIdentity,
) -> None:
    changed = [record for record in records if record.get("action_changed") is True]
    if len(changed) != 1 or changed[0].get("event_identity") != asdict(target):
        raise ValueError("targeted replay did not make exactly the declared switch")
    record = changed[0]
    if (
        record.get("native_action_id") != target.native_action_id
        or record.get("adapted_action_id") != target.runner_action_id
        or int(record.get("step", -1)) != target.step
    ):
        raise ValueError("targeted replay action identity drift")


def validate_exact_prefix(
    native_trace: list[dict], treatment_trace: list[dict], target_step: int,
) -> None:
    if target_step < 0 or len(native_trace) <= target_step or len(treatment_trace) <= target_step:
        raise ValueError("exact-replay trace is shorter than the target prefix")
    keys = ("act", "ghost_vp", "cur_vp", "front_vp", "back_path_len")
    for step, (native, treatment) in enumerate(zip(
        native_trace[:target_step], treatment_trace[:target_step], strict=True,
    )):
        if any(native.get(key) != treatment.get(key) for key in keys):
            raise ValueError(f"exact-replay prefix mismatch at step {step}")


def _write_feature(
    path: Path,
    instruction: torch.Tensor,
    history: torch.Tensor,
    native: torch.Tensor,
    runner: torch.Tensor,
) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise RuntimeError("exact-replay feature path is not a fresh file")
    arrays = {
        "instruction": instruction.detach().cpu().float().numpy(),
        "checkpoint": history.detach().cpu().float().numpy(),
        "native": native.detach().cpu().float().numpy(),
        "alternative": runner.detach().cpu().float().numpy(),
    }
    if any(value.shape != (768,) or not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("exact-replay feature value drift")
    part = path.with_name(path.name + ".part")
    if part.exists() or part.is_symlink():
        raise RuntimeError("stale exact-replay feature partial")
    with part.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(part, path)


class ExactReplayController:
    """Observe MF3ZG proposals and optionally execute one sealed switch."""

    policy_fusion_features = True

    def __init__(
        self,
        proposal_controller,
        trace: Path,
        feature_dir: Path,
        *,
        dataset: str,
        episode_id: str,
        scene_id: str,
        mode: str,
        target: ProposalEventIdentity | None = None,
    ) -> None:
        if mode not in {"native_shadow", "targeted_switch"}:
            raise ValueError("invalid exact-replay controller mode")
        if (mode == "native_shadow") != (target is None):
            raise ValueError("target presence does not match exact-replay mode")
        if target is not None and (
            target.dataset != dataset
            or target.episode_id != str(episode_id)
            or target.scene_id != scene_id
        ):
            raise ValueError("sealed target does not match the controller episode")
        if getattr(proposal_controller, "revision", None) != "mf3zg":
            raise ValueError("exact replay requires the frozen MF3ZG hierarchy")
        if not feature_dir.is_dir() or trace.exists() or trace.is_symlink():
            raise ValueError("exact-replay output paths are not fresh")
        proposal_controller.core_return_gate = _AlwaysAbstain()
        proposal_controller.expansion_return_gate = _AlwaysAbstain()
        proposal_controller.intervened = False
        self.proposal = proposal_controller
        self.trace = trace
        self.feature_dir = feature_dir
        self.dataset = dataset
        self.episode_id = str(episode_id)
        self.scene_id = scene_id
        self.mode = mode
        self.target = target
        self.records: list[dict] = []
        self.events: list[dict] = []
        self.switched = False
        self.previous_hash = "0" * 64
        self.parameters = dict(proposal_controller.parameters)
        self.checkpoint = proposal_controller.checkpoint
        self.trace.write_text("")

    def record_language(self, embedding, mask) -> None:
        self.proposal.record_language(embedding, mask)

    def record_panorama(self, embedding, mask) -> None:
        self.proposal.record_panorama(embedding, mask)

    def navigation(self, kwargs: dict, result: dict) -> dict:
        before = len(self.proposal.records)
        observed = self.proposal.navigation(kwargs, result)
        if len(self.proposal.records) != before + 1:
            raise RuntimeError("MF3ZG proposal observer record cardinality drift")
        inner = self.proposal.records[-1]
        if inner.get("action_changed") is not False:
            raise RuntimeError("abstaining MF3ZG observer changed an action")
        if observed is not result and not torch.equal(
            observed["global_logits"], result["global_logits"]
        ):
            raise RuntimeError("abstaining MF3ZG observer changed logits")

        ids = kwargs["gmap_vp_ids"][0]
        native_index = int(inner["native_action_index"])
        adapted_index = native_index
        event = None
        evidence = inner.get("return_gate")
        if isinstance(evidence, dict) and evidence.get("tier") in {"core", "expansion"}:
            native_id = str(inner["feature_native_action_id"])
            runner_id = str(inner["feature_alternative_action_id"])
            if native_id not in ids or runner_id not in ids:
                raise RuntimeError("exact-replay event action left the global map")
            runner_index = ids.index(runner_id)
            current_indices = [
                index for index, identity in enumerate(ids)
                if str(identity) in set(inner["current_local_action_ids"])
            ]
            validate_action_identity(
                ids, current_indices, native_index, runner_index,
                declared_native_id=native_id,
                declared_adapted_id=runner_id,
                require_non_stop=True,
            )
            identity = ProposalEventIdentity(
                dataset=self.dataset,
                episode_id=self.episode_id,
                scene_id=self.scene_id,
                step=int(inner["step"]),
                tier=str(evidence["tier"]),
                native_action_id=native_id,
                runner_action_id=runner_id,
            )
            if any(item["event_identity"] == asdict(identity) for item in self.events):
                raise RuntimeError("duplicate exact-replay event identity")
            feature = self.feature_dir / (
                f"event_{identity.tier}_step_{identity.step}.npz"
            )
            _write_feature(
                feature,
                self.proposal.instruction,
                self.proposal.latest_history,
                kwargs["gmap_img_fts"][0, native_index],
                kwargs["gmap_img_fts"][0, runner_index],
            )
            event = {
                "event_identity": asdict(identity),
                "decision": {
                    key: inner[key] for key in (
                        "step", "minimum_top2_advantage",
                        "median_top2_advantage", "robust_top2_advantage",
                        "ensemble_mad", "cold_start_floor_ratio",
                        "cold_start_relative_mad", "policy_risk_adjusted_score",
                        "native_margin", "current_local_action_ids",
                    )
                },
                "feature_path": str(feature),
            }
            self.events.append(event)
            if self.mode == "targeted_switch" and not self.switched:
                if identity.step == self.target.step and identity != self.target:
                    raise RuntimeError("targeted event identity drift at sealed step")
                if identity == self.target:
                    adapted_index = runner_index
                    self.switched = True

        if (
            self.mode == "targeted_switch"
            and not self.switched
            and int(inner["step"]) >= self.target.step
            and (event is None or event["event_identity"] != asdict(self.target))
        ):
            raise RuntimeError("sealed target proposal was not reproduced")

        output = result
        if adapted_index != native_index:
            logits = result["global_logits"].clone()
            logits[0, adapted_index] = logits[0, native_index] + 1e-4
            if int(torch.argmax(logits[0])) != adapted_index:
                raise RuntimeError("exact-replay forced action did not win argmax")
            output = dict(result)
            output["global_logits"] = logits
        record = {
            "schema_version": "revealnav-mf3zl-exact-replay-decision/1",
            "step": int(inner["step"]),
            "mode": self.mode,
            "native_action_index": native_index,
            "adapted_action_index": adapted_index,
            "native_action_id": None if native_index == 0 else str(ids[native_index]),
            "adapted_action_id": None if adapted_index == 0 else str(ids[adapted_index]),
            "action_changed": adapted_index != native_index,
            "event_identity": None if event is None else event["event_identity"],
            "previous_hash": self.previous_hash,
        }
        record["record_hash"] = __import__("hashlib").sha256(json.dumps(
            record, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        self.previous_hash = record["record_hash"]
        self.records.append(record)
        with self.trace.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return output
