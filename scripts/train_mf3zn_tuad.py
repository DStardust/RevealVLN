#!/usr/bin/env python3
"""Run fixed five-fold MF3ZN-TUAD development training.

The command requires a sealed protocol, a PASS identifiability result, and a
PASS exact-lattice audit.  It always runs the declared model controls with all
three fixed seeds and elementwise-median aggregation; there is no model,
regularization, threshold, epoch, or seed selection argument.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revealnav_mf3.temporal_action_value import (  # noqa: E402
    NativeAnchoredActionValue,
    choose_native_inclusive_action,
    native_anchored_huber_loss,
)
from revealnav_mf3.temporal_uad_model import (  # noqa: E402
    RevealExpiryTargets,
    TemporalRevealExpiryEncoder,
    TemporalRevealExpiryLoss,
    freeze_temporal_encoder,
    last_causal_state,
)
from revealnav_mf3.temporal_uad_features import (  # noqa: E402
    NATIVE_MARGIN_INDEX,
    NATIVE_SCORE_INDEX,
    STRUCTURAL_FEATURE_NAMES,
    causal_sequence_features,
)
from revealnav_mf3.temporal_uad_labels import derive_uad  # noqa: E402
from revealnav_mf3.temporal_uad_schema import (  # noqa: E402
    TemporalSequence,
    temporal_record_list_from_mapping,
)
from revealnav_mf3.tuad_protocol import (  # noqa: E402
    ADAM_LEARNING_RATE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CATASTROPHIC_THRESHOLD,
    ENSEMBLE_REDUCTION,
    FIXED_SEEDS,
    FIXED_WEIGHT_DECAY,
    OUTER_FOLDS,
    STAGE_1_EPOCHS,
    STAGE_2_EPOCHS,
    TUADProtocolError,
    sha256_file,
    verify_protocol,
)
from revealnav_mf3.tuad_identifiability import (  # noqa: E402
    canonical_audit_event_id,
)
from revealnav_mf3.tuad_selection import (  # noqa: E402
    assemble_development_policies,
    assign_tuad_scene_folds,
    evaluate_tuad_development,
    validate_lattice_fold_integrity,
)


ORACLE_KEYS = frozenset({
    "event_id", "delta_utility", "target_in_set", "candidate_separated",
    "evidence_closed",
    "reveal_event", "expiry_event", "factor_mask", "reveal_at_risk",
    "expiry_at_risk", "reveal_offset", "expiry_offset",
})
TRAINED_ARMS = (
    "TUAD-full",
    "current-only",
    "temporal-no-UAD-supervision",
    "oracle-UAD",
    "runner-only-support",
)
COLLECTION_ENTRYPOINT = PROJECT_ROOT / "scripts/collect_mf3zn_temporal_lattice.py"


def _load_temporal_records(path: Path) -> tuple[tuple[TemporalSequence, ...], str]:
    value = _json(path, "causal temporal record list")
    try:
        return temporal_record_list_from_mapping(value)
    except (TypeError, ValueError) as error:
        raise TUADProtocolError("invalid causal temporal record list") from error


def _causal_arrays_from_records(
    records: Sequence[TemporalSequence], plan: dict,
) -> dict[str, np.ndarray]:
    """Rebuild every production GRU tensor through the strict causal builder."""

    events = plan.get("seal", {}).get("events")
    if not isinstance(events, list) or not events:
        raise TUADProtocolError("sealed collection plan has no temporal population")
    by_identity: dict[tuple[str, str, str, int], dict] = {}
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("snapshot"), dict):
            raise TUADProtocolError("sealed temporal population schema drift")
        snapshot = event["snapshot"]
        identity = (
            snapshot.get("dataset"), snapshot.get("scene_id"),
            snapshot.get("episode_id"), snapshot.get("decision_step"),
        )
        if identity in by_identity:
            raise TUADProtocolError("sealed temporal population repeats an event")
        by_identity[identity] = event
    if len(records) != len(by_identity):
        raise TUADProtocolError("causal records do not cover the sealed lattice")

    feature_rows: list[np.ndarray] = []
    event_ids: list[str] = []
    scenes: list[str] = []
    datasets: list[str] = []
    episodes: list[str] = []
    lattices: list[str] = []
    physical_prefixes: list[str] = []
    seen: set[tuple[str, str, str, int]] = set()
    for record in records:
        identity = (
            record.dataset, record.scene_id, record.episode_id,
            record.decision_step,
        )
        event = by_identity.get(identity)
        if event is None or identity in seen:
            raise TUADProtocolError("causal temporal identity differs from collection seal")
        seen.add(identity)
        if record.steps[-1].step != record.decision_step:
            raise TUADProtocolError("causal temporal record omits the decision prefix")
        snapshot = event["snapshot"]
        score_by_action = dict(zip(
            snapshot["global_action_ids"], snapshot["policy_scores"], strict=True,
        ))
        executable = [
            snapshot["global_action_ids"][index]
            for index in snapshot["executable_action_indices"]
        ]
        ranked = sorted(executable, key=lambda value: (-score_by_action[value], value))
        current = record.steps[-1]
        if list(current.candidate_action_ids) != ranked:
            raise TUADProtocolError(
                "causal candidate support/rank differs from the frozen snapshot"
            )
        runner_id = event["alternative_action_ids"][0]
        expected_margin = abs(
            float(score_by_action[snapshot["native_action_id"]])
            - float(score_by_action[runner_id])
        )
        if (
            current.native_action_id != snapshot["native_action_id"]
            or float(current.policy_features[NATIVE_SCORE_INDEX])
            != float(score_by_action[snapshot["native_action_id"]])
            or float(current.policy_features[NATIVE_MARGIN_INDEX])
            != expected_margin
        ):
            raise TUADProtocolError("causal native policy state differs from the seal")
        feature_rows.append(np.asarray(causal_sequence_features(record), dtype=np.float32))
        event_ids.append(canonical_audit_event_id(*identity))
        datasets.append(record.dataset)
        scenes.append(record.scene_id)
        episodes.append(record.episode_id)
        lattices.append(event["lattice_id"])
        physical_prefixes.append(snapshot["native_prefix_sha256"])

    widths = {value.shape[1] for value in feature_rows}
    if len(widths) != 1:
        raise TUADProtocolError("causal feature width changes across records")
    maximum_steps = max(len(value) for value in feature_rows)
    sequence_features = np.zeros(
        (len(records), maximum_steps, next(iter(widths))), dtype=np.float32,
    )
    sequence_mask = np.zeros((len(records), maximum_steps), dtype=bool)
    for row, value in enumerate(feature_rows):
        sequence_features[row, : len(value)] = value
        sequence_mask[row, : len(value)] = True
    return {
        "event_id": np.asarray(event_ids),
        "scene_id": np.asarray(scenes),
        "dataset": np.asarray(datasets),
        "episode_id": np.asarray(episodes),
        "lattice_id": np.asarray(lattices),
        "native_prefix_sha256": np.asarray(physical_prefixes),
        "sequence_features": sequence_features,
        "sequence_mask": sequence_mask,
    }


def _action_arrays_from_sealed_sources(
    records: Sequence[TemporalSequence],
    causal: dict[str, np.ndarray],
    plan: dict,
    exact: dict,
) -> dict[str, np.ndarray]:
    """Derive model actions and exact labels; accept no caller-supplied tensor."""

    events = plan.get("seal", {}).get("events")
    outcomes = exact.get("outcomes")
    if not isinstance(events, list) or not isinstance(outcomes, list):
        raise TUADProtocolError("sealed action/outcome inventory is missing")
    event_by_lattice = {
        event.get("lattice_id"): event
        for event in events if isinstance(event, dict)
    }
    outcome_by_arm: dict[tuple[str, str], dict] = {}
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise TUADProtocolError("exact outcome inventory schema drift")
        key = (outcome.get("lattice_id"), outcome.get("action_id"))
        if key in outcome_by_arm:
            raise TUADProtocolError("exact outcome inventory repeats an arm")
        outcome_by_arm[key] = outcome

    rows = len(records)
    maximum_actions = max(
        len(event["alternative_action_ids"]) + 1
        for event in event_by_lattice.values()
    )
    if maximum_actions not in {2, 3}:
        raise TUADProtocolError("sealed action lattice width drift")
    action_widths = {
        record.steps[-1].action_embeddings.shape[1] for record in records
    }
    if len(action_widths) != 1:
        raise TUADProtocolError("action embedding width changes across records")
    action_width = next(iter(action_widths))
    native_embedding = np.zeros((rows, action_width), dtype=np.float32)
    action_embedding = np.zeros(
        (rows, maximum_actions, action_width), dtype=np.float32,
    )
    action_features = np.zeros((rows, maximum_actions, 1), dtype=np.float64)
    action_mask = np.zeros((rows, maximum_actions), dtype=bool)
    is_native = np.zeros((rows, maximum_actions), dtype=bool)
    utility = np.zeros((rows, maximum_actions), dtype=np.float64)
    catastrophic = np.zeros((rows, maximum_actions), dtype=bool)
    proposal_score = np.zeros(rows, dtype=np.float64)
    native_margin = np.zeros(rows, dtype=np.float64)
    action_id_rows: list[list[str]] = []

    lattices = causal["lattice_id"].astype(str)
    for row, (record, lattice_id) in enumerate(zip(records, lattices, strict=True)):
        event = event_by_lattice.get(lattice_id)
        if event is None or not isinstance(event.get("snapshot"), dict):
            raise TUADProtocolError("causal record has no sealed action event")
        snapshot = event["snapshot"]
        action_ids = [snapshot["native_action_id"], *event["alternative_action_ids"]]
        current = record.steps[-1]
        embedding_by_id = dict(zip(
            current.candidate_action_ids, current.action_embeddings, strict=True,
        ))
        score_by_id = dict(zip(
            snapshot["global_action_ids"], snapshot["policy_scores"], strict=True,
        ))
        native_id = snapshot["native_action_id"]
        runner_id = event["alternative_action_ids"][0]
        native_score = float(score_by_id[native_id])
        proposal_score[row] = float(score_by_id[runner_id])
        native_margin[row] = abs(native_score - proposal_score[row])
        padded_ids = [*action_ids, *([""] * (maximum_actions - len(action_ids)))]
        action_id_rows.append(padded_ids)
        for column, action_id in enumerate(action_ids):
            outcome = outcome_by_arm.get((lattice_id, action_id))
            if outcome is None:
                raise TUADProtocolError("exact outcome inventory omits a sealed arm")
            expected_type = "native" if column == 0 else "treatment"
            delta = outcome.get("delta_utility")
            catastrophe = outcome.get("catastrophic")
            if (
                outcome.get("arm_type") != expected_type
                or isinstance(delta, bool)
                or not isinstance(delta, (int, float))
                or not np.isfinite(float(delta))
                or type(catastrophe) is not bool
                or (column == 0 and (float(delta) != 0.0 or catastrophe))
            ):
                raise TUADProtocolError("exact outcome value/schema drift")
            action_embedding[row, column] = np.asarray(
                embedding_by_id[action_id], dtype=np.float32,
            )
            action_features[row, column, 0] = float(score_by_id[action_id]) - native_score
            action_mask[row, column] = True
            is_native[row, column] = column == 0
            utility[row, column] = float(delta)
            catastrophic[row, column] = catastrophe
        native_embedding[row] = action_embedding[row, 0]
    if len(outcome_by_arm) != int(action_mask.sum()):
        raise TUADProtocolError("exact outcome inventory contains an unsealed arm")
    return {
        "event_id": np.asarray(causal["event_id"]).astype(str),
        "lattice_id": lattices,
        "action_id": np.asarray(action_id_rows),
        "action_list_commitment_sha256": np.asarray(
            exact["action_list_commitment_sha256"]
        ),
        "native_embedding": native_embedding,
        "action_embedding": action_embedding,
        "action_features": action_features,
        "action_mask": action_mask,
        "is_native": is_native,
        "delta_utility": utility,
        "catastrophic": catastrophic,
        "proposal_score": proposal_score,
        "native_margin": native_margin,
    }


def _load_npz(path: Path, keys: frozenset[str], name: str) -> dict[str, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise TUADProtocolError(f"invalid {name}: {path}")
    with np.load(path, allow_pickle=False) as source:
        observed = set(source.files)
        if observed != keys:
            raise TUADProtocolError(
                f"{name} schema drift; missing={sorted(keys - observed)}, "
                f"extra={sorted(observed - keys)}"
            )
        return {key: np.array(source[key], copy=True) for key in keys}


def _strings(value: np.ndarray, rows: int | None, name: str) -> np.ndarray:
    if value.ndim != 1 or value.dtype.kind not in "US" or len(value) == 0:
        raise TUADProtocolError(f"{name} must be a nonempty string vector")
    result = value.astype(str)
    if rows is not None and len(result) != rows:
        raise TUADProtocolError(f"{name} has the wrong length")
    if any(not item for item in result.tolist()):
        raise TUADProtocolError(f"{name} contains an empty value")
    return result


def _string_scalar(value: np.ndarray, name: str) -> str:
    if value.ndim != 0 or value.dtype.kind not in "US" or not str(value.item()):
        raise TUADProtocolError(f"{name} must be a nonempty string scalar")
    return str(value.item())


def _json(path: Path, name: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise TUADProtocolError(f"invalid {name}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TUADProtocolError(f"{name} must be a JSON object")
    return value


def _project_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise TUADProtocolError(f"{name} project-relative path drift")
    root = PROJECT_ROOT.resolve()
    path = (root / value).resolve()
    if root not in path.parents:
        raise TUADProtocolError(f"{name} escaped project root")
    return path


def _require_project_local(path: Path, name: str) -> None:
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    if root not in resolved.parents:
        raise TUADProtocolError(f"{name} escaped project root")


def _verify_inventory_item(item: object, name: str) -> Path:
    if not isinstance(item, dict):
        raise TUADProtocolError(f"{name} provenance is missing")
    path = _project_path(item.get("path"), name)
    if (
        not path.is_file()
        or path.is_symlink()
        or item.get("sha256") != sha256_file(path)
        or ("bytes" in item and item.get("bytes") != path.stat().st_size)
    ):
        raise TUADProtocolError(f"{name} source drift")
    return path


def _recompute_exact_audit(exact: dict) -> dict:
    plan_path = _project_path(exact.get("collection_plan_path"), "collection plan")
    result_path = _project_path(exact.get("collection_result_path"), "collection result")
    spec = importlib.util.spec_from_file_location(
        "sealed_mf3zn_collection_validator_for_training", COLLECTION_ENTRYPOINT
    )
    if spec is None or spec.loader is None:
        raise TUADProtocolError("cannot load sealed exact-lattice validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.validate_collection_result(plan_path, result_path)
    except Exception as error:
        raise TUADProtocolError("exact-lattice audit revalidation failed") from error


def _verify_gates(
    protocol_path: Path,
    identifiability_path: Path,
    exact_audit_path: Path,
) -> tuple[dict, dict, dict]:
    protocol = verify_protocol(protocol_path, root=PROJECT_ROOT)
    identifiability = _json(identifiability_path, "identifiability result")
    if (
        identifiability.get("status") != "MF3ZN_IDENTIFIABILITY_PASS"
        or identifiability.get("collection_authorized") is not True
        or identifiability.get("public_authorization") is not False
        or identifiability.get("provenance", {}).get("protocol", {}).get("sha256")
        != sha256_file(protocol_path)
    ):
        raise TUADProtocolError("Stop A blocks TUAD training or provenance drifted")
    provenance = identifiability.get("provenance", {})
    for field in ("causal_probe", "oracle_labels", "label_reviews"):
        _verify_inventory_item(provenance.get(field), f"identifiability {field}")
    exact = _json(exact_audit_path, "exact-lattice audit")
    if (
        exact.get("schema_version") != "revealnav-mf3zn-teal-exact-audit/1"
        or exact.get("status") != "MF3ZN_EXACT_LATTICE_AUDIT_PASS"
        or exact.get("public_authorization") is not False
    ):
        raise TUADProtocolError("exact action-lattice audit did not pass")
    if _recompute_exact_audit(exact) != exact:
        raise TUADProtocolError("exact action-lattice audit artifact drift")
    plan_path = _project_path(exact.get("collection_plan_path"), "collection plan")
    plan = _json(plan_path, "collection plan")
    if (
        plan.get("protocol", {}).get("sha256") != sha256_file(protocol_path)
        or plan.get("identifiability_result", {}).get("sha256")
        != sha256_file(identifiability_path)
        or _project_path(
            plan.get("identifiability_result", {}).get("path"),
            "plan identifiability result",
        ).resolve()
        != identifiability_path.resolve()
    ):
        raise TUADProtocolError(
            "training gates differ from the sealed collection provenance"
        )
    return protocol, identifiability, exact


def _validate_training_arrays(
    causal: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    event_id = _strings(causal["event_id"], None, "event_id")
    rows = len(event_id)
    if len(set(event_id.tolist())) != rows:
        raise TUADProtocolError("duplicate training event identity")
    for source, name in ((oracle, "oracle"), (action, "action")):
        other = _strings(source["event_id"], rows, f"{name} event_id")
        if not np.array_equal(event_id, other):
            raise TUADProtocolError(f"causal/{name} event identities are not aligned")
    scenes = _strings(causal["scene_id"], rows, "scene_id")
    datasets = _strings(causal["dataset"], rows, "dataset")
    episodes = _strings(causal["episode_id"], rows, "episode_id")
    lattices = _strings(causal["lattice_id"], rows, "lattice_id")
    prefixes = _strings(causal["native_prefix_sha256"], rows, "native prefix")
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in prefixes.tolist()
    ):
        raise TUADProtocolError("causal native prefix digest drift")
    if set(datasets.tolist()) != {"R2R", "RxR"}:
        raise TUADProtocolError("training must report R2R and RxR separately")

    features = np.asarray(causal["sequence_features"])
    sequence_mask = np.asarray(causal["sequence_mask"])
    if (
        features.ndim != 3
        or features.shape[0] != rows
        or features.shape[1] < 1
        or features.shape[2] < 1
        or not np.issubdtype(features.dtype, np.floating)
        or not np.isfinite(features).all()
        or sequence_mask.dtype != np.bool_
        or sequence_mask.shape != features.shape[:2]
        or np.any(~sequence_mask[:, :-1] & sequence_mask[:, 1:])
        or np.any(sequence_mask.sum(axis=1) == 0)
    ):
        raise TUADProtocolError("invalid causal sequence tensors")
    temporal_shape = sequence_mask.shape
    for key in (
        "target_in_set", "candidate_separated", "evidence_closed",
        "reveal_event", "expiry_event",
    ):
        value = np.asarray(oracle[key])
        if (
            value.shape != temporal_shape
            or value.dtype != np.bool_
        ):
            raise TUADProtocolError(f"invalid oracle tensor {key}")
    for key in ("factor_mask", "reveal_at_risk", "expiry_at_risk"):
        value = np.asarray(oracle[key])
        if value.shape != temporal_shape or value.dtype != np.bool_:
            raise TUADProtocolError(f"invalid oracle mask {key}")
    if any(
        np.any(np.asarray(oracle[key]) & ~sequence_mask)
        for key in (
            "target_in_set", "candidate_separated", "evidence_closed",
            "reveal_event", "expiry_event", "factor_mask",
            "reveal_at_risk", "expiry_at_risk",
        )
    ):
        raise TUADProtocolError("oracle supervision is nonzero outside causal prefixes")
    if (
        np.any(oracle["candidate_separated"] & ~oracle["target_in_set"])
        or np.any(oracle["evidence_closed"] & ~oracle["target_in_set"])
        or np.any(oracle["reveal_event"] & ~oracle["reveal_at_risk"])
        or np.any(oracle["expiry_event"] & ~oracle["expiry_at_risk"])
    ):
        raise TUADProtocolError("oracle factor/hazard semantics drift")
    audit_target = np.asarray(oracle["delta_utility"])
    if (
        audit_target.ndim != 1
        or len(audit_target) != rows
        or not np.issubdtype(audit_target.dtype, np.floating)
        or not np.isfinite(audit_target).all()
    ):
        raise TUADProtocolError("invalid sealed Audit-A utility target")
    for key in ("reveal_offset", "expiry_offset"):
        value = np.asarray(oracle[key])
        if (
            value.ndim != 1
            or len(value) != rows
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
        ):
            raise TUADProtocolError(f"invalid oracle offset {key}")
    if not np.array_equal(oracle["factor_mask"], sequence_mask):
        raise TUADProtocolError("every causal prefix requires oracle factor labels")
    _derived_final_uad(oracle, sequence_mask)

    native = np.asarray(action["native_embedding"])
    actions = np.asarray(action["action_embedding"])
    action_features = np.asarray(action["action_features"])
    action_mask = np.asarray(action["action_mask"])
    is_native = np.asarray(action["is_native"])
    utility = np.asarray(action["delta_utility"])
    catastrophic = np.asarray(action["catastrophic"])
    proposal_score = np.asarray(action["proposal_score"])
    native_margin = np.asarray(action["native_margin"])
    action_lattices = _strings(action["lattice_id"], rows, "action lattice_id")
    if not np.array_equal(lattices, action_lattices):
        raise TUADProtocolError("causal/action lattice identity mismatch")
    action_ids = np.asarray(action["action_id"])
    if (
        native.ndim != 2 or native.shape[0] != rows
        or actions.ndim != 3 or actions.shape[0] != rows
        or actions.shape[2] != native.shape[1]
        or action_features.ndim != 3 or action_features.shape[:2] != actions.shape[:2]
        or action_features.shape[2] != 1
        or action_mask.dtype != np.bool_ or action_mask.shape != actions.shape[:2]
        or is_native.dtype != np.bool_ or is_native.shape != action_mask.shape
        or utility.shape != action_mask.shape
        or catastrophic.dtype != np.bool_ or catastrophic.shape != action_mask.shape
        or proposal_score.ndim != 1 or len(proposal_score) != rows
        or native_margin.ndim != 1 or len(native_margin) != rows
        or action_ids.dtype.kind not in "US" or action_ids.shape != action_mask.shape
        or not all(np.issubdtype(value.dtype, np.floating) for value in (native, actions, action_features, utility))
        or not all(np.isfinite(value).all() for value in (
            native, actions, action_features, utility, proposal_score, native_margin
        ))
        or np.any(is_native.sum(axis=1) != 1)
        or np.any(is_native & ~action_mask)
        or not np.allclose(
            actions[is_native], native, rtol=0.0, atol=0.0
        )
        or np.any(utility[is_native] != 0.0)
        or np.any(catastrophic[is_native])
        or np.any(np.abs(utility[action_mask]) > 1.0)
        or np.any(
            catastrophic[action_mask & ~is_native]
            != (utility[action_mask & ~is_native] <= CATASTROPHIC_THRESHOLD)
        )
        or np.any(~action_mask & ((utility != 0.0) | catastrophic))
        or np.any(actions[~action_mask] != 0.0)
        or np.any(action_features[~action_mask] != 0.0)
        or np.any(native_margin < 0.0)
        or np.any(action_mask.sum(axis=1) < 2)
    ):
        raise TUADProtocolError("invalid exact action-value tensors")
    for row in range(rows):
        live_ids = action_ids[row, action_mask[row]].astype(str).tolist()
        padded_ids = action_ids[row, ~action_mask[row]].astype(str).tolist()
        if (
            any(not value for value in live_ids)
            or len(live_ids) != len(set(live_ids))
            or any(value for value in padded_ids)
        ):
            raise TUADProtocolError("sealed action ID/mask alignment drift")
    return rows, event_id, scenes, datasets, episodes, lattices


def _verify_exact_action_binding(
    causal: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    exact: dict,
) -> None:
    commitment = _string_scalar(
        np.asarray(action["action_list_commitment_sha256"]),
        "action-list commitment",
    )
    if commitment != exact.get("action_list_commitment_sha256"):
        raise TUADProtocolError("training actions use another sealed action list")
    plan_path = _project_path(exact.get("collection_plan_path"), "collection plan")
    plan = _json(plan_path, "collection plan")
    events = plan.get("seal", {}).get("events")
    if not isinstance(events, list):
        raise TUADProtocolError("sealed collection plan event inventory drift")
    expected = {}
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("snapshot"), dict):
            raise TUADProtocolError("sealed action event schema drift")
        action_ids = [event["native_action_id"], *event["alternative_action_ids"]]
        snapshot = event["snapshot"]
        score_by_action = dict(zip(
            snapshot["global_action_ids"], snapshot["policy_scores"], strict=True,
        ))
        runner_id = event["alternative_action_ids"][0]
        expected[event["lattice_id"]] = {
            "action_ids": action_ids,
            "native_prefix_sha256": snapshot["native_prefix_sha256"],
            "dataset": snapshot["dataset"],
            "scene_id": snapshot["scene_id"],
            "episode_id": snapshot["episode_id"],
            "event_id": canonical_audit_event_id(
                snapshot["dataset"], snapshot["scene_id"],
                snapshot["episode_id"], snapshot["decision_step"],
            ),
            "proposal_score": float(score_by_action[runner_id]),
            "native_margin": abs(
                float(score_by_action[snapshot["native_action_id"]])
                - float(score_by_action[runner_id])
            ),
            "action_score_difference": [
                float(score_by_action[action_id])
                - float(score_by_action[snapshot["native_action_id"]])
                for action_id in action_ids
            ],
        }
    lattices = causal["lattice_id"].astype(str)
    if len(expected) != len(lattices) or set(expected) != set(lattices.tolist()):
        raise TUADProtocolError("training population does not cover the exact lattice")
    action_ids = action["action_id"].astype(str)
    action_mask = np.asarray(action["action_mask"], dtype=bool)
    prefixes = causal["native_prefix_sha256"].astype(str)
    event_ids = causal["event_id"].astype(str)
    datasets = causal["dataset"].astype(str)
    scenes = causal["scene_id"].astype(str)
    episodes = causal["episode_id"].astype(str)
    proposal_scores = np.asarray(action["proposal_score"], dtype=np.float64)
    native_margins = np.asarray(action["native_margin"], dtype=np.float64)
    action_features = np.asarray(action["action_features"], dtype=np.float64)
    is_native = np.asarray(action["is_native"], dtype=bool)
    for row, lattice_id in enumerate(lattices):
        value = expected[lattice_id]
        live = action_mask[row]
        if (
            action_ids[row, live].tolist() != value["action_ids"]
            or is_native[row, live].tolist()
            != [True, *([False] * (len(value["action_ids"]) - 1))]
            or prefixes[row] != value["native_prefix_sha256"]
            or event_ids[row] != value["event_id"]
            or datasets[row] != value["dataset"]
            or scenes[row] != value["scene_id"]
            or episodes[row] != value["episode_id"]
            or proposal_scores[row] != value["proposal_score"]
            or native_margins[row] != value["native_margin"]
            or action_features[row, live, 0].tolist()
            != value["action_score_difference"]
        ):
            raise TUADProtocolError(
                "training identity, baseline feature, action order, or prefix differs from seal"
            )


def _tensor(value: np.ndarray, device: torch.device, *, boolean: bool = False) -> torch.Tensor:
    return torch.as_tensor(
        value,
        dtype=torch.bool if boolean else torch.float32,
        device=device,
    )


def _targets(
    oracle: dict[str, np.ndarray], rows: np.ndarray, device: torch.device,
) -> RevealExpiryTargets:
    return RevealExpiryTargets(
        target_in_set=_tensor(oracle["target_in_set"][rows], device),
        separation=_tensor(oracle["candidate_separated"][rows], device),
        evidence=_tensor(oracle["evidence_closed"][rows], device),
        reveal_event=_tensor(oracle["reveal_event"][rows], device),
        expiry_event=_tensor(oracle["expiry_event"][rows], device),
        factor_mask=_tensor(oracle["factor_mask"][rows], device, boolean=True),
        reveal_at_risk=_tensor(oracle["reveal_at_risk"][rows], device, boolean=True),
        expiry_at_risk=_tensor(oracle["expiry_at_risk"][rows], device, boolean=True),
    )


def _seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parameter_sha256(*models: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for model in models:
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _current_only_view(
    features: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a true single-prefix view with all history dynamics neutralized."""

    structural_width = len(STRUCTURAL_FEATURE_NAMES)
    if features.shape[-1] <= structural_width:
        raise TUADProtocolError("causal feature width omits the frozen structural suffix")
    lengths = mask.sum(dim=1)
    rows = torch.arange(len(lengths), device=features.device)
    current = features[rows, lengths - 1].clone()
    candidate_count = current[:, -structural_width].clone()
    current[:, -structural_width:] = 0.0
    current[:, -structural_width] = candidate_count
    # An isolated snapshot has no preceding prefix.  Birth count therefore
    # equals its current candidate count and empty-to-current Jaccard is one.
    current[:, -structural_width + 1] = candidate_count
    current[:, -2] = 1.0
    return current[:, None, :], torch.ones(
        len(lengths), 1, dtype=torch.bool, device=features.device
    )


def _train_supervised_encoder(
    features: torch.Tensor,
    mask: torch.Tensor,
    oracle: dict[str, np.ndarray],
    fit: np.ndarray,
    *,
    seed: int,
    current_only: bool,
    device: torch.device,
    epochs: int = STAGE_1_EPOCHS,
) -> tuple[TemporalRevealExpiryEncoder, float]:
    _seed(seed)
    model = TemporalRevealExpiryEncoder(features.shape[-1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ADAM_LEARNING_RATE, weight_decay=FIXED_WEIGHT_DECAY
    )
    fit_index = np.flatnonzero(fit)
    fit_features = features[fit_index]
    fit_mask = mask[fit_index]
    targets = _targets(oracle, fit_index, device)
    if current_only:
        fit_features, fit_mask = _current_only_view(fit_features, fit_mask)
        last = torch.as_tensor(mask[fit_index].sum(dim=1).cpu().numpy() - 1, dtype=torch.long)

        def final(value: torch.Tensor) -> torch.Tensor:
            rows = torch.arange(len(last), device=value.device)
            return value[rows, last.to(value.device)][:, None]

        targets = RevealExpiryTargets(
            target_in_set=final(targets.target_in_set),
            separation=final(targets.separation),
            evidence=final(targets.evidence),
            reveal_event=final(targets.reveal_event),
            expiry_event=final(targets.expiry_event),
            factor_mask=final(targets.factor_mask),
            reveal_at_risk=final(targets.reveal_at_risk),
            expiry_at_risk=final(targets.expiry_at_risk),
        )
    objective = TemporalRevealExpiryLoss()
    loss = torch.tensor(float("nan"), device=device)
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(fit_features, fit_mask), targets, fit_mask)
        loss.backward()
        optimizer.step()
    freeze_temporal_encoder(model)
    return model, float(loss.detach().cpu())


def _states(
    model: TemporalRevealExpiryEncoder,
    features: torch.Tensor,
    mask: torch.Tensor,
    *,
    current_only: bool,
) -> torch.Tensor:
    model_features, model_mask = (
        _current_only_view(features, mask) if current_only else (features, mask)
    )
    with torch.no_grad():
        return last_causal_state(model(model_features, model_mask), model_mask)


def _train_action_head(
    state: torch.Tensor,
    action: dict[str, np.ndarray],
    fit: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int = STAGE_2_EPOCHS,
    support_mask: np.ndarray | None = None,
) -> tuple[NativeAnchoredActionValue, float]:
    _seed(seed + 10_000)
    native = _tensor(action["native_embedding"], device)
    actions = _tensor(action["action_embedding"], device)
    features = _tensor(action["action_features"], device)
    effective_mask = np.asarray(action["action_mask"], dtype=bool)
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != effective_mask.shape or np.any(support & ~effective_mask):
            raise TUADProtocolError("action-head support is outside the sealed lattice")
        effective_mask = effective_mask & support
    action_mask = _tensor(effective_mask, device, boolean=True)
    is_native = _tensor(action["is_native"], device, boolean=True)
    target = _tensor(action["delta_utility"], device)
    head = NativeAnchoredActionValue(
        native.shape[-1], features.shape[-1]
    ).to(device)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=ADAM_LEARNING_RATE, weight_decay=FIXED_WEIGHT_DECAY
    )
    fit_index = torch.as_tensor(np.flatnonzero(fit), dtype=torch.long, device=device)
    loss = torch.tensor(float("nan"), device=device)
    for _ in range(int(epochs)):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        value = head(
            state[fit_index], native[fit_index], actions[fit_index],
            features[fit_index], is_native=is_native[fit_index],
        )
        loss = native_anchored_huber_loss(
            value, target[fit_index], action_mask[fit_index], is_native[fit_index]
        )
        loss.backward()
        optimizer.step()
    head.eval()
    return head, float(loss.detach().cpu())


def _predict_action_head(
    head: NativeAnchoredActionValue,
    state: torch.Tensor,
    action: dict[str, np.ndarray],
    rows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    index = torch.as_tensor(rows, dtype=torch.long, device=device)
    with torch.no_grad():
        value = head(
            state[index],
            _tensor(action["native_embedding"][rows], device),
            _tensor(action["action_embedding"][rows], device),
            _tensor(action["action_features"][rows], device),
            is_native=_tensor(action["is_native"][rows], device, boolean=True),
        )
    return value.cpu().numpy()


def _oracle_state(
    oracle: dict[str, np.ndarray], mask: np.ndarray, device: torch.device,
) -> torch.Tensor:
    uad_state = _derived_final_uad(oracle, np.asarray(mask, dtype=bool))
    values = np.column_stack([
        uad_state == "U",
        uad_state == "A",
        uad_state == "D",
        oracle["reveal_offset"],
        oracle["expiry_offset"],
    ]).astype(np.float32)
    state = np.zeros((len(mask), 64), dtype=np.float32)
    state[:, : values.shape[1]] = values
    return _tensor(state, device)


def _derived_final_uad(
    oracle: dict[str, np.ndarray], sequence_mask: np.ndarray,
) -> np.ndarray:
    """Derive decision-time U/A/D; never trust a supplied class label."""

    result = []
    for row, length in enumerate(np.asarray(sequence_mask, dtype=bool).sum(axis=1)):
        stop = int(length)
        states = derive_uad(
            np.asarray(oracle["target_in_set"])[row, :stop].astype(bool),
            np.asarray(oracle["candidate_separated"])[row, :stop].astype(bool),
            np.asarray(oracle["evidence_closed"])[row, :stop].astype(bool),
        )
        result.append(states[-1].value)
    return np.asarray(result)


def fixed_scene_oof_train(
    causal: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    *,
    device: torch.device,
    stage_1_epochs: int = STAGE_1_EPOCHS,
    stage_2_epochs: int = STAGE_2_EPOCHS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train every declared learned control with one common OOF partition.

    Epoch arguments exist for small correctness fixtures; the production CLI
    never exposes them and always supplies the sealed constants.
    """

    rows, event_id, scenes, datasets, episodes, lattices = _validate_training_arrays(
        causal, oracle, action
    )
    folds, mapping = assign_tuad_scene_folds(scenes)
    validate_lattice_fold_integrity(scenes, episodes, lattices, folds)
    sequence = _tensor(causal["sequence_features"], device)
    sequence_mask = _tensor(causal["sequence_mask"], device, boolean=True)
    action_count = action["action_mask"].shape[1]
    per_seed = {
        arm: np.full((len(FIXED_SEEDS), rows, action_count), np.nan, dtype=np.float32)
        for arm in (
            "TUAD-full", "current-only", "temporal-no-UAD-supervision",
            "oracle-UAD", "runner-only-support",
        )
    }
    sealed_action_mask = np.asarray(action["action_mask"], dtype=bool)
    sealed_is_native = np.asarray(action["is_native"], dtype=bool)
    runner_support_mask = sealed_is_native.copy()
    for row in range(rows):
        alternatives = np.flatnonzero(
            sealed_action_mask[row] & ~sealed_is_native[row]
        )
        runner_support_mask[row, alternatives[0]] = True
    diagnostics = []
    for fold in range(OUTER_FOLDS):
        fit = folds != fold
        held = np.flatnonzero(folds == fold)
        for seed_index, seed in enumerate(FIXED_SEEDS):
            for arm, current_only in (("TUAD-full", False), ("current-only", True)):
                encoder, stage_1_loss = _train_supervised_encoder(
                    sequence, sequence_mask, oracle, fit,
                    seed=seed, current_only=current_only, device=device,
                    epochs=stage_1_epochs,
                )
                state = _states(
                    encoder, sequence, sequence_mask, current_only=current_only
                )
                head, stage_2_loss = _train_action_head(
                    state, action, fit, seed=seed, device=device,
                    epochs=stage_2_epochs,
                )
                per_seed[arm][seed_index, held] = _predict_action_head(
                    head, state, action, held, device
                )
                diagnostics.append({
                    "fold": fold, "seed": seed, "arm": arm,
                    "stage_1_loss": stage_1_loss, "stage_2_loss": stage_2_loss,
                    "parameter_sha256": _parameter_sha256(encoder, head),
                    "temporal_encoder_utility_supervision": False,
                })
                if arm == "TUAD-full":
                    runner_head, runner_loss = _train_action_head(
                        state, action, fit, seed=seed, device=device,
                        epochs=stage_2_epochs,
                        support_mask=runner_support_mask,
                    )
                    runner_prediction = _predict_action_head(
                        runner_head, state, action, held, device
                    )
                    held_support = runner_support_mask[held]
                    runner_prediction[~held_support] = 0.0
                    per_seed["runner-only-support"][seed_index, held] = (
                        runner_prediction
                    )
                    diagnostics.append({
                        "fold": fold, "seed": seed,
                        "arm": "runner-only-support",
                        "stage_1_loss": stage_1_loss,
                        "stage_2_loss": runner_loss,
                        "parameter_sha256": _parameter_sha256(
                            encoder, runner_head
                        ),
                        "temporal_encoder_utility_supervision": False,
                        "action_support": "native_plus_frozen_runner_only",
                    })

            # No-UAD control: causal GRU is trained only by exact utility.
            _seed(seed)
            encoder = TemporalRevealExpiryEncoder(sequence.shape[-1]).to(device)
            native = _tensor(action["native_embedding"], device)
            actions = _tensor(action["action_embedding"], device)
            action_features = _tensor(action["action_features"], device)
            action_mask = _tensor(action["action_mask"], device, boolean=True)
            is_native = _tensor(action["is_native"], device, boolean=True)
            target = _tensor(action["delta_utility"], device)
            head = NativeAnchoredActionValue(native.shape[-1], action_features.shape[-1]).to(device)
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(head.parameters()),
                lr=ADAM_LEARNING_RATE, weight_decay=FIXED_WEIGHT_DECAY,
            )
            fit_index = torch.as_tensor(np.flatnonzero(fit), dtype=torch.long, device=device)
            no_uad_loss = torch.tensor(float("nan"), device=device)
            for _ in range(int(stage_2_epochs)):
                encoder.train(); head.train(); optimizer.zero_grad(set_to_none=True)
                state = last_causal_state(
                    encoder(sequence[fit_index], sequence_mask[fit_index]),
                    sequence_mask[fit_index],
                )
                value = head(
                    state, native[fit_index], actions[fit_index],
                    action_features[fit_index], is_native=is_native[fit_index],
                )
                no_uad_loss = native_anchored_huber_loss(
                    value, target[fit_index], action_mask[fit_index], is_native[fit_index]
                )
                no_uad_loss.backward(); optimizer.step()
            encoder.eval(); head.eval()
            with torch.no_grad():
                held_index = torch.as_tensor(held, dtype=torch.long, device=device)
                state = last_causal_state(
                    encoder(sequence[held_index], sequence_mask[held_index]),
                    sequence_mask[held_index],
                )
                value = head(
                    state, native[held_index], actions[held_index],
                    action_features[held_index], is_native=is_native[held_index],
                )
            per_seed["temporal-no-UAD-supervision"][seed_index, held] = value.cpu().numpy()
            diagnostics.append({
                "fold": fold, "seed": seed, "arm": "temporal-no-UAD-supervision",
                "stage_1_loss": None, "stage_2_loss": float(no_uad_loss.detach().cpu()),
                "parameter_sha256": _parameter_sha256(encoder, head),
                "temporal_encoder_utility_supervision": True,
            })

            oracle_state = _oracle_state(oracle, causal["sequence_mask"], device)
            oracle_head, oracle_loss = _train_action_head(
                oracle_state, action, fit, seed=seed, device=device,
                epochs=stage_2_epochs,
            )
            per_seed["oracle-UAD"][seed_index, held] = _predict_action_head(
                oracle_head, oracle_state, action, held, device
            )
            diagnostics.append({
                "fold": fold, "seed": seed, "arm": "oracle-UAD",
                "stage_1_loss": None, "stage_2_loss": oracle_loss,
                "parameter_sha256": _parameter_sha256(oracle_head),
                "diagnostic_only": True,
                "temporal_encoder_utility_supervision": False,
            })

    action_mask = sealed_action_mask
    is_native = sealed_is_native
    outputs: dict[str, np.ndarray] = {
        "fold": folds,
        "event_id": event_id,
        "action_mask": action_mask,
        "is_native": is_native,
        "action_id": np.asarray(action["action_id"]).astype(str),
        "lattice_id": np.asarray(action["lattice_id"]).astype(str),
    }
    for arm, seeds in per_seed.items():
        if not np.isfinite(seeds[:, action_mask]).all():
            raise RuntimeError(f"{arm} outer OOF predictions are incomplete")
        seeds[:, ~action_mask] = 0.0
        median = np.median(seeds, axis=0)
        median[~action_mask] = 0.0
        median[is_native] = 0.0
        deployment_mask = (
            runner_support_mask if arm == "runner-only-support" else action_mask
        )
        chosen = choose_native_inclusive_action(
            torch.from_numpy(median), torch.from_numpy(deployment_mask),
            torch.from_numpy(is_native)
        ).numpy()
        key = arm.replace("-", "_")
        outputs[f"q_{key}"] = median
        outputs[f"chosen_{key}"] = chosen
        outputs[f"seed_q_{key}"] = seeds

    chosen_by_arm = {
        "TUAD-full": outputs["chosen_TUAD_full"],
        "current-only": outputs["chosen_current_only"],
        "temporal-no-UAD-supervision": outputs[
            "chosen_temporal_no_UAD_supervision"
        ],
        "oracle-UAD": outputs["chosen_oracle_UAD"],
        "runner-only-support": outputs["chosen_runner_only_support"],
    }
    policies = assemble_development_policies(
        chosen_by_arm,
        action["delta_utility"],
        action["catastrophic"],
        action["action_mask"],
        action["is_native"],
        action["proposal_score"],
        action["native_margin"],
        folds,
        datasets,
        event_id,
    )
    development_result = evaluate_tuad_development(
        policies,
        scenes,
        datasets,
        folds,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    return outputs, {
        "schema_version": "revealnav-mf3zn-fixed-oof-training-diagnostics/1",
        "method_id": "mf3zn_tuad_v1",
        "arms": list(TRAINED_ARMS),
        "fixed_seeds": list(FIXED_SEEDS),
        "ensemble_reduction": ENSEMBLE_REDUCTION,
        "scene_fold_mapping": mapping,
        "complete_five_fold_oof": True,
        "model_selection_performed": False,
        "checkpoint_written": False,
        "public_authorization": False,
        "development_result": development_result,
        "stop_b_applied": (
            development_result["status"] != "TUAD_DEVELOPMENT_PASS"
        ),
        "fits": diagnostics,
    }


def _atomic_outputs(prefix: Path, arrays: dict[str, np.ndarray], metadata: dict) -> tuple[Path, Path]:
    array_path = prefix.with_suffix(".npz")
    metadata_path = prefix.with_suffix(".json")
    if any(path.exists() or path.is_symlink() for path in (array_path, metadata_path)):
        raise TUADProtocolError("refusing to overwrite TUAD OOF output")
    array_path.parent.mkdir(parents=True, exist_ok=True)
    array_partial = array_path.with_name(array_path.name + ".part")
    metadata_partial = metadata_path.with_name(metadata_path.name + ".part")
    if any(path.exists() or path.is_symlink() for path in (array_partial, metadata_partial)):
        raise TUADProtocolError("stale TUAD OOF partial output")
    with array_partial.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    with metadata_partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(array_partial, array_path)
    os.replace(metadata_partial, metadata_path)
    return array_path, metadata_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--identifiability", type=Path, required=True)
    parser.add_argument("--exact-lattice-audit", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for path, name in (
        (args.protocol, "protocol"),
        (args.identifiability, "identifiability result"),
        (args.exact_lattice_audit, "exact-lattice audit"),
        (args.output_prefix, "training output"),
    ):
        _require_project_local(path, name)
    protocol, identifiability, exact_audit = _verify_gates(
        args.protocol, args.identifiability, args.exact_lattice_audit
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise TUADProtocolError("CUDA was requested but is unavailable")
    plan_path = _project_path(
        exact_audit.get("collection_plan_path"), "collection plan"
    )
    plan = _json(plan_path, "collection plan")
    causal_inventory = plan.get("causal_temporal_record_source")
    causal_source_path = _verify_inventory_item(
        causal_inventory, "causal temporal record list"
    )
    oracle_source_path = _verify_inventory_item(
        identifiability.get("provenance", {}).get("oracle_labels"),
        "identifiability oracle labels",
    )
    if causal_source_path.resolve() == oracle_source_path.resolve():
        raise TUADProtocolError("causal records and oracle labels are not isolated")
    records, source_identity = _load_temporal_records(causal_source_path)
    if source_identity != protocol["source_population"]["canonical_identity_sha256"]:
        raise TUADProtocolError("causal record population differs from the audited universe")
    causal = _causal_arrays_from_records(records, plan)
    oracle = _load_npz(oracle_source_path, ORACLE_KEYS, "oracle Stage-1 labels")
    action = _action_arrays_from_sealed_sources(
        records, causal, plan, exact_audit
    )
    _verify_exact_action_binding(causal, action, exact_audit)
    arrays, metadata = fixed_scene_oof_train(
        causal, oracle, action,
        device=torch.device(args.device),
        stage_1_epochs=protocol["model"]["stage_1"]["epochs"],
        stage_2_epochs=protocol["model"]["stage_2"]["epochs"],
        bootstrap_replicates=protocol["development"]["bootstrap"]["replicates"],
    )
    metadata["provenance"] = {
        "protocol_sha256": sha256_file(args.protocol),
        "identifiability_sha256": sha256_file(args.identifiability),
        "exact_lattice_audit_sha256": sha256_file(args.exact_lattice_audit),
        "causal_path": str(causal_source_path.relative_to(PROJECT_ROOT)),
        "causal_sha256": sha256_file(causal_source_path),
        "oracle_path": str(oracle_source_path.relative_to(PROJECT_ROOT)),
        "oracle_sha256": sha256_file(oracle_source_path),
        "actions_source": "recomputed_from_exact_lattice_audit_and_causal_records",
    }
    array_path, metadata_path = _atomic_outputs(args.output_prefix, arrays, metadata)
    development_status = metadata["development_result"]["status"]
    print(json.dumps({
        "status": development_status,
        "arrays": str(array_path),
        "metadata": str(metadata_path),
        "stop_b_applied": metadata["stop_b_applied"],
        "checkpoint_written": False,
        "public_authorization": False,
    }, indent=2, sort_keys=True))
    return 0 if development_status == "TUAD_DEVELOPMENT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
