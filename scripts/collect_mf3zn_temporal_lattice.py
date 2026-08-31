#!/usr/bin/env python3
"""Seal and audit MF3ZN temporal exact-action lattice collection.

This entry point has no public-split mode.  ``seal`` converts causal target-step
snapshots into a complete, immutable native+top-two task list only after the
identifiability gate passes.  ``validate`` audits externally produced native
and treatment traces against that list; it does not choose replacement arms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IDENTIFIABILITY_AUDIT_ENTRYPOINT = (
    PROJECT_ROOT / "scripts/audit_mf3zn_uad_identifiability.py"
)

from revealnav_mf3.temporal_exact_lattice import (  # noqa: E402
    CausalActionSnapshot,
    LatticeArmIdentity,
    seal_action_lattice,
    validate_exact_lattice_treatment,
)
from revealnav_mf3.temporal_uad_features import (  # noqa: E402
    NATIVE_MARGIN_INDEX,
    NATIVE_SCORE_INDEX,
)
from revealnav_mf3.temporal_uad_schema import (  # noqa: E402
    temporal_record_list_from_mapping,
)
from revealnav_mf3.tuad_protocol import (  # noqa: E402
    CATASTROPHIC_THRESHOLD,
    LATTICE_ID,
    OUTCOME_METRICS,
    TUADProtocolError,
    UTILITY_WEIGHTS,
    sha256_file,
    verify_protocol,
)


SNAPSHOT_INPUT_SCHEMA = "revealnav-mf3zn-causal-action-snapshot-list/1"
COLLECTION_PLAN_SCHEMA = "revealnav-mf3zn-teal-collection-plan/1"
COLLECTION_RESULT_SCHEMA = "revealnav-mf3zn-teal-collection-result/1"
EXACT_AUDIT_SCHEMA = "revealnav-mf3zn-teal-exact-audit/1"
TASK_METRICS_SCHEMA = "revealnav-mf3zn-task-metrics/1"
SNAPSHOT_INPUT_KEYS = frozenset({
    "schema_version",
    "status",
    "frozen_continuation_source",
    "native_baseline_source",
    "snapshots",
})
INVENTORY_KEYS = frozenset({"path", "bytes", "sha256"})
PLAN_KEYS = frozenset({
    "schema_version", "status", "method_id", "lattice_revision",
    "protocol", "identifiability_result", "causal_snapshot_source",
    "causal_temporal_record_source",
    "frozen_continuation_source", "native_baseline_source",
    "outcome_fields_used_for_selection", "treatment_results_read", "seal",
    "arms", "public_split_access",
})
RESULT_KEYS = frozenset({
    "schema_version", "action_list_commitment_sha256", "events",
})
TASK_METRICS_KEYS = frozenset({
    "schema_version", "dataset", "scene_id", "episode_id", "decision_step",
    "native_prefix_sha256", "action_id", "metrics",
    "task_metric_payload_read_by_worker", "public_split_access",
})
RESULT_EVENT_KEYS = frozenset({
    "lattice_id", "native_prior_intervention_count",
    "native_second_intervention_count",
    "native_baseline_sha256",
    "native_continuation_controller_sha256", "native_physical_trace",
    "native_decision_trace", "native_outcome_source", "treatments",
})
TREATMENT_KEYS = frozenset({
    "action_id", "prior_intervention_count", "second_intervention_count",
    "continuation_controller_sha256", "physical_trace", "decision_trace",
    "outcome_source",
})


def _json(path: Path, name: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise TUADProtocolError(f"invalid {name}: {path}")
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise TUADProtocolError(f"invalid {name} JSON") from error
    if not isinstance(value, dict):
        raise TUADProtocolError(f"{name} must be a JSON object")
    return value


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError as error:
        raise TUADProtocolError(f"artifact escaped project root: {path}") from error


def _project_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise TUADProtocolError(f"collection plan {name} path drift")
    root = PROJECT_ROOT.resolve()
    path = (root / value).resolve()
    if root not in path.parents:
        raise TUADProtocolError(f"collection plan {name} escaped project root")
    return path


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _verified_inventory(value: object, name: str) -> tuple[dict, Path]:
    """Validate one canonical project-local regular-file inventory."""

    if not isinstance(value, Mapping) or set(value) != INVENTORY_KEYS:
        raise TUADProtocolError(f"{name} inventory schema drift")
    declared_path = PROJECT_ROOT / str(value.get("path", ""))
    path = _project_path(value.get("path"), name)
    if not path.is_file() or declared_path.is_symlink():
        raise TUADProtocolError(f"{name} inventory is not a regular file")
    normalized = {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if dict(value) != normalized:
        raise TUADProtocolError(f"{name} inventory provenance drift")
    return normalized, path


def _inventory(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise TUADProtocolError(f"invalid inventory source: {path}")
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _finite_metric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TUADProtocolError(f"task metric {name} must be a finite real")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TUADProtocolError(f"task metric {name} must lie in [0,1]")
    return result


def _require_zero_count(value: object, name: str) -> None:
    if type(value) is not int or value != 0:
        raise TUADProtocolError(f"{name} must be the integer zero")


def _arm_outcome(
    inventory_value: object,
    event,
    action_id: str,
) -> tuple[dict[str, float], dict]:
    """Read and identity-check one immutable per-arm task-metric source."""

    inventory, path = _verified_inventory(
        inventory_value, f"outcome {event.lattice_id}/{action_id}"
    )
    value = _json(path, "per-arm task metrics")
    if set(value) != TASK_METRICS_KEYS or value.get("schema_version") != TASK_METRICS_SCHEMA:
        raise TUADProtocolError("per-arm task-metric source schema drift")
    expected_identity = {
        "dataset": event.dataset,
        "scene_id": event.scene_id,
        "episode_id": event.episode_id,
        "decision_step": event.decision_step,
        "native_prefix_sha256": event.native_prefix_sha256,
        "action_id": action_id,
    }
    if any(value.get(key) != expected for key, expected in expected_identity.items()):
        raise TUADProtocolError("per-arm task-metric identity drift")
    if (
        value.get("task_metric_payload_read_by_worker") is not False
        or value.get("public_split_access") is not False
    ):
        raise TUADProtocolError("task metrics crossed the controller/public boundary")
    raw_metrics = value.get("metrics")
    if not isinstance(raw_metrics, Mapping) or set(raw_metrics) != set(OUTCOME_METRICS):
        raise TUADProtocolError("per-arm task metric inventory drift")
    metrics = {
        name: _finite_metric(raw_metrics[name], name)
        for name in OUTCOME_METRICS
    }
    return metrics, inventory


def _utility(metrics: Mapping[str, float]) -> float:
    return float(sum(metrics[name] * weight for name, weight in UTILITY_WEIGHTS.items()))


def _outcome_record(
    event,
    action_id: str,
    arm_type: str,
    metrics: Mapping[str, float],
    native_metrics: Mapping[str, float],
    source: Mapping[str, object],
) -> dict:
    delta = {
        name: float(metrics[name] - native_metrics[name])
        for name in OUTCOME_METRICS
    }
    delta_utility = float(_utility(metrics) - _utility(native_metrics))
    if arm_type == "native":
        # Avoid signed-zero/platform subtraction drift in the anchor record.
        delta = {name: 0.0 for name in OUTCOME_METRICS}
        delta_utility = 0.0
    payload = {
        "lattice_id": event.lattice_id,
        "action_id": action_id,
        "arm_type": arm_type,
        "outcome_source": dict(source),
        "metrics": dict(metrics),
        "absolute_utility": _utility(metrics),
        "native_relative_delta": delta,
        "delta_utility": delta_utility,
        "catastrophic": bool(
            arm_type != "native" and delta_utility <= CATASTROPHIC_THRESHOLD
        ),
    }
    return {
        **payload,
        "outcome_commitment_sha256": _canonical_sha256(payload),
    }


def _atomic_json(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise TUADProtocolError(f"refusing to overwrite sealed collection artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise TUADProtocolError(f"stale collection partial: {partial}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _recompute_identifiability(
    protocol_path: Path,
    provenance: Mapping[str, object],
) -> dict:
    """Rerun the source-sealed deterministic Stop-A audit from its sources."""

    source_paths = {}
    for field in ("causal_probe", "oracle_labels", "label_reviews"):
        item = provenance.get(field)
        if not isinstance(item, Mapping):
            raise TUADProtocolError(f"identifiability {field} provenance drift")
        source_paths[field] = _project_path(item.get("path"), field)
    spec = importlib.util.spec_from_file_location(
        "sealed_mf3zn_identifiability_for_collection",
        IDENTIFIABILITY_AUDIT_ENTRYPOINT,
    )
    if spec is None or spec.loader is None:
        raise TUADProtocolError("cannot load sealed identifiability audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.run_audit(
            protocol_path,
            source_paths["causal_probe"],
            source_paths["oracle_labels"],
            source_paths["label_reviews"],
        )
    except Exception as error:
        raise TUADProtocolError("identifiability audit revalidation failed") from error


def _verify_identifiability(path: Path, protocol_path: Path) -> dict:
    result = _json(path, "identifiability result")
    if (
        result.get("schema_version") != "revealnav-mf3zn-identifiability-result/1"
        or result.get("status") != "MF3ZN_IDENTIFIABILITY_PASS"
        or result.get("collection_authorized") is not True
        or result.get("public_authorization") is not False
    ):
        raise TUADProtocolError(
            "identifiability did not pass; Stop A forbids treatment collection"
        )
    expected_protocol = result.get("provenance", {}).get("protocol", {}).get("sha256")
    if expected_protocol != sha256_file(protocol_path):
        raise TUADProtocolError("identifiability result is bound to another protocol")
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TUADProtocolError("identifiability provenance is missing")
    for field in ("causal_probe", "oracle_labels", "label_reviews"):
        item = provenance.get(field)
        if not isinstance(item, Mapping):
            raise TUADProtocolError(f"identifiability {field} provenance drift")
        source_path = _project_path(item.get("path"), field)
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or item.get("sha256") != sha256_file(source_path)
            or item.get("bytes") != source_path.stat().st_size
        ):
            raise TUADProtocolError(f"identifiability {field} source drift")
    if _recompute_identifiability(protocol_path, provenance) != result:
        raise TUADProtocolError("identifiability PASS artifact was not reproducible")
    return result


def _load_snapshots(
    path: Path,
) -> tuple[list[CausalActionSnapshot], dict, dict]:
    value = _json(path, "causal action snapshot list")
    if set(value) != SNAPSHOT_INPUT_KEYS:
        raise TUADProtocolError(
            f"snapshot-list schema drift; missing={sorted(SNAPSHOT_INPUT_KEYS - set(value))}, "
            f"extra={sorted(set(value) - SNAPSHOT_INPUT_KEYS)}"
        )
    if (
        value["schema_version"] != SNAPSHOT_INPUT_SCHEMA
        or value["status"] != "CAUSAL_SNAPSHOTS_FROZEN"
    ):
        raise TUADProtocolError("causal action snapshots are not frozen")
    if not isinstance(value["snapshots"], list) or not value["snapshots"]:
        raise TUADProtocolError("snapshot list is empty")
    snapshots = [CausalActionSnapshot.from_mapping(item) for item in value["snapshots"]]
    continuation, _ = _verified_inventory(
        value["frozen_continuation_source"], "frozen continuation"
    )
    baseline, _ = _verified_inventory(
        value["native_baseline_source"], "native baseline"
    )
    return (
        snapshots,
        continuation,
        baseline,
    )


def _validate_causal_temporal_records(
    path: Path,
    protocol: Mapping[str, object],
    snapshots: Sequence[CausalActionSnapshot],
) -> None:
    """Bind strict causal histories exactly to the pre-outcome action cohort."""

    try:
        records, source_commitment = temporal_record_list_from_mapping(
            _json(path, "causal temporal record list")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TUADProtocolError("invalid strict causal temporal record list") from error
    source_population = protocol.get("source_population")
    if (
        not isinstance(source_population, Mapping)
        or source_commitment
        != source_population.get("canonical_identity_sha256")
    ):
        raise TUADProtocolError("causal temporal records use another source universe")
    snapshot_by_identity = {
        (
            value.dataset,
            value.scene_id,
            value.episode_id,
            value.decision_step,
        ): value
        for value in snapshots
    }
    record_by_identity = {
        (
            value.dataset,
            value.scene_id,
            value.episode_id,
            value.decision_step,
        ): value
        for value in records
    }
    if set(record_by_identity) != set(snapshot_by_identity):
        raise TUADProtocolError(
            "causal temporal records and action snapshots cover different decisions"
        )
    for identity, record in record_by_identity.items():
        snapshot = snapshot_by_identity[identity]
        final = record.steps[-1]
        if final.step != record.decision_step:
            raise TUADProtocolError(
                "causal temporal sequence does not end at its decision step"
            )
        if final.native_action_id != snapshot.native_action_id:
            raise TUADProtocolError("causal temporal native action drift")
        score_by_action = dict(zip(
            snapshot.global_action_ids, snapshot.policy_scores, strict=True,
        ))
        ranked = tuple(sorted(
            snapshot.frozen_candidate_action_ids,
            key=lambda action_id: (-score_by_action[action_id], action_id),
        ))
        runner_id = next(
            action_id for action_id in ranked
            if action_id != snapshot.native_action_id
        )
        expected_margin = abs(
            score_by_action[snapshot.native_action_id]
            - score_by_action[runner_id]
        )
        if final.candidate_action_ids != ranked:
            raise TUADProtocolError(
                "causal temporal candidate support drift: frozen rank mismatch"
            )
        if (
            float(final.policy_features[NATIVE_SCORE_INDEX])
            != score_by_action[snapshot.native_action_id]
            or float(final.policy_features[NATIVE_MARGIN_INDEX])
            != expected_margin
        ):
            raise TUADProtocolError("causal temporal native score/margin drift")


def build_collection_plan(
    protocol_path: Path,
    identifiability_path: Path,
    snapshot_path: Path,
    causal_temporal_record_path: Path,
) -> dict:
    protocol = verify_protocol(protocol_path, root=PROJECT_ROOT)
    identifiability = _verify_identifiability(
        identifiability_path, protocol_path
    )
    snapshots, continuation, baseline = _load_snapshots(snapshot_path)
    temporal_records = _inventory(causal_temporal_record_path)
    if (
        temporal_records
        != identifiability.get("provenance", {}).get("causal_probe")
    ):
        raise TUADProtocolError(
            "collection causal temporal records differ from Audit-B provenance"
        )
    _validate_causal_temporal_records(
        causal_temporal_record_path, protocol, snapshots
    )
    continuation_sha = continuation["sha256"]
    seal = seal_action_lattice(snapshots)
    arms = []
    for event in seal.events:
        for action_id in event.action_ids:
            arms.append({
                "lattice_id": event.lattice_id,
                "dataset": event.dataset,
                "scene_id": event.scene_id,
                "episode_id": event.episode_id,
                "decision_step": event.decision_step,
                "native_prefix_sha256": event.native_prefix_sha256,
                "action_id": action_id,
                "arm_type": "native" if action_id == event.native_action_id else "treatment",
                "prior_intervention": "abstain",
                "target_step_action_changes": 0 if action_id == event.native_action_id else 1,
                "maximum_second_interventions": 0,
                "continuation_controller_sha256": continuation_sha,
            })
    return {
        "schema_version": COLLECTION_PLAN_SCHEMA,
        "status": "SEALED_BEFORE_TREATMENT_OUTCOMES",
        "method_id": protocol["revision"],
        "lattice_revision": LATTICE_ID,
        "protocol": {"path": _relative(protocol_path), "sha256": sha256_file(protocol_path)},
        "identifiability_result": {
            "path": _relative(identifiability_path),
            "sha256": sha256_file(identifiability_path),
            "status": "MF3ZN_IDENTIFIABILITY_PASS",
        },
        "causal_snapshot_source": _inventory(snapshot_path),
        "causal_temporal_record_source": temporal_records,
        "frozen_continuation_source": continuation,
        "native_baseline_source": baseline,
        "outcome_fields_used_for_selection": [],
        "treatment_results_read": False,
        "seal": seal.as_manifest(),
        "arms": arms,
        "public_split_access": False,
    }


def _rebuild_plan_seal(plan: Mapping[str, object]):
    seal_payload = plan.get("seal")
    if not isinstance(seal_payload, Mapping) or not isinstance(seal_payload.get("events"), list):
        raise TUADProtocolError("collection plan has no sealed action lattice")
    snapshots = []
    for item in seal_payload["events"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("snapshot"), Mapping):
            raise TUADProtocolError("collection plan event schema drift")
        snapshot = dict(item["snapshot"])
        snapshot.pop("schema_version", None)
        snapshots.append(CausalActionSnapshot.from_mapping(snapshot))
    rebuilt = seal_action_lattice(snapshots)
    if rebuilt.as_manifest() != dict(seal_payload):
        raise TUADProtocolError("collection plan action-list commitment drift")
    expected_arms = []
    continuation = plan.get("frozen_continuation_source")
    continuation_sha = (
        continuation.get("sha256") if isinstance(continuation, Mapping) else None
    )
    for event in rebuilt.events:
        for action_id in event.action_ids:
            expected_arms.append({
                "lattice_id": event.lattice_id,
                "dataset": event.dataset,
                "scene_id": event.scene_id,
                "episode_id": event.episode_id,
                "decision_step": event.decision_step,
                "native_prefix_sha256": event.native_prefix_sha256,
                "action_id": action_id,
                "arm_type": "native" if action_id == event.native_action_id else "treatment",
                "prior_intervention": "abstain",
                "target_step_action_changes": 0 if action_id == event.native_action_id else 1,
                "maximum_second_interventions": 0,
                "continuation_controller_sha256": continuation_sha,
            })
    if plan.get("arms") != expected_arms:
        raise TUADProtocolError("collection plan arm inventory drift")
    return rebuilt


def _verify_plan_provenance(plan: Mapping[str, object]) -> None:
    """Rebuild the complete plan from its immutable sources and exact-match it."""

    if set(plan) != PLAN_KEYS:
        raise TUADProtocolError("collection plan top-level schema drift")
    if (
        plan.get("schema_version") != COLLECTION_PLAN_SCHEMA
        or plan.get("status") != "SEALED_BEFORE_TREATMENT_OUTCOMES"
    ):
        raise TUADProtocolError("collection plan was not sealed before outcomes")
    protocol = plan.get("protocol")
    identifiability = plan.get("identifiability_result")
    snapshot = plan.get("causal_snapshot_source")
    temporal_records = plan.get("causal_temporal_record_source")
    if not all(isinstance(value, Mapping) for value in (
        protocol, identifiability, snapshot, temporal_records
    )):
        raise TUADProtocolError("collection plan provenance schema drift")
    if set(protocol) != {"path", "sha256"}:
        raise TUADProtocolError("collection plan protocol provenance schema drift")
    if set(identifiability) != {"path", "sha256", "status"}:
        raise TUADProtocolError("collection plan identifiability provenance schema drift")
    protocol_path = _project_path(protocol.get("path"), "protocol")
    identifiability_path = _project_path(
        identifiability.get("path"), "identifiability"
    )
    snapshot_path = _project_path(snapshot.get("path"), "causal snapshot")
    temporal_record_path = _project_path(
        temporal_records.get("path"), "causal temporal record list"
    )
    _verified_inventory(snapshot, "causal snapshot source")
    _verified_inventory(temporal_records, "causal temporal record list")
    expected = build_collection_plan(
        protocol_path,
        identifiability_path,
        snapshot_path,
        temporal_record_path,
    )
    if dict(plan) != expected:
        raise TUADProtocolError(
            "collection plan differs from its hashed causal snapshot sources"
        )


def validate_collection_result(plan_path: Path, result_path: Path) -> dict:
    plan = _json(plan_path, "collection plan")
    _verify_plan_provenance(plan)
    seal = _rebuild_plan_seal(plan)
    result = _json(result_path, "collection result")
    if set(result) != RESULT_KEYS:
        raise TUADProtocolError("collection result top-level schema drift")
    if result["schema_version"] != COLLECTION_RESULT_SCHEMA:
        raise TUADProtocolError("collection result schema version drift")
    if result["action_list_commitment_sha256"] != seal.action_list_commitment_sha256:
        raise TUADProtocolError("collection result used another action list")
    if not isinstance(result["events"], list):
        raise TUADProtocolError("collection result events must be a list")
    by_lattice = {event.lattice_id: event for event in seal.events}
    observed: set[str] = set()
    audits = []
    outcomes = []
    continuation_sha = plan["frozen_continuation_source"]["sha256"]
    baseline_sha = plan["native_baseline_source"]["sha256"]
    for row in result["events"]:
        if not isinstance(row, dict) or set(row) != RESULT_EVENT_KEYS:
            raise TUADProtocolError("collection result event schema drift")
        lattice_id = row["lattice_id"]
        if not isinstance(lattice_id, str) or not lattice_id:
            raise TUADProtocolError("collection result lattice identity drift")
        event = by_lattice.get(lattice_id)
        if event is None or event.lattice_id in observed:
            raise TUADProtocolError("unknown or duplicate collection lattice")
        observed.add(event.lattice_id)
        _require_zero_count(
            row["native_prior_intervention_count"], "native prior intervention count"
        )
        _require_zero_count(
            row["native_second_intervention_count"], "native second intervention count"
        )
        if row["native_baseline_sha256"] != baseline_sha:
            raise TUADProtocolError("native baseline source drift")
        if row["native_continuation_controller_sha256"] != continuation_sha:
            raise TUADProtocolError("native continuation controller drift")
        native_metrics, native_source = _arm_outcome(
            row["native_outcome_source"], event, event.native_action_id
        )
        outcomes.append(_outcome_record(
            event,
            event.native_action_id,
            "native",
            native_metrics,
            native_metrics,
            native_source,
        ))
        if not isinstance(row["treatments"], list):
            raise TUADProtocolError("treatment arms must be a list")
        if any(not isinstance(item, dict) for item in row["treatments"]):
            raise TUADProtocolError("treatment result must be an object")
        if any(set(item) != TREATMENT_KEYS for item in row["treatments"]):
            raise TUADProtocolError("treatment result schema drift")
        treatment_ids = [item.get("action_id") for item in row["treatments"]]
        if any(not isinstance(value, str) or not value for value in treatment_ids):
            raise TUADProtocolError("treatment action identity drift")
        if len(treatment_ids) != len(set(treatment_ids)):
            raise TUADProtocolError("duplicate collection treatment action")
        treatment_by_action = {
            item["action_id"]: item for item in row["treatments"]
        }
        if set(treatment_by_action) != set(event.alternative_action_ids):
            raise TUADProtocolError("collection treatment-arm coverage drift")
        native_arm = LatticeArmIdentity(
            event.dataset, event.scene_id, event.episode_id, event.decision_step,
            event.native_prefix_sha256, event.native_action_id,
        )
        for action_id in event.alternative_action_ids:
            treatment = treatment_by_action[action_id]
            _require_zero_count(
                treatment["prior_intervention_count"],
                "treatment prior intervention count",
            )
            _require_zero_count(
                treatment["second_intervention_count"],
                "treatment second intervention count",
            )
            if treatment["continuation_controller_sha256"] != continuation_sha:
                raise TUADProtocolError("treatment continuation controller drift")
            treatment_arm = LatticeArmIdentity(
                event.dataset, event.scene_id, event.episode_id, event.decision_step,
                event.native_prefix_sha256, action_id,
            )
            audits.append(validate_exact_lattice_treatment(
                event,
                native_arm,
                treatment_arm,
                row["native_physical_trace"],
                treatment["physical_trace"],
                row["native_decision_trace"],
                treatment["decision_trace"],
            ))
            treatment_metrics, treatment_source = _arm_outcome(
                treatment["outcome_source"], event, action_id
            )
            outcomes.append(_outcome_record(
                event,
                action_id,
                "treatment",
                treatment_metrics,
                native_metrics,
                treatment_source,
            ))
    if observed != set(by_lattice):
        raise TUADProtocolError("collection result is missing a sealed lattice")
    return {
        "schema_version": EXACT_AUDIT_SCHEMA,
        "status": "MF3ZN_EXACT_LATTICE_AUDIT_PASS",
        "collection_plan_path": _relative(plan_path),
        "collection_plan_sha256": sha256_file(plan_path),
        "collection_result_path": _relative(result_path),
        "collection_result_sha256": sha256_file(result_path),
        "action_list_commitment_sha256": seal.action_list_commitment_sha256,
        "outcome_definition": {
            "metrics": list(OUTCOME_METRICS),
            "utility_weights": dict(UTILITY_WEIGHTS),
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
            "catastrophic_rule": "delta_utility <= catastrophic_threshold",
        },
        "lattices": len(seal.events),
        "treatments": len(audits),
        "audits": audits,
        "outcomes": outcomes,
        "public_authorization": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--protocol", type=Path, required=True)
    seal_parser.add_argument("--identifiability", type=Path, required=True)
    seal_parser.add_argument("--snapshots", type=Path, required=True)
    seal_parser.add_argument(
        "--causal-temporal-records", type=Path, required=True
    )
    seal_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--plan", type=Path, required=True)
    validate_parser.add_argument("--result", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _relative(args.output)
    if args.command == "seal":
        value = build_collection_plan(
            args.protocol,
            args.identifiability,
            args.snapshots,
            args.causal_temporal_records,
        )
    else:
        value = validate_collection_result(args.plan, args.result)
    _atomic_json(args.output, value)
    print(json.dumps(
        {"status": value["status"], "output": str(args.output)},
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
