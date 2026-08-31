"""Pre-result protocol sealing and fail-closed verification for MF3ZO."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

from .mf3zo_pilot import (
    EXPECTED_CANONICAL_IDENTITY,
    PILOT_EVENTS,
    inventory_file,
    sha256_file,
)
from .mf3zo_probes import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    HUBER_DELTA,
    OUTER_FOLDS,
    RIDGE_L2,
    SCENE_FOLD_SALT,
    assign_scene_folds,
)
from .mf3zo_temporal_schema import UAD_STABILITY_PREFIXES


REVISION = "mf3zo_temporal_oracle_gap_v1"
STATUS = "SEALED_BEFORE_TEMPORAL_ORACLE_GAP_RESULTS"
SCHEMA_VERSION = "revealnav-mf3zo-temporal-oracle-gap-protocol/1"
EXPECTED_PUBLIC_ACCESS = {
    "val_seen": False,
    "val_unseen": False,
    "test": False,
    "test_challenge": False,
}
METHOD_PATH = "METHOD_REVISION_3ZO_TEMPORAL_ORACLE_GAP.md"
OUTPUT_RELATIVE = "artifacts/training/mf3zo_temporal_oracle_gap_v1"
PROTOCOL_NAME = "MF3ZO_TEMPORAL_ORACLE_GAP_PROTOCOL.json"
EXPECTED_PARENT_MF3ZN_SHA256 = (
    "b502629d898879c65031a92b91496fd39d640e7c0f09097bd8bce8ebd9118772"
)
PILOT_FILES = (
    "MF3ZO_PILOT_SELECTION.json",
    "MF3ZO_CAUSAL_TEMPORAL_RECORDS.jsonl",
    "MF3ZO_TEMPORAL_ORACLE_LABELS.jsonl",
    "MF3ZO_PILOT_DATA_AUDIT.json",
)
IMPLEMENTATION_FILES = (
    METHOD_PATH,
    "revealnav_mf3/mf3zo_temporal_schema.py",
    "revealnav_mf3/mf3zo_pilot.py",
    "revealnav_mf3/mf3zo_probes.py",
    "revealnav_mf3/mf3zo_protocol.py",
    "scripts/run_mf3zo_temporal_oracle_gap.py",
)


class ProtocolError(RuntimeError):
    pass


def _strict_json(path: Path) -> dict:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ProtocolError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid JSON source: {path}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"protocol source is not an object: {path}")
    return value


def _strict_jsonl(path: Path) -> list[dict]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"invalid JSONL {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ProtocolError(f"JSONL row is not an object: {path}:{line_number}")
        values.append(value)
    return values


def validate_protocol(value: Mapping[str, object]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("MF3ZO protocol schema drift")
    if value.get("revision") != REVISION or value.get("status") != STATUS:
        raise ProtocolError("MF3ZO protocol identity/status drift")
    if value.get("public_split_access") != EXPECTED_PUBLIC_ACCESS:
        raise ProtocolError("MF3ZO public split access must remain all false")
    authorization = value.get("authorization")
    if not isinstance(authorization, dict) or authorization != {
        "checkpoint_generation": False,
        "formal_teal_collection": False,
        "full_tuad_training": False,
        "public_evaluation": False,
    }:
        raise ProtocolError("MF3ZO authorization boundary drift")
    if value.get("family_tombstone") != {
        "name": "FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED",
        "value": True,
    }:
        raise ProtocolError("single-decision gate tombstone is not active")
    fixed = value.get("fixed_probe_configuration")
    if not isinstance(fixed, dict) or any(
        fixed.get(name) not in ([], False, None)
        for name in (
            "architecture_grid",
            "feature_subset_search",
            "hyperparameter_search",
            "regularization_grid",
            "seed_selection",
            "threshold_grid",
        )
    ):
        raise ProtocolError("MF3ZO protocol permits forbidden model search")
    if fixed.get("ridge_l2") != RIDGE_L2 or fixed.get("outer_folds") != OUTER_FOLDS:
        raise ProtocolError("MF3ZO fixed probe parameters drift")
    if fixed.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
        raise ProtocolError("MF3ZO bootstrap count drift")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise ProtocolError("invalid source commit")
    return result


def build_protocol(root: Path) -> dict:
    root = root.resolve()
    output = root / OUTPUT_RELATIVE
    selection = _strict_json(output / PILOT_FILES[0])
    audit = _strict_json(output / PILOT_FILES[3])
    oracle_rows = _strict_jsonl(output / PILOT_FILES[2])
    events = selection.get("events")
    if not isinstance(events, list) or len(events) != PILOT_EVENTS:
        raise ProtocolError("MF3ZO pilot selection must contain 150 events")
    event_ids = [str(value["event_id"]) for value in events]
    if len(set(event_ids)) != PILOT_EVENTS:
        raise ProtocolError("MF3ZO pilot event identities are not unique")
    scenes = [str(value["scene_id"]) for value in events]
    datasets = [str(value["dataset"]) for value in events]
    if Counter(datasets) != {"R2R": 75, "RxR": 75}:
        raise ProtocolError("MF3ZO pilot domain allocation drift")
    folds, mapping = assign_scene_folds(scenes)
    parent = _strict_json(
        root / "artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json"
    )
    parent_path = root / "artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json"
    if (
        sha256_file(parent_path) != EXPECTED_PARENT_MF3ZN_SHA256
        or parent.get("status") != "SEALED_BEFORE_IDENTIFIABILITY_RESULTS"
        or parent.get("authorization", {}).get("new_treatment_collection") is not False
        or parent.get("authorization", {}).get("tuad_training") is not False
        or parent.get("authorization", {}).get("public_split_access")
        != EXPECTED_PUBLIC_ACCESS
    ):
        raise ProtocolError("formal MF3ZN parent protocol boundary drift")
    car = _strict_json(root / "artifacts/training/mf3zm_car_v1/MF3ZM_CAR_PROTOCOL.json")
    blacklist = sorted(str(value) for value in car["known_consumed_scene_ids"])
    intersection = sorted(set(scenes) & set(blacklist))
    if intersection:
        raise ProtocolError("consumed confirmation scene entered MF3ZO pilot")
    if any(value.get("status") != "UNAVAILABLE" for value in oracle_rows):
        raise ProtocolError("unreviewed oracle row was marked available")
    source_files = {
        path: inventory_file(root / path, root) for path in IMPLEMENTATION_FILES
    }
    source_files.update({
        "parent_mf3zn_protocol": inventory_file(
            root / "artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json",
            root,
        ),
        "parent_car_protocol": inventory_file(
            root / "artifacts/training/mf3zm_car_v1/MF3ZM_CAR_PROTOCOL.json",
            root,
        ),
        "canonical_loader": inventory_file(
            root / "scripts/train_mf3zm_car.py", root,
        ),
    })
    pilot_files = {
        name: inventory_file(output / name, root) for name in PILOT_FILES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "revision": REVISION,
        "purpose": "one-shot train-development temporal observability/oracle-gap pilot",
        "source_commit": _git_commit(root),
        "family_tombstone": {
            "name": "FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED",
            "value": True,
        },
        "parent_mf3zn_protocol_sha256": EXPECTED_PARENT_MF3ZN_SHA256,
        "canonical_source": {
            "events": 1540,
            "raw_mp3d_scenes": 39,
            "domain_counts": {"R2R": 543, "RxR": 997},
            "canonical_identity_sha256": EXPECTED_CANONICAL_IDENTITY,
        },
        "pilot": {
            "events": PILOT_EVENTS,
            "domain_counts": dict(Counter(datasets)),
            "raw_mp3d_scenes": len(set(scenes)),
            "scene_ids": sorted(set(scenes)),
            "event_ids": event_ids,
            "scene_fold_salt": SCENE_FOLD_SALT,
            "scene_fold_mapping": mapping,
            "event_folds": folds.tolist(),
            "selection_outcome_blind": True,
        },
        "consumed_confirmation_blacklist": blacklist,
        "consumed_confirmation_intersection": intersection,
        "causal_information_boundary": {
            "latest_step": "j <= decision_step",
            "oracle_storage_separate": True,
            "missing_fields_imputed": False,
            "forbidden_inference": [
                "future frame/candidate set", "target", "delta_utility",
                "counterfactual outcome", "route oracle truth", "navmesh",
                "simulator pose", "oracle label",
            ],
        },
        "uad_semantics": {
            "stability_prefixes": UAD_STABILITY_PREFIXES,
            "U": "target branch absent",
            "A": "target present but separation or evidence closure incomplete",
            "D": "presence, separation, and closure stable for K causal prefixes",
            "learned_redefinition": False,
        },
        "fixed_probe_configuration": {
            "outer_folds": OUTER_FOLDS,
            "fold_unit": "raw_mp3d_scene",
            "shared_scene_cross_dataset_same_fold": True,
            "fold_fit_only_standardization": True,
            "ridge_l2": RIDGE_L2,
            "huber_delta": HUBER_DELTA,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "raw_mp3d_scene",
            "architecture_grid": [],
            "feature_subset_search": False,
            "hyperparameter_search": False,
            "regularization_grid": [],
            "seed_selection": False,
            "threshold_grid": [],
            "probe_a_oracle_interval_representation": [
                "reveal_lower_minus_decision_step",
                "reveal_upper_minus_decision_step",
                "expiry_step_minus_decision_step",
            ],
            "probe_b_primary_metrics": [
                "uad_macro_f1_improvement",
                "reveal_nll_improvement",
                "expiry_nll_improvement",
            ],
            "probe_c_decision_rule": "oof_prediction > 0",
            "probe_c_temporal_encoder": {
                "type": "single_predeclared_causal_GRU",
                "hidden_dim": 64,
                "trained_only_on": [
                    "target_in_set", "candidate_separation",
                    "evidence_closure", "reveal_hazard", "expiry_hazard",
                ],
                "delta_utility_gradient_allowed": False,
                "frozen_before_action_value_probe": True,
            },
        },
        "probe_order": ["A_oracle_relevance", "B_temporal_observability", "C_learned_state_relevance"],
        "pass_fail": {
            "stop_at_first_failure": True,
            "probe_a": "both domains delta_huber observed and lower_95 > 0",
            "probe_b": "all three primary improvements observed and lower_95 > 0 in both domains",
            "probe_c": (
                "both domains positive predictive/utility/LOSO evidence, no "
                "negative or zero-coverage fold, catastrophic rate <= strongest "
                "matched deterministic baseline"
            ),
            "unavailable_required_oracle_supervision": "TEMPORAL_ORACLE_RELEVANCE_FAIL",
            "no_post_result_revision": True,
        },
        "authorization": {
            "checkpoint_generation": False,
            "formal_teal_collection": False,
            "full_tuad_training": False,
            "public_evaluation": False,
        },
        "public_split_access": dict(EXPECTED_PUBLIC_ACCESS),
        "source_files": source_files,
        "pilot_files": pilot_files,
        "pilot_data_status_at_seal": audit["status"],
        "oracle_complete_labels_at_seal": int(
            audit["oracle_coverage"]["complete_verified_labels"]
        ),
        "old_confirmation_reused": False,
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise ProtocolError(f"refusing to overwrite protocol: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ProtocolError(f"stale protocol partial: {partial}")
    partial.write_text(json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n", encoding="utf-8")
    os.replace(partial, path)


def seal_protocol(root: Path) -> tuple[Path, dict]:
    path = root.resolve() / OUTPUT_RELATIVE / PROTOCOL_NAME
    value = build_protocol(root)
    validate_protocol(value)
    _atomic_json(path, value)
    return path, value


def verify_protocol(path: Path, root: Path) -> dict:
    root = root.resolve()
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("MF3ZO protocol is unavailable")
    value = _strict_json(path)
    validate_protocol(value)
    for collection_name in ("source_files", "pilot_files"):
        collection = value.get(collection_name)
        if not isinstance(collection, dict):
            raise ProtocolError(f"missing protocol inventory: {collection_name}")
        for item in collection.values():
            source = root / item["path"]
            if (
                not source.is_file()
                or source.is_symlink()
                or source.stat().st_size != int(item["bytes"])
                or sha256_file(source) != str(item["sha256"])
            ):
                raise ProtocolError(f"MF3ZO sealed source drift: {item['path']}")
    if set(value["pilot"]["scene_ids"]) & set(value["consumed_confirmation_blacklist"]):
        raise ProtocolError("consumed confirmation scene entered sealed MF3ZO pilot")
    return value


__all__ = [
    "EXPECTED_PUBLIC_ACCESS",
    "OUTPUT_RELATIVE",
    "PROTOCOL_NAME",
    "ProtocolError",
    "REVISION",
    "SCHEMA_VERSION",
    "STATUS",
    "build_protocol",
    "seal_protocol",
    "validate_protocol",
    "verify_protocol",
]
