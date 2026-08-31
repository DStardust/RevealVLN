"""Machine-readable pre-registration for MF3ZN-TUAD v1.

This module has no training, collection, result-loading, confirmation, or
public-evaluation entry point.  It defines the immutable scientific contract
that must be sealed before the MF3ZN identifiability audits are inspected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


TUAD_PROTOCOL_SCHEMA = "revealnav-mf3zn-tuad-protocol/1"
TUAD_REVISION = "mf3zn_tuad_v1"
TEAL_REVISION = "mf3zn_teal_v1"
FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED = True

# Stable integration aliases.  Training, audit, and collection scripts must
# import these names rather than copy protocol literals.
METHOD_ID = TUAD_REVISION
LATTICE_ID = TEAL_REVISION

GRU_HIDDEN_SIZE = 64
ACTION_VALUE_HIDDEN_SIZE = 64
RAW_SCENE_OOF_FOLDS = 5
OUTER_FOLDS = RAW_SCENE_OOF_FOLDS
FIXED_SEEDS = (20260831, 20260832, 20260833)
FIXED_REPORTING_SEEDS = FIXED_SEEDS
FIXED_WEIGHT_DECAY = 0.0
STAGE_1_EPOCHS = 200
STAGE_2_EPOCHS = 200
ADAM_LEARNING_RATE = 0.001
HUBER_DELTA = 1.0
ENSEMBLE_REDUCTION = "elementwise_median"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260830
OUTCOME_METRICS = ("success", "spl", "ndtw", "sdtw")
UTILITY_WEIGHTS = {"ndtw": 0.50, "sdtw": 0.25, "spl": 0.25}
CATASTROPHIC_THRESHOLD = -0.10
LABEL_VALIDITY_PILOT_ROWS = 300
UAD_KAPPA_MINIMUM = 0.65
EVIDENCE_CLOSURE_KAPPA_MINIMUM = 0.70
IDENTIFIABILITY_EXPECTED_ROWS = 1540
IDENTIFIABILITY_EXPECTED_SCENES = 39
IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS = {"R2R": 543, "RxR": 997}
IDENTIFIABILITY_CANONICAL_IDENTITY_SHA256 = (
    "7047fe8e3514d6037926f77a2883e9f0cdf094d5b077aa82febba64260b07bae"
)

ALLOWED_COMMANDS = ("seal", "verify")
IDENTIFIABILITY_AUDITS = (
    "oracle_relevance",
    "causal_observability",
    "label_validity",
)
CONTROLS = (
    "TUAD-full",
    "current-only",
    "temporal-no-UAD-supervision",
    "oracle-UAD",
    "runner-only-support",
    "frozen-native",
    "matched-high-proposal-score",
    "matched-low-native-margin",
    "matched-random",
)
SOURCE_PATHS = {
    "car_source_protocol": (
        "artifacts/training/mf3zm_car_v1/MF3ZM_CAR_PROTOCOL.json"
    ),
    "method_revision": "METHOD_REVISION_3ZN_TUAD.md",
    "temporal_schema": "revealnav_mf3/temporal_uad_schema.py",
    "temporal_labels": "revealnav_mf3/temporal_uad_labels.py",
    "temporal_features": "revealnav_mf3/temporal_uad_features.py",
    "temporal_model": "revealnav_mf3/temporal_uad_model.py",
    "temporal_action_value": "revealnav_mf3/temporal_action_value.py",
    "temporal_exact_lattice": "revealnav_mf3/temporal_exact_lattice.py",
    "tuad_identifiability": "revealnav_mf3/tuad_identifiability.py",
    "tuad_selection": "revealnav_mf3/tuad_selection.py",
    "protocol_implementation": "revealnav_mf3/tuad_protocol.py",
    "protocol_sealer": "scripts/seal_mf3zn_tuad_protocol.py",
    "identifiability_audit_entrypoint": (
        "scripts/audit_mf3zn_uad_identifiability.py"
    ),
    "lattice_collection_entrypoint": (
        "scripts/collect_mf3zn_temporal_lattice.py"
    ),
    "tuad_training_entrypoint": "scripts/train_mf3zn_tuad.py",
}

FORBIDDEN_INFERENCE_FIELDS = (
    "target",
    "delta_utility",
    "treatment_result",
    "future_*",
    "navmesh",
    "pose",
    "oracle_*",
)


class TUADProtocolError(RuntimeError):
    """The sealed MF3ZN scientific contract is missing or has drifted."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a protocol deterministically and reject non-JSON numbers."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _inventory(project_root: Path, relative_path: str) -> dict[str, Any]:
    root = project_root.resolve()
    path = root / relative_path
    if not path.is_file() or path.is_symlink():
        raise TUADProtocolError(f"invalid MF3ZN protocol source: {relative_path}")
    resolved = path.resolve()
    if root not in resolved.parents:
        raise TUADProtocolError(f"MF3ZN protocol source escaped root: {relative_path}")
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def source_inventory(project_root: Path) -> dict[str, dict[str, Any]]:
    """Inventory every source whose exact bytes define the sealed protocol."""

    return {
        name: _inventory(project_root, relative_path)
        for name, relative_path in sorted(SOURCE_PATHS.items())
    }


def _verify_parent_population(project_root: Path) -> None:
    """Bind MF3ZN audits to the already sealed MF3ZM event universe."""

    relative_path = SOURCE_PATHS["car_source_protocol"]
    path = project_root.resolve() / relative_path
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TUADProtocolError("invalid sealed CAR source protocol") from exc
    if (
        value.get("rows") != IDENTIFIABILITY_EXPECTED_ROWS
        or value.get("scenes") != IDENTIFIABILITY_EXPECTED_SCENES
        or value.get("domain_counts") != IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS
        or value.get("canonical_identity_sha256")
        != IDENTIFIABILITY_CANONICAL_IDENTITY_SHA256
        or value.get("consumed_confirmation_reused") is not False
        or any(value.get("public_split_access", {}).values())
    ):
        raise TUADProtocolError("sealed CAR source universe drift")


def build_protocol(project_root: Path) -> dict[str, Any]:
    """Build the deterministic, result-independent MF3ZN protocol."""

    _verify_parent_population(project_root)
    return {
        "schema_version": TUAD_PROTOCOL_SCHEMA,
        "status": "SEALED_BEFORE_IDENTIFIABILITY_RESULTS",
        "revision": TUAD_REVISION,
        "dataset_revision": TEAL_REVISION,
        "source_files": source_inventory(project_root),
        "family_tombstone": {
            "name": "FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED",
            "value": FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED,
            "forbids": [
                "car_v2",
                "rcsp_v2",
                "single_snapshot_architecture_search",
                "decision_threshold_sweep",
                "catastrophe_threshold_sweep",
                "loss_reweighting_on_consumed_outcomes",
                "additional_weight_decay_or_training_length_search",
            ],
        },
        "hypothesis": (
            "irreversible intervention value is identifiable from strictly "
            "causal evidence and candidate-availability evolution, including "
            "UAD and reveal/expiry dynamics"
        ),
        "source_population": {
            "rows": IDENTIFIABILITY_EXPECTED_ROWS,
            "raw_mp3d_scenes": IDENTIFIABILITY_EXPECTED_SCENES,
            "domain_counts": dict(IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS),
            "canonical_identity_sha256": (
                IDENTIFIABILITY_CANONICAL_IDENTITY_SHA256
            ),
            "event_id_rule": (
                "sha256(canonical-json(dataset,scene_id,episode_id,decision_step))"
            ),
            "runtime_parent_source_reverification_required": True,
            "cohort_substitution_allowed": False,
        },
        "causal_information_boundary": {
            "latest_allowed_step_relation": "step <= decision_step",
            "future_observations_allowed": False,
            "oracle_and_inference_storage_separated": True,
            "forbidden_inference_fields": list(FORBIDDEN_INFERENCE_FIELDS),
            "required_invariance_tests": [
                "future_mutation_invariance",
                "treatment_outcome_isolation",
                "oracle_isolation",
                "prefix_hash_identity",
                "one_action_invariant",
                "candidate_executability",
                "scene_fold_integrity",
                "episode_treatment_integrity",
                "uad_monotonic_semantics",
                "no_outcome_adaptive_collection",
            ],
        },
        "action_support": {
            "native_included": True,
            "maximum_non_native_actions": 2,
            "alternative_rule": (
                "highest frozen-policy-ranked executable non-native actions"
            ),
            "native_margin_definition": (
                "abs(native_policy_score - highest_ranked_executable_"
                "non_native_policy_score)"
            ),
            "sealed_before_treatment_outcomes": True,
            "outcome_adaptive_support": False,
            "causal_temporal_record_source": {
                "project_local_regular_file": True,
                "content_inventory_fields": ["path", "bytes", "sha256"],
                "sealed_in_collection_plan_before_outcomes": True,
                "training_must_rebuild_tensors_from_strict_records": True,
                "seal_time_validation": [
                    "strict_record_list_parser",
                    "source_commitment_equals_protocol_population",
                    "inventory_equals_identifiability_causal_probe_provenance",
                    "decision_identity_set_equals_action_snapshots",
                    "final_step_equals_decision_step",
                    "native_and_executable_support_equal_action_snapshot",
                ],
            },
            "treatment_contract": {
                "same_native_prefix": True,
                "exactly_one_target_step_action_change": True,
                "frozen_continuation": True,
                "second_intervention_count": 0,
                "cross_episode_pairing": False,
            },
            "outcome_contract": {
                "metrics": list(OUTCOME_METRICS),
                "utility_weights": dict(UTILITY_WEIGHTS),
                "catastrophic_rule": (
                    f"native_relative_delta_utility <= {CATASTROPHIC_THRESHOLD}"
                ),
                "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
                "per_arm_hashed_outcome_source_required": True,
                "validator_recomputes_outcomes": True,
                "controller_may_read_task_metrics": False,
            },
        },
        "identifiability_gate": {
            "required_before_collection": True,
            "all_required": list(IDENTIFIABILITY_AUDITS),
            "protocol_architecture_and_stop_rules_presealed": True,
            "domain_compensation_allowed": False,
            "audits": {
                "oracle_relevance": {
                    "population": "existing_1540_exact_native_runner_events",
                    "probes": ["fixed_current_control", "fixed_oracle_augmented"],
                    "model_search": False,
                    "loss": "Huber",
                    "requirements_per_domain": [
                        "oof_delta_huber > 0",
                        "scene_cluster_bootstrap_95pct_lower_bound > 0",
                    ],
                    "failure_status": "TEMPORAL_ORACLE_RELEVANCE_FAIL",
                },
                "causal_observability": {
                    "probes": ["fixed_snapshot", "fixed_causal_temporal_summary"],
                    "future_inputs": False,
                    "absolute_f1_threshold": None,
                    "requirements_per_domain": [
                        "uad_macro_f1_temporal_minus_snapshot > 0",
                        "reveal_nll_snapshot_minus_temporal > 0",
                        "expiry_nll_snapshot_minus_temporal > 0",
                        "all_scene_bootstrap_95pct_lower_bounds > 0",
                    ],
                    "failure_status": "TEMPORAL_CAUSAL_OBSERVABILITY_FAIL",
                },
                "label_validity": {
                    "scene_balanced_pilot_rows": LABEL_VALIDITY_PILOT_ROWS,
                    "uad_kappa_minimum": UAD_KAPPA_MINIMUM,
                    "evidence_closure_kappa_minimum": (
                        EVIDENCE_CLOSURE_KAPPA_MINIMUM
                    ),
                    "failure_status": "TEMPORAL_LABEL_VALIDITY_FAIL",
                },
            },
            "collection_authorized_at_seal": False,
            "collection_recomputes_audits_from_hashed_provenance": True,
            "copied_pass_json_is_sufficient": False,
        },
        "model": {
            "joint_end_to_end_training": False,
            "stage_1": {
                "name": "TemporalRevealExpiryEncoder",
                "encoder": "causal_gru",
                "hidden_size": GRU_HIDDEN_SIZE,
                "epochs": STAGE_1_EPOCHS,
                "optimizer": "Adam",
                "learning_rate": ADAM_LEARNING_RATE,
                "projections": "fixed",
                "utility_labels_allowed": False,
                "heads": [
                    "target_in_set",
                    "candidate_separation",
                    "evidence_closure",
                    "reveal_discrete_hazard",
                    "expiry_discrete_hazard",
                ],
                "loss": "unweighted_mean_of_five_masked_binary_cross_entropies",
                "loss_term_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
                "uad_head": None,
                "uad_derivation": "deterministic_frozen_semantics",
                "frozen_before_stage_2": True,
            },
            "stage_2": {
                "name": "NativeAnchoredActionValue",
                "loss": "Huber",
                "huber_delta": HUBER_DELTA,
                "epochs": STAGE_2_EPOCHS,
                "optimizer": "Adam",
                "learning_rate": ADAM_LEARNING_RATE,
                "hidden_size": ACTION_VALUE_HIDDEN_SIZE,
                "activation": "GELU",
                "native_value": 0.0,
                "native_bypasses_network": True,
                "deployment": "argmax_over_native_inclusive_sealed_support",
                "risk_head": False,
                "quantile_head": False,
            },
        },
        "development": {
            "architecture_grid": [],
            "architecture_selection": False,
            "weight_decay": FIXED_WEIGHT_DECAY,
            "weight_decay_grid": [],
            "weight_decay_selection": False,
            "decision_threshold": None,
            "threshold_grid": [],
            "threshold_selection": False,
            "fixed_reporting_seeds": list(FIXED_REPORTING_SEEDS),
            "seed_use": "declared_ensemble_and_reporting_only",
            "seed_selection": False,
            "ensemble_reduction": ENSEMBLE_REDUCTION,
            "individual_seed_results_reported": True,
            "outer_folds": RAW_SCENE_OOF_FOLDS,
            "fold_unit": "raw_mp3d_scene",
            "inner_model_selection": False,
            "complete_oof_required": True,
            "bootstrap": {
                "unit": "raw_mp3d_scene",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "confidence": 0.95,
            },
        },
        "controls": {
            "arms": list(CONTROLS),
            "oracle_uad_authorizes": False,
            "fixed_random_is_sanity_only": True,
            "matched_budget_unit": "fold_and_domain",
            "strongest_simple_baseline_rule": (
                "greater per-domain OOF total utility between "
                "matched-high-proposal-score and matched-low-native-margin; "
                "exact ties choose matched-high-proposal-score"
            ),
        },
        "scientific_gates": {
            "correctness": [
                "all_causality_tests_pass",
                "exact_lattice_audit_pass",
                "no_scene_leakage",
                "no_episode_arm_leakage",
                "no_public_access",
                "complete_five_fold_oof",
            ],
            "per_domain_utility": [
                "total_utility > 0",
                "every_fold_domain_total_utility >= 0",
                "minimum_leave_one_selected_scene_out_total > 0",
            ],
            "per_domain_catastrophic_risk": (
                "rate <= strongest_matched_simple_baseline_rate"
            ),
            "temporal_contribution_per_domain": [
                "tuad_full_minus_current_only_utility > 0",
                "scene_cluster_bootstrap_95pct_lower_bound > 0",
            ],
            "uad_contribution_per_domain": (
                "tuad_full_utility > temporal_no_uad_supervision_utility"
            ),
            "simple_baseline_per_domain": (
                "tuad_full_utility > strongest_matched_simple_baseline_utility"
            ),
            "hide_domain_results": False,
        },
        "stop_rules": {
            "stop_a": {
                "trigger": "any_identifiability_audit_fails",
                "action": (
                    "stop temporal-UAD intervention research before new "
                    "treatment collection"
                ),
            },
            "stop_b": {
                "trigger": "identifiability_passes_but_complete_development_fails",
                "action": (
                    "stop learned irreversible-intervention policy development "
                    "on this consumed development universe"
                ),
                "forbids": [
                    "tuad_v2",
                    "different_temporal_architecture",
                    "different_history_window",
                    "different_uad_definition",
                    "new_threshold_or_loss",
                    "new_risk_head",
                    "additional_weight_decay",
                    "outcome_adaptive_tuning_on_39_scenes",
                ],
            },
        },
        "authorization": {
            "identifiability_audit": True,
            "new_treatment_collection": False,
            "tuad_training": False,
            "confirmation": False,
            "public_unseen": False,
            "public_split_access": {
                "val_seen": False,
                "val_unseen": False,
                "test": False,
                "test_challenge": False,
            },
        },
        "entrypoints": {
            "allowed_scope": "protocol_sealer_only",
            "allowed": list(ALLOWED_COMMANDS),
            "default": "seal",
            "source_sealed_scientific": [
                "identifiability_audit",
                "gated_lattice_seal_and_validation",
                "gated_fixed_oof_training",
            ],
            "confirmation": None,
            "public_evaluation": None,
        },
    }


def protocol_payload(root: Path) -> dict[str, Any]:
    """Stable integration name for the deterministic protocol builder."""

    return build_protocol(root)


def validate_protocol(value: Mapping[str, Any], project_root: Path) -> None:
    """Fail closed unless ``value`` exactly matches the current sealed sources."""

    if not isinstance(value, Mapping):
        raise TUADProtocolError("MF3ZN protocol must be a mapping")
    if value.get("revision") != METHOD_ID:
        raise TUADProtocolError("MF3ZN method identifier drift")
    if value.get("dataset_revision") != LATTICE_ID:
        raise TUADProtocolError("MF3ZN lattice identifier drift")
    if value.get("status") != "SEALED_BEFORE_IDENTIFIABILITY_RESULTS":
        raise TUADProtocolError("MF3ZN protocol status drift")
    expected = build_protocol(project_root)
    if dict(value) != expected:
        raise TUADProtocolError("MF3ZN protocol or source inventory drift")
    if value["family_tombstone"]["value"] is not True:
        raise TUADProtocolError("single-decision gate family tombstone drift")
    if tuple(value["identifiability_gate"]["all_required"]) != (
        IDENTIFIABILITY_AUDITS
    ):
        raise TUADProtocolError("MF3ZN identifiability gate drift")
    if value["authorization"]["new_treatment_collection"] is not False:
        raise TUADProtocolError("MF3ZN collection was authorized at seal")
    if value["authorization"]["public_unseen"] is not False:
        raise TUADProtocolError("MF3ZN public evaluation was authorized at seal")
    if any(value["authorization"]["public_split_access"].values()):
        raise TUADProtocolError("MF3ZN public split access drift")
    development = value["development"]
    if (
        tuple(development["fixed_reporting_seeds"]) != FIXED_SEEDS
        or development["outer_folds"] != OUTER_FOLDS
        or development["ensemble_reduction"] != ENSEMBLE_REDUCTION
        or development["seed_selection"] is not False
        or development["architecture_selection"] is not False
        or development["weight_decay_selection"] is not False
        or development["threshold_selection"] is not False
    ):
        raise TUADProtocolError("MF3ZN fixed development configuration drift")


def verify_protocol(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Read and verify a canonical sealed protocol and all source hashes."""

    project_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[1]
    )
    protocol_path = Path(path)
    if not protocol_path.is_file() or protocol_path.is_symlink():
        raise TUADProtocolError(f"MF3ZN protocol is unavailable: {protocol_path}")
    try:
        value = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TUADProtocolError("invalid MF3ZN protocol JSON") from exc
    validate_protocol(value, project_root)
    if protocol_path.read_bytes() != canonical_json_bytes(value):
        raise TUADProtocolError("MF3ZN protocol JSON is not canonical")
    return value


__all__ = [
    "ACTION_VALUE_HIDDEN_SIZE",
    "ALLOWED_COMMANDS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CATASTROPHIC_THRESHOLD",
    "CONTROLS",
    "ADAM_LEARNING_RATE",
    "ENSEMBLE_REDUCTION",
    "EVIDENCE_CLOSURE_KAPPA_MINIMUM",
    "FIXED_SEEDS",
    "FIXED_REPORTING_SEEDS",
    "FIXED_WEIGHT_DECAY",
    "FORBIDDEN_INFERENCE_FIELDS",
    "FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED",
    "GRU_HIDDEN_SIZE",
    "IDENTIFIABILITY_AUDITS",
    "HUBER_DELTA",
    "IDENTIFIABILITY_CANONICAL_IDENTITY_SHA256",
    "IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS",
    "IDENTIFIABILITY_EXPECTED_ROWS",
    "IDENTIFIABILITY_EXPECTED_SCENES",
    "LATTICE_ID",
    "LABEL_VALIDITY_PILOT_ROWS",
    "RAW_SCENE_OOF_FOLDS",
    "METHOD_ID",
    "OUTER_FOLDS",
    "OUTCOME_METRICS",
    "STAGE_1_EPOCHS",
    "STAGE_2_EPOCHS",
    "TEAL_REVISION",
    "TUAD_PROTOCOL_SCHEMA",
    "TUAD_REVISION",
    "TUADProtocolError",
    "UAD_KAPPA_MINIMUM",
    "UTILITY_WEIGHTS",
    "build_protocol",
    "canonical_json_bytes",
    "sha256_file",
    "source_inventory",
    "protocol_payload",
    "validate_protocol",
    "verify_protocol",
]
