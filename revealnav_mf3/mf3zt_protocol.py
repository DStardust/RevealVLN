"""Sealed protocol for the MF3ZT evidence-memory decision probe.

MF3ZT is fail-closed at its first prerequisite: a legal, pre-existing,
candidate-aligned target must exist in both R2R and RxR.  This module freezes
the complete conditional probe design without authorizing downstream work when
that prerequisite fails.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
REVISION = "mf3zt_evidence_memory_decision_probe_v1"
OUTPUT = ROOT / "artifacts" / "training" / REVISION
PROTOCOL_PATH = OUTPUT / "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_PROTOCOL.json"
TARGET_AUDIT_PATH = OUTPUT / "MF3ZT_DECISION_TARGET_SUPPORT_AUDIT.json"
RESULT_PATH = OUTPUT / "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_RESULT.json"

# MF3ZT was requested against this clean main revision.  It must remain an
# ancestor, rather than equal HEAD, after this versioned implementation lands.
BASE_COMMIT = "e24c4f6a62b6e86cd143e911d0dd9ae103daa209"

PUBLIC_CLOSED = {
    "val_seen": False,
    "val_unseen": False,
    "test": False,
    "test_challenge": False,
}
DOMAINS = ("R2R", "RxR")
EVIDENCE_ONTOLOGY = (
    "LANDMARK_SEEN",
    "LANDMARK_PASSED",
    "RELATION_SATISFIED",
    "ORDINAL_COUNT",
    "DIRECTIONAL_CONTEXT",
)
CONFIDENCE_CLASSES = ("OBSERVED", "AMBIGUOUS", "ABSENT")
ARMS = (
    "ETP_CURRENT",
    "ETP_PLUS_EVIDENCE_MEMORY",
    "ETP_PLUS_SHUFFLED_MEMORY",
)
K_MEM = 8
FOLDS = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_901
MIN_MEMORY_REQUIRED_DECISIONS = 50
MIN_MEMORY_REQUIRED_SCENES = 10

RXR_TARGET_PROTOCOL = ROOT / "artifacts/phase1/mf3b_uad_online/dataset_v1/MF3B_ONLINE_DATA_PROTOCOL.json"
RXR_TARGET_MANIFEST = ROOT / "artifacts/phase1/mf3b_uad_online/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json"
MF3ZP_OBSERVATION_STATUS = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2/MF3ZP_OBSERVATION_COLLECTION_STATUS.json"
MF3ZQ_POPULATION = ROOT / "artifacts/training/mf3zq_oracle_revealskill_headroom_v1/MF3ZQ_ORACLE_HEADROOM_POPULATION.jsonl"
MF3ZR_BINDING_AUDIT = ROOT / "artifacts/training/mf3zr_option_bound_support_v1/MF3ZR_OPTION_BINDING_AUDIT.json"
MF3ZR_RESULT = ROOT / "artifacts/training/mf3zr_option_bound_support_v1/MF3ZR_OPTION_BOUND_SUPPORT_RESULT.json"
R2R_TRAIN_GT = ROOT / "third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_gt.json.gz"
ETP_TRAINER = ROOT / "third_party/ETP-R1/vlnce_baselines/ss_trainer_ETP_R1.py"
MF3ZK_R2R_OUTCOME_MANIFEST = ROOT / "artifacts/training/mf3zk_joint_v1/r2r_collection/MF3ZK_R2R_DIRECT_SWITCH_MANIFEST.json"

R2R_ETP_CHECKPOINT = ROOT / "third_party/ETP-R1/data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
RXR_ETP_CHECKPOINT = ROOT / "third_party/ETP-R1/data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth"
ETP_BACKBONE = ROOT / "third_party/ETP-R1/pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt"

EXPECTED_SOURCE_SHA256 = {
    "rxr_target_protocol": "c857b0863062074987d497bb1b5b4a3cfbf768d1396ff15308e52ec6583e7346",
    "rxr_target_manifest": "36884bae31718bb859f5856103654f5b3f25979fdfd0319c1ca344f00328e034",
    "mf3zp_observation_status": "c8a4e6825da0873a1f8421c8cdfae499acf1b4cfdfeb99e96873108ac5b7a3ec",
    "mf3zq_population": "76095a16939b35a1f201b3ffa72d094dda00fae71ab1d428e3b6293ebf5724aa",
    "mf3zr_binding_audit": "09c6238db20630eb2bb1a88488b9d1f86c396540419d1354d7f4c57538ba3763",
    "mf3zr_result": "c513d6a9ed527276cc9ff7df074d18ca6e837dc484cdb8c9750c082752328ce6",
    "r2r_train_gt": "b63fa10e5c57d1b80b241c49bdeddc3d481bbe54b41ca0d59f9584c10da7bd5d",
    "etp_trainer": "9500d55628480981d2a4d9348c7e6e8da848723c553087ae29d93c7e3f5cdae8",
    "mf3zk_r2r_outcome_manifest": "1a3d1a45bd7c8cbaa61fc2d1d4b2ecfe11af0a032c8774ad7672369151a4e8b0",
    "r2r_etp_checkpoint": "8f90cebba7eefb9648054aa74e8c8664f23e643073ef033b99edfbe85c54f61c",
    "rxr_etp_checkpoint": "3796c9c94ff8674b8cfe99f2b4aab0f4b391f0d4c9c1e167e4736b3848f27821",
    "etp_backbone": "203fe62cc22c63261a5c5b6a3638bc52fd3b08a7f09dd31d8539bf2beab6c3cf",
}

SOURCE_PATHS = {
    "rxr_target_protocol": RXR_TARGET_PROTOCOL,
    "rxr_target_manifest": RXR_TARGET_MANIFEST,
    "mf3zp_observation_status": MF3ZP_OBSERVATION_STATUS,
    "mf3zq_population": MF3ZQ_POPULATION,
    "mf3zr_binding_audit": MF3ZR_BINDING_AUDIT,
    "mf3zr_result": MF3ZR_RESULT,
    "r2r_train_gt": R2R_TRAIN_GT,
    "etp_trainer": ETP_TRAINER,
    "mf3zk_r2r_outcome_manifest": MF3ZK_R2R_OUTCOME_MANIFEST,
    "r2r_etp_checkpoint": R2R_ETP_CHECKPOINT,
    "rxr_etp_checkpoint": RXR_ETP_CHECKPOINT,
    "etp_backbone": ETP_BACKBONE,
}

IMPLEMENTATION_FILES = (
    "METHOD_REVISION_3ZT_EVIDENCE_MEMORY_DECISION_PROBE.md",
    "revealnav_mf3/evidence_memory_probe.py",
    "revealnav_mf3/mf3zt_protocol.py",
    "scripts/seal_mf3zt_protocol.py",
    "scripts/build_mf3zt_decision_population.py",
    "scripts/audit_mf3zt_result.py",
)


class ProtocolError(RuntimeError):
    """Raised when a sealed MF3ZT boundary is missing or drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path, *, expected_sha256: str | None = None) -> dict[str, object]:
    resolved = path.resolve()
    root = ROOT.resolve()
    if not path.is_file() or path.is_symlink() or root not in resolved.parents:
        raise ProtocolError(f"invalid project-local inventory path: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProtocolError(f"source hash drift: {resolved.relative_to(root)}")
    return {
        "path": str(resolved.relative_to(root)),
        "bytes": int(path.stat().st_size),
        "sha256": digest,
    }


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ensure_base_commit() -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProtocolError("MF3ZT reviewed base commit is not an ancestor of HEAD")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def _source_inventory() -> dict[str, dict[str, object]]:
    return {
        name: inventory(path, expected_sha256=EXPECTED_SOURCE_SHA256[name])
        for name, path in SOURCE_PATHS.items()
    }


def build_protocol() -> dict[str, object]:
    _ensure_base_commit()
    source_inventory = _source_inventory()
    implementation_inventory = {
        name: inventory(ROOT / name) for name in IMPLEMENTATION_FILES
    }

    rxr_protocol = _read_object(RXR_TARGET_PROTOCOL)
    if rxr_protocol.get("dataset") != "RxR train guide en-US/en-IN only":
        raise ProtocolError("RxR target protocol scope drift")
    if rxr_protocol.get("public_unseen_authorized") is not False:
        raise ProtocolError("RxR target protocol opens public unseen")

    return {
        "schema_version": "revealnav-mf3zt-evidence-memory-decision-probe-protocol/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_MF3ZT_RESULTS",
        "source_commit": BASE_COMMIT,
        "seal_commit": current_commit(),
        "scientific_question": "Does explicit instruction-conditioned semantic evidence memory improve held-scene frozen-ETP candidate decisions?",
        "scope": {
            "decision_probe_only": True,
            "train_development_only": True,
            "full_navigation": False,
            "SR_SPL_optimization": False,
            "UAD_reveal_expiry_oracle_returnability": False,
            "policy_gradient_or_RL": False,
            "ETP_fine_tuning": False,
        },
        "source_inventory": source_inventory,
        "implementation_inventory": implementation_inventory,
        "target_support_gate": {
            "must_pass_before_population": True,
            "required_domains": list(DOMAINS),
            "accepted_priority": [
                "exact_train_native_action_or_candidate_supervision",
                "exact_same_episode_same_prefix_branch_supervision",
                "existing_frozen_causal_decision_target",
            ],
            "requires_preexisting_materialization": True,
            "requires_exact_candidate_set_alignment": True,
            "forbidden_target_derivations": [
                "frozen_native_action_self_label",
                "public_split_target",
                "cross_episode_pairing",
                "nearest_candidate_mapping",
                "route_truth_or_shortest_path_reconstruction",
                "route_level_reward_or_utility",
                "SR_SPL_nDTW_SDTW",
                "CAR_post_hoc_rescue_label",
            ],
            "status_on_fail": "MF3ZT_DECISION_TARGET_SUPPORT_FAIL",
            "stop_before": [
                "decision_population_materialization",
                "memory_required_classification",
                "evidence_memory_materialization",
                "reranker_implementation",
                "training",
                "cross_validation",
                "bootstrap",
                "full_navigation",
            ],
            "audited_preexisting_source_classes": {
                "exact_train_native_candidate_supervision": [
                    "MF3B RxR exact current-ghost teacher arrays",
                    "R2R train_gt low-level trajectory schema",
                    "ETP-R1 runtime dynamic teacher implementation",
                ],
                "exact_same_episode_same_prefix_branch_supervision": [
                    "MF3ZK R2R exact-one-switch route-return manifest"
                ],
                "existing_frozen_causal_decision_target": [
                    "MF3ZQ/MF3ZP frozen native causal observations",
                    "MF3ZR option-binding audit",
                ],
            },
            "negative_existence_claim_scope": "sealed audited source inventory, not a claim about data that could be newly generated",
        },
        "population": {
            "status": "NOT_MATERIALIZED_PENDING_TARGET_SUPPORT_GATE",
            "sha256": None,
            "decision_target_source_sha256": None,
            "evidence_memory_sha256": None,
            "R2R_decisions": None,
            "RxR_decisions": None,
            "raw_scene_count": None,
            "scene_disjoint_folds": FOLDS,
            "shared_R2R_RxR_scene_same_fold": True,
            "memory_required_min_per_domain": MIN_MEMORY_REQUIRED_DECISIONS,
            "memory_required_scene_min_per_domain": MIN_MEMORY_REQUIRED_SCENES,
        },
        "evidence": {
            "ontology": list(EVIDENCE_ONTOLOGY),
            "confidence_classes": list(CONFIDENCE_CLASSES),
            "K_MEM": K_MEM,
            "retrieval_trainable": False,
            "mean_pooling": True,
            "future_evidence_forbidden": True,
            "materialized_before_training": True,
            "record_schema": [
                "evidence_id",
                "event_id",
                "source_step",
                "source_node_id",
                "instruction_atom_id",
                "evidence_type",
                "semantic_value",
                "confidence_class",
                "source_observation_sha256",
            ],
            "retrieval_order": [
                "active_instruction_atom_order",
                "source_step_descending",
                "evidence_id",
            ],
        },
        "memory_required_definition": {
            "MEMORY_REQUIRED": "at least one instruction-relevant evidence item appeared in causal history, is absent or insufficient in the current observation, and is semantically required for the current candidate decision",
            "MEMORY_NOT_REQUIRED": "the current observation is sufficient or the decision does not require historical semantic state",
            "allowed_inputs": [
                "instruction_semantics",
                "causal_visual_history",
                "current_visual_observation",
                "current_candidate_geometry_or_appearance",
                "instruction_constraint_structure",
            ],
            "forbidden_inputs": [
                "candidate_target",
                "success",
                "reward",
                "utility",
                "future_frame",
            ],
            "classified_before_target_evaluation": True,
        },
        "frozen_ETP": {
            "ETP_frozen": True,
            "candidate_generator_frozen": True,
            "visual_backbone_frozen": True,
            "topology_encoder_frozen": True,
            "R2R_checkpoint": source_inventory["r2r_etp_checkpoint"],
            "RxR_checkpoint": source_inventory["rxr_etp_checkpoint"],
            "shared_pretrained_backbone": source_inventory["etp_backbone"],
            "baseline_extraction_started": False,
        },
        "conditional_model": {
            "arms": list(ARMS),
            "reranker": "fixed_two_layer_candidate_residual_MLP",
            "memory_pooling": "mean_then_linear_projection",
            "candidate_interaction": "concat(candidate,memory,candidate_elementwise_memory)",
            "loss": "candidate_set_cross_entropy_or_preexisting_multi_positive_equivalent",
            "architecture_sweep": False,
            "threshold_search": False,
            "multi_seed_rescue": False,
            "shuffled_memory": {
                "permutation_scope": "different event within training fold",
                "preserve_memory_count_and_feature_distribution": True,
                "held_target_as_donor": False,
            },
        },
        "evaluation": {
            "folds": FOLDS,
            "split_unit": "raw_MP3D_scene",
            "metrics": ["Acc@1", "MRR", "MeanRank", "pairwise_accuracy"],
            "subgroups": ["ALL", "MEMORY_REQUIRED", "MEMORY_NOT_REQUIRED"],
            "bootstrap": {
                "cluster": "raw_MP3D_scene",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
            "standardization_fit_on_train_fold_only": True,
        },
        "pass_fail": {
            "both_domains_required": True,
            "memory_required_B_minus_A_Acc_positive": True,
            "memory_required_B_minus_A_Acc_lower95_positive": True,
            "memory_required_B_minus_A_MRR_positive": True,
            "memory_required_B_minus_C_Acc_positive": True,
            "memory_required_B_minus_C_Acc_lower95_positive": True,
            "memory_not_required_B_minus_A_Acc_min": -0.01,
            "all_B_minus_A_Acc_min": 0.0,
            "status_on_pass": "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_PASS",
            "status_on_fail": "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_FAIL",
        },
        "execution": {
            "target_support_audit_started": False,
            "population_built": False,
            "evidence_memory_built": False,
            "training_started": False,
            "OOF_evaluation_run": False,
            "bootstrap_run": False,
            "full_navigation_run": False,
            "checkpoint_generated": False,
            "checkpoint_for_deployment": False,
            "public_split_access": dict(PUBLIC_CLOSED),
        },
        "public_split_access": dict(PUBLIC_CLOSED),
        "correctness_checks": [
            "no_future_observation",
            "no_future_candidate",
            "no_public_split",
            "no_utility_or_outcome_based_population_selection",
            "memory_required_label_does_not_read_candidate_target",
            "ETP_weights_frozen",
            "candidate_generator_frozen",
            "evidence_generated_before_training_result",
            "raw_scene_split",
            "shared_scenes_same_fold",
            "no_held_fold_normalization_leakage",
            "retrieval_K_exactly_8",
            "evidence_source_causal",
            "shuffled_memory_train_fold_safe",
            "held_fold_memory_not_sourced_from_another_held_target",
            "result_immutable",
            "no_full_navigation",
            "no_deployment_checkpoint",
            "historical_failed_revisions_unchanged",
            "full_regression_pass",
        ],
    }


def seal_protocol() -> dict[str, object]:
    if PROTOCOL_PATH.exists() or PROTOCOL_PATH.is_symlink():
        raise ProtocolError("MF3ZT protocol already sealed; refusing overwrite")
    if TARGET_AUDIT_PATH.exists() or RESULT_PATH.exists():
        raise ProtocolError("MF3ZT result material exists before protocol seal")
    value = build_protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    partial = PROTOCOL_PATH.with_name(PROTOCOL_PATH.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ProtocolError("stale MF3ZT protocol partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, PROTOCOL_PATH)
    return value


def verify_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    _ensure_base_commit()
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("MF3ZT protocol is missing")
    value = _read_object(path)
    if value.get("revision") != REVISION:
        raise ProtocolError("MF3ZT revision drift")
    if value.get("status") != "SEALED_BEFORE_MF3ZT_RESULTS":
        raise ProtocolError("MF3ZT protocol status drift")
    if value.get("source_commit") != BASE_COMMIT:
        raise ProtocolError("MF3ZT source commit drift")
    if value.get("public_split_access") != PUBLIC_CLOSED:
        raise ProtocolError("MF3ZT public split opened")
    execution = value.get("execution", {})
    if not isinstance(execution, Mapping) or execution.get("public_split_access") != PUBLIC_CLOSED:
        raise ProtocolError("MF3ZT execution public split opened")
    for key in (
        "population_built",
        "evidence_memory_built",
        "training_started",
        "OOF_evaluation_run",
        "bootstrap_run",
        "full_navigation_run",
        "checkpoint_generated",
        "checkpoint_for_deployment",
    ):
        if execution.get(key) is not False:
            raise ProtocolError(f"MF3ZT protocol prematurely authorizes {key}")
    if value.get("evidence", {}).get("ontology") != list(EVIDENCE_ONTOLOGY):
        raise ProtocolError("MF3ZT evidence ontology drift")
    if value.get("evidence", {}).get("K_MEM") != K_MEM:
        raise ProtocolError("MF3ZT retrieval budget drift")
    if value.get("conditional_model", {}).get("arms") != list(ARMS):
        raise ProtocolError("MF3ZT arm set drift")

    for section in ("source_inventory", "implementation_inventory"):
        records = value.get(section)
        if not isinstance(records, Mapping):
            raise ProtocolError(f"malformed MF3ZT {section}")
        for name, item in records.items():
            if not isinstance(item, Mapping):
                raise ProtocolError(f"malformed MF3ZT inventory item: {name}")
            current = inventory(ROOT / str(item["path"]))
            if current != dict(item):
                raise ProtocolError(f"MF3ZT inventory drift: {item['path']}")
    return value


__all__ = [
    "ROOT",
    "REVISION",
    "OUTPUT",
    "PROTOCOL_PATH",
    "TARGET_AUDIT_PATH",
    "RESULT_PATH",
    "BASE_COMMIT",
    "PUBLIC_CLOSED",
    "DOMAINS",
    "EVIDENCE_ONTOLOGY",
    "CONFIDENCE_CLASSES",
    "ARMS",
    "K_MEM",
    "FOLDS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "MIN_MEMORY_REQUIRED_DECISIONS",
    "MIN_MEMORY_REQUIRED_SCENES",
    "RXR_TARGET_PROTOCOL",
    "RXR_TARGET_MANIFEST",
    "MF3ZP_OBSERVATION_STATUS",
    "MF3ZQ_POPULATION",
    "MF3ZR_BINDING_AUDIT",
    "MF3ZR_RESULT",
    "R2R_TRAIN_GT",
    "ETP_TRAINER",
    "MF3ZK_R2R_OUTCOME_MANIFEST",
    "R2R_ETP_CHECKPOINT",
    "RXR_ETP_CHECKPOINT",
    "ETP_BACKBONE",
    "EXPECTED_SOURCE_SHA256",
    "SOURCE_PATHS",
    "IMPLEMENTATION_FILES",
    "ProtocolError",
    "sha256_file",
    "inventory",
    "build_protocol",
    "seal_protocol",
    "verify_protocol",
]
