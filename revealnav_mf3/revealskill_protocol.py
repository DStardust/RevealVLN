"""Sealing and verification for MF3ZP-REVEALSKILL v1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

from .evidence_uad import UAD_STABILITY_PREFIXES
from .qwen_evidence_annotation import (
    EVIDENCE_SYSTEM_PROMPT,
    INSTRUCTION_SYSTEM_PROMPT,
    QWEN_MODEL,
    prompt_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
REVISION = "mf3zp_revealskill_v1"
SCHEMA_VERSION = "revealnav-mf3zp-revealskill-protocol/1"
EXPECTED_SOURCE_COMMIT = "4ff282cb3b6ed45a9402903134a1aaf1ffb2a4b6"
OUTPUT = ROOT / "artifacts/training/mf3zp_revealskill_v1"
PROTOCOL_PATH = OUTPUT / "MF3ZP_REVEALSKILL_PROTOCOL.json"
PILOT_SELECTION = OUTPUT / "MF3ZP_REVEAL_PILOT_SELECTION.json"
PILOT_EVENTS = OUTPUT / "MF3ZP_REVEAL_EVENTS.jsonl"
PILOT_AUDIT = OUTPUT / "MF3ZP_REVEAL_PILOT_DATA_AUDIT.json"
PARENT_TUAD = ROOT / "artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json"
SOURCE_ANNOTATION_PROTOCOL = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2/MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
SOURCE_REQUESTS = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2/MF3ZP_ANNOTATION_REQUESTS.jsonl"

MEMORY_BUDGET = 8
ROLLOUT_HORIZON = 8
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260901

IMPLEMENTATION_FILES = (
    "METHOD_REVISION_3ZP_REVEALSKILL.md",
    "revealnav_mf3/evidence_constraints.py",
    "revealnav_mf3/evidence_chain.py",
    "revealnav_mf3/evidence_memory.py",
    "revealnav_mf3/evidence_uad.py",
    "revealnav_mf3/option_graph.py",
    "revealnav_mf3/option_evidence_binding.py",
    "revealnav_mf3/revealskill_schema.py",
    "revealnav_mf3/revealskill_features.py",
    "revealnav_mf3/revealskill_ree_loss.py",
    "revealnav_mf3/revealskill_q.py",
    "revealnav_mf3/revealskill_policy.py",
    "revealnav_mf3/reveal_event_data.py",
    "revealnav_mf3/qwen_evidence_annotation.py",
    "revealnav_mf3/revealskill_protocol.py",
    "scripts/build_mf3zp_reveal_pilot.py",
    "scripts/seal_mf3zp_revealskill_protocol.py",
    "scripts/annotate_mf3zp_qwen.py",
    "scripts/audit_mf3zp_labels.py",
    "scripts/run_mf3zp_oracle_headroom.py",
    "scripts/train_mf3zp_ree.py",
    "scripts/collect_mf3zp_skill_rollouts.py",
    "scripts/train_mf3zp_skill_policy.py",
    "scripts/evaluate_mf3zp_development.py",
)


class ProtocolError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or (resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents):
        raise ProtocolError(f"invalid project-local file: {path}")
    return {
        "path": str(resolved.relative_to(ROOT.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProtocolError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON object required: {path}")
    return value


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent_boundaries() -> tuple[dict[str, object], dict[str, object]]:
    if sha256_file(PARENT_TUAD) != "b502629d898879c65031a92b91496fd39d640e7c0f09097bd8bce8ebd9118772":
        raise ProtocolError("sealed MF3ZN parent SHA drift")
    parent = read_json(PARENT_TUAD)
    if parent.get("family_tombstone") != {
        "forbids": [
            "car_v2", "rcsp_v2", "single_snapshot_architecture_search",
            "decision_threshold_sweep", "catastrophe_threshold_sweep",
            "loss_reweighting_on_consumed_outcomes",
            "additional_weight_decay_or_training_length_search",
        ],
        "name": "FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED",
        "value": True,
    }:
        raise ProtocolError("single-decision family tombstone drift")
    source = read_json(SOURCE_ANNOTATION_PROTOCOL)
    public = source.get("authorization", {}).get("public_split_access")
    if public != {"test": False, "test_challenge": False, "val_seen": False, "val_unseen": False}:
        raise ProtocolError("source annotation protocol public boundary drift")
    if source.get("observation", {}).get("target_or_outcome_input") is not False:
        raise ProtocolError("source observation outcome boundary drift")
    return parent, source


def build_protocol() -> dict[str, object]:
    if current_commit() != EXPECTED_SOURCE_COMMIT:
        raise ProtocolError("source commit does not match reviewed base")
    parent, source = _validate_parent_boundaries()
    pilot = read_json(PILOT_SELECTION)
    audit = read_json(PILOT_AUDIT)
    if pilot.get("event_count") != 300 or pilot.get("domain_counts") != {"R2R": 150, "RxR": 150}:
        raise ProtocolError("pilot population is not the predeclared 300-event balance")
    if audit.get("status") != "MF3ZP_REVEAL_PILOT_DATA_PASS":
        raise ProtocolError("pilot audit did not pass")
    blacklist = list(source["consumed_confirmation_blacklist"])
    scenes = sorted({str(event["scene_id"]) for event in pilot["events"]})
    if set(scenes) & set(blacklist):
        raise ProtocolError("consumed confirmation scene entered pilot")
    implementation = {name: inventory(ROOT / name) for name in IMPLEMENTATION_FILES}
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": REVISION,
        "status": "SEALED_BEFORE_MF3ZP_RESULTS",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "family_tombstone": parent["family_tombstone"],
        "scientific_scope": "evidence-preserving high-level RevealSkill feasibility; not a single-decision gate",
        "source_files": {
            "mf3zn_parent": inventory(PARENT_TUAD),
            "causal_observation_protocol": inventory(SOURCE_ANNOTATION_PROTOCOL),
            "causal_annotation_requests": inventory(SOURCE_REQUESTS),
            "pilot_selection": inventory(PILOT_SELECTION),
            "pilot_events": inventory(PILOT_EVENTS),
            "pilot_data_audit": inventory(PILOT_AUDIT),
        },
        "implementation_inventory": implementation,
        "pilot": {
            "event_count": 300,
            "domain_counts": {"R2R": 150, "RxR": 150},
            "scene_ids": scenes,
            "raw_scene_count": len(scenes),
            "event_ids_sha256": pilot["event_ids_sha256"],
            "fold_count": 5,
            "fold_assignment": {str(event["event_id"]): int(event["scene_fold"]) for event in pilot["events"]},
            "selection_outcome_blind": True,
            "target_payload_read": False,
            "outcome_payload_read": False,
        },
        "consumed_confirmation_blacklist": blacklist,
        "method": {
            "constraint_kinds": ["ENTITY", "RELATION", "DIRECTION", "ORDINAL", "TEMPORAL_ORDER", "EXCLUSION", "GOAL"],
            "uad_stability_k": UAD_STABILITY_PREFIXES,
            "instruction_representation": "acyclic decisive evidence graph",
            "instruction_level_uad": False,
            "memory_budget_m": MEMORY_BUDGET,
            "bounded_skill_rollout_horizon_h": ROLLOUT_HORIZON,
            "temporal_gru_hidden": 64,
            "constraint_readout_width": 64,
            "loss_weight_search": False,
            "hyperparameter_search": False,
            "architecture_search": False,
            "deployment_threshold_search": False,
            "utility": {"nDTW": 0.50, "SDTW": 0.25, "SPL": 0.25},
        },
        "qwen_annotation": {
            "role": ["instruction_semantic_decomposition", "strictly_causal_visual_evidence_grounding"],
            "forbidden_roles": ["action_reward", "delta_utility", "correct_action", "policy_selection", "simulator_oracle", "intervention_outcome"],
            "api_model_identifier": QWEN_MODEL,
            "model_snapshot": QWEN_MODEL,
            "model_snapshot_authority": "explicit_user_authorized_API_identifier",
            "provider_response_model_must_equal_request": True,
            "rolling_alias_fallback": False,
            "formal_label_validity_blocked_until_human_review": True,
            "instruction_prompt_sha256": prompt_sha256(INSTRUCTION_SYSTEM_PROMPT),
            "evidence_prompt_sha256": prompt_sha256(EVIDENCE_SYSTEM_PROMPT),
            "temperature": 0.0,
            "structured_output": True,
        },
        "label_validity": {
            "reviewers": 3,
            "adjudicators": 1,
            "reviewers_blinded_to_outcomes": True,
            "uad_kappa_min": 0.65,
            "evidence_closure_kappa_min": 0.70,
            "qwen_preannotation_is_gold": False,
        },
        "scientific_gates": {
            "order": ["label_validity", "oracle_headroom", "ree_learnability", "full_skill_development"],
            "label_validity_fail": "MF3ZP_LABEL_VALIDITY_FAIL",
            "oracle_headroom": {
                "pcr_relative_reduction_min": 0.25,
                "per_domain_delta_utility": ">0",
                "per_domain_raw_scene_bootstrap_lower_95": ">0",
                "fail": "MF3ZP_ORACLE_HEADROOM_FAIL",
            },
            "ree_learnability": {
                "primary": ["per_constraint_uad_macro_f1", "reveal_nll", "expiry_nll"],
                "evidence_memory_over_temporal_only_per_domain": True,
                "raw_scene_bootstrap_lower_95": ">0",
                "fail": "MF3ZP_REE_LEARNABILITY_FAIL",
            },
            "full_skill": {
                "per_domain_delta_utility": ">0",
                "pcr_relative_reduction_min": 0.25,
                "no_clear_sr_spl_regression": True,
                "required_ablations": ["evidence_memory_only", "no_expiry_option_memory"],
                "fail": "MF3ZP_FULL_SKILL_DEVELOPMENT_FAIL",
            },
        },
        "interval_handling": {
            "reveal_training": "interval_censored_discrete_hazard_likelihood",
            "oracle_pcr": "conservative_bounds; definite premature if T_C<lower, possible premature if T_C<upper; PASS uses base-lower versus oracle-upper",
            "post_result_single_point_selection": False,
        },
        "bootstrap": {"cluster": "raw_mp3d_scene", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "authorization": {
            "qwen_preannotation": True,
            "human_label_validity": True,
            "oracle_headroom": False,
            "ree_training": False,
            "skill_rollout_collection": False,
            "skill_policy_training": False,
            "checkpoint_generation": False,
            "public_evaluation": False,
        },
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
        "stop_rule": "first failed scientific gate stops this one-shot revision; no post-result feature/model/threshold revision on pilot scenes",
    }


def atomic_write_protocol(value: Mapping[str, object]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists() or PROTOCOL_PATH.is_symlink():
        raise ProtocolError(f"refusing to overwrite sealed protocol: {PROTOCOL_PATH}")
    partial = PROTOCOL_PATH.with_name(PROTOCOL_PATH.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ProtocolError(f"stale protocol partial: {partial}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, PROTOCOL_PATH)


def seal_protocol() -> dict[str, object]:
    value = build_protocol()
    atomic_write_protocol(value)
    return value


def verify_protocol() -> dict[str, object]:
    value = read_json(PROTOCOL_PATH)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("revision") != REVISION or value.get("status") != "SEALED_BEFORE_MF3ZP_RESULTS":
        raise ProtocolError("MF3ZP protocol identity/status drift")
    if value.get("source_commit") != EXPECTED_SOURCE_COMMIT or current_commit() != EXPECTED_SOURCE_COMMIT:
        raise ProtocolError("source commit drift")
    if value.get("public_split_access") != {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}:
        raise ProtocolError("public split boundary drift")
    auth = value.get("authorization", {})
    if any(auth.get(key) is not False for key in ("oracle_headroom", "ree_training", "skill_rollout_collection", "skill_policy_training", "checkpoint_generation", "public_evaluation")):
        raise ProtocolError("downstream authorization was opened")
    _validate_parent_boundaries()
    for section in ("source_files", "implementation_inventory"):
        entries = value.get(section)
        if not isinstance(entries, Mapping):
            raise ProtocolError(f"missing protocol inventory: {section}")
        for expected in entries.values():
            path = ROOT / str(expected["path"])
            if inventory(path) != expected:
                raise ProtocolError(f"protocol inventory drift: {path}")
    if value.get("method", {}).get("uad_stability_k") != 3:
        raise ProtocolError("UAD stability K drift")
    return value


__all__ = [
    "BOOTSTRAP_REPLICATES", "BOOTSTRAP_SEED", "EXPECTED_SOURCE_COMMIT",
    "MEMORY_BUDGET", "OUTPUT", "PROTOCOL_PATH", "ProtocolError", "REVISION",
    "ROLLOUT_HORIZON", "build_protocol", "inventory", "seal_protocol",
    "sha256_file", "verify_protocol",
]
