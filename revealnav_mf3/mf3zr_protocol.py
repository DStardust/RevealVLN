"""Sealing and verification for MF3ZR option-bound support.

The protocol is intentionally a data/observation-support revision.  It does
not authorize an oracle rollout, learner training, checkpoint creation, or
public-split access.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
REVISION = "mf3zr_option_bound_support_v1"
OUTPUT = ROOT / "artifacts/training" / REVISION
PROTOCOL_PATH = OUTPUT / "MF3ZR_OPTION_BOUND_SUPPORT_PROTOCOL.json"
REVIEW_SOURCE_PATH = OUTPUT / "MF3ZR_OPTION_BINDING_REVIEW_SOURCE.json"
REVIEW_RECEIPT_PATH = OUTPUT / "MF3ZR_OPTION_BINDING_REVIEW_SOURCE_RECEIPT.json"
BINDINGS_PATH = OUTPUT / "MF3ZR_OPTION_BINDINGS.jsonl"
BINDING_AUDIT_PATH = OUTPUT / "MF3ZR_OPTION_BINDING_AUDIT.json"
RETURNABILITY_PATH = OUTPUT / "MF3ZR_RETURNABILITY_RECORDS.jsonl"
RETURNABILITY_AUDIT_PATH = OUTPUT / "MF3ZR_RETURNABILITY_AUDIT.json"
REVEAL_EXPIRY_PATH = OUTPUT / "MF3ZR_REVEAL_EXPIRY_SUPPORT.jsonl"
JOINT_AUDIT_PATH = OUTPUT / "MF3ZR_JOINT_SUPPORT_AUDIT.json"
RESULT_PATH = OUTPUT / "MF3ZR_OPTION_BOUND_SUPPORT_RESULT.json"

BASE_COMMIT = "5e4fbcbd14d6a28b5d8edc1d590b6ca8f9a1adfb"
MF3ZQ_DIR = ROOT / "artifacts/training/mf3zq_oracle_revealskill_headroom_v1"
SOURCE_POPULATION = MF3ZQ_DIR / "MF3ZQ_ORACLE_HEADROOM_POPULATION.jsonl"
SOURCE_POPULATION_AUDIT = MF3ZQ_DIR / "MF3ZQ_ORACLE_HEADROOM_POPULATION_AUDIT.json"
SOURCE_MF3ZQ_PROTOCOL = MF3ZQ_DIR / "MF3ZQ_ORACLE_HEADROOM_PROTOCOL.json"
SOURCE_MF3ZQ_RESULT = MF3ZQ_DIR / "MF3ZQ_ORACLE_HEADROOM_RESULT.json"
SOURCE_VISUAL_LABELS = ROOT / "artifacts/training/mf3zp_codex_visual_review_v1/MF3ZP_CODEX_VISUAL_REVIEW_LABELS.jsonl"
FORMAL_MF3ZP_PROTOCOL = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEALSKILL_PROTOCOL.json"
MF3ZP_FORMAL_SHA = "d0f09395b86804d3afc58f4ec946afc7dfaffd1637c7b8a66a776d58a17cc0c9"
SOURCE_POPULATION_SHA = "76095a16939b35a1f201b3ffa72d094dda00fae71ab1d428e3b6293ebf5724aa"
SOURCE_VISUAL_LABEL_SHA = "e664f8c637db0e8780e3ed47b8bea455a2fec7fd08d9277e1ad10981ef6bd9ba"

PUBLIC_CLOSED = {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}
K_STABILITY = 3
MEMORY_BUDGET = 8
RETURN_HORIZON = 8
JOINT_COVERAGE_MIN = 0.80
DOMAIN_COVERAGE_MIN = 0.75
MIN_SUPPORTED_EPISODES_PER_DOMAIN = 30

# These are the files whose bytes define this revision's implementation.  All
# are written before protocol sealing and are checked on every audit command.
IMPLEMENTATION_FILES = (
    "METHOD_REVISION_3ZR_OPTION_BOUND_SUPPORT.md",
    "revealnav_mf3/option_binding_schema.py",
    "revealnav_mf3/option_identity.py",
    "revealnav_mf3/evidence_option_graph.py",
    "revealnav_mf3/option_binding_audit.py",
    "revealnav_mf3/frozen_returnability.py",
    "revealnav_mf3/reveal_expiry_support.py",
    "revealnav_mf3/mf3zr_protocol.py",
    "scripts/build_mf3zr_binding_review.py",
    "scripts/materialize_mf3zr_option_bindings.py",
    "scripts/run_mf3zr_returnability_audit.py",
    "scripts/audit_mf3zr_joint_support.py",
    "scripts/seal_mf3zr_protocol.py",
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
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in resolved.parents:
        raise ProtocolError(f"invalid project-local inventory path: {path}")
    return {"path": str(resolved.relative_to(ROOT.resolve())), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def current_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _ensure_base_commit() -> None:
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=ROOT).returncode != 0:
        raise ProtocolError("MF3ZR reviewed base commit is not an ancestor of HEAD")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ProtocolError(f"JSON object required at {path}:{number}")
        rows.append(value)
    return rows


def _closed(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if value.get("public_split_access") != PUBLIC_CLOSED:
        raise ProtocolError(f"public split is open in {path}")
    return value


def _fixed_source_checks() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not SOURCE_POPULATION.is_file() or sha256_file(SOURCE_POPULATION) != SOURCE_POPULATION_SHA:
        raise ProtocolError("MF3ZQ source population hash drift")
    if not SOURCE_VISUAL_LABELS.is_file() or sha256_file(SOURCE_VISUAL_LABELS) != SOURCE_VISUAL_LABEL_SHA:
        raise ProtocolError("MF3ZP visual-label source hash drift")
    if not FORMAL_MF3ZP_PROTOCOL.is_file() or sha256_file(FORMAL_MF3ZP_PROTOCOL) != MF3ZP_FORMAL_SHA:
        raise ProtocolError("formal MF3ZP protocol byte identity drift")
    population = _read_jsonl(SOURCE_POPULATION)
    labels = _read_jsonl(SOURCE_VISUAL_LABELS)
    if len(population) != 80 or len(labels) != 80:
        raise ProtocolError("MF3ZR fixed population must contain 80 rows")
    if Counter(str(row.get("dataset")) for row in population) != Counter({"R2R": 40, "RxR": 40}):
        raise ProtocolError("MF3ZR source domain allocation drift")
    if len({(str(row.get("dataset")), str(row.get("episode_id"))) for row in population}) != 80:
        raise ProtocolError("MF3ZR source episodes are not unique")
    if len({str(row.get("event_id")) for row in population}) != 80:
        raise ProtocolError("MF3ZR source event IDs are not unique")
    if {str(row.get("event_id")) for row in population} != {str(row.get("event_id")) for row in labels}:
        raise ProtocolError("MF3ZR population/visual-label event IDs differ")
    if SOURCE_POPULATION_AUDIT.is_file():
        audit = _read_json(SOURCE_POPULATION_AUDIT)
        if audit.get("events") != 80 or audit.get("public_split_access") != PUBLIC_CLOSED:
            raise ProtocolError("historical MF3ZQ population audit drift")
    # The old result is hashed for provenance only; its numerical fields are
    # never parsed or used for any MF3ZR decision.
    if not SOURCE_MF3ZQ_PROTOCOL.is_file() or not SOURCE_MF3ZQ_RESULT.is_file():
        raise ProtocolError("historical MF3ZQ artifacts are missing")
    _closed(SOURCE_MF3ZQ_PROTOCOL)
    return population, labels


def build_protocol() -> dict[str, object]:
    _ensure_base_commit()
    population, labels = _fixed_source_checks()
    if not REVIEW_SOURCE_PATH.is_file() or not REVIEW_RECEIPT_PATH.is_file():
        raise ProtocolError("build MF3ZR binding review source before sealing")
    implementation = {name: inventory(ROOT / name) for name in IMPLEMENTATION_FILES}
    scenes = sorted({str(row["scene_id"]) for row in population})
    episodes = [{"dataset": str(row["dataset"]), "episode_id": str(row["episode_id"])} for row in sorted(population, key=lambda x: (str(x["dataset"]), str(x["episode_id"]))) ]
    return {
        "schema_version": "revealnav-mf3zr-option-bound-support-protocol/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_MF3ZR_SUPPORT_RESULTS",
        "source_commit": BASE_COMMIT,
        "seal_commit": current_commit(),
        "scientific_scope": "data_and_observation_support_only; no learner, policy, oracle rollout, or public evaluation",
        "historical_boundary": {
            "single_decision_gate_family_stopped": True,
            "mf3zq_status": "MF3ZQ_EXPLORATORY_ORACLE_HEADROOM_FAIL",
            "mf3zq_failure_type": "FAIL_AT_POPULATION_SUPPORT",
            "mf3zq_numeric_evidence": "NOT_OBSERVED",
            "prohibited_reruns": ["CAR", "RCSP", "DSR", "MF3ZN", "MF3ZO", "MF3ZP", "MF3ZQ"],
            "formal_mf3zp_protocol": inventory(FORMAL_MF3ZP_PROTOCOL),
            "mf3zq_protocol": inventory(SOURCE_MF3ZQ_PROTOCOL),
            "mf3zq_result": inventory(SOURCE_MF3ZQ_RESULT),
        },
        "source_files": {
            "mf3zq_population": inventory(SOURCE_POPULATION),
            "mf3zq_visual_labels": inventory(SOURCE_VISUAL_LABELS),
            "mf3zq_population_audit": inventory(SOURCE_POPULATION_AUDIT),
            "binding_review_source": inventory(REVIEW_SOURCE_PATH),
            "binding_review_receipt": inventory(REVIEW_RECEIPT_PATH),
        },
        "implementation_inventory": implementation,
        "population": {
            "events": 80,
            "unique_episodes": 80,
            "raw_mp3d_scenes": len(scenes),
            "scene_ids": scenes,
            "episode_ids": episodes,
            "domain_counts": {"R2R": 40, "RxR": 40},
            "fixed_source_no_replacement": True,
        },
        "binding": {
            "schema_version": "revealnav-mf3zr-option-evidence-binding/1",
            "states": ["SUPPORTS", "CONTRADICTS", "UNRESOLVED", "SHARED_CONTEXT", "NOT_APPLICABLE"],
            "context_and_discriminative_mutually_exclusive": True,
            "candidate_rank_is_not_semantic_truth": True,
            "option_id_rule": "sha256({event_id, first_seen_step, candidate_id})",
            "candidate_identity_source": "causal_opaque_alias_only",
            "missing_dec_binding_is_failure": True,
        },
        "support_definition": {
            "K_stability": K_STABILITY,
            "memory_budget": MEMORY_BUDGET,
            "return_horizon": RETURN_HORIZON,
            "joint_coverage_min": JOINT_COVERAGE_MIN,
            "domain_coverage_min": DOMAIN_COVERAGE_MIN,
            "minimum_supported_unique_episodes_per_domain": MIN_SUPPORTED_EPISODES_PER_DOMAIN,
            "no_replacement": True,
            "returnability_must_be_control_backed": True,
            "expiry_censoring_allowed": True,
            "reveal_censoring_allowed": True,
        },
        "forbidden_information": ["reward", "utility", "delta_utility", "success", "SR", "SPL", "nDTW", "SDTW", "catastrophe", "route_truth", "correct_action", "shortest_path", "future_frame", "public_split"],
        "execution": {
            "qwen_calls": 0,
            "qwen_reads": 0,
            "outcome_payload_read": False,
            "oracle_arms_run": [],
            "checkpoint_generated": False,
            "public_split_access": dict(PUBLIC_CLOSED),
        },
        "public_split_access": dict(PUBLIC_CLOSED),
        "pass_fail": {
            "status_on_pass": "SUPPORT_PASS",
            "status_on_fail": "SUPPORT_FAIL",
            "joint_coverage": ">=0.80",
            "each_domain_coverage": ">=0.75",
            "each_domain_supported_unique_episodes": ">=30",
            "oracle_numerical_evidence": "NOT_OBSERVED",
        },
    }


def seal_protocol() -> dict[str, object]:
    if PROTOCOL_PATH.exists() or PROTOCOL_PATH.is_symlink():
        raise ProtocolError("MF3ZR protocol already sealed; refusing overwrite")
    value = build_protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    partial = PROTOCOL_PATH.with_name(PROTOCOL_PATH.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ProtocolError("stale MF3ZR protocol partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, PROTOCOL_PATH)
    return value


def verify_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("MF3ZR protocol is missing")
    value = _read_json(path)
    if value.get("status") != "SEALED_BEFORE_MF3ZR_SUPPORT_RESULTS":
        raise ProtocolError("MF3ZR protocol status drift")
    if value.get("source_commit") != BASE_COMMIT:
        raise ProtocolError("MF3ZR source commit drift")
    if value.get("public_split_access") != PUBLIC_CLOSED or value.get("execution", {}).get("public_split_access") != PUBLIC_CLOSED:
        raise ProtocolError("MF3ZR public split opened")
    execution = value.get("execution", {})
    if execution.get("checkpoint_generated") is not False or execution.get("oracle_arms_run") != []:
        raise ProtocolError("MF3ZR downstream authorization opened")
    if value.get("population", {}).get("events") != 80 or value.get("population", {}).get("unique_episodes") != 80:
        raise ProtocolError("MF3ZR fixed population drift")
    # Verify source files and implementation bytes exactly as sealed.  The
    # protocol itself is intentionally not part of implementation_inventory.
    for section in ("source_files", "implementation_inventory"):
        records = value.get(section, {})
        if not isinstance(records, Mapping):
            raise ProtocolError(f"malformed {section}")
        for item in records.values():
            if not isinstance(item, Mapping):
                raise ProtocolError("malformed inventory item")
            current = inventory(ROOT / str(item["path"]))
            if current != dict(item):
                raise ProtocolError(f"MF3ZR inventory drift: {item['path']}")
    if sha256_file(SOURCE_POPULATION) != SOURCE_POPULATION_SHA or sha256_file(SOURCE_VISUAL_LABELS) != SOURCE_VISUAL_LABEL_SHA:
        raise ProtocolError("fixed historical source bytes changed")
    if sha256_file(FORMAL_MF3ZP_PROTOCOL) != MF3ZP_FORMAL_SHA:
        raise ProtocolError("formal MF3ZP protocol changed")
    return value


__all__ = [
    "ROOT", "REVISION", "OUTPUT", "PROTOCOL_PATH", "REVIEW_SOURCE_PATH", "REVIEW_RECEIPT_PATH",
    "BINDINGS_PATH", "BINDING_AUDIT_PATH", "RETURNABILITY_PATH",
    "RETURNABILITY_AUDIT_PATH", "REVEAL_EXPIRY_PATH", "JOINT_AUDIT_PATH",
    "RESULT_PATH", "BASE_COMMIT", "SOURCE_POPULATION", "SOURCE_VISUAL_LABELS",
    "FORMAL_MF3ZP_PROTOCOL", "MF3ZP_FORMAL_SHA", "SOURCE_POPULATION_SHA",
    "SOURCE_VISUAL_LABEL_SHA", "PUBLIC_CLOSED", "K_STABILITY", "MEMORY_BUDGET",
    "RETURN_HORIZON", "JOINT_COVERAGE_MIN", "DOMAIN_COVERAGE_MIN",
    "MIN_SUPPORTED_EPISODES_PER_DOMAIN", "IMPLEMENTATION_FILES", "ProtocolError",
    "sha256_file", "inventory", "build_protocol", "seal_protocol", "verify_protocol",
]
