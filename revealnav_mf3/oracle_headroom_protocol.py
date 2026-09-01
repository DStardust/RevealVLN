"""Protocol sealing and verification for MF3ZQ Oracle RevealSkill headroom.

This is an exploratory, one-shot protocol.  It is intentionally separate from
the sealed MF3ZP formal protocol and never changes that file or its
authorization flags.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
REVISION = "mf3zq_oracle_revealskill_headroom_v1"
OUTPUT = ROOT / "artifacts/training" / REVISION
PROTOCOL_PATH = OUTPUT / "MF3ZQ_ORACLE_HEADROOM_PROTOCOL.json"
POPULATION_PATH = OUTPUT / "MF3ZQ_ORACLE_HEADROOM_POPULATION.jsonl"
AUDIT_PATH = OUTPUT / "MF3ZQ_ORACLE_HEADROOM_POPULATION_AUDIT.json"
RESULT_PATH = OUTPUT / "MF3ZQ_ORACLE_HEADROOM_RESULT.json"
BASE_COMMIT = "c356a226a1e5bffa05ddc86a0f3f821b490a2dc1"
VISUAL_LABELS = ROOT / "artifacts/training/mf3zp_codex_visual_review_v1/MF3ZP_CODEX_VISUAL_REVIEW_LABELS.jsonl"
VISUAL_MANIFEST = ROOT / "artifacts/training/mf3zp_codex_visual_review_v1/MF3ZP_CODEX_VISUAL_REVIEW_LABEL_MANIFEST.json"
VISUAL_PROTOCOL = ROOT / "artifacts/training/mf3zp_codex_visual_review_v1/MF3ZP_CODEX_VISUAL_REVIEW_PROTOCOL.json"
FORMAL_MF3ZP = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEALSKILL_PROTOCOL.json"
EVENTS = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEAL_EVENTS.jsonl"
SOURCE_OBSERVATIONS = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2/observations"

PUBLIC_CLOSED = {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}
MEMORY_BUDGET = 8
RETURN_HORIZON = 8
UAD_K = 3
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260901

IMPLEMENTATION_FILES = (
    "METHOD_REVISION_3ZQ_ORACLE_REVEALSKILL_HEADROOM.md",
    "revealnav_mf3/oracle_revealskill_schema.py",
    "revealnav_mf3/oracle_option_memory.py",
    "revealnav_mf3/oracle_reveal_expiry.py",
    "revealnav_mf3/oracle_revealskill_policy.py",
    "revealnav_mf3/oracle_headroom_metrics.py",
    "revealnav_mf3/oracle_headroom_protocol.py",
    "scripts/build_mf3zq_oracle_headroom_population.py",
    "scripts/seal_mf3zq_oracle_headroom_protocol.py",
    "scripts/run_mf3zq_oracle_headroom.py",
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
    return {"path": str(resolved.relative_to(ROOT.resolve())), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def current_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _ensure_base_commit() -> None:
    # Sealing is tied to the reviewed parent.  A descendant is allowed once
    # the new revision is committed, but an unrelated history is not.
    result = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=ROOT)
    if result.returncode != 0:
        raise ProtocolError("reviewed base commit is not an ancestor of HEAD")


def _closed_protocol(path: Path) -> dict[str, object]:
    value = _read_json(path)
    public = value.get("public_split_access")
    if public != PUBLIC_CLOSED:
        raise ProtocolError(f"public split is not closed in {path}")
    return value


def build_protocol() -> dict[str, object]:
    _ensure_base_commit()
    if not POPULATION_PATH.is_file() or not AUDIT_PATH.is_file():
        raise ProtocolError("build population and audit before sealing MF3ZQ")
    labels_protocol = _closed_protocol(VISUAL_PROTOCOL)
    formal = _closed_protocol(FORMAL_MF3ZP)
    if formal.get("authorization", {}).get("oracle_headroom") is not False:
        raise ProtocolError("formal MF3ZP oracle authorization unexpectedly opened")
    population_audit = _read_json(AUDIT_PATH)
    if population_audit.get("status") != "MF3ZQ_POPULATION_AUDIT_PASS":
        raise ProtocolError("population audit did not pass")
    implementation = {name: inventory(ROOT / name) for name in IMPLEMENTATION_FILES}
    population = [json.loads(line) for line in POPULATION_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(population) != 80:
        raise ProtocolError("MF3ZQ population must contain exactly 80 events")
    scenes = sorted({str(row["scene_id"]) for row in population})
    episodes = sorted({(str(row["dataset"]), str(row["episode_id"])) for row in population})
    if len(episodes) != 80:
        raise ProtocolError("MF3ZQ requires unique complete episodes")
    return {
        "schema_version": "revealnav-mf3zq-oracle-headroom-protocol/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_MF3ZQ_ORACLE_RESULTS",
        "source_commit": BASE_COMMIT,
        "seal_commit": current_commit(),
        "scientific_scope": "exploratory cognitive-state oracle headroom; not a formal MF3ZP gate",
        "frozen_history": {
            "single_decision_gate_family_stopped": True,
            "prohibited_reruns": ["CAR", "RCSP", "DSR", "MF3ZN", "MF3ZO", "MF3ZP"],
            "formal_mf3zp_protocol_byte_identity": inventory(FORMAL_MF3ZP),
            "formal_mf3zp_oracle_headroom_authorized": False,
        },
        "source_files": {
            "visual_labels": inventory(VISUAL_LABELS),
            "visual_label_manifest": inventory(VISUAL_MANIFEST),
            "visual_review_protocol": inventory(VISUAL_PROTOCOL),
            "mf3zp_reveal_events": inventory(EVENTS),
            "population": inventory(POPULATION_PATH),
            "population_audit": inventory(AUDIT_PATH),
        },
        "implementation_inventory": implementation,
        "population": {
            "events": 80,
            "unique_episodes": 80,
            "raw_scene_count": len(scenes),
            "scene_ids": scenes,
            "episode_ids": [{"dataset": d, "episode_id": e} for d, e in episodes],
            "domain_counts": {"R2R": sum(row["dataset"] == "R2R" for row in population), "RxR": sum(row["dataset"] == "RxR" for row in population)},
            "outcome_blind_selection": True,
        },
        "oracle_boundary": {
            "allowed": ["decision_specific_DEC", "prerequisite_status", "true_SGE", "option_identity", "option_birth", "option_persistence", "control_backed_returnability", "true_reveal", "true_expiry"],
            "forbidden": ["future_action_sequence", "shortest_path", "optimal_low_level_motion", "final_reward", "SR", "SPL", "nDTW", "SDTW", "delta_utility", "CAR_result", "direct_best_action", "teleport"],
        },
        "method": {
            "arms": ["A_BASELINE", "B_ORACLE_DEC", "C_DEC_OPTION_MEMORY", "D_FULL_REVEALSKILL"],
            "skills": ["FOLLOW", "INSPECT", "EXPLORE", "BACKTRACK", "COMMIT", "STOP"],
            "uad_stability_k": UAD_K,
            "memory_budget_m": MEMORY_BUDGET,
            "return_horizon_h": RETURN_HORIZON,
            "option_order": ["frozen_candidate_rank", "first_seen_step", "option_id"],
            "utility": {"nDTW": 0.50, "SDTW": 0.25, "SPL": 0.25},
            "catastrophic_delta_utility": -0.10,
            "no_learning": True,
            "no_threshold_search": True,
        },
        "bootstrap": {"cluster": "raw_mp3d_scene", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "public_split_access": dict(PUBLIC_CLOSED),
        "authorization": {
            "formal_label_validity": False,
            "formal_oracle_gate": False,
            "checkpoint_generation": False,
            "public_evaluation": False,
        },
        "pass_fail": {
            "per_domain_delta_utility": ">0",
            "delta_utility_bootstrap_lower_95": ">0",
            "pcr_relative_reduction_min": 0.25,
            "pcr_bootstrap_lower_95": ">0",
            "catastrophe_rate": "<= baseline; if baseline=0, full=0",
            "unsupported_episode_count": 0,
            "stop_status_on_failure": "MF3ZQ_EXPLORATORY_ORACLE_HEADROOM_FAIL",
        },
    }


def seal_protocol() -> dict[str, object]:
    if PROTOCOL_PATH.exists() or PROTOCOL_PATH.is_symlink():
        raise ProtocolError("MF3ZQ protocol already sealed; refusing overwrite")
    value = build_protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    partial = PROTOCOL_PATH.with_name(PROTOCOL_PATH.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ProtocolError("stale protocol partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, PROTOCOL_PATH)
    return value


def verify_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("MF3ZQ protocol is missing")
    value = _read_json(path)
    if value.get("status") != "SEALED_BEFORE_MF3ZQ_ORACLE_RESULTS":
        raise ProtocolError("MF3ZQ protocol status drift")
    if value.get("source_commit") != BASE_COMMIT:
        raise ProtocolError("MF3ZQ source commit drift")
    if value.get("public_split_access") != PUBLIC_CLOSED:
        raise ProtocolError("MF3ZQ public split opened")
    if value.get("authorization", {}).get("checkpoint_generation") is not False:
        raise ProtocolError("MF3ZQ checkpoint authorization opened")
    for section in ("source_files", "implementation_inventory"):
        for item in value.get(section, {}).values():
            if not isinstance(item, Mapping):
                raise ProtocolError("malformed MF3ZQ inventory")
            current = inventory(ROOT / str(item["path"]))
            if current != dict(item):
                raise ProtocolError(f"MF3ZQ inventory drift: {item['path']}")
    if inventory(FORMAL_MF3ZP) != value["frozen_history"]["formal_mf3zp_protocol_byte_identity"]:
        raise ProtocolError("historical MF3ZP formal protocol changed")
    formal = _closed_protocol(FORMAL_MF3ZP)
    if formal.get("authorization", {}).get("oracle_headroom") is not False:
        raise ProtocolError("historical MF3ZP oracle flag changed")
    return value


__all__ = [
    "ROOT", "REVISION", "OUTPUT", "PROTOCOL_PATH", "POPULATION_PATH", "AUDIT_PATH", "RESULT_PATH",
    "BASE_COMMIT", "PUBLIC_CLOSED", "MEMORY_BUDGET", "RETURN_HORIZON", "UAD_K", "BOOTSTRAP_REPLICATES", "BOOTSTRAP_SEED",
    "ProtocolError", "sha256_file", "inventory", "build_protocol", "seal_protocol", "verify_protocol",
]
