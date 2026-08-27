#!/usr/bin/env python3
"""Stage 7: Phase 0 technical evidence refresh.

Sequence:
  1. snapshot the old canonical evidence file (byte copy + SHA-256):
       artifacts/phase0/evidence_pre_runtime_refresh.json
  2. build artifacts/phase0/PHASE0_TECHNICAL_EVIDENCE_REFRESH.json recording
     the updated technical facts, each backed by SHA-256-cited project-local
     artifacts.

Frozen-evaluator compatibility constraint (documented inside the refresh
file): the frozen evaluator (scripts/evaluate_phase0.py ->
toporeveal.evidence.load_phase0_snapshot) implements artifact verifiers only
for the evidence keys {official_metadata_verified, screening_counts}.  Any
snapshot that asserts the runtime booleans would require evidence keys whose
semantic verifier is not implemented, which the frozen evaluator rejects as
unverifiable input (exit 2) rather than a valid NO-GO (exit 1).  Because
toporeveal/ is frozen and must not be modified in this batch, the canonical
evaluator-consumed snapshot artifacts/phase0/evidence_current.json keeps
claim values inside the frozen verifier's supported set (so the evaluator
still returns a valid NO-GO with its blocker list).  The refreshed runtime
facts live in PHASE0_TECHNICAL_EVIDENCE_REFRESH.json with SHA-256 citations;
promoting them into the evaluator-consumed snapshot requires extending the
frozen verifier first.  This is recorded as a remaining blocker.

Mandatory holds (never relaxed in this batch):
  mp3d_access_authorized = false (no written authorization document exists
  in the project; user statements or sent emails do not count),
  manually_reviewed_count = 0, valid_candidate_count = 0,
  validated_event_count = 0, unique_expiry_count = 0,
  provisional expiry proposals are never counted as validated.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = "/mnt/daiyang/vla"
PHASE0_DIR = os.path.join(PROJECT_ROOT, "artifacts", "phase0")
CURRENT = os.path.join(PHASE0_DIR, "evidence_current.json")
PRE_SNAPSHOT = os.path.join(PHASE0_DIR,
                            "evidence_pre_runtime_refresh.json")
REFRESH = os.path.join(PHASE0_DIR, "PHASE0_TECHNICAL_EVIDENCE_REFRESH.json")
STAGE0 = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                      "phase0_reveal_closure", "STAGE0_PREFLIGHT.json")
RXR_GATE = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                        "RXR_EN_RUNTIME_GATE.json")
STAGE2 = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                      "phase0_reveal_closure",
                      "COLLECTOR_ENGINEERING_VALIDATION.json")
STAGE4 = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                      "phase0_reveal_closure",
                      "STAGE4_TRACE_GENERATION_SUMMARY.json")
MAPPING = os.path.join(PHASE0_DIR, "REVEAL_QUEUE_50_MAPPING.json")
PACKET = os.path.join(PHASE0_DIR, "REVIEW_PACKET_50.json")
GUIDE = os.path.join(PHASE0_DIR, "REVIEW_GUIDE.md")
WITNESS = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                       "phase0_reveal_closure", "witness",
                       "WITNESS_RETURN_EXPIRY_FIRST5.json")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, PROJECT_ROOT)


def cite(p):
    return {"path": rel(p), "sha256": sha256_file(p),
            "bytes": os.path.getsize(p)}


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


def main():
    # 1. snapshot old evidence
    if not os.path.isfile(PRE_SNAPSHOT):
        shutil.copyfile(CURRENT, PRE_SNAPSHOT)
    pre_sha = sha256_file(PRE_SNAPSHOT)
    cur_sha = sha256_file(CURRENT)

    # 2. run the frozen evaluator on the canonical snapshot (expected NO-GO)
    eval_proc = subprocess.run(
        [os.path.join(PROJECT_ROOT, ".envs", "phase0-tools", "bin",
                      "python"),
         os.path.join(PROJECT_ROOT, "scripts", "evaluate_phase0.py"),
         rel(CURRENT)],
        capture_output=True, text=True, cwd=PROJECT_ROOT)
    evaluator_output = None
    try:
        evaluator_output = json.loads(eval_proc.stdout)
    except json.JSONDecodeError:
        pass

    stage0 = load_json(STAGE0)
    rxr_gate = load_json(RXR_GATE)
    stage2 = load_json(STAGE2)
    stage4 = load_json(STAGE4) if os.path.isfile(STAGE4) else {}
    mapping = load_json(MAPPING)
    packet = load_json(PACKET) if os.path.isfile(PACKET) else {}
    witness = load_json(WITNESS) if os.path.isfile(WITNESS) else {}

    inputs_ok = (
        stage0.get("status") == "PASS"
        and rxr_gate.get("verdict") == "RUNTIME_GATE_PASS"
        and stage2.get("status") == "PASS"
        and stage4.get("total_units") == 50
        and stage4.get("counts", {}).get("failed") == 0
        and mapping.get("unique_mapping_50_of_50") is True
        and packet.get("status") == "PASS"
        and packet.get("row_count") == 50
        and packet.get("reviewed_true_count") == 0
        and packet.get("all_rows_pending") is True
        and witness.get("status") == "PASS"
        and witness.get("validated_tx_count") == 0
    )
    evaluator_ok = (
        eval_proc.returncode == 1
        and isinstance(evaluator_output, dict)
        and evaluator_output.get("go") is False
        and cur_sha == pre_sha
    )

    refresh = {
        "manifest": "RevealNav Phase 0 technical evidence refresh "
                    "(runtime closure batch, 2026-08-24)",
        "label": "rxr_train_val_seen_engineering_only",
        "canonical_snapshot": {
            "path": rel(CURRENT),
            "sha256_before_refresh": cur_sha,
            "pre_refresh_snapshot_path": rel(PRE_SNAPSHOT),
            "pre_refresh_snapshot_sha256": pre_sha,
            "snapshot_changed_by_this_batch": cur_sha != pre_sha,
            "frozen_evaluator_exit_code_on_snapshot": eval_proc.returncode,
            "frozen_evaluator_output": evaluator_output,
            "compatibility_note":
                "The frozen evaluator implements artifact verifiers only for "
                "evidence keys {official_metadata_verified, "
                "screening_counts}. Runtime booleans asserted in the "
                "evaluator-consumed snapshot would require evidence keys "
                "with no implemented semantic verifier, which the frozen "
                "evaluator rejects as unverifiable input (exit 2) rather "
                "than a valid NO-GO (exit 1). toporeveal/ is frozen, so the "
                "canonical snapshot keeps verifier-supported claim values; "
                "the refreshed runtime facts are recorded here with SHA-256 "
                "citations. Promoting them into the canonical snapshot "
                "requires extending the frozen verifier first.",
        },
        "updated_technical_facts": {
            "project_self_contained": {
                "value": True,
                "basis": "environment, checkpoints, scenes, caches and "
                         "datasets all under /mnt/daiyang/vla; verified at "
                         "preflight",
                "evidence": [cite(STAGE0)],
            },
            "mp3d_scene_count": {
                "value": 90,
                "basis": "90 scene directories each with glb/navmesh/house; "
                         "Stage 0 re-verification of the accepted MP3D gate",
                "evidence": [cite(STAGE0)],
            },
            "mp3d_access_authorized": {
                "value": False,
                "basis": "no written authorization document exists in the "
                         "project; verbal statements or sent application "
                         "emails do not constitute authorization",
                "evidence": [],
            },
            "habitat_ready": {
                "value": True,
                "basis": "Habitat-Sim 0.1.7 headless RGB/depth smoke PASS on "
                         "the selected GPU; torch 2.11.0+cu128 GPU smoke "
                         "PASS; uv pip check clean at 109 packages with the "
                         "accepted freeze SHA",
                "evidence": [cite(STAGE0)],
            },
            "waypoint_frontend_reproduced": {
                "value": True,
                "basis": "accepted checkpoint gate: waypoint_hfov63/90 "
                         "42/42 keys strict load with finite forward; "
                         "re-verified by SHA provenance in the RxR gate",
                "evidence": [
                    cite(os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                                      "ETPR1_CHECKPOINT_MAIN_ACCEPTANCE.json")),
                    cite(os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                                      "rxr_en_gate",
                                      "RXR_EN_CONFIG_GATE.json")),
                ],
            },
            "etpr1_reproduced": {
                "value": True,
                "basis": "R2R val_seen runtime gate accepted (8 episodes, "
                         "deterministic replay) and RxR train/val_seen "
                         "runtime gate PASS with strict checkpoint load",
                "evidence": [
                    cite(os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                                      "R2R_VAL_SEEN_MAIN_ACCEPTANCE.json")),
                    cite(RXR_GATE),
                ],
            },
            "rxr_english_runtime_status": {
                "value": rxr_gate.get("verdict"),
                "basis": "RxR en-US/en-IN train cold run + fresh-process "
                         "replay identical; two val_seen episodes on "
                         "distinct scenes; zero network attempts; GPU "
                         "memory reclaimed after every run",
                "evidence": [cite(RXR_GATE)],
            },
            "fixed_screening_queue_size": {
                "value": 50,
                "basis": "frozen queue maps 50/50 uniquely to the runtime "
                         "payload with zero mismatches/duplicates",
                "evidence": [cite(MAPPING)],
            },
            "review_packet_generated": {
                "value": packet.get("status") == "PASS",
                "basis": "50 rows, all annotation_status=PENDING and "
                         "reviewed=false; private media manifest recorded",
                "evidence": ([cite(PACKET), cite(GUIDE)]
                             if os.path.isfile(PACKET) else []),
            },
            "prefix_trace_generated_count": {
                "value": (stage4.get("counts", {}).get("ok", 0)
                          + stage4.get("counts", {}).get("ambiguous", 0)),
                "basis": "collector runs producing hash-chained prefix "
                         "traces (OK + deterministic AMBIGUOUS fail-closed "
                         "traces); failed runs excluded",
                "evidence": ([cite(STAGE4)] if os.path.isfile(STAGE4)
                             else []),
            },
            "deterministic_trace_replay_passed": {
                "value": stage2.get("status") == "PASS",
                "basis": "R2R ep326 and RxR train ep43629 collector runs "
                         "replayed in fresh processes with identical chain "
                         "roots",
                "evidence": [cite(STAGE2)],
            },
        },
        "mandatory_holds": {
            "mp3d_access_authorized": False,
            "manually_reviewed_count": 0,
            "valid_candidate_count": 0,
            "validated_event_count": 0,
            "unique_expiry_count": 0,
            "provisional_expiry_counted_as_validated": False,
            "note": "no human review has occurred in this batch; the "
                    "executing agent does not act as a human reviewer; "
                    "Stage 5 expiry proposals are PROVISIONAL/AMBIGUOUS "
                    "engineering witnesses only",
        },
        "witness_summary": {
            "path": rel(WITNESS) if os.path.isfile(WITNESS) else None,
            "episodes": witness.get("episodes") if witness else None,
            "status": witness.get("status") if witness else None,
            "observed_unique_expiry_prefix_count": witness.get(
                "observed_unique_expiry_prefix_count", 0),
            "validated_tx_count": witness.get("validated_tx_count", 0),
            "validated_expiry_claimed": False,
        },
        "remaining_blockers_for_phase0_go": [
            "MP3D access authorization provenance not recorded (written "
            "approval document absent)",
            "manual review of the 50-item packet not performed "
            "(reviewed=0/50; valid-rate undefined)",
            "no RevealEvent has passed full artifact validation",
            "event-count projection below 300 while reviewed count is 0",
            "34/50 collector traces are fail-closed AMBIGUOUS under the "
            "frozen persistent-candidate identity rule; only 16/50 are "
            "identity-clean",
            "the first-five witness engineering probe observed no unique "
            "expiry boundary (four right-censored PROVISIONAL cases and "
            "one AMBIGUOUS case)",
            "frozen evaluator semantic verifiers for runtime claims not "
            "implemented (canonical snapshot cannot yet carry them without "
            "a toporeveal change)",
        ],
        "explicit_non_conclusions": {
            "overall_phase0_go": False,
            "paper_benchmark_claim": False,
            "sota_claim": False,
            "val_unseen_or_test_used": False,
            "training_performed": False,
        },
        "refresh_validation": {
            "inputs_ok": inputs_ok,
            "canonical_evaluator_remains_valid_nogo": evaluator_ok,
        },
    }
    refresh["status"] = "PASS" if inputs_ok and evaluator_ok else "FAIL"
    with open(REFRESH, "w") as fh:
        json.dump(refresh, fh, indent=2)
    print(json.dumps({
        "refresh": rel(REFRESH),
        "pre_snapshot": rel(PRE_SNAPSHOT),
        "canonical_sha_before": cur_sha,
        "evaluator_exit_on_canonical": eval_proc.returncode,
        "evaluator_go": (evaluator_output or {}).get("go"),
    }, indent=2))
    return 0 if refresh["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
