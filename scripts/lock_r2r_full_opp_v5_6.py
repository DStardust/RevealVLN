#!/usr/bin/env python3
"""Lock the complete V5.6 transitive source closure after its dev gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / (
    "artifacts/evaluation/mf2_r2r_full_opp_v5_6_seen_active_dev/"
    "R2R_FULL_OPP_RESULT_V5_6.json"
)
LOCK = ROOT / "locks/R2R_FULL_OPP_CONTROLLER_V5_6.json"
SOURCES = (
    "scripts/r2r_full_opp_worker_v5_6.py",
    "scripts/r2r_native_preservation_worker_v5_5.py",
    "scripts/r2r_continuous_controller_worker_v5_4.py",
    "scripts/r2r_continuous_controller_worker_v5_3.py",
    "scripts/r2r_continuous_controller_worker_v5_2.py",
    "scripts/r2r_action_enabled_pilot_worker_v5.py",
    "revealnav_mf2r4/__init__.py",
    "revealnav_mf2r4/controller.py",
    "revealnav_mf2r4/fusion.py",
    "revealnav_mf2r4/integrated_controller.py",
    "revealnav_mf2r4/model.py",
    "revealnav_mf2r4/post_excursion.py",
    "artifacts/design/MF2_FULL_OPP_ACTION_ORDER_CORRECTION_V5_6.md",
    "artifacts/evaluation/mf2_ecog_opp_shared_calibration_v3/RXR_ECOG_OPP_SHARED_CALIBRATION_RESULT_V3.json",
    "artifacts/evaluation/mf2_r2r_full_opp_v5_6_seen_active_dev/R2R_FULL_OPP_PROTOCOL_V5_6.json",
    "artifacts/evaluation/mf2_r2r_full_opp_v5_6_seen_active_dev/R2R_FULL_OPP_RESULT_V5_6.json",
    "locks/REE_Q_FUSION_CONTROLLER_V4_4.json",
    "locks/POST_EXCURSION_INTEGRATED_CONTROLLER_V4_9.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    result = json.loads(RESULT.read_text())
    if not (
        result.get("status")
        == "V5_6_ENGINEERING_PASS_DIRECTIONALLY_POSITIVE_INCONCLUSIVE"
        and all(result.get("engineering_gates", {}).values())
        and result.get("scientific_gates", {}).get("directional_positive") is True
        and result.get("unseen_or_test_accessed") is False
    ):
        raise RuntimeError("V5.6 development gate is not lockable")
    sources = {}
    for relative in SOURCES:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"invalid V5.6 source: {relative}")
        sources[relative] = {
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
        }
    value = {
        "schema_version": "revealnav-r2r-full-opp-controller-lock/5.6",
        "status": "LOCKED_FOR_FRESH_VAL_SEEN_CONFIRMATION",
        "scientific_status": "directionally_positive_development_only",
        "frozen_thresholds": {
            "persistence_k": 3, "opv_threshold": 0.025,
            "discriminable_threshold": 0.7, "evidence_threshold": 0.5,
            "target_threshold": 0.3, "expiry_threshold": 0.3,
            "reveal_threshold": 0.5, "wrong_commitment_weight": 5.0,
        },
        "source_closure": sources,
        "next_gate": "outcome-blind activation screen and paired evaluation on previously unexecuted R2R val_seen episodes",
        "unseen_access_authorized": False,
        "paper_result": False,
    }
    if LOCK.exists() and json.loads(LOCK.read_text()) != value:
        raise RuntimeError("existing V5.6 lock differs")
    if not LOCK.exists():
        part = LOCK.with_name(LOCK.name + ".part")
        part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(part, LOCK)
    print(json.dumps({
        "status": value["status"], "sources": len(sources),
        "lock_sha256": sha256_file(LOCK),
    }))


if __name__ == "__main__":
    main()
