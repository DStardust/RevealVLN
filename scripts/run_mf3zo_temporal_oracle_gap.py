#!/usr/bin/env python3
"""Build, seal, and execute the one-shot MF3ZO A->B->C pilot.

There are deliberately no collection, training, confirmation, or public-split
commands.  Execution stops at the first scientific/support failure and never
creates a deployment checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zo_pilot import (  # noqa: E402
    EXPECTED_CANONICAL_IDENTITY,
    PilotDataError,
    build_pilot,
    canonical_event_id,
    load_causal_records,
    load_oracle_labels,
    sha256_file,
)
from revealnav_mf3.mf3zo_probes import (  # noqa: E402
    current_snapshot_features,
    oracle_feature_vector,
    probe_a_oracle_relevance,
)
from revealnav_mf3.mf3zo_protocol import (  # noqa: E402
    EXPECTED_PUBLIC_ACCESS,
    OUTPUT_RELATIVE,
    PROTOCOL_NAME,
    ProtocolError,
    seal_protocol,
    verify_protocol,
)


OUTPUT = ROOT / OUTPUT_RELATIVE
PROTOCOL = OUTPUT / PROTOCOL_NAME
SELECTION = OUTPUT / "MF3ZO_PILOT_SELECTION.json"
RECORDS = OUTPUT / "MF3ZO_CAUSAL_TEMPORAL_RECORDS.jsonl"
ORACLE = OUTPUT / "MF3ZO_TEMPORAL_ORACLE_LABELS.jsonl"
AUDIT = OUTPUT / "MF3ZO_PILOT_DATA_AUDIT.json"
PROBE_A = OUTPUT / "MF3ZO_PROBE_A_ORACLE_RELEVANCE.json"
PROBE_B = OUTPUT / "MF3ZO_PROBE_B_TEMPORAL_OBSERVABILITY.json"
PROBE_C = OUTPUT / "MF3ZO_PROBE_C_LEARNED_STATE_RELEVANCE.json"
FINAL = OUTPUT / "MF3ZO_FINAL_RESULT.json"
CHECKPOINT = OUTPUT / "gates/MF3ZO_MODEL.pt"


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite MF3ZO result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise RuntimeError(f"stale MF3ZO result partial: {partial}")
    partial.write_text(json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load project module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection() -> dict:
    value = json.loads(SELECTION.read_text(encoding="utf-8"))
    if value.get("status") != "SEALED_INPUT_POPULATION_SELECTED_OUTCOME_BLIND":
        raise RuntimeError("MF3ZO pilot selection status drift")
    return value


def _targets(event_ids: list[str]) -> np.ndarray:
    """Open exact outcomes only after every required oracle label is verified."""

    car = _load_module(ROOT / "scripts/train_mf3zm_car.py", "mf3zo_probe_target_source")
    car.verify_protocol()
    rows = car._canonical_rows()
    if car._identity_hash(rows) != EXPECTED_CANONICAL_IDENTITY:
        raise RuntimeError("MF3ZO exact outcome source drift")
    by_event = {
        canonical_event_id(
            row["dataset"], row["scene_id"], row["episode_id"],
            int(row["decision"]["step"]),
        ): float(row["target"])
        for row in rows
    }
    if any(event_id not in by_event for event_id in event_ids):
        raise RuntimeError("MF3ZO selected event outcome is unavailable")
    result = np.asarray([by_event[event_id] for event_id in event_ids])
    if not np.isfinite(result).all():
        raise RuntimeError("MF3ZO exact outcome is non-finite")
    return result


def _final_failure(
    protocol: dict,
    *,
    status: str,
    first_failure: str,
    reason: str,
    target_payload_read: bool,
    probe_stages_reached: list[str],
    probes_numerically_executed: list[str],
) -> dict:
    return {
        "schema_version": "revealnav-mf3zo-final-result/1",
        "revision": "mf3zo_temporal_oracle_gap_v1",
        "status": status,
        "first_scientific_failure": first_failure,
        "failure_reason": reason,
        "stop_rule_triggered": True,
        "probe_stages_reached": probe_stages_reached,
        "probes_numerically_executed": probes_numerically_executed,
        "later_probes_skipped": [
            value for value in ("Probe A", "Probe B", "Probe C")
            if value not in probe_stages_reached
        ],
        "target_payload_read": target_payload_read,
        "checkpoint_generated": CHECKPOINT.exists(),
        "formal_teal_collection_started": False,
        "full_tuad_training_started": False,
        "public_split_access": dict(EXPECTED_PUBLIC_ACCESS),
        "protocol_sha256": sha256_file(PROTOCOL),
        "authorization": protocol["authorization"],
        "scientific_scope": (
            "This result applies only to the one-shot MF3ZO pilot stage that "
            "failed; it is not evidence that all temporal information or full "
            "TUAD is mathematically ineffective."
        ),
    }


def run() -> dict:
    protocol = verify_protocol(PROTOCOL, ROOT)
    if any(path.exists() or path.is_symlink() for path in (PROBE_A, PROBE_B, PROBE_C, FINAL)):
        raise RuntimeError("MF3ZO result already exists; one-shot rerun is forbidden")
    if CHECKPOINT.exists() or CHECKPOINT.is_symlink():
        raise RuntimeError("unexpected MF3ZO checkpoint before scientific probes")
    selection = _selection()
    event_rows = selection["events"]
    event_ids = [str(value["event_id"]) for value in event_rows]
    labels = load_oracle_labels(ORACLE)
    if set(labels) != set(event_ids):
        raise RuntimeError("MF3ZO oracle-label identity set differs from pilot")
    complete = [event_id for event_id in event_ids if labels[event_id].complete]
    if len(complete) != len(event_ids):
        unavailable = {
            field: sum(field in labels[event_id].unavailable_fields for event_id in event_ids)
            for field in (
                "target_in_set", "candidate_separated", "evidence_closed",
                "reveal_interval", "expiry_step", "resolvable",
            )
        }
        probe_a = {
            "schema_version": "revealnav-mf3zo-probe-a/1",
            "probe": "A_oracle_relevance",
            "status": "TEMPORAL_ORACLE_RELEVANCE_FAIL",
            "executed": False,
            "failure_kind": "REQUIRED_ORACLE_SUPERVISION_UNAVAILABLE",
            "events": len(event_ids),
            "complete_verified_oracle_labels": len(complete),
            "unavailable_counts": unavailable,
            "surrogate_labels_substituted": False,
            "target_payload_read": False,
            "reason": (
                "Probe A cannot be estimated honestly because no selected event "
                "has a complete independently verified UAD/Reveal/Expiry oracle "
                "record. The protocol forbids surrogate substitution."
            ),
        }
        _atomic_json(PROBE_A, probe_a)
        final = _final_failure(
            protocol,
            status="TEMPORAL_ORACLE_RELEVANCE_FAIL",
            first_failure="Probe A: required oracle supervision unavailable",
            reason=probe_a["reason"],
            target_payload_read=False,
            probe_stages_reached=["Probe A"],
            probes_numerically_executed=[],
        )
        _atomic_json(FINAL, final)
        return final

    # This branch is only reachable with physically present verified labels.
    records = load_causal_records(RECORDS, ROOT)
    if set(records) != set(event_ids):
        raise RuntimeError("MF3ZO causal-record identity set differs from pilot")
    current = np.stack([current_snapshot_features(records[value]) for value in event_ids])
    oracle = np.stack([
        oracle_feature_vector(labels[value], records[value].decision_step)
        for value in event_ids
    ])
    target = _targets(event_ids)
    scenes = np.asarray([str(value["scene_id"]) for value in event_rows])
    datasets = np.asarray([str(value["dataset"]) for value in event_rows])
    folds = np.asarray(protocol["pilot"]["event_folds"], dtype=np.int64)
    probe_a = probe_a_oracle_relevance(
        current, oracle, target, scenes, datasets, folds,
    )
    probe_a.update({"executed": True, "target_payload_read": True})
    _atomic_json(PROBE_A, probe_a)
    if probe_a["status"] != "ORACLE_RELEVANCE_PASS":
        final = _final_failure(
            protocol,
            status="TEMPORAL_ORACLE_RELEVANCE_FAIL",
            first_failure="Probe A: oracle relevance",
            reason="At least one domain failed the presealed DeltaHuber criterion.",
            target_payload_read=True,
            probe_stages_reached=["Probe A"],
            probes_numerically_executed=["Probe A"],
        )
        _atomic_json(FINAL, final)
        return final

    # Probe B requires complete per-prefix checkpoint and executable-candidate
    # embeddings.  The support check precedes any temporal model fitting.
    incomplete = [value for value in event_ids if not records[value].full_prefix_embedding_complete]
    if incomplete:
        probe_b = {
            "schema_version": "revealnav-mf3zo-probe-b/1",
            "probe": "B_temporal_observability",
            "status": "TEMPORAL_CAUSAL_OBSERVABILITY_FAIL",
            "executed": False,
            "failure_kind": "REQUIRED_CAUSAL_PREFIX_EMBEDDINGS_UNAVAILABLE",
            "incomplete_events": len(incomplete),
            "reason": (
                "The sealed pilot lacks complete per-prefix checkpoint and "
                "executable-candidate embeddings; no imputation is permitted."
            ),
        }
        _atomic_json(PROBE_B, probe_b)
        final = _final_failure(
            protocol,
            status="TEMPORAL_CAUSAL_OBSERVABILITY_FAIL",
            first_failure="Probe B: causal temporal input support",
            reason=probe_b["reason"],
            target_payload_read=True,
            probe_stages_reached=["Probe A", "Probe B"],
            probes_numerically_executed=["Probe A"],
        )
        _atomic_json(FINAL, final)
        return final

    raise RuntimeError(
        "sealed MF3ZO pilot unexpectedly has complete Probe-B support; the "
        "pre-result implementation must be independently reviewed before "
        "executing supervised temporal fitting"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build-pilot", "seal", "run"),
        help="execute exactly one predeclared MF3ZO stage",
    )
    args = parser.parse_args()
    if args.command == "build-pilot":
        value = build_pilot(ROOT, OUTPUT)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "seal":
        path, value = seal_protocol(ROOT)
        print(json.dumps({
            "status": value["status"],
            "protocol": str(path.relative_to(ROOT)),
            "protocol_sha256": sha256_file(path),
            "events": value["pilot"]["events"],
            "scenes": value["pilot"]["raw_mp3d_scenes"],
        }, indent=2, sort_keys=True))
        return 0
    value = run()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PilotDataError, ProtocolError, RuntimeError, ValueError) as error:
        print(f"MF3ZO_FAIL_CLOSED: {error}", file=sys.stderr)
        raise SystemExit(1)
