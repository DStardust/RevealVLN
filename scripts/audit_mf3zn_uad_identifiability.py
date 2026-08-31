#!/usr/bin/env python3
"""Run the sealed MF3ZN UAD identifiability audits before collection.

Causal probe inputs, oracle supervision, and human review labels must be three
different files.  The script refuses extra keys and aligns them by an immutable
event ID, so a convenient mixed feature/label shard cannot become an inference
input by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revealnav_mf3.tuad_identifiability import (  # noqa: E402
    canonical_audit_event_id,
    causal_observability_audit,
    decision_time_uad_truth,
    deterministic_review_pilot_indices,
    identifiability_gate,
    label_validity_audit,
    oracle_relevance_audit,
)
from revealnav_mf3.temporal_uad_features import (  # noqa: E402
    TEMPORAL_SUMMARY_NAMES,
    causal_sequence_features,
    causal_temporal_summary,
)
from revealnav_mf3.temporal_uad_schema import (  # noqa: E402
    TemporalSequence,
    temporal_record_list_from_mapping,
)
from revealnav_mf3.tuad_protocol import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    IDENTIFIABILITY_CANONICAL_IDENTITY_SHA256,
    IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS,
    IDENTIFIABILITY_EXPECTED_ROWS,
    IDENTIFIABILITY_EXPECTED_SCENES,
    LABEL_VALIDITY_PILOT_ROWS,
    TUADProtocolError,
    sha256_file,
    verify_protocol,
)


ORACLE_KEYS = frozenset({
    "event_id",
    "delta_utility",
    "target_in_set",
    "candidate_separated",
    "evidence_closed",
    "factor_mask",
    "reveal_offset",
    "expiry_offset",
    "reveal_event",
    "reveal_at_risk",
    "expiry_event",
    "expiry_at_risk",
})
REVIEW_KEYS = frozenset({
    "event_id",
    "scene_id",
    "uad_rater_a",
    "uad_rater_b",
    "evidence_rater_a",
    "evidence_rater_b",
})
PARENT_CAR_TRAINER = PROJECT_ROOT / "scripts/train_mf3zm_car.py"


def _strict_json_text(payload: str, name: str) -> object:
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise TUADProtocolError(f"duplicate {name} JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(token: str):
        raise TUADProtocolError(f"non-finite {name} JSON constant: {token}")

    return json.loads(
        payload,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def _load_npz(path: Path, exact_keys: frozenset[str], name: str) -> dict[str, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise TUADProtocolError(f"invalid {name} artifact: {path}")
    with np.load(path, allow_pickle=False) as source:
        observed = set(source.files)
        if observed != exact_keys:
            raise TUADProtocolError(
                f"{name} schema drift; missing={sorted(exact_keys - observed)}, "
                f"extra={sorted(observed - exact_keys)}"
            )
        return {key: np.array(source[key], copy=True) for key in exact_keys}


def _load_causal_records(path: Path) -> tuple[tuple[TemporalSequence, ...], str]:
    if not path.is_file() or path.is_symlink():
        raise TUADProtocolError(f"invalid causal record artifact: {path}")
    try:
        value = _strict_json_text(
            path.read_text(encoding="utf-8"), "causal record",
        )
        if not isinstance(value, dict):
            raise TypeError("causal record list must be an object")
        return temporal_record_list_from_mapping(value)
    except Exception as error:
        # Keep cohort/schema substitution on the same fail-closed boundary.
        raise TUADProtocolError(
            "identifiability source universe drift: invalid causal record list"
        ) from error


def _string_vector(value: np.ndarray, name: str) -> np.ndarray:
    if value.ndim != 1 or value.dtype.kind not in "US" or len(value) == 0:
        raise TUADProtocolError(f"{name} must be a nonempty string vector")
    result = value.astype(str)
    if any(not item for item in result.tolist()):
        raise TUADProtocolError(f"{name} contains an empty value")
    return result


def _load_reviews(path: Path) -> dict[str, list]:
    if not path.is_file() or path.is_symlink():
        raise TUADProtocolError(f"invalid label-review artifact: {path}")
    value = _strict_json_text(
        path.read_text(encoding="utf-8"), "label-review",
    )
    if not isinstance(value, dict) or set(value) != REVIEW_KEYS:
        observed = set(value) if isinstance(value, dict) else set()
        raise TUADProtocolError(
            f"label-review schema drift; missing={sorted(REVIEW_KEYS - observed)}, "
            f"extra={sorted(observed - REVIEW_KEYS)}"
        )
    if any(not isinstance(value[key], list) for key in REVIEW_KEYS):
        raise TUADProtocolError("all label-review fields must be lists")
    lengths = {len(value[key]) for key in REVIEW_KEYS}
    if len(lengths) != 1 or next(iter(lengths), 0) == 0:
        raise TUADProtocolError("label-review columns have mismatched lengths")
    return value


def _event_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"revealnav-mf3zn-audit-events/1\0")
    for value in values.tolist():
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sealed_parent_population(
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reverify the original CAR files/data and return its exact event order."""

    spec = importlib.util.spec_from_file_location(
        "sealed_mf3zm_car_source_for_tuad", PARENT_CAR_TRAINER
    )
    if spec is None or spec.loader is None:
        raise TUADProtocolError("cannot load sealed MF3ZM source verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.verify_protocol()
        rows = module._canonical_rows()
    except Exception as error:
        raise TUADProtocolError("sealed MF3ZM source verification failed") from error
    event_ids = np.asarray([
        canonical_audit_event_id(
            row["dataset"], row["scene_id"], row["episode_id"],
            int(row["decision"]["step"]),
        )
        for row in rows
    ])
    return (
        event_ids,
        np.asarray([str(row["scene_id"]) for row in rows]),
        np.asarray([str(row["dataset"]) for row in rows]),
        np.asarray([float(row["target"]) for row in rows], dtype=np.float64),
    )


def _causal_probe_tensors(
    sequences: tuple[TemporalSequence, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild both fixed probes at every prefix from strict records."""

    if not sequences:
        raise TUADProtocolError("causal sequence population is empty")
    maximum = max(len(sequence.steps) for sequence in sequences)
    snapshot_rows: list[list[np.ndarray]] = []
    temporal = np.zeros(
        (len(sequences), maximum, len(TEMPORAL_SUMMARY_NAMES)),
        dtype=np.float64,
    )
    mask = np.zeros((len(sequences), maximum), dtype=np.bool_)
    for row, sequence in enumerate(sequences):
        if sequence.steps[-1].step != sequence.decision_step:
            raise TUADProtocolError(
                "causal temporal sequence does not end at its decision step"
            )
        event_snapshot_rows = []
        for prefix_index, step in enumerate(sequence.steps):
            prefix = TemporalSequence.create(
                dataset=sequence.dataset,
                scene_id=sequence.scene_id,
                episode_id=sequence.episode_id,
                decision_step=step.step,
                steps=sequence.steps[: prefix_index + 1],
            )
            isolated_current = TemporalSequence.create(
                dataset=sequence.dataset,
                scene_id=sequence.scene_id,
                episode_id=sequence.episode_id,
                decision_step=step.step,
                steps=(step,),
            )
            event_snapshot_rows.append(
                np.asarray(causal_sequence_features(isolated_current)[-1])
            )
            temporal[row, prefix_index] = causal_temporal_summary(prefix)
            mask[row, prefix_index] = True
        snapshot_rows.append(event_snapshot_rows)
    widths = {
        value.shape for event in snapshot_rows for value in event
    }
    if len(widths) != 1:
        raise TUADProtocolError(
            "current-only causal feature width changes across records"
        )
    width = next(iter(widths))[0]
    snapshots = np.zeros(
        (len(sequences), maximum, width), dtype=np.float64,
    )
    for row, event in enumerate(snapshot_rows):
        snapshots[row, :len(event)] = np.stack(event)
    return snapshots, temporal, mask


def _validate_oracle_arrays(
    oracle: dict[str, np.ndarray],
    event_ids: np.ndarray,
    expected_mask: np.ndarray,
) -> None:
    rows, prefixes = expected_mask.shape
    oracle_ids = _string_vector(oracle["event_id"], "oracle event_id")
    if not np.array_equal(event_ids, oracle_ids):
        raise TUADProtocolError("causal/oracle audit event identity mismatch")
    for key in (
        "target_in_set", "candidate_separated", "evidence_closed",
        "factor_mask", "reveal_event", "reveal_at_risk", "expiry_event",
        "expiry_at_risk",
    ):
        value = np.asarray(oracle[key])
        if value.dtype != np.bool_ or value.shape != (rows, prefixes):
            raise TUADProtocolError(
                f"oracle {key} must be an exact Boolean prefix matrix"
            )
    if not np.array_equal(oracle["factor_mask"], expected_mask):
        raise TUADProtocolError(
            "oracle factor mask is not aligned to strict causal prefixes"
        )
    delta = np.asarray(oracle["delta_utility"])
    if (
        delta.ndim != 1
        or len(delta) != rows
        or not np.issubdtype(delta.dtype, np.floating)
        or not np.isfinite(delta).all()
    ):
        raise TUADProtocolError("oracle delta_utility must be a finite float vector")
    for key in ("reveal_offset", "expiry_offset"):
        value = np.asarray(oracle[key])
        if (
            value.ndim != 1
            or len(value) != rows
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
        ):
            raise TUADProtocolError(f"oracle {key} must be a finite float vector")


def _atomic_json(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise TUADProtocolError(f"refusing to overwrite identifiability result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise TUADProtocolError(f"stale identifiability result partial: {partial}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _project_relative(path: Path, name: str) -> str:
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    if root not in resolved.parents:
        raise TUADProtocolError(f"{name} escaped the project root")
    return str(resolved.relative_to(root))


def run_audit(
    protocol_path: Path,
    causal_path: Path,
    oracle_path: Path,
    reviews_path: Path,
) -> dict:
    for path, name in (
        (protocol_path, "protocol"),
        (causal_path, "causal probe"),
        (oracle_path, "oracle labels"),
        (reviews_path, "label reviews"),
    ):
        _project_relative(path, name)
    protocol = verify_protocol(protocol_path, root=PROJECT_ROOT)
    if protocol["status"] != "SEALED_BEFORE_IDENTIFIABILITY_RESULTS":
        raise TUADProtocolError("identifiability protocol was not sealed pre-result")
    if causal_path.resolve() == oracle_path.resolve():
        raise TUADProtocolError("causal and oracle audit data must be separate files")
    initial_inventory = {
        name: (path.stat().st_size, sha256_file(path))
        for name, path in (
            ("protocol", protocol_path),
            ("causal_probe", causal_path),
            ("oracle_labels", oracle_path),
            ("label_reviews", reviews_path),
        )
    }

    sequences, source_identity = _load_causal_records(causal_path)
    causal_ids = np.asarray([
        canonical_audit_event_id(
            sequence.dataset,
            sequence.scene_id,
            sequence.episode_id,
            sequence.decision_step,
        )
        for sequence in sequences
    ])
    scenes = np.asarray([sequence.scene_id for sequence in sequences])
    datasets = np.asarray([sequence.dataset for sequence in sequences])
    rows = len(sequences)
    if len(set(causal_ids.tolist())) != rows:
        raise TUADProtocolError("duplicate causal audit event ID")
    domain_counts = {
        domain: int(np.sum(datasets == domain))
        for domain in sorted(set(datasets.tolist()))
    }
    expected_population = protocol["source_population"]
    if (
        rows != IDENTIFIABILITY_EXPECTED_ROWS
        or len(set(scenes.tolist())) != IDENTIFIABILITY_EXPECTED_SCENES
        or domain_counts != IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS
        or source_identity != IDENTIFIABILITY_CANONICAL_IDENTITY_SHA256
        or expected_population["rows"] != rows
        or expected_population["raw_mp3d_scenes"]
        != len(set(scenes.tolist()))
        or expected_population["domain_counts"] != domain_counts
        or expected_population["canonical_identity_sha256"] != source_identity
    ):
        raise TUADProtocolError("identifiability source universe drift")
    (
        expected_ids,
        expected_scenes,
        expected_datasets,
        expected_delta_utility,
    ) = _sealed_parent_population()
    if not (
        np.array_equal(causal_ids, expected_ids)
        and np.array_equal(scenes, expected_scenes)
        and np.array_equal(datasets, expected_datasets)
    ):
        raise TUADProtocolError("identifiability event-level source identity drift")

    snapshots, temporal, causal_mask = _causal_probe_tensors(sequences)
    oracle = _load_npz(oracle_path, ORACLE_KEYS, "oracle label")
    _validate_oracle_arrays(oracle, causal_ids, causal_mask)
    if not np.array_equal(
        np.asarray(oracle["delta_utility"], dtype=np.float64),
        expected_delta_utility,
    ):
        raise TUADProtocolError(
            "oracle delta_utility differs from sealed exact CAR outcomes"
        )
    reviews = _load_reviews(reviews_path)
    review_ids = np.asarray(reviews["event_id"], dtype=str)
    review_scenes = np.asarray(reviews["scene_id"], dtype=str)
    pilot_indices = deterministic_review_pilot_indices(
        causal_ids,
        scenes,
        pilot_events=LABEL_VALIDITY_PILOT_ROWS,
        required_scenes=IDENTIFIABILITY_EXPECTED_SCENES,
    )
    expected_review_ids = causal_ids[pilot_indices]
    expected_review_scenes = scenes[pilot_indices]
    if not (
        np.array_equal(review_ids, expected_review_ids)
        and np.array_equal(review_scenes, expected_review_scenes)
    ):
        raise TUADProtocolError(
            "label-validity reviews differ from the deterministic presealed pilot"
        )

    uad_state = decision_time_uad_truth(
        oracle["target_in_set"],
        oracle["candidate_separated"],
        oracle["evidence_closed"],
        oracle["factor_mask"],
    )
    oracle_features = np.column_stack((
        uad_state == "U",
        uad_state == "A",
        uad_state == "D",
        oracle["reveal_offset"],
        oracle["expiry_offset"],
    ))
    decision_rows = oracle["factor_mask"].sum(axis=1) - 1
    current_features = snapshots[np.arange(rows), decision_rows]
    relevance = oracle_relevance_audit(
        current_features,
        oracle_features,
        oracle["delta_utility"],
        scenes,
        datasets,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    observability = causal_observability_audit(
        snapshots,
        temporal,
        oracle["target_in_set"],
        oracle["candidate_separated"],
        oracle["evidence_closed"],
        oracle["factor_mask"],
        oracle["reveal_event"],
        oracle["reveal_at_risk"],
        oracle["expiry_event"],
        oracle["expiry_at_risk"],
        scenes,
        datasets,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED + 101,
    )
    validity = label_validity_audit(
        reviews["uad_rater_a"],
        reviews["uad_rater_b"],
        reviews["evidence_rater_a"],
        reviews["evidence_rater_b"],
        reviews["scene_id"],
        minimum_events=LABEL_VALIDITY_PILOT_ROWS,
        expected_scene_count=IDENTIFIABILITY_EXPECTED_SCENES,
        scene_capacities={
            scene: int(np.sum(scenes == scene))
            for scene in sorted(set(scenes.tolist()))
        },
    )
    result = identifiability_gate(relevance, observability, validity)
    final_inventory = {
        name: (path.stat().st_size, sha256_file(path))
        for name, path in (
            ("protocol", protocol_path),
            ("causal_probe", causal_path),
            ("oracle_labels", oracle_path),
            ("label_reviews", reviews_path),
        )
    }
    if final_inventory != initial_inventory:
        raise TUADProtocolError("identifiability input changed during audit")
    result["provenance"] = {
        "protocol": {
            "path": _project_relative(protocol_path, "protocol"),
            "sha256": sha256_file(protocol_path),
        },
        "causal_probe": {
            "path": _project_relative(causal_path, "causal probe"),
            "bytes": causal_path.stat().st_size,
            "sha256": sha256_file(causal_path),
        },
        "oracle_labels": {
            "path": _project_relative(oracle_path, "oracle labels"),
            "bytes": oracle_path.stat().st_size,
            "sha256": sha256_file(oracle_path),
        },
        "label_reviews": {
            "path": _project_relative(reviews_path, "label reviews"),
            "bytes": reviews_path.stat().st_size,
            "sha256": sha256_file(reviews_path),
        },
        "event_count": rows,
        "scene_count": len(set(scenes.tolist())),
        "domain_counts": domain_counts,
        "source_canonical_identity_sha256": source_identity,
        "event_identity_sha256": _event_digest(causal_ids),
        "causal_prefix_identity_sha256": _event_digest(np.asarray([
            sequence.prefix_sha256 for sequence in sequences
        ])),
        "causal_features_recomputed_from_strict_records": True,
        "review_pilot": {
            "selection": "deterministic_capacity_balanced_sha256_round_robin",
            "events": LABEL_VALIDITY_PILOT_ROWS,
            "scenes": IDENTIFIABILITY_EXPECTED_SCENES,
            "event_identity_sha256": _event_digest(expected_review_ids),
        },
        "audit_parameters": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "uad_stability_prefixes": 3,
        },
        "causal_and_oracle_stored_separately": causal_path.resolve() != oracle_path.resolve(),
    }
    return result


def _project_artifact(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise TUADProtocolError(f"{name} provenance path drift")
    root = PROJECT_ROOT.resolve()
    path = (root / value).resolve()
    if root not in path.parents:
        raise TUADProtocolError(f"{name} provenance escaped project root")
    return path


def verify_identifiability_result(path: Path) -> dict:
    """Re-run the fixed audit and require byte-semantic result equality.

    Gated downstream entrypoints can call this helper instead of trusting the
    mutable top-level PASS Boolean in an identifiability JSON artifact.
    """

    if not path.is_file() or path.is_symlink():
        raise TUADProtocolError(f"invalid identifiability result: {path}")
    observed = _strict_json_text(
        path.read_text(encoding="utf-8"), "identifiability result",
    )
    if not isinstance(observed, dict):
        raise TUADProtocolError("identifiability result must be a JSON object")
    provenance = observed.get("provenance")
    if not isinstance(provenance, dict):
        raise TUADProtocolError("identifiability result lacks provenance")
    try:
        protocol_path = _project_artifact(
            provenance["protocol"]["path"], "protocol"
        )
        causal_path = _project_artifact(
            provenance["causal_probe"]["path"], "causal probe"
        )
        oracle_path = _project_artifact(
            provenance["oracle_labels"]["path"], "oracle labels"
        )
        reviews_path = _project_artifact(
            provenance["label_reviews"]["path"], "label reviews"
        )
    except (KeyError, TypeError) as error:
        raise TUADProtocolError(
            "identifiability result provenance schema drift"
        ) from error
    expected = run_audit(
        protocol_path, causal_path, oracle_path, reviews_path,
    )
    if observed != expected:
        raise TUADProtocolError(
            "identifiability result differs from deterministic recomputation"
        )
    return expected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _project_relative(args.output, "identifiability output")
    result = run_audit(args.protocol, args.causal, args.oracle, args.reviews)
    _atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "collection_authorized": result["collection_authorized"],
        "public_authorization": result["public_authorization"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0 if result["collection_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
