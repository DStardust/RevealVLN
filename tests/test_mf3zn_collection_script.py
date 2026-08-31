"""Gated collection-plan and exact-result entrypoint integration test."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.temporal_exact_lattice import canonical_prefix_sha256
from revealnav_mf3.temporal_uad_schema import (
    CausalTemporalStep,
    TEMPORAL_RECORD_LIST_SCHEMA,
    TEMPORAL_RECORD_LIST_STATUS,
    TemporalSequence,
)


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEALER = _load("scripts/seal_mf3zn_tuad_protocol.py", "mf3zn_sealer_collection_test")
COLLECT = _load("scripts/collect_mf3zn_temporal_lattice.py", "mf3zn_collect_test")


def physical(action: str) -> list[dict]:
    return [
        {"act": 4, "ghost_vp": "prefix", "cur_vp": "s", "front_vp": "f0", "back_path_len": 0},
        {"act": 4, "ghost_vp": action, "cur_vp": "prefix", "front_vp": "f1", "back_path_len": 1},
        {"act": 4, "ghost_vp": "later", "cur_vp": action, "front_vp": "f2", "back_path_len": 2},
    ]


def decisions(adapted: str, changed: bool) -> list[dict]:
    index = {"native": 1, "alternative": 2}[adapted]
    return [
        {
            "step": 0, "native_action_index": 1, "adapted_action_index": 1,
            "native_action_id": "native", "adapted_action_id": "native",
            "action_changed": False,
        },
        {
            "step": 1, "native_action_index": 1, "adapted_action_index": index,
            "native_action_id": "native", "adapted_action_id": adapted,
            "action_changed": changed,
        },
        {
            "step": 2, "native_action_index": 1, "adapted_action_index": 1,
            "native_action_id": "native", "adapted_action_id": "native",
            "action_changed": False,
        },
    ]


def inventory(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": COLLECT.sha256_file(path),
    }


def write_outcome(
    path: Path,
    *,
    prefix: str,
    action_id: str,
    metrics: dict[str, float],
) -> dict:
    path.write_text(json.dumps({
        "schema_version": COLLECT.TASK_METRICS_SCHEMA,
        "dataset": "RxR",
        "scene_id": "scene",
        "episode_id": "episode",
        "decision_step": 1,
        "native_prefix_sha256": prefix,
        "action_id": action_id,
        "metrics": metrics,
        "task_metric_payload_read_by_worker": False,
        "public_split_access": False,
    }), encoding="utf-8")
    return inventory(path)


def write_temporal_records(
    path: Path,
    source_commitment: str,
    *,
    candidates: tuple[str, ...] = ("native", "alternative"),
) -> None:
    steps = tuple(CausalTemporalStep(
        step=index,
        native_action_id="native",
        candidate_action_ids=candidates,
        policy_features=np.asarray([1.0, 0.5, 0.2, 0.1, 0.0]),
        instruction_embedding=np.asarray([0.1, 0.2]),
        checkpoint_embedding=np.asarray([0.3, 0.4]),
        action_embeddings=np.eye(len(candidates), dtype=np.float64),
    ) for index in range(2))
    sequence = TemporalSequence.create(
        dataset="RxR",
        scene_id="scene",
        episode_id="episode",
        decision_step=1,
        steps=steps,
    )
    path.write_text(json.dumps({
        "schema_version": TEMPORAL_RECORD_LIST_SCHEMA,
        "status": TEMPORAL_RECORD_LIST_STATUS,
        "source_canonical_identity_sha256": source_commitment,
        "records": [{
            "dataset": sequence.dataset,
            "scene_id": sequence.scene_id,
            "episode_id": sequence.episode_id,
            "decision_step": sequence.decision_step,
            "prefix_sha256": sequence.prefix_sha256,
            "steps": [{
                "step": step.step,
                "native_action_id": step.native_action_id,
                "candidate_action_ids": list(step.candidate_action_ids),
                "policy_features": step.policy_features.tolist(),
                "instruction_embedding": step.instruction_embedding.tolist(),
                "checkpoint_embedding": step.checkpoint_embedding.tolist(),
                "action_embeddings": step.action_embeddings.tolist(),
            } for step in sequence.steps],
        }],
        "public_split_access": False,
    }), encoding="utf-8")


class CollectionEntrypointTest(unittest.TestCase):
    @mock.patch.object(COLLECT, "_recompute_identifiability")
    def test_pass_gate_seals_all_arms_and_revalidates_exact_treatment(
        self, recompute_identifiability,
    ):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            SEALER.seal_protocol(protocol_path, project_root=ROOT)
            causal_source = root / "causal.npz"
            oracle_source = root / "oracle.npz"
            review_source = root / "reviews.json"
            temporal_source = root / "causal-temporal-records.json"
            continuation_source = root / "frozen-controller.bin"
            baseline_source = root / "native-baseline.json"
            np.savez(causal_source, marker=np.asarray([1]))
            np.savez(oracle_source, marker=np.asarray([2]))
            review_source.write_text("{}", encoding="utf-8")
            sealed_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            write_temporal_records(
                temporal_source,
                sealed_protocol["source_population"][
                    "canonical_identity_sha256"
                ],
            )
            continuation_source.write_bytes(b"frozen controller")
            baseline_source.write_text("{}", encoding="utf-8")
            identifiability_path = root / "identifiability.json"
            identifiability = {
                "schema_version": "revealnav-mf3zn-identifiability-result/1",
                "status": "MF3ZN_IDENTIFIABILITY_PASS",
                "collection_authorized": True,
                "public_authorization": False,
                "provenance": {
                    "protocol": {
                        "path": str(protocol_path.relative_to(ROOT)),
                        "sha256": COLLECT.sha256_file(protocol_path),
                    },
                    "causal_probe": {
                        "path": str(temporal_source.relative_to(ROOT)),
                        "bytes": temporal_source.stat().st_size,
                        "sha256": COLLECT.sha256_file(temporal_source),
                    },
                    "oracle_labels": {
                        "path": str(oracle_source.relative_to(ROOT)),
                        "bytes": oracle_source.stat().st_size,
                        "sha256": COLLECT.sha256_file(oracle_source),
                    },
                    "label_reviews": {
                        "path": str(review_source.relative_to(ROOT)),
                        "bytes": review_source.stat().st_size,
                        "sha256": COLLECT.sha256_file(review_source),
                    },
                },
            }
            identifiability_path.write_text(
                json.dumps(identifiability), encoding="utf-8"
            )
            recompute_identifiability.return_value = identifiability
            prefix = canonical_prefix_sha256(physical("native"), 1)
            snapshot_path = root / "snapshots.json"
            snapshot_path.write_text(json.dumps({
                "schema_version": COLLECT.SNAPSHOT_INPUT_SCHEMA,
                "status": "CAUSAL_SNAPSHOTS_FROZEN",
                "frozen_continuation_source": inventory(continuation_source),
                "native_baseline_source": inventory(baseline_source),
                "snapshots": [{
                    "dataset": "RxR",
                    "scene_id": "scene",
                    "episode_id": "episode",
                    "decision_step": 1,
                    "native_action_id": "native",
                    "global_action_ids": ["STOP", "native", "alternative"],
                    "executable_action_indices": [1, 2],
                    "policy_scores": [0.0, 1.0, 0.5],
                    "native_prefix_sha256": prefix,
                }],
            }), encoding="utf-8")
            forged_recomputation = dict(identifiability)
            forged_recomputation["status"] = "MF3ZN_IDENTIFIABILITY_FAIL"
            recompute_identifiability.return_value = forged_recomputation
            with self.assertRaisesRegex(
                COLLECT.TUADProtocolError, "not reproducible"
            ):
                COLLECT.build_collection_plan(
                    protocol_path,
                    identifiability_path,
                    snapshot_path,
                    temporal_source,
                )
            recompute_identifiability.return_value = identifiability
            plan = COLLECT.build_collection_plan(
                protocol_path,
                identifiability_path,
                snapshot_path,
                temporal_source,
            )
            temporal_bytes = temporal_source.read_bytes()
            temporal_value = json.loads(temporal_bytes)
            temporal_value["source_canonical_identity_sha256"] = "0" * 64
            temporal_source.write_text(json.dumps(temporal_value), encoding="utf-8")
            with self.assertRaisesRegex(
                COLLECT.TUADProtocolError, "another source universe"
            ):
                COLLECT._validate_causal_temporal_records(
                    temporal_source,
                    sealed_protocol,
                    COLLECT._load_snapshots(snapshot_path)[0],
                )
            temporal_source.write_bytes(temporal_bytes)
            write_temporal_records(
                temporal_source,
                sealed_protocol["source_population"][
                    "canonical_identity_sha256"
                ],
                candidates=("native", "unsealed-alternative"),
            )
            with self.assertRaisesRegex(
                COLLECT.TUADProtocolError, "candidate support drift"
            ):
                COLLECT._validate_causal_temporal_records(
                    temporal_source,
                    sealed_protocol,
                    COLLECT._load_snapshots(snapshot_path)[0],
                )
            temporal_source.write_bytes(temporal_bytes)
            substituted_temporal = root / "substituted-temporal-records.json"
            substituted_temporal.write_bytes(temporal_bytes)
            with self.assertRaisesRegex(
                COLLECT.TUADProtocolError, "differ from Audit-B provenance"
            ):
                COLLECT.build_collection_plan(
                    protocol_path,
                    identifiability_path,
                    snapshot_path,
                    substituted_temporal,
                )
            self.assertEqual(plan["outcome_fields_used_for_selection"], [])
            self.assertEqual(len(plan["arms"]), 2)
            self.assertEqual(
                plan["causal_temporal_record_source"],
                inventory(temporal_source),
            )
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            event = plan["seal"]["events"][0]
            native_outcome = write_outcome(
                root / "native-outcome.json",
                prefix=prefix,
                action_id="native",
                metrics={"success": 1.0, "spl": 0.8, "ndtw": 0.8, "sdtw": 0.8},
            )
            treatment_outcome = write_outcome(
                root / "treatment-outcome.json",
                prefix=prefix,
                action_id="alternative",
                metrics={"success": 0.0, "spl": 0.6, "ndtw": 0.6, "sdtw": 0.6},
            )
            continuation_sha = plan["frozen_continuation_source"]["sha256"]
            result_path = root / "result.json"
            result_path.write_text(json.dumps({
                "schema_version": COLLECT.COLLECTION_RESULT_SCHEMA,
                "action_list_commitment_sha256": plan["seal"]["action_list_commitment_sha256"],
                "events": [{
                    "lattice_id": event["lattice_id"],
                    "native_prior_intervention_count": 0,
                    "native_second_intervention_count": 0,
                    "native_baseline_sha256": plan[
                        "native_baseline_source"
                    ]["sha256"],
                    "native_continuation_controller_sha256": continuation_sha,
                    "native_physical_trace": physical("native"),
                    "native_decision_trace": decisions("native", False),
                    "native_outcome_source": native_outcome,
                    "treatments": [{
                        "action_id": "alternative",
                        "prior_intervention_count": 0,
                        "second_intervention_count": 0,
                        "continuation_controller_sha256": continuation_sha,
                        "physical_trace": physical("alternative"),
                        "decision_trace": decisions("alternative", True),
                        "outcome_source": treatment_outcome,
                    }],
                }],
            }), encoding="utf-8")
            audit = COLLECT.validate_collection_result(plan_path, result_path)
            self.assertEqual(audit["status"], "MF3ZN_EXACT_LATTICE_AUDIT_PASS")
            self.assertEqual(audit["treatments"], 1)
            self.assertEqual(len(audit["outcomes"]), 2)
            native, treatment = audit["outcomes"]
            self.assertEqual(native["delta_utility"], 0.0)
            self.assertAlmostEqual(treatment["delta_utility"], -0.2)
            self.assertTrue(treatment["catastrophic"])
            self.assertEqual(len(treatment["outcome_commitment_sha256"]), 64)
            self.assertFalse(audit["public_authorization"])

            continuation_bytes = continuation_source.read_bytes()
            continuation_source.write_bytes(b"post-seal replacement")
            with self.assertRaisesRegex(
                COLLECT.TUADProtocolError, "inventory provenance drift"
            ):
                COLLECT.validate_collection_result(plan_path, result_path)
            continuation_source.write_bytes(continuation_bytes)

            tampered_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            tampered_plan["arms"][0]["scene_id"] = "outcome-adaptive-scene"
            tampered_plan_path = root / "tampered-plan.json"
            tampered_plan_path.write_text(
                json.dumps(tampered_plan), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                COLLECT.TUADProtocolError, "hashed causal snapshot sources"
            ):
                COLLECT.validate_collection_result(
                    tampered_plan_path, result_path
                )

            duplicate = json.loads(result_path.read_text(encoding="utf-8"))
            duplicate["events"][0]["treatments"].append(
                dict(duplicate["events"][0]["treatments"][0])
            )
            duplicate_path = root / "duplicate-treatment-result.json"
            duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(
                COLLECT.TUADProtocolError, "duplicate collection treatment"
            ):
                COLLECT.validate_collection_result(plan_path, duplicate_path)

            extra_top_level = json.loads(result_path.read_text(encoding="utf-8"))
            extra_top_level["caller_delta_utility"] = 123.0
            extra_path = root / "extra-result-field.json"
            extra_path.write_text(json.dumps(extra_top_level), encoding="utf-8")
            with self.assertRaisesRegex(
                COLLECT.TUADProtocolError, "top-level schema"
            ):
                COLLECT.validate_collection_result(plan_path, extra_path)

            (root / "treatment-outcome.json").write_text(
                json.dumps({"post_hoc": "replacement"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                COLLECT.TUADProtocolError, "inventory provenance drift"
            ):
                COLLECT.validate_collection_result(plan_path, result_path)


if __name__ == "__main__":
    unittest.main()
