import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from revealnav_mf3.mf3zu_evidence_memory import (
    CANDIDATE_EVIDENCE_FEATURE_DIM,
    EVIDENCE_FEATURE_DIM,
    ConfidenceClass,
    EvidenceJudgement,
    EvidenceRecord,
    EvidenceType,
    K_MEM,
    MF3ZUContractError,
    candidate_memory_feature,
    evidence_contract,
    evidence_numeric_feature,
    instruction_request,
    memory_required,
    parse_instruction_response,
    reject_sensitive_mapping,
    retrieve_evidence,
    validate_evidence_response,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collection = _load_script(
    "mf3zu_collection_test_module",
    "scripts/collect_mf3zu_rxr_observations.py",
)
annotation = _load_script(
    "mf3zu_annotation_test_module",
    "scripts/annotate_mf3zu_rxr_evidence.py",
)


def graph():
    return parse_instruction_response(
        {
            "instruction_atoms": [
                {
                    "instruction_atom_id": "a01",
                    "text": "pass the painting",
                    "semantic_kind": "PASSING",
                    "depends_on": [],
                },
                {
                    "instruction_atom_id": "a02",
                    "text": "take the second left",
                    "semantic_kind": "ORDINAL",
                    "depends_on": ["a01"],
                },
            ]
        },
        instruction="Pass the painting, then take the second left.",
    )


class Mf3zuEvidencePipelineTest(unittest.TestCase):
    def test_annotation_and_memory_builder_cannot_name_separate_label_artifact(self):
        for relative in (
            "scripts/annotate_mf3zu_rxr_evidence.py",
            "scripts/build_mf3zu_evidence_memory.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("MF3ZU_RXR_EXACT_TARGETS", text)
            self.assertNotIn("EXACT_TARGETS_PATH", text)

    def test_replay_collector_never_opens_teacher_bearing_shadow_source(self):
        text = (
            ROOT / "scripts/collect_mf3zu_rxr_observations.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("uad_shadow.jsonl", text)
        self.assertNotIn("source_shadow_trace", text)
        self.assertNotIn("jsonl(shadow", text)
        # Opaque hashes copied from the sanitized population are allowed and
        # retain provenance without opening the underlying mixed-content row.
        self.assertIn("sealed_shadow_provenance", text)
        self.assertIn('"teacher_bearing_source_opened": False', text)

    def test_replay_audit_uses_only_sanitized_population_provenance(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            source_dir = base / "source"
            run_dir = base / "replay"
            arrays_dir = run_dir / "arrays"
            media_dir = run_dir / "media"
            source_dir.mkdir()
            arrays_dir.mkdir(parents=True)
            media_dir.mkdir()

            instruction = np.arange(768, dtype=np.float32)
            history = np.stack((instruction + 1, instruction + 2))
            candidates = np.zeros((2, 3, 768), dtype=np.float32)
            candidates[0, 0] = instruction + 10
            candidates[0, 2] = instruction + 20
            candidates[1, 1] = instruction + 30
            mask = np.asarray(
                [[True, False, True], [False, True, False]], dtype=bool
            )
            scores = np.asarray(
                [[1.0, -np.inf, 2.0], [-np.inf, 3.0, -np.inf]],
                dtype=np.float32,
            )
            feature_path = source_dir / "online_feature.npz"
            np.savez(
                feature_path,
                instruction_embedding=instruction,
                history_embeddings=history,
                candidate_embeddings=candidates,
                candidate_mask=mask,
                native_scores=scores,
            )
            actions = [
                {
                    "i": 0, "act": 4, "cur_vp": "n0", "ghost_vp": "g2",
                    "front_vp": "n0", "back_path_len": 0, "tryout": True,
                },
                {
                    "i": 1, "act": 0, "cur_vp": "n1", "ghost_vp": None,
                    "front_vp": "n1", "back_path_len": 0, "tryout": False,
                },
            ]

            def write_jsonl(path, rows):
                path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )

            source_trace = source_dir / "base_trace.jsonl"
            write_jsonl(source_trace, actions)
            write_jsonl(run_dir / "base_trace.jsonl", actions)
            records = []
            panoramas = []
            candidate_orders = ((2, 0), (1,))
            candidate_ids = (("g2", "g0"), ("g1",))
            for step, (slot_order, ids) in enumerate(
                zip(candidate_orders, candidate_ids, strict=True)
            ):
                arrays_path = arrays_dir / f"prefix_{step:03d}.npz"
                np.savez(
                    arrays_path,
                    instruction=instruction,
                    checkpoint=history[step],
                    action_embeddings=candidates[step, list(slot_order)],
                    policy_scores=scores[step, list(slot_order)],
                )
                panorama = media_dir / f"prefix_{step:03d}.jpg"
                panorama.write_bytes(f"synthetic-panorama-{step}".encode())
                records.append({
                    "step": step,
                    "candidate_action_ids": list(ids),
                    "native_action_id": ids[0],
                    "candidate_relative_heading_rad": [0.1] * len(ids),
                    "arrays": collection.inventory(arrays_path),
                })
                panoramas.append({
                    "step": step,
                    **collection.inventory(panorama),
                })
            write_jsonl(run_dir / "causal_prefix_records.jsonl", records)
            write_jsonl(run_dir / "panorama_manifest.jsonl", panoramas)
            (run_dir / "RUN_SUMMARY.json").write_text(json.dumps({
                "status": "PASS",
                "dataset": "RxR",
                "split": "train",
                "source_prefix_replay_exact": True,
                "source_target_action_compared": False,
                "no_outcome_or_target_input": True,
                "public_split_access": False,
            }), encoding="utf-8")

            feature_inventory = collection.inventory(feature_path)
            trace_inventory = collection.inventory(source_trace)
            task = {
                "scene_id": "scene",
                "episode_id": "episode",
                "feature_rows": 2,
                "physical_steps": 2,
                "ranking_eligible_decisions": 1,
                "source_feature": feature_inventory,
                "source_native_trace": trace_inventory,
                "population_decisions": [{
                    "event_id": "RxR:scene:episode:0",
                    "scene_id": "scene",
                    "episode_id": "episode",
                    "decision_step": 0,
                    "feature_row_index": 0,
                    "candidate_action_ids": ["g2", "g0"],
                    "candidate_graph_indices": [12, 10],
                    "active_candidate_feature_slots": [0, 2],
                    "native_action_id": "g2",
                    "source_feature_path": feature_inventory["path"],
                    "source_feature_sha256": feature_inventory["sha256"],
                    "source_native_trace_path": trace_inventory["path"],
                    "source_native_trace_sha256": trace_inventory["sha256"],
                    "source_shadow_sha256": "a" * 64,
                    "source_shadow_record_hash": "b" * 64,
                }],
            }
            audit = collection.audit_replay(task, run_dir)
            self.assertEqual(audit["status"], "PASS")
            self.assertFalse(audit["teacher_bearing_source_opened"])
            output_rows = collection.jsonl(
                run_dir / "MF3ZU_CAUSAL_DECISIONS.jsonl"
            )
            self.assertEqual(
                output_rows[0]["candidate_id_to_feature_slot"],
                {"g2": 2, "g0": 0},
            )
            self.assertEqual(
                output_rows[0]["sealed_shadow_provenance"],
                {"source_sha256": "a" * 64, "record_hash": "b" * 64},
            )
            self.assertFalse((source_dir / "uad_shadow.jsonl").exists())

    def test_instruction_request_is_fixed_single_extractor(self):
        value = instruction_request("Walk past the sofa and turn left.")
        self.assertEqual(value["model"], "qwen3.8-max")
        self.assertEqual(value["temperature"], 0.0)
        self.assertEqual(value["max_tokens"], 8000)
        self.assertIs(value["enable_thinking"], False)
        reject_sensitive_mapping(value)

    def test_instruction_graph_has_fixed_ontology_and_order(self):
        value = graph()
        self.assertEqual(
            [item.evidence_type for item in value.atoms],
            [EvidenceType.LANDMARK_PASSED, EvidenceType.ORDINAL_COUNT],
        )
        with self.assertRaises(MF3ZUContractError):
            parse_instruction_response(
                {
                    "instruction_atoms": [{
                        "instruction_atom_id": "a02",
                        "text": "left",
                        "semantic_kind": "DIRECTION",
                        "depends_on": [],
                    }]
                },
                instruction="left",
            )

    def test_memory_required_is_outcome_blind_deterministic_rule(self):
        value = validate_evidence_response(
            {
                "atoms": [
                    {
                        "instruction_atom_id": "a01",
                        "active_for_current_ranking": True,
                        "relevant_to_current_ranking": True,
                        "historical_status": "OBSERVED",
                        "current_status": "ABSENT",
                        "source_step": 1,
                        "candidate_ids": ["L00"],
                        "semantic_value": "painting was passed",
                    },
                    {
                        "instruction_atom_id": "a02",
                        "active_for_current_ranking": True,
                        "relevant_to_current_ranking": True,
                        "historical_status": "AMBIGUOUS",
                        "current_status": "ABSENT",
                        "source_step": None,
                        "candidate_ids": [],
                        "semantic_value": "left count is unclear",
                    },
                ]
            },
            graph=graph(),
            decision_step=3,
            allowed_candidate_ids=["L00", "L01"],
        )
        self.assertTrue(memory_required(value))
        encoded = json.dumps([item.as_mapping() for item in value]).casefold()
        for forbidden in ("teacher", "reward", "utility", "outcome"):
            self.assertNotIn(forbidden, encoded)

    def test_no_current_or_later_step_can_be_historical_source(self):
        response = {
            "atoms": [
                {
                    "instruction_atom_id": "a01",
                    "active_for_current_ranking": True,
                    "relevant_to_current_ranking": True,
                    "historical_status": "OBSERVED",
                    "current_status": "ABSENT",
                    "source_step": 3,
                    "candidate_ids": [],
                    "semantic_value": "painting",
                },
                {
                    "instruction_atom_id": "a02",
                    "active_for_current_ranking": False,
                    "relevant_to_current_ranking": False,
                    "historical_status": "ABSENT",
                    "current_status": "ABSENT",
                    "source_step": None,
                    "candidate_ids": [],
                    "semantic_value": "ordinal absent",
                },
            ]
        }
        with self.assertRaises(MF3ZUContractError):
            validate_evidence_response(
                response,
                graph=graph(),
                decision_step=3,
                allowed_candidate_ids=["L00", "L01"],
            )

    def test_evidence_contract_has_exact_causal_history(self):
        contract = evidence_contract(
            instruction="Pass the painting, then take the second left.",
            graph=graph(),
            decision_step=3,
            current_candidates=[
                {"candidate_id": "L00", "relative_heading_rad": -1.2},
                {"candidate_id": "L01", "relative_heading_rad": 0.4},
            ],
            historical_steps=[0, 1, 2],
        )
        self.assertEqual(contract["historical_panorama_steps"], [0, 1, 2])
        reject_sensitive_mapping(contract)
        with self.assertRaises(MF3ZUContractError):
            evidence_contract(
                instruction="x",
                graph=graph(),
                decision_step=3,
                current_candidates=[
                    {"candidate_id": "L00"}, {"candidate_id": "L01"}
                ],
                historical_steps=[0, 2],
            )

    def test_fixed_77d_and_candidate_conditioned_78d_features(self):
        record = EvidenceRecord(
            evidence_id="e1",
            event_id="event",
            source_step=1,
            source_node_id="node",
            instruction_atom_id="a01",
            evidence_type=EvidenceType.LANDMARK_PASSED,
            semantic_value="painting was passed",
            confidence_class=ConfidenceClass.OBSERVED,
            current_status=ConfidenceClass.ABSENT,
            candidate_ids=("g7",),
            source_observation_sha256="a" * 64,
        )
        first = evidence_numeric_feature(record, decision_step=4)
        second = evidence_numeric_feature(record, decision_step=4)
        self.assertEqual(first.shape, (EVIDENCE_FEATURE_DIM,))
        self.assertTrue(np.array_equal(first, second))
        bound = candidate_memory_feature(
            [record],
            active_instruction_atom_ids=["a01"],
            decision_step=4,
            candidate_id="g7",
        )
        unbound = candidate_memory_feature(
            [record],
            active_instruction_atom_ids=["a01"],
            decision_step=4,
            candidate_id="g8",
        )
        self.assertEqual(bound.shape, (CANDIDATE_EVIDENCE_FEATURE_DIM,))
        self.assertEqual(float(bound[-1]), 1.0)
        self.assertEqual(float(unbound[-1]), 0.0)
        self.assertTrue(np.array_equal(bound[:-1], unbound[:-1]))

    def test_retrieval_order_and_budget_are_fixed(self):
        records = [
            EvidenceRecord(
                evidence_id=f"e{index:02d}",
                event_id="event",
                source_step=index,
                source_node_id=f"n{index}",
                instruction_atom_id="a01" if index % 2 else "a02",
                evidence_type=EvidenceType.LANDMARK_SEEN,
                semantic_value=f"landmark {index}",
                confidence_class=ConfidenceClass.OBSERVED,
                current_status=ConfidenceClass.ABSENT,
                candidate_ids=(),
                source_observation_sha256=f"{index % 16:x}" * 64,
            )
            for index in range(10)
        ]
        selected = retrieve_evidence(
            records,
            active_instruction_atom_ids=["a01", "a02"],
        )
        self.assertEqual(len(selected), K_MEM)
        self.assertEqual([row.source_step for row in selected[:5]], [9, 7, 5, 3, 1])
        with self.assertRaises(MF3ZUContractError):
            retrieve_evidence(
                records,
                active_instruction_atom_ids=["a01"],
                budget=7,
            )

    def test_candidate_slot_mapping_uses_bytes_not_graph_indices(self):
        source = np.arange(4 * 768, dtype=np.float32).reshape(4, 768)
        scores = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        active = np.asarray([0, 2, 3])
        replay_order = [3, 0, 2]
        mapped, direct = collection.match_replay_candidates_to_feature_slots(
            replay_embeddings=source[replay_order],
            replay_scores=scores[replay_order],
            source_embeddings=source,
            source_scores=scores,
            active_slots=active,
        )
        self.assertEqual(mapped, replay_order)
        self.assertFalse(direct)

    def test_ambiguous_candidate_slot_mapping_fails_closed(self):
        source = np.zeros((2, 768), dtype=np.float32)
        scores = np.zeros((2,), dtype=np.float32)
        with self.assertRaises(collection.MF3ZUCollectionError):
            collection.match_replay_candidates_to_feature_slots(
                replay_embeddings=source.copy(),
                replay_scores=scores.copy(),
                source_embeddings=source,
                source_scores=scores,
                active_slots=np.asarray([0, 1]),
            )

    def test_storyboard_preserves_complete_panorama_content(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            source = base / "full.jpg"
            image = np.zeros((774, 896, 3), dtype=np.uint8)
            image[:387, :448] = (10, 20, 30)
            image[:387, 448:] = (40, 50, 60)
            image[387:, :448] = (70, 80, 90)
            image[387:, 448:] = (100, 110, 120)
            self.assertTrue(cv2.imwrite(str(source), image))
            item = annotation.inventory(source)
            output = base / "storyboard.jpg"
            result = annotation.build_full_panorama_storyboard(
                [{"decision_step": 0, "full_panorama": item}], output
            )
            rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertIsNotNone(rendered)
            self.assertEqual(rendered.shape[:2], (413, 448))
            # Bottom-right source quadrant survives deterministic downscaling;
            # the old yaw-0-only crop would make this region unavailable.
            pixel = rendered[-30, -30].astype(int)
            self.assertLess(np.max(np.abs(pixel - np.asarray([100, 110, 120]))), 15)
            self.assertEqual(result["steps"], [0])


if __name__ == "__main__":
    unittest.main()
