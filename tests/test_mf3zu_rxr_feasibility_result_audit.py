from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from revealnav_mf3.mf3zu_evidence_memory_metrics import (
    ARM_CURRENT,
    ARM_MEMORY,
    ARM_SHUFFLED,
    ARMS,
    apply_fixed_rxr_gates,
    evaluate_three_arm_probe,
)
from revealnav_mf3.mf3zu_protocol import PUBLIC_CLOSED, REVISION, scene_fold_mapping


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_mf3zu_rxr_feasibility_for_test",
    ROOT / "scripts/audit_mf3zu_rxr_feasibility_result.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import MF3ZU result audit")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class MF3ZURxRFeasibilityResultAuditTest(unittest.TestCase):
    def result(self):
        required_rows, control_rows = 60, 20
        rows = required_rows + control_rows
        target = np.zeros(rows, dtype=np.int64)
        mask = np.ones((rows, 2), dtype=bool)
        required = np.zeros(rows, dtype=bool)
        required[:required_rows] = True
        scenes = np.asarray(
            [f"scene-{index // 5}" for index in range(required_rows)]
            + [f"control-{index // 2}" for index in range(control_rows)]
        )
        episodes = np.asarray([f"episode-{index}" for index in range(rows)])
        folds = scene_fold_mapping(scenes.tolist())
        current = np.tile([[-1.0, 1.0]], (rows, 1))
        memory = current.copy()
        memory[:required_rows] = [1.0, -1.0]
        evaluation = evaluate_three_arm_probe(
            {
                ARM_CURRENT: current,
                ARM_MEMORY: memory,
                ARM_SHUFFLED: current.copy(),
            },
            target,
            mask,
            scenes,
            required,
            bootstrap_replicates=20,
        )
        gates = apply_fixed_rxr_gates(evaluation)
        fold_rows = [{
            "held_fold": fold,
            "normalization_fit_train_fold_only": True,
            "B_C_common_initialization": True,
            "B_C_common_batch_order": True,
            "shuffled_memory": {
                "train_derangement": True,
                "held_donors_train_only": True,
                "outcome_or_target_used": False,
            },
        } for fold in range(5)]
        result = {
            "revision": REVISION,
            "status": gates["status"],
            "final_PASS_FAIL": gates["final_PASS_FAIL"],
            "scope": {"dataset": "RxR", "R2R_evaluated": False},
            "population": {
                "target_blind_decisions": rows,
                "rankable_exact_target_decisions": rows,
                "episodes": len(set(episodes.tolist())),
                "raw_scenes": len(set(scenes.tolist())),
            },
            "memory_required": evaluation["subgroup_support"]["MEMORY_REQUIRED"],
            "arms": list(ARMS),
            "metrics_per_domain": {"RxR": evaluation["metrics"]},
            "pairwise_deltas": evaluation["pairwise_deltas"],
            "scene_bootstrap_CI": evaluation["scene_bootstrap_CI"],
            "training": {
                "complete_five_fold_oof": True,
                "checkpoint_written": False,
                "public_split_access": dict(PUBLIC_CLOSED),
                "full_navigation_run": False,
                "ETP_frozen": True,
                "candidate_generator_frozen": True,
                "visual_backbone_frozen": True,
                "topology_encoder_frozen": True,
                "fixed_schedule": {
                    "epochs": 40,
                    "batch_size": 64,
                    "optimizer": "AdamW",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "seed": 20_260_901,
                    "early_stopping": False,
                    "best_checkpoint_selection": False,
                    "model_or_threshold_selection": False,
                },
                "folds": fold_rows,
            },
            "fixed_gates": gates,
            "public_split_access": dict(PUBLIC_CLOSED),
            "full_navigation_run": False,
            "checkpoint_generated": False,
            "checkpoint_for_deployment": False,
            "deployment_authorized": False,
            "MF3ZT_two_domain_pass_claimed": False,
        }
        oof = []
        for index in range(rows):
            oof.append({
                "event_id": f"event-{index}",
                "scene_id": str(scenes[index]),
                "episode_id": str(episodes[index]),
                "decision_step": index,
                "scene_fold": folds[str(scenes[index])],
                "memory_required": bool(required[index]),
                "candidate_action_ids": ["candidate-0", "candidate-1"],
                "target_index": int(target[index]),
                "scores": {
                    ARM_CURRENT: [float(value) for value in current[index]],
                    ARM_MEMORY: [float(value) for value in memory[index]],
                    ARM_SHUFFLED: [float(value) for value in current[index]],
                },
            })
        probe = AUDIT.ProbeArrays(
            event_id=np.asarray([row["event_id"] for row in oof]),
            scene_id=scenes,
            episode_id=episodes,
            decision_step=np.arange(rows, dtype=np.int64),
            scene_fold=np.asarray(
                [folds[str(scene)] for scene in scenes], dtype=np.int64
            ),
            candidate_action_ids=tuple(
                tuple(row["candidate_action_ids"]) for row in oof
            ),
            candidate_features=np.zeros((rows, 2, 768), dtype=np.float32),
            base_scores=current.astype(np.float32),
            candidate_mask=mask,
            memory_features=np.zeros((rows, 2, 78), dtype=np.float32),
            memory_count=np.zeros(rows, dtype=np.int64),
            memory_required=required,
            target_index=target,
            source_feature_path=tuple("fixture.npz" for _ in range(rows)),
            source_feature_row=np.arange(rows, dtype=np.int64),
            population_rows_before_target=rows,
            evidence_diagnostics={},
        )
        return result, oof, probe

    def test_valid_result_recomputes_gates(self):
        result, oof, probe = self.result()
        audit = AUDIT.audit_result(
            result,
            oof_rows=oof,
            frozen_probe=probe,
            enforce_frozen_counts=False,
            bootstrap_replicates=20,
            require_frozen_provenance=True,
        )
        self.assertTrue(audit["passed"], audit["failures"])
        self.assertTrue(audit["OOF_metrics_recomputed"])
        self.assertTrue(audit["frozen_OOF_provenance_verified"])

    def test_public_or_checkpoint_mutation_fails(self):
        value, oof, probe = self.result()
        value["checkpoint_generated"] = True
        value["public_split_access"] = {**PUBLIC_CLOSED, "val_unseen": True}
        audit = AUDIT.audit_result(
            value,
            oof_rows=oof,
            frozen_probe=probe,
            enforce_frozen_counts=False,
            bootstrap_replicates=20,
            require_frozen_provenance=True,
        )
        self.assertFalse(audit["passed"])
        self.assertIn("public_split_access", audit["failures"])
        self.assertIn("checkpoint_generated", audit["failures"])

    def test_oof_performance_mutation_cannot_pass_self_reported_metrics(self):
        value, oof, probe = self.result()
        oof[0]["scores"][ARM_MEMORY] = [-10.0, 10.0]
        audit = AUDIT.audit_result(
            value,
            oof_rows=oof,
            frozen_probe=probe,
            enforce_frozen_counts=False,
            bootstrap_replicates=20,
            require_frozen_provenance=True,
        )
        self.assertFalse(audit["passed"])
        self.assertIn("metrics_do_not_match_OOF", audit["failures"])

    def test_formal_audit_rejects_nonfrozen_counts_before_bootstrap(self):
        value, oof, probe = self.result()
        audit = AUDIT.audit_result(
            value,
            oof_rows=oof,
            frozen_probe=probe,
            enforce_frozen_counts=True,
            bootstrap_replicates=10_000,
            require_frozen_provenance=True,
        )
        self.assertFalse(audit["passed"])
        self.assertTrue(any(
            failure.startswith("OOF_recomputation_failed:formal OOF support count drift")
            for failure in audit["failures"]
        ))

    def test_bootstrap_replicate_drift_is_recomputed_and_rejected(self):
        value, oof, probe = self.result()
        audit = AUDIT.audit_result(
            value,
            oof_rows=oof,
            frozen_probe=probe,
            enforce_frozen_counts=False,
            bootstrap_replicates=21,
            require_frozen_provenance=True,
        )
        self.assertFalse(audit["passed"])
        self.assertIn("scene_bootstrap_does_not_match_OOF", audit["failures"])

    def test_every_frozen_provenance_field_is_mutation_detected(self):
        value, original, probe = self.result()
        mutations = {
            "event_id": lambda row: row.__setitem__("event_id", "wrong-event"),
            "scene_id": lambda row: row.__setitem__("scene_id", "wrong-scene"),
            "episode_id": lambda row: row.__setitem__("episode_id", "wrong-episode"),
            "decision_step": lambda row: row.__setitem__("decision_step", 9999),
            "scene_fold": lambda row: row.__setitem__(
                "scene_fold", (int(row["scene_fold"]) + 1) % 5
            ),
            "candidate_order": lambda row: row.__setitem__(
                "candidate_action_ids", list(reversed(row["candidate_action_ids"]))
            ),
            "target_index": lambda row: row.__setitem__("target_index", 1),
            "memory_required": lambda row: row.__setitem__(
                "memory_required", not row["memory_required"]
            ),
            "ETP_CURRENT": lambda row: row["scores"].__setitem__(
                ARM_CURRENT, [2.0, -2.0]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(field=name):
                oof = copy.deepcopy(original)
                mutate(oof[0])
                audit = AUDIT.audit_result(
                    value,
                    oof_rows=oof,
                    frozen_probe=probe,
                    enforce_frozen_counts=False,
                    bootstrap_replicates=20,
                    require_frozen_provenance=True,
                )
                self.assertFalse(audit["passed"])
                self.assertTrue(any(
                    failure.startswith("frozen_OOF_provenance_failed:")
                    for failure in audit["failures"]
                ), audit["failures"])


if __name__ == "__main__":
    unittest.main()
