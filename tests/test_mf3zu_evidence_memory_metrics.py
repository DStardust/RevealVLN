from __future__ import annotations

import unittest

import numpy as np

from revealnav_mf3.mf3zu_evidence_memory_metrics import (
    ARM_CURRENT,
    ARM_MEMORY,
    ARM_SHUFFLED,
    apply_fixed_rxr_gates,
    evaluate_three_arm_probe,
    scene_cluster_bootstrap_paired_delta,
)


class MF3ZUEvidenceMemoryMetricsTest(unittest.TestCase):
    def passing_fixture(self, replicates=200):
        required_rows = 60
        control_rows = 20
        rows = required_rows + control_rows
        target = np.zeros(rows, dtype=np.int64)
        mask = np.ones((rows, 2), dtype=bool)
        required = np.zeros(rows, dtype=bool)
        required[:required_rows] = True
        scenes = np.asarray(
            [f"scene-{index // 5:02d}" for index in range(required_rows)]
            + [f"control-{index // 2:02d}" for index in range(control_rows)]
        )
        current = np.tile(np.asarray([[-1.0, 1.0]]), (rows, 1))
        memory = current.copy()
        memory[:required_rows] = np.asarray([1.0, -1.0])
        shuffled = current.copy()
        scores = {
            ARM_CURRENT: current,
            ARM_MEMORY: memory,
            ARM_SHUFFLED: shuffled,
        }
        return evaluate_three_arm_probe(
            scores, target, mask, scenes, required,
            bootstrap_replicates=replicates,
        )

    def test_metrics_and_bootstrap_pass_the_fixed_pattern(self):
        evaluation = self.passing_fixture()
        gates = apply_fixed_rxr_gates(evaluation)
        self.assertTrue(gates["passed"])
        self.assertEqual(
            gates["status"], "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_PASS"
        )
        self.assertGreater(
            evaluation["scene_bootstrap_CI"]["B_minus_A"]["MEMORY_REQUIRED"]["Acc@1"]["lower_95"],
            0.0,
        )
        self.assertEqual(
            evaluation["pairwise_deltas"]["B_minus_A"]["MEMORY_NOT_REQUIRED"]["Acc@1"],
            0.0,
        )

    def test_bootstrap_is_scene_clustered_and_deterministic(self):
        memory = np.asarray([1.0, 1.0, 0.0, 0.0])
        control = np.zeros(4)
        scenes = np.asarray(["a", "a", "b", "b"])
        selected = np.ones(4, dtype=bool)
        first = scene_cluster_bootstrap_paired_delta(
            memory, control, scenes, selected, replicates=100
        )
        second = scene_cluster_bootstrap_paired_delta(
            memory, control, scenes, selected, replicates=100
        )
        self.assertEqual(first, second)
        self.assertEqual(first["scene_count"], 2)

    def test_support_and_specificity_have_distinct_fail_status(self):
        support = self.passing_fixture()
        support["subgroup_support"]["MEMORY_REQUIRED"]["decisions"] = 49
        gate = apply_fixed_rxr_gates(support)
        self.assertEqual(gate["status"], "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL")
        self.assertEqual(gate["final_PASS_FAIL"], "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_FAIL")

        specificity = self.passing_fixture()
        specificity["pairwise_deltas"]["B_minus_C"]["MEMORY_REQUIRED"]["Acc@1"] = 0.0
        specificity["scene_bootstrap_CI"]["B_minus_C"]["MEMORY_REQUIRED"]["Acc@1"]["lower_95"] = 0.0
        gate = apply_fixed_rxr_gates(specificity)
        self.assertEqual(gate["status"], "MF3ZU_RXR_EVIDENCE_SPECIFICITY_FAIL")


if __name__ == "__main__":
    unittest.main()
