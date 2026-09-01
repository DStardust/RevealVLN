from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_mf3zu_rxr_feasibility_pipeline_for_test",
    ROOT / "scripts/run_mf3zu_rxr_feasibility_pipeline.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import MF3ZU pipeline supervisor")
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


class MF3ZUPipelineSupervisorTest(unittest.TestCase):
    def test_fixed_stage_order_and_resources(self):
        stages = PIPELINE.build_stage_plan(
            python="/fixed/python", gpu_id=0, qwen_workers=8
        )
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "causal_observation_replay",
                "fixed_qwen_annotation",
                "freeze_outcome_blind_evidence",
                "once_only_five_fold_training",
                "immutable_result_audit",
                "full_regression",
            ],
        )
        commands = "\n".join(" ".join(stage.command) for stage in stages)
        self.assertNotIn("R2R", commands)
        self.assertNotIn("val_seen", commands)
        self.assertNotIn("val_unseen", commands)
        self.assertNotIn("checkpoint", commands)
        self.assertIn("--max-workers 1 --gpu-ids 0", commands)
        self.assertIn("run --max-workers 8", commands)
        self.assertIn("--device cuda:0", commands)

    def test_support_fail_is_the_only_nonzero_scientific_stage_code(self):
        stages = PIPELINE.build_stage_plan(python="python")
        accepted = {stage.name: stage.accepted_returncodes for stage in stages}
        self.assertEqual(
            accepted["freeze_outcome_blind_evidence"],
            (0, 3),
        )
        for name, values in accepted.items():
            if name != "freeze_outcome_blind_evidence":
                self.assertEqual(values, (0,))


if __name__ == "__main__":
    unittest.main()
