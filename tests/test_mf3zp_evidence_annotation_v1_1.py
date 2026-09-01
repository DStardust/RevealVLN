import importlib.util
from pathlib import Path
import unittest

from revealnav_mf3.qwen_evidence_annotation_v1_1 import validate_evidence_response_v1_1


class EvidenceAnnotationV11Test(unittest.TestCase):
    def row(self, *, instantiated=False, distinguishable=False, resolved=True):
        return {"constraints":{"c1":{"instantiated":instantiated,"distinguishable":distinguishable,"resolved":resolved,"bbox_xyxy":None,"candidate_ids":[],"evidence_image_indices":[0],"evidence":"past evidence remains resolved"}}}

    def test_s_g_e_are_independent(self):
        value=validate_evidence_response_v1_1(self.row(),active_constraint_ids=["c1"],allowed_candidate_ids=[],image_count=1)
        self.assertTrue(value["c1"]["resolved"]); self.assertFalse(value["c1"]["instantiated"])

    def test_structural_boundaries_remain_strict(self):
        bad=self.row(); bad["constraints"]["c1"]["evidence_image_indices"]=[2]
        with self.assertRaises(ValueError): validate_evidence_response_v1_1(bad,active_constraint_ids=["c1"],allowed_candidate_ids=[],image_count=1)
        bad=self.row(); bad["constraints"]["c1"]["candidate_ids"]=["B9"]
        with self.assertRaises(ValueError): validate_evidence_response_v1_1(bad,active_constraint_ids=["c1"],allowed_candidate_ids=["B1"],image_count=1)

    def test_sealed_population_partitions_into_383_plus_155(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "scripts/repair_mf3zp_qwen_evidence_v1_1.py"
        spec = importlib.util.spec_from_file_location("mf3zp_correctness_test", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        valid, selected = module.task_partition()
        self.assertEqual(len(valid), 383)
        self.assertEqual(len(selected), 155)
        self.assertEqual(len({str(task["request_id"]) for task in valid + selected}), 538)


if __name__ == "__main__": unittest.main()
