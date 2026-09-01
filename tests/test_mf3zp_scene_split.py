import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pilot_builder", ROOT / "scripts/build_mf3zp_reveal_pilot.py")
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)


class SceneSplitTest(unittest.TestCase):
    def test_raw_scene_fold_ignores_dataset(self):
        scene = "17DRP5sb8fy"
        self.assertEqual(builder.scene_fold(scene), builder.scene_fold(scene))
        self.assertTrue(0 <= builder.scene_fold(scene) < 5)

    def test_outcome_field_rejected(self):
        with self.assertRaises(builder.PilotBuildError):
            builder.reject_outcome_keys({"safe": {"delta_utility": 1}})


if __name__ == "__main__":
    unittest.main()
