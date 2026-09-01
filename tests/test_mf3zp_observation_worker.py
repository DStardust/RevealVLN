import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mf3zp_observation_worker",
    ROOT / "scripts/mf3zp_observation_worker.py",
)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class MF3ZPObservationWorkerTest(unittest.TestCase):
    def test_run_contract_is_train_only(self):
        argv = WORKER.run_argv("R2R", "7", ROOT / "artifacts/x")
        joined = " ".join(argv)
        self.assertIn("EVAL.SPLIT train", joined)
        self.assertIn("TASK_CONFIG.DATASET.SPLIT train", joined)
        self.assertNotIn("val_seen", joined)
        self.assertNotIn("val_unseen", joined)
        self.assertNotIn("test", joined)
        self.assertEqual(argv[argv.index("VIDEO_OPTION") + 1], "[]")

    def test_action_signature_ignores_outcome_metrics(self):
        left = {
            "act": 4,
            "ghost_vp": "g1",
            "front_vp": "0",
            "back_path_len": 2,
            "tryout": True,
            "reward": 1.0,
        }
        right = {**left, "reward": -100.0, "done": True}
        self.assertEqual(
            WORKER._action_signature(left),
            WORKER._action_signature(right),
        )

    def test_panorama_capture_is_prefix_local(self):
        observations = {}
        for yaw in WORKER.PANORAMA_YAWS:
            observations[WORKER._sensor_key(yaw)] = torch.full(
                (1, 16, 16, 3),
                yaw % 255,
                dtype=torch.uint8,
            )
        waypoint = {
            "cand_img_idxes": [[0, 3]],
            "cand_angles": [[0.0, 1.57]],
            "cand_distances": [[1.0, 2.0]],
        }
        with tempfile.TemporaryDirectory(
            dir=ROOT / "artifacts",
        ) as directory:
            capture = WORKER.PanoramaCapture(Path(directory))
            capture.capture(observations, waypoint)
            self.assertEqual(len(capture.records), 1)
            record = capture.records[0]
            self.assertEqual(record["step"], 0)
            self.assertEqual(
                [value["local_marker"] for value in record["local_candidates"]],
                ["L00", "L01"],
            )
            self.assertTrue((ROOT / record["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
