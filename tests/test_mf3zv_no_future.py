import tempfile
import unittest
from pathlib import Path

from revealnav_mf3.progress_state_audit import validate_causal_evidence


class Mf3zvNoFutureTest(unittest.TestCase):
    def test_future_step_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / "a.jpg", root / "b.jpg"]
            for path in files:
                path.write_bytes(b"image")
            with self.assertRaises(ValueError):
                validate_causal_evidence(
                    decision_step=2, evidence_steps=[1, 3], evidence_paths=files
                )

    def test_past_and_current_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / "a.jpg", root / "b.jpg"]
            for path in files:
                path.write_bytes(b"image")
            validate_causal_evidence(
                decision_step=2, evidence_steps=[1, 2], evidence_paths=files
            )


if __name__ == "__main__":
    unittest.main()

