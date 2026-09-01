import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicFailClosedTest(unittest.TestCase):
    def test_parent_public_flags_closed_and_no_checkpoint(self):
        parent = json.loads((ROOT / "artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json").read_text())
        self.assertEqual(parent["authorization"]["public_split_access"], {"test": False, "test_challenge": False, "val_seen": False, "val_unseen": False})
        self.assertFalse((ROOT / "artifacts/training/mf3zp_revealskill_v1/gates/MF3ZP_REVEALSKILL_MODEL.pt").exists())


if __name__ == "__main__":
    unittest.main()
