import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Mf3zrProtocolTest(unittest.TestCase):
    def test_fixed_source_and_public_closure(self):
        path = ROOT / "artifacts/training/mf3zr_option_bound_support_v1/MF3ZR_OPTION_BOUND_SUPPORT_PROTOCOL.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "SEALED_BEFORE_MF3ZR_SUPPORT_RESULTS")
        self.assertEqual(value["population"]["events"], 80)
        self.assertEqual(value["population"]["domain_counts"], {"R2R": 40, "RxR": 40})
        self.assertEqual(value["public_split_access"], {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False})
        self.assertEqual(value["execution"]["oracle_arms_run"], [])
        self.assertFalse(value["execution"]["checkpoint_generated"])

    def test_historical_formal_protocol_hash_is_fixed(self):
        path = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEALSKILL_PROTOCOL.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), "d0f09395b86804d3afc58f4ec946afc7dfaffd1637c7b8a66a776d58a17cc0c9")


if __name__ == "__main__":
    unittest.main()
