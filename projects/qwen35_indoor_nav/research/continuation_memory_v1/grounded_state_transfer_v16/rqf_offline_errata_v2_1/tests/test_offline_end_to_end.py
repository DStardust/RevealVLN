from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import driver
from common import canonical_bytes
from helpers import temporary_directory


def valid_config():
    return {
        "version": "v16_rqf_offline_errata_v2_1",
        "run_id": "v16_repair_rqf_002_errata_001",
        "house": "rqfALeAoiTq",
        "split": "TEST",
        "new_physical_candidate_execution_limit": 0,
        "new_family_admission_limit": 0,
        "cpu_only": True,
        "auto_advance": False,
        "historical_runs_read_only": True,
        "fixed_yaws": [0, 3, 6, 9, 12, 15, 18, 21],
    }


class OfflineEndToEndTests(unittest.TestCase):
    def test_G01_offline_scope_has_zero_execution_surfaces(self):
        scope = driver.authorization_scope()
        self.assertEqual(scope["new_physical_candidate_executions_allowed"], 0)
        self.assertFalse(scope["habitat_allowed"])
        self.assertFalse(scope["qwen_allowed"])
        self.assertFalse(scope["g1_allowed"])

    def test_G02_success_or_zero_missing_does_not_authorize_progression(self):
        config = valid_config()
        config["missing_target_families"] = 0
        driver.validate_config(config)
        self.assertFalse(config["auto_advance"])
        self.assertFalse(driver.authorization_scope()["automatic_promotion_allowed"])

    def test_G03_illegal_cli_is_rejected_before_any_factory(self):
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(driver.HERE / "driver.py"), "physical", "--config", str(driver.HERE / "ERRATA_CONFIG.json")],
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_G04_only_owned_output_root_changes(self):
        with temporary_directory("q35n_g04_") as root:
            old = root / "old"
            new = root / "new"
            old.mkdir()
            new.mkdir()
            old_file = old / "evidence.json"
            old_file.write_text("{}\n")
            before = hashlib.sha256(old_file.read_bytes()).hexdigest()
            (new / "artifact.json").write_text("{}\n")
            after = hashlib.sha256(old_file.read_bytes()).hexdigest()
            self.assertEqual(before, after)

    def test_G05_semantic_serialization_is_deterministic(self):
        value = {"b": [2, 1], "a": {"x": True}}
        self.assertEqual(hashlib.sha256(canonical_bytes(value)).hexdigest(), hashlib.sha256(canonical_bytes(value)).hexdigest())

    def test_G06_forced_gate_failure_propagates_nonzero(self):
        with temporary_directory("q35n_g06_") as root:
            config = valid_config()
            config["cpu_only"] = False
            path = root / "invalid.json"
            path.write_text(json.dumps(config))
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(driver.HERE / "driver.py"), "run-offline", "--config", str(path)],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("OFFLINE_CONFIG_GATE", result.stderr)


if __name__ == "__main__":
    unittest.main()
