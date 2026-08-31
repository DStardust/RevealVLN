import copy
import unittest

from revealnav_mf3.mf3zo_protocol import (
    EXPECTED_PUBLIC_ACCESS,
    ProtocolError,
    SCHEMA_VERSION,
    STATUS,
    validate_protocol,
)


def valid_protocol():
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": "mf3zo_temporal_oracle_gap_v1",
        "status": STATUS,
        "public_split_access": dict(EXPECTED_PUBLIC_ACCESS),
        "authorization": {
            "checkpoint_generation": False,
            "formal_teal_collection": False,
            "full_tuad_training": False,
            "public_evaluation": False,
        },
        "family_tombstone": {
            "name": "FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED",
            "value": True,
        },
        "fixed_probe_configuration": {
            "architecture_grid": [],
            "feature_subset_search": False,
            "hyperparameter_search": False,
            "regularization_grid": [],
            "seed_selection": False,
            "threshold_grid": [],
            "ridge_l2": 1.0,
            "outer_folds": 5,
            "bootstrap_replicates": 10000,
        },
    }


class MF3ZOProtocolTest(unittest.TestCase):
    def test_fixed_protocol_validates(self):
        validate_protocol(valid_protocol())

    def test_public_split_and_search_fail_closed(self):
        public = valid_protocol()
        public["public_split_access"]["val_seen"] = True
        with self.assertRaises(ProtocolError):
            validate_protocol(public)
        search = valid_protocol()
        search["fixed_probe_configuration"]["threshold_grid"] = [0.0]
        with self.assertRaises(ProtocolError):
            validate_protocol(search)

    def test_checkpoint_authorization_cannot_be_enabled(self):
        value = valid_protocol()
        value["authorization"]["checkpoint_generation"] = True
        with self.assertRaises(ProtocolError):
            validate_protocol(value)


if __name__ == "__main__":
    unittest.main()
