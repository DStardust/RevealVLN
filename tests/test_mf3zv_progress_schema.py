import unittest

from revealnav_mf3.progress_schema import LandmarkProgress, OrdinalProgress, ProgressAtom


class Mf3zvProgressSchemaTest(unittest.TestCase):
    def test_valid_atoms(self):
        ProgressAtom("a", "ORDINAL", "left_turn", "COUNT_TARGET", "2", "second left")
        ProgressAtom(
            "b", "PASSED_LANDMARK", "painting", "PASSED", "true", "past the painting"
        )

    def test_relation_is_family_fixed(self):
        with self.assertRaises(ValueError):
            ProgressAtom("a", "ORDINAL", "door", "PASSED", "2", "second door")

    def test_state_invariants(self):
        OrdinalProgress(1, 2)
        LandmarkProgress(True, True)
        with self.assertRaises(ValueError):
            LandmarkProgress(False, True)


if __name__ == "__main__":
    unittest.main()

