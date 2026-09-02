import unittest

from revealnav_mf3.progress_language_filter import InstructionRecord, propose_progress_atoms


class Mf3zvLanguageFilterTest(unittest.TestCase):
    def row(self, text):
        return InstructionRecord("R2R", "1", "scene", text, None)

    def test_numeric_ordinal_navigation(self):
        proposals = propose_progress_atoms(self.row("Walk down the hall and take the second door."))
        self.assertEqual(proposals[0].atom.family, "ORDINAL")
        self.assertEqual(proposals[0].atom.target_value, "2")
        self.assertEqual(proposals[0].mechanical_review_status, "VALID_PROGRESS_ATOM")

    def test_next_to_is_not_an_ordinal(self):
        proposals = propose_progress_atoms(self.row("Stop next to the chair."))
        self.assertEqual(proposals, [])

    def test_after_passing_landmark(self):
        proposals = propose_progress_atoms(
            self.row("After passing the painting, take the first doorway.")
        )
        self.assertEqual(proposals[0].atom.family, "PASSED_LANDMARK")
        self.assertEqual(proposals[0].atom.subject, "painting")

    def test_static_second_bedroom_is_ambiguous(self):
        proposals = propose_progress_atoms(self.row("Wait in the second bedroom."))
        self.assertEqual(proposals[0].mechanical_review_status, "AMBIGUOUS_PROGRESS_ATOM")


if __name__ == "__main__":
    unittest.main()

