import unittest

from revealnav_mf3.progress_language_filter import AtomProposal, deterministic_scene_round_robin
from revealnav_mf3.progress_schema import ProgressAtom


def proposal(scene, episode):
    return AtomProposal(
        "R2R",
        episode,
        scene,
        "take the first door",
        None,
        ProgressAtom(episode, "ORDINAL", "door", "COUNT_TARGET", "1", "first door"),
        9,
        19,
        "VALID_PROGRESS_ATOM",
        "fixed",
    )


class Mf3zvSceneBalanceTest(unittest.TestCase):
    def test_round_robin_uses_scenes_before_second_rows(self):
        rows = [proposal("a", "1"), proposal("a", "2"), proposal("b", "3")]
        selected = deterministic_scene_round_robin(rows, 2)
        self.assertEqual({row.scene_id for row in selected}, {"a", "b"})

    def test_selection_is_deterministic(self):
        rows = [proposal("a", "1"), proposal("a", "2"), proposal("b", "3")]
        self.assertEqual(
            [row.episode_id for row in deterministic_scene_round_robin(rows, 3)],
            [row.episode_id for row in deterministic_scene_round_robin(list(reversed(rows)), 3)],
        )


if __name__ == "__main__":
    unittest.main()

