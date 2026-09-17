"""CPU tests: threshold, reset, caps, STOP preservation and invalid causal input."""
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location('causal_guard_under_test', HERE / 'guard.py')
g = importlib.util.module_from_spec(s); s.loader.exec_module(g)
RGB = bytes(224 * 224 * 3)
CHANGED = bytes([1]) + RGB[1:]


class GuardTests(unittest.TestCase):
    def ready(self, count=8):
        x = g.VisualStallGuard(); x.observe_rgb(RGB)
        for _ in range(count): x.observe_rgb(RGB, 'move_forward')
        return x

    def test_no_early_intervention(self):
        for n in range(8):
            self.assertFalse(self.ready(n).choose('move_forward')['intervened'])

    def test_exact_threshold_and_action(self):
        r = self.ready().choose('move_forward')
        self.assertEqual((r['action'], r['interventions_after']), ('turn_left', 1))

    def test_stop_never_overridden(self):
        self.assertEqual(self.ready().choose('STOP')['action'], 'STOP')

    def test_turns_never_overridden(self):
        for action in ('turn_left', 'turn_right'):
            self.assertFalse(self.ready().choose(action)['intervened'])

    def test_one_byte_change_resets(self):
        x = self.ready(); x.observe_rgb(CHANGED, 'move_forward')
        self.assertFalse(x.choose('move_forward')['intervened'])

    def test_executed_turn_resets_even_identical_rgb(self):
        for action in ('turn_left', 'turn_right'):
            x = self.ready(); x.observe_rgb(RGB, action)
            self.assertEqual(x.stagnant_forwards, 0)

    def test_four_interventions_max(self):
        x = self.ready(0)
        for i in range(9):
            for _ in range(8): x.observe_rgb(RGB, 'move_forward')
            r = x.choose('move_forward')
            self.assertEqual(r['intervened'], i < 4)
            x.observe_rgb(RGB, r['action'])
        self.assertEqual(x.interventions, 4)

    def test_new_episode_reset(self):
        x = self.ready(); x.choose('move_forward'); x.observe_rgb(RGB)
        self.assertEqual((x.stagnant_forwards, x.interventions), (0, 0))

    def test_lane_isolation(self):
        a, b = self.ready(), self.ready(0)
        a.choose('move_forward')
        self.assertEqual((b.stagnant_forwards, b.interventions), (0, 0))

    def test_no_pose_or_privileged_payload(self):
        for data in ({'rgb': RGB, 'position': [0, 0, 0]}, 'frame.png', RGB[:-1]):
            with self.assertRaises(ValueError): g.VisualStallGuard().observe_rgb(data)

    def test_requires_reset(self):
        x = g.VisualStallGuard()
        with self.assertRaises(ValueError): x.choose('move_forward')
        with self.assertRaises(ValueError): x.observe_rgb(RGB, 'move_forward')

    def test_no_fake_executed_stop_or_unknown_action(self):
        for action in ('STOP', 'collision', 'turn_180'):
            with self.assertRaises(ValueError): self.ready().observe_rgb(RGB, action)
        with self.assertRaises(ValueError): self.ready().choose('teleport')


if __name__ == '__main__': unittest.main()
