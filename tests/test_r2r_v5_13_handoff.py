import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import watch_r2r_v5_13_1_handoff as handoff


class AutomaticHandoffTest(unittest.TestCase):
    def test_current_process_is_alive(self) -> None:
        self.assertTrue(handoff.process_alive(os.getpid()))
        self.assertFalse(handoff.process_alive(None))

    def test_val_seen_failure_blocks_unseen(self) -> None:
        states = []
        with (
            mock.patch.object(handoff, "wait_for_training", return_value={}),
            mock.patch.object(
                handoff, "launch_or_attach", return_value={"status": "FAIL"}
            ) as launch,
            mock.patch.object(
                handoff, "write_state",
                side_effect=lambda stage, **values: states.append(stage),
            ),
        ):
            with self.assertRaisesRegex(
                handoff.HandoffError, "val_seen paired result"
            ):
                handoff.run("0", 5)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.args[0], "val_seen")
        self.assertEqual(states[-1], "BLOCKED_VAL_SEEN_SCIENTIFIC_GATE")

    def test_val_seen_pass_launches_unseen(self) -> None:
        states = []
        results = iter((
            {"status": "PASS"},
            {"status": "PASS", "paper_result": True},
        ))
        with (
            mock.patch.object(handoff, "wait_for_training", return_value={}),
            mock.patch.object(
                handoff, "launch_or_attach",
                side_effect=lambda *unused: next(results),
            ) as launch,
            mock.patch.object(
                handoff, "write_state",
                side_effect=lambda stage, **values: states.append(stage),
            ),
        ):
            self.assertEqual(handoff.run("0", 5), 0)
        self.assertEqual(
            [row.args[0] for row in launch.call_args_list],
            ["val_seen", "val_unseen"],
        )
        self.assertEqual(states, ["VAL_SEEN_GATE_PASS", "COMPLETE"])


if __name__ == "__main__":
    unittest.main()
