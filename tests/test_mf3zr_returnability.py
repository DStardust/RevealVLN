import unittest

from revealnav_mf3.frozen_returnability import (
    RETURN_HORIZON,
    ReturnabilityStatus,
    unavailable_adapter,
    reject_snapshot_as_skill,
)


class Mf3zrReturnabilityTest(unittest.TestCase):
    def test_unavailable_callback_is_explicit(self):
        result = unavailable_adapter().audit(
            event_id="e", option_id="o", from_step=0, anchor_checkpoint_id="a",
            state={"step": 0}, option={"option_id": "o"},
        )
        self.assertEqual(result.status, ReturnabilityStatus.EXECUTION_UNAVAILABLE)
        self.assertFalse(result.attempted)
        self.assertEqual(RETURN_HORIZON, 8)

    def test_snapshot_cannot_count_as_return(self):
        with self.assertRaises(ValueError):
            reject_snapshot_as_skill({"used_as_return": True})


if __name__ == "__main__":
    unittest.main()
