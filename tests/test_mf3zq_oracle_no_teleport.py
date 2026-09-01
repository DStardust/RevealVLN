import unittest

from revealnav_mf3.oracle_revealskill_policy import execute_skill_with_frozen_controller
from revealnav_mf3.oracle_revealskill_schema import OracleSkill


class OracleNoTeleportTest(unittest.TestCase):
    def test_only_injected_executor_is_called(self):
        calls = []
        result = execute_skill_with_frozen_controller(
            type("Decision", (), {"skill": OracleSkill.FOLLOW, "option_id": None})(),
            frozen_executor=lambda skill, option: calls.append((skill, option)) or "ok",
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, [(OracleSkill.FOLLOW, None)])

    def test_non_callable_executor_rejected(self):
        with self.assertRaises(TypeError):
            execute_skill_with_frozen_controller(None, frozen_executor=None)


if __name__ == "__main__":
    unittest.main()
