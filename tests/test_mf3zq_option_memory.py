import unittest

from revealnav_mf3.oracle_option_memory import OracleOptionMemory
from revealnav_mf3.oracle_revealskill_schema import OracleOption


class OptionMemoryTest(unittest.TestCase):
    def _option(self, ident, rank, returnable=False, expiry=None):
        return OracleOption(ident, ident, "cp", rank, 0, 0, (), ("c",), returnable=returnable, expiry_step=expiry)

    def test_budget_is_fixed_and_deterministic(self):
        memory = OracleOptionMemory()
        for index in range(9):
            memory.observe(self._option(f"o{index}", index))
        self.assertEqual(memory.max_usage(), 8)
        self.assertNotIn("o8", {item.option_id for item in memory.options()})

    def test_order_prefers_returnable_earliest_expiry(self):
        memory = OracleOptionMemory()
        for index in range(8):
            memory.observe(self._option(f"o{index}", index))
        memory.observe(self._option("new", 99, returnable=True, expiry=1))
        self.assertIn("new", {item.option_id for item in memory.options()})


if __name__ == "__main__":
    unittest.main()
