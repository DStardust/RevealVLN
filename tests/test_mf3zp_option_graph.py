import unittest

from revealnav_mf3.option_graph import OptionGraph, OptionNode, OptionStatus


def node():
    return OptionNode("o1", "cp1", "B1", 1, 1, ("c1",), (), ("c1",), (), "return:cp1", OptionStatus.ACTIVE)


class OptionGraphTest(unittest.TestCase):
    def test_preserve_and_commit_require_readiness(self):
        graph = OptionGraph([node()])
        graph.preserve("o1")
        self.assertEqual(graph.get("o1").status, OptionStatus.PRESERVED)
        with self.assertRaises(ValueError):
            graph.commit("o1", readiness="A")
        graph.commit("o1", readiness="D")
        self.assertEqual(graph.get("o1").status, OptionStatus.COMMITTED)

    def test_identity_conflict(self):
        graph = OptionGraph([node()])
        with self.assertRaises(ValueError):
            graph.add(OptionNode("o1", "cp2", "B2", 1, 1, ("c1",), (), ("c1",), (), "r", "ACTIVE"))


if __name__ == "__main__":
    unittest.main()
