import unittest

from revealnav_mf3.mf3zv_protocol import eligible_domains, final_status


class Mf3zvDomainGateTest(unittest.TestCase):
    def rows(self, dataset, count, scenes=10):
        return [
            {"dataset": dataset, "episode_id": str(i), "scene_id": f"s{i % scenes}"}
            for i in range(count)
        ]

    def test_single_domain_is_predeclared(self):
        domains = eligible_domains(self.rows("RxR", 30))
        self.assertEqual(domains, ["RxR"])
        self.assertEqual(final_status(domains), "MF3ZV_PROGRESS_SUPPORT_PASS_RXR_ONLY")

    def test_threshold_does_not_relax(self):
        self.assertEqual(eligible_domains(self.rows("R2R", 29)), [])


if __name__ == "__main__":
    unittest.main()

