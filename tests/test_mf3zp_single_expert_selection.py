from __future__ import annotations

from collections import Counter
import copy
import unittest

from revealnav_mf3.single_expert_dec_scout import (
    select_retest_events,
    select_scout_events,
)


def _population() -> list[dict[str, object]]:
    rows = []
    for domain in ("R2R", "RxR"):
        for index in range(150):
            rows.append({
                "dataset": domain,
                "scene_id": f"scene-{index % 39:02d}",
                "event_id": f"{domain}-{index:03d}",
                "delta_utility": index * 1000,
                "qwen_factor_label": "D" if index % 2 else "U",
            })
    return rows


class SingleExpertSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_balanced_and_label_blind(self) -> None:
        rows = _population()
        baseline = select_scout_events(rows)
        mutated = copy.deepcopy(rows)
        for row in mutated:
            row["delta_utility"] = -float(row["delta_utility"])
            row["qwen_factor_label"] = "A"
        second = select_scout_events(mutated)
        ids = [row["event_id"] for row in baseline]
        self.assertEqual(ids, [row["event_id"] for row in second])
        self.assertEqual(Counter(row["dataset"] for row in baseline), {"R2R": 40, "RxR": 40})
        self.assertEqual(len(set(ids)), 80)

    def test_retest_is_preselected_ten_per_domain(self) -> None:
        selected = select_scout_events(_population())
        first = select_retest_events(selected)
        second = select_retest_events(list(reversed(selected)))
        self.assertEqual([row["event_id"] for row in first], [row["event_id"] for row in second])
        self.assertEqual(Counter(row["dataset"] for row in first), {"R2R": 10, "RxR": 10})
        self.assertEqual(len(first), 20)


if __name__ == "__main__":
    unittest.main()
