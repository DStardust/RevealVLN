"""CPU-only accounting and evaluation selection regression checks."""
import unittest
import prepare


class PreparationTests(unittest.TestCase):
    def row(self, route, steps, house='h'):
        return dict(scene_group=house, physical_source_route_sha256=route, decisions=steps)

    def test_aliases_do_not_inflate_physical_decisions(self):
        counts = prepare.count_rows([self.row('a', 10), self.row('a', 10), self.row('b', 4)])
        self.assertEqual(counts, dict(strict_routes=2, instruction_records=3,
                                     unique_route_decisions=14, instruction_conditioned_decisions=24))

    def test_conflicting_route_lengths_fail(self):
        with self.assertRaisesRegex(ValueError, 'PHYSICAL_ROUTE_DECISION_CONFLICT'):
            prepare.count_rows([self.row('a', 10), self.row('a', 11)])

    def test_changed_receipt_count_fails(self):
        with self.assertRaisesRegex(ValueError, 'RECEIPT_COUNT_MISMATCH'):
            prepare.require_counts({'strict_routes': 3}, {'strict_routes': 4})

    def test_dev_is_whole_house_source_metadata_only(self):
        rows = []
        for house, split in [('train', 'FIT'), ('dev', 'INTERNAL_DEV'), ('confirm', 'INTERNAL_CONFIRM')]:
            for source in ('R2R', 'RxR'):
                for i in range(12):
                    for alias in range(2):
                        rows.append(dict(scene_id=house, source=source, split=split,
                                         physical_source_route_sha256=f'{house}-{source}-{i}',
                                         episode_id=str(2*i+alias), runtime_certified=False))
        class Reader:
            def read(self, *_):
                return list(reversed(rows))
        selected = prepare.dev_sources(Reader(), {'INTERNAL_DEV': ['dev']})
        self.assertEqual(len(selected), 20)
        self.assertEqual({r['scene_id'] for r in selected}, {'dev'})
        self.assertEqual(len({r['physical_source_route_sha256'] for r in selected}), 20)
        self.assertTrue(all(r['usage'] == 'closed_loop_evaluation_only_not_training' for r in selected))
        self.assertFalse(any(r['simulation_evaluated'] for r in selected))

    def test_dev_shortage_is_not_filled_from_training(self):
        rows = [dict(scene_id='dev', source='RxR', split='INTERNAL_DEV',
                     physical_source_route_sha256='only', episode_id='1')]
        rows += [dict(scene_id='train', source='R2R', split='FIT',
                      physical_source_route_sha256=str(i), episode_id=str(i)) for i in range(50)]
        class Reader:
            def read(self, *_):
                return rows
        selected = prepare.dev_sources(Reader(), {'INTERNAL_DEV': ['dev']})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['physical_source_route_sha256'], 'only')


if __name__ == '__main__':
    unittest.main()
