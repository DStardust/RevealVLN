import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path

s = importlib.util.spec_from_file_location('adapter', Path(__file__).with_name('server.py'))
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def row(t, n, u):
    return {'unix': t, 'cursor': {'updates': u}, 'metrics': {'mean_ce': 1.},
            'cumulative_compute': {'decisions': n}}


class Tests(unittest.TestCase):
    def test_reserve_jump(self):
        p = m.points_for_segments([('old', [row(1, 10, 1), row(2, 20, 2)]),
                                  ('new', [row(3, 9990, 1), row(4, 10000, 2)])])
        self.assertTrue(p[2]['discontinuity'])
        self.assertIsNone(p[2]['throughput'])
        self.assertEqual(p[3]['throughput'], 10)

    def test_rollback(self):
        p = m.points_for_segments([('run', [row(1, 20, 2), row(2, 10, 1), row(3, 15, 2)])])
        self.assertIsNone(p[1]['throughput'])
        self.assertEqual(p[2]['throughput'], 5)

    def test_scope(self):
        with self.assertRaises(ValueError):
            m.resolve_run({'run_dir': '/tmp/outside'})

    def test_ui(self):
        text = m.html().decode()
        self.assertIn('至少4000', text)
        self.assertNotIn('训练cursor软停止界限', text)

    def test_live_schema(self):
        d = m.collect(include_gpu=False)
        self.assertEqual(d['monitor_version'], 'recovery_v1')
        self.assertIsNone(d['probe']['error'])
        self.assertTrue(d['legacy_history_available'])
        self.assertEqual(d['decisions'], d['progress']['data']['global_plan_decisions'])
        self.assertGreater(d['charged_compute_decisions'], d['decisions'])

    def test_stale_supervisor_not_green(self):
        original = m.v3.original_collect
        def stale(*args, **kwargs):
            d = original(*args, **kwargs)
            d['status']['stale'] = True
            d['status']['data']['status'] = 'TRAINING'
            return d
        with patch.object(m.v3, 'original_collect', side_effect=stale):
            d = m.collect(include_gpu=False)
        self.assertEqual(d['display_state'], 'STALE_NO_CONFIRMED_PROGRESS')
        self.assertIsNone(d['steady_decisions_per_second'])
        self.assertIsNone(d['estimated_segment_remaining_seconds'])


if __name__ == '__main__':
    unittest.main()
