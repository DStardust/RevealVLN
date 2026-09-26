"""Resume and ownership checks with CPU files/process identity only."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as p


class PipelineTests(unittest.TestCase):
    def test_unfinished_attempt_duration_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); (run / 'attempts').mkdir()
            p.u.write(run / 'attempts/gpu0_test.start.json',
                dict(pid=2147483647, start_ticks='old', started_unix=10., gpu=0, ids=[1]))
            p.account_interrupted_attempts(run, 70.)
            receipt = p.u.read(run / 'attempts/gpu0_test.json')
            self.assertEqual(receipt['wall_seconds'], 60.)
            self.assertEqual(receipt['duration_kind'], 'UPPER_BOUND_INCLUDES_POSSIBLE_DOWNTIME')
            p.account_interrupted_attempts(run, 90.)
            self.assertEqual(p.u.read(run / 'attempts/gpu0_test.json'), receipt)

    def test_current_process_is_not_treated_as_dead_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); (run / 'attempts').mkdir()
            p.u.write(run / 'attempts/gpu0_live.start.json',
                dict(pid=os.getpid(), start_ticks=p.process_start(os.getpid()), started_unix=1.))
            with self.assertRaisesRegex(RuntimeError, 'STILL_RUNNING'):
                p.account_interrupted_attempts(run, 10.)

    def test_cleanup_rejects_identity_mismatch(self):
        class Process:
            pid = os.getpid()
            def poll(self):
                return None
        self.assertFalse(p.owned(dict(proc=Process(), start_ticks='wrong-start')))

    def test_only_sealed_complete_groups_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); (run / 'SOURCE_LOCK.json').write_text('{}')
            episode = run / 'capture/attempt/episodes/1'
            episode.mkdir(parents=True)
            p.u.write(episode / 'COMPLETE.json', dict(trajectory_id=1))
            self.assertEqual(len(p.records(run, False)), 1)
            self.assertEqual(p.records(run), {})
            p.u.write(run / 'capture/attempt/STATE_SEAL.json',
                dict(base_before='same', base_after='same', complete_ids=[1],
                     source_lock_sha256=p.u.sha(run / 'SOURCE_LOCK.json')))
            self.assertEqual(set(p.records(run)), {1})


if __name__ == '__main__':
    unittest.main()
