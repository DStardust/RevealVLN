"""Regression: slow NAS sampling cannot abort producers; complete data stays immutable."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from disk_watch_r2 import DiskWatch
from resume_r2 import quarantine_array_temporaries,preserve_complete_metadata
runtime=load('test_scale_runtime_r2',HERE/'pipeline_r2.py')

class Tests(unittest.TestCase):
    def test_disk_timeout_nonfatal_nonblocking(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)
            write(run/'DISK.json',dict(bytes=123,unix=10))
            watch=DiskWatch(run,runtime.monitor,write,append,1000,interval=60,timeout=.04,
                command=[sys.executable,'-I','-c','import time; time.sleep(60)'])
            try:
                began=time.monotonic();watch.tick()
                self.assertLess(time.monotonic()-began,1)
                self.assertIsNone(watch.proc.poll())
                time.sleep(.06);watch.tick()
                value=read(run/'DISK.json')
                self.assertEqual(value['status'],'SCAN_TIMEOUT')
                self.assertEqual(value['bytes'],123);self.assertTrue(value['stale'])
                self.assertIsNone(watch.proc)
            finally:watch.stop()
    def test_real_exceeded_limit_still_stops(self):
        with tempfile.TemporaryDirectory() as td:
            watch=DiskWatch(Path(td),runtime.monitor,write,append,99,interval=60,
                command=[sys.executable,'-I','-c','print("100 /own/path")'])
            try:
                watch.tick();watch.proc.wait(timeout=5)
                with self.assertRaisesRegex(RuntimeError,'ARTIFACT_LIMIT'):watch.tick()
            finally:watch.stop()
    def test_nonzero_scan_does_not_fabricate_zero(self):
        with tempfile.TemporaryDirectory() as td:
            watch=DiskWatch(Path(td),runtime.monitor,write,append,99,interval=60,
                command=[sys.executable,'-I','-c','raise SystemExit(1)'])
            try:
                watch.tick();watch.proc.wait(timeout=5);watch.tick()
                value=read(Path(td)/'DISK.json');self.assertIsNone(value['bytes']);self.assertEqual(value['status'],'SCAN_FAILED')
            finally:watch.stop()
    def test_only_temporary_arrays_preserved(self):
        (HERE/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='cpu_r2_') as td:
            run=Path(td);p=run/'collect/H/content';p.mkdir(parents=True)
            name='a'*64+'.rgb.npy';(p/name).write_bytes(b'complete')
            (p/(name+'.tmp')).write_bytes(b'interrupted');(p/'user.tmp').write_bytes(b'not ours')
            rows=quarantine_array_temporaries(run,'H')
            self.assertEqual(len(rows),1);self.assertEqual((p/name).read_bytes(),b'complete')
            self.assertEqual((run/rows[0]['retained_as']).read_bytes(),b'interrupted')
            self.assertEqual((p/'user.tmp').read_bytes(),b'not ours')
            self.assertEqual(quarantine_array_temporaries(run,'H'),[])
    def test_quarantine_rejects_outside_run(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'RUN_BOUNDARY'):quarantine_array_temporaries(Path(td),'H')
    def test_completed_metadata_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='cpu_r2_') as td:
            run=Path(td);p=run/'collect/H/position_001';p.mkdir(parents=True)
            write(p/'AUDIT.json',dict(verified=True))
            write(p/'FAMILY.json',dict(audit=dict(path=str((p/'AUDIT.json').relative_to(LINE)),sha256=sha(p/'AUDIT.json'))))
            self.assertEqual(preserve_complete_metadata(run),1);self.assertEqual(preserve_complete_metadata(run),1)
            write(p/'FAMILY.json',dict(audit=dict(path=str((p/'AUDIT.json').relative_to(LINE)),sha256=sha(p/'AUDIT.json')),changed=True))
            with self.assertRaisesRegex(ValueError,'COMPLETE_FAMILY_CHANGED'):preserve_complete_metadata(run)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(HERE/'CPU_RUNTIME_R2_RESULT.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        new_simulator_runs=0,new_training_updates=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)

