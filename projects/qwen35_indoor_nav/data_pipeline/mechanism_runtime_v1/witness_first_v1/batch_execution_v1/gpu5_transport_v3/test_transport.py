import copy
import importlib.util
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport as t
import prepare as p

class Tests(unittest.TestCase):
    def document(self):
        d=t.read(t.LINE/'authorizations/WITNESS_BATCH02R1_GPU5_HOLDER_IDENTITY_V1.json')
        d.update(pid=1234567,pane_pid=1234567,starttime_ticks=100)
        return d
    def test_only_pid_binding_changes(self):
        d=self.document();identity=t.bind_identity(d)
        self.assertEqual(identity['pid'],1234567)
        self.assertEqual(t.base.COMMAND,d['cmdline'] and ' '.join(d['cmdline']))
    def test_invalid_identity_rejected(self):
        for patch in [dict(pid=True),dict(pane_pid=2),dict(cwd='/tmp'),dict(pane_id='%999'),dict(proc_uid=1),dict(starttime_ticks=0),dict(cmdline=['python'])]:
            d=self.document();d.update(patch)
            with self.assertRaises(ValueError):t.bind_identity(d)
    def test_sources_exact(self):
        t.verify_sources()
        self.assertEqual(t.base.SOURCE_LOCK[t.BE/'shared.py'],t.sha(t.BE/'shared.py'))
    def test_preparation_binding_locked(self):
        source=p.adapted((t.BE/'prepare.py').read_text())
        compile(source,'cpu','exec')
        self.assertIn("cfg['identity_binding_version']='gpu5_transport_v3'",source)
        self.assertIn('gpu5_transport_v3/transport.py',p.p.LAUNCHER)
    def test_lease_implementation_unmodified(self):
        self.assertEqual(t.base.run_lease.__code__.co_filename,str(t.OLD/'transport.py'))
        self.assertEqual(t.base.restore_dead_holder.__code__.co_filename,str(t.OLD/'transport.py'))

if __name__=='__main__':unittest.main()
