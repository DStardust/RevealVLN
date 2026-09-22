"""CPU contracts and an actual old physical-family reread; never generated training data."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from quality import validate_trace,isolated_position,counts,audit_family

class Tests(unittest.TestCase):
    def trace(self):
        return dict(actions=['L','R','S'],observations=[dict(evidence_complete=True)]*3,
                    complete=True,collisions=0,interior_state_assignments=0)
    def test_stop_has_no_new_frame(self):
        x=self.trace();validate_trace(x);x['observations'].append(dict(evidence_complete=True))
        with self.assertRaisesRegex(ValueError,'STOP_MUST'):validate_trace(x)
    def test_500_includes_stop(self):
        x=self.trace();x['actions']=['L']*499+['S'];x['observations']=[dict(evidence_complete=True)]*500;validate_trace(x)
        x['actions'].insert(0,'L')
        with self.assertRaisesRegex(ValueError,'STOP_OR_BUDGET'):validate_trace(x)
    def test_unknown_is_not_negative(self):
        x=self.trace();x['observations']=[dict(evidence_complete=False)]*3
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):validate_trace(x)
    def test_no_collision_or_teleport(self):
        for field in ('collisions','interior_state_assignments'):
            x=self.trace();x[field]=1
            with self.assertRaises(ValueError):validate_trace(x)
    def test_real_stop_required(self):
        x=self.trace();x['actions']=['L','R','L']
        with self.assertRaisesRegex(ValueError,'STOP_OR_BUDGET'):validate_trace(x)
    def test_spatial_parent_dedup(self):
        self.assertFalse(isolated_position([0,0,0],[[.5,0,0]],1))
        self.assertTrue(isolated_position([0,0,0],[[1,0,0]],1))
    def test_variants_do_not_inflate_parents(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);write(r/'PROTOCOL.json',dict(houses=['FIT_ONLY'],families_per_house=64))
            for folder,id_ in [('position_000','P'),('absent_00','A')]:
                p=r/'collect/FIT_ONLY'/folder;p.mkdir(parents=True)
                write(p/'FAMILY.json',dict(family_id=id_,parent_family_id='P',training_admission='CONTROLLED_SEE2_ARRAY_CERTIFIED_FIT'))
            x,_=counts(r);self.assertEqual(x['new_parents'],1);self.assertEqual(x['new_variants'],2);self.assertEqual(x['certified_physical_cross_executions'],24)
    def test_frozen_existing_real_arrays(self):
        collector=load('scale_test_collect',HERE/'collect.py')
        old=read(CPU/'DATA.json')
        f=next(f for f in old['raw_families'] if f['split']=='FIT' and f.get('stratum')=='terminal_present')
        result=audit_family(f,collector.present.certify,{})
        self.assertEqual(result['trace_count'],12);self.assertEqual(result['crossed_labels'],24)
    def test_holdout_exclusion(self):
        manifest=read(HERE/'HOUSE_MANIFEST.json')
        self.assertFalse({h['house'] for h in manifest['houses']}&set(manifest['excluded']))
        self.assertTrue(all(h['split']=='FIT' for h in manifest['houses']))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        success=result.wasSuccessful(),new_physical_data_generated=0,scope='CPU invariants plus read-only recertification of one existing family'))
    raise SystemExit(0 if result.wasSuccessful() else 1)

