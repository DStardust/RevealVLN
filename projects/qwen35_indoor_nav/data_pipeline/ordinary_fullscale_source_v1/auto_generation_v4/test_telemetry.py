import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import telemetry

def guard(snapshot,own=None,restorable=False):
    external={pid:mib for pid,mib in snapshot['processes'].items() if pid!=own}
    assert all(0<=m<=768 for m in external.values()) and sum(external.values())<=2048,'EXTERNAL_GPU_RESOURCE'
    assert sum(snapshot['processes'].values())<=snapshot['memory_mib'],'GPU_MEMORY_ACCOUNTING'
    upper=snapshot['memory_mib']-sum(external.values())
    if own is not None:assert 0<=upper<4096,'OWN_GPU_UPPER_BOUND'
    if restorable:assert snapshot['memory_mib']<1024,'RESTORE_NOT_SAFE_EXTERNAL_LOAD'
    return upper

class Tests(unittest.TestCase):
    def check(self,total,processes,own=3):
        records=[];snap=dict(memory_mib=total,processes=processes);before=copy.deepcopy(snap)
        result=telemetry.active_assessment(snap,own,guard,records.append)
        self.assertEqual(snap,before);return result,records
    def test_actual_failed_snapshot(self):
        old=HERE.parent/'auto_generation_v3/lanes/gpu_6/attempt_000'
        row=json.loads((old/'GPU_SNAPSHOTS.jsonl').read_text().splitlines()[-1])
        row['processes']={int(k):v for k,v in row['processes'].items()}
        self.assertEqual(row['memory_mib'],733);self.assertEqual(sum(row['processes'].values()),737)
        own=json.loads((old/'PROCESS_2.json').read_text())['pid']
        self.assertIn('contexts',(old/'SUPERVISOR_EXCEPTION.json').read_text())
        with self.assertRaisesRegex(AssertionError,'GPU_MEMORY_ACCOUNTING'):guard(row,own)
        result,records=self.check(row['memory_mib'],row['processes'],own)
        self.assertEqual(result['conservative_upper_mib'],245)
        self.assertTrue(result['amended_acceptance']);self.assertFalse(result['original_guard_passed'])
        self.assertEqual([r['status'] for r in records],['PENDING','ACCEPTED'])
    def test_consistent_preserves_original_upper(self):
        r,_=self.check(900,{1:246,2:246,3:245});self.assertEqual(r['guard_upper_mib'],408)
        self.assertTrue(r['original_guard_passed']);self.assertFalse(r['amended_acceptance'])
    def test_inconsistent_total_cap(self):
        for total,processes in ((4095,{1:100,3:4000}),(4100,{1:200,3:4000}),(4000,{1:246,3:3850})):
            with self.assertRaisesRegex(AssertionError,'INCONSISTENT_TOTAL_GPU_CAP'):self.check(total,processes)
    def test_consistent_own_overage(self):
        with self.assertRaisesRegex(AssertionError,'OWN_GPU_UPPER_BOUND'):self.check(4596,{1:500,3:4096})
    def test_external_limits_not_relaxed(self):
        for p in ({1:769,3:1},{1:700,2:700,4:700,3:10}):
            with self.assertRaisesRegex(AssertionError,'EXTERNAL_GPU_RESOURCE'):self.check(500,p)
    def test_negative_nan_inf_and_boolean(self):
        for v in (-1,float('nan'),float('inf'),True):
            with self.assertRaisesRegex(AssertionError,'INVALID_GPU_MEMORY_VALUE'):self.check(v,{3:1})
            with self.assertRaisesRegex(AssertionError,'INVALID_GPU_MEMORY_VALUE'):self.check(500,{3:v})
    def test_no_own_not_available(self):
        with self.assertRaisesRegex(AssertionError,'ACTIVE_OWN_PID_REQUIRED'):self.check(733,{1:246,2:246,3:245},None)
    def test_absent_own_not_amended(self):
        with self.assertRaisesRegex(AssertionError,'GPU_MEMORY_ACCOUNTING'):self.check(733,{1:246,2:246,4:245},3)
    def test_unknown_original_exception_rejected(self):
        def broken(*args):raise OSError('unknown')
        with self.assertRaises(OSError):telemetry.active_assessment(dict(memory_mib=1,processes={3:2}),3,broken,lambda x:None)
    def test_unknown_assertion_rejected(self):
        def broken(*args):raise AssertionError('UNKNOWN')
        with self.assertRaisesRegex(AssertionError,'UNKNOWN'):telemetry.active_assessment(dict(memory_mib=1,processes={3:2}),3,broken,lambda x:None)
    def test_log_failure_stops(self):
        def failed(*args):raise OSError('disk')
        with self.assertRaises(OSError):telemetry.active_assessment(dict(memory_mib=733,processes={3:737}),3,guard,failed)
    def test_restore_and_idle_original_still_reject(self):
        row=dict(memory_mib=733,processes={1:246,2:246,3:245})
        for kwargs in ({},{'restorable':True}):
            with self.assertRaisesRegex(AssertionError,'GPU_MEMORY_ACCOUNTING'):guard(row,**kwargs)
if __name__=='__main__':unittest.main(verbosity=2)
