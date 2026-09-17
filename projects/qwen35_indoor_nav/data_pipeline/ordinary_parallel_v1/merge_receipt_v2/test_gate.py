import copy
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import gate
import merge


class Tests(unittest.TestCase):
    def fixture(self):
        old=dict(pid=1,starttime_ticks=10,proc_uid=0,cwd=str(gate.ROOT),cmdline=['python','holder'],gpu_uuid='GPU-x')
        sleeper=dict(pid=2,starttime_ticks=20,proc_uid=0,cwd=str(gate.ROOT),cmdline=[''])
        result=dict(error=None,restoration=dict(restored=False,error="AssertionError('SLEEPER_IDENTITY_CHANGED')"))
        before=dict(identity=old);active=dict(sleeper=sleeper)
        pre=dict(gpu=6,holder_identity=old,frozen_empty_argv=sleeper,independently_verified_sleeper=dict(sleeper,cmdline=['sleep','24000']))
        new=dict(pid=3,starttime_ticks=30,proc_uid=0,cwd=str(gate.ROOT),cmdline=['python','holder'])
        recovery=dict(gpu=6,old_receipts_unchanged=True,old_runtime_result_overridden=False,external_processes_stopped=0,
            restoration=dict(restored=True,new_identity=new,gpu=dict(gpu=6,uuid='GPU-x',processes={'3':29000}),recovery_pane_preserved_if_failed=False))
        return result,before,active,pre,recovery
    def test_explicit_external_receipt_old_false_preserved(self):
        values=self.fixture();result=gate.validate_evidence(6,*values)
        self.assertFalse(values[0]['restoration']['restored'])
        self.assertFalse(result['original_restored']);self.assertTrue(result['recovery_restored'])
    def test_no_override_or_changed_old_receipt(self):
        for key,value in [('old_receipts_unchanged',False),('old_runtime_result_overridden',True),('external_processes_stopped',1)]:
            values=list(self.fixture());values[-1][key]=value
            with self.assertRaises(AssertionError):gate.validate_evidence(6,*values)
    def test_wrong_new_identity(self):
        for key,value in [('cwd','/tmp'),('cmdline',['unknown']),('proc_uid',22),('starttime_ticks',15)]:
            values=list(self.fixture());values[-1]['restoration']['new_identity'][key]=value
            with self.assertRaises(AssertionError):gate.validate_evidence(6,*values)
    def test_memory_and_uuid_wrong(self):
        values=list(self.fixture());values[-1]['restoration']['gpu']['processes']={'3':1}
        with self.assertRaises(AssertionError):gate.validate_evidence(6,*values)
        values=list(self.fixture());values[-1]['restoration']['gpu']['uuid']='GPU-other'
        with self.assertRaises(AssertionError):gate.validate_evidence(6,*values)
    def test_foreign_recovery_or_failure(self):
        values=list(self.fixture());values[-1]['restoration']['restored']=False
        with self.assertRaises(AssertionError):gate.validate_evidence(6,*values)
        with self.assertRaises(AssertionError):gate.validate_evidence(4,*self.fixture())
    def test_strict_merge_transport_only(self):
        source=merge.transported_source();compile(source,'CPU_merge_transport','exec')
        self.assertIn("strict.audit_route(root,jobmap[ident])",source)
        self.assertIn("assert fresh==stored,'STRICT_INDEX_REAUDIT_MISMATCH'",source)
        self.assertIn("restoration_receipts[str(gpu)+':'+attempt.name]=gate.verify(attempt,result,gpu)",source)
        self.assertIn('physical_duplicates=0,alias_duplicates=0,complete_alias_ownership=True',source)


if __name__=='__main__':unittest.main()
