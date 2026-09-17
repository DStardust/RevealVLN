import importlib.util
from pathlib import Path
import unittest

def module():
    spec=importlib.util.spec_from_file_location('recovery_under_test',Path(__file__).parent/'prepare_recovery.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class RecoveryTests(unittest.TestCase):
    def test_exact_resource_retry(self):
        m=module();r=dict(error="AssertionError('EXTERNAL_RESOURCE_LOAD')",cleanup_complete=True,external_processes_stopped=0)
        self.assertEqual(m.classify(225,'FAILED_NO_AUTOMATIC_RETRY',r),'EXPLICIT_ONCE_MANUAL_RETRY_RESOURCE_FAILURE')
    def test_unclean_or_foreign_signals_rejected(self):
        for override in ({'cleanup_complete':False},{'external_processes_stopped':1}):
            r=dict(error="AssertionError('EXTERNAL_RESOURCE_LOAD')",cleanup_complete=True,external_processes_stopped=0);r.update(override)
            with self.assertRaises(ValueError):module().classify(225,'FAILED_NO_AUTOMATIC_RETRY',r)
    def test_scientific_failure_not_retried(self):
        with self.assertRaises(ValueError):module().classify(225,'FAILED_NO_AUTOMATIC_RETRY',dict(error='SEMANTIC_FAILURE',cleanup_complete=True))
    def test_preworker_only(self):
        r=dict(status='LAUNCH_FAILED',exception=dict(type='AssertionError',message='EXTERNAL_RESOURCE_LOAD'),external_processes_stopped_by_launcher=0,process_record_present=False,supervisor_result_present=False)
        module().classify(244,'FAILED_NO_AUTOMATIC_RETRY',r)
        r['process_record_present']=True
        with self.assertRaises(ValueError):module().classify(244,'FAILED_NO_AUTOMATIC_RETRY',r)
    def test_real_preworker_receipt(self):
        m=module();r=m.read(m.BE/'batch_244/run_v1/LAUNCH_RESULT.json')
        self.assertEqual(m.classify(244,'FAILED_NO_AUTOMATIC_RETRY',r),'EXPLICIT_ONCE_MANUAL_RETRY_RESOURCE_FAILURE')
    def test_pending_only(self):
        self.assertEqual(module().classify(226,'PENDING',{}),'UNATTEMPTED_RESERVATION_MIGRATION')
        for state in ('RUNNING','AUDITED_SEE_PER_ITEM_QUALITY','FAILED_NO_AUTOMATIC_RETRY'):
            with self.assertRaises(ValueError):module().classify(226,state,{})
    def test_exact_targets_and_scope(self):
        m=module();self.assertEqual([a for a,b in m.PAIRS],[225,244,226,245,227,246,228,247])
        self.assertEqual(m.LINE.name,'qwen35_indoor_nav');self.assertEqual(m.ROOT.name,'vla')

if __name__=='__main__':unittest.main()
