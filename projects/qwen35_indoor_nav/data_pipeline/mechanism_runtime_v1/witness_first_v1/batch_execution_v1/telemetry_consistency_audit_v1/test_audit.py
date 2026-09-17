import copy
import importlib.util
import json
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=load('tested_raw_audit',HERE/'audit.py')
w=load('tested_raw_wrapper',HERE.parent/'telemetry_consistency_v1/wrapper.py')
original=load('tested_raw_original',HERE.parents[2]/'compact_loop_v2/run.py')
original.UUID='GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b'

def xml(total=750,own=220,external=246):
    rows=''.join('<process_info><pid>'+str(pid)+'</pid><type>'+kind+'</type><used_memory>'+str(mib)+' MiB</used_memory></process_info>'
                for pid,kind,mib in [(1,'C',external),(2,'C',246),(99,'G',own)])
    return '<nvidia_smi_log><gpu><uuid>'+original.UUID+'</uuid><fb_memory_usage><used>'+str(total)+' MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes>'+rows+'</processes></gpu></nvidia_smi_log>'

def fixture(cycles=None):
    clock=[100.];events=[];resources=[]
    for raws in cycles or [[xml()],[xml(total=600),xml()]]:
        iterator=iter(raws)
        def query(timeout):clock[0]+=.01;return next(iterator)
        def record(event):events.append(copy.deepcopy(event));clock[0]+=.001
        snapshot,upper=w.sample(query,original.parse_gpu,original.check_gpu,record,99,wall_deadline=3995.,clock=lambda:clock[0])
        resources.append(dict(snapshot,own_memory_upper_mib=upper,elapsed=clock[0]-95.))
        clock[0]+=5.
    return a.normalized(events),a.normalized(resources)

class Tests(unittest.TestCase):
    def audit(self,events,resources):return a.audit_samples(events,resources,original.parse_gpu,original.check_gpu,w.retry_eligible,99)
    def test_two_calls_one_retry_exact_xml_resource_binding(self):
        result=self.audit(*fixture())
        self.assertEqual(result['raw_queries_verified'],3);self.assertEqual(result['resource_passes_bound'],2)
        self.assertEqual(result['retained_failed_accounting_samples'],1)
        self.assertFalse(result['durable_ack_timestamps_directly_recorded'])
    def test_pid_json_strings_reparsed_from_xml(self):
        events,resources=fixture([[xml()]])
        self.assertIn('99',resources[0]['processes']);self.assertTrue(self.audit(events,resources)['raw_protocol_pass'])
    def test_raw_xml_sha_tamper(self):
        events,resources=fixture();events[0]['raw_xml']+=' '
        with self.assertRaisesRegex(ValueError,'RAW_XML_HASH'):self.audit(events,resources)
    def test_parsed_snapshot_tamper(self):
        events,resources=fixture();events[1]['snapshot']['memory_mib']+=1
        with self.assertRaisesRegex(ValueError,'PARSED_VALUE_CHANGED'):self.audit(events,resources)
    def test_resource_value_tamper(self):
        events,resources=fixture();resources[0]['memory_mib']+=1
        with self.assertRaisesRegex(ValueError,'RESOURCE_RAW_VALUE_MISMATCH'):self.audit(events,resources)
    def test_resource_missing_pass(self):
        events,resources=fixture()
        with self.assertRaisesRegex(ValueError,'RESOURCE_PASS_COUNT_MISMATCH'):self.audit(events,resources[:-1])
    def test_resource_reorder(self):
        events,resources=fixture()
        with self.assertRaises(ValueError):self.audit(events,list(reversed(resources)))
    def test_prior_guard_failure_cannot_be_deleted(self):
        events,resources=fixture();events=[e for e in events if e['event']!='guard_error']
        with self.assertRaises(ValueError):self.audit(events,resources)
    def test_prior_unsafe_high_sum_not_washed_by_later_low(self):
        events,resources=fixture([[xml(total=600),xml()]])
        raw=xml(total=1000,own=3700)
        events[0]['raw_xml']=raw;events[0]['utf8_sha256']=a.hashlib.sha256(raw.encode()).hexdigest()
        events[1]['snapshot']=a.normalized(original.parse_gpu(raw))
        with self.assertRaisesRegex(AssertionError,'INCONSISTENT_SAMPLE_GROSS_MEMORY_NOT_SAFE'):self.audit(events,resources)
    def test_true_external_limit_not_retryable(self):
        events,resources=fixture([[xml(total=600),xml()]])
        raw=xml(total=1500,external=769)
        events[0]['raw_xml']=raw;events[0]['utf8_sha256']=a.hashlib.sha256(raw.encode()).hexdigest()
        events[1]['snapshot']=a.normalized(original.parse_gpu(raw))
        with self.assertRaisesRegex(ValueError,'NONRETRYABLE'):self.audit(events,resources)
    def test_retry_window_reset(self):
        events,resources=fixture([[xml(total=600),xml()]])
        next(e for e in events if e['event']=='guard_error')['retry_deadline']+=1.
        with self.assertRaisesRegex(ValueError,'RETRY_WINDOW_RESET'):self.audit(events,resources)
    def test_retry_timeout_over_remaining(self):
        events,resources=fixture([[xml(total=600),xml()]])
        next(e for e in events if e['event']=='raw_xml' and e['attempt']==1)['timeout']=2.
        with self.assertRaisesRegex(ValueError,'RETRY_DEADLINE'):self.audit(events,resources)
    def test_retry_attempt_three_rejected(self):
        events,resources=fixture([[xml(total=600),xml()]])
        for event in events:
            if event['attempt']==1:event['attempt']=3
        with self.assertRaises(ValueError):self.audit(events,resources)
    def test_clock_rollback_rejected(self):
        events,resources=fixture();events[-1]['monotonic']=0.
        with self.assertRaisesRegex(ValueError,'EVENT_CLOCK_ORDER'):self.audit(events,resources)
    def test_resource_wall_boundary_rejected(self):
        events,resources=fixture();resources[-1]['elapsed']=3900.
        with self.assertRaisesRegex(ValueError,'WALL_LIMIT'):self.audit(events,resources)
    def test_pass_then_stop_not_accepted(self):
        events,resources=fixture();events.append(dict(event='sampling_stopped',attempt=1,monotonic=events[-1]['monotonic']))
        with self.assertRaises(ValueError):self.audit(events,resources)
    def test_duplicate_json_keys(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON_KEY'):a.lines(b'{"x":1,"x":2}\n')
    def test_partial_log(self):
        with self.assertRaisesRegex(ValueError,'PARTIAL_LOG'):a.lines(b'{}')
    def test_nonfinite_log(self):
        with self.assertRaisesRegex(ValueError,'NONFINITE_JSON'):a.lines(b'{"x":NaN}\n')
    def test_two_retries_allowed_but_every_failed_raw_preserved(self):
        events,resources=fixture([[xml(total=600),xml(total=600),xml()]])
        result=self.audit(events,resources)
        self.assertEqual(result['max_extra_queries_per_call'],2)
        self.assertEqual(result['retained_failed_accounting_samples'],2)

if __name__=='__main__':unittest.main()
