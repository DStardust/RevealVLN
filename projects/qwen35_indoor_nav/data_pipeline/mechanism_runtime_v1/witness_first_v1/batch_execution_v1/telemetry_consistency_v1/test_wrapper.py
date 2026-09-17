import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import wrapper as w
source=HERE.parents[2]/'compact_loop_v2/run.py'
spec=importlib.util.spec_from_file_location('original_cpu_gpu_guard',source)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


def xml(total,own,external=246,second=246,uuid=original.UUID):
    rows=''.join(f'<process_info><pid>{pid}</pid><type>{kind}</type><used_memory>{m} MiB</used_memory></process_info>' for pid,m,kind in [(1,external,'C'),(2,second,'C'),(99,own,'G')])
    return f'<nvidia_smi_log><gpu><uuid>{uuid}</uuid><fb_memory_usage><used>{total} MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes>{rows}</processes></gpu></nvidia_smi_log>'


class Tests(unittest.TestCase):
    def invoke(self,raws,*,costs=None,recorder=None,deadline=100):
        clock=[0.0];calls=[];events=[];raws=iter(raws);costs=iter(costs or [0]*10)
        def query(timeout):
            calls.append(timeout);clock[0]+=next(costs);return next(raws)
        def record(event):
            events.append(event)
            if recorder:recorder(event,clock)
        self.events=events;self.calls=calls
        return w.sample(query,original.parse_gpu,original.check_gpu,record,99,wall_deadline=deadline,clock=lambda:clock[0])
    def test_consistent_original_value_unchanged(self):
        raw=xml(532,8);result,upper=self.invoke([raw])
        self.assertEqual(result,original.parse_gpu(raw));self.assertEqual(upper,40);self.assertEqual(len(self.calls),1)
    def test_exact_error_one_retry_then_original_pass(self):
        result,upper=self.invoke([xml(600,220),xml(750,220)])
        self.assertEqual(upper,258);self.assertEqual(len(self.calls),2)
        self.assertEqual([e['event'] for e in self.events[:3]],['raw_xml','parsed_sample','guard_error'])
        self.assertIn("MEMORY_ACCOUNTING",self.events[-1]['first_error'])
    def test_three_inconsistent_stops_no_fourth(self):
        with self.assertRaisesRegex(AssertionError,'MEMORY_ACCOUNTING'):self.invoke([xml(600,220)]*4)
        self.assertEqual(len(self.calls),3)
    def test_prior_process_sum_over_limit_cannot_be_washed_by_low(self):
        with self.assertRaisesRegex(AssertionError,'GROSS_MEMORY'):self.invoke([xml(1000,4000),xml(532,8)])
        self.assertEqual(len(self.calls),1)
    def test_4096_sum_boundary_rejected(self):
        with self.assertRaisesRegex(AssertionError,'GROSS_MEMORY'):self.invoke([xml(1000,3604)])
    def test_total_overlimit_immediate(self):
        with self.assertRaisesRegex(AssertionError,'OWN_GPU'):self.invoke([xml(5000,4600),xml(532,8)])
        self.assertEqual(len(self.calls),1)
    def test_external_perpid_not_retryable(self):
        with self.assertRaisesRegex(AssertionError,'EXTERNAL'):self.invoke([xml(2000,300,external=769),xml(532,8)])
        self.assertEqual(len(self.calls),1)
    def test_negative_own_not_retryable(self):
        with self.assertRaisesRegex(AssertionError,'OWN_GPU'):self.invoke([xml(400,8),xml(532,8)])
    def test_wrong_gpu_not_retryable_raw_preserved(self):
        with self.assertRaisesRegex(AssertionError,'GPU_IDENTITY'):self.invoke([xml(532,8,uuid='wrong')])
        self.assertEqual(self.events[0]['event'],'raw_xml');self.assertEqual(self.events[1]['event'],'parse_error')
    def test_malformed_xml_saved_before_parse(self):
        with self.assertRaises(Exception):self.invoke(['<broken'])
        self.assertEqual(self.events[0]['raw_xml'],'<broken');self.assertEqual(len(self.calls),1)
    def test_log_failure_no_retry(self):
        def record(event,clock):raise OSError('disk full')
        with self.assertRaisesRegex(OSError,'disk full'):self.invoke([xml(600,220),xml(750,220)],recorder=record)
        self.assertEqual(len(self.calls),1)
    def test_retry_timeout_reduced_and_late_pass_rejected(self):
        with self.assertRaisesRegex(AssertionError,'MEMORY_ACCOUNTING'):self.invoke([xml(600,220),xml(750,220)],costs=[.2,1.01])
        self.assertLessEqual(self.calls[1],1.0)
    def test_log_time_counts_toward_retry_window(self):
        def record(event,clock):
            if event['event']=='guard_error':clock[0]+=1.01
        with self.assertRaisesRegex(AssertionError,'MEMORY_ACCOUNTING'):self.invoke([xml(600,220),xml(750,220)],recorder=record)
        self.assertEqual(len(self.calls),1)
    def test_original_wall_not_reset(self):
        with self.assertRaisesRegex(AssertionError,'MEMORY_ACCOUNTING'):self.invoke([xml(600,220),xml(750,220)],costs=[.2,.4],deadline=.5)
        self.assertLessEqual(self.calls[1],.3)
    def test_no_original_time_remaining_no_query(self):
        with self.assertRaisesRegex(TimeoutError,'WALL_EXHAUSTED'):self.invoke([xml(532,8)],deadline=0)
        self.assertEqual(self.calls,[])
    def test_duplicate_pid_parse_error_no_retry(self):
        with self.assertRaises(AssertionError):self.invoke([xml(532,8).replace('<pid>2</pid>','<pid>1</pid>')])
        self.assertEqual(len(self.calls),1);self.assertEqual(self.events[-1]['event'],'parse_error')
    def test_unknown_units_parse_error_no_retry(self):
        with self.assertRaises(AssertionError):self.invoke([xml(532,8).replace('532 MiB','532 MB')])
        self.assertEqual(len(self.calls),1)
    def test_external_total_not_retryable(self):
        raw=xml(3000,100,external=750,second=750).replace('</processes>','<process_info><pid>3</pid><type>C</type><used_memory>750 MiB</used_memory></process_info></processes>')
        with self.assertRaisesRegex(AssertionError,'EXTERNAL'):self.invoke([raw])
    def test_durable_log_fsync_failure_propagates(self):
        logger=object.__new__(w.DurableLog);logger.stream=mock.Mock()
        with mock.patch.object(w.os,'fsync',side_effect=OSError('fsync failed')):
            with self.assertRaisesRegex(OSError,'fsync failed'):logger(dict(event='test'))
        logger.stream.write.assert_called_once();logger.stream.flush.assert_called_once()


if __name__=='__main__':unittest.main(verbosity=2)
