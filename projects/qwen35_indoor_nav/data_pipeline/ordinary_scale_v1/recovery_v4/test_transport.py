import ast
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
worker=load('gpu5_worker',HERE/'worker.py')
run=load('gpu5_launcher',HERE/'run.py')

class TransportTests(unittest.TestCase):
    def test_exactly_one_real_source_change(self):
        original=(HERE.parent/'worker.py').read_text()
        tree,lines=worker.transport(original)
        self.assertEqual(len(lines),1)
        expected=ast.parse(original.replace('cfg.gpu_device_id=2','cfg.gpu_device_id=5'))
        self.assertEqual(ast.dump(tree),ast.dump(expected))
    def test_missing_assignment_rejected(self):
        with self.assertRaises(AssertionError):worker.transport('pass')
    def test_multiple_assignment_rejected(self):
        with self.assertRaises(AssertionError):worker.transport('cfg.gpu_device_id=2\ncfg.gpu_device_id=2')
    def test_other_device_rejected(self):
        with self.assertRaises(AssertionError):worker.transport('cfg.gpu_device_id=7')
    def test_xml_includes_graphics_process(self):
        xml='<nvidia_smi_log><gpu><uuid>'+run.UUID+'</uuid><fb_memory_usage><used>1024 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes><process_info><pid>1</pid><type>G</type><used_memory>900 MiB</used_memory></process_info></processes></gpu></nvidia_smi_log>'
        result=run.parse_gpu(xml)
        self.assertEqual(result['processes'],{1:900})
        self.assertEqual(result['process_rows'][0]['type'],'G')
    def test_unattributed_gpu_memory_is_conservative(self):
        with self.assertRaises(AssertionError):
            run.validate_contexts(dict(memory_mib=5000,processes={1:100,2:500}),1)
    def test_small_new_external_context_allowed(self):
        self.assertEqual(run.validate_contexts(dict(memory_mib=1300,processes={1:700,2:500}),1),{2:500})

if __name__=='__main__':unittest.main()
