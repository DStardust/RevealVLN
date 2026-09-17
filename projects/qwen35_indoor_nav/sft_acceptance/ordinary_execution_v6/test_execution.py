import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

class Tests(unittest.TestCase):
    def test_v5_shared_partition(self):
        m=load('supervise');b=dict(external_per_process_mib=4608,external_gpu_mib=5120,max_gpu_memory_bytes=26*1024**3)
        s=dict(total_mib=32607,used_mib=30250,processes=[dict(pid=1,mib=25878),dict(pid=2,mib=3850),dict(pid=3,mib=492)])
        m.guard(s,1,b)
        s['processes'][1]['mib']=4609
        with self.assertRaisesRegex(AssertionError,'EXTERNAL_PROCESS'):m.guard(s,1,b)
    def test_v5_partition_cannot_overbook_device(self):
        m=load('supervise');b=dict(external_per_process_mib=4608,external_gpu_mib=5120,max_gpu_memory_bytes=28*1024**3)
        with self.assertRaisesRegex(AssertionError,'PARTITIONS_EXCEED'):m.guard(dict(total_mib=32607),None,b)
    def test_partial_route_cursor(self):
        m=load('execute'); c=dict(epoch=0,route_position=2,step=0,updates=0)
        self.assertEqual(m.next_cursor(c,4,10,7)['step'],4)
        self.assertEqual(c['step'],0)
    def test_route_boundary(self):
        m=load('execute'); c=dict(epoch=0,route_position=2,step=8)
        self.assertEqual(m.next_cursor(c,10,10,7),dict(epoch=0,route_position=3,step=0))
    def test_epoch_boundary(self):
        m=load('execute'); c=dict(epoch=0,route_position=6,step=8)
        self.assertEqual(m.next_cursor(c,10,10,7),dict(epoch=1,route_position=0,step=0))
    def test_external_guard(self):
        m=load('supervise');b=dict(external_per_process_mib=768,external_gpu_mib=2048,max_gpu_memory_bytes=28*1024**3)
        s=dict(total_mib=32768,used_mib=522,processes=[dict(pid=1,mib=246),dict(pid=2,mib=246)])
        m.guard(s,None,b)
        s['processes'].append(dict(pid=3,mib=1380))
        with self.assertRaisesRegex(AssertionError,'EXTERNAL_PROCESS'):m.guard(s,None,b)
    def test_owned_memory(self):
        m=load('supervise');b=dict(external_per_process_mib=768,external_gpu_mib=2048,max_gpu_memory_bytes=28*1024**3)
        s=dict(total_mib=32768,used_mib=31000,processes=[dict(pid=1,mib=30000)])
        with self.assertRaisesRegex(AssertionError,'OWN_GPU'):m.guard(s,1,b)
    def test_graphics_included(self):
        m=load('supervise')
        s=m.parse_gpu('<nvidia_smi_log><gpu><uuid>GPU-X</uuid><fb_memory_usage><total>32768 MiB</total><used>500 MiB</used></fb_memory_usage><processes><process_info><pid>2</pid><used_memory>400 MiB</used_memory><type>G</type></process_info></processes></gpu></nvidia_smi_log>','GPU-X')
        self.assertEqual(s['processes'][0]['type'],'G')
    def test_unknown_memory_rejected(self):
        m=load('supervise')
        with self.assertRaises(AssertionError):m.parse_gpu('<nvidia_smi_log><gpu><uuid>GPU-X</uuid><fb_memory_usage><total>32768 MiB</total><used>N/A</used></fb_memory_usage></gpu></nvidia_smi_log>','GPU-X')

if __name__=='__main__':unittest.main()
