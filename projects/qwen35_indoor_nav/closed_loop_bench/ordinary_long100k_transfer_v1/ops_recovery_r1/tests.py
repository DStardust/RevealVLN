import importlib.util,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('disk_repair',HERE/'disk_monitor.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

class Tests(unittest.TestCase):
    def test_apparent_bytes_links_and_no_symlink_traversal(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'data';root.mkdir();(root/'file').write_bytes(b'x'*123);os.link(root/'file',root/'hard');(root/'external').symlink_to(Path(t)/'missing')
            expected=root.lstat().st_size+(root/'file').lstat().st_size+(root/'external').lstat().st_size
            result=d.scan(root,Path(t)/'result.json',0)
            self.assertEqual(result['bytes'],expected);self.assertEqual(result['files'],2);self.assertEqual(result['status'],'COMPLETE')
    def test_partial_inventory_is_marked_partial(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'data';root.mkdir();(root/'file').write_bytes(b'x');records=[];original=d.write
            try:
                d.write=lambda path,value:records.append(dict(value));d.scan(root,Path(t)/'result',0)
            finally:d.write=original
            self.assertTrue(any(r['status']=='RUNNING' for r in records));self.assertEqual(records[-1]['status'],'COMPLETE')
    def test_prior_and_current_gpu_time_not_double_charged(self):
        rows=[dict(gpu=True,done=True,charged_hours=2,start=0),dict(gpu=True,done=False,start=3600),dict(gpu=None,done=False,start=0)]
        self.assertEqual(d.charged_gpu_hours(9,rows,7200),12)
    def test_poll_does_not_wait_for_scanning_worker(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);m=d.Monitor(root,None,sys.executable);m.folder=root/'scan';m.folder.mkdir();m.last_start=time.monotonic()
            proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],stdin=subprocess.DEVNULL)
            try:
                m.proc=proc;began=time.monotonic();value=m.poll();self.assertLess(time.monotonic()-began,1);self.assertIsNone(value['last_complete']);self.assertIsNone(proc.poll())
            finally:proc.terminate();proc.wait(timeout=5)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    (HERE/'CPU_TEST_RESULT.json').write_text(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),scope='Async inventory semantics, no symlink traversal, accurate resumed GPU hours and nonblocking health polling'),indent=2)+'\n')
    raise SystemExit(not r.wasSuccessful())
