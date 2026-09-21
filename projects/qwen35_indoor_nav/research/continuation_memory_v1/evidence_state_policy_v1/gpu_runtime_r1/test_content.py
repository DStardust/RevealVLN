"""Regression for the failed real service boundary, without starting a GPU context."""
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import HERE,read
from continuation_service import ContentStore

class ContentBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (HERE/'runs').mkdir(exist_ok=True)

    def test_storage_and_escape(self):
        (HERE/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='cpu_store_') as temp:
            base=Path(temp);store=ContentStore(base/'content',4096)
            array=np.arange(24,dtype=np.uint8).reshape(2,4,3)
            first=store.put_array(array,'rgb');size=store.bytes
            self.assertEqual(first,store.put_array(array,'rgb'));self.assertEqual(size,store.bytes)
            self.assertTrue(np.array_equal(np.load(base/'content'/(first+'.rgb.npy'),allow_pickle=False),array))
            with self.assertRaisesRegex(RuntimeError,'ARTIFACT_LIMIT'):
                ContentStore(base/'limited',1).put_array(array,'rgb')
            with tempfile.TemporaryDirectory() as outside:
                rejected=Path(outside)/'must_not_create'
                with self.assertRaisesRegex(ValueError,'CONTENT_PATH'):ContentStore(rejected,4096)
                self.assertFalse(rejected.exists())
                (base/'link').symlink_to(outside,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'CONTENT_PATH'):ContentStore(base/'link'/'child',4096)
                self.assertFalse((Path(outside)/'child').exists())

    def test_actual_habitat_python_service_handshake(self):
        cfg=read(HERE/'PROTOCOL.json')
        with tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='cpu_service_') as temp:
            run=Path(temp);session=run/'session';session.mkdir()
            (run/'DATA.json').write_text(json.dumps(dict(raw_families=[])))
            (run/'EVALUATION_REGISTRY.json').write_text('{}')
            (session/'CONFIG.json').write_text(json.dumps(dict(run=str(run),artifact_gib=1)))
            sock,child=socket.socketpair();sock.settimeout(20)
            proc=subprocess.Popen([cfg['sim_python'],'-I','-B',str(HERE/'continuation_service.py'),str(child.fileno()),str(session)],
                pass_fds=(child.fileno(),),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            child.close()
            try:
                sock.sendall(b'{"op":"close"}\n')
                self.assertEqual(json.loads(sock.recv(4096)),dict(closed=True))
                stdout,stderr=proc.communicate(timeout=20)
                self.assertEqual(proc.returncode,0,stderr.decode())
                self.assertEqual(read(session/'SERVICE_RESULT.json')['completed'],0)
            finally:
                sock.close()
                if proc.poll() is None:proc.terminate();proc.wait(timeout=10)

if __name__=='__main__':unittest.main()
