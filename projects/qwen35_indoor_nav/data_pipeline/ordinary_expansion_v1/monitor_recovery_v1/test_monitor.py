import copy
import json
from pathlib import Path
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import monitor as m

def fixture():
    return dict(completed=9,total_routes=9661,strict_routes=9,strict_decisions=589,
        lanes=[dict(gpu=g,stage='PRODUCING',completed=3,target=3253 if g!=5 else 3155,
            strict_routes=3,strict_decisions=100,result={}) for g in (3,4,5)],
        old_data_audit={},final_merge={},note='safe')

class Tests(unittest.TestCase):
    def test_01_empty_cache_explicit(self):
        cache=m.SnapshotCache(lambda:fixture(),clock=lambda:100)
        value=cache.snapshot()
        self.assertFalse(value['monitor']['has_snapshot'])
        self.assertTrue(value['monitor']['stale'])
        self.assertNotIn('completed',value)

    def test_02_failure_preserves_snapshot(self):
        collector=mock.Mock(side_effect=[fixture(),ValueError('partial JSON')])
        cache=m.SnapshotCache(collector,clock=lambda:100)
        self.assertTrue(cache.refresh());self.assertFalse(cache.refresh())
        value=cache.snapshot()
        self.assertEqual(value['completed'],9)
        self.assertTrue(value['monitor']['stale'])
        self.assertIn('partial JSON',value['monitor']['source_error'])

    def test_03_recovery_clears_error(self):
        updated=fixture();updated['completed']=20
        collector=mock.Mock(side_effect=[fixture(),OSError('disk'),updated])
        cache=m.SnapshotCache(collector,clock=lambda:100)
        cache.refresh();cache.refresh();cache.refresh()
        self.assertEqual(cache.snapshot()['completed'],20)
        self.assertFalse(cache.snapshot()['monitor']['stale'])

    def test_04_no_silent_nonfinite(self):
        data=fixture();data['completed']=float('nan')
        cache=m.SnapshotCache(lambda:data)
        self.assertFalse(cache.refresh())
        self.assertFalse(cache.snapshot()['monitor']['has_snapshot'])

    def test_05_snapshots_copied_and_age(self):
        now=[100];cache=m.SnapshotCache(fixture,clock=lambda:now[0]);cache.refresh()
        view=cache.snapshot();view['lanes'][0]['completed']=999
        self.assertEqual(cache.snapshot()['lanes'][0]['completed'],3)
        now[0]=130
        self.assertTrue(cache.snapshot()['monitor']['stale'])

    def test_06_server_render_and_safe_embedding(self):
        data=fixture();data['lanes'][0]['result']={'error':'</script><img src=x onerror=evil()>'}
        cache=m.SnapshotCache(lambda:data);cache.refresh();page=m.page(cache.snapshot()).decode()
        self.assertIn('已处理 9 / 9661',page)
        self.assertNotIn('</script><img',page)
        self.assertIn('\\u003c/script\\u003e',page)
        self.assertIn('new URL("api/status", doc.baseURI)',page)
        self.assertNotIn("fetch('/api/status'",page)

    def test_07_http_query_methods_and_error(self):
        cache=m.SnapshotCache(fixture)
        server=m.Server(('127.0.0.1',0),m.handler_for(cache))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_address[1])
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base+'/api/status',timeout=3)
            self.assertEqual(caught.exception.code,503)
            cache.refresh()
            for path in ('/api/status?reload=1','/healthz'):
                with urllib.request.urlopen(base+path,timeout=3) as response:
                    self.assertEqual(response.status,200)
                    self.assertIn('application/json',response.headers['Content-Type'])
                    json.load(response)
            for method in ('POST','PUT','DELETE','PATCH'):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(urllib.request.Request(base+'/api/status',method=method),timeout=3)
                self.assertEqual(caught.exception.code,405)
            with urllib.request.urlopen(base+'/?reload=1',timeout=3) as response:
                self.assertIn('已处理 9 / 9661',response.read().decode())
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)

    def test_08_live_frozen_collector_readonly(self):
        before=m.sha(m.OLD/'monitor.py'),m.sha(m.OLD/'INPUT_LOCK.json')
        value=m.load_collector()()
        self.assertEqual(len(value['lanes']),3)
        self.assertEqual(value['training_status'],'STOPPED')
        self.assertIn(value['benchmark']['status'],('COMPLETE','EVALUATING'))
        self.assertEqual(before,(m.sha(m.OLD/'monitor.py'),m.sha(m.OLD/'INPUT_LOCK.json')))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    with (HERE/'CPU_TESTS.json').open('x') as f:
        json.dump(dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
            time_unix=time.time(),tested_sha256={p.name:m.sha(p) for p in (HERE/'monitor.py',HERE/'client.js',HERE/'index.html',HERE/'test_monitor.py')}),f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
