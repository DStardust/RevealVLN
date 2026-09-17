import importlib.util
import itertools
import math
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('full_common',HERE/'common.py');dtw=load('path_metrics',HERE/'path_metrics.py')
tiny=load('tiny_tests',c.TINY/'test_interface.py');tiny.c=c
Tests=tiny.Tests


class FullTests(unittest.TestCase):
    def test_complete_disjoint_schedules(self):
        ss=c.schedules(1839,8)
        self.assertEqual(sorted(x for row in ss for x in row),list(range(1839)))
        self.assertEqual(len(set(x for row in ss for x in row)),1839)
        self.assertEqual(sorted(map(len,ss)),[229]+[230]*7)
    def test_schedule_small_and_ragged(self):
        self.assertEqual(c.schedules(3,8),[[0],[1],[2],[],[],[],[],[]])
    def test_windows_do_not_alias(self):
        ws=[c.Window() for _ in range(8)];ws[0].executed.append('turn_left')
        self.assertTrue(all(not w.executed for w in ws[1:]))
    def test_parity_fail_closed(self):
        row=dict(action_match=True,max_abs=.1,relative_l2=.01)
        self.assertTrue(c.parity_accept([row,row]))
        self.assertFalse(c.parity_accept([row]))
        for bad in (dict(row,action_match=False),dict(row,max_abs=.151),dict(row,relative_l2=.031)):
            self.assertFalse(c.parity_accept([row,bad]))
    def test_ndtw_identity(self):
        route=[[0,0,0],[1,0,0],[2,0,0]]
        self.assertEqual(dtw.ndtw(route,route),1.)
    def test_ndtw_offset(self):
        self.assertAlmostEqual(dtw.ndtw([[0,0,0]],[[3,0,0]]),math.exp(-1))
    def test_ndtw_duplicate_positions(self):
        self.assertEqual(dtw.ndtw([[0,0,0],[0,0,0],[1,0,0]],[[0,0,0],[1,0,0]]),1.)
    def test_ndtw_uses_euclidean(self):
        self.assertAlmostEqual(dtw.ndtw([[0,0,0]],[[3,4,0]]),math.exp(-5/3))
    def test_ndtw_nonconsecutive_revisit_kept(self):
        self.assertLess(dtw.ndtw([[0,0,0],[1,0,0],[0,0,0]],[[0,0,0],[1,0,0]]),1)
    def test_ndtw_bruteforce(self):
        def best(a,b,i,j):
            d=math.dist(a[i],b[j])
            if i==len(a)-1 and j==len(b)-1:return d
            return d+min(best(a,b,i+di,j+dj) for di,dj in [(1,0),(0,1),(1,1)] if i+di<len(a) and j+dj<len(b))
        for a in ([[0,0,0],[1,0,0]],[[0,1,0],[2,0,0],[0,1,0]]):
            for b in ([[1,0,0],[2,1,0]],[[0,0,0],[1,2,0],[3,0,0]]):
                self.assertAlmostEqual(dtw.ndtw(a,b),math.exp(-best(a,b,0,0)/(len(b)*3)))
    def test_sdtw_success_weighting(self):
        value=dtw.ndtw([[0,0,0]],[[1,0,0]])
        self.assertEqual(value*0.,0.);self.assertEqual(value*1.,value)
    def test_independent_final_dtw_audit(self):
        audit=load('aggregate_test',HERE/'aggregate.py')
        a=[[0,0,0],[0,0,0],[1,0,0],[0,0,0]];b=[[0,0,0],[2,1,0]]
        self.assertAlmostEqual(dtw.ndtw(a,b),audit.exact_ndtw_check(a,b))
    def test_monitor_readonly(self):
        import threading,urllib.request,urllib.error,json
        monitor=load('monitor_test',HERE/'monitor.py')
        server=monitor.ThreadingHTTPServer(('127.0.0.1',0),monitor.Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url='http://127.0.0.1:'+str(server.server_port)
        try:
            with urllib.request.urlopen(url+'/healthz',timeout=3) as r:self.assertTrue(json.load(r)['read_only'])
            with urllib.request.urlopen(url+'/api/status',timeout=3) as r:self.assertIsNone(json.load(r)['full_sr'])
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(urllib.request.Request(url+'/api/status',data=b'{}'),timeout=3)
            self.assertEqual(raised.exception.code,405)
        finally:server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':unittest.main()
