"""CPU-only bounded-read and chart semantics tests; never start an HTTP server."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('chart_monitor', HERE/'server.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class ChartsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cpu_', dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def row(self, t, d, ce=1.0, speed=2.0):
        return {'unix':t, 'cursor':{'decisions':d, 'updates':d//4, 'epoch':0},
                'metrics':{'last_chunk':{'ce':ce}}, 'throughput':speed}

    def test_tail_is_byte_and_point_bounded(self):
        path = self.root/'history.jsonl'
        path.write_bytes(b''.join(json.dumps(self.row(i,i*4)).encode()+b'\n' for i in range(50)))
        result = s.tail_records(path,self.root,max_bytes=900,max_points=3)
        self.assertTrue(result['tail_truncated'])
        self.assertTrue(result['points_truncated'])
        self.assertEqual(len(result['records']),3)
        self.assertEqual(result['records'][-1]['cursor']['decisions'],196)

    def test_bad_and_partial_log_lines_are_not_invented(self):
        path = self.root/'history.jsonl'
        path.write_bytes(json.dumps(self.row(1,4)).encode()+b'\nBAD\n{"unfinished"')
        result = s.tail_records(path,self.root)
        self.assertEqual(len(result['records']),1)
        self.assertEqual(result['malformed_lines'],1)
        self.assertTrue(result['partial_last_line'])

    def test_real_cursor_coordinates_missing_and_reset(self):
        rows = [self.row(100,80,2),self.row(120,100,4),self.row(140,120,None),
                self.row(160,60,8),{'metrics':[]}, {'unix':200,'metrics':{}}]
        points = s.chart_points(rows)
        self.assertEqual([p['decisions'] for p in points],[80,100,120,60])
        self.assertEqual(points[1]['rolling_ce'],3)
        self.assertIsNone(points[2]['last_chunk_ce'])
        self.assertIsNone(points[2]['rolling_ce'])
        self.assertEqual(points[3]['rolling_ce'],8)
        self.assertTrue(points[3]['discontinuity'])

    def test_sliding_mean_uses_last_twenty_logged_values(self):
        points = s.chart_points([self.row(i,i*4,float(i)) for i in range(1,26)])
        self.assertEqual(points[-1]['rolling_ce'],sum(range(6,26))/20)

    def test_confusion_counts_recall_and_unobserved_class(self):
        matrix = [[3,1,0,0],[0,0,0,0],[1,0,2,0],[1,0,0,1]]
        a = s.action_summary({'confusion':matrix})
        self.assertEqual(a['targets'],[4,0,3,2])
        self.assertEqual(a['predictions'],[5,1,2,1])
        self.assertEqual(a['recall'],[.75,None,2/3,.5])
        self.assertIsNone(s.action_summary({})['confusion'])

    def test_json_limits_redaction_staleness(self):
        path=self.write('STATUS.json',{'status':'RUNNING','api_key':'sensitive','metrics':{'loss':float('nan'),'auth_token':'sensitive','forward_tokens':17}})
        os.utime(path,(100,100))
        out=s.read_json(path,self.root,now=300)
        self.assertTrue(out['stale'])
        self.assertNotIn('sensitive',json.dumps(out,allow_nan=False))
        self.assertEqual(out['data']['metrics']['forward_tokens'],17)
        with mock.patch.object(s,'MAX_JSON',20):
            self.assertEqual(s.read_json(path,self.root)['error'],'ValueError')

    def test_path_escape_missing_and_corruption(self):
        path=self.root/'STATUS.json';path.symlink_to(HERE/'server.py')
        self.assertEqual(s.read_json(path,self.root)['error'],'ValueError')
        self.assertEqual(s.tail_records(path,self.root)['error'],'ValueError')
        self.assertEqual(s.read_json(self.root/'missing',self.root)['error'],'MISSING')
        bad=self.root/'bad';bad.write_text('{')
        self.assertEqual(s.read_json(bad,self.root)['error'],'JSONDecodeError')

    def test_collect_true_limits_eta_and_no_fake_history_point(self):
        record=self.row(100,40,1,2);record.update(budget={'max_decisions':100,'wall_seconds':1000,'max_updates':50},latest_checkpoint='/a/checkpoint.pt',cumulative_compute={'decisions':44})
        record['metrics']['last_chunk']['elapsed_seconds']=390
        self.write('exec/run_1/PROGRESS.json',record)
        self.write('snapshot.json',{'counts':{'instruction_conditioned_decisions':200}})
        out=s.collect(self.root/'exec',self.root/'snapshot.json',self.root/'queues',self.root,False)
        self.assertEqual(out['decisions'],40)
        self.assertEqual(out['budget_eta_seconds'],28)
        self.assertEqual(out['wall_remaining_seconds'],10)
        self.assertEqual(out['estimated_segment_remaining_seconds'],10)
        self.assertEqual(out['updates_remaining'],40)
        self.assertEqual(out['epoch_eta_seconds'],80)
        self.assertEqual(out['checkpoint_name'],'checkpoint.pt')
        self.assertEqual(out['points'],[])
        self.assertEqual(out['metric_scope'],'ONLINE_TRAINING_NOT_DEV_OR_CLOSED_LOOP')

    def test_queues_new_plan_and_failure_visible(self):
        self.write('queues/manual_failure_recovery_gpu1_20260910_v1/PLAN.json',{'jobs':[{'gpu':1,'id':'500'}]})
        out=s.queue_table(self.root/'queues',self.root)
        self.assertEqual(out[0]['state'],'NOT_LAUNCHED')
        self.write('queues/manual_failure_recovery_gpu1_20260910_v1/lane_gpu_1/RESULT.json',{'error':'test failure','states':{'500':'FAILED'}})
        out=s.queue_table(self.root/'queues',self.root)
        self.assertEqual(out[0]['state'],'STOPPED_ERROR')
        self.assertEqual(out[0]['error'],'test failure')

    def test_gpu_numeric_only_and_timeout(self):
        with mock.patch.object(s.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='2, 200, 300, 50\nbad\n')) as run:
            out=s.gpu_status()
        self.assertEqual(out['devices'][0]['used_mib'],200)
        self.assertEqual(run.call_args.kwargs['timeout'],2)
        with mock.patch.object(s.subprocess,'run',side_effect=subprocess.TimeoutExpired('nvidia-smi',2)):
            self.assertEqual(s.gpu_status()['error'],'TimeoutExpired')

    def test_queue_exact_relative_identity(self):
        plan=s.QUEUES/'special_holder_gpu3_v1/PLAN.json'
        argv=['python3','-I','-S',str((s.QUEUES/'queue.py').relative_to(s.ROOT)),
              '--plan',str(plan.relative_to(s.ROOT)),'--gpu','3']
        self.assertTrue(s.matches_queue(argv,s.ROOT,plan,3))
        self.assertFalse(s.matches_queue(argv,s.ROOT,plan,4))
        self.assertFalse(s.matches_queue(argv,s.ROOT,s.QUEUES/'other/PLAN.json',3))
        self.assertFalse(s.matches_queue(argv,s.HERE,plan,3))

    def test_routes_and_write_methods_never_browse(self):
        cls=s.make_handler()
        for path in ('/../STATUS.json','/server.py','/api/status?path=anything','/%2e%2e'):
            h=object.__new__(cls);h.path=path;h.respond=mock.Mock();h.do_GET()
            self.assertEqual(h.respond.call_args.args[0],404)
        for method in ('do_POST','do_PUT','do_DELETE','do_PATCH'):
            h=object.__new__(cls);h.respond=mock.Mock();getattr(h,method)()
            self.assertEqual(h.respond.call_args.args[0],405)
        for path in ('/','/healthz','/api/status'):
            h=object.__new__(cls);h.path=path;h.respond=mock.Mock()
            with mock.patch.object(s,'collect',return_value={'read_only':True}):h.do_GET()
            self.assertEqual(h.respond.call_args.args[0],200)
        self.assertEqual((s.HOST,s.PORT),('127.0.0.1',18766))

    def test_static_view_has_no_external_assets_or_html_injection(self):
        html=(HERE/'index.html').read_text()
        self.assertNotIn('https://',html)
        self.assertNotIn('innerHTML',html)
        self.assertIn('textContent',html)
        for name in ('loss','throughput','matrix','queues','health'):
            self.assertIn('id="'+name+'"',html)


if __name__=='__main__':unittest.main(verbosity=2)
