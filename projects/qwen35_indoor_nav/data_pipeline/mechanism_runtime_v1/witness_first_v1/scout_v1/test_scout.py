import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common
run=common.load('exact_scout_supervisor',HERE/'run.py')

class Budget:
    def __init__(self,cutoff=None):self.used=0;self.cutoff=cutoff
    def check_time(self):
        if self.cutoff is not None and self.used>=self.cutoff:raise common.BudgetExceeded('synthetic cutoff')
    def reserve_action(self):self.check_time();self.used+=1
    def snapshot(self):return dict(total_reserved_actions=self.used)

class GeometryBackend:
    def __init__(self):self.eligible={'old':[99]};self.objects={}
    def snap_position(self,p):return list(p) if p[0]>=0 else None

class FakeBackend:
    def __init__(self):
        self.position=[0,0,0]
        self.np=SimpleNamespace(asarray=lambda p,dtype:list(p),float32=float)
        class ShortestPath:pass
        class Follower:
            def __init__(self,*args,**kwargs):pass
            def next_action_along(self,target):return 'move_forward'
        def find_path(path):path.geodesic_distance=1 if self.position[2]==0 else 0;return True
        self.sim=SimpleNamespace(pathfinder=SimpleNamespace(find_path=find_path),get_agent=lambda i:object())
        self.hs=SimpleNamespace(ShortestPath=ShortestPath,GreedyGeodesicFollower=Follower,
            errors=SimpleNamespace(GreedyFollowerError=RuntimeError))
    def _pose(self):
        p=dict(position=list(self.position),rotation=[1.,0.,0.,0.])
        return dict(p,sensors={k:copy.deepcopy(p) for k in ('rgb','semantic')})
    def reset(self,p,yaw,seed):self.position=list(p)
    def step(self,action):
        if action=='F':self.position=[0,0,-1]
        return False
    def observe(self):return dict(pose=self._pose(),pixels={'1':300},evidence_complete=True,rgb_hash='a'*64,semantic_hash='b'*64)

class ScoutTests(unittest.TestCase):
    def test_probe_does_not_modify_permission_roles(self):
        backend=GeometryBackend();proxy=common.ProbeBackend(backend,[1,2])
        self.assertEqual(proxy.eligible,{'probe':[1,2]});self.assertEqual(backend.eligible,{'old':[99]})
    def test_hubs_diverse_and_groups_capped(self):
        roles=[dict(signature=['chair','living room',str(i)],eligible_ids=[i+1]) for i in range(20)]
        def targets(proxy,p,role,limit):return [dict(distance=1+proxy.eligible['probe'][0]/100,target=p,center=p)]
        hubs,diag=common.select_hubs(GeometryBackend(),roles,[[0,0,0],[.2,0,0],[2,0,0],[-1,0,0]],Budget(),targets)
        self.assertEqual(len(hubs),2);self.assertEqual([h['position'][0] for h in hubs],[0,2])
        self.assertTrue(all(len(h['groups'])==12 for h in hubs))
        self.assertFalse(diag['physical_or_visibility_certified'])
    def test_hub_selection_deterministic(self):
        role=[dict(signature=['chair','living room','chair'],eligible_ids=[1])]
        fn=lambda b,p,r,limit:[dict(distance=1,target=p,center=p)]
        args=(GeometryBackend(),role)
        self.assertEqual(common.select_hubs(*args,[[0,0,0],[2,0,0]],Budget(),fn),
                         common.select_hubs(*args,[[2,0,0],[0,0,0]],Budget(),fn))
    def test_compact_inverse_matches_sealed_math(self):
        for actions in ('F','FLFR','LFFR','FRRFFLLF','LLLLFFFFRR'):
            expected=list(actions)+common.factory.inverse(list(actions))[1]
            self.assertEqual(common.compact_actions(list(actions)),expected)
    def test_compact_rejects_stop(self):
        with self.assertRaises(ValueError):common.compact_actions(['S'])
    def runner(self,budget,events):
        compiler=common.Compiler({'x':('chair','living room')},{'t':dict(anchor='x',terminal='x',instruction='fixture')},{'x':[1]})
        return common.PartialTraceRunner(FakeBackend(),compiler,budget,lambda k,v:events.append((k,v)))
    def test_feedback_with_partial_base_normal_interface(self):
        events=[];runner=self.runner(Budget(),events)
        trace=common.ScoutFeedbackRunner(runner).navigate([0,0,0],0,[0,0,-1],[0,0,-2])
        self.assertTrue(common.compiler.complete(trace));self.assertEqual(trace['actions'],['F','L','R'])
    def test_feedback_cutoff_preserves_diagnostic_trace(self):
        events=[];runner=self.runner(Budget(cutoff=1),events)
        with self.assertRaises(common.BudgetExceeded):
            common.ScoutFeedbackRunner(runner).navigate([0,0,0],0,[0,0,-1],[0,0,-2])
        traces=[v for k,v in events if k=='trace']
        self.assertEqual(len(traces),1);self.assertFalse(traces[0]['complete'])
        self.assertEqual(traces[0]['actions'],['F'])
        self.assertFalse(common.compiler.complete(traces[0]))
        self.assertTrue(any(k=='feedback_exception' for k,v in events))
    def test_store_scope_is_exact_new_root(self):
        store=common.store_class()
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            with store(Path(folder)/'content',1024**2) as instance:pass
            self.assertTrue(instance.final_audit['audit_pass'])
    def test_supervisor_adaptation_limits_and_environment(self):
        source=run.adapted_source();compile(source,'adapted_supervisor','exec')
        self.assertIn("sample['elapsed'] < 4500",source)
        self.assertIn("sample['disk_bytes'] < 8*1024**3",source)
        self.assertIn("'nvidia-smi','-i','1'",source)
        self.assertEqual(common.LINE.name,'qwen35_indoor_nav')
        self.assertTrue((common.LINE/'.envs/q35n_habitat_v017_g0r/bin/python3').is_file())
    def test_effective_config_requires_approval_and_enables_execution(self):
        prepared={'runtime_allowed':False,'training_allowed':False,'executable':False}
        data={'INPUT_LOCK.json':'{}','PREPARED_CONFIG.json':__import__('json').dumps(prepared),
            'MAIN_AGENT_APPROVAL.json':__import__('json').dumps(dict(approved=True,gpu=1,input_lock_sha256='fixture_hash'))}
        with mock.patch.object(common,'sha',return_value='fixture_hash'),mock.patch.object(Path,'read_text',lambda path:data[path.name]):
            effective=common.runtime_config()
        self.assertTrue(effective['runtime_allowed'])
        self.assertTrue(effective['executable'])
        self.assertFalse(effective['training_allowed'])
        self.assertFalse(prepared['runtime_allowed'])
        self.assertFalse(prepared['executable'])
    def test_unapproved_config_never_becomes_executable(self):
        data={'INPUT_LOCK.json':'{}','MAIN_AGENT_APPROVAL.json':'{"approved":false,"gpu":1,"input_lock_sha256":"fixture_hash"}'}
        with mock.patch.object(common,'sha',return_value='fixture_hash'),mock.patch.object(Path,'read_text',lambda path:data[path.name]):
            with self.assertRaises(AssertionError):common.runtime_config()

if __name__=='__main__':unittest.main()
