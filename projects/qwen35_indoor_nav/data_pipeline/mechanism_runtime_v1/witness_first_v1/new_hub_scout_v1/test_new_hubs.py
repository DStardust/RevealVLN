"""CPU fixtures: no physical reachability or navigation claims."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('newhub_selector_test',HERE/'selector.py');a=load('newhub_adapter_test',HERE/'adapters.py')
def geom(pos,count=4):return {'position':pos,'status':'GEOMETRY_CHECKED_NOT_VISIBILITY_CERTIFIED','snapped':pos,'reachable_groups':count}
class Tests(unittest.TestCase):
    def test_four_unique_full3d(self):
        pos=[[i*2.,0.,0.] for i in range(6)];selected,ledger=m.choose(pos,list(map(geom,pos)),[[0.,0.,0.]])
        self.assertEqual(len(selected),4);self.assertTrue(all(r['position'][0]>=2 for r in selected));self.assertEqual(len(ledger),6)
    def test_1m_boundary(self):
        pos=[[1.,0.,0.],[.999,0.,0.]];selected,ledger=m.choose(pos,list(map(geom,pos)),[[0.,0.,0.]])
        self.assertEqual([r['position'] for r in selected],[[1.,0.,0.]])
    def test_three_dimensions_preserved(self):
        pos=[[0.,1.,0.],[0.,0.,0.]];selected,_=m.choose(pos,list(map(geom,pos)),[[0.,0.,0.]])
        self.assertEqual(selected[0]['position'],[0.,1.,0.])
    def test_duplicate_source_ledger(self):
        pos=[[2.,0.,0.],[2.,0.,0.]];selected,ledger=m.choose(pos,[geom(pos[0])],[])
        self.assertEqual(len(selected),1);self.assertEqual(ledger[1]['status'],'DUPLICATE_SOURCE_POSITION')
    def test_unreachable_and_unknown_retained(self):
        pos=[[i*2.,0.,0.] for i in range(3)];g=[geom(pos[0],0),geom(pos[1])];g[1]['snapped']=[2.1,0.,0.]
        selected,ledger=m.choose(pos,g,[]);self.assertFalse(selected)
        self.assertEqual({r['status'] for r in ledger},{'NO_PRIOR_REACHABLE_ROLE_GROUP','PRIOR_COORDINATE_NOT_EXACT_NAVMESH','PRIOR_REACHABILITY_UNTESTED'})
    def test_selected_mutual_distance(self):
        pos=[[2.,0.,0.],[2.1,0.,0.],[4.,0.,0.]];selected,ledger=m.choose(pos,list(map(geom,pos)),[])
        self.assertEqual(len(selected),2);self.assertIn('WITHIN_1M_NEW_SELECTED_HUB',[r['status'] for r in ledger])
    def test_deterministic_tie(self):
        pos=[[i*2.,0.,0.] for i in range(8)]
        self.assertEqual([x['position'] for x in m.choose(pos,list(map(geom,pos)),[])[0]],
                         [x['position'] for x in m.choose(list(reversed(pos)),list(map(geom,pos)),[])[0]])
    def test_runtime_overlap_refused(self):
        pos=[[i*2.,0.,0.] for i in range(4)];cfg={'candidates':[{'house_id':'h'}],'new_hub_plan':{'house_id':'h','selected_positions':pos,'excluded_positions':[[0,0,0]]}}
        with self.assertRaises(AssertionError):m.runtime_select(None,None,[],pos,None,cfg,None)
    def test_runtime_live_geometry_called(self):
        pos=[[i*2.,0.,0.] for i in range(4)];cfg={'candidates':[{'house_id':'h'}],'new_hub_plan':{'house_id':'h','selected_positions':pos,'excluded_positions':[]}}
        calls=[]
        def original(*args):calls.append(args);return [{'position':p,'yaw_bin':0} for p in pos],{}
        hubs,report=m.runtime_select(original,None,[],pos,None,cfg,None)
        self.assertEqual(len(calls),1);self.assertEqual(len(hubs),4);self.assertTrue(report['all_four_runtime_geometry_retained'])
    def test_common_only_hub_selection_algorithm_change(self):
        source=a.common_source();body=source.split('\n_new_hub_selector=')[0]
        body=body.replace(repr(str(HERE)),repr(str(a.a.shard_root(0))))
        body=body.replace('def _original_select_four_hubs(','def select_hubs(').replace('if len(selected)==4:break','if len(selected)==2:break')
        self.assertEqual(body,a.a.common_source(0))
    def test_worker_exact_transport_only(self):
        source=a.worker_source().replace(repr(str(HERE)),repr(str(a.a.shard_root(0))))
        source=source.replace('from new_hub_scoped_common_v1 import (','from bulk_scout_scoped_common_00 import (')
        source=source.replace("HabitatBackend(candidate['scene_glb'],2,candidate['roles']","HabitatBackend(candidate['scene_glb'],1,candidate['roles']")
        self.assertEqual(source,a.a.worker_source(0))
    def test_original_physical_resource_thresholds_retained(self):
        common=a.common_source();worker=a.worker_source();supervisor=a.supervisor_source()
        for token in ('reachable[:12]','limit=2','0<=t[\'distance\']<=8'):self.assertIn(token,common)
        for token in ('max_actions=140','len(actions)>504','6*1024**3'):self.assertIn(token,worker)
        for token in ("sample['elapsed'] < 3000","sample['disk_bytes'] < 7*1024**3",'upper < 4096','sum(external) <= 2048'):self.assertIn(token,supervisor)
    def test_wrong_limit_invalid_position(self):
        with self.assertRaises(ValueError):m.choose([],[],[],5)
        with self.assertRaises(ValueError):m.point([0,0])

if __name__=='__main__':unittest.main()
