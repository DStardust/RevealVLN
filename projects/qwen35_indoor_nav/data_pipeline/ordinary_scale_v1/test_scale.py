import copy
import importlib.util
from pathlib import Path
import unittest

OUT=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location(name,OUT/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=load('prepare')
a=load('audit')

class ScaleTests(unittest.TestCase):
    def test_split_immutable_pilot_and_disjoint(self):
        houses=[str(i) for i in range(61)]
        x=p.split_houses(houses,houses[:5])
        self.assertEqual([len(x[k]) for k in ('FIT','INTERNAL_DEV','INTERNAL_CONFIRM')],[51,5,5])
        self.assertTrue(set(houses[:5])<=set(x['FIT']))
        self.assertEqual(len(set().union(*map(set,x.values()))),61)
        self.assertEqual(x,p.split_houses(list(reversed(houses)),houses[:5]))

    def test_fair_selection(self):
        jobs=[dict(scene_id=s,physical_source_route_sha256=str(i)) for s in 'abc' for i in range(9)]
        selected=p.fair_select(jobs,6)
        self.assertEqual([j['scene_id'] for j in selected],list('abcabc'))
        self.assertEqual(len(p.fair_select([],500)),0)

    def test_physical_identity_not_instruction(self):
        e=dict(scene_id='mp3d/x/x.glb',start_position=[0,0,0],start_rotation=[0,0,0,1],
               reference_path=[[0,0,0]],goals=[{'position':[0,0,0]}],instruction='one')
        b=dict(e,instruction='two')
        self.assertEqual(p.route_key(e),p.route_key(b))
        b['start_rotation']=[0,1,0,0]
        self.assertNotEqual(p.route_key(e),p.route_key(b))

    def fixture(self):
        frame=dict(position=[0,0,0])
        job=dict(physical_source_route_sha256='x',episode=dict(reference_path=[[0,0,0]],goals=[dict(position=[0,0,0])]))
        checks=[dict(index=i,decision=0,euclidean_error=0,geodesic_error=0) for i in range(2)]
        sup=dict(actions=['STOP'],frames=[frame],physical_source_route_sha256='x',
                 natural_language_full_semantics_certified=False,waypoint_checks=checks)
        diag=dict(actions_attempted=['STOP'],observations=[frame],collision_count=0,maximum_segment_corridor_offset_m=0)
        cert=dict(exact_all_frame_pose_rgb_semantic_match=True,replay_count=2,frames_compared=1)
        return job,sup,diag,cert

    def test_minimal_valid_export(self):
        self.assertEqual(a.check_structure(*self.fixture()),1)

    def test_missing_stop_and_collision_rejected(self):
        for case in ('stop','collision','replay','waypoint','claim'):
            j,s,d,c=copy.deepcopy(self.fixture())
            if case=='stop':s['actions']=['turn_left'];d['actions_attempted']=s['actions']
            if case=='collision':d['collision_count']=1
            if case=='replay':c['exact_all_frame_pose_rgb_semantic_match']=False
            if case=='waypoint':s['waypoint_checks'][0]['geodesic_error']=float('nan')
            if case=='claim':s['natural_language_full_semantics_certified']=True
            with self.subTest(case=case),self.assertRaises(AssertionError):a.check_structure(j,s,d,c)

if __name__=='__main__':unittest.main()
