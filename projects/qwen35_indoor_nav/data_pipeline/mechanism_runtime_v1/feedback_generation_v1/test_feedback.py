import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from feedback import FeedbackRunner

class Backend:
    def __init__(self,collision=False,mutation=False):
        self.collision=collision;self.mutation=mutation;self.steps=0
        self.np=NS(asarray=lambda v,**kw:list(v),float32=float)
        class Path:pass
        def find(path):
            path.geodesic_distance=math.dist(path.requested_start,path.requested_end);return True
        class Follower:
            def next_action_along(_,target):
                if self.mutation:self.p[0]+=1
                return 'move_forward'
        self.sim=NS(pathfinder=NS(find_path=find),get_agent=lambda _:object())
        self.hs=NS(ShortestPath=Path,GreedyGeodesicFollower=lambda *args,**kw:Follower(),
                   errors=NS(GreedyFollowerError=RuntimeError))
    def reset(self,p,y,seed):
        self.p=list(p);self.y=y
    def _pose(self):
        return dict(position=self.p[:],rotation=[math.cos(self.y*math.pi/24),0,math.sin(self.y*math.pi/24),0])
    def step(self,a):
        self.steps+=1
        if a=='F':self.p[2]-=.25
        else:self.y+=(1 if a=='L' else -1)
        return self.collision

class Budget:
    def __init__(self,backend,limit=100):
        self.backend=backend;self.n=0;self.limit=limit
    def check_time(self):pass
    def reserve_action(self):
        if self.n>=self.limit:raise RuntimeError('BUDGET')
        assert self.n==self.backend.steps
        self.n+=1

def make(collision=False,mutation=False,limit=100):
    b=Backend(collision,mutation);budget=Budget(b,limit);events=[]
    runner=NS(backend=b,budget=budget,compiler=NS(atoms=lambda obs:[{} for _ in obs]),
              observe=lambda t:dict(step=t,pose=copy.deepcopy(b._pose())),
              emit=lambda k,v:events.append((k,copy.deepcopy(v))))
    return FeedbackRunner(runner),budget,b,events

class Tests(unittest.TestCase):
    def test_real_actions_recorded_and_budgeted(self):
        r,credit,b,ev=make()
        tr=r.navigate([0,0,0],0,[0,0,-.5],[0,0,-1])
        self.assertTrue(tr['complete'])
        self.assertEqual(tr['actions'],['F','L','R'])
        self.assertEqual(len(tr['observations']),4)
        self.assertEqual(credit.n,b.steps)
        self.assertEqual(ev[-1][0],'trace')
    def test_collision_stops(self):
        r,_,b,ev=make(collision=True)
        tr=r.navigate([0,0,0],0,[0,0,-.5],[0,0,-1])
        self.assertFalse(tr['complete']);self.assertEqual(tr['collisions'],1)
        self.assertEqual(b.steps,1)
    def test_follower_mutation_fatal(self):
        r,_,b,ev=make(mutation=True)
        with self.assertRaisesRegex(ValueError,'MUTATED'):r.navigate([0,0,0],0,[0,0,-.5],[0,0,-1])
        self.assertEqual(b.steps,0);self.assertFalse(ev[-1][1]['complete'])
    def test_budget_failure_preserves_partial_trace(self):
        r,_,b,ev=make(limit=1)
        with self.assertRaisesRegex(RuntimeError,'BUDGET'):r.navigate([0,0,0],0,[0,0,-.5],[0,0,-1])
        self.assertEqual(b.steps,1);self.assertFalse(ev[-1][1]['complete'])
    def test_action_limit_rejected(self):
        r,_,b,ev=make()
        tr=r.navigate([0,0,0],0,[0,0,-.5],[0,0,-1],max_actions=1)
        self.assertFalse(tr['complete']);self.assertEqual(tr['failure_reason'],'FEEDBACK_ACTION_CAP')

if __name__=='__main__':unittest.main()

