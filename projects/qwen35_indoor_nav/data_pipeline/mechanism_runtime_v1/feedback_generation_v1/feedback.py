"""Feedback-discovered real trajectories; all probes are recorded and budgeted.

No modifications to the sealed factory, checker or backend. The new routes()
override is explicitly a *physical discovery* operator, not a pure geometric
proposal. The inherited factory subsequently independently replays its actions.
"""
import math
from pathlib import Path
import sys

RUNTIME=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(RUNTIME))
from core_bridge import FamilyFactory, TraceRunner, Reject, digest
from factory import yaw_bin, rotations
from habitat_backend import NAMES, heading

def candidate_targets(backend,position,role,limit=8):
    np,hs=backend.np,backend.hs
    candidates=[];seen=set()
    for instance in sorted(backend.eligible[role]):
        center=backend.objects[instance]['center']
        for radius in (.75,1.25):
            for angle in range(8):
                p=[center[0]+radius*math.cos(angle*math.pi/4),center[1],
                   center[2]+radius*math.sin(angle*math.pi/4)]
                target=backend.sim.pathfinder.snap_point(np.asarray(p,dtype=np.float32))
                if not np.isfinite(target).all():continue
                path=hs.ShortestPath();path.requested_start=np.asarray(position,dtype=np.float32)
                path.requested_end=target
                if not backend.sim.pathfinder.find_path(path):continue
                distance=float(path.geodesic_distance)
                if not math.isfinite(distance) or distance>8:continue
                key=tuple(target.tolist())
                if key in seen:continue
                seen.add(key)
                candidates.append(dict(instance=instance,target=target.tolist(),center=center,
                                       distance=distance,radius=radius,angle=angle))
    return sorted(candidates,key=lambda r:(r['distance'],r['instance'],r['radius'],r['angle']))[:limit]

class FeedbackRunner:
    def __init__(self,trace_runner):
        self.base=trace_runner
        self.backend=trace_runner.backend
        self.compiler=trace_runner.compiler
        self.budget=trace_runner.budget
        self.emit=trace_runner.emit

    def navigate(self,position,yaw,target,center,max_actions=140,seed=1109):
        b=self.backend;observations=[];actions=[];collisions=0;complete=False;reason=None
        try:
            self.budget.check_time()
            b.reset(position,yaw,seed)
            observations.append(self.base.observe(0))
            follower=b.hs.GreedyGeodesicFollower(b.sim.pathfinder,b.sim.get_agent(0),goal_radius=.35)
            def move(a):
                nonlocal collisions
                if len(actions)>=max_actions:raise Reject('FEEDBACK_ACTION_CAP')
                self.budget.reserve_action()
                actions.append(a)
                collision=b.step(a)
                if type(collision) is not bool:raise ValueError('COLLISION_TYPE')
                self.emit('action_completed',dict(step=len(actions)-1,action=a,collision=collision,
                                                  phase='feedback_discovery'))
                observations.append(self.base.observe(len(actions)))
                if collision:
                    collisions+=1
                    raise Reject('FEEDBACK_COLLISION')
            while True:
                self.budget.check_time()
                p=observations[-1]['pose']['position']
                path=b.hs.ShortestPath()
                path.requested_start=b.np.asarray(p,dtype=b.np.float32)
                path.requested_end=b.np.asarray(target,dtype=b.np.float32)
                if not b.sim.pathfinder.find_path(path):raise Reject('FEEDBACK_TARGET_UNREACHABLE')
                if float(path.geodesic_distance)<=.35 and math.dist(p,target)<=.35:break
                before=digest(b._pose())
                try:native=follower.next_action_along(b.np.asarray(target,dtype=b.np.float32))
                except b.hs.errors.GreedyFollowerError:raise Reject('FEEDBACK_FOLLOWER_ERROR')
                if digest(b._pose())!=before:raise ValueError('FOLLOWER_MUTATED_ACTUAL_STATE')
                inverse_names={v:k for k,v in NAMES.items()}
                if native not in inverse_names:raise Reject('FEEDBACK_PREMATURE_STOP')
                move(inverse_names[native])
            pose=observations[-1]['pose']
            desired=heading(center[0]-pose['position'][0],center[2]-pose['position'][2])
            for a in rotations(desired-yaw_bin(pose))+['L','R']:
                move(a)
            complete=True
        except Reject as error:
            reason=error.code
        finally:
            trace=dict(actions=actions,observations=observations,complete=complete,
                       collisions=collisions,seed=seed,initial_position=list(position),
                       initial_yaw_bin=yaw,normalization_events=[],
                       discovery_kind='actual_state_feedback_v1',failure_reason=reason)
            for record,atoms in zip(observations,self.compiler.atoms(observations)):
                record['see2']=atoms
            trace['trace_hash']=digest(trace)
            self.emit('trace',trace)
        return trace

class FeedbackFactory(FamilyFactory):
    def routes(self,position,yaw,role):
        self.budget.check_time()
        targets=candidate_targets(self.backend,position,role)
        self.emit('feedback_targets',dict(position=position,yaw=yaw,role=role,targets=targets))
        feedback=FeedbackRunner(self.runner)
        seen=set()
        for row in targets:
            trace=feedback.navigate(position,yaw,row['target'],row['center'])
            actions=trace['actions']
            if trace['complete'] and tuple(actions) not in seen:
                seen.add(tuple(actions))
                yield actions

