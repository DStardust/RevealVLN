import collections
import copy
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'assembly_v1'))
from method import WitnessFactory,Reject,closure_report

def matched_revisit(a,b,tail):
    # Same completed-subgoal revisit at two placements. B repeats B, never A.
    irrelevant=list(a)+list(tail)
    rows={'H_A':irrelevant+list(a),'H_B':list(b)+list(b),'H_A_I':list(a)+irrelevant}
    counts={k:collections.Counter(v) for k,v in rows.items()}
    if len({v['F'] for v in counts.values()})!=1:raise Reject('REVISIT_FORWARD_COUNTS_DIFFER')
    if len({v['L']-v['R'] for v in counts.values()})!=1:raise Reject('REVISIT_NET_TURNS_DIFFER')
    target=max(v['L'] for v in counts.values())
    for name in rows:rows[name]+=['L','R']*(target-counts[name]['L'])
    if len({tuple(v) for v in rows.values()})!=3:raise Reject('REVISIT_HISTORY_DUPLICATE')
    if len(rows['H_A'])+8>512:raise Reject('REVISIT_HISTORY_LENGTH')
    assert len({tuple(sorted(collections.Counter(v).items())) for v in rows.values()})==1
    return rows,irrelevant

class RevisitFactory(WitnessFactory):
    def histories(self,position,yaw):
        initial=self.runner.run(position,yaw,[])['observations'][0]['pose']
        a,b=self.components['a'],self.components['b']
        rows,irrelevant=matched_revisit(a,b,self.components['i'])
        if collections.Counter(irrelevant)['F']<2:raise Reject('REVISIT_NOT_MOVING')
        trace=self.runner.run(position,yaw,irrelevant)
        if not self.pattern(trace,[self.a,self.irrelevant],[self.b,self.end]):raise Reject('REVISIT_NOT_STATE_PRESERVING_COMPONENT')
        if not closure_report(initial,trace['observations'][-1]['pose'])['pass']:raise Reject('REVISIT_NOT_CLOSED')
        for name,actions in rows.items():
            trace=self.runner.run(position,yaw,actions)
            required=[self.b] if name=='H_B' else [self.a]+([self.irrelevant] if name=='H_A_I' else [])
            forbidden=[self.a] if name=='H_B' else [self.b]
            if not self.pattern(trace,required,forbidden):raise Reject('REVISIT_HISTORY_PATTERN',history=name)
            if not closure_report(initial,trace['observations'][-1]['pose'])['pass']:raise Reject('REVISIT_HISTORY_CLOSURE',history=name)
        self.emit('balanced_histories',dict(action_counts={k:dict(collections.Counter(v)) for k,v in rows.items()},
            exact_action_count_match=True,irrelevant=irrelevant,
            control_type='completed_subgoal_revisit_placement_not_event_free_detour',
            control_requires_measured_same_Y_rows=True))
        return rows,dict(step=len(rows['H_A']),position=position,yaw_bin=yaw,target_pose=initial,registered_histories=list(rows.values()))
