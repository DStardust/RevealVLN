"""Evidence-first assembly with exact history action-count controls.

All component reuse is a proposal. Actual assembled histories and the complete
crossed matrix still use the sealed validator and independent physical replay.
"""
import collections
import copy
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
sys.path[:0]=[str(RUNTIME),str(RUNTIME/'compact_loop_v2')]
from core_bridge import FamilyFactory, Reject, digest
from compact import closure_report

def count(a): return collections.Counter(a)

def balanced_histories(a,b,i,neutral):
    rows={'H_A':list(a),'H_B':list(b),'H_A_I':list(a)+list(i)}
    nf=count(neutral)['F']
    if nf<=0:raise Reject('NEUTRAL_HAS_NO_TRANSLATION')
    target_f=max(count(v)['F'] for v in rows.values())
    for name, actions in rows.items():
        gap=target_f-count(actions)['F']
        if gap%nf:raise Reject('FORWARD_COUNT_NOT_DIVISIBLE')
        filler=list(neutral)*(gap//nf)
        rows[name]=filler+actions if name!='H_A_I' else actions+filler
    # Only pairwise L/R padding is allowed here. Full spins are not assumed safe.
    turns={count(v)['L']-count(v)['R'] for v in rows.values()}
    if len(turns)!=1:raise Reject('TURN_IMBALANCE_NEEDS_NEW_PHYSICAL_PROPOSAL')
    target_l=max(count(v)['L'] for v in rows.values())
    for name, actions in rows.items():
        rows[name]=actions+['L','R']*(target_l-count(actions)['L'])
    if len({tuple(v) for v in rows.values()})!=3:raise Reject('DUPLICATE_HISTORY_ACTION_SEQUENCE')
    if max(map(len,rows.values()))+8>512:raise Reject('BALANCED_HISTORY_LENGTH')
    if len({tuple(sorted(count(v).items())) for v in rows.values()})!=1:raise AssertionError('COUNT_BALANCE')
    return rows

class WitnessFactory(FamilyFactory):
    def __init__(self,*args,components,**kwargs):
        super().__init__(*args,**kwargs)
        self.components=copy.deepcopy(components)

    def histories(self,position,yaw):
        initial=self.runner.run(position,yaw,[])['observations'][0]['pose']
        a,b=self.components['a'],self.components['b']
        for role,actions,forbidden in [(self.a,a,[self.b]),(self.b,b,[self.a])]:
            trace=self.runner.run(position,yaw,actions)
            if not self.pattern(trace,[role],forbidden):raise Reject('REUSED_LOOP_PATTERN',role=role)
            if not closure_report(initial,trace['observations'][-1]['pose'])['pass']:raise Reject('REUSED_LOOP_NOT_CLOSED')
        neutral=None
        # Two directions, increasing shuttle distance. They are proposals, not
        # assumed reversible. Every one is stepped and checked before reuse.
        for n in (1,2,3,4):
            for outturn,returnturn in [('L','R'),('R','L')]:
                actions=['F']*n+[outturn]*12+['F']*n+[returnturn]*12
                trace=self.runner.run(position,yaw,actions)
                if self.pattern(trace,[],[self.a,self.b]) and closure_report(initial,trace['observations'][-1]['pose'])['pass']:
                    neutral=actions;break
            if neutral is not None:break
        if neutral is None:raise Reject('NO_ACTUAL_NEUTRAL_TRANSLATION_LOOP')
        irrelevant=list(self.components['i'])+neutral
        trace=self.runner.run(position,yaw,irrelevant)
        # A terminal sighting without STOP is not completion under the unchanged
        # task program. The unused old I-loop terminal-avoidance heuristic is not
        # part of the sealed final matrix requirement.
        if not self.pattern(trace,[self.irrelevant],[self.b]) or not closure_report(initial,trace['observations'][-1]['pose'])['pass']:
            raise Reject('IRRELEVANT_MOVING_LOOP_FAILED')
        rows=balanced_histories(a,b,irrelevant,neutral)
        for name,actions in rows.items():
            trace=self.runner.run(position,yaw,actions)
            required=[self.b] if name=='H_B' else [self.a]+([self.irrelevant] if name=='H_A_I' else [])
            forbidden=[self.a] if name=='H_B' else [self.b]
            if not self.pattern(trace,required,forbidden):raise Reject('BALANCED_HISTORY_PATTERN',history=name)
            if not closure_report(initial,trace['observations'][-1]['pose'])['pass']:raise Reject('BALANCED_HISTORY_NOT_CLOSED',history=name)
        self.emit('balanced_histories',dict(action_counts={k:dict(count(v)) for k,v in rows.items()},
            exact_action_count_match=True,neutral=neutral,irrelevant=irrelevant,
            interpretation='irrelevant moving-loop placement variation; not absence-of-irrelevant-event proof'))
        length=len(rows['H_A'])
        return rows,dict(step=length,position=position,yaw_bin=yaw,target_pose=initial,registered_histories=list(rows.values()))

    def continuations(self,position,yaw):
        t=self.components['terminal']
        return {'C0':t+['S'],'C_A':self.components['a']+t+['S'],'C_B':self.components['b']+t+['S']}
