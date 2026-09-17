"""Exact action-count placement control. All physical checks remain at runtime."""
import collections
import copy
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
sys.path.insert(0,str(RUNTIME))
from core_bridge import FamilyFactory,Reject,factory


def matched(a,b,i,k,j):
    if type(k) is not int or type(j) is not int or not 2<=k<=16 or not 1<=j<=16:
        raise Reject('BALANCE_MULTIPLIER_RANGE')
    if any(not x or any(t not in ('F','L','R') for t in x) for x in (a,b,i)):
        raise Reject('COMPONENT_MOTION_ONLY_NONEMPTY')
    rows={'H_A':list(a)+list(i)+list(a)*(k-1),
          'H_A_I':list(a)*k+list(i),'H_B':list(b)*j}
    counts={name:collections.Counter(value) for name,value in rows.items()}
    if len({c['F'] for c in counts.values()})!=1:raise Reject('FORWARD_COUNTS_DIFFER')
    if len({c['L']-c['R'] for c in counts.values()})!=1:raise Reject('NET_TURNS_DIFFER')
    target=max(c['L'] for c in counts.values())
    for name in rows:rows[name]+=['L','R']*(target-counts[name]['L'])
    if max(map(len,rows.values()))>504:raise Reject('BALANCED_HISTORY_GT_504')
    if len({tuple(v) for v in rows.values()})!=3:raise Reject('PLACEMENT_HISTORY_DUPLICATE')
    if len({tuple(collections.Counter(v)[key] for key in ('F','L','R')) for v in rows.values()})!=1:
        raise AssertionError('INTERNAL_COUNT_BALANCE')
    return {name:rows[name] for name in ('H_A','H_B','H_A_I')}


def solve(a,b,i):
    accepted=[];attempts=[]
    for k in range(2,17):
        for j in range(1,17):
            try:
                rows=matched(a,b,i,k,j)
                accepted.append({'k':k,'j':j,'history_actions':len(rows['H_A']),
                                 'action_counts':{key:rows['H_A'].count(key) for key in ('F','L','R')}})
                attempts.append([k,j,'ACCEPTED_COUNT_PLAN'])
            except Reject as error:attempts.append([k,j,error.code])
    return sorted(accepted,key=lambda r:(r['history_actions'],r['k'],r['j'])),attempts


def closed(first,last):
    if set(first.get('sensors',{}))!={'rgb','semantic'} or set(last.get('sensors',{}))!={'rgb','semantic'}:return False
    return all(max(factory.pose_distance(a,b))<=1e-5 for a,b in
               [(first,last)]+[(first['sensors'][k],last['sensors'][k]) for k in ('rgb','semantic')])


class BalancedFactory(FamilyFactory):
    """Original matrix/query/checker are inherited unchanged; no GPU on import."""
    def __init__(self,*args,components,balance,**kwargs):
        super().__init__(*args,**kwargs)
        if set(components)!={'a','b','i','terminal'} or set(balance)!={'k','j'}:
            raise Reject('BALANCED_CONFIG_FIELDS')
        self.components=copy.deepcopy(components)
        self.balance=copy.deepcopy(balance)
        self.planned_histories=matched(components['a'],components['b'],components['i'],**balance)
        if any(a not in ('F','L','R') for a in components['terminal']) or not components['terminal']:
            raise Reject('TERMINAL_COMPONENT_MOTION')
        if max(len(v) for v in self.continuations(None,None).values())>160:
            raise Reject('CONTINUATION_ACTION_CAP')

    def histories(self,position,yaw):
        initial=self.runner.run(position,yaw,[])['observations'][0]['pose']
        patterns=[('a',[self.a],[self.b]),('b',[self.b],[self.a]),
                  ('i',[self.irrelevant],[self.b,self.end])]
        for name,required,forbidden in patterns:
            trace=self.runner.run(position,yaw,self.components[name])
            if not self.pattern(trace,required,forbidden):raise Reject('BALANCED_COMPONENT_EVENT',component=name)
            if not closed(initial,trace['observations'][-1]['pose']):raise Reject('BALANCED_COMPONENT_NOT_CLOSED',component=name)
        rows=copy.deepcopy(self.planned_histories)
        for name,actions in rows.items():
            trace=self.runner.run(position,yaw,actions)
            required=[self.b] if name=='H_B' else [self.a,self.irrelevant]
            forbidden=[self.a] if name=='H_B' else [self.b]
            if not self.pattern(trace,required,forbidden):raise Reject('BALANCED_HISTORY_EVENT',history=name)
            if not closed(initial,trace['observations'][-1]['pose']):raise Reject('BALANCED_HISTORY_NOT_CLOSED',history=name)
        self.emit('balanced_histories',{'control_type':'completed_subgoal_revisit_or_irrelevant_loop_placement',
            'H_A_also_contains_I':True,'event_free_detour_claim':False,'balance':self.balance,
            'exact_F_L_R_counts':{name:{key:actions.count(key) for key in ('F','L','R')} for name,actions in rows.items()},
            'same_Y_rows_still_requires_matrix_replay':True})
        return rows,{'step':len(rows['H_A']),'position':position,'yaw_bin':yaw,
                     'target_pose':initial,'registered_histories':list(rows.values())}

    def continuations(self,position,yaw):
        c=self.components;t=c['terminal']+['S']
        return {'C0':list(t),'C_A':c['a']+t,'C_B':c['b']+t}
