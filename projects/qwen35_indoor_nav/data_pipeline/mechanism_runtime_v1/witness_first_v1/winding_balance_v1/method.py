"""Conditional winding padding: a proposal until actual neutral-spin replay."""
import collections
import copy
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
sys.path.insert(0,str(RUNTIME))
from core_bridge import FamilyFactory,Reject,factory
CONTROL_TYPE='completed_subgoal_revisit_placement_not_event_free_detour'


def plan(a,b,i,k,j):
    if type(k) is not int or type(j) is not int or not 2<=k<=16 or not 1<=j<=16:raise Reject('MULTIPLIER_RANGE')
    if any(not x or any(t not in ('F','L','R') for t in x) for x in (a,b,i)):raise Reject('COMPONENT_MOTION_ONLY_NONEMPTY')
    if i.count('F')<2:raise Reject('IRRELEVANT_LOOP_REQUIRES_TWO_FORWARD_ACTIONS')
    rows={'H_A':list(a)+list(i)+list(a)*(k-1),'H_B':list(b)*j,'H_A_I':list(a)*k+list(i)}
    counts={name:collections.Counter(value) for name,value in rows.items()}
    if len({c['F'] for c in counts.values()})!=1:raise Reject('FORWARD_COUNTS_DIFFER')
    if len({(c['L']-c['R'])%24 for c in counts.values()})!=1:raise Reject('NONINTEGER_WINDING_DIFFERENCE')
    base_l=max(c['L'] for c in counts.values());base_r=max(c['R'] for c in counts.values())
    residue=next(iter(counts.values()))['L']-next(iter(counts.values()))['R']
    # Minimal nonnegative target adjustment; never change source rows or final
    # equal-count/physical thresholds. Ties deterministically prefer raising L.
    add_l=(residue-(base_l-base_r))%24
    add_r=((base_l-base_r)-residue)%24
    adjustment_direction='L' if add_l<=add_r else 'R'
    target_l=base_l+(add_l if adjustment_direction=='L' else 0)
    target_r=base_r+(add_r if adjustment_direction=='R' else 0)
    pads={};required=set()
    for name in rows:
        gap_l=target_l-counts[name]['L'];gap_r=target_r-counts[name]['R']
        if (gap_l-gap_r)%24:raise Reject('NONINTEGER_GAP_WINDING')
        direction='L' if gap_l>gap_r else 'R' if gap_r>gap_l else None
        revolutions=abs(gap_l-gap_r)//24
        pairs=min(gap_l,gap_r)
        actions=([direction]*24*revolutions if direction else [])+['L','R']*pairs
        if direction:required.add(direction)
        pads[name]={'direction':direction,'revolutions':revolutions,'LR_pairs':pairs,'actions':actions}
        rows[name]+=actions
    if max(map(len,rows.values()))>504:raise Reject('BALANCED_HISTORY_GT_504')
    if len({tuple(v) for v in rows.values()})!=3:raise Reject('PLACEMENT_HISTORY_DUPLICATE')
    profiles={name:{t:row.count(t) for t in ('F','L','R')} for name,row in rows.items()}
    if len({tuple(c[t] for t in ('F','L','R')) for c in profiles.values()})!=1:raise AssertionError('COUNT_NOT_MATCHED')
    return {'histories':rows,'pads':pads,'required_neutral_spin_directions':sorted(required),
            'target_alignment':{'original_max_L':base_l,'original_max_R':base_r,
                'raise_direction':adjustment_direction,'raise_steps':min(add_l,add_r),
                'target_L':target_l,'target_R':target_r,'common_net_residue':residue%24},
            'history_actions':len(rows['H_A']),'action_counts':profiles['H_A']}


def matched(a,b,i,k,j):return plan(a,b,i,k,j)['histories']


def solve(a,b,i):
    accepted=[];attempts=[]
    af=a.count('F');bf=b.count('F');inf=i.count('F')
    for k in range(2,17):
        for j in range(1,17):
            if inf>=2 and k*af+inf!=j*bf:
                attempts.append([k,j,'FORWARD_COUNTS_DIFFER']);continue
            try:
                value=plan(a,b,i,k,j)
                accepted.append({'k':k,'j':j,**{name:value[name] for name in
                    ('history_actions','action_counts','pads','required_neutral_spin_directions','target_alignment')}})
                attempts.append([k,j,'COUNT_PLAN_REQUIRES_REAL_SPIN_VALIDATION'])
            except Reject as error:attempts.append([k,j,error.code])
    return sorted(accepted,key=lambda r:(r['history_actions'],r['k'],r['j'])),attempts


def closed(first,last):
    if set(first.get('sensors',{}))!={'rgb','semantic'} or set(last.get('sensors',{}))!={'rgb','semantic'}:return False
    return all(max(factory.pose_distance(a,b))<=1e-5 for a,b in
        [(first,last)]+[(first['sensors'][k],last['sensors'][k]) for k in ('rgb','semantic')])


class WindingBalancedFactory(FamilyFactory):
    def __init__(self,*args,components,balance,**kwargs):
        super().__init__(*args,**kwargs)
        if set(components)!={'a','b','i','terminal'} or set(balance)!={'k','j'}:raise Reject('CONFIG_FIELDS')
        self.components=copy.deepcopy(components);self.balance=copy.deepcopy(balance)
        self.padding_plan=plan(components['a'],components['b'],components['i'],**balance)
        if not components['terminal'] or any(a not in ('F','L','R') for a in components['terminal']):raise Reject('TERMINAL_COMPONENT_MOTION')
        if max(map(len,self.continuations(None,None).values()))>160:raise Reject('CONTINUATION_ACTION_CAP')

    def neutral_spins(self,position,yaw,initial):
        evidence={}
        for direction in self.padding_plan['required_neutral_spin_directions']:
            trace=self.runner.run(position,yaw,[direction]*24)
            # Full original pattern includes complete trace / no collision.
            # Seeing T without STOP is allowed by the unchanged task checker.
            if not self.pattern(trace,[],[self.a,self.b]):raise Reject('HUB_SPIN_NOT_ANCHOR_NEUTRAL',direction=direction)
            if not closed(initial,trace['observations'][-1]['pose']):raise Reject('HUB_SPIN_NOT_CLOSED',direction=direction)
            evidence[direction]={'trace_hash':trace['trace_hash'],'actions':24,'neutral_for':[self.a,self.b],
                                 'terminal_visibility_without_STOP_permitted':True}
            self.emit('actual_neutral_full_spin',{'direction':direction,**evidence[direction]})
        return evidence

    def histories(self,position,yaw):
        initial=self.runner.run(position,yaw,[])['observations'][0]['pose']
        # Cheap decisive test first. A forbidden witness cannot be hidden by
        # proceeding to repetitions or by calling the spin a generic neutral pad.
        spin_evidence=self.neutral_spins(position,yaw,initial)
        for name,required,forbidden in [('a',[self.a],[self.b]),('b',[self.b],[self.a]),
                                      ('i',[self.irrelevant],[self.b,self.end])]:
            trace=self.runner.run(position,yaw,self.components[name])
            if not self.pattern(trace,required,forbidden):raise Reject('COMPONENT_EVENT',component=name)
            if not closed(initial,trace['observations'][-1]['pose']):raise Reject('COMPONENT_NOT_CLOSED',component=name)
        rows=copy.deepcopy(self.padding_plan['histories'])
        for name,actions in rows.items():
            trace=self.runner.run(position,yaw,actions)
            required=[self.b] if name=='H_B' else [self.a,self.irrelevant]
            forbidden=[self.a] if name=='H_B' else [self.b]
            if not self.pattern(trace,required,forbidden):raise Reject('HISTORY_EVENT',history=name)
            if not closed(initial,trace['observations'][-1]['pose']):raise Reject('HISTORY_NOT_CLOSED',history=name)
        self.emit('balanced_histories',{'control_type':CONTROL_TYPE,'H_A_also_contains_I':True,
            'balance':self.balance,'padding':self.padding_plan['pads'],'actual_neutral_spin_evidence':spin_evidence,
            'exact_F_L_R_counts':{name:{t:actions.count(t) for t in ('F','L','R')} for name,actions in rows.items()},
            'event_free_detour_claim':False,'same_Y_rows_still_requires_matrix_replay':True})
        return rows,{'step':len(rows['H_A']),'position':position,'yaw_bin':yaw,
                     'target_pose':initial,'registered_histories':list(rows.values())}

    def continuations(self,position,yaw):
        c=self.components;t=c['terminal']+['S']
        return {'C0':list(t),'C_A':c['a']+t,'C_B':c['b']+t}


BalancedFactory=WindingBalancedFactory
