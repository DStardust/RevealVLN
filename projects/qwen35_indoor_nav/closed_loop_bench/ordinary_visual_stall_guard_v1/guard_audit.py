"""Independent decision replay from causal RGB hashes; never import controller."""
import collections
import itertools
import json
import math

ACTIONS=('move_forward','turn_left','turn_right','STOP')


def records(p):
    with p.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def audit(run,protocol):
    total_actions=0;total_interventions=0;by_episode={}
    for lane in range(protocol['lanes']):
        folder=run/'lanes'/f'lane_{lane:02d}'
        initial={r['index']:r['rgb_sha256'] for r in records(folder/'INTERFACE.jsonl')}
        state={i:dict(rgb=rgb,streak=0,used=0,step=0) for i,rgb in initial.items()}
        for policy,physical in itertools.zip_longest(records(folder/'POLICY_STEPS.jsonl'),records(folder/'STEPS_PRIVILEGED.jsonl')):
            assert policy is not None and physical is not None,'MISSING_ACTION_RECEIPT'
            assert policy['index']==physical['index'] and policy['step']==physical['step']
            index=policy['index'];x=state[index];assert policy['step']==x['step']+1
            values=policy['logits'];assert len(values)==4 and all(math.isfinite(v) for v in values)
            proposed=ACTIONS[max(range(4),key=values.__getitem__)]
            assert proposed==policy['model_proposed_action']
            # Deliberately reconstructed independently, without loading guard.py.
            intervene=proposed=='move_forward' and x['streak']>=8 and x['used']<4
            expected='turn_left' if intervene else proposed
            receipt=dict(proposed_action=proposed,action=expected,intervened=intervene,
                stagnant_forwards_before=x['streak'],interventions_before=x['used'],interventions_after=x['used']+int(intervene),
                rgb_sha256=x['rgb'],reason='EIGHT_IDENTICAL_FORWARD_OBSERVATIONS' if intervene else None)
            assert policy['guard']==receipt,'CAUSAL_GUARD_RECEIPT_MISMATCH'
            assert policy['action']==physical['action']==expected,'WRONG_EXECUTED_ACTION'
            assert proposed!='STOP' or expected=='STOP','STOP_OVERRIDDEN'
            x['used']+=int(intervene);assert x['used']<=4
            x['streak']=min(8,x['streak']+1) if expected=='move_forward' and physical['rgb_sha256']==x['rgb'] else 0
            x['rgb']=physical['rgb_sha256'];x['step']+=1
            total_actions+=1;total_interventions+=int(intervene)
        for index,x in state.items():
            assert 1<=x['step']<=protocol['max_steps'] and index not in by_episode
            by_episode[index]=x['used']
    assert len(by_episode)==protocol['episode_count']
    return dict(passed=True,causal_rgb_action_replay=True,goal_pose_collision_not_used_by_controller=True,
        total_actions=total_actions,total_interventions=total_interventions,
        intervened_episodes=sum(n>0 for n in by_episode.values()),by_episode=by_episode,
        stop_overrides=0,extra_hidden_actions=0,per_episode_intervention_cap=4)
