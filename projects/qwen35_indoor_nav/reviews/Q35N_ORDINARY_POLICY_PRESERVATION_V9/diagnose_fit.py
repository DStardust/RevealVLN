"""FIT-only offline accounting; old-model actions are behavior, not oracle truth."""
import collections,hashlib,json,math,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
DATA=LINE/'data_pipeline/ordinary_onpolicy_recovery_v2/run_001'
FIT=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2/run_001'
V6=LINE/'sft_acceptance/ordinary_onpolicy_adapt_v6'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=HERE/'FIT_BEHAVIOR_DIAGNOSTIC.json';assert not out.exists()
    episodes={x['index']:x for x in (read(p) for p in FIT.glob('lanes/lane_*/episode_*.json'))}
    policy={(x['index'],x['step']-1):x for p in FIT.glob('lanes/lane_*/POLICY_STEPS.jsonl') for x in rows(p)}
    assert len(episodes)==64 and sum(x['success'] for x in episodes.values())==16 and len(policy)==7225
    selected={x['advice_record_id'] for x in read(V6/'TRAINING_ROWS.json')[37114:]}
    assert len(selected)==4983
    counts=collections.defaultdict(collections.Counter)
    inputs={x['record_id']:x for x in rows(DATA/'POLICY_INPUTS.jsonl')}
    files=[V6/'TRAINING_ROWS.json',DATA/'SUPERVISION_ONLY.jsonl',DATA/'POLICY_INPUTS.jsonl']
    examples=[]
    for x in rows(DATA/'SUPERVISION_ONLY.jsonl'):
        if x['record_id'] not in selected:continue
        e=episodes[x['source_index']];p=policy[x['source_index'],x['decision_step']]
        assert x['split']=='FIT' and x['episode_id']==e['episode_id'] and p['action']==x['original_action']
        group='original_successful_route' if e['success'] else 'original_failed_route'
        c=counts[group];c['inputs']+=1;c['teacher_disagrees']+=int(x['teacher_action']!=x['original_action'])
        c['teacher_stop']+=int(x['target']==3)
        c['teacher_stop_before_original_stop']+=int(x['target']==3 and x['original_action']!='STOP')
        if e['success'] and x['teacher_action']!=x['original_action'] and len(examples)<5:
            examples.append(dict(house=e['house'],episode_id=e['episode_id'],step=x['decision_step'],
                original_action=x['original_action'],teacher_action=x['teacher_action'],
                instruction=inputs[x['record_id']]['instruction'],original_route_eventually_successful=True,
                caution='Different teacher action is not proof of a wrong label; geometric shortest-path teacher need not preserve language intermediate route.'))
    assert sum(x['inputs'] for x in counts.values())==4983
    result=dict(status='COMPLETE',unix=time.time(),data_split='FIT only',original_episodes=64,
        original_successful_episodes=16,eligible_new_inputs=4983,by_original_outcome=dict(counts),
        examples=examples,files={str(p):sha(p) for p in files},training_updates=0,simulator_actions=0,
        hypothesis='Unconstrained adaptation changes both failed and already-successful behavior; output preservation may reduce loss of old successes.',
        supported='Successful original FIT routes also receive alternative geometric targets.',
        untested='KL preservation improves closed-loop navigation; not established by these counts.',
        novel_contribution=False)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['examples','files']},ensure_ascii=False))
if __name__=='__main__':main()
