"""FIT-only temporal visibility accounting; no proposed navigation gain from metadata."""
import collections,hashlib,json,math,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
FIT=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2/run_001'
DATA=LINE/'data_pipeline/ordinary_onpolicy_recovery_v2/run_001'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=HERE/'RESULT.json';assert not out.exists()
    episodes={x['index']:x for x in (read(p) for p in FIT.glob('lanes/lane_*/episode_*.json'))}
    assert len(episodes)==64 and {x['house'] for x in episodes.values()}==set(read(DATA/'RESULT.json')['houses'])
    c=collections.Counter();source={}
    for folder in sorted(FIT.glob('lanes/lane_*')):
        starts={x['index']:x for x in rows(folder/'INTERFACE.jsonl')}
        ps=collections.defaultdict(list);ss=collections.defaultdict(list)
        for x in rows(folder/'POLICY_STEPS.jsonl'):ps[x['index']].append(x)
        for x in rows(folder/'STEPS_PRIVILEGED.jsonl'):ss[x['index']].append(x)
        for i,e in episodes.items():
            if i not in starts:continue
            seen=[starts[i]['rgb_sha256']];actions=[];positions=[e['positions'][0]]
            assert len(ps[i])==len(ss[i])==e['steps']
            for p,s in zip(ps[i],ss[i]):
                t=len(actions);assert p['step']==s['step']==t+1 and p['action']==s['action']
                old=[seen[-1]] if t==0 else seen[-2:]
                new=[seen[-1]] if t==0 else [seen[max(0,t-8)],seen[t]]
                assert len(old)==len(new)==min(2,t+1)
                c['decisions']+=1;c['image_count_unchanged']+=1
                if t>=8:
                    c['full_eight_action_windows']+=1
                    c['older_frame_differs_from_recent_previous']+=int(new[0]!=old[0])
                    c['adjacent_pair_identical']+=int(old[0]==old[1])
                    c['aligned_pair_identical']+=int(new[0]==new[1])
                    c['adjacent_identical_but_aligned_distinct']+=int(old[0]==old[1] and new[0]!=new[1])
                source[i,t]=dict(rgb_sha256=new,executed_actions=list(actions[-8:]))
                seen.append(s['rgb_sha256']);actions.append(s['action']);positions.append(s['position'])
    assert c['decisions']==7225
    inputs={x['record_id']:x for x in rows(DATA/'POLICY_INPUTS.jsonl')}
    eligible={x['advice_record_id'] for x in read(LINE/'sft_acceptance/ordinary_onpolicy_adapt_v6/TRAINING_ROWS.json')[37114:]}
    selected={};conflicting=[]
    for x in rows(DATA/'SUPERVISION_ONLY.jsonl'):
        if x['record_id'] not in eligible:continue
        value=source[x['source_index'],x['decision_step']]
        assert value['executed_actions']==inputs[x['record_id']]['executed_actions']
        # Retain first immutable occurrence per old causal ID; no change to labels or draws.
        if x['record_id'] in selected:continue
        selected[x['record_id']]=dict(record_id=x['record_id'],instruction=inputs[x['record_id']]['instruction'],**value)
    assert len(selected)==4983
    fingerprints=collections.defaultdict(set)
    targets={x['advice_record_id']:x['target'] for x in read(LINE/'sft_acceptance/ordinary_onpolicy_adapt_v6/TRAINING_ROWS.json')[37114:]}
    for key,value in selected.items():
        digest=hashlib.sha256(json.dumps([value['instruction'],value['rgb_sha256'],value['executed_actions']],ensure_ascii=False).encode()).hexdigest()
        fingerprints[digest].add(targets[key])
        for rgb in value['rgb_sha256']:assert (DATA/'content'/(rgb+'.png')).is_file()
    conflicting=[k for k,v in fingerprints.items() if len(v)>1]
    result=dict(status='COMPLETE',unix=time.time(),scope='FIT-only read-only temporal visibility and reconstructability',
        stride=8,stride_reason='aligned with the already-fixed eight executed-action window; no stride sweep',
        counts=dict(c),recovery_inputs_reconstructable=len(selected),new_view_target_conflicts=len(conflicting),
        new_view_unique_fingerprints=len(fingerprints),extra_training_updates=0,extra_simulator_actions=0,
        stronger_navigation_unproven=True,not_a_novel_architecture=True,
        files={str(p):sha(p) for p in [DATA/'POLICY_INPUTS.jsonl',DATA/'SUPERVISION_ONLY.jsonl',Path(__file__)]})
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
