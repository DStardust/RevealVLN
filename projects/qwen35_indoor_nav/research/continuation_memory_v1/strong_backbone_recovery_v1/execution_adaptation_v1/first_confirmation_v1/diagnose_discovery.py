"""Recompute completed discovery results and localize differences without training."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent;ADAPT=HERE.parent
sys.path.insert(0,str(ADAPT.parent))
import common as u
spec=importlib.util.spec_from_file_location('frozen_scope_review',ADAPT/'scope_validation_v1/review.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def main():
    run=ADAPT/'scope_validation_v1/runs/scope_001/unseen';rows=old.records(run)
    if len(rows)!=80:raise ValueError('DISCOVERY_NOT_COMPLETE')
    config=u.read(run/'PROTOCOL.json');names=['NATIVE',*[a for a in config['heads'] if '_FIRST_' in a]]
    totals={};differences=[];trace_cache={}
    def trace(i,arm):
        key=(i,arm)
        if key not in trace_cache:
            path=Path(rows[i]['path']).parent/arm/'TRACE.jsonl'
            if u.sha(path)!=rows[i]['trace_hashes'][arm]:raise ValueError('TRACE_CHANGED')
            trace_cache[key]=[json.loads(line) for line in path.read_text().splitlines()]
        return trace_cache[key]
    for arm in names[1:]:
        buckets={c:dict(n=0,spl_difference_sum=0.,step_difference_sum=0.) for c in ['both_success','rescue','regression','both_fail']}
        counts=Counter()
        for i,r in rows.items():
            a,n=r['outcomes'][arm],r['outcomes']['NATIVE']
            c='both_success' if a['success'] and n['success'] else 'rescue' if a['success'] else 'regression' if n['success'] else 'both_fail'
            buckets[c]['n']+=1;buckets[c]['spl_difference_sum']+=a['spl']-n['spl'];buckets[c]['step_difference_sum']+=a['steps']-n['steps']
            actions=[e for e in trace(i,arm) if e['event']=='action'];last=actions[-1]
            counts['far_stop']+=last['executed_action']==0 and last['distance']>=3
            counts['budget_exhausted']+=last['executed_action']!=0 and len(actions)>=500
            counts['entered_range_without_success']+=any(e['distance']<3 for e in actions) and not a['success']
        totals[arm]=dict(decomposition=buckets,failures=dict(counts),
            delta_sr=(buckets['rescue']['n']-buckets['regression']['n'])/80,
            delta_spl=sum(x['spl_difference_sum'] for x in buckets.values())/80,
            delta_steps=sum(x['step_difference_sum'] for x in buckets.values())/80)
    for seed in config['seeds']:
        current,delta=f'CURRENT_FIRST_s{seed}',f'DELTA_FIRST_s{seed}'
        key=f'DELTA_minus_CURRENT_FIRST_s{seed}'
        for i,r in sorted(rows.items()):
            c,d,n=[r['outcomes'][a]['success'] for a in [current,delta,'NATIVE']]
            if c==d:continue
            evidence=r['seed_pair_audits'][key]['first_generated_difference']
            if evidence is None:raise ValueError('OUTCOME_DIFFERENCE_WITHOUT_GENERATION_DIFFERENCE')
            local={}
            for arm in [current,delta]:
                events=trace(i,arm);g=[e for e in events if e['event']=='generation'][evidence['query_index']]
                token=g['tokens'][evidence['token_offset']];native=sorted(token['native_logits'],reverse=True);method=sorted(token['method_logits'],reverse=True)
                local[arm]=dict(native_action=token['native_action'],method_action=token['method_action'],
                    native_logits=token['native_logits'],method_logits=token['method_logits'],residual=token['residual'],
                    native_margin=native[0]-native[1],method_margin=method[0]-method[1],
                    actor_feature_sha256=token['actor_feature_sha256'],query_memory_sha256=token['query_memory_sha256'])
            if local[current]['actor_feature_sha256']!=local[delta]['actor_feature_sha256']:
                raise ValueError('FIRST_DIFFERENCE_FEATURES_NOT_PAIRED')
            differences.append(dict(seed=seed,id=i,house=r['house'],native_success=n,current_success=c,delta_success=d,
                category=('delta_extra_rescue' if not n else 'delta_retains_success') if d>c else ('delta_missed_current_rescue' if not n else 'delta_regresses_native_success'),
                first_difference=evidence,local_logits=local,
                interpretation='First differing decision on a matched prefix; downstream failure is not proven to have one cause.'))
    result=dict(discovery_routes=80,executions=1040,source_result_sha256=u.sha(run/'RESULT.json'),
        source_protocol_sha256=u.sha(run/'PROTOCOL.json'),totals=totals,delta_current_outcome_differences=differences,
        per_seed_categories={str(s):dict(Counter(r['category'] for r in differences if r['seed']==s)) for s in config['seeds']},
        base_updates=0,optimizer_updates=0,gpu_hours=0,posthoc_diagnosis=True)
    u.write(HERE/'DISCOVERY_DIAGNOSIS.json',result)
    u.write(HERE/'DISCOVERY_REVIEW_RECOMPUTED.json',old.summarize(config,u.read(run/'DATA_MANIFEST.json')['episodes'],rows))
    print(json.dumps(result['per_seed_categories']))


if __name__=='__main__':main()
