"""Read sealed navigation and fixed FIT/DEV caches; no GPU or optimization."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ADAPT = HERE.parent
BASE = ADAPT.parent
sys.path[:0] = [str(ADAPT), str(BASE)]
import common as u


def dist(values):
    values = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
        p50=values[len(values)//2] if values else None,
        p95=values[min(len(values)-1, int(len(values)*.95))] if values else None)


def traces():
    spec = importlib.util.spec_from_file_location('sealed_review', ADAPT/'operation_v1/review.py')
    review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
    run = ADAPT/'runs/experiment_002/unseen'
    rows = review.records(run)
    u.write(HERE/'SEALED_GROUP_SNAPSHOT.json', dict(unix=time.time(), groups=rows,
        planned=369, selection='all complete state-sealed groups at snapshot; no outcome filtering'))
    stats = defaultdict(Counter); vectors = defaultdict(lambda: defaultdict(list))
    for i, group in sorted(rows.items()):
        for arm, outcome in group['outcomes'].items():
            p = Path(group['path']).parent/arm/'TRACE.jsonl'
            if u.sha(p) != group['trace_hashes'][arm]: raise ValueError('TRACE_CHANGED')
            events = [json.loads(line) for line in p.read_text().splitlines()]
            s = stats[arm]; s['episodes'] += 1; s['successes'] += outcome['success']
            native = group['outcomes']['NATIVE']['success']
            tag = 'rescue' if outcome['success'] > native else 'regression' if outcome['success'] < native else 'retained'
            s[tag] += 1
            actions = [e for e in events if e['event']=='action']
            final = actions[-1]
            stop = final['executed_action']==0
            s['active_stop'] += stop; s['far_stop'] += stop and final['distance']>=3
            s['budget_exhausted'] += not stop and len(actions)>=500
            s['entered_range_without_success'] += any(e['distance']<3 for e in actions) and not outcome['success']
            s['collisions'] += sum(e['collision'] for e in actions)
            vectors[arm]['steps'].append(len(actions)); vectors[arm]['spl'].append(outcome['spl'])
            if arm!='NATIVE':
                audit = group['audits'][arm]
                first = audit['first_generated_difference']
                s['bitwise_prefix_pairs'] += audit['logits_bitwise_equal']
                if first:
                    offset=first['token_offset']; s[f'first_difference_offset_{offset}'] += 1
                    s[f'{tag}_first_offset_{offset}'] += 1
                    vectors[arm]['first_difference_environment_step'].append(first['query_environment_step'])
            for e in events:
                if e['event']!='generation': continue
                for t in e['tokens']:
                    if not t['residual_applied']: continue
                    offset=t['offset']; s['adjustable_tokens'] += 1
                    flip=t['native_token']!=t['method_token']; s['token_flips'] += flip
                    s[f'token_flips_offset_{offset}'] += flip
                    s['native_continue_to_method_stop'] += t['native_action']!=0 and t['method_action']==0
                    s['native_stop_to_method_continue'] += t['native_action']==0 and t['method_action']!=0
                    vectors[arm]['stop_advantage_residual'].append(t['residual'][0]-sum(t['residual'][1:])/3)
                    vectors[arm]['residual_max_abs'].append(max(map(abs,t['residual'])))
    result = dict(scope='PARTIAL_SEALED_PAIRED_DIAGNOSIS_NOT_FULL_SR', complete_groups=len(rows), planned=369,
        metrics={arm:dict(counts=dict(stats[arm]), distributions={k:dist(v) for k,v in vectors[arm].items()}) for arm in stats})
    u.write(HERE/'TRAJECTORY_DIAGNOSIS.json', result)
    print(json.dumps({'trace_diagnosis':{k:v['counts'] for k,v in result['metrics'].items()}}),flush=True)


def heads():
    import torch
    import torch.nn.functional as F
    from data import load_rows
    from model import ExecutionAdaptation
    torch.set_num_threads(4)
    cfg=u.read(ADAPT/'runs/train_001/TRAINING_CONFIG.json')
    # Entire DEV, and a score-independent systematic FIT sample for scales.
    dev=load_rows(cfg['capture_run'],'DEV'); fit=load_rows(cfg['capture_run'],'FIT')[::8]
    result={}; began=time.time()
    for seed in cfg['seeds']:
        for mode in cfg['modes']:
            arm=f'{mode}_s{seed}'; model=ExecutionAdaptation(3584,mode).eval()
            saved=torch.load(ADAPT/f'runs/train_001/{arm}/FINAL.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']); counters=defaultdict(Counter); values=defaultdict(lambda:defaultdict(list))
            with torch.inference_mode():
                for row in dev+fit:
                    split=row['partition']+'_'+row['kind']; c=counters[split]; v=values[split]
                    x=row['memory_features']; m=model.reset(); at={}
                    for t in range(len(x)):
                        action=row['executed_actions'][t-1:t] if t else torch.tensor([4])
                        prev=x[t-1:t] if t else None
                        main=model.writer(model.norm(x[t:t+1]))
                        change=(x[t:t+1]-prev if t else torch.zeros_like(x[t:t+1])) if mode=='DELTA' else x[t:t+1]
                        extra=model.change_writer(change); action_term=model.executed_action(action)
                        candidate=main+extra+action_term+model.recurrent(m.flatten(1))
                        v['main_writer_rms'].append(float(main.square().mean().sqrt()))
                        v['extra_writer_rms'].append(float(extra.square().mean().sqrt()))
                        v['action_writer_rms'].append(float(action_term.square().mean().sqrt()))
                        v['tanh_saturation_fraction'].append(float((candidate.abs()>3).float().mean()))
                        m=model.update(x[t:t+1],m,prev,action); at[t]=m
                        v['memory_l2_before_unit_norm'].append(float(m.norm()))
                    context=torch.cat([at[int(t)] for t in row['actor_context_steps']])
                    af=row['actor_features']; bias=model.action_delta(af,context)
                    zero=model.action_delta(af,torch.zeros_like(context))
                    base=row['base_logits']; pred=(base+bias).argmax(-1); original=base.argmax(-1)
                    label=row['targets']; known=row['known']; c['rows']+=1; c['tokens']+=int(known.sum())
                    c['native_correct']+=int((original[known]==label[known]).sum())
                    c['method_correct']+=int((pred[known]==label[known]).sum())
                    c['rescued_tokens']+=int(((pred==label)&(original!=label)&known).sum())
                    c['regressed_tokens']+=int(((pred!=label)&(original==label)&known).sum())
                    c['zero_memory_action_flips']+=int((((base+zero).argmax(-1)!=pred)&known).sum())
                    c['stop_labels']+=int(((label==0)&known).sum()); c['stop_predictions']+=int(((pred==0)&known).sum())
                    c['false_stop']+=int(((pred==0)&(label!=0)&known).sum())
                    c['missed_stop']+=int(((pred!=0)&(label==0)&known).sum())
                    for offset in range(4):
                        mask=known&(row['action_token_offsets']==offset)
                        c[f'offset_{offset}_n']+=int(mask.sum())
                        c[f'offset_{offset}_native_correct']+=int((original[mask]==label[mask]).sum())
                        c[f'offset_{offset}_method_correct']+=int((pred[mask]==label[mask]).sum())
                    v['bias_l2'].extend(bias[known].norm(dim=-1).tolist())
                    v['memory_effect_l2'].extend((bias-zero)[known].norm(dim=-1).tolist())
                    v['stop_advantage_residual'].extend((bias[:,0]-bias[:,1:].mean(-1))[known].tolist())
                    v['teacher_ce'].append(float(F.cross_entropy((base+bias)[known],label[known])))
            result[arm]={s:dict(counts=dict(c),distributions={k:dist(v) for k,v in values[s].items()}) for s,c in counters.items()}
            print(json.dumps({'head':arm,'counts':{s:dict(c) for s,c in counters.items()},'elapsed':time.time()-began}),flush=True)
            u.write(HERE/'HEAD_DIAGNOSIS.json',dict(scope='FROZEN_FEATURE_CPU_DIAGNOSTIC_NO_UPDATES',
                dev_ids=[r['id'] for r in dev],fit_sample_ids=[r['id'] for r in fit],heads=result,gpu_hours=0,optimizer_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['traces','heads'],required=True);args=p.parse_args()
    if args.phase=='traces': traces()
    else: heads()
