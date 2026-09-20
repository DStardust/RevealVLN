"""Read-only exact-state and matched-memory diagnostics; never select checkpoints."""
from collections import defaultdict,Counter
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
import objective as o
from select_action import tensor_indices

def visible_scale(trace,eligible,t):
    pixels=trace['observations'][t]['pixels'];previous=trace['observations'][t-1]['pixels'] if t else {}
    count=max((min(pixels.get(str(i),0),previous.get(str(i),0)) for i in eligible),default=0)
    return dict(same_instance_two_frame_pixels=count,
        scale='clear' if count>=1024 else 'near_threshold' if count>=256 else 'below_threshold' if count else 'absent')

def main(run):
    data=read(run/'DATA.json');config=read(run/'PROTOCOL.json');diagnostic_split=config.get('diagnostic_split','TEST')
    cache=torch.load(run/'features/FEATURES.pt',map_location='cpu',weights_only=True)
    torch.set_num_threads(4);events=[];interventions=[];excluded=[];error_counts=defaultdict(Counter);queries=[]
    raw_families={f['family_id']:f for f in data['raw_families']}
    ordinary=read(HERE.parent/'natural_transfer_v9/DATA.json')
    ordinary_cache=torch.load(run/'features/ORDINARY_FEATURES.pt',map_location='cpu',weights_only=True)
    ordinary_rows=[]
    for seed in config['seeds']:
        for arm in ('B1','B2','Ours'):
            net=o.initialize(seed);net.load_state_dict(torch.load(run/'train'/f'{arm}_{seed}'/'FINAL.pt',weights_only=True));net.eval()
            with torch.no_grad():
                for record in ordinary['records']:
                    if record['partition']!='check':continue
                    features=ordinary_cache['features'][record['features']].unsqueeze(0)
                    native=ordinary_cache['logits'][record['features']]
                    memory=net.encode(features)[0][0]
                    method=net.action_logits(memory,native,features[0]);target=torch.tensor(record['targets'])
                    predicted=tensor_indices(method);base=tensor_indices(native)
                    ordinary_rows.append(dict(arm=arm,seed=seed,house=record['row']['scene_group'],N=len(target),
                        method_correct=int((predicted==target).sum()),native_correct=int((base==target).sum()),
                        native_stop_method_continue=int(((base==3)&(predicted!=3)).sum())))
            for family in data['families']:
                if family['split']=='FIT':continue
                b=o.batch(family,'cpu')
                with torch.no_grad():
                    x=cache['features'][b['indices']];states,_=o.encode_masked(net,x,b['alive'])
                    if arm!='B1':
                        probability=o.compose_state_probability(net.state_head(states.flatten(2)).sigmoid(),b['query']) if arm=='B2' else net.reader(states.flatten(0,1),b['query'].flatten(0,1)).sigmoid().view_as(b['y'])
                        mask=b['query_mask'];correct=((probability>=.5)==b['y'].bool())
                        queries.append(dict(family=family['family_id'],house=family['house'],split=family['split'],seed=seed,arm=arm,
                            valid=int(mask.sum()),correct=int((correct*mask).sum()),
                            rule='analytic exact-state composition' if arm=='B2' else 'trained result reader',
                            future_query_used_only_in_readonly_diagnostic=True))
                    if arm=='B2':
                        p=net.state_head(states.flatten(2)).sigmoid()
                        actions=tensor_indices(net.action_logits(states.flatten(0,1),cache['logits'][b['indices']].flatten(0,1),x.flatten(0,1))).view_as(b['indices'])
                        for i,row in enumerate(family['sequences']):
                            raw=raw_families[family['family_id']];trace=read(DATA_LINE/row['trace_path'])
                            anchor=None if row['task_id']=='task_T' else raw['compiler']['tasks'][row['task_id']]['anchor']
                            first=None;first_detected=None
                            for t,z in enumerate(row['state_targets']):
                                if z[1] and first is None:first=t;first_detected=bool(p[i,t,1]>=.5)
                                truth=torch.tensor(z,dtype=torch.bool);prediction=p[i,t]>=.5
                                visibility=visible_scale(trace,raw['compiler']['eligible'][anchor or 'terminal'],t)
                                gap=None if first is None or anchor is None else t-first
                                gap_bin='not_applicable' if anchor is None else 'absent' if gap is None else '0-8' if gap<=8 else '9-64' if gap<=64 else '65+'
                                group=(family['house'],family['split'],seed,visibility['scale'],gap_bin,row['task_id'])
                                errors=dict(visible_witness_missed=anchor is not None and t==first and not bool(prediction[1]),
                                    truly_absent_false_positive=not z[1] and bool(prediction[1]),
                                    retention_lost_after_detected_witness=anchor is not None and first is not None and t-first>8 and first_detected and not bool(prediction[1]),
                                    wrong_terminal_readiness=bool(prediction[3])!=bool(z[3]),
                                    wrong_action_given_exact_state=bool(row['action_masks'][t]) and bool(torch.equal(prediction,truth)) and int(actions[i,t])!=row['targets'][t])
                                error_counts[group].update({k:int(v) for k,v in errors.items()});error_counts[group]['causal_decisions']+=1
                                events.append(dict(house=family['house'],family=family['family_id'],split=family['split'],seed=seed,
                                    history=row['history_id'],task=row['task_id'],continuation=row['continuation'],step=t,
                                    target=z,probability=p[i,t].tolist(),gap=gap,gap_bin=gap_bin,
                                    stratum='absent' if not z[1] else 'first_witness' if t==first else 'long_retention' if t-first>8 else 'recent',
                                    terminal=z[3],trace_path=row['trace_path'],errors=errors,visibility_role=anchor or 'terminal',**visibility))
                    # Mechanism comparisons require actual matched windows AND geometry.
                    cases=family.get('mechanism_cases',[])
                    if not cases:excluded.append(dict(family=family['family_id'],seed=seed,arm=arm,reason='NO_PREREGISTERED_GEOMETRY_WINDOW_AND_SHAM_MATCH'));continue
                    for case in cases:
                        predictions={}
                        for mode in ('correct','wrong','sham','zero'):
                            idx=case[mode] if mode!='zero' else case['correct'];mem=states[idx[0],idx[1]].unsqueeze(0)
                            if mode=='zero':mem=torch.zeros_like(mem)
                            current=cache['features'][case['feature']].unsqueeze(0);native=cache['logits'][case['feature']].unsqueeze(0)
                            predictions[mode]=int(tensor_indices(net.action_logits(mem,native,current)))
                        interventions.append(dict(family=family['family_id'],house=family['house'],split=family['split'],seed=seed,arm=arm,case=case,predictions=predictions))
    write(run/'EVENT_STATE_DIAGNOSIS.json',dict(rows=events,only_trained_B2_state_head=True,scope='correlated teacher-route causal decisions, not independent rollouts'))
    write(run/'ORDINARY_ACTION_REVIEW.json',dict(rows=ordinary_rows,
        scope='Previously exposed ordinary CHECK action accuracy; no new R2R closed-loop SR, no checkpoint selection.'))
    write(run/'QUERY_DIAGNOSIS.json',dict(rows=queries,B1_query_head_not_evaluated=True,
        scope='Auxiliary diagnostics only; cannot establish closed-loop increment. B2 always uses compositional program state, never an untrained result head.'))
    target=[r for r in interventions if r['split']==diagnostic_split and r['arm']=='Ours']
    mechanism=[r for r in target if r['case']['endpoint']=='mechanism'];control=[r for r in target if r['case']['endpoint']=='control']
    accurate=lambda mode:sum(r['predictions'][mode]==r['case']['target'] for r in mechanism)
    supports=bool(mechanism and control and accurate('correct')>accurate('wrong') and accurate('sham')>=accurate('correct') and all(r['predictions']['correct']==r['predictions']['wrong']==r['predictions']['sham'] for r in control))
    identifiable=bool(mechanism and control)
    write(run/'MEMORY_INTERVENTIONS.json',dict(status='READ_ONLY_ACTION_INTERVENTIONS_COMPLETE' if identifiable else 'UNIDENTIFIABLE',diagnostic_split=diagnostic_split,
        rows=interventions,excluded=excluded,supports_history_specific_effect=supports,
        mechanism_counts={mode:accurate(mode) for mode in ('correct','wrong','sham','zero')},diagnostic_mechanism_N=len(mechanism),matched_control_N=len(control),
        reason='A positive continuation decision requires identifiable matched task_T controls and registered correct/wrong/sham evidence; absent matching is not a negative model result.'))
    write(run/'FAILURE_LOCALIZATION.json',dict(state_diagnosis='EVENT_STATE_DIAGNOSIS.json',
        by_house_seed_scale_gap=[dict(house=k[0],split=k[1],seed=k[2],scale=k[3],gap=k[4],task=k[5],counts=dict(v)) for k,v in error_counts.items()],
        claims_not_supported=['larger base required','memory capacity lower bound','Ours independently improves R2R'],
        interpretation='Current-frame witness errors, retention errors and correct-state action errors must be separated; no architectural change is executed.'))

if __name__=='__main__':main(Path(sys.argv[1]))
