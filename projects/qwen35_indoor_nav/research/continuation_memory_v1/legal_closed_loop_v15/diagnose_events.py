"""Read-only localization of B2 state errors along actual causal prefixes."""
from pathlib import Path
import sys
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import objective as o
c=o.c


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data=c.read(HERE/'DATA.json')
    cache=torch.load(HERE/'features_run_001/FEATURES.pt',map_location='cpu',weights_only=True)
    train=HERE/'train_run_001';rows=[]
    with torch.inference_mode():
        for size in ('S1','L3'):
            for seed in (1209,1210,1211):
                tag=f'{size}_B2_{seed}';record=c.read(train/(tag+'_RESULT.json'))
                net=o.initialize(seed)
                net.load_state_dict(torch.load(train/(tag+'_MEMORY.pt'),map_location='cpu',weights_only=True));net.eval()
                assert c.model_identity(net)['sha256']==record['final_state_sha256']
                for family in data['families']:
                    b=o.batch(family)
                    states,_=net.encode(cache['features'][b['indices']])
                    probability=net.state_head(states.flatten(2)).sigmoid()
                    for i,prefix in enumerate(family['prefixes']):
                        if prefix['task_id']=='task_T':continue
                        target=prefix['state_targets'];n=len(target)
                        first=next((t for t,z in enumerate(target) if z[1]),None)
                        values=probability[i,:n,1].tolist()
                        rows.append(dict(model=tag,family_id=family['family_id'],split=family['split'],
                            history_id=prefix['history_id'],task_id=prefix['task_id'],decisions=n,
                            first_anchor_witness=first,anchor_present=bool(target[-1][1]),
                            anchor_probability_at_first_witness=values[first] if first is not None else None,
                            anchor_probability_at_cutoff=values[-1],
                            positive_before_actual_witness=any(x>.5 for x in values[:first]) if first is not None else any(x>.5 for x in values),
                            actual_anchor_state=[z[1] for z in target],predicted_anchor_probability=values))
                assert c.model_identity(net)['sha256']==record['final_state_sha256']
                print(tag,flush=True)
    summaries=[]
    for size in ('S1','L3'):
        fit_ids=c.read(HERE/'TRAIN_PROTOCOL.json')['data_sizes'][size]
        for seed in (1209,1210,1211):
            for scope in ('trained_fit','untrained_fit','check'):
                selected=[row for row in rows if row['model']==f'{size}_B2_{seed}' and
                    ('check' if row['split']=='check' else 'trained_fit' if row['family_id'] in fit_ids else 'untrained_fit')==scope]
                if not selected:continue
                present=[x for x in selected if x['anchor_present']];absent=[x for x in selected if not x['anchor_present']]
                summaries.append(dict(size=size,seed=seed,scope=scope,present_prefixes=len(present),absent_prefixes=len(absent),
                    present_first_witness_detected=sum(x['anchor_probability_at_first_witness']>.5 for x in present),
                    present_cutoff_detected=sum(x['anchor_probability_at_cutoff']>.5 for x in present),
                    present_detected_first_but_lost_by_cutoff=sum(x['anchor_probability_at_first_witness']>.5 and x['anchor_probability_at_cutoff']<=.5 for x in present),
                    absent_cutoff_false_positive=sum(x['anchor_probability_at_cutoff']>.5 for x in absent)))
    c.write(HERE/'EVENT_STATE_DIAGNOSIS.json',dict(summaries=summaries,rows=rows,
        optimizer_updates=0,GPU_used=False,checkpoint_parameters_unchanged=True,
        selection='All six frozen B2 models, all ordered-task prefixes; fixed 0.5 classification threshold, no refitting or threshold search.',
        scope='Learned B2 exact-state readout on cached causal features. Does not prove information is absent from frozen features or localize errors uniquely to encoder versus memory versus readout.',
        independent_units='One FIT house, one CHECK house; neutral/history/task variants and seeds are correlated.'),True)


if __name__=='__main__':main()
