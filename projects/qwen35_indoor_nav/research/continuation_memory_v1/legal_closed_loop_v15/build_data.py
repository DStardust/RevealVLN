"""Build causal inputs and separate supervision from the audited raw families."""
import hashlib
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
sys.path.insert(0,str(LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15'))
from common import c,load
from task_controls import terminal_only


def main():
    run=LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008'
    audit=c.read(HERE/'raw_run_008_AUDIT.json');config=c.read(run/'CONFIG.json')
    assert len(audit['families'])==len(config['families'])==6
    compilers=load('v15_dataset_compiler',LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    features=[];lookup={};contents={};families=[]
    def feature(trace,instruction,t):
        row=dict(instruction=instruction,rgb_refs=['sha256:'+o['rgb_hash'] for o in trace['observations'][max(0,t-1):t+1]],
            executed=[{'F':'move_forward','L':'turn_left','R':'turn_right'}[a] for a in trace['actions'][max(0,t-8):t]])
        key=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key not in lookup:
            lookup[key]=len(features);features.append(dict(key=key,**row))
            for reference in row['rgb_refs']:
                if reference not in contents:
                    path=run/'content'/(reference[7:]+'.rgb.npy')
                    contents[reference]=dict(line_relative_path=str(path.relative_to(LINE)),file_sha256=c.sha(path))
        return lookup[key]
    for family in config['families']:
        checked=next(x for x in audit['families'] if x['family_id']==family['family_id'])
        assert checked['relations_pass'] and checked['raw_join_rechecked']
        compiler=compilers.Compiler(**family['compiler']);prefixes=[];cells=[]
        tasks={**{k:dict(v) for k,v in compiler.tasks.items()},'task_T':family['task_terminal_only']}
        for h,actions in family['candidate']['histories'].items():
            cut=len(actions);base=c.read(run/family['family_id']/(h+'__C0.json'))
            for task_id,task in tasks.items():
                events=compiler.atoms(base['observations']);states=[];seen=False
                for e in events[:cut+1]:
                    terminal=bool(e['terminal'])
                    if task_id=='task_T':state=[1,1,int(terminal),int(terminal)]
                    else:
                        anchor=bool(e[task['anchor']]);state=[int(seen),int(seen or anchor),int(terminal),int(seen and terminal)];seen=seen or anchor
                    states.append(state)
                indices=[feature(base,task['instruction'],t) for t in range(cut+1)]
                pindex=len(prefixes)
                prefixes.append(dict(history_id=h,task_id=task_id,features=indices,state_targets=states,state_masks=[1]*len(states)))
                for q,suffix in family['candidate']['continuations'].items():
                    trace=c.read(run/family['family_id']/(h+'__'+q+'.json'));events=compiler.atoms(trace['observations'])
                    label=terminal_only(compiler,trace) if task_id=='task_T' else compiler.evaluate(trace,task_id)
                    assert label in ('pass','fail')
                    tail=[dict(step=t,feature=feature(trace,task['instruction'],t),target='FLRS'.index(trace['actions'][t]),mask=int(label=='pass'))
                          for t in range(cut,len(trace['actions']))]
                    context=dict(stop_at_cutoff=len(trace['observations'])-1==cut,stops=trace['actions'][-1]=='S',
                        suffix_anchor_before_final=(True if task_id=='task_T' else any(bool(e[task['anchor']]) for e in events[cut+1:-1])),
                        terminal_at_final=bool(events[-1]['terminal']))
                    cells.append(dict(audit_cell_id=h+'__'+task_id+'__'+q,prefix=pindex,query=0,
                        y=int(label=='pass'),mask=1,tail=tail,query_context=context,continuation_id=q))
        for task_id in tasks:
            selected=[p for p in prefixes if p['task_id']==task_id]
            assert len({p['features'][-1] for p in selected})==1
        families.append(dict(family_id=family['family_id'],house=family['house'],split=family['partition'],
            prefixes=prefixes,cells=cells,queries=[[0]],source_raw_run=str(run.relative_to(LINE)),
            new_raw_physical_controls_verified=True,original_legacy_assets_relabelled=False))
        print(family['family_id'],len(features),flush=True)
    data=dict(families=families,features=features,contents=contents,query_vocabulary=['STRUCTURED_QUERY_ONLY'],
        scope='New raw physical SEE2 debug data, three FIT families in one house and three CHECK families in a different house. One-house CHECK is not broad generalization.',
        terminal_control_context='For task_T no prerequisite is required; suffix_anchor_before_final encodes unconditional prerequisite satisfaction, not an observed anchor event. The task and suffix determine this field before reference to history or Y. Query remains training-reader-only.',
        data_quality_audit_sha256=c.sha(HERE/'raw_run_008_AUDIT.json'),base_or_encoder_trained=False)
    c.write(HERE/'DATA.json',data,True)
    c.write(HERE/'DATA_SUMMARY.json',dict(families=len(families),features=len(features),unique_rgb=len(contents),
        supervised_cells=sum(len(f['cells']) for f in families),house_split={f['house']:f['split'] for f in families},
        data_sha256=c.sha(HERE/'DATA.json'),model_results_available=False),True)


if __name__=='__main__':main()
