"""Dense causal cuts from actually executed suffixes; no invented cross cells."""
from collections import Counter
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
from evaluator_v16 import legacy, state_sequence, query_contexts, compose, evaluate

def mechanism_cases(family):
    from continuation_service import exact_pose
    rows=family['sequences'];cells=family['cells'];prefixes=family['prefixes'];admission=family['teacher_admission']
    lookup={(p['history_id'],p['task_id']):i for i,p in enumerate(prefixes)}
    seq_lookup={(r['history_id'],r['task_id'],r['continuation']):i for i,r in enumerate(rows)}
    traces={r['trace_path']:read(LINE/r['trace_path']) for r in rows};cases=[];excluded=[]
    def same(i,t,j,u):
        return rows[i]['features'][t]==rows[j]['features'][u] and exact_pose(traces[rows[i]['trace_path']]['observations'][t]['pose'],traces[rows[j]['trace_path']]['observations'][u]['pose'])
    for task in ('task_A','task_B'):
        choices=[admission['selected_cells'].get(str(lookup[(h,task)]),[]) for h in ('H_A','H_B','H_A_R','H_B_R')]
        if any(len(x)!=1 for x in choices):excluded.append(dict(task=task,reason='NO_UNIQUE_TESTED_PASS_TEACHER'));continue
        a,b,sa,sb=[cells[x[0]]['sequence'] for x in choices]
        for offset in range(min(len(rows[i]['features'])-rows[i]['cutoff'] for i in (a,b,sa,sb))):
            ta,tb,tsa,tsb=[rows[i]['cutoff']+offset for i in (a,b,sa,sb)]
            if not same(a,ta,b,tb):excluded.append(dict(task=task,reason='WINDOW_OR_GEOMETRY_DIFFERS_BEFORE_FORK'));break
            if rows[a]['targets'][ta]==rows[b]['targets'][tb]:continue
            valid=same(a,ta,sa,tsa) and same(b,tb,sb,tsb)
            valid &= rows[a]['state_targets'][ta]==rows[sa]['state_targets'][tsa] and rows[b]['state_targets'][tb]==rows[sb]['state_targets'][tsb]
            valid &= rows[a]['state_targets'][ta][1]!=rows[b]['state_targets'][tb][1]
            if not valid:excluded.append(dict(task=task,reason='SHAM_OR_STATE_MATCH_MISSING'));break
            for i,t,j,u,sh,st in ((a,ta,b,tb,sa,tsa),(b,tb,a,ta,sb,tsb)):
                cases.append(dict(task=task,endpoint='mechanism',correct=[i,t],wrong=[j,u],sham=[sh,st],feature=rows[i]['features'][t],target=rows[i]['targets'][t]))
                ci,cj,cs=[seq_lookup[(rows[k]['history_id'],'task_T',rows[k]['continuation'])] for k in (i,j,sh)]
                if same(ci,t,cj,u) and same(ci,t,cs,st):
                    cases.append(dict(task='task_T',endpoint='control',correct=[ci,t],wrong=[cj,u],sham=[cs,st],feature=rows[ci]['features'][t],target=rows[ci]['targets'][t]))
                else:excluded.append(dict(task='task_T',reason='MATCHED_CONTROL_NOT_AVAILABLE'))
            break
    return cases,excluded

def build(run, require_complete=True):
    families=[f for p in sorted(run.glob('HOUSE_*.json')) for f in read(p)['families']]
    features=[];lookup={};contents={};built=[];sufficient=[];history_owners={};split_windows={}
    def feature(trace,instruction,t,root,split):
        row=dict(instruction=instruction,rgb_refs=['sha256:'+o['rgb_hash'] for o in trace['observations'][max(0,t-1):t+1]],
                 executed=[{'F':'move_forward','L':'turn_left','R':'turn_right'}[a] for a in trace['actions'][max(0,t-8):t]])
        key=digest(row)
        if key not in lookup:
            lookup[key]=len(features);features.append(dict(key=key,split=split,**row))
            for ref in row['rgb_refs']:
                if ref not in contents:
                    p=LINE/root/(ref[7:]+'.rgb.npy');contents[ref]=dict(line_relative_path=str(p.relative_to(LINE)),file_sha256=sha(p))
        elif split_windows[key]!=split:raise ValueError('CROSS_SPLIT_DERIVED_INPUT')
        split_windows[key]=split
        return lookup[key]
    for family in families:
        compiler=legacy.Compiler(**family['compiler']);prefixes=[];cells=[];sequences=[]
        for h,actions in family['histories'].items():
            cut=len(actions);ref=read(LINE/family['traces'][h+'__C0']['path'])
            physical=digest(dict(actions=actions,rgb=[o['rgb_hash'] for o in ref['observations'][:cut+1]]))
            if physical in history_owners and history_owners[physical]!=family['split']:raise ValueError('CROSS_SPLIT_PHYSICAL_HISTORY')
            history_owners[physical]=family['split']
            for task in ('task_A','task_B','task_T'):
                instruction=family['terminal_instruction'] if task=='task_T' else compiler.tasks[task]['instruction']
                state=state_sequence(compiler,ref['observations'],task)
                p=len(prefixes)
                prefixes.append(dict(history_id=h,task_id=task,
                    features=[feature(ref,instruction,t,family['content_root'],family['split']) for t in range(cut+1)],
                    state_targets=state[:cut+1],state_masks=[1]*(cut+1)))
                for q in ('C0','C_A','C_B'):
                    source=family['traces'][h+'__'+q];path=LINE/source['path']
                    if sha(path)!=source['sha256']:raise ValueError('RAW_TRACE_CHANGED')
                    trace=read(path);result=evaluate(compiler,trace,task,cut)
                    valid=legacy.complete(trace) and len(trace['actions'])<=500
                    y=int(result['safe_v16_label']=='PASS')
                    states=state_sequence(compiler,trace['observations'],task)
                    ids=[feature(trace,instruction,t,family['content_root'],family['split']) for t in range(len(trace['actions']))]
                    queries=query_contexts(compiler,trace,task)
                    reconstruction=[int(compose(states[t],queries[t])) for t in range(len(ids))]
                    if valid and any(z!=y for z in reconstruction):raise ValueError('EXACT_STATE_QUERY_LABEL_DISAGREEMENT')
                    sufficient.append(dict(family=family['family_id'],history=h,task=task,query=q,cuts=len(ids),valid=valid,all_reconstructed=valid and all(z==y for z in reconstruction)))
                    index=len(sequences)
                    seq=dict(features=ids,targets=['FLRS'.index(a) for a in trace['actions']],
                        state_targets=states[:len(ids)],state_masks=[int(valid)]*len(ids),
                        query_contexts=queries,y=[y]*len(ids),query_masks=[int(valid)]*len(ids),
                        action_masks=[0]*len(ids),cutoff=cut,history_id=h,task_id=task,continuation=q,
                        trace_path=source['path'],trace_sha256=source['sha256'])
                    sequences.append(seq)
                    tail=[dict(feature=ids[t],target=seq['targets'][t],mask=int(valid and y),step=t) for t in range(cut,len(ids))]
                    cells.append(dict(prefix=p,tail=tail,y=y,mask=int(valid),query_context=queries[cut],continuation_id=q,sequence=index))
        f=dict(family_id=family['family_id'],house=family['house'],split=family['split'],prefixes=prefixes,cells=cells,sequences=sequences)
        teacher=load('v16_real_teacher',HERE.parent/'cost_teacher_v13/teacher.py');admission=teacher.admit(f)
        for cell,masks in zip(cells,admission['masks']):
            seq=sequences[cell['sequence']];seq['action_masks'][seq['cutoff']:]=masks
        # Shared actual recovery actions before takeover; perturbations are inputs, never teacher targets.
        owned=set()
        for cell in cells:
            seq=sequences[cell['sequence']];h=seq['history_id'];rec=family['recovery'].get(h)
            if not rec or not cell['mask'] or not cell['y']:continue
            for t in range(rec['perturbation_start']+len(rec['actions']),seq['cutoff']):
                key=(tuple(seq['features'][:t+1]),seq['targets'][t])
                if key not in owned:seq['action_masks'][t]=1;owned.add(key)
        f['teacher_admission']=admission
        f['mechanism_cases'],f['mechanism_excluded']=mechanism_cases(f)
        built.append(f)
        write(run/'DATA_BUILD_PROGRESS.json',dict(completed_families=len(built),total_families=len(families),features=len(features)))
    count=Counter(f['split'] for f in built)
    complete=dict(count)==dict(FIT=16,DEV=2,TEST=8)
    if require_complete and not complete:raise ValueError('REGISTERED_FAMILY_COUNT_SHORTFALL: '+str(count))
    ordinary=read(HERE.parent/'natural_transfer_v9/DATA.json');ordinary_houses={r['row']['scene_group'] for r in ordinary['records'] if r['partition']=='fit'}
    memory_houses={f['house'] for f in built}
    if ordinary_houses & memory_houses:raise ValueError('ORDINARY_MEMORY_HOUSE_LEAK')
    data=dict(families=built,features=features,contents=contents,raw_families=families,
        training_admission='V16_REGISTERED_SCOPE_ONLY' if complete else 'FIT_DEV_GENERATOR_NUMERICAL_PILOT_ONLY',old_admission_flags_modified=False,
        policy_fields=['instruction','last2_RGB','last8_actual_motion_actions'],future_queries_only_reader=True)
    immutable(run/'DATA.json',data)
    immutable(run/'SPLIT_AUDIT.json',dict(houses={s:sorted({f['house'] for f in built if f['split']==s}) for s in count},
        ordinary_overlap=[],derived_cross_split_overlap=[],physical_histories=len(history_owners),
        template_families=read(HERE/'DATA_MANIFEST.json')['template_families'],base_training_unseen_claim=False))
    immutable(run/'QUERY_STATE_SUFFICIENCY.json',dict(rows=sufficient,exact_state_sufficient_for_all_valid_cells=all(r['all_reconstructed'] for r in sufficient if r['valid']),query_not_more_information=True))
    lineage=[];ancestor=run
    while True:
        attempts=c.records(ancestor/'COLLECTION_ATTEMPTS.jsonl') if (ancestor/'COLLECTION_ATTEMPTS.jsonl').exists() else []
        lineage.append(dict(run=str(ancestor.relative_to(HERE)),started=sum(r['status']=='START' for r in attempts),
            certified=sum(r['status']=='CERTIFIED' for r in attempts),rejected=sum(r['status']=='REJECTED' for r in attempts)))
        if not (ancestor/'REUSED_PHYSICAL_ASSETS.json').exists():break
        ancestor=Path(read(ancestor/'REUSED_PHYSICAL_ASSETS.json')['source'])
    counts=dict(families=len(built),by_split=dict(count),physical_histories=len(history_owners),
        physical_executions=sum(len(f['traces']) for f in families),cross_labels=sum(len(f['cells']) for f in built),
        causal_feature_windows=len(features),dense_state_cut_supervisions=sum(sum(s['state_masks']) for f in built for s in f['sequences']),
        dense_query_cut_supervisions=sum(sum(s['query_masks']) for f in built for s in f['sequences']),
        action_owners=sum(sum(s['action_masks']) for f in built for s in f['sequences']),
        collection_lineage=lineage,proposal_attempts=sum(r['started'] for r in lineage),
        registered_scale_complete=complete)
    immutable(run/'COUNTS.json',counts)
    immutable(run/'MECHANISM_REGISTRY.json',dict(families=[dict(family=f['family_id'],split=f['split'],cases=f['mechanism_cases'],excluded=f['mechanism_excluded']) for f in built],
        model_scores_read=False,matching='exact current window and physical pose, actual teacher conflict, state-matched recovery donor'))
    return data

if __name__=='__main__':build(Path(sys.argv[1]))
