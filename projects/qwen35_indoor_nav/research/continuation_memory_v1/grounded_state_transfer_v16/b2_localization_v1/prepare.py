"""Recover exact SEE2 events; deduplicate current windows and causal histories."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from collections import Counter


def main():
    OUT.mkdir(exist_ok=False);cfg=read(HERE/'PROTOCOL.json');sources={}
    for name,expected in read(HERE/'SOURCE_LOCK.json')['files'].items():
        if sha(HERE/name)!=expected:raise ValueError('DIAGNOSTIC_SOURCE_CHANGED')
    for path,expected in read(RUN/'SOURCE_LOCK.json')['files'].items():
        if sha(LINE/path)!=expected:raise ValueError('SOURCE_CHANGED:'+path)
    for path in [RUN/'DATA.json',RUN/'DATA_BINDING.json',RUN/'features/FEATURE_RESULT.json',RUN/'features/FEATURES.pt']:
        sources[str(path)]=sha(path)
    if sources[str(RUN/'DATA.json')]!=read(RUN/'DATA_BINDING.json')['data_sha256']:raise ValueError('DATA_CHANGED')
    if sources[str(RUN/'features/FEATURES.pt')]!=read(RUN/'features/FEATURE_RESULT.json')['file_sha256']:raise ValueError('CACHE_CHANGED')
    data=read(RUN/'DATA.json');raw={r['family_id']:r for r in data['raw_families']}
    for tag in read(RUN/'EVALUATION_REGISTRY.json')['models']:
        path=RUN/'train'/tag/'FINAL.pt';expected=read(path.parent/'RESULT.json')
        if sha(path)!=expected['checkpoint_sha256'] or expected['updates']!=1200:raise ValueError('HEAD_CHANGED')
        sources[str(path)]=sha(path)
    trace_cache={};events={};visits={};sequences=[];parents={};houses={}
    for f in data['families']:
        parent=f['parent_family_id'];split=f['split'];house=f['house']
        if parent in parents and parents[parent]!=split or house in houses and houses[house]!=split:raise ValueError('SPLIT_LEAK')
        parents[parent]=split;houses[house]=split;compiler=legacy.Compiler(**raw[f['family_id']]['compiler'])
        for ri,row in enumerate(f['sequences']):
            path=LINE/row['source_trace']['path']
            if path not in trace_cache:
                if sha(path)!=row['source_trace']['sha256']:raise ValueError('TRACE_CHANGED')
                trace_cache[path]=read(path);sources[str(path)]=row['source_trace']['sha256']
            trace=trace_cache[path]
            if not legacy.complete(trace):raise ValueError('INCOMPLETE_TRACE')
            atoms=compiler.atoms(trace['observations']);states=state_sequence(compiler,trace['observations'],row['task'])
            if states[:len(row['features'])]!=row['state_targets']:raise ValueError('STATE_RECOMPUTE')
            teacher=any(row['action_masks']);last=None;chain='';selected=[];visit_ids=[]
            for t,idx in enumerate(row['features']):
                ev=[int(bool(atoms[t][role])) for role in ('anchor','terminal')]
                if any(atoms[t][role] is None for role in ('anchor','terminal')):raise ValueError('UNKNOWN_EVENT')
                if ev[0]:last=t
                meta=dict(parent=parent,house=house,split=split,family=f['family_id'],stratum=f['stratum'],task=row['task'],history=row['history'],t=t,index=idx,
                    age=None if last is None else t-last,event=ev,state=states[t],cutoff=row['cutoff'])
                if row['task']=='task_A' and t>0:
                    key=(parent,idx)
                    if key in events and events[key]['event']!=ev:raise ValueError('SAME_INPUT_EVENT_LABEL_CONFLICT')
                    if key not in events:
                        scale=[]
                        for role in ('anchor','terminal'):
                            scale.append(max((min(int(trace['observations'][t]['pixels'].get(str(i),0)),int(trace['observations'][t-1]['pixels'].get(str(i),0))) for i in compiler.eligible[role]),default=0))
                        events[key]=dict(meta,scale=scale)
                if not teacher:continue
                chain=digest([chain,idx]);key=(parent,row['task'],chain)
                item=dict(meta,teacher_targets=[row['targets'][t]] if row['action_masks'][t] else [],action_mask=row['action_masks'][t],preservation=t<min(156,row['cutoff']+1),first_witness=ev[0]==1 and states[t][0]==0)
                if key in visits:
                    merge_teacher(visits[key],item)
                else:
                    item['row_id']=len(visits);visits[key]=item;selected.append([t,item['row_id']])
                visit_ids.append(visits[key]['row_id'])
            if teacher:sequences.append(dict(family=f['family_id'],sequence=ri,selected=selected,visit_ids=visit_ids))
    events=list(events.values());visits=list(visits.values())
    fitkeys={r['index'] for r in events if r['split']=='FIT'};devkeys={r['index'] for r in events if r['split']=='DEV'}
    if fitkeys&devkeys:raise ValueError('INPUT_SPLIT_LEAK')
    immutable(OUT/'DATA.json',dict(events=events,visits=visits,sequences=sequences))
    immutable(OUT/'BINDING.json',dict(sources=sources,protocol_sha256=sha(HERE/'PROTOCOL.json'),sources_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),data_sha256=sha(OUT/'DATA.json')))
    immutable(OUT/'COUNTS.json',dict(unique_event_inputs=len(events),unique_causal_states=len(visits),teacher_sequences=len(sequences),teacher_causal_states=sum(v['action_mask'] for v in visits),teacher_multiaction_states=sum(len(v['teacher_targets'])>1 for v in visits),parents=dict(Counter(parents.values())),houses=dict(Counter(houses.values())),
        events_by_split={split:dict(n=sum(r['split']==split for r in events),positive=[sum(r['event'][b] for r in events if r['split']==split) for b in range(2)]) for split in ('FIT','DEV')},
        scope='Original exposed FIT/DEV houses; variants and repeated windows are not independent families. Full teacher suffixes used only at their causal time.'))
    write(OUT/'STATUS.json',dict(status='PREPARED',phase='prepare'))

if __name__=='__main__':main()
