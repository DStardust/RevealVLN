"""Index actual existing event supervision and difficult negatives; collect no new data."""
from collections import Counter,defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import HERE,BASE,LINE,CPU,read,sha,immutable,legacy

def main(out):
    out=out.resolve();data=read(CPU/'DATA.json');raw={f['family_id']:f for f in data['raw_families']}
    rows={};traces={};verified={}
    for family in data['families']:
        f=raw[family['family_id']];compiler=legacy.Compiler(**f['compiler'])
        for seq in family['sequences']:
            ref=seq['source_trace'];path=LINE/ref['path']
            if path not in verified:
                if sha(path)!=ref['sha256']:raise ValueError('SOURCE_TRACE_CHANGED')
                verified[path]=ref['sha256'];traces[path]=read(path)
            trace=traces[path];atoms=compiler.atoms(trace['observations'])
            for t,index in enumerate(seq['features']):
                for bit,role in [(0,'anchor'),(1,'terminal')]:
                    if not seq['event_masks'][t][bit]:continue
                    target=seq['event_targets'][t][bit]
                    if atoms[t][role] is None or int(bool(atoms[t][role]))!=target:raise ValueError('EVENT_LABEL_CHANGED')
                    key=(index,role);ids=f['compiler']['eligible'][role];obs=trace['observations']
                    before={i:obs[t-1]['pixels'].get(str(i),0) for i in ids} if t else {i:0 for i in ids}
                    now={i:obs[t]['pixels'].get(str(i),0) for i in ids}
                    pair_pixels=max((min(before[i],now[i]) for i in ids),default=0)
                    if target:kind='positive_near' if pair_pixels<1024 else 'positive_clear'
                    elif not t:kind='initial_single_observation'
                    elif any(before[i]>=256 for i in ids) and any(now[i]>=256 for i in ids):kind='different_instance_across_frames'
                    elif any(max(before[i],now[i])>=256 for i in ids):kind='single_frame_witness'
                    elif any(max(before[i],now[i])>0 for i in ids):kind='below_pixel_threshold'
                    else:kind='no_target_pixels'
                    spec=f['roles'][role];lexeme=[spec['raw_match']['value'],spec['room']]
                    if key in rows:
                        r=rows[key]
                        if r['target']!=target or r['split']!=family['split'] or r['negative_or_scale']!=kind or r['lexeme']!=lexeme:raise ValueError('INPUT_LABEL_OR_LEXEME_CONFLICT')
                    else:
                        r=dict(feature_index=index,input_key=data['features'][index]['key'],role=role,target=target,
                            split=family['split'],negative_or_scale=kind,lexeme=lexeme,house=family['house'],
                            parent_families=[],physical_traces=[],example=dict(path=ref['path'],sha256=ref['sha256'],step=t),
                            policy_input_fields='Original causal feature only; role/lexeme/truth/provenance fields are loss/audit only.')
                        rows[key]=r
                    if family['parent_family_id'] not in r['parent_families']:r['parent_families'].append(family['parent_family_id'])
                    if ref['path'] not in r['physical_traces']:r['physical_traces'].append(ref['path'])
    records=sorted(rows.values(),key=lambda r:(r['split'],r['house'],r['feature_index'],r['role']))
    summaries=[]
    for split in ('FIT','DEV'):
        for role in ('anchor','terminal'):
            selected=[r for r in records if r['split']==split and r['role']==role]
            summaries.append(dict(split=split,role=role,unique_inputs=len(selected),positive=sum(r['target'] for r in selected),negative=sum(not r['target'] for r in selected),
                types=dict(Counter(r['negative_or_scale'] for r in selected)),houses=len({r['house'] for r in selected}),
                target_word_room_combinations=sorted({tuple(r['lexeme']) for r in selected})))
    fit_lexemes={tuple(r['lexeme']) for r in records if r['split']=='FIT'};fit_words={r['lexeme'][0] for r in records if r['split']=='FIT'}
    coverage=[]
    for house,word,room in sorted({(r['house'],*r['lexeme']) for r in records if r['split']=='DEV'}):
        selected=[r for r in records if r['split']=='DEV' and r['house']==house and r['lexeme']==[word,room]]
        coverage.append(dict(house=house,word=word,room=room,word_seen_in_fit=word in fit_words,word_room_seen_in_fit=(word,room) in fit_lexemes,
            n=len(selected),positive=sum(r['target'] for r in selected)))
    immutable(out/'EVENT_INDEX.json',dict(records=records,source_data_sha256=sha(CPU/'DATA.json'),
        physical_executions_created=0,new_houses=0,new_images=0,optimizer_updates=0,
        split_policy='FIT and exposed DEV remain separate. Derived windows follow all parent families and physical traces. This index is not additional independent data.',
        intended_use='FIT event diagnostics and candidate sampler input; no new training authorization inferred from this file.'))
    immutable(out/'EVENT_SUPPORT.json',dict(summary=summaries,dev_lexical_coverage=coverage,source_traces=len(verified),
        distinct_feature_role_labels=len(records),new_samples_collected=0,
        note='Counts are unique model-input/role keys; histories and houses remain correlated. Initial observations cannot establish SEE2. Pixel thresholds and instance consistency unchanged.'))
    print(read(out/'EVENT_SUPPORT.json'))

if __name__=='__main__':main(Path(sys.argv[1]))
