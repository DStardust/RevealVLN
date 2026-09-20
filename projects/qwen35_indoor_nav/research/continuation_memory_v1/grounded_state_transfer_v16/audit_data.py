"""Recheck raw RGB/semantic content and record actual coverage, without model scores."""
import collections
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
from evaluator_v16 import legacy,state_sequence

def main(run):
    data=read(run/'DATA.json');verified={};counts=collections.Counter();witnesses=[];initial_states=[];roles=collections.defaultdict(set)
    for family in data['raw_families']:
        compiler=legacy.Compiler(**family['compiler']);root=LINE/family['content_root']
        for role,signature in compiler.roles.items():roles[family['split']].add(tuple(signature))
        for cell,source in family['traces'].items():
            path=LINE/source['path']
            if sha(path)!=source['sha256']:raise ValueError('TRACE_FILE_CHANGED')
            trace=read(path);events=compiler.atoms(trace['observations']);history=cell.split('__')[0];cut=len(family['histories'][history])
            for t,observation in enumerate(trace['observations']):
                arrays={}
                for kind in ('rgb','semantic'):
                    key=(str(root),observation[kind+'_hash'],kind);p=root/(key[1]+'.'+kind+'.npy')
                    if key not in verified:
                        a=np.load(p,allow_pickle=False)
                        expected=(224,224,3) if kind=='rgb' else (224,224)
                        dtype=np.uint8 if kind=='rgb' else np.uint32
                        if a.shape!=expected or a.dtype!=dtype or hashlib.sha256(a.tobytes()).hexdigest()!=key[1]:raise ValueError('RAW_CONTENT_ARRAY_MISMATCH')
                        verified[key]=dict(path=str(p.relative_to(LINE)),sha256=sha(p),kind=kind)
                    if kind=='semantic':arrays[kind]=np.load(p,allow_pickle=False)
                ids,num=np.unique(arrays['semantic'],return_counts=True);pixels={str(int(k)):int(v) for k,v in zip(ids,num)}
                if pixels!=observation['pixels']:raise ValueError('SEMANTIC_PIXEL_COUNT_MISMATCH')
                counts['observations']+=1
                if cell.endswith('__C0'):
                    for role in compiler.roles:
                        ids=events[t][role]
                        prior=trace['observations'][t-1]['pixels'] if t else {}
                        max_pixels=max((min(pixels.get(str(i),0),prior.get(str(i),0)) for i in compiler.eligible[role]),default=0)
                        witnesses.append(dict(family=family['family_id'],house=family['house'],split=family['split'],history=history,step=t,role=role,
                            witnessed=bool(ids),max_same_instance_min_pixels=max_pixels,
                            scale='clear' if max_pixels>=1024 else 'near_threshold' if max_pixels>=256 else 'below_threshold' if max_pixels else 'absent',
                            phase='prefix' if t<cut else 'takeover' if t==cut else 'suffix'))
            if cell.endswith('__C0'):
                initial_states.append(dict(family=family['family_id'],history=history,initial_rgb=trace['observations'][0]['rgb_hash'],
                    state_A=state_sequence(compiler,trace['observations'],'task_A')[cut],state_B=state_sequence(compiler,trace['observations'],'task_B')[cut]))
            counts['complete_crossed_executions']+=1
    snap=LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001'
    index=snap/'TRAINING_INDEX.jsonl';protocol=read(LINE/'sft_acceptance/ordinary_expanded_v1/PROTOCOL_FILESTORE.json')
    if sha(index)!=protocol['snapshot']['training_index_sha256']:raise ValueError('KNOWN_BASE_TRAINING_INDEX_CHANGED')
    base_houses={json.loads(row)['scene_group'] for row in index.read_text().splitlines()}
    all_houses={f['house'] for f in data['raw_families']}
    base_audit=dict(index_path=str(index.relative_to(LINE)),index_sha256=sha(index),known_training_houses=sorted(base_houses),
        memory_house_overlap=sorted(all_houses&base_houses),claim='Only this registered navigation training pool is audited; no assertion about general vision-language pretraining exposure.')
    write(run/'BASE_TRAINING_SCENE_AUDIT.json',base_audit)
    by_initial=collections.defaultdict(set)
    for row in initial_states:by_initial[(row['family'],row['initial_rgb'])].add((row['state_A'][1],row['state_B'][1]))
    audit=dict(status='RAW_CONTENT_AND_LABEL_INPUTS_RECHECKED',counts=dict(counts),unique_arrays=len(verified),
        content_entries=list(verified.values()),initial_input_to_state=[dict(family=k[0],initial_rgb=k[1],states=sorted(v)) for k,v in by_initial.items()],
        semantic_vocabulary={k:[list(x) for x in sorted(v)] for k,v in roles.items()},
        TEST_roles_absent_from_FIT=[list(x) for x in sorted(roles['TEST']-roles['FIT'])],witness_strata=dict(collections.Counter(r['scale'] for r in witnesses)),
        no_method_scores_read=True,old_training_admission_modified=False)
    write(run/'RAW_DATA_AUDIT.json',audit)
    write(run/'VISIBLE_EVENT_COVERAGE.json',dict(rows=witnesses))
    print(json.dumps(dict(arrays=len(verified),executions=counts['complete_crossed_executions'],base_overlap=base_audit['memory_house_overlap'])))

if __name__=='__main__':main(Path(sys.argv[1]))
