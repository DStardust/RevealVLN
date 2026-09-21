"""Merge verified inputs, retaining parent groups and original split assignments."""
import copy
from collections import Counter
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *


def main(run):
    if (run/'DATA.json').exists():
        binding=read(run/'DATA_BINDING.json')
        if sha(run/'DATA.json')!=binding['data_sha256']:raise ValueError('DATA_CHANGED')
        for name,value in binding['sources'].items():
            if sha(LINE/name)!=value:raise ValueError('SOURCE_DATA_CHANGED')
        return
    bindings={};features=[];lookup={};owners={};contents={};families=[];raw=[]
    datasets=[(DATA,V16/'identifiable_data_v1/runs/collect_002/DATASET.json','terminal_present'),
              (COVERAGE/'verification_000',COVERAGE/'DATASET.json','terminal_absent')]
    for folder,raw_path,stratum in datasets:
        names=['POLICY_INPUTS.json','SUPERVISION.json','FAMILIES.json','CONTENT_MANIFEST.json','RESULT.json']
        for path in [folder/n for n in names]+[raw_path]:bindings[str(path.relative_to(LINE))]=sha(path)
        result=read(folder/'RESULT.json')
        if not result['training_admission']:raise ValueError('DATA_NOT_ADMITTED')
        local_features=read(folder/'POLICY_INPUTS.json');records=read(folder/'SUPERVISION.json')
        local_families=read(folder/'FAMILIES.json');raw_by_id={f['family_id']:f for f in read(raw_path)['families']}
        groups={f['family_id']:dict(f,sequences=[]) for f in local_families};remap={};record_map={}
        for r in records:
            split=r['split']
            if split not in ('FIT','DEV'):raise ValueError('TEST_LEAK')
            for i in r['features']:
                item=local_features[i];key=item['key']
                if key in owners and owners[key]!=split:raise ValueError('CROSS_SPLIT_INPUT')
                owners[key]=split
                if key not in lookup:lookup[key]=len(features);features.append(dict(item,split=split))
                remap[i]=lookup[key]
        for old_index,r in enumerate(records):
            row=copy.deepcopy(r);row['features']=[remap[i] for i in r['features']]
            group=groups[r['family_id']];record_map[old_index]=(r['family_id'],len(group['sequences']))
            group['sequences'].append(row)
        for f in groups.values():
            source=copy.deepcopy(raw_by_id[f['family_id']]);parent=source.get('parent_family_id',source['family_id'])
            for case in f['mechanism_cases']:
                for key in ('correct','wrong','sham'):
                    fid,index=record_map[case[key][0]]
                    if fid!=f['family_id']:raise ValueError('CROSS_FAMILY_MECHANISM_INDEX')
                    case[key][0]=index
            f.update(parent_family_id=parent,stratum=stratum);source.update(parent_family_id=parent,stratum=stratum)
            families.append(f);raw.append(source)
        for item in read(folder/'CONTENT_MANIFEST.json'):
            if item['kind']=='rgb':contents.setdefault('sha256:'+item['pixel_sha256'],dict(line_relative_path=item['path'],file_sha256=item['file_sha256']))
    parent_splits={}
    for f in families:
        p=f['parent_family_id'];s=f['split']
        if p in parent_splits and parent_splits[p]!=s:raise ValueError('PARENT_SPLIT_LEAK')
        parent_splits[p]=s
    if Counter(f['split'] for f in families)!=Counter(FIT=59,DEV=16):raise ValueError('REGISTERED_VARIANT_COUNT')
    if Counter(parent_splits.values())!=Counter(FIT=32,DEV=8):raise ValueError('REGISTERED_PARENT_COUNT')
    natural=read(V16.parent/'natural_transfer_v9/DATA.json')
    if {r['row']['scene_group'] for r in natural['records'] if r['partition']=='fit'} & {f['house'] for f in families}:
        raise ValueError('ORDINARY_HOUSE_LEAK')
    immutable(run/'DATA.json',dict(features=features,contents=contents,families=families,raw_families=raw))
    immutable(run/'DATA_BINDING.json',dict(data_sha256=sha(run/'DATA.json'),sources=bindings,
        unique_inputs=len(features),variants=75,independent_parent_families=40,FIT_parents=32,DEV_parents=8,
        exposed_DEV_houses=1,labels_changed=False,new_independent_test_houses=0))

if __name__=='__main__':
    import sys
    run=Path(sys.argv[1]);main(run)
    import train
    from evaluate_continuations import registry_value
    data=read(run/'DATA.json');cfg=read(run/'PROTOCOL.json');fit=train.training_families(data,cfg)
    immutable(run/'FIT_AUXILIARY_WEIGHTS.json',train.o.class_weights(fit))
    natural=read(V16.parent/'natural_transfer_v9/DATA.json')
    allowed=[i for i,r in enumerate(natural['records']) if r['partition']=='fit']
    (run/'train').mkdir(exist_ok=True)
    for seed in cfg['seeds']:
        old=read(V16.parent/'natural_transfer_v9/run_001'/f'SCHEDULE_{seed}.json')['ordinary_indices']
        immutable(run/'train'/f'SCHEDULE_{seed}.json',train.schedule_for_seed(fit,old,allowed,seed))
    immutable(run/'EVALUATION_REGISTRY.json',registry_value(data['raw_families'],cfg))
