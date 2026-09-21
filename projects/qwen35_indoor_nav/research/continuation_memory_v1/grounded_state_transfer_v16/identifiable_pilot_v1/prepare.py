"""Bind already verified data; no new labels, families or TEST access."""
from collections import Counter
from shared import *

def main(run):
    source=read(DATA/'EVIDENCE_MANIFEST.json')
    for name,expected in source['files'].items():
        if sha(DATA.parent/name)!=expected:raise ValueError('DATA_EVIDENCE_CHANGED:'+name)
    info=read(DATA/'RESULT.json')
    if not info['target_complete'] or info['counts']['families']!=40:raise ValueError('DATA_NOT_COMPLETE')
    features=read(DATA/'POLICY_INPUTS.json');records=read(DATA/'SUPERVISION.json');fams=read(DATA/'FAMILIES.json')
    groups={f['family_id']:dict(f,sequences=[]) for f in fams};owners={}
    for row in records:
        if row['split'] not in ('FIT','DEV'):raise ValueError('NONDEVELOPMENT_INPUT')
        groups[row['family_id']]['sequences'].append(row)
        for i in row['features']:
            if i in owners and owners[i]!=row['split']:raise ValueError('SPLIT_INPUT_LEAK')
            owners[i]=row['split']
    for i,row in enumerate(features):row['split']=owners[i]
    contents={}
    for r in read(DATA/'CONTENT_MANIFEST.json'):
        if r['kind']=='rgb':contents['sha256:'+r['pixel_sha256']]=dict(line_relative_path=r['path'],file_sha256=r['file_sha256'])
    raw=read(V16/'identifiable_data_v1/runs/collect_002/DATASET.json')['families']
    if Counter(f['split'] for f in raw)!=Counter(FIT=32,DEV=8):raise ValueError('REGISTERED_DATA_COUNT')
    ordinary=read(V16.parent/'natural_transfer_v9/DATA.json')
    fit_ordinary={r['row']['scene_group'] for r in ordinary['records'] if r['partition']=='fit'}
    if fit_ordinary & {f['house'] for f in raw}:raise ValueError('ORDINARY_MEMORY_HOUSE_OVERLAP')
    immutable(run/'DATA.json',dict(features=features,contents=contents,families=list(groups.values()),raw_families=raw))
    immutable(run/'DATA_BINDING.json',dict(source_result_sha256=sha(DATA/'RESULT.json'),source_manifest_sha256=sha(DATA/'EVIDENCE_MANIFEST.json'),
        records_sha256=sha(DATA/'SUPERVISION.json'),data_sha256=sha(run/'DATA.json'),ordinary_data_sha256=sha(V16.parent/'natural_transfer_v9/DATA.json'),
        scope='Previously exposed development houses; no ordinary benchmark or generalization claim',labels_changed=False))
