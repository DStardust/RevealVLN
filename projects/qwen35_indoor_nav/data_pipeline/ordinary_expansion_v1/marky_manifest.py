"""Stream official generated English train annotations, FIT-only; not CE ready."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def execute():
    out=HERE/'marky_manifest_v1';out.mkdir(exist_ok=False)
    source=HERE/'marky_acquisition_v1/rxr_marky_train_guide.jsonl.gz'
    fit=set(json.loads((HERE.parent/'ordinary_fullscale_source_v1/SPLIT_FREEZE.json').read_text())['FIT'])
    seen=set();ids=set();counts=collections.Counter();houses=collections.Counter();route_lengths=collections.Counter()
    with gzip.open(source,'rt',encoding='utf-8') as inp,(out/'OFFICIAL_ENGLISH_FIT_SOURCE.jsonl').open('x') as target:
        for line in inp:
            row=json.loads(line);counts['raw_rows']+=1;counts['language_'+row['language']]+=1
            if row['language']!='en':continue
            assert row.get('split','train')=='train','UNEXPECTED_SOURCE_SPLIT'
            if row['scan'] not in fit:counts['non_fit_english']+=1;continue
            assert row['instruction'].strip() and len(row['path'])>=2
            assert all(isinstance(v,str) and len(v)==32 for v in row['path'])
            assert row['instruction_id'] not in ids;ids.add(row['instruction_id'])
            key=hashlib.sha256(json.dumps({k:row[k] for k in ('scan','path','heading')},sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if key in seen:counts['duplicate_official_path_and_heading']+=1;continue
            seen.add(key);counts['english_fit_unique_source_routes']+=1;houses[row['scan']]+=1;route_lengths[len(row['path'])]+=1
            record=dict(row,source='OFFICIAL_MARKY_MATTERPORT_TRAIN',source_grade='MODEL_GENERATED_NOT_HUMAN',
                source_geometry_sha256=key,split='FIT',continuous_runtime_certified=False,official_source_filename=source.name)
            target.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
    h=hashlib.sha256()
    with (out/'OFFICIAL_ENGLISH_FIT_SOURCE.jsonl').open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    report=dict(executable=False,counts=dict(counts),per_house=dict(sorted(houses.items())),source_waypoint_length_histogram=dict(sorted(route_lengths.items())),
        source_sha256=json.loads((HERE/'marky_acquisition_v1/RESULT.json').read_text())['sha256'],manifest_sha256=h.hexdigest(),
        source_grade='OFFICIAL_MODEL_GENERATED_NOT_HUMAN_ANNOTATED',official_eval_content_read=False,
        source_routes_are_not_generated_trajectories=True,requires_official_connectivity_and_validated_CE_conversion=True,
        existing_physical_CE_route_dedup_pending=True,scientific_pass=False)
    with (out/'RESULT.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)
if __name__=='__main__':execute()
