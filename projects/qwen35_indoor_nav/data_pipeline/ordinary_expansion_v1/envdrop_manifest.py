"""Official CE augmentation source, FIT-only dedup; real replay still required."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import json_stream
ROOT=HERE.parents[3]
PIPE=HERE.parent
ACQ=HERE/'envdrop_acquisition_v1'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def geometry(e):
    q=list(e['start_rotation'])
    if next((x for x in reversed(q) if abs(x)>1e-12),1)<0:q=[-x for x in q]
    def xyz(v):return [round(x,4) for x in v]
    return dict(scene_id=e['scene_id'],start_position=xyz(e['start_position']),start_rotation=[round(v,6) for v in q],
        reference_path=[xyz(v) for v in e['reference_path']],goals=[xyz(v['position']) for v in e['goals']])
def geometry_key(e):return hashlib.sha256(canonical(geometry(e))).hexdigest()
def gt_key(g):return hashlib.sha256(canonical({k:g[k] for k in ('locations','actions')})).hexdigest()
def validate(e):
    assert e['instruction']['instruction_text'].strip() and e['reference_path'] and len(e['goals'])==1
    vectors=[e['start_position'],e['start_rotation'],*[g['position'] for g in e['goals']],*e['reference_path']]
    assert all(all(type(x) in (float,int) and math.isfinite(x) for x in v) for v in vectors)
    assert len(e['start_position'])==3 and len(e['start_rotation'])==4
    assert all(len(p)==3 for p in e['reference_path'])
    assert abs(sum(x*x for x in e['start_rotation'])-1)<1e-5
    assert max(abs(a-b) for a,b in zip(e['start_position'],e['reference_path'][0]))<1e-5
    assert max(abs(a-b) for a,b in zip(e['goals'][0]['position'],e['reference_path'][-1]))<1e-5
def select(candidates,n=20000):
    pools=collections.defaultdict(list)
    for j in sorted(candidates,key=lambda j:j['physical_source_route_sha256']):pools[j['scene_id']].append(j)
    order=[]
    for i in range(max(map(len,pools.values()))):
        for house in sorted(pools):
            if i<len(pools[house]):order.append(pools[house][i])
            if len(order)==n:return order
    raise AssertionError('INSUFFICIENT_UNIQUE_SOURCE_CAPACITY')
def execute():
    out=HERE/'envdrop_source_v1';out.mkdir(exist_ok=False)
    split_path=PIPE/'ordinary_fullscale_source_v1/SPLIT_FREEZE.json';fit=set(read(split_path)['FIT'])
    sources=[('R2R',ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz',ROOT/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_gt.json.gz'),
             ('RxR',ROOT/'data/phase0/raw/rxr_vlnce_v0/train/train_guide.json.gz',ROOT/'third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide_gt.json.gz')]
    seen=set();oldgt=set();counts=collections.Counter();paths=[split_path,ACQ/'envdrop.json.gz',ACQ/'envdrop_gt.json.gz',ACQ/'RESULT.json',HERE/'json_stream.py',HERE/'envdrop_manifest.py',HERE/'test_expansion.py']
    for tag,p,gp in sources:
        wanted=set();paths += [p,gp]
        for e in json_stream.episodes(p):
            if Path(e['scene_id']).parent.name not in fit or not e['instruction'].get('language','en').startswith('en'):continue
            seen.add(geometry_key(e));wanted.add(str(e['episode_id']))
        for eid,g in json_stream.entries(gp):
            if str(eid) in wanted:oldgt.add(gt_key(g))
    oldgeom=sorted(seen);candidates={};hashsource=sha(ACQ/'envdrop.json.gz')
    for e in json_stream.episodes(ACQ/'envdrop.json.gz'):
        counts['official_envdrop_episodes']+=1
        scene=Path(e['scene_id']).parent.name
        if scene not in fit:counts['non_fit_excluded']+=1;continue
        counts['fit_source_episodes']+=1
        try:validate(e)
        except (AssertionError,KeyError,TypeError,ValueError):counts['invalid_source_geometry_or_text']+=1;continue
        key=geometry_key(e)
        if key in seen:counts['old_or_duplicate_geometry_excluded']+=1;continue
        seen.add(key)
        e['instruction']={'instruction_text':e['instruction']['instruction_text'],'language':'en'}
        pkey=hashlib.sha256(canonical({k:e[k] for k in ('scene_id','start_position','start_rotation','reference_path','goals')})).hexdigest()
        ident=str(e['episode_id']);assert ident not in candidates
        candidates[ident]=dict(job_id='envdrop_'+pkey[:20],source='ENVDROP_OFFICIAL_CE_TRAIN',
            source_grade='OFFICIAL_SYNTHETIC_ENGLISH_INSTRUCTION_NOT_HUMAN',source_path=str((ACQ/'envdrop.json.gz').relative_to(ROOT)),source_sha256=hashsource,
            scene_id=scene,split='FIT',physical_source_route_sha256=pkey,geometry_sha256=key,
            episode=e,instruction_alias_episodes=[e],full_natural_language_semantics_certified=False)
    newgt=set();accepted=[];found=set()
    for eid,g in json_stream.entries(ACQ/'envdrop_gt.json.gz'):
        ident=str(eid)
        if ident not in candidates:continue
        assert ident not in found;found.add(ident);j=candidates[ident]
        if not g.get('actions') or not g.get('locations') or g['actions'][-1]!=0 or not set(g['actions'])<={0,1,2,3}:
            counts['invalid_official_gt']+=1;continue
        if not all(len(p)==3 and all(math.isfinite(x) for x in p) for p in g['locations']):counts['invalid_official_gt']+=1;continue
        if max(abs(a-b) for a,b in zip(j['episode']['start_position'],g['locations'][0]))>1e-4:
            counts['official_gt_start_mismatch']+=1;continue
        key=gt_key(g)
        if key in oldgt or key in newgt:counts['old_or_duplicate_gt_excluded']+=1;continue
        newgt.add(key);j['official_gt_sha256']=key;j['official_gt_actions']=len(g['actions']);accepted.append(j)
    counts['missing_official_gt']=len(set(candidates)-found)
    chosen=select(accepted);sentinel=chosen[:3];rest=sorted(chosen[3:],key=lambda j:(j['scene_id'],j['physical_source_route_sha256']))
    shards=[sentinel]+[rest[i:i+1000] for i in range(0,len(rest),1000)]
    assert len(chosen)==20000 and len(shards)==21 and len(shards[-1])==997
    assert len({j['physical_source_route_sha256'] for j in chosen})==20000
    save(out/'JOBS.json',sentinel+rest)
    save(out/'EXCLUSION_MANIFEST.json',dict(geometry_keys=oldgeom,physical_route_keys=[]))
    # Exact physical-key exclusion retained as a second independent check.
    original_jobs=[PIPE/'ordinary_pilot_v1/JOBS.json',PIPE/'ordinary_scale_v1/JOBS.json',PIPE/'ordinary_parallel_v1/JOBS.json',PIPE/'ordinary_fullscale_source_v1/JOBS.json']
    oldphysical=sorted({j['physical_source_route_sha256'] for p in original_jobs for j in read(p)})
    assert len(oldphysical)==8742 and set(oldphysical).isdisjoint(j['physical_source_route_sha256'] for j in chosen)
    save(out/'PHYSICAL_EXCLUSION.json',dict(physical_route_keys=oldphysical))
    paths+=original_jobs
    manifest=[]
    for s,jobs in enumerate(shards):
        folder=out/'manifests'/f'shard_{s:04d}';folder.mkdir(parents=True)
        save(folder/'JOBS.json',jobs);paths.append(folder/'JOBS.json')
        manifest.append(dict(shard=s,routes=len(jobs),jobs_path=str((folder/'JOBS.json').relative_to(out)),jobs_sha256=sha(folder/'JOBS.json'),sentinel=s==0))
    inventory=dict(sources=[dict(source='ENVDROP_OFFICIAL_CE_TRAIN',path=str((ACQ/'envdrop.json.gz').relative_to(ROOT)),sha256=hashsource,
                               grade='OFFICIAL_SYNTHETIC_ENGLISH_NOT_HUMAN')],language='en',source_split='envdrop_train_augmentation',official_eval_content_read=False)
    save(out/'SOURCE_INVENTORY.json',inventory)
    plan=dict(executable=False,runtime_allowed=False,available_source_routes=len(accepted),selected_routes=20000,selected_aliases=20000,
        selected_houses=len({j['scene_id'] for j in chosen}),counts=dict(counts),shards=manifest,
        official_gt_reference_action_count=sum(j['official_gt_actions'] for j in chosen),
        generated_action_count=0,full_natural_language_semantics_certified=False,
        source_grade='OFFICIAL_SYNTHETIC_ENGLISH_CE_PORT_NOT_HUMAN',
        selection='whole-house round-robin by canonical physical hash, first20000; first3 prospective sentinel; rest house grouped',
        sentinel_required_strict_routes=3,original_quality_thresholds_unchanged=True,scientific_pass=False)
    save(out/'PLAN.json',plan)
    paths += [out/name for name in ('JOBS.json','EXCLUSION_MANIFEST.json','PHYSICAL_EXCLUSION.json','SOURCE_INVENTORY.json','PLAN.json')]
    save(out/'INPUT_LOCK.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    print(json.dumps(plan),flush=True)
if __name__=='__main__':execute()
