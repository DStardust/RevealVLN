"""Prospective strict source-alignment screen; not renderer or semantic certification."""
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
ACQ=HERE/'acquisition_v3'
def read(p):return json.loads(p.read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def vector_equal(a,b):return len(a)==len(b) and max(abs(x-y) for x,y in zip(a,b))<=1e-5
def pose_equal(a,b):
    q=a['start_rotation'];r=b['start_rotation']
    return vector_equal(a['start_position'],b['start_position']) and (vector_equal(q,r) or vector_equal(q,[-x for x in r]))
def ordered_cover(guide,follower):
    cursor=0
    for point in follower:
        if cursor<len(guide) and vector_equal(guide[cursor],point):cursor+=1
    return cursor==len(guide)
def physical(e):return {k:e[k] for k in ('scene_id','start_position','start_rotation','reference_path','goals')}
def geom(e):return dict(scene_id=e['scene_id'],start_position=e['start_position'],start_rotation=e['start_rotation'],reference_path=e['reference_path'],goal_positions=[g['position'] for g in e['goals']])
def screen(e,g):
    if e['info'].get('role')!='follower':return 'NOT_OFFICIAL_FOLLOWER'
    if e['instruction']['instruction_text']!=g['instruction']['instruction_text']:return 'INSTRUCTION_CHANGED'
    if e['scene_id']!=g['scene_id'] or e['goals']!=g['goals']:return 'ORIGINAL_GOAL_CHANGED'
    if not pose_equal(e,g):return 'ORIGINAL_START_POSE_CHANGED'
    metrics=e['info']['metrics']
    if not all(math.isfinite(metrics[k]) for k in ('sr','ndtw','ne')):return 'INVALID_METRICS'
    if metrics['sr']!=1 or metrics['ndtw']<.9:return 'OFFICIAL_FOLLOWER_ALIGNMENT_LOW'
    if not ordered_cover(g['reference_path'],e['reference_path']):return 'GUIDE_WAYPOINT_ORDER_NOT_COVERED'
    return None
def execute():
    out=HERE/'follower_manifest_v1';out.mkdir(exist_ok=False)
    fit=set(read(PIPE/'ordinary_fullscale_source_v1/SPLIT_FREEZE.json')['FIT'])
    guide_path=ROOT/'data/phase0/raw/rxr_vlnce_v0/train/train_guide.json.gz'
    guides={};seen_geom=set();counts=collections.Counter();candidates=[]
    # Existing R2R/RxR official FIT geometry excludes aliases/radius-only changes.
    for p in (ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz',guide_path):
        for e in json_stream.episodes(p):
            scene=Path(e['scene_id']).parent.name
            if scene not in fit or not e['instruction'].get('language','en').startswith('en'):continue
            seen_geom.add(hashlib.sha256(canonical(geom(e))).hexdigest())
            if p==guide_path:guides[str(e['instruction']['instruction_id'])]=e
    sourcepath=ACQ/'train_follower.json.gz';sourcehash=sha(sourcepath)
    for e in json_stream.episodes(sourcepath):
        counts['raw_follower_episodes']+=1
        if not e['instruction']['language'].startswith('en'):counts['non_english_excluded']+=1;continue
        if Path(e['scene_id']).parent.name not in fit:counts['non_fit_excluded']+=1;continue
        counts['english_fit']+=1
        g=guides.get(str(e['instruction']['instruction_id']))
        if g is None:counts['guide_binding_missing']+=1;continue
        reason=screen(e,g)
        if reason:counts[reason]+=1;continue
        key=hashlib.sha256(canonical(geom(e))).hexdigest()
        if key in seen_geom:counts['existing_or_duplicate_geometry']+=1;continue
        seen_geom.add(key)
        e['instruction']={k:v for k,v in e['instruction'].items() if k in ('instruction_id','instruction_text','language')}
        pkey=hashlib.sha256(canonical(physical(e))).hexdigest()
        candidates.append(dict(job_id='rxrf_'+pkey[:20],source='RxR_FOLLOWER_OFFICIAL_TRAIN',
            source_grade='OFFICIAL_HUMAN_FOLLOWER_WITH_SOURCE_ALIGNMENT_SCREEN_NOT_FULL_SEMANTIC_CERTIFICATION',
            source_path=str(sourcepath.relative_to(ROOT)),source_sha256=sourcehash,scene_id=Path(e['scene_id']).parent.name,
            split='FIT',physical_source_route_sha256=pkey,geometry_sha256=key,episode=e,instruction_alias_episodes=[e],
            original_guide_episode_id=g['episode_id'],original_guide_instruction_id=g['instruction']['instruction_id'],
            alignment_checks=dict(unchanged_instruction=True,unchanged_goal=True,unchanged_start_pose=True,
                ordered_guide_waypoints_covered=True,official_sr=1,official_ndtw_min=.9),runtime_certified=False))
    # Source-GT coverage and temporal fingerprints; final actions still require real replay.
    byid={str(j['episode']['episode_id']):j for j in candidates};gt_seen={};new=[]
    for eid,gt in json_stream.entries(ACQ/'train_follower_gt.json.gz'):
        if str(eid) not in byid:continue
        j=byid[str(eid)];assert isinstance(gt,dict) and gt.get('locations') and gt.get('actions')
        key=hashlib.sha256(canonical(gt)).hexdigest()
        if key in gt_seen:counts['duplicate_follower_gt']+=1;continue
        gt_seen[key]=eid;j['official_gt_sha256']=key;j['official_gt_actions']=len(gt['actions']);new.append(j)
    assert len(new)+counts['duplicate_follower_gt']==len(candidates),'MISSING_OFFICIAL_FOLLOWER_GT'
    new.sort(key=lambda j:(j['scene_id'],j['physical_source_route_sha256']))
    paths=[guide_path,ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz',sourcepath,ACQ/'train_follower_gt.json.gz',ACQ/'RESULT.json',PIPE/'ordinary_fullscale_source_v1/SPLIT_FREEZE.json',HERE/'json_stream.py',HERE/'follower_manifest.py']
    report=dict(executable=False,runtime_allowed=False,source_accepted_candidates=len(new),instruction_aliases=len(new),
        houses=len({j['scene_id'] for j in new}),counts=dict(counts),actual_generated_routes=0,
        full_natural_language_semantics_certified=False,official_gt_cross_comparison_with_guide_pending=True,
        quality_gate='requires_original_strict_compiler_and_double_replay_then_strict_audit',
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths})
    for name,value in (('JOBS.json',new),('RESULT.json',report)):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(report),flush=True)
if __name__=='__main__':execute()
