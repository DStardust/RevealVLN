"""Independent exported-data audit. No simulator/model/GPU initialization."""
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ROOT=LINE.parents[1]
spec=importlib.util.spec_from_file_location('causal_loader',OUT/'loader.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    started=time.time();assert not (OUT/'AUDIT_RESULT.json').exists()
    lock=read(OUT/'GPU3_CODE_AND_INPUT_LOCK.json')
    for name,h in lock.items():assert sha(OUT/name)==h,('INPUT_OR_CODE_CHANGED',name)
    assets=read(OUT/'ASSET_LOCK.json')
    for r in assets['files']:
        p=(ROOT/r['path']).resolve();assert p.is_relative_to(ROOT)
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    assert sha(ROOT/assets['source_path'])==assets['source_sha256']
    jobs=read(OUT/'JOBS.json');by_id={j['job_id']:j for j in jobs}
    results=[json.loads(s) for s in (OUT/'LEDGER.jsonl').read_text().splitlines()]
    route_roots={r['job_id']:OUT for r in results}
    recovery=OUT/'recovery_v1'
    if (recovery/'LEDGER.jsonl').exists():
        for name,h in read(recovery/'INPUT_LOCK.json').items():assert sha(OUT/name)==h,('RECOVERY_CODE_CHANGED',name)
        additional=[json.loads(s) for s in (recovery/'LEDGER.jsonl').read_text().splitlines()]
        assert not set(route_roots)&{r['job_id'] for r in additional}
        route_roots.update({r['job_id']:recovery for r in additional});results+=additional
    assert len(results)==len(jobs)==100 and {r['job_id'] for r in results}==set(by_id)
    split=read(OUT/'SPLIT_FREEZE.json');assert set(split['fit_pilot']).isdisjoint(split['reserved_unassigned'])
    frame_checks={};indices=[];decisions=0;waypoint_checks=0;primitive_checks=0;max_waypoint_error=0.;timing=[]
    pixel_bytes=0
    for r in results:
        j=by_id[r['job_id']];data_root=route_roots[r['job_id']];d=data_root/'routes'/r['job_id'];diag=read(d/'diagnostic.json')
        assert j['scene_id'] in split['fit_pilot'] and read(d/'result.json')==r
        if r['status']!='CERTIFIED':
            assert not (d/'supervision_only.json').exists() and not list(d.glob('policy_*.json'))
            continue
        timing.append(r['elapsed_seconds']);s=read(d/'supervision_only.json')
        cert=read(d/'replay_certificate.json')
        assert cert['exact_all_frame_pose_rgb_semantic_match'] and cert['frames_compared']==len(s['frames'])
        assert len(s['actions'])==len(s['frames'])==r['decisions']<=513
        assert diag['collision_count']==0 and diag['maximum_segment_corridor_offset_m']<=.75
        assert s['actions']==diag['actions_attempted'] and s['frames']==diag['observations']
        assert s['actions'][-1]=='STOP' and 'STOP' not in s['actions'][:-1]
        assert s['physical_source_route_sha256']==j['physical_source_route_sha256']
        assert not s['natural_language_full_semantics_certified']
        targets=j['episode']['reference_path']+[j['episode']['goals'][0]['position']]
        assert len(s['waypoint_checks'])==len(targets)
        previous=-1
        for k,c in enumerate(s['waypoint_checks']):
            assert c['index']==k and previous<=c['decision']<len(s['frames']);previous=c['decision']
            error=float(np.linalg.norm(np.array(s['frames'][c['decision']]['position'])-np.array(targets[k])))
            assert error<=.350001 and abs(error-c['euclidean_error'])<1e-5 and 0<=c['geodesic_error']<=.35
            max_waypoint_error=max(max_waypoint_error,error);waypoint_checks+=1
        assert previous==len(s['actions'])-1
        for t,a in enumerate(s['actions'][:-1]):
            one,two=s['frames'][t:t+2]
            vector=np.array(one['position'])-two['position']
            delta=float(np.linalg.norm(vector))
            assert a in {'move_forward','turn_left','turn_right'}
            # Navmesh traversal may change height on stairs; 0.25m is the horizontal primitive.
            if a=='move_forward':assert delta>0 and float(np.linalg.norm(vector[[0,2]]))<.251
            else:assert delta<1e-6
            primitive_checks+=1
        for alias in j['instruction_alias_episodes']:
            p=d/f'policy_{alias["episode_id"]}.json';pol=read(p)
            assert set(pol)=={'instruction','rgb_sequence'}
            assert pol['instruction']==alias['instruction']['instruction_text']
            assert len(pol['rgb_sequence'])==len(s['actions'])
            for t,ref in enumerate(pol['rgb_sequence']):
                file=(data_root/ref).resolve();assert file.is_relative_to(data_root/'content')
                storage_key=str(file.relative_to(OUT))
                expected=s['frames'][t]['rgb_sha256']
                if storage_key not in frame_checks:
                    with Image.open(file) as img:arr=np.asarray(img)
                    assert arr.shape==(224,224,3) and arr.dtype==np.uint8
                    h=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
                    assert h==expected and file.stem==h
                    frame_checks[storage_key]=h;pixel_bytes+=file.stat().st_size
                assert frame_checks[storage_key]==expected
                ex=loader.decision_example(pol,s,t)
                assert ex['policy_input']=={'instruction':pol['instruction'],'rgb_history':pol['rgb_sequence'][:t+1],'executed_actions':s['actions'][:t]}
                assert ex['target_action']==s['actions'][t]
                decisions+=1
            indices.append({'policy_file':str(p.relative_to(OUT)),
                            'supervision_file':str((d/'supervision_only.json').relative_to(OUT)),
                            'job_id':r['job_id'],'split':'FIT_PILOT','scene_group':j['scene_id'],
                            'rgb_reference_root':str(data_root.relative_to(OUT))})
    gen=read(OUT/'GENERATION_RESULT.json')
    assert len(indices)==gen['instruction_records']
    assert sum(r['status']=='CERTIFIED' for r in results)==gen['certified_routes']
    assert len(frame_checks)>0
    # Commit the data index only after all per-record checks pass.
    tmp=OUT/'TRAINING_INDEX.jsonl.tmp'
    with tmp.open('x') as f:
        for x in indices:f.write(json.dumps(x)+'\n')
    tmp.replace(OUT/'TRAINING_INDEX.jsonl')
    report={'integrity_pass':True,'runtime_replay_witness_scope':'worker executed and compared every frame; this audit rechecks exported data independently, not a third simulation replay',
        'source_and_asset_hashes_verified':len(assets['files'])+1,'locked_prelaunch_files_verified':len(lock),
        'certified_routes':gen['certified_routes'],'instruction_records':len(indices),
        'verified_png_files':len(frame_checks),'unique_png_frames':len(set(frame_checks.values())),'png_bytes':pixel_bytes,
        'instruction_conditioned_supervised_decisions':decisions,
        'unique_route_decisions':gen['unique_route_decisions'],
        'euclidean_waypoint_checks':waypoint_checks,'primitive_displacement_checks':primitive_checks,
        'maximum_waypoint_error_m':max_waypoint_error,
        'certified_route_seconds_median':statistics.median(timing),
        'failure_counts':dict(collections.Counter(r.get('reason') for r in results if r['status']!='CERTIFIED')),
        'wall_seconds':time.time()-started,'gpu_or_model_initialized':False,
        'tensor_gradient_isolation_pass':None,'scientific_pass':False,'training_started':False}
    with (OUT/'AUDIT_RESULT.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
