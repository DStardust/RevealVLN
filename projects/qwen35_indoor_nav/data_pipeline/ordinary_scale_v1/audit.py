"""Export verifier, independent from the simulator/compiler."""
import hashlib
import json
import math
from pathlib import Path

LEGAL={'move_forward','turn_left','turn_right','STOP'}

def norm(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def check_structure(job,sup,diag,cert):
    acts=sup['actions'];frames=sup['frames']
    assert 1<=len(acts)<=513 and len(acts)==len(frames)
    assert acts[-1]=='STOP' and 'STOP' not in acts[:-1] and set(acts)<=LEGAL
    assert acts==diag['actions_attempted'] and frames==diag['observations']
    assert diag['collision_count']==0 and diag['maximum_segment_corridor_offset_m']<=.75
    assert cert['exact_all_frame_pose_rgb_semantic_match'] and cert['replay_count']==2
    assert cert['frames_compared']==len(frames)
    assert sup['physical_source_route_sha256']==job['physical_source_route_sha256']
    assert not sup['natural_language_full_semantics_certified']
    targets=job['episode']['reference_path']+[job['episode']['goals'][0]['position']]
    assert len(targets)==len(sup['waypoint_checks'])
    prev=-1
    for k,(target,c) in enumerate(zip(targets,sup['waypoint_checks'])):
        assert c['index']==k and prev<=c['decision']<len(frames)
        prev=c['decision'];err=norm(frames[prev]['position'],target)
        assert err<=.350001 and abs(err-c['euclidean_error'])<1e-5
        assert math.isfinite(c['geodesic_error']) and 0<=c['geodesic_error']<=.35
    assert prev==len(acts)-1
    for t,act in enumerate(acts[:-1]):
        a,b=frames[t]['position'],frames[t+1]['position']
        delta=norm(a,b)
        if act=='move_forward':
            assert delta>0 and norm([a[0],a[2]],[b[0],b[2]])<.251
        else:assert delta<1e-6
    return len(acts)

def audit_route(root,job):
    from PIL import Image
    import numpy as np
    d=root/'routes'/job['job_id']
    read=lambda name:json.loads((d/name).read_text())
    result=read('result.json')
    assert result['job_id']==job['job_id'] and result['scene_id']==job['scene_id']
    if result['status']!='CERTIFIED':
        assert not (d/'supervision_only.json').exists() and not list(d.glob('policy_*.json'))
        return [],set(),0
    sup=read('supervision_only.json')
    steps=check_structure(job,sup,read('diagnostic.json'),read('replay_certificate.json'))
    refs=set();rows=[]
    for e in job['instruction_alias_episodes']:
        name='policy_'+str(e['episode_id'])+'.json';pol=read(name)
        assert set(pol)=={'instruction','rgb_sequence'}
        assert pol['instruction']==e['instruction']['instruction_text']
        assert len(pol['rgb_sequence'])==steps
        for t,ref in enumerate(pol['rgb_sequence']):
            file=(root/ref).resolve(strict=True)
            assert file.is_relative_to((root/'content').resolve())
            expected=sup['frames'][t]['rgb_sha256']
            assert file.stem==expected
            if ref not in refs:
                with Image.open(file) as im:arr=np.asarray(im)
                assert arr.shape==(224,224,3) and arr.dtype==np.uint8
                assert hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()==expected
                refs.add(ref)
        # The causal loader selects RGB[:t+1], executed actions[:t]; target is separate.
        rows.append(dict(policy_file=str((d/name).relative_to(root)),
            supervision_file=str((d/'supervision_only.json').relative_to(root)),
            rgb_reference_root='.',job_id=job['job_id'],source=job['source'],
            source_sha256=job['source_sha256'],split='FIT',scene_group=job['scene_id'],
            physical_source_route_sha256=job['physical_source_route_sha256'],
            decisions=steps,quality_tier='ORDINARY_REPLAY_AND_EXPORT_VERIFIED'))
    return rows,refs,steps

def commit_shard(root,shard_id,jobs):
    out=root/'shards';out.mkdir(exist_ok=True)
    target=out/('shard_'+str(shard_id).zfill(4)+'.jsonl')
    assert not target.exists()
    rows=[];frames=set();route_steps=0
    for j in jobs:
        r,f,n=audit_route(root,j);rows+=r;frames.update(f);route_steps+=n
    tmp=target.with_suffix('.pending')
    with tmp.open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(target)
    cert=target.with_suffix('.audit.json')
    with cert.open('x') as f:json.dump(dict(integrity_pass=True,attempted_routes=len(jobs),
        certified_routes=len({r['job_id'] for r in rows}),instruction_records=len(rows),
        unique_rgb_files=len(frames),unique_route_decisions=route_steps,
        instruction_conditioned_decisions=sum(r['decisions'] for r in rows),
        index_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        runtime_replay_witness='compiler_second_actual_replay; independent_CPU_export_recheck',
        scientific_pass=False),f,indent=2)
    return json.loads(cert.read_text())

