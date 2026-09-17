"""Strict per-route quarantine; thresholds inherited unchanged from sealed V1."""
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import traceback

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
s=importlib.util.spec_from_file_location('strict_v1_audit',BASE/'audit.py')
strict=importlib.util.module_from_spec(s);s.loader.exec_module(strict)

def commit_shard(root,shard_id,jobs):
    out=root/'shards';out.mkdir(exist_ok=True)
    target=out/('shard_'+str(shard_id).zfill(4)+'.jsonl')
    assert not target.exists()
    rows=[];frames=set();steps=0;quarantine=[]
    for j in jobs:
        try:
            r,f,n=strict.audit_route(root,j)
        except AssertionError:
            # Invalid export is excluded, not relabelled as task failure.
            quarantine.append(dict(job_id=j['job_id'],source=j['source'],scene_id=j['scene_id'],
                stage='strict_export_audit',status='QUARANTINED_NOT_TRAINING_DATA',
                traceback=traceback.format_exc()))
            continue
        rows+=r;frames.update(f);steps+=n
    q=HERE/('quarantine_'+str(shard_id).zfill(4)+'.json')
    with q.open('x') as f:json.dump(quarantine,f,indent=2)
    tmp=target.with_suffix('.pending')
    with tmp.open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
        f.flush();os.fsync(f.fileno())
    tmp.replace(target)
    result=dict(integrity_pass=True,attempted_routes=len(jobs),
        certified_routes=len({r['job_id'] for r in rows}),
        quarantined_routes=len(quarantine),quarantine_manifest=str(q.relative_to(root)),
        instruction_records=len(rows),unique_rgb_files=len(frames),unique_route_decisions=steps,
        instruction_conditioned_decisions=sum(r['decisions'] for r in rows),
        index_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        quality_rule='V1_unchanged_thresholds_per_route_quarantine',
        scientific_pass=False)
    with target.with_suffix('.audit.json').open('x') as f:json.dump(result,f,indent=2)
    return result

