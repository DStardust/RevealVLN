"""Bounded ordinary production; inherited sealed compiler is read-only."""
import collections
import importlib.util
import json
import os
from pathlib import Path
import time

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ROOT=LINE.parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

prep=module('scale_prepare',OUT/'prepare.py')
audit=module('scale_audit',OUT/'audit.py')

def atomic(name,obj):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.pending')
    with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False)
    tmp.replace(p)

def main():
    for path,h in json.loads((OUT/'INPUT_LOCK.json').read_text()).items():
        assert prep.sha(ROOT/path)==h,('LOCK_CHANGED',path)
    source=json.loads((OUT/'SOURCE_INVENTORY.json').read_text())
    for s in source['sources']:assert prep.sha(ROOT/s['path'])==s['sha256']
    core=module('sealed_ordinary_compiler',LINE/'data_pipeline/ordinary_pilot_v1/worker.py')
    core.OUT=OUT
    jobs=json.loads((OUT/'JOBS.json').read_text())
    split=json.loads((OUT/'SPLIT_FREEZE.json').read_text())
    assert all(j['scene_id'] in split['FIT'] for j in jobs)
    core_hs=core.hs
    def simulator(scene):
        cfg=core_hs.SimulatorConfiguration()
        cfg.scene_id=str(ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{scene}/{scene}.glb')
        cfg.gpu_device_id=2;cfg.enable_physics=False;cfg.allow_sliding=False
        ac=core_hs.agent.AgentConfiguration();ac.height=1.5;ac.radius=.1
        ac.sensor_specifications=[]
        for name,kind in [('rgb',core_hs.SensorType.COLOR),('semantic',core_hs.SensorType.SEMANTIC)]:
            s=core_hs.SensorSpec();s.uuid=name;s.sensor_type=kind;s.resolution=[224,224]
            s.position=[0,1.25,0];s.orientation=[0,0,0];s.parameters['hfov']='90'
            s.gpu2gpu_transfer=False;ac.sensor_specifications.append(s)
        ac.action_space={n:core_hs.agent.ActionSpec(n,core_hs.agent.ActuationSpec(amount=v)) for n,v in core.ACTIONS.items()}
        sim=core_hs.Simulator(core_hs.Configuration(cfg,[ac]))
        if not sim.pathfinder.is_loaded:sim.close();raise RuntimeError('NAVMESH_NOT_LOADED')
        return sim
    counts={k:0 for k in ('primitive_actions','observations','resets','logical_stops','collisions','simulator_constructions')}
    prior_results={}
    if (OUT/'LEDGER.jsonl').exists():
        for line in (OUT/'LEDGER.jsonl').read_text().splitlines():
            r=json.loads(line);assert r['job_id'] not in prior_results
            prior_results[r['job_id']]=r
    assert set(prior_results)<={j['job_id'] for j in jobs}
    # Interrupted directories are never overwritten or silently labelled negative.
    for j in jobs:
        if (OUT/'routes'/j['job_id']).exists() and j['job_id'] not in prior_results:
            raise RuntimeError('INTERRUPTED_JOB_REQUIRES_VERSIONED_RECOVERY:'+j['job_id'])
    all_results=dict(prior_results);sim=None;scene=None;started=time.time();asset_locks={}
    try:
        for i,j in enumerate(jobs):
            if j['job_id'] not in prior_results:
                if scene!=j['scene_id']:
                    if sim:sim.close();sim=None
                    scene=j['scene_id'];records=[]
                    for suffix in ('.glb','.navmesh','.house','_semantic.ply'):
                        p=ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{scene}/{scene}{suffix}'
                        records.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=prep.sha(p)))
                    asset_locks[scene]=records
                    atomic('ASSET_LOCK_LIVE.json',asset_locks)
                    sim=simulator(scene);counts['simulator_constructions']+=1
                    with (OUT/'SCENE_EXPOSURE.jsonl').open('a') as f:
                        f.write(json.dumps(dict(scene_id=scene,split='FIT',kind='actual_renderer_loaded',time=time.time()))+'\n')
                all_results[j['job_id']]=core.run_job(sim,j,counts)
            if (i+1)%50==0 or i+1==len(jobs):
                sid=i//50
                if not (OUT/'shards'/('shard_'+str(sid).zfill(4)+'.jsonl')).exists():
                    audit.commit_shard(OUT,sid,jobs[sid*50:i+1])
            passed=[r for r in all_results.values() if r['status']=='CERTIFIED']
            audits=[json.loads(p.read_text()) for p in (OUT/'shards').glob('*.audit.json')] if (OUT/'shards').exists() else []
            atomic('PROGRESS.json',dict(completed=len(all_results),target=len(jobs),
                replay_certified_routes=len(passed),audited_routes=sum(a['certified_routes'] for a in audits),
                audited_instruction_records=sum(a['instruction_records'] for a in audits),
                audited_instruction_conditioned_decisions=sum(a['instruction_conditioned_decisions'] for a in audits),
                houses_completed=len({r['scene_id'] for r in all_results.values()}),
                failure_counts=dict(collections.Counter(r.get('reason') for r in all_results.values() if r['status']!='CERTIFIED')),
                worker_elapsed_seconds=time.time()-started,counts_this_invocation=counts,
                scientific_pass=False,training_started=False))
    finally:
        if sim:sim.close()
        atomic('COUNTS_LAST_INVOCATION.json',dict(counts,elapsed_seconds=time.time()-started))
    atomic('GENERATION_COMPLETE.json',dict(completed=len(all_results),target=len(jobs),all_jobs_terminal=len(all_results)==len(jobs),
        audited_shards=len(list((OUT/'shards').glob('*.audit.json'))),scientific_pass=False))

if __name__=='__main__':main()

