"""CPU-only freeze after main supplies exact identities/authority; never query GPUs."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport


def validate_identity(identity,gpu,root):
    required={'gpu_device','gpu_uuid','pid','starttime_ticks','proc_uid','cwd','cmdline','pane_id','pane_target'}
    assert required<=set(identity)
    assert identity['gpu_device']==gpu and identity['gpu_uuid'].startswith('GPU-')
    assert type(identity['pid']) is int and identity['pid']>0
    assert type(identity['starttime_ticks']) is int and identity['starttime_ticks']>0
    assert identity['cwd']==str(root)
    argv=identity['cmdline']
    assert 'scripts/occupy_idle_gpu.py' in argv
    assert argv.count('--gpu')==1 and argv[argv.index('--gpu')+1]==str(gpu)
    assert '--tag' in argv and identity['pane_id'].startswith('%')


def execute(identities,authorization,queues):
    assert not (HERE/'INPUT_LOCK.json').exists() and not (HERE/'SETUP.json').exists(),'FRESH_RUNTIME_REQUIRED'
    lanes=transport.selected_queues(queues)
    for path in (identities,authorization):assert path.resolve(strict=True).is_relative_to(transport.ROOT)
    ids=json.loads(identities.read_text());auth=json.loads(authorization.read_text())
    assert auth['approved'] and not auth['training_allowed']
    # Main must expressly authorize exactly this source/shard allocation, not an old batch.
    assert auth['gpu_to_sequential_shards']=={str(g):list(v) for g,v in lanes.items()},'AUTH_QUEUE_MISMATCH'
    line=HERE.parents[2]
    assert (line/auth['source_plan']).resolve()==HERE.parent/'PLAN.json'
    assert (line/auth['runtime']).resolve()==HERE
    assert (line/auth['holder_identities']).resolve()==identities.resolve()
    assert auth['max_wall_seconds_per_gpu_chain']>=7200
    selected=sorted(s for rows in lanes.values() for s in rows)
    source_plan=json.loads((HERE.parent/'PLAN.json').read_text())
    assert auth['max_new_routes']>=sum(source_plan['shards'][s]['routes'] for s in selected)
    assert auth['max_new_total_gib']>=49*len(selected)+4
    assert auth['max_worker_ram_gib']>=12 and auth['max_own_gpu_mib']>=4096
    assert not auth['actual_external_tasks_may_be_stopped']
    assert auth['restore_holders_on_success_failure_or_interruption']
    assert {r['gpu_device'] for r in ids['holders']}==set(lanes)
    assert len(ids['holders'])==len(lanes)
    for identity in ids['holders']:validate_identity(identity,identity['gpu_device'],transport.ROOT)
    cfg=dict(gpu_to_shards={str(g):list(v) for g,v in lanes.items()},
        identities_path=str(identities.resolve().relative_to(transport.ROOT)),
        authorization_path=str(authorization.resolve().relative_to(transport.ROOT)),executable=False)
    with (HERE/'SETUP.json').open('x') as f:json.dump(cfg,f,indent=2)
    # Import only after the immutable prospective transport mapping is written.
    import common as c
    assert c.LANES==lanes and c.ROOT==transport.ROOT
    lock=c.read(c.PARALLEL/'INPUT_LOCK.json')
    for p,h in lock.items():assert c.sha(c.ROOT/p)==h,p
    for p,h in c.read(c.PARALLEL/'MANIFEST_HASHES.json').items():assert c.sha(c.PARALLEL/p)==h,p
    for name in transport.EXPECTED:transport.original(name)
    plan=c.read(c.PARALLEL/'PLAN.json');assert plan['remaining_routes']==6642 and len(plan['shards'])==7
    jobs={s:c.read(c.PARALLEL/plan['shards'][s]['jobs_path']) for s in c.SELECTED_SHARDS}
    houses=sorted({j['scene_id'] for rows in jobs.values() for j in rows})
    assets={}
    for house in houses:
        for suffix in ('.glb','.navmesh','.house','_semantic.ply'):
            p=c.ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}{suffix}'
            assets[str(p.relative_to(c.ROOT))]=c.sha(p)
    c.save(HERE/'ASSETS.json',assets)
    outputs=[]
    for shard,rows in jobs.items():
        root=c.shard_root(shard);root.mkdir(parents=True,exist_ok=False)
        (root/'quality').mkdir()
        for name,value in {'JOBS.json':rows,'SPLIT_FREEZE.json':c.read(c.PARALLEL/'SPLIT_FREEZE.json'),
                           'SOURCE_INVENTORY.json':c.read(c.PARALLEL/'SOURCE_INVENTORY.json')}.items():
            c.save(root/name,value);outputs.append(root/name)
    prepared=dict(node='Q35N_ORDINARY_FULLSCALE_RUNTIME_V2',runtime_allowed=False,executable=False,training_allowed=False,
        gpu_to_shards=cfg['gpu_to_shards'],selected_shards=list(c.SELECTED_SHARDS),all_source_shards=7,
        shard_seconds=3600,lane_seconds=7200,worker_seconds_per_shard_with_cleanup_margin=3480,
        per_shard_max_bytes=49*1024**3,metadata_and_merge_max_bytes=4*1024**3,
        total_max_bytes=(49*len(c.SELECTED_SHARDS)+4)*1024**3,
        worker_ram_bytes=12*1024**3,own_gpu_mib=4096,scene_assets=len(assets),
        source_plan_sha256=c.sha(c.PARALLEL/'PLAN.json'),routes=sum(map(len,jobs.values())),
        restore_exact_holder=True,scientific_pass=False)
    c.save(HERE/'PREPARED_CONFIG.json',prepared)
    paths=list(HERE.glob('*.py'))+[HERE/'SETUP.json',HERE/'ASSETS.json',HERE/'PREPARED_CONFIG.json',
        c.AUTH,c.IDENTITIES,c.PARALLEL/'INPUT_LOCK.json',c.PARALLEL/'MANIFEST_HASHES.json']+outputs
    paths += [c.PARALLEL/p for p in c.read(c.PARALLEL/'MANIFEST_HASHES.json')]
    paths += [transport.SOURCE/name for name in transport.EXPECTED]
    # Strict compiler, quarantine adapter, and conservative size are transitive executable dependencies.
    paths += [c.BASE/p for p in ['worker.py','prepare.py','audit.py','recovery_v1/audit.py','recovery_v4/safe_size.py']]
    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})
    c.save(HERE/'INPUT_LOCK.json',lock)
    for shard in c.SELECTED_SHARDS:c.save(c.shard_root(shard)/'INPUT_LOCK.json',lock)
    print(json.dumps(dict(executable=False,routes=prepared['routes'],assets=len(assets),lock_sha256=c.sha(HERE/'INPUT_LOCK.json'),
        approvals={str(g):c.approval_value(g) for g in c.LANES},gpu_operations=0),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identities',required=True,type=Path)
    p.add_argument('--authorization',required=True,type=Path)
    p.add_argument('--queues',default='{"3":[0],"4":[1]}')
    args=p.parse_args();execute(args.identities,args.authorization,json.loads(args.queues))
