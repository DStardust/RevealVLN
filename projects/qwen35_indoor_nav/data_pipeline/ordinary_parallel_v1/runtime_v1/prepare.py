"""Fresh per-shard immutable metadata and real scene hashes; no GPU import."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c


def main():
    assert not (HERE/'INPUT_LOCK.json').exists(),'FRESH_RUNTIME_REQUIRED'
    previous=c.read(c.PARALLEL/'INPUT_LOCK.json')['immutable']
    for p,h in previous.items():assert c.sha(c.ROOT/p)==h,p
    for p,h in c.read(c.PARALLEL/'MANIFEST_HASHES.json').items():assert c.sha(c.PARALLEL/p)==h,p
    plan=c.read(c.PARALLEL/'PLAN.json');assert [s['route_count'] for s in plan['shards']]==[250,244,253,253]
    assert c.read(c.AUTH)['approved'] and not c.read(c.AUTH)['training_allowed']
    identities=c.read(c.IDENTITIES)['holders'];assert [r['gpu_device'] for r in identities]==[6,7]
    assets={}
    for house in sorted(plan['house_to_shard']):
        for suffix in ('.glb','.navmesh','.house','_semantic.ply'):
            p=c.ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}{suffix}'
            assets[str(p.relative_to(c.ROOT))]=c.sha(p)
    c.save(HERE/'ASSETS.json',assets)
    outputs=[]
    for shard in range(4):
        root=c.shard_root(shard);root.mkdir(parents=True,exist_ok=False)
        (root/'quality').mkdir()
        data={'JOBS.json':c.read(c.PARALLEL/plan['shards'][shard]['jobs_path']),
            'SPLIT_FREEZE.json':c.read(c.BASE/'SPLIT_FREEZE.json'),
            'SOURCE_INVENTORY.json':c.read(c.BASE/'SOURCE_INVENTORY.json')}
        for name,value in data.items():c.save(root/name,value);outputs.append(root/name)
    cfg=dict(node='Q35N_ORDINARY_PARALLEL_RUNTIME_V1',runtime_allowed=False,executable=False,training_allowed=False,
        gpu_to_shards={str(k):list(v) for k,v in c.LANES.items()},shard_seconds=3600,
        lane_seconds=7200,worker_seconds_per_shard_with_cleanup_margin=3480,
        per_shard_max_bytes=49*1024**3,metadata_and_merge_max_bytes=4*1024**3,
        total_max_bytes=200*1024**3,worker_ram_bytes=12*1024**3,own_gpu_mib=4096,
        source_plan_sha256=c.sha(c.PARALLEL/'PLAN.json'),routes=1000,scene_assets=len(assets),
        restore_exact_holder=True,scientific_pass=False)
    c.save(HERE/'PREPARED_CONFIG.json',cfg)
    paths=list(HERE.glob('*.py'))+[HERE/'README_ZH.md',HERE/'ASSETS.json',HERE/'PREPARED_CONFIG.json',c.AUTH,c.IDENTITIES,
        c.PARALLEL/'INPUT_LOCK.json',c.PARALLEL/'MANIFEST_HASHES.json']+outputs
    paths += [c.PARALLEL/p for p in c.read(c.PARALLEL/'MANIFEST_HASHES.json')]
    paths += [c.BASE/p for p in ['worker.py','prepare.py','audit.py','recovery_v1/audit.py','recovery_v4/worker.py','recovery_v4/run.py','recovery_v4/safe_size.py']]
    paths += [c.PARALLEL.parent/'ordinary_pilot_v1/worker.py']
    lock=dict(previous)
    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})
    c.save(HERE/'INPUT_LOCK.json',lock)
    for shard in range(4):c.save(c.shard_root(shard)/'INPUT_LOCK.json',lock)
    print(json.dumps(dict(status='CPU_PREPARED_MAIN_REVIEW_REQUIRED',source_assets=len(assets),immutable_inputs=len(lock),
        input_lock_sha256=c.sha(HERE/'INPUT_LOCK.json'),executable=False,gpu_operations=0)))


if __name__=='__main__':main()
