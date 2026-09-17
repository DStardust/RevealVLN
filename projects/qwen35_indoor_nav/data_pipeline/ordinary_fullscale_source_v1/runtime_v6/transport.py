"""GPU4 prospective rescue of 953 NEVER-attempted jobs; no old-root writes."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE_V3=HERE.parent/'runtime_v3'
RESCUE=HERE.parent/'salvage_rescue_v1/run_v1'
RESCUE_SHA='d68f813bf7d21f3d013acc81116f21f0774d1d15b1bba09f012639e3f0ef554a'
raw=(SOURCE_V3/'transport.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='148432857e063e2c6b81fbb0de609c896e167e1476213f734338e582821382fe'
exec(compile(raw,str(__file__),'exec'),globals())
_v3_setup=setup
_v3_common_source=common_source
_v3_run_source=run_source
_v3_prepare_source=prepare_source
_v3_merge_source=merge_source


def setup():
    cfg=_v3_setup()
    if not (HERE/'SETUP.json').exists():cfg=dict(cfg,gpu_to_shards={'4':[0,1]})
    assert cfg['gpu_to_shards']=={'4':[0,1]},'GPU4_RESCUE_0_1_ONLY'
    return cfg


def rescue_jobs():
    """Only exact frozen subset members with no original ledger/directory."""
    import json
    assert hashlib.sha256((RESCUE/'RESCUE_JOBS.json').read_bytes()).hexdigest()==RESCUE_SHA
    rows=json.loads((RESCUE/'RESCUE_JOBS.json').read_text())
    byid={j['job_id']:j for j in rows};assert len(byid)==len(rows)==953
    result={};found=set();physical=set();aliases=set()
    for shard,count in ((0,279),(1,674)):
        root=HERE.parent/'production'/f'shard_{shard:04d}'
        full=json.loads((root/'JOBS.json').read_text())
        assert full==json.loads((HERE.parent/'manifests'/f'shard_{shard:04d}'/'JOBS.json').read_text())
        ledger=(root/'LEDGER.jsonl').read_bytes();assert ledger.endswith(b'\n')
        terminal={json.loads(line)['job_id'] for line in ledger.splitlines()}
        directories={p.name for p in (root/'routes').iterdir()}
        subset=[j for j in full if j['job_id'] in byid]
        assert len(subset)==count
        for job in subset:
            ident=job['job_id'];assert ident not in found
            assert job==byid[ident],'SOURCE_JOB_OR_ALIAS_CHANGED'
            assert ident not in terminal and ident not in directories,'ORIGINAL_ROUTE_ATTEMPTED'
            key=job['physical_source_route_sha256'];assert key not in physical
            keys={(job['source'],str(e['episode_id'])) for e in job['instruction_alias_episodes']}
            assert len(keys)==len(job['instruction_alias_episodes']) and aliases.isdisjoint(keys)
            physical.add(key);aliases.update(keys);found.add(ident)
        result[shard]=subset
    assert found==set(byid)
    assert [j for s in (0,1) for j in result[s]]==rows,'PRESERVE_FROZEN_SUBSET_ORDER'
    return result


def common_source():
    source=exact(_v3_common_source(),"return PARALLEL/'production'/f'shard_{shard:04d}'",
        "return PARALLEL/'rescue_production'/f'shard_{shard:04d}'")
    source+='''

def claim_original_source_lanes():
    import fcntl
    import transport
    handles=[]
    for gpu in (3,4):
        path=PARALLEL/'runtime_v2/lanes'/f'gpu_{gpu}/PRODUCER.lock'
        handle=path.open('rb');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(handle)
    jobs=transport.rescue_jobs()
    assert all(read(shard_root(s)/'JOBS.json')==rows for s,rows in jobs.items())
    return handles
'''
    return source


def run_source():
    source=_v3_run_source()
    source=exact(source,"        call('tmux','respawn-pane','-t',identity['pane_target'],'-c',str(c.ROOT),'sleep 24000')\n        sleeper_identity=wait_stable_sleeper(identity);sleeper=sleeper_identity['pid']",
        '        # Keep original holder pane dead; no temporary sleep process.')
    source=exact(source,"    limits=c.read(HERE/'PREPARED_CONFIG.json')",
        "    old_source_locks=c.claim_original_source_lanes()\n    limits=c.read(HERE/'PREPARED_CONFIG.json')")
    return exact(source,'choices=[3,4,6,7]','choices=[4]')


def prepare_source():
    source=_v3_prepare_source()
    source=exact(source,"default='{\"6\":[2,4],\"7\":[3,5]}'","default='{\"4\":[0,1]}'")
    source=exact(source,"node='Q35N_ORDINARY_FULLSCALE_RUNTIME_V3'","node='Q35N_ORDINARY_UNATTEMPTED_RESCUE_RUNTIME_V6'")
    source=exact(source,'    lanes=transport.selected_queues(queues)',
        "    lanes=transport.selected_queues(queues)\n    assert lanes=={4:(0,1)}\n    assert not (HERE.parent/'rescue_production').exists(),'FRESH_RESCUE_PRODUCTION_REQUIRED'")
    source=exact(source,"    assert auth['approved'] and not auth['training_allowed']",
        "    assert auth['approved'] and not auth['training_allowed']\n    assert auth['rescue_jobs_sha256']==transport.RESCUE_SHA\n    assert (HERE.parents[2]/auth['rescue_source']).resolve()==transport.RESCUE\n    assert (HERE.parents[2]/auth['new_output_parent']).resolve()==HERE.parent/'rescue_production'\n    assert auth['expected_unattempted_per_old_shard']=={'0':279,'1':674}\n    assert not auth['old_failed_and_interrupted_routes_may_be_retried']\n    assert not auth['old_production_roots_may_be_modified']")
    source=exact(source,"sum(source_plan['shards'][s]['routes'] for s in selected)", '953')
    source=exact(source,"    jobs={s:c.read(c.PARALLEL/plan['shards'][s]['jobs_path']) for s in c.SELECTED_SHARDS}",
        "    import fcntl\n    old_source_locks=[]\n    for gpu in (3,4):\n        handle=(HERE.parent/'runtime_v2/lanes'/f'gpu_{gpu}/PRODUCER.lock').open('rb')\n        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);old_source_locks.append(handle)\n    jobs=transport.rescue_jobs()")
    source=exact(source,"HERE/'budget_preflight_v1'/name", "transport.SOURCE_V3/'budget_preflight_v1'/name")
    source=exact(source,'restore_exact_holder=True,scientific_pass=False)',
        "restore_exact_holder=True,scientific_pass=False,source_scope='UNATTEMPTED_SUBSET_ONLY_NOT_ORIGINAL_SHARD_RETRY',\n        rescue_snapshot=str(transport.RESCUE.relative_to(c.ROOT)),rescue_jobs_sha256=transport.RESCUE_SHA,\n        output_roots={str(s):str(c.shard_root(s).relative_to(c.ROOT)) for s in c.SELECTED_SHARDS},\n        original_failed_resource_receipts_preserved=True,previous_interrupted_routes_retried=0)")
    source=exact(source,'    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})',
        "    paths += [transport.SOURCE_V3/'transport.py']\n    paths += [transport.RESCUE/name for name in ('RESCUE_JOBS.json','RESULT.json','INPUT_HASHES.json','INTERRUPTED.json','QUARANTINE.json','RESTORATION_EVIDENCE.json','TRAINING_INDEX.jsonl')]\n    for oldshard,gpu in ((0,3),(1,4)):\n        paths += [c.PARALLEL/'production'/f'shard_{oldshard:04d}'/name for name in ('JOBS.json','LEDGER.jsonl','INPUT_LOCK.json')]\n        paths += [c.PARALLEL/'runtime_v2/lanes'/f'gpu_{gpu}/attempt_000'/name for name in ('RESULT.json','RESTORATION.json')]\n    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})")
    return source


def merge_source():
    return exact(_v3_merge_source(),"scope='SELECTED_WAVE_NOT_FULL_POOL'",
        "scope='UNATTEMPTED_RESCUE_SUBSET_ONLY_NOT_ORIGINAL_FULL_SHARDS'")
