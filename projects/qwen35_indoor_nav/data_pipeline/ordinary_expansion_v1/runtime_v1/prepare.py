"""Freeze source/asset/transport inputs without creating any simulator or GPU context."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import transport

SOURCE_JOBS_SHA='e1d79e741c6ba63cc7b85f828dfc1d8d30b4d6dcf9dc35064ab75076cf4b118b'
SOURCE_LOCK_SHA='3b1f3a2d38ec1acf58c029bce1934c9a959a19862ce4c2a90599b6722c8d9adf'

def validate_authorization(auth):
    expected=dict(approved=True,training_allowed=False,project_root=str(c.ROOT),
        holder_identity=str(c.IDENTITIES.relative_to(c.ROOT)),gpu_to_shards={'6':list(range(21))},
        source_root=str(transport.MANIFEST.relative_to(c.ROOT)),source_jobs_sha256=SOURCE_JOBS_SHA,
        source_input_lock_sha256=SOURCE_LOCK_SHA,new_output_parent=str(transport.DATA.relative_to(c.ROOT)),
        max_new_routes=20000,sentinel_required_strict_routes=3,
        shard_wall_seconds={str(s):600 if s==0 else 4200 for s in range(21)},lane_wall_seconds={'6':82800},
        cleanup_margin_seconds=120,transport_upper_seconds=83400,audit_upper_seconds=1800,
        job_max_seconds=85200,queue_max_seconds=86400,max_new_disk_gib=512,per_shard_max_disk_gib=24,
        metadata_max_disk_gib=8,max_worker_ram_gib=12,max_own_gpu_mib=4096,
        stop_before_tail_without_full_phase_budget=True,retry_failed_or_partial=False,
        gpu_accounting_amendment='conservative_max_active_v1',quantitative_memory_limits_unchanged=True,
        raw_gpu_readings_preserved=True,restore_exact_holder=True,split='FIT_ONLY',
        synthetic_official_instructions_not_human_labels=True,original_quality_thresholds_unchanged=True,
        external_process_signals_allowed=False,main_runtime_approval_after_tests_required=True)
    for key,value in expected.items():assert auth[key]==value,('AUTHORIZATION_MISMATCH',key)
    return expected

def source_checks():
    source=transport.MANIFEST
    assert c.sha(source/'JOBS.json')==SOURCE_JOBS_SHA
    assert c.sha(source/'INPUT_LOCK.json')==SOURCE_LOCK_SHA
    lock=c.read(source/'INPUT_LOCK.json')
    for name,sha in lock.items():assert c.sha(c.ROOT/name)==sha,name
    jobs=c.read(source/'JOBS.json');plan=c.read(source/'PLAN.json')
    assert plan['selected_routes']==20000 and plan['selected_aliases']==20000 and plan['selected_houses']==50
    assert plan['executable'] is False and plan['runtime_allowed'] is False
    shards=[]
    for shard,entry in enumerate(plan['shards']):
        assert entry['shard']==shard
        p=source/entry['jobs_path'];assert c.sha(p)==entry['jobs_sha256']
        rows=c.read(p);assert len(rows)==entry['routes'];shards.append(rows)
    assert len(shards)==21 and [j for rows in shards for j in rows]==jobs
    assert len(shards[0])==3 and len({j['scene_id'] for j in shards[0]})==3
    assert [j['official_gt_actions'] for j in shards[0]]==[36,69,52],'FIXED_SENTINEL_SOURCE_NO_SELECTION_CHANGES'
    assert len(jobs)==len({j['job_id'] for j in jobs})==len({j['physical_source_route_sha256'] for j in jobs})==20000
    assert len({j['geometry_sha256'] for j in jobs})==len({j['official_gt_sha256'] for j in jobs})==20000
    excluded=set(c.read(source/'PHYSICAL_EXCLUSION.json')['physical_route_keys'])
    assert len(excluded)==8742 and excluded.isdisjoint(j['physical_source_route_sha256'] for j in jobs)
    fit=set(c.read(c.BASE/'SPLIT_FREEZE.json')['FIT'])
    aliases=[]
    for j in jobs:
        assert j['split']=='FIT' and j['scene_id'] in fit
        assert j['source']=='ENVDROP_OFFICIAL_CE_TRAIN'
        assert j['instruction_alias_episodes']==[j['episode']]
        assert j['episode']['instruction']['language']=='en'
        assert j['full_natural_language_semantics_certified'] is False
        aliases.append((j['source'],str(j['episode']['episode_id'])))
    assert len(set(aliases))==20000
    return jobs,shards,lock

def main():
    assert not (HERE/'INPUT_LOCK.json').exists() and not transport.DATA.exists(),'FRESH_RUNTIME_AND_DATA_REQUIRED'
    auth=validate_authorization(c.read(c.AUTH));jobs,shards,lock=source_checks()
    identities=c.read(c.IDENTITIES)['holders'];assert len(identities)==1 and identities[0]['gpu_device']==6
    assert identities[0].get('project_cache_environment',{})=={},'GPU6_EMPTY_RESTORE_ENV_REQUIRED'
    assert identities[0]['cwd']==str(c.ROOT)
    assert not transport.DATA.is_relative_to(HERE) and transport.DATA.parent==HERE.parent
    previous=c.read(transport.V4/'INPUT_LOCK.json')
    for name,sha in previous.items():assert c.sha(c.ROOT/name)==sha,name
    lock.update(previous)
    assets={}
    for house in sorted({j['scene_id'] for j in jobs}):
        for suffix in ('.glb','.navmesh','.house','_semantic.ply'):
            p=c.ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}{suffix}'
            assets[str(p.relative_to(c.ROOT))]=c.sha(p)
    c.save(HERE/'ASSETS.json',assets)
    outputs=[]
    for shard,rows in enumerate(shards):
        root=c.shard_root(shard);root.mkdir(parents=True,exist_ok=False);(root/'quality').mkdir()
        for name,value in {'JOBS.json':rows,'SPLIT_FREEZE.json':c.read(c.BASE/'SPLIT_FREEZE.json'),
            'SOURCE_INVENTORY.json':c.read(transport.MANIFEST/'SOURCE_INVENTORY.json')}.items():
            c.save(root/name,value);outputs.append(root/name)
    cfg=dict(node='Q35N_ORDINARY_OFFICIAL_ENVDROP_RUNTIME_V1',runtime_allowed=False,executable=False,
        training_allowed=False,gpu_to_shards=auth['gpu_to_shards'],shard_seconds=auth['shard_wall_seconds'],
        lane_seconds=auth['lane_wall_seconds'],cleanup_margin_seconds=120,
        per_shard_max_bytes=24*1024**3,metadata_and_merge_max_bytes=8*1024**3,total_max_bytes=512*1024**3,
        worker_ram_bytes=12*1024**3,own_gpu_mib=4096,routes=20000,aliases=20000,scene_assets=len(assets),
        source_plan_sha256=c.sha(transport.MANIFEST/'PLAN.json'),source_jobs_sha256=SOURCE_JOBS_SHA,
        source_grade='OFFICIAL_SYNTHETIC_ENGLISH_CE_PORT_NOT_HUMAN',sentinel_required_strict_routes=3,
        sentinel_original_gt_actions=[36,69,52],sentinel_different_fit_houses=True,
        selection='PREEXISTING_FROZEN_HOUSE_ROUND_ROBIN_AND_PHYSICAL_HASH_NOT_LENGTH_STRATIFIED',
        original_reference_gt_actions=1228720,generated_actions=0,
        full_natural_language_semantics_certified=False,metadata_and_production_disjoint=True,
        output_namespace=str(transport.DATA.relative_to(c.ROOT)),disk_census_unchanged=True,
        quantitative_memory_limits_unchanged=True,gpu_accounting_guard_amended=True,
        gpu_accounting_amendment='conservative_max_active_v1',raw_gpu_readings_preserved=True,
        stop_before_tail_without_full_phase_budget=True,retry_failed_or_partial=False,
        restore_exact_holder=True,scientific_pass=False)
    c.save(HERE/'PREPARED_CONFIG.json',cfg)
    paths=list(HERE.glob('*.py'))+[HERE/'ASSETS.json',HERE/'PREPARED_CONFIG.json',c.AUTH,c.IDENTITIES,
        transport.MANIFEST/'INPUT_LOCK.json',transport.V4/'INPUT_LOCK.json']+outputs
    paths += [transport.SOURCE/name for name in transport.EXPECTED]
    paths += [transport.V4/'telemetry.py']
    lock.update(assets)
    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})
    c.save(HERE/'INPUT_LOCK.json',lock)
    for shard in range(21):c.save(c.shard_root(shard)/'INPUT_LOCK.json',lock)
    scheduler=dict(id='ordinary_envdrop_first_20000_gpu6_v1',gpu=6,
        command=[str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-B',str(HERE/'run.py'),'--gpu','6'],
        max_seconds=85200,production_timeout_seconds=83400,audit_timeout_seconds=1800,
        completion=[dict(path=str(HERE/'lanes/gpu_6/attempt_000/RESULT.json'),equals={'error':None,'restoration.restored':True})],
        audit_command=[str(c.ENV/'bin/python3'),'-I','-B',str(HERE/'merge.py')],
        source_jobs_sha256=SOURCE_JOBS_SHA,input_lock_sha256=c.sha(HERE/'INPUT_LOCK.json'),executable=False)
    c.save(HERE/'SCHED_JOB.json',scheduler)
    print(json.dumps(dict(status='CPU_PREPARED_MAIN_APPROVAL_REQUIRED',routes=20000,assets=len(assets),
        immutable_inputs=len(lock),input_lock_sha256=c.sha(HERE/'INPUT_LOCK.json'),approval_value=c.approval_value(6),executable=False)))

if __name__=='__main__':main()
