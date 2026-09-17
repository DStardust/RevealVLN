"""CPU-only deterministic next-batch manifests. Never launch simulators or GPUs."""
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
ROOT=LINE.parents[1]
PREVIOUS=HERE.parent/'ordinary_scale_v1'
PILOT=HERE.parent/'ordinary_pilot_v1'

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def digest(path):
    path=Path(path).resolve(strict=True)
    assert path.is_relative_to(ROOT),path
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def save(name,obj):
    path=HERE/name
    assert path.resolve().is_relative_to(HERE)
    content=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if path.exists():
        assert path.read_text()==content,('IMMUTABLE_PLAN_REQUIRES_NEW_VERSION',name)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as handle:handle.write(content)

def route_key(episode):
    fields=('scene_id','start_position','start_rotation','reference_path','goals')
    return hashlib.sha256(canonical({key:episode[key] for key in fields})).hexdigest()

def fair_select(jobs,limit):
    pools=collections.defaultdict(list)
    for job in sorted(jobs,key=lambda j:j['physical_source_route_sha256']):pools[job['scene_id']].append(job)
    selected=[]
    for rank in range(max(map(len,pools.values()),default=0)):
        for house in sorted(pools):
            if rank<len(pools[house]):selected.append(pools[house][rank])
            if len(selected)==limit:return selected
    return selected

def select_sources(pools,excluded,per_source=500):
    selected=[];used=set(excluded);stats={}
    for source in ('R2R','RxR'):
        candidates=[job for job in pools[source] if job['physical_source_route_sha256'] not in used]
        batch=fair_select(candidates,per_source)
        assert len({j['physical_source_route_sha256'] for j in batch})==len(batch)
        used.update(j['physical_source_route_sha256'] for j in batch)
        selected.extend(batch)
        stats[source]=dict(requested_routes=per_source,eligible_after_exclusion=len(candidates),
            selected_routes=len(batch),shortfall_routes=per_source-len(batch))
    return sorted(selected,key=lambda j:(j['scene_id'],j['source'],j['physical_source_route_sha256'])),stats

def partition(jobs,count=4):
    """Indivisible whole-house groups, deterministic balanced route-count assignment."""
    houses=collections.defaultdict(list)
    for job in jobs:houses[job['scene_id']].append(job)
    ranks=sorted(houses,key=lambda house:(-len(houses[house]),
        hashlib.sha256(('Q35N_PARALLEL_V1:'+house).encode()).hexdigest(),house))
    shards=[[] for _ in range(count)];assignment={}
    for house in ranks:
        sid=min(range(count),key=lambda i:(len(shards[i]),i))
        assignment[house]=sid
        shards[sid].extend(houses[house])
    for shard in shards:shard.sort(key=lambda j:(j['scene_id'],j['source'],j['physical_source_route_sha256']))
    return shards,assignment

def main():
    frozen=json.loads((PREVIOUS/'INPUT_LOCK.json').read_text())
    for name,expected in frozen.items():assert digest(ROOT/name)==expected,name
    inputs={name:expected for name,expected in frozen.items()}
    for path in (PREVIOUS/'INPUT_LOCK.json',PILOT/'JOBS.json'):
        inputs[str(path.relative_to(ROOT))]=digest(path)
    pilot=json.loads((PILOT/'JOBS.json').read_text())
    previous=json.loads((PREVIOUS/'JOBS.json').read_text())
    assert len(pilot)==100 and len(previous)==1000
    excluded={j['physical_source_route_sha256'] for j in pilot+previous}
    assert len(excluded)==1100
    split=json.loads((PREVIOUS/'SPLIT_FREEZE.json').read_text())
    fit=set(split['FIT'])
    assert not fit.intersection(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    inventory=json.loads((PREVIOUS/'SOURCE_INVENTORY.json').read_text())
    pools={};source_stats=[]
    for source in inventory['sources']:
        tag=source['source'];path=ROOT/source['path']
        assert digest(path)==source['sha256']
        inputs[source['path']]=source['sha256']
        with gzip.open(path,'rt') as handle:episodes=json.load(handle)['episodes']
        grouped=collections.defaultdict(list);seen_ids=set();english=0
        for episode in episodes:
            eid=str(episode['episode_id'])
            assert eid not in seen_ids;seen_ids.add(eid)
            lang=episode['instruction'].get('language','en')
            if not lang.startswith('en'):continue
            english+=1
            scene=Path(episode['scene_id']).parent.name
            assert Path(episode['scene_id']).parts==('mp3d',scene,scene+'.glb')
            if scene not in fit:continue
            assert episode['reference_path'] and episode['instruction']['instruction_text'].strip()
            grouped[route_key(episode)].append(episode)
        pool=[]
        for key,aliases in sorted(grouped.items()):
            aliases.sort(key=lambda e:int(e['episode_id']))
            for alias in aliases:
                alias['instruction']={k:v for k,v in alias['instruction'].items()
                    if k in ('instruction_text','language','instruction_id')}
            episode=aliases[0];scene=Path(episode['scene_id']).parent.name
            pool.append(dict(job_id=tag.lower()+'_'+key[:20],source=tag,
                source_path=source['path'],source_sha256=source['sha256'],scene_id=scene,split='FIT',
                physical_source_route_sha256=key,episode=episode,instruction_alias_episodes=aliases))
        pools[tag]=pool
        source_stats.append(dict(source=tag,path=source['path'],sha256=source['sha256'],
            raw_episodes=len(episodes),english_episodes=english,fit_unique_physical_routes=len(pool)))
    jobs,selection=select_sources(pools,excluded)
    shards,assignment=partition(jobs)
    chosen={j['physical_source_route_sha256'] for j in jobs}
    assert len(chosen)==len(jobs)<=1000 and not chosen.intersection(excluded)
    assert {j['scene_id'] for j in jobs}<=fit
    save('JOBS.json',jobs)
    save('SPLIT_FREEZE_REFERENCE.json',dict(path=str((PREVIOUS/'SPLIT_FREEZE.json').relative_to(ROOT)),
        sha256=digest(PREVIOUS/'SPLIT_FREEZE.json'),FIT=split['FIT'],
        INTERNAL_DEV=split['INTERNAL_DEV'],INTERNAL_CONFIRM=split['INTERNAL_CONFIRM'],
        no_global_unexposed_claim=True))
    shard_stats=[]
    for sid,shard in enumerate(shards):
        name=f'manifests/shard_{sid:04d}/JOBS.json'
        save(name,shard)
        shard_stats.append(dict(shard_id=sid,jobs_path=name,jobs_sha256=digest(HERE/name),
            output_root=f'production/shard_{sid:04d}',route_count=len(shard),
            instruction_count=sum(len(j['instruction_alias_episodes']) for j in shard),
            houses=sorted({j['scene_id'] for j in shard}),
            source_routes=dict(collections.Counter(j['source'] for j in shard)),executable=False))
    save('EXCLUSION_MANIFEST.json',dict(pilot_candidates_excluded=100,prior_batch_candidates_excluded=1000,
        unique_physical_route_keys_excluded=sorted(excluded),
        exclusion_policy='all previously assigned candidates, regardless of generation/rejection/quarantine status'))
    save('PLAN.json',dict(status='CPU_MANIFESTS_READY_NOT_GENERATED',executable=False,gpu_launcher_present=False,
        jobs=len(jobs),unique_physical_routes=len(chosen),houses=len({j['scene_id'] for j in jobs}),
        instruction_records=sum(len(j['instruction_alias_episodes']) for j in jobs),selection=selection,
        shards=shard_stats,house_to_shard=assignment,partition_rule='whole_house_greedy_route_count_balancing_sha256_tie_break',
        generated_routes=0,generated_action_records=0,scientific_pass=False,official_eval_content_read=False,
        independent_output_roots=True,shared_writer_to_old_batch=False,
        merged_index='merge/TRAINING_INDEX.jsonl',merge_implemented=False,
        merge_required_checks=['all_shards_terminal_or_explicitly_censored','strict_route_export_audit',
            'content_hash_check','disjoint_physical_routes','complete_alias_ownership',
            'relative_references_resolved_under_own_shard','quarantine_and_failure_counts_preserved'],
        source_statistics=source_stats))
    for path in (HERE/'plan.py',HERE/'test_plan.py',HERE/'README.md'):
        inputs[str(path.relative_to(ROOT))]=digest(path)
    save('INPUT_LOCK.json',dict(immutable=inputs,active_ledgers_read=False))
    outputs=['JOBS.json','SPLIT_FREEZE_REFERENCE.json','EXCLUSION_MANIFEST.json','PLAN.json','INPUT_LOCK.json']
    outputs += [s['jobs_path'] for s in shard_stats]
    save('MANIFEST_HASHES.json',{name:digest(HERE/name) for name in outputs})
    print(json.dumps(dict(routes=len(jobs),houses=len(assignment),
        instructions=sum(len(j['instruction_alias_episodes']) for j in jobs),
        selection=selection,shards=shard_stats,generated_routes=0,executable=False),indent=2))

if __name__=='__main__':main()
