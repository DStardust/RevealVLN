"""Frozen official-train manifest, no simulator or GPU imports."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
ROOT = LINE.parents[1]
PHYSICAL = ('scene_id', 'start_position', 'start_rotation', 'reference_path', 'goals')

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def route_key(e):
    return hashlib.sha256(canonical({k:e[k] for k in PHYSICAL})).hexdigest()

def sha(p):
    p = p.resolve(strict=True)
    assert p.is_relative_to(ROOT), p
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def save(name, x):
    content = json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    p = OUT/name
    if p.exists():
        assert p.read_text() == content, ('VERSION_REQUIRED', name)
    else:
        with p.open('x') as f:f.write(content)

def split_houses(houses, pilot):
    rest = sorted(set(houses)-set(pilot), key=lambda h:hashlib.sha256(('Q35N_SCALE_SPLIT_V1:'+h).encode()).hexdigest())
    assert len(rest)>=10
    return {'FIT':sorted(set(houses)-set(rest[:10])),
            'INTERNAL_DEV':sorted(rest[:5]), 'INTERNAL_CONFIRM':sorted(rest[5:10])}

def fair_select(jobs, limit):
    pools = collections.defaultdict(list)
    for j in sorted(jobs, key=lambda j:j['physical_source_route_sha256']):
        pools[j['scene_id']].append(j)
    selected=[]
    for rank in range(max(map(len,pools.values()), default=0)):
        for house in sorted(pools):
            if rank<len(pools[house]): selected.append(pools[house][rank])
            if len(selected)==limit:return selected
    return selected

def main():
    old=LINE/'data_pipeline/ordinary_pilot_v1'
    prior=json.loads((old/'SPLIT_FREEZE.json').read_text())
    houses=set(prior['fit_pilot']+prior['reserved_unassigned'])
    groups=split_houses(houses,prior['fit_pilot'])
    save('SPLIT_FREEZE.json',dict(groups,prior_fit_pilot=prior['fit_pilot'],
        grouping='whole_house_all_sources_histories_continuations_and_wordings',
        split_rule='sha256(Q35N_SCALE_SPLIT_V1:house); prior FIT immutable; first 5 dev next 5 confirm',
        legacy_status=prior['legacy_status'],no_global_unexposed_claim=True,
        official_eval_content_read=False))
    old_keys={j['physical_source_route_sha256'] for j in json.loads((old/'JOBS.json').read_text())}
    sources=[('R2R','data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz'),
             ('RxR','data/phase0/raw/rxr_vlnce_v0/train/train_guide.json.gz')]
    stats=[]; manifest=[];chosen=[];global_keys=set();all_jobs={}
    for tag,path in sources:
        p=ROOT/path; digest=sha(p)
        if tag=='R2R':assert digest=='f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34'
        with gzip.open(p,'rt') as f:es=json.load(f)['episodes']
        grouped=collections.defaultdict(list);languages=collections.Counter()
        ids=set();selected_language=0
        for e in es:
            eid=str(e['episode_id']);assert eid not in ids;ids.add(eid)
            lang=e['instruction'].get('language','en');languages[lang]+=1
            if not lang.startswith('en'):continue
            selected_language+=1
            scene=Path(e['scene_id']).parent.name
            assert scene in houses and Path(e['scene_id']).parts==('mp3d',scene,scene+'.glb')
            assert e['reference_path'] and e['instruction']['instruction_text'].strip()
            key=route_key(e);grouped[key].append(e)
            group=next(g for g,ss in groups.items() if scene in ss)
            manifest.append(dict(source=tag,source_path=path,source_sha256=digest,episode_id=eid,
                scene_id=scene,split=group,language=lang,physical_source_route_sha256=key,
                reference_waypoints=len(e['reference_path']),record_type='SOURCE_CANDIDATE',
                runtime_certified=False))
        pool=[]
        for key,aliases in sorted(grouped.items()):
            aliases.sort(key=lambda e:int(e['episode_id']))
            e=aliases[0];scene=Path(e['scene_id']).parent.name
            global_keys.add(key)
            if scene not in groups['FIT'] or key in old_keys:continue
            j=dict(job_id=tag.lower()+'_'+key[:20],source=tag,source_path=path,source_sha256=digest,
                scene_id=scene,split='FIT',physical_source_route_sha256=key,episode=e,
                instruction_alias_episodes=aliases)
            # Metadata irrelevant to the policy is not copied into policy files.
            for a in j['instruction_alias_episodes']:
                a['instruction']={k:v for k,v in a['instruction'].items()
                                  if k in ('instruction_text','language','instruction_id')}
            pool.append(j)
        all_jobs[tag]=pool
        stats.append(dict(source=tag,path=path,bytes=p.stat().st_size,sha256=digest,
            raw_episodes=len(es),languages=dict(languages),selected_english_episodes=selected_language,
            selected_language_physical_routes=len(grouped),fit_new_physical_routes=len(pool),
            selected_language_houses=len({Path(v[0]['scene_id']).parent.name for v in grouped.values()})))
        del es
    used=set()
    for tag in ('R2R','RxR'):
        candidates=[j for j in all_jobs[tag] if j['physical_source_route_sha256'] not in used]
        batch=fair_select(candidates,500)
        chosen+=batch;used.update(j['physical_source_route_sha256'] for j in batch)
    chosen.sort(key=lambda j:(j['scene_id'],j['source'],j['physical_source_route_sha256']))
    assert len(chosen)==1000 and len(used)==1000
    save('JOBS.json',chosen)
    counts=collections.Counter((r['source'],r['split']) for r in manifest)
    save('SOURCE_INVENTORY.json',dict(sources=stats,
        source_rows_by_split={a+':'+b:n for (a,b),n in sorted(counts.items())},
        unique_physical_routes_across_selected_sources=len(global_keys),batch_routes=len(chosen),
        batch_houses=len({j['scene_id'] for j in chosen}),
        batch_instruction_records=sum(len(j['instruction_alias_episodes']) for j in chosen),
        generated_records=0,official_validation_test_read=False))
    manifest_text=''.join(canonical(r).decode()+'\n' for r in manifest)
    m=OUT/'SOURCE_MANIFEST.jsonl'
    if m.exists():assert m.read_text()==manifest_text
    else:
        with m.open('x') as f:f.write(manifest_text)
    files={}
    for name in ('JOBS.json','SPLIT_FREEZE.json','SOURCE_INVENTORY.json','SOURCE_MANIFEST.jsonl','SPEC_ZH.md',
                 'prepare.py','worker.py','audit.py','run.py'):
        files[str((OUT/name).relative_to(ROOT))]=sha(OUT/name)
    for name in ('worker.py','loader.py'):
        files[str((old/name).relative_to(ROOT))]=sha(old/name)
    files[str((LINE/'MAINLINE_FREEZE_V3.md').relative_to(ROOT))]=sha(LINE/'MAINLINE_FREEZE_V3.md')
    save('INPUT_LOCK.json',files)
    print(json.dumps(json.loads((OUT/'SOURCE_INVENTORY.json').read_text()),indent=2))

if __name__=='__main__':main()

