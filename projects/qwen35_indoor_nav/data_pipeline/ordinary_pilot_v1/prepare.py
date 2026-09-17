import collections
import gzip
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
ROOT = LINE.parents[1]


def guarded(path):
    path = path.resolve()
    assert path.is_relative_to(ROOT), path
    return path


def sha(path):
    h = hashlib.sha256()
    with guarded(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    content = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    path = OUT/name
    if path.exists():
        assert path.read_text() == content, f'Use new version: {name}'
    else:
        with path.open('x') as f: f.write(content)


def main():
    source = guarded(ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz')
    assert sha(source) == 'f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34'
    with gzip.open(source, 'rt') as f: episodes = json.load(f)['episodes']
    candidates = [json.loads(s) for s in (OUT.parent/'v1/SOURCE_CANDIDATES.jsonl').read_text().splitlines()]
    scenes = {e['scene_id'] for e in candidates}
    mentions = collections.defaultdict(list)
    ledger_records = []
    def strings(x):
        if isinstance(x, str): yield x
        elif isinstance(x, dict):
            for v in x.values(): yield from strings(v)
        elif isinstance(x, list):
            for v in x: yield from strings(v)
    for p in sorted((ROOT/'artifacts/experiments').glob('*/SCENE_EXPOSURE.json')):
        p = guarded(p)
        obj = json.loads(p.read_text())
        found = sorted(scenes & set(strings(obj)))
        ledger_records.append({'path':str(p.relative_to(ROOT)), 'sha256':sha(p), 'mentioned_scenes':found})
        for s in found: mentions[s].append(str(p.relative_to(ROOT)))
    audit_path = guarded(ROOT/'artifacts/experiments/B27_B_DISJOINT_REPLICATION_V1/B_EXPOSURE_AUDIT.json')
    audit = json.loads(audit_path.read_text())
    positives = collections.defaultdict(list)
    for w in audit['witnesses']:
        if w['kind'] != 'actual_B_trajectory': continue
        p = guarded(ROOT/w['path'])
        if p.is_file():
            for s in w['scenes']:
                if s in scenes: positives[s].append({'path':w['path'], 'sha256':sha(p)})
    grouped = collections.defaultdict(dict)
    aliases = collections.defaultdict(list)
    for row in sorted(candidates, key=lambda x:int(x['episode_id'])):
        key = row['physical_source_route_sha256']
        grouped[row['scene_id']].setdefault(key, row)
        aliases[key].append(row['episode_id'])
    selected = sorted(s for s in positives if len(grouped[s])>=20)[:5]
    assert len(selected)==5
    jobs = []
    by_id = {str(e['episode_id']):e for e in episodes}
    for s in selected:
        rows = sorted(grouped[s].values(), key=lambda x:int(x['episode_id']))[:20]
        for row in rows:
            jobs.append({'job_id':f'route_{len(jobs):03d}', 'scene_id':s,
                         'physical_source_route_sha256':row['physical_source_route_sha256'],
                         'episode':by_id[row['episode_id']],
                         'instruction_alias_episodes':[by_id[eid] for eid in aliases[row['physical_source_route_sha256']]]})
    save('EXPOSURE_AUDIT.json', {'scope':'18 legacy scene ledgers plus positive actual-trajectory witnesses from B27; not exhaustive unseen proof',
         'ledgers':ledger_records, 'positive_audit_source':str(audit_path.relative_to(ROOT)),
         'positive_audit_sha256':sha(audit_path), 'actual_trajectory_witnesses':dict(positives),
         'no_global_unseen_claim':True})
    save('SPLIT_FREEZE.json', {'fit_pilot':selected,
         'reserved_unassigned':sorted(scenes-set(selected)), 'dev':[], 'confirm':[],
         'grouping':'whole raw house including all histories/continuations/wordings',
         'official_eval_content_read':False, 'no_clean_confirm_claim':True,
         'legacy_status':{s:('POSITIVE_TRAJECTORY_EVIDENCE' if s in positives else 'LEDGER_MENTION' if s in mentions else 'UNKNOWN') for s in sorted(scenes)}})
    save('JOBS.json', jobs)
    files=[]
    for s in selected:
        for suffix in ('.glb','.navmesh','.house','_semantic.ply'):
            p=guarded(ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{s}/{s}{suffix}')
            files.append({'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size, 'sha256':sha(p)})
    save('ASSET_LOCK.json', {'source_path':str(source.relative_to(ROOT)), 'source_sha256':sha(source), 'files':files})
    save('PREPARATION_RESULT.json', {'candidate_routes':len(jobs),'houses':selected,
         'instruction_aliases':sum(len(j['instruction_alias_episodes']) for j in jobs),
         'runtime_replays':0,'preparation_pass':True})
    print(json.dumps({'jobs':len(jobs), 'houses':selected},indent=2))


if __name__=='__main__': main()
