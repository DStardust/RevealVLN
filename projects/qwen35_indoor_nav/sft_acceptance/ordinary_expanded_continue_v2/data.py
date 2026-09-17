import importlib.util as _iu
from pathlib import Path as _P
_s = _iu.spec_from_file_location('expanded_reuse_data', _P(__file__).with_name('reuse.py'))
_r = _iu.module_from_spec(_s); _s.loader.exec_module(_r)
_r.execute('data.py', globals())


def load_rows():
    import collections
    snap = LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001'
    seal = json.loads((snap/'SEAL.json').read_text())
    for name, digest in seal.items(): require(sha256(snap/name) == digest, 'EXPANDED_SEAL:'+name)
    result = json.loads((snap/'RESULT.json').read_text())
    require(result['status'] == 'STRICT_MERGED_TRAINING_SNAPSHOT_READY', 'SNAPSHOT_NOT_READY')
    rows = [json.loads(x) for x in (snap/'TRAINING_INDEX.jsonl').read_text().splitlines()]
    split = json.loads((snap/'SPLIT.json').read_text())
    fit = set(split['FIT']); require(fit.isdisjoint(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM']), 'SPLIT_OVERLAP')
    require(len(rows) == 37114 and len({r['record_id'] for r in rows}) == len(rows), 'RECORD_COUNT')
    require(sum(r['decisions'] for r in rows) == 2650347, 'DECISION_COUNT')
    owners = {}; aliases = set()
    for row in rows:
        require(row['split'] == 'FIT' and row['scene_group'] in fit, 'HOUSE_LEAKAGE')
        key = row['physical_source_route_sha256']; owner=(row['sourceRoot'],row['job_id'],row['scene_group'])
        require(owners.setdefault(key, owner) == owner, 'DUPLICATE_PHYSICAL')
        alias=(row['source'],Path(row['policy_file']).name)
        require(alias not in aliases, 'DUPLICATE_ALIAS'); aliases.add(alias)
        root=(LINE.parents[1]/row['sourceRoot']).resolve(strict=True)
        require(root.is_relative_to(LINE), 'SOURCE_SCOPE')
        for field in ('policy','supervision'):
            path=(root/row[field+'_file']).resolve(strict=True)
            require(path.is_relative_to(root), 'PATH_ESCAPE')
            require(len(row[field+'_sha256']) == 64, 'MISSING_CONTENT_HASH')
    require(len(owners) == 25415, 'ROUTE_COUNT')
    return rows, dict(training_index_sha256=sha256(snap/'TRAINING_INDEX.jsonl'),
                     records=len(rows), decisions_per_epoch=2650347, snapshot_seal_sha256=sha256(snap/'SEAL.json'))
