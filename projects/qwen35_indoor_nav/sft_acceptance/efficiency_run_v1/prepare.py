"""CPU selection and source freeze. No model imports or GPU operations."""
import hashlib
import json
from pathlib import Path
import random

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
OLD = OUT.parent / 'v1'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def schedule(rows, rank, chunks=800):
    rng = random.Random(1109 + rank)
    indices = list(range(rank, len(rows), 2))
    result = []
    epoch = 0
    while len(result) < chunks:
        order = list(indices)
        rng.shuffle(order)
        for i in order:
            for start in range(0, rows[i]['decisions'], 4):
                result.append(dict(row=i, start=start, end=min(start+4, rows[i]['decisions']), epoch=epoch))
                if len(result) == chunks:
                    return result
        epoch += 1

def main():
    assert not (OUT/'LOCK.json').exists(), 'No overwrite/retry'
    sealed = 0
    for line in (OLD/'SHA256SUMS').read_text().splitlines():
        h, rel = line.split('  ', 1)
        p = (OLD/rel).resolve()
        assert p.is_relative_to(OLD.resolve()) and digest(p) == h, rel
        sealed += 1
    split = json.loads((OLD/'SPLIT.json').read_text())
    rows, seen = [], {}
    for row in split['train']:
        jobs = seen.setdefault(row['scene_group'], set())
        if row['job_id'] not in jobs and len(jobs) < 2:
            rows.append(row)
            jobs.add(row['job_id'])
    assert len(rows) == 10 and len(seen) == 5 and all(len(s)==2 for s in seen.values())
    assert not {r['job_id'] for r in rows} & {r['job_id'] for r in split['seen_house_route_dev']}
    data = LINE/'data_pipeline/ordinary_pilot_v1'
    sources = [OLD/'SHA256SUMS', OLD/'SPLIT.json', OLD/'EXPERIMENT_SPEC.json', OLD/'result.json',
               OLD/'recovery_r1/policy.py', OUT.parent/'d1_v1/worker.py', OUT.parent/'efficiency_v1/frozen_cache.py', OLD/'checkpoints/initial.pt', OLD/'recovery_r1/checkpoints/terminal.pt',
               LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json']
    for row in rows:
        p, s = data/row['policy_file'], data/row['supervision_file']
        sources.extend([p, s])
        policy = json.loads(p.read_text())
        actions = json.loads(s.read_text())['actions']
        assert len(actions)==row['decisions']==len(policy['rgb_sequence'])
        assert actions[-1]=='STOP' and 'STOP' not in actions[:-1]
        sources.extend((data/row['rgb_reference_root']/ref).resolve() for ref in policy['rgb_sequence'])
    with (OUT/'SUBSET.json').open('x') as f:
        json.dump(dict(rows=rows, schedules=[schedule(rows,r) for r in range(2)]),f,indent=2)
    lock = dict(original_seal_files_verified=sealed, sources={str(p.relative_to(LINE)):digest(p) for p in set(sources)},
                code={p.name:digest(p) for p in OUT.iterdir() if p.suffix in {'.py','.md'}}, subset_sha256=digest(OUT/'SUBSET.json'))
    with (OUT/'LOCK.json').open('x') as f: json.dump(lock,f,indent=2)
    print(json.dumps(dict(routes=[r['job_id'] for r in rows], decisions=sum(r['decisions'] for r in rows),
                         source_files=len(lock['sources']), old_seal_verified=sealed)))

if __name__ == '__main__': main()

