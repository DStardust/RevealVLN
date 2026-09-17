"""Build the frozen FIT diagnostic panel: stratified fixed decisions.

Covers all four actions and many FIT houses/routes. Deterministic given the
sample index and seed. CPU only; no model, no GPU, no pixels decoded.
"""
import json
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402

PANEL_SIZE = 512
QUOTA = {0: 205, 1: 128, 2: 128, 3: 51}  # F/L/R/STOP
SEED = 1109


def main():
    rows, report = data.load_rows()
    acc = json.loads((HERE / 'acceptance/ACCEPTANCE.json').read_text())
    samples = data.load_sample_index(HERE / 'SAMPLE_INDEX.jsonl', acc['sample_index_sha256'],
                                     sum(r['decisions'] for r in rows))
    by_class = {0: [], 1: [], 2: [], 3: []}
    for i, s in enumerate(samples):
        by_class[s['target']].append(i)
    rng = random.Random(SEED)
    picked = []
    for cls, quota in QUOTA.items():
        pool = by_class[cls]
        data.require(len(pool) >= quota, 'CLASS_POOL_TOO_SMALL:%d' % cls)
        chosen = rng.sample(pool, quota)
        picked.extend(chosen)
    rng.shuffle(picked)
    houses = {rows[samples[i]['record_idx']]['scene_group'] for i in picked}
    routes = {(rows[samples[i]['record_idx']]['physical_source_route_sha256']) for i in picked}
    out = HERE / 'PANEL_FIT.jsonl'
    data.require(not out.exists(), 'PANEL_EXISTS')
    with out.open('x') as stream:
        for i in picked:
            s = samples[i]
            stream.write(json.dumps([s['record_idx'], s['t']]) + '\n')
    receipt = dict(status='PANEL_FIT_BUILT', unix=time.time(), seed=SEED, size=len(picked),
                   quota=QUOTA, houses=len(houses), routes=len(routes),
                   sha256=data.sha256(out), sample_index_sha256=acc['sample_index_sha256'],
                   snapshot=report['training_index_sha256'],
                   note='fixed teacher-forced diagnostic panel; never trained on differently; not DEV')
    with (HERE / 'PANEL_FIT_BUILD.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(dict(size=len(picked), houses=len(houses), routes=len(routes),
                          sha256=receipt['sha256'][:16])))


if __name__ == '__main__':
    main()
