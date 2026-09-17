"""Freeze a prediction-independent R2R diagnostic and audit causal label binding."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
SNAP = LINE / 'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001'
ACTIONS = ['move_forward', 'turn_left', 'turn_right', 'STOP']


def digest(value):
    return hashlib.sha256(value).hexdigest()


def rank(value):
    return digest(('1209:' + value).encode())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    assert not (HERE / 'PREPARATION.json').exists(), 'Already prepared'
    blob = (SNAP / 'TRAINING_INDEX.jsonl').read_bytes()
    assert digest(blob) == json.loads((SNAP / 'SEAL.json').read_text())['TRAINING_INDEX.jsonl']
    rows = [json.loads(line) for line in blob.splitlines()]
    fit = set(json.loads((SNAP / 'SPLIT.json').read_text())['FIT'])
    scenes = sorted({r['scene_group'] for r in rows if r['source'] == 'R2R'}, key=rank)
    groups = {'train': scenes[:12], 'check': scenes[12:16]}
    assert len(scenes) >= 16 and set(scenes) <= fit
    seen = {}
    conflicts = defaultdict(list)
    counts = Counter()
    bindings = Counter()
    for i, row in enumerate(rows):
        if row['source'] != 'R2R':
            continue
        assert row['split'] == 'FIT' and row['instruction_provenance'] == 'official_human_instruction'
        source = (ROOT / row['sourceRoot']).resolve()
        assert source.is_relative_to(LINE)
        blobs = [(source / row[k + '_file']).read_bytes() for k in ('policy', 'supervision')]
        assert all(digest(b) == row[k + '_sha256'] for b, k in zip(blobs, ('policy', 'supervision')))
        policy, supervision = map(json.loads, blobs)
        assert set(policy) == {'instruction', 'rgb_sequence'}
        actions = [a if a in ACTIONS else a.lower() for a in supervision['actions']]
        refs = policy['rgb_sequence']
        assert len(refs) == len(actions) == row['decisions']
        assert actions[-1] == 'STOP' and 'STOP' not in actions[:-1]
        assert all(a in ACTIONS for a in actions)
        frames = supervision['frames']
        assert len(frames) == len(refs)
        assert all(Path(ref).stem == f['rgb_sha256'] for ref, f in zip(refs, frames))
        counts['records'] += 1
        for t, action in enumerate(actions):
            payload = [policy['instruction'], [Path(r).stem for r in refs[max(0, t-1):t+1]], actions[max(0, t-8):t]]
            key = digest(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode())
            entry = dict(input_id=key, record_idx=i, t=t, scene=row['scene_group'],
                         route=row['physical_source_route_sha256'], target=ACTIONS.index(action))
            if key in seen:
                counts['duplicate_occurrences'] += 1
                if entry['target'] != seen[key]['target'] or key in conflicts:
                    if key not in conflicts:
                        conflicts[key].append(seen[key])
                    conflicts[key].append(entry)
            else:
                seen[key] = entry
            counts['decisions'] += 1
            if t + 1 < len(frames):
                distance = sum((x-y)**2 for x, y in zip(frames[t]['position'], frames[t+1]['position']))**.5
                bindings['turns' if action.startswith('turn') else 'moves'] += 1
                if action.startswith('turn'):
                    assert distance <= 1e-5, 'TURN_FRAME_ALIGNMENT'
                else:
                    horizontal = sum((frames[t]['position'][j]-frames[t+1]['position'][j])**2 for j in (0,2))**.5
                    bindings['moves_3d_over_0251'] += distance > .251
                    assert horizontal <= .251, 'MOVE_HORIZONTAL_FRAME_ALIGNMENT'
        if counts['records'] % 1000 == 0:
            print(json.dumps(dict(records=counts['records'], decisions=counts['decisions'])), flush=True)
    selected = []
    for group, houses in groups.items():
        for target in range(4):
            per_house = {house: sorted([x for key, x in seen.items() if key not in conflicts
                and x['scene'] == house and x['target'] == target], key=lambda x: rank(x['input_id'])) for house in houses}
            number = 64 if group == 'train' else 32
            offsets = Counter()
            while sum(offsets.values()) < number:
                before = sum(offsets.values())
                for house in houses:
                    if offsets[house] < len(per_house[house]) and sum(offsets.values()) < number:
                        selected.append(dict(per_house[house][offsets[house]], group=group))
                        offsets[house] += 1
                assert sum(offsets.values()) > before, 'INSUFFICIENT_REAL_SAMPLES'
    for group in groups:
        assert len({x['scene'] for x in selected if x['group'] == group}) == len(groups[group])
    assert {x['route'] for x in selected if x['group']=='train'}.isdisjoint(x['route'] for x in selected if x['group']=='check')
    with (HERE / 'INPUTS.jsonl').open('x') as inputs, (HERE / 'SUPERVISION.jsonl').open('x') as labels:
        for x in selected:
            target = x.pop('target')
            labels.write(json.dumps(dict(input_id=x['input_id'], target=target)) + '\n')
            inputs.write(json.dumps(x) + '\n')
    write('CONFLICTS.json', dict(conflicts))
    write('PREPARATION.json', dict(status='CPU_BINDING_CHECKED_PENDING_PIXEL_AND_MODEL_ENTRY',
        snapshot_sha256=digest(blob), groups=groups, counts=dict(counts), motion_bindings=dict(bindings),
        unique_inputs=len(seen), conflicting_inputs=len(conflicts), excluded_conflict_occurrences=sum(map(len,conflicts.values())),
        selected=len(selected), class_counts={'train':[64]*4,'check':[32]*4},
        input_sha256=digest((HERE/'INPUTS.jsonl').read_bytes()), label_sha256=digest((HERE/'SUPERVISION.jsonl').read_bytes()),
        original_frame_target_index='observation[t] -> actions[t]; executed=actions[max(0,t-8):t]',
        selection_uses_predictions=False, official_eval_read=False, navigation_gain=None))
    print((HERE / 'PREPARATION.json').read_text())


if __name__ == '__main__':
    main()
