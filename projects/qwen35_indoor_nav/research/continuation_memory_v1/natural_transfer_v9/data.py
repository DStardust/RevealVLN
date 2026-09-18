"""Freeze ordinary, real teacher trajectories without looking at model outcomes."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
ROOT = LINE.parents[1]
SNAPSHOT = LINE / 'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(rows, special_split):
    houses = {}
    for row in sorted(rows, key=lambda r: r['record_id']):
        if row['source'] == 'R2R':
            houses.setdefault(row['scene_group'], {}).setdefault(row['physical_source_route_sha256'], row)
    check = {h for h, split in special_split.items() if split == 'check'}
    extra = sorted(set(houses) - set(special_split), key=lambda h: hashlib.sha256(('v9-check:' + h).encode()).hexdigest())
    check.update(extra[:3])
    chosen = []
    for house in sorted(houses):
        ordered = sorted(houses[house].values(), key=lambda r: hashlib.sha256(('v9-natural:' + r['physical_source_route_sha256']).encode()).hexdigest())
        chosen.extend(dict(row=row, partition='check' if house in check else 'fit') for row in ordered[:12])
    return chosen


def prepare():
    for name, digest in json.loads((SNAPSHOT / 'SEAL.json').read_text()).items():
        assert sha(SNAPSHOT / name) == digest, 'SNAPSHOT_CHANGED:' + name
    rows = [json.loads(line) for line in (SNAPSHOT / 'TRAINING_INDEX.jsonl').read_text().splitlines()]
    split = json.loads((SNAPSHOT / 'SPLIT.json').read_text())
    special = json.loads((HERE.parent / 'multifamily_v7/DATA.json').read_text())
    chosen = select(rows, special['audit']['house_split'])
    adapter = load('v9_ordinary_adapter', LINE / 'sft_acceptance/ordinary_baseline_v2/data.py')
    records = []
    offset = 0
    for item in chosen:
        row = item['row']
        assert row['split'] == 'FIT' and row['scene_group'] in split['FIT']
        assert row['scene_group'] not in split['INTERNAL_DEV'] + split['INTERNAL_CONFIRM']
        record = adapter.OrdinaryRecord(row)
        assert record.actions[-1] == 'STOP' and 'STOP' not in record.actions[:-1]
        records.append(dict(**item, features=list(range(offset, offset + len(record))),
            targets=[adapter.ACTIONS.index(action) for action in record.actions],
            instruction_sha256=hashlib.sha256(record.instruction.encode()).hexdigest()))
        offset += len(record)
    fit_houses = {r['row']['scene_group'] for r in records if r['partition'] == 'fit'}
    check_houses = {r['row']['scene_group'] for r in records if r['partition'] == 'check'}
    assert fit_houses.isdisjoint(check_houses)
    assert not fit_houses.intersection(h for h,s in special['audit']['house_split'].items() if s == 'check')
    assert not check_houses.intersection(h for h,s in special['audit']['house_split'].items() if s == 'fit')
    output = dict(records=records, decisions=offset, source_index_sha256=sha(SNAPSHOT / 'TRAINING_INDEX.jsonl'),
        source_seal_sha256=sha(SNAPSHOT / 'SEAL.json'),
        selection='R2R only; one lexicographic instruction per physical route; first12 sha256(v9-natural:route) per house; all available if fewer',
        split='Preserve two SEE2 CHECK houses; three additional houses by sha256(v9-check:house); remaining original FIT houses train',
        fit_houses=sorted(fit_houses), check_houses=sorted(check_houses),
        no_navigation_evaluation_used_for_training=True, backbone_exposed_houses=True, independent_blind_test=False,
        original_labels_changed=False, new_simulator_decisions=0,
        policy_fields=['instruction', 'last_at_most_two_RGB', 'last_at_most_eight_executed_motion_actions'])
    with (HERE / 'DATA.json').open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(records=len(records), decisions=offset, fit_houses=len(fit_houses), check_houses=len(check_houses))))


if __name__ == '__main__':
    prepare()
