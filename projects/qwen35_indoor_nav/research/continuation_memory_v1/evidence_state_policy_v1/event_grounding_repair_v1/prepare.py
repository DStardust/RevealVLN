"""Bind the unchanged teacher pool and the previously audited real event labels."""
from collections import Counter
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import *
from event_loss import Pool, ROLES


def copy_bound(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha(src) != sha(dst):
            raise ValueError('COPY_CHANGED:' + str(dst))
    else:
        shutil.copyfile(src, dst)


def main(run):
    run = run.resolve()
    cfg = config(run)
    bound = read(CPU/'BINDING.json')
    for path, expected in bound['files'].items():
        if sha(Path(path)) != expected:
            raise ValueError('ORIGINAL_ASSET_CHANGED:' + path)
    data = read(CPU/'DATA.json')
    ordinary = PARENT.parent/'natural_transfer_v9/DATA.json'
    dev = {f['house'] for f in data['families'] if f['split'] == 'DEV'}
    if dev & {r['row']['scene_group'] for r in read(ordinary)['records'] if r['partition'] == 'fit'}:
        raise ValueError('ORDINARY_DEV_HOUSE_LEAK')
    copies = {CPU/'DATA.json': run/'DATA.json', CPU/'SCHEDULES.json': run/'SCHEDULES.json',
              ordinary: run/'ORDINARY_DATA.json',
              Path(bound['feature_result']['ordinary_path']): run/'features/ORDINARY_FEATURES.pt',
              CONTROL/'features/FEATURES.pt': run/'features/FEATURES.pt',
              CONTROL/'features/FEATURE_RESULT.json': run/'features/FEATURE_RESULT.json'}
    if sha(Path(bound['feature_result']['ordinary_path'])) != bound['feature_result']['ordinary_sha256']:
        raise ValueError('ORDINARY_CACHE_CHANGED')
    for path in (CONTROL/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
        copies[path] = run/path.relative_to(CONTROL)
        copies[path.parent/'STATE_SEAL.json'] = run/path.parent.relative_to(CONTROL)/'STATE_SEAL.json'
    for src, dst in copies.items():
        copy_bound(src, dst)
    copy_bound(CPU/'DATA.json', run/'WARMUP_DATA.json')
    source = PARENT/'recovery_localization_v1/runs/diagnosis_001/EVENT_INDEX.json'
    index = read(source)
    if index['source_data_sha256'] != sha(CPU/'DATA.json'):
        raise ValueError('EVENT_INDEX_SOURCE_CHANGED')
    records = [r for r in index['records'] if r['split'] == 'FIT']
    pool = Pool(records)
    expected = {}
    action_counts = Counter()
    for family in data['families']:
        if family['split'] != 'FIT':
            continue
        for row in family['sequences']:
            for feature, targets, masks in zip(row['features'], row['event_targets'], row['event_masks']):
                for role in range(2):
                    if masks[role]:
                        key = (feature, ROLES[role]); value = (int(targets[role]), family['house'])
                        if key in expected and expected[key] != value:
                            raise ValueError('INCONSISTENT_INPUT_EVENT_LABEL')
                        expected[key] = value
            for action, mask in zip(row['targets'], row['action_masks']):
                if mask:
                    action_counts[str(action)] += 1
    actual = {(r['feature_index'], r['role']): (r['target'], r['house']) for r in records}
    if actual != expected:
        raise ValueError('AUDITED_EVENT_POOL_DIFFERS_FROM_ORIGINAL_LABELS')
    for row in records:
        if row['input_key'] != data['features'][row['feature_index']]['key']:
            raise ValueError('INPUT_KEY_MISMATCH')
    houses = sorted({r['house'] for r in records})
    if len(houses) != 4 or dev & set(houses):
        raise ValueError('HOUSE_SPLIT_CHANGED')
    immutable(run/'EVENT_POOL.json', dict(records=records, source=str(source), source_sha256=sha(source),
        policy_input='Original causal Qwen features only. All metadata is supervision/audit only.',
        training_split='FIT', new_physical_executions=0))
    immutable(run/'EVENT_AUDIT.json', dict(fold_houses=houses, unique_fit_input_role_labels=len(records),
        cells=[dict(house=k[0], role=k[1], n=len(ids), positive=sum(records[i]['target'] for i in ids))
               for k, ids in sorted(pool.cells.items())],
        original_action_counts=dict(action_counts), original_action_masks_and_targets_unchanged=True,
        positive_negative_reweighted=False, loss_mass_sum=sum(pool.mass()),
        normalization='Equal house-role cells, empirical unique-input prevalence inside each cell.',
        original_event_index_sha256=sha(source), dev_not_used_in_probe=True,
        scope='Existing exposed data, not newly collected independent evidence.'))
    from evaluate_continuations import registry_value
    registry = registry_value(data['raw_families'], cfg)
    prior = read(PARENT/'teacher_alignment_v1/runs/aligned_001/EVALUATION_REGISTRY.json')
    if registry['conditions'] != prior['conditions'] or len(registry['slots']) != cfg['planned_rollouts']:
        raise ValueError('REGISTERED_DEV_CONDITIONS_CHANGED')
    immutable(run/'EVALUATION_REGISTRY.json', registry)
    immutable(run/'DATA_AUDIT.json', dict(data_sha256=sha(run/'DATA.json'), original_data_sha256=sha(CPU/'DATA.json'),
        registry_sha256=sha(run/'EVALUATION_REGISTRY.json'), unchanged_original_pool=True,
        fit_variants=59, fit_parents=32, fit_houses=4, dev_variants=16, dev_parents=8, dev_houses=1,
        cache_regenerated=False, independent_test_accessed=False, new_physical_executions=0))
    files = {str(p): sha(p) for p in copies.values()}
    files[str(source)] = sha(source)
    for name in ('EVENT_POOL.json', 'EVENT_AUDIT.json', 'PROTOCOL.json', 'SOURCE_LOCK.json'):
        files[str(run/name)] = sha(run/name)
    immutable(run/'BINDING.json', dict(files=files, feature_result=bound['feature_result']))
    immutable(run/'PREPARE_RESULT.json', dict(status='PREPARED_ORIGINAL_POOL', base_loaded=False,
        optimizer_updates=0, source_binding_sha256=sha(CPU/'BINDING.json'), event_labels=len(records)))
    print(read(run/'PREPARE_RESULT.json'), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
