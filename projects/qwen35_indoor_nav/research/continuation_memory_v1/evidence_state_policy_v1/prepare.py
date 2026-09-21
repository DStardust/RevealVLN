"""Recompute supervision from complete real traces; never generate new events."""
import argparse
from collections import Counter
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *


def event_labels(compiler, observations, task):
    events = compiler.atoms(observations)
    labels = [[int(bool(e[k])) for k in ('anchor', 'terminal')] for e in events]
    masks = [[int(e[k] is not None) for k in ('anchor', 'terminal')] for e in events]
    if task == 'task_T':
        # The instruction does not specify any anchor; don't train its hidden role.
        masks = [[0, m[1]] for m in masks]
    return labels, masks


def main(run):
    run.mkdir(parents=True, exist_ok=False)
    cfg = read(HERE / 'PROTOCOL.json')
    source_data = SOURCE / 'DATA.json'
    data = read(source_data)
    if sha(source_data) != read(SOURCE / 'DATA_BINDING.json')['data_sha256']:
        raise ValueError('SOURCE_DATA_IDENTITY')
    feature_result = read(SOURCE / 'features/FEATURE_RESULT.json')
    cache = SOURCE / 'features/FEATURES.pt'
    if not feature_result['parameters_unchanged'] or sha(cache) != feature_result['file_sha256']:
        raise ValueError('FROZEN_FEATURE_IDENTITY')
    sources = {str(source_data): sha(source_data), str(cache): feature_result['file_sha256']}
    raw = {r['family_id']: r for r in data['raw_families']}
    houses, parents, feature_splits, traces = {}, {}, {}, {}
    counts = Counter(); supports = {}
    for family in data['families']:
        split, house, parent = (family[k] for k in ('split', 'house', 'parent_family_id'))
        if split not in ('FIT', 'DEV'):
            raise ValueError('UNREGISTERED_SPLIT')
        for mapping, key in ((houses, house), (parents, parent)):
            if key in mapping and mapping[key] != split:
                raise ValueError('FAMILY_OR_HOUSE_SPLIT_LEAK')
            mapping[key] = split
        compiler = checker.legacy.Compiler(**raw[family['family_id']]['compiler'])
        for row in family['sequences']:
            path = LINE / row['source_trace']['path']
            if path not in traces:
                if sha(path) != row['source_trace']['sha256']:
                    raise ValueError('SOURCE_TRACE_CHANGED')
                traces[path] = read(path); sources[str(path)] = row['source_trace']['sha256']
            trace = traces[path]
            if not checker.legacy.complete(trace):
                raise ValueError('INCOMPLETE_OR_COLLIDING_TRACE')
            n = len(row['features'])
            if checker.state_sequence(compiler, trace['observations'], row['task'])[:n] != row['state_targets']:
                raise ValueError('STATE_LABEL_MISMATCH')
            labels, masks = event_labels(compiler, trace['observations'], row['task'])
            row['event_targets'], row['event_masks'] = labels[:n], masks[:n]
            if len(labels) < n:
                raise ValueError('EVENT_ALIGNMENT')
            for idx in row['features']:
                feature = data['features'][idx]
                if feature['split'] != split or idx in feature_splits and feature_splits[idx] != split:
                    raise ValueError('FEATURE_SPLIT_LEAK')
                feature_splits[idx] = split
            counts[split + '_sequences'] += 1
            counts[split + '_causal_steps_with_reuse'] += n
            counts[split + '_teacher_decisions_with_reuse'] += sum(row['action_masks'])
            for role in range(2):
                for label, mask in zip(row['event_targets'], row['event_masks']):
                    counts[f'{split}_event{role}_' + ('unknown' if not mask[role] else str(label[role]))] += 1
            for t, (state, idx) in enumerate(zip(row['state_targets'], row['features'])):
                key = (parent, row['task'], tuple(row['features'][:t+1]))
                if key in supports and supports[key] != state:
                    raise ValueError('CAUSAL_STATE_LABEL_CONFLICT')
                supports[key] = state
    # Bind the unchanged ordinary pool/schedules for the eventual matched train.
    ordinary = HERE.parent / 'natural_transfer_v9/DATA.json'
    ordinary_data = read(ordinary)
    training_ordinary = [r for r in ordinary_data['records'] if r['partition'] == 'fit']
    if any(r['row']['scene_group'] in houses for r in training_ordinary):
        raise ValueError('ORDINARY_HOUSE_LEAK')
    for path in (ordinary, Path(feature_result['ordinary_path'])):
        sources[str(path)] = sha(path)
    if sources[feature_result['ordinary_path']] != feature_result['ordinary_sha256']:
        raise ValueError('ORDINARY_CACHE_CHANGED')
    schedules = {}
    for seed in cfg['seeds']:
        path = SOURCE / 'train' / f'SCHEDULE_{seed}.json'
        sources[str(path)] = sha(path); schedules[str(seed)] = read(path)
        fit_ids = {f['family_id'] for f in data['families'] if f['split'] == 'FIT'}
        if len(schedules[str(seed)]) != cfg['steps'] or any(r['family'] not in fit_ids for r in schedules[str(seed)]):
            raise ValueError('TRAINING_SCHEDULE_NOT_FIT')
    write(run / 'DATA.json', data, True)
    write(run / 'SCHEDULES.json', schedules, True)
    write(run / 'PROTOCOL.json', cfg, True)
    count_record = dict(counts=counts, houses=dict(Counter(houses.values())), parents=dict(Counter(parents.values())),
                        variants=len(data['families']), unique_source_traces=len(traces), unique_causal_states=len(supports),
                        cached_features=len(data['features']), raw_contents=len(data['contents']), new_physical_executions=0,
                        semantics='SEE2 same eligible instance, original pixel threshold, no synthetic STOP observation',
                        scope='Exposed FIT/DEV only. Derived windows do not increase independent histories or houses.')
    write(run / 'DATA_AUDIT.json', count_record, True)
    for p in HERE.glob('*.py'):
        sources[str(p)] = sha(p)
    for p, value in read(SOURCE / 'SOURCE_LOCK.json')['files'].items():
        if sha(LINE / p) != value:
            raise ValueError('FROZEN_SOURCE_CHANGED:' + p)
        sources[str(LINE / p)] = value
    for seed in cfg['seeds']:
        p = HERE.parent / 'fork_balanced_v14/strong_state_run_001' / f'INITIAL_{seed}.pt'
        sources[str(p)] = sha(p)
    for p in (run / 'DATA.json', run / 'PROTOCOL.json', run / 'SCHEDULES.json'):
        sources[str(p.resolve())] = sha(p)
    write(run / 'BINDING.json', dict(files=sources, feature_result=feature_result, source_commit=cfg['source_commit']), True)
    write(run / 'STATUS.json', dict(status='CPU_DATA_READY', gpu_started=False), True)
    print(json.dumps(count_record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
