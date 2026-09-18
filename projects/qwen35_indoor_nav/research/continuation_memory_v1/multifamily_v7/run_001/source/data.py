"""Real SEE2 families, whole-house split, globally typed query vocabulary."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
ASSETS = LINE / 'data_pipeline/mechanism_runtime_v1/witness_first_v1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_tokens(query):
    """Names carry type and meaning across families; local integer IDs are excluded."""
    tokens = ['QUERY_BEGIN', 'FRAME:' + query['coordinate_frame']]
    for item in query['sequence']:
        if item['kind'] == 'movement':
            tokens += ['MOVEMENT', 'ACTION:' + item['action'], 'REPEAT:' + str(item['repeat'])]
        elif item['kind'] == 'observe':
            tokens += ['OBSERVE', 'CATEGORY:' + item['object_category'], 'ROOM:' + item['room_category'],
                       'PIXELS:' + str(item['min_pixels']), 'FRAMES:' + str(item['consecutive_frames'])]
        elif item['kind'] == 'act' and item['action'] == 'STOP':
            tokens += ['ACTION:STOP']
        else:
            raise ValueError('UNSUPPORTED_QUERY_ITEM')
    return tokens


def split_houses(houses):
    ordered = sorted(set(houses), key=lambda house: hashlib.sha256(('v7-house-split:' + house).encode()).hexdigest())
    assert len(ordered) >= 5
    return {house: 'check' if i < 2 else 'fit' for i, house in enumerate(ordered)}


def prepare():
    loaders = load('v7_loader', LINE / 'data_pipeline/mechanism_runtime_v1/loader.py')
    core = load('v7_compiler', LINE / 'data_pipeline/mechanism_factory_v2/compiler.py')
    exact = load('v7_exact_state', HERE.parent / 'check_tasks.py')
    # Fixed existing cohort identified before model results, not a score-selected pool.
    paths = sorted(ASSETS.glob('batch_execution_v1/batch_50*/run_v1/bundles/*/export_v4/MANIFEST.json'))
    paths += sorted(ASSETS.glob('diversity_v1*/run_v1/bundles/*/export_v4/MANIFEST.json'))
    manifests = [json.loads(path.read_text()) for path in paths]
    split = split_houses(m['group']['house_id'] for m in manifests)
    features, feature_lookup, contents, families, audit = [], {}, {}, [], []
    all_tokens = {'PAD'}

    def feature_index(record):
        loaders.validate_policy(record)
        clean = dict(instruction=record['instruction'], rgb_refs=[o['rgb_ref'] for o in record['observations']],
                     executed=[a['action'].lower() for a in record['executed_actions']])
        key = hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()
        if key not in feature_lookup:
            feature_lookup[key] = len(features)
            features.append(dict(key=key, **clean))
        return feature_lookup[key]

    for path, manifest in zip(paths, manifests):
        compiler = core.Compiler(**manifest['compiler_config'])
        loader = loaders.FamilyLoader(path.parent, compiler)
        checked = loader.validate_supervision_contract()
        for key, value in loader.contents.items():
            if key.endswith('.rgb'):
                # Identical pixels may be stored at multiple legitimate source paths.
                contents.setdefault('sha256:' + key[:-4], value)
        prefix_ids, prefixes = list(loader.prefix_index), []
        for pid in prefix_ids:
            cell = next(c for c in loader.cells if c['prefix_id'] == pid)
            records = loader.prefix_records(pid)
            trace = loader.read(cell['trace_path'])
            states = exact.exact_state_targets(compiler, trace, cell['task_id'], cell['prefix_cutoff'])
            fields = ['anchor_seen_strictly_before', 'anchor_seen_through_current', 'terminal_witness_now', 'ordered_ready_to_stop']
            prefixes.append(dict(history_id=cell['history_id'], task_id=cell['task_id'],
                features=[feature_index(row) for row in records],
                state_targets=[[int(s[k]) if s[k] is not None else 0 for k in fields] for s in states],
                state_masks=[s['loss_mask'] for s in states]))
        queries, query_lookup, cells, contexts = [], {}, [], {}
        for cell in loader.cells:
            semantic = compiler.semantic_query(cell['query'])
            compiler.encode_query(semantic)  # Keep the frozen schema/legality validation.
            tokens = query_tokens(semantic)
            all_tokens.update(tokens)
            key = tuple(tokens)
            if key not in query_lookup:
                query_lookup[key] = len(queries)
                queries.append(tokens)
            full = loader.action_stream(cell['cell_id'])
            action_by_step = {t: (target, mask) for t, target, mask in zip(cell['action_steps'], cell['action_targets'], cell['action_loss_mask'])}
            tail = []
            if any(cell['action_loss_mask']):
                for t in range(cell['prefix_cutoff'], len(full)):
                    target, mask = action_by_step.get(t, ('STOP', 0))
                    tail.append(dict(step=t, feature=feature_index(full[t]),
                        target=['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP'].index(target), mask=mask))
            trace = loader.read(cell['trace_path'])
            events = compiler.atoms(trace['observations'])
            task = compiler.tasks[cell['task_id']]
            cut, last = cell['prefix_cutoff'], len(events)-1
            context = dict(stop_at_cutoff=last == cut, stops=trace['actions'][-1] == 'S',
                suffix_anchor_before_final=any(bool(e[task['anchor']]) for e in events[cut+1:last]),
                terminal_at_final=bool(events[last][task['terminal']]) if last > cut else None)
            context_key = (cell['task_id'], cell['continuation_id'])
            assert contexts.setdefault(context_key, context) == context, 'HISTORY_LEAK_IN_QUERY'
            state = prefixes[prefix_ids.index(cell['prefix_id'])]['state_targets'][-1]
            expected = bool(context['stops'] and (state[3] if context['stop_at_cutoff'] else
                context['terminal_at_final'] and (state[1] or context['suffix_anchor_before_final'])))
            assert not cell['bce_mask'] or int(expected) == cell['y'], 'EXACT_STATE_NOT_SUFFICIENT'
            cells.append(dict(audit_cell_id=cell['cell_id'], prefix=prefix_ids.index(cell['prefix_id']),
                query=query_lookup[key], y=cell['y'] if cell['y'] is not None else 0,
                mask=cell['bce_mask'], tail=tail, query_context=context))
        owners = sum(x['mask'] for cell in cells for x in cell['tail'])
        assert owners == manifest['ce_unique_owners']
        assert len({len(p['features']) for p in prefixes}) == 1
        # Verify the core opposing-label pair really has identical last-window input.
        matched = 0
        for left in cells:
            for right in cells:
                pl, pr = prefixes[left['prefix']], prefixes[right['prefix']]
                if pl['history_id'] == 'H_A' and pr['history_id'] == 'H_B' and pl['task_id'] == pr['task_id'] and left['query'] == right['query']:
                    assert pl['features'][-1] == pr['features'][-1], 'UNMATCHED_CAUSAL_SHORT_WINDOW'
                    matched += int(left['y'] != right['y'] and left['mask'] and right['mask'])
        assert matched > 0
        house = manifest['group']['house_id']
        families.append(dict(family_id=manifest['family_id'], house=house, split=split[house],
            prefixes=prefixes, cells=cells, queries=queries))
        audit.append(dict(export=str(path.parent.relative_to(LINE)), manifest_sha256=sha(path),
            house=house, split=split[house], original_training_admission=manifest['training_admission'],
            original_scientific_pass=manifest['scientific_pass'], labels=checked, opposing_label_pairs=matched,
            prefix_observations=len(prefixes[0]['features']), source_provenance=manifest['provenance']))
        print('PREPARED', len(families), '/', len(paths), house, len(features), flush=True)
    vocabulary = ['PAD'] + sorted(all_tokens - {'PAD'})
    index = {token: i for i, token in enumerate(vocabulary)}
    for family in families:
        family['queries'] = [[index[token] for token in tokens] for tokens in family['queries']]
    assert all(reference in contents for row in features for reference in row['rgb_refs'])
    result = dict(families=families, features=features, contents=contents, query_vocabulary=vocabulary,
        audit=dict(scope='EXPLORATORY_EXISTING_REAL_SEE2_FAMILIES', original_training_admission=False,
            normalization_and_missing_controls_retained=True, independent_blind_test=False,
            house_split=split, families=audit, family_count=len(families), causal_input_count=len(features),
            query_encoding='Globally named typed tokens; schema vocabulary fixed before training, not label-derived',
            policy_fields=['instruction', 'rgb', 'executed_actions']))
    with (HERE / 'DATA.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False)
    print('COMPLETE', len(families), 'families', len(features), 'causal inputs', split, flush=True)


if __name__ == '__main__':
    prepare()
