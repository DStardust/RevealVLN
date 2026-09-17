"""Export real factory traces using instance-scoped V4 compiler, CPU only."""
import collections
import importlib.util
import json
import os
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('q35n_runtime_v4_loader_export', HERE / 'loader.py')
_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader)
require, sha, npy_pixels = _loader.require, _loader.sha, _loader.npy_pixels


def canonical(value):
    def clean(x):
        if isinstance(x, dict):
            require(len({str(k) for k in x}) == len(x), 'JSON_KEY_COLLISION')
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    return json.dumps(clean(value), sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha(canonical(value))


def write(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        if jsonl:
            for row in value:
                handle.write(canonical(row).decode() + '\n')
        else:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export_family(output_dir, compiler, candidate, traces, content_dir, *, family_id,
                  split='interface_only', provenance=None, repeated_traces=None):
    """Nine canonical traces keyed (history, continuation); repeats same mapping.

    Caller establishes real runtime provenance. This function revalidates evidence
    and exports, but never grants physical certification or training admission.
    Content references are line-relative read-only; source files are never written.
    """
    output, source = Path(output_dir).resolve(), Path(content_dir).resolve()
    require(output.is_relative_to(HERE) and source.is_relative_to(LINE), 'SCOPE')
    require(not output.exists() and source.is_dir(), 'OUTPUT_EXISTS_OR_NO_CONTENT')
    require(split in ('interface_only', 'candidate_fit_pool'), 'SPLIT_NOT_AUTHORIZED')
    require(isinstance(candidate.get('context', {}).get('house_id'), str)
            and bool(candidate['context']['house_id']), 'HOUSE_GROUP_REQUIRED')
    histories, continuations = candidate['histories'], candidate['continuations']
    require(len(histories) == 3 and len(continuations) == 3 and len(compiler.tasks) == 2, 'FAMILY_DIMENSIONS')
    require(set(traces) == {(h, c) for h in histories for c in continuations}, 'TRACE_GRID')
    require(len(set(map(len, histories.values()))) == 1, 'HISTORY_LENGTH_MISMATCH')
    cutoff = len(next(iter(histories.values())))
    require(8 <= cutoff <= 512, 'HISTORY_BUDGET')
    contents, semantic_counts, prefix_signatures, queries, merge = {}, {}, {}, {}, None
    all_traces = [traces] + list(repeated_traces or [])
    for grid in all_traces:
        require(set(grid) == set(traces), 'REPEAT_GRID')
        for (h, c), trace in grid.items():
            require(compiler.complete(trace), 'INCOMPLETE_TRACE_UNKNOWN_NOT_EXPORTED')
            require(trace['actions'] == list(histories[h]) + list(continuations[c]), 'CANDIDATE_ACTION_MISMATCH')
            require(0 < len(continuations[c]) <= 160 and continuations[c][-1] == 'S', 'CONTINUATION_BUDGET_STOP')
            require(digest({'actions': trace['actions'], 'observations': trace['observations']}) ==
                    digest({'actions': traces[h, c]['actions'], 'observations': traces[h, c]['observations']}), 'SEED_TRACE_DIFFERENCE')
            # All content, including numerical raw/canonical boundaries, is verified.
            obs = list(trace['observations'])
            for event in trace.get('normalization_events', []):
                obs += [event['raw_record'], event['canonical_record']]
            for observation in obs:
                for kind in ('rgb', 'semantic'):
                    expected = observation[kind + '_hash']
                    key = expected + '.' + kind
                    if key not in contents:
                        path = _loader.safe_path(source, key + '.npy')
                        blob = path.read_bytes()
                        pixels = npy_pixels(blob, expected, kind)
                        contents[key] = {'line_relative_path': str(path.relative_to(LINE)),
                                         'file_sha256': sha(blob), 'raw_pixel_sha256': expected,
                                         'byte_count': len(blob), 'kind': kind, 'source_read_only': True}
                        if kind == 'semantic':
                            semantic_counts[expected] = dict(collections.Counter(str(x[0]) for x in struct.iter_unpack('<I', pixels)))
                require({str(k): v for k, v in observation['pixels'].items() if v} == semantic_counts[observation['semantic_hash']], 'SEMANTIC_COUNTS_MISMATCH')
            ph = digest(trace['observations'][:cutoff+1])
            require(prefix_signatures.setdefault(h, ph) == ph, 'PREFIX_CHANGED_ACROSS_CONTINUATIONS')
            short = digest([{k: o[k] for k in ('rgb_hash', 'semantic_hash', 'pose')}
                            for o in trace['observations'][cutoff-1:cutoff+1]])
            merge = short if merge is None else merge
            require(short == merge, 'COMMON_SHORT_WINDOW_MISMATCH')
            query = compiler.query_from_trace(compiler.slice_continuation(trace, cutoff))
            compiler.encode_query(query)
            require(len(query['sequence']) <= 160, 'QUERY_BUDGET')
            require(queries.setdefault(c, query) == query, 'QUERY_CHANGED_ACROSS_HISTORIES')
    output.mkdir(parents=True)
    prefixes, cells, owners, payloads, label_counts = {}, [], {}, {}, collections.Counter()
    for hi, h in enumerate(histories):
        history_hash = 'sha256:' + prefix_signatures[h]
        representative = traces[h, next(iter(continuations))]
        for ti, task in enumerate(compiler.tasks):
            prefix_id = f'p{hi}_{ti}'
            path = f'prefixes/{prefix_id}.jsonl'
            records = [compiler.policy_at(representative, task, t, f'{prefix_id}.{t}') for t in range(cutoff+1)]
            write(output / path, records, True)
            prefixes[prefix_id] = {'path': path, 'decision_count': len(records), 'task_id': task, 'history_id': h}
            for ci, c in enumerate(continuations):
                trace = traces[h, c]
                cell_id = f'cell{hi}_{ti}_{ci}'
                outcome = compiler.evaluate(trace, task)
                label_counts[outcome] += 1
                y = {'pass': 1, 'fail': 0, 'unknown': None}[outcome]
                keys, values = compiler.action_keys(trace, task, history_hash, cutoff) if outcome == 'pass' else ([], [])
                masks = []
                for index, (key, value) in enumerate(zip(keys, values)):
                    require(payloads.setdefault(key, value) == value, 'CE_HASH_COLLISION')
                    masks.append(int(key not in owners))
                    owners.setdefault(key, [cell_id, index])
                full_path = f'full_policy/{cell_id}.jsonl'
                write(output / full_path, [compiler.policy_at(trace, task, t, f'{cell_id}.{t}')
                                          for t in range(len(trace['actions']))], True)
                cells.append({'schema_version': 'q35n.supervision.v4', 'cell_id': cell_id,
                              'prefix_id': prefix_id, 'task_id': task, 'history_id': h, 'continuation_id': c,
                              'query': queries[c], 'outcome': outcome, 'y': y, 'bce_mask': int(y is not None),
                              'prefix_cutoff': cutoff, 'm2_program_state': compiler.m2(trace, task),
                              'action_targets': [v['target_action'] for v in values],
                              'action_steps': [v['decision_step'] for v in values],
                              'action_keys': keys, 'action_loss_mask': masks, 'full_policy_path': full_path,
                              'trace_path': f'traces/t{hi}_{ci}.json'})
        for ci, c in enumerate(continuations):
            write(output / f'traces/t{hi}_{ci}.json', traces[h, c])
    manifest = {'schema_version': 'q35n.family_export.v4', 'family_id': family_id, 'split': split,
                'group': {'house_id': candidate.get('context', {}).get('house_id'), 'family_id': family_id},
                'candidate': candidate, 'provenance': provenance or {},
                'compiler_config': {'roles': {k: list(v) for k, v in compiler.roles.items()},
                                    'eligible': {k: list(v) for k, v in compiler.eligible.items()},
                                    'tasks': {k: dict(v) for k, v in compiler.tasks.items()}, 'task_revision': compiler.task_revision},
                'canonical_traces': len(traces), 'repeat_grids_checked': len(all_traces)-1,
                'prefixes': len(prefixes), 'prefix_decisions': sum(p['decision_count'] for p in prefixes.values()),
                'cells': len(cells), 'outcomes': dict(label_counts), 'ce_unique_owners': len(owners),
                'ce_payloads': len(payloads), 'content_arrays_verified': len(contents),
                'training_admission': False, 'scientific_pass': False,
                'runtime_certification': 'CALLER_OWNED_NOT_GRANTED_BY_EXPORTER',
                'source_content_read_only': True}
    write(output / 'MANIFEST.json', manifest)
    write(output / 'CONTENT_INDEX.json', contents)
    write(output / 'PREFIX_INDEX.json', prefixes)
    write(output / 'SUPERVISION_ONLY.jsonl', cells, True)
    write(output / 'ACTION_OWNERS.json', {'owners': owners, 'payloads': payloads})
    files = sorted(p for p in output.rglob('*') if p.is_file())
    with (output / 'SHA256SUMS').open('x') as handle:
        for path in files:
            handle.write(sha(path.read_bytes()) + '  ' + str(path.relative_to(output)) + '\n')
    return manifest
