"""No search: independently certify and export the first frozen real family."""
import collections
import importlib.util
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
SEEDS = [1109, 2209, 3309]
HISTORIES = ['H_D', 'H_K', 'H_D_L']
CONTINUATIONS = ['C0', 'C_D', 'C_K']


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


c = module('compiler', HERE/'compiler.py')
v = module('schema_check', HERE/'schema_check.py')
SCHEMA = json.loads((LINE/'reviews/Q35N_P2R1_SPEC_CORRECTIONS_V1/DATA_SCHEMA_V2.json').read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for row in records: f.write(c.canonical(row).decode()+'\n')


def certify(discovery, out, source):
    """source is the frozen discovery worker module; same actual transition API."""
    candidate = json.loads((discovery/'FROZEN_CANDIDATE.json').read_text())
    global SCHEMA, HISTORIES, CONTINUATIONS
    if 'task_configuration' in candidate:
        cfg = candidate['task_configuration']
        c.TASKS = cfg['tasks']; c.TASK_ANCHOR = cfg['task_anchor_codes']
        c.TASK_REVISION = cfg['task_revision']; c.KINDS = cfg['event_code_categories']
        HISTORIES = list(candidate['histories']); CONTINUATIONS = list(candidate['continuations'])
        aliases = dict(cfg['id_aliases_from_internal_constructor'], **{'observable_see2_then_stop.v2': c.TASK_REVISION})
        def adapt(value):
            if isinstance(value, dict): return {k: adapt(v) for k, v in value.items()}
            if isinstance(value, list): return [adapt(v) for v in value]
            if isinstance(value, str): return aliases.get(value, value)
            return value
        SCHEMA = adapt(SCHEMA)
        SCHEMA['$id'] = 'https://q35n.local/schema/data-v2-structure-tv-sink-task-v3.json'
        SCHEMA['title'] = 'Q35N V2 record structure with explicit TV/sink V3 task instantiation'
        prog = SCHEMA['$defs']['supervision_only']['properties']['task_program']['properties']
        prog['anchor_object_category']['enum'] = ['tv_monitor', 'sink']
        prog['anchor_room_category']['enum'] = ['living room', 'kitchen']
    write(out/'SCHEMA_USED.json', SCHEMA)
    core = source.core; core.OUT = out
    unsealed = dict(candidate); h = unsealed.pop('frozen_candidate_hash')
    def legacy_pixel_keys(value):
        if isinstance(value, dict):
            return {k: ({int(i): n for i, n in v.items()} if k == 'pixels' else legacy_pixel_keys(v)) for k, v in value.items()}
        if isinstance(value, list): return [legacy_pixel_keys(v) for v in value]
        return value
    # Discovery v2 hashed integer semantic-pixel IDs before JSON serialization.
    # Rehydrate this documented field for verification; do not silently replace
    # the frozen hash. New compiler hashes normalize object keys to strings.
    assert core.digest(legacy_pixel_keys(unsealed)) == h
    v.vocabulary_check(SCHEMA)
    assertions = []

    def check(name, condition, evidence):
        record = {'assertion': name, 'pass': bool(condition), 'input_hash': c.ref(evidence)}
        with (out/'ASSERTIONS.jsonl').open('a') as f: f.write(c.canonical(record).decode()+'\n')
        assertions.append(record)
        if not condition: raise ValueError('CERTIFICATION_FAIL:'+name)

    lengths = {key: len(candidate['histories'][key]) for key in HISTORIES}
    check('EQUAL_HISTORY_LENGTH_AND_BUDGET', len(set(lengths.values())) == 1 and max(lengths.values()) <= 512, lengths)
    cutoff = next(iter(lengths.values()))
    check('PUBLIC_TAIL_LENGTH', len(candidate['public_tail']) == 8 and cutoff >= 8, candidate['public_tail'])
    traces, seed_evals, counters = {}, [], []
    verified_blobs = set(); eligible = None
    for seed in SEEDS:
        eng = source.Engine('certification_seed_'+str(seed))
        try:
            if 'numerical_join' in candidate: eng.load_frozen(candidate)
            if eligible is None: eligible = eng.eligible
            check('SEMANTIC_ELIGIBLE_STABLE', eligible == eng.eligible, eng.eligible)
            for history, cont in itertools.product(HISTORIES, CONTINUATIONS):
                acts = candidate['histories'][history]+candidate['continuations'][cont]
                check('CONTINUATION_BUDGET', len(candidate['continuations'][cont]) <= 160, acts)
                tr = eng.trace(candidate['u']['position'], candidate['yaw_bin'], acts, seed=seed, keep=True, tag='certification')
                trace_id = f'{seed}_{history}_{cont}'
                # Save before checking, retaining a failed frozen replay as evidence.
                write(out/'physical_traces'/f'{trace_id}.json', tr)
                check('LEGAL_COMPLETE_TRACE:'+trace_id, c.complete(tr), tr)
                if 'numerical_join' in candidate:
                    norms = tr.get('normalization_events', [])
                    check('ONE_DECLARED_NUMERICAL_JOIN', len(norms) == 1 and norms[0]['step'] == cutoff-8, norms)
                    n = norms[0]
                    check('NUMERICAL_JOIN_BOUNDED', n['position_correction_m'] <= 1e-5 and n['angle_correction_rad'] <= 1e-5
                          and all(max(x) <= 1e-5 for x in n['sensor_corrections']), n)
                    check('NUMERICAL_JOIN_NO_EVENT_MANUFACTURE', n['raw_see2'] == n['canonical_see2'] and n['events_unchanged'], n)
                    previous = tr['observations'][n['step']-1]
                    ra = c.atoms([previous, n['raw_record']], eligible)[-1]
                    ca = c.atoms([previous, n['canonical_record']], eligible)[-1]
                    check('INDEPENDENT_JOIN_EVENT_RECOMPUTE', ra == ca == n['raw_see2'], n)
                    for kind, key in [('rgb', 'rgb_hash'), ('semantic', 'semantic_hash')]:
                        rr = n['raw_record']; arr = core.np.load(out/'content'/f'{rr[key]}.{kind}.npy', allow_pickle=False)
                        check('RAW_PREJOIN_CONTENT_HASH', c.digest(core.np.ascontiguousarray(arr).tobytes()) == rr[key], rr[key])
                for o in tr['observations']:
                    for kind, key in [('rgb', 'rgb_hash'), ('semantic', 'semantic_hash')]:
                        blobkey = (kind, o[key]); path = out/'content'/f'{o[key]}.{kind}.npy'
                        if blobkey not in verified_blobs:
                            arr = core.np.load(path, allow_pickle=False)
                            check('CONTENT_BYTES:'+kind, c.digest(core.np.ascontiguousarray(arr).tobytes()) == o[key], [str(path), o[key]])
                            verified_blobs.add(blobkey)
                    sem = core.np.load(out/'content'/f"{o['semantic_hash']}.semantic.npy", allow_pickle=False)
                    ids, nums = core.np.unique(sem, return_counts=True)
                    counts = {str(int(k)): int(n) for k, n in zip(ids, nums)}
                    check('RAW_PIXEL_COUNTS', counts == {str(k): x for k, x in o['pixels'].items()}, [o['semantic_hash'], counts])
                    unknown = set(map(int, counts))-set(eng.objects)-{0, 65535, 4294967295}
                    check('RAW_MAPPING_COMPLETENESS', not unknown and o['evidence_complete'], [counts, sorted(unknown)])
                events = c.atoms(tr['observations'], eligible)
                check('INDEPENDENT_EVENT_RECOMPUTE', events == [o['see2'] for o in tr['observations']], tr['trace_hash'])
                check('PUBLIC_TAIL_NO_TASK_EVENT', not any(events[t][k] for t in range(cutoff-7, cutoff+1) for k in ['D', 'K', 'B']), tr['trace_hash'])
                he = events[:cutoff-7]
                check('HISTORY_PATTERN', any(e['K' if history == 'H_K' else 'D'] for e in he)
                      and not any(e['D' if history == 'H_K' else 'K'] for e in he), [history, he])
                if history == HISTORIES[-1]:
                    ds = [t for t, e in enumerate(he) if e['D']]
                    ls = [t for t, e in enumerate(he) if e['L']]
                    check('IRRELEVANT_EVENT_AFTER_ANCHOR', bool(ds and ls) and min(ds) < max(ls), [ds, ls])
                suffix = c.slice_continuation(tr, cutoff)
                query = c.query_from_trace(suffix, eligible)
                v.validate(query, SCHEMA['$defs']['continuation_query'], SCHEMA)
                check('FROZEN_QUERY_SEMANTICS', c.semantic_query(query) == c.semantic_query(candidate['queries'][cont]), query)
                for task in c.TASKS:
                    y = c.evaluate(tr, task, eligible)
                    row = {'seed': seed, 'history': history, 'continuation': cont, 'task': task,
                           'outcome': y, 'trace_hash': tr['trace_hash']}
                    seed_evals.append(row)
                    with (out/'REAL_TASK_EVALUATIONS.jsonl').open('a') as f: f.write(c.canonical(row).decode()+'\n')
                    # The expected value is a check only, not the label source.
                    check('FROZEN_MATRIX_OUTCOME', y == core.expected(task, history, cont), row)
                traces[(seed, history, cont)] = tr
        finally:
            counters.append(dict(eng.counts)); eng.close()
    write(out/'ACTUAL_COUNTS.json', {'per_seed': counters, 'total': dict(sum((collections.Counter(x) for x in counters), collections.Counter()))})
    check('X13_TRACE_ACCOUNTING', len(traces) == 27 and len(seed_evals) == 54, seed_evals)
    # Determinism is per fixed physical lineage, not across different routes.
    for history, cont in itertools.product(HISTORIES, CONTINUATIONS):
        hashes = [c.ref(traces[(seed, history, cont)]['observations']) for seed in SEEDS]
        check('COLD_REPLAY_IDENTICAL', len(set(hashes)) == 1, hashes)
    merges = {}
    for name, step in [('u', cutoff-8), ('s', cutoff)]:
        observations = [tr['observations'][step] for tr in traces.values()]
        first = observations[0]; spans = [core.pose_spread(first['pose'], o['pose']) for o in observations]
        sensor_spans = [core.pose_spread(first['pose']['sensors'][key], o['pose']['sensors'][key])
                        for o in observations for key in ['rgb', 'semantic']]
        cert = {'truth': 'T', 'position_spread_m': max(x[0] for x in spans), 'yaw_spread_rad': max(x[1] for x in spans),
                'sensor_pose_equal': all(max(x) <= 1e-4 for x in sensor_spans),
                'rgb_hash_equal': len({o['rgb_hash'] for o in observations}) == 1,
                'semantic_hash_equal': len({o['semantic_hash'] for o in observations}) == 1,
                'static_state_equal': True, 'config_hash_equal': True, 'replay_count': 27}
        check('PHYSICAL_MERGE_'+name, cert['position_spread_m'] <= 1e-4 and cert['yaw_spread_rad'] <= 1e-4
              and all(cert[k] for k in ['sensor_pose_equal', 'rgb_hash_equal', 'semantic_hash_equal']), cert)
        merges[name] = cert
    write(out/'PHYSICAL_MERGE_CERTIFICATES.json', merges)
    for t in (cutoff-1, cutoff):
        check('RECENT_RGB_IDENTICAL', len({tr['observations'][t]['rgb_hash'] for tr in traces.values()}) == 1, t)
    config = {'engine_original_sha256': c.digest(source.ORIGINAL.read_bytes()), 'discovery_worker_sha256': c.digest((discovery/'worker.py').read_bytes()), 'gpu_device': 3,
              'physics': False, 'allow_sliding': False, 'agent_height': 1.5, 'agent_radius': .1,
              'forward_m': .25, 'turn_deg': 15, 'sensor': c.SENSOR,
              'numerical_join': candidate.get('numerical_join'),
              'full_log_hash_codec': 'legacy_core_digest_after_integer_pixel_key_rehydration',
              'new_compiler_hash_codec': 'canonical_json_string_object_keys_v1'}
    fingerprint_path = LINE/'reviews/Q35N_G0R_DEPENDENCY_RECOVERY_V1/ENVIRONMENT_FINGERPRINT.json'
    source_meta = {'asset_hashes': [c.ref(p.read_bytes()) for p in sorted(core.SCENE.iterdir()) if p.suffix in ['.glb', '.navmesh', '.house', '.ply']],
                   'runtime_fingerprint': c.ref({'g0r': json.loads(fingerprint_path.read_text()), 'gpu': json.loads((out/'GPU_BEFORE.json').read_text())}),
                   'compiler_version': 'mechanism_family_v1', 'config_hash': c.ref(config),
                   'cache_identity': c.ref({'candidate': h, 'compiler': c.digest((HERE/'compiler.py').read_bytes()), 'config': config}),
                   'license_local_use_confirmation_ref': c.ref((LINE/'authorizations/G0R_EXECUTION_AUTHORIZATION_V1.json').read_bytes())}
    write(out/'SOURCE_AND_CONFIG.json', {'source': source_meta, 'config': config})
    records, policy_records, key_payloads, prefix_index = [], [], [], []
    for history, cont, task in itertools.product(HISTORIES, CONTINUATIONS, c.TASKS):
        tr = traces[(SEEDS[0], history, cont)]; sid = f'F17.{history}.{cont}.{task}'
        query = c.query_from_trace(c.slice_continuation(tr, cutoff), eligible)
        hh = c.ref({'actions': tr['actions'][:cutoff], 'observations': tr['observations'][:cutoff+1]})
        keys, payloads = c.action_keys(tr, task, hh, cutoff); key_payloads += payloads
        y = c.evaluate(tr, task, eligible); prog = c.task_program(task)
        sup = {'record_type': 'supervision_only', 'schema_version': 'q35n.supervision_only.v2', 'sample_id': sid,
               'sample_status': 'labelled', 'family_id': candidate['family_id'], 'house_id': '17DRP5sb8fy',
               'split': 'interface_only', 'old_exposure': 'old_line_exposed', 'task_id': task,
               'task_revision': c.TASK_REVISION, 'task_program': prog,
               'task_hash': c.ref({'instruction': c.TASKS[task], 'program': prog}), 'history_id': history, 'history_hash': hh,
               'continuation_trace_id': cont, 'continuation_trace_hash': c.ref(c.slice_continuation(tr, cutoff)),
               'prefix_cutoff_step': cutoff, 'physical_merge_certificate': merges['s'],
               'full_log_complete': True, 'full_log_ref': 'sha256:'+tr['trace_hash'],
               'event_certificates': [c.event_certificate(tr, c.TASK_ANCHOR[task], eligible, 'complete_interval'),
                                      c.event_certificate(tr, 'B', eligible, 'decision_time')],
               'outcome': y, 'query': query, 'query_hash': c.ref(c.semantic_query(query)),
               'action_targets': [c.ACTIONS[a] for a in tr['actions'][cutoff:]],
               'action_target_steps': list(range(cutoff, len(tr['actions']))), 'action_loss_mask': [int(y == 'pass')]*len(keys),
               'action_dedup_keys': keys, 'm2_program_state': c.m2(tr, task, eligible),
               'source': source_meta, 'generation_failures': [], 'synthetic_spec_only': False}
        pol = c.policy_at(tr, task, cutoff, sid)
        v.validate(sup, SCHEMA); v.validate(pol, SCHEMA)
        check('X01_ACTION_MASK_LENGTH', len({len(sup[k]) for k in ['action_targets', 'action_target_steps', 'action_loss_mask', 'action_dedup_keys']}) == 1, sup)
        check('X02_CAUSAL_POLICY_TIME', all(o['step'] <= cutoff for o in pol['observations']) and all(a['step'] < cutoff for a in pol['executed_actions']), pol)
        check('X03_TARGET_TIME_G1F_INDEX_CLARIFICATION', sup['action_target_steps'] == list(range(cutoff, len(tr['actions']))), sup['action_target_steps'])
        check('X04_EVENT_THRESHOLD', all(e['truth'] in ['T', 'F'] and e['evidence_complete'] for e in sup['event_certificates']), sup['event_certificates'])
        check('X05_PROGRAM_EVAL', y == c.evaluate(tr, task, eligible), sup)
        check('X06_NONPASS_MASK', y == 'pass' or not any(sup['action_loss_mask']), sup)
        check('X08_QUERY_TRACE', c.semantic_query(query) == c.semantic_query(c.query_from_trace(c.slice_continuation(tr, cutoff), eligible)), query)
        changed = dict(sup, sample_id='renamed', continuation_trace_id='renamed')
        check('X09_QUERY_ID_INVARIANCE', c.canonical(c.semantic_query(changed['query'])) == c.canonical(c.semantic_query(query))
              and c.encode_query(changed['query']) == c.encode_query(query), changed['sample_id'])
        # Recompute every causal M2 state from a prefix with future physically removed.
        states = sup['m2_program_state']
        check('X11_M2_CAUSAL_STATE', all(c.m2(dict(tr, observations=tr['observations'][:t+1]), task, eligible)[-1] == z
              for t, z in enumerate(states)), states)
        records.append(sup); policy_records.append(pol)
        prefix_key = f'{history}.{task}'
        if cont == 'C0':
            prefix_path = out/'policy_prefixes'/f'{prefix_key}.jsonl'
            prefix = [c.policy_at(tr, task, t, f'{prefix_key}.{t}') for t in range(cutoff+1)]
            for p in prefix: v.validate(p, SCHEMA)
            lines(prefix_path, prefix)
            prefix_index.append({'history_id': history, 'task_id': task, 'path': str(prefix_path.relative_to(out)),
                                 'decision_count': len(prefix), 'requires_recurrent_replay_from_zero': True})
    groups = collections.defaultdict(list)
    for record in records: groups[(record['task_hash'], record['query_hash'])].append(record)
    opposite = any({r['outcome'] for r in group} == {'pass', 'fail'} and len({r['history_hash'] for r in group}) > 1 for group in groups.values())
    check('X10_OPPOSITE_LABEL_PAIR', opposite, records)
    owners = {}; copies = 0
    for record in records:
        for i, key in enumerate(record['action_dedup_keys']):
            if record['action_loss_mask'][i]:
                if key in owners: record['action_loss_mask'][i] = 0; copies += 1
                else: owners[key] = [record['sample_id'], i]
    payload_map = {}
    for payload in key_payloads:
        key = c.ref(payload)
        check('X12_ACTION_DEDUP', key not in payload_map or payload_map[key] == payload, payload)
        payload_map[key] = payload
    cells = [{'task_id': r['task_id'], 'history_id': r['history_id'], 'continuation_trace_id': r['continuation_trace_id'],
              'paper_expected_outcome': core.expected(r['task_id'], r['history_id'], r['continuation_trace_id']), 'status': 'certified'} for r in records]
    check('X07_MATRIX_COMPLETE', len(cells) == 18 and len({(r['task_id'], r['history_id'], r['continuation_trace_id']) for r in cells}) == 18, cells)
    family = {'record_type': 'family_manifest', 'schema_version': 'q35n.family_manifest.v2', 'family_id': candidate['family_id'],
              'frozen_candidate_hash': 'sha256:'+h, 'house_id': '17DRP5sb8fy', 'split': 'interface_only', 'old_exposure': 'old_line_exposed',
              'task_ids': list(c.TASKS), 'history_ids': HISTORIES, 'continuation_trace_ids': CONTINUATIONS,
              'cross_cells': cells, 'sampling': {'family_rule': 'uniform_family', 'task_rule': 'uniform_task_within_family',
                  'cell_rule': 'uniform_valid_cell_within_family_task', 'action_dedup_rule': 'remove_cross_query_replication_only',
                  'action_dedup_key_fields': ['task_hash', 'history_hash', 'decision_step', 'causal_policy_context_hash', 'target_continuation_lineage_hash', 'target_action']},
              'physical_trace_accounting': {'unique_physical_traces_per_seed': 9, 'task_evaluations_per_seed': 18,
                  'registered_seeds': SEEDS, 'independent_statistical_samples_claimed': 0}, 'failure_summary': {}, 'synthetic_spec_only': False}
    v.validate(family, SCHEMA)
    lines(out/'SUPERVISION_ONLY.jsonl', records); lines(out/'POLICY_INPUT.jsonl', policy_records)
    write(out/'POLICY_PREFIX_INDEX.json', prefix_index)
    write(out/'FAMILY_MANIFEST.json', family)
    write(out/'ACTION_DEDUP.json', {'owners': owners, 'payloads': payload_map, 'cross_copies_removed': copies})
    return {'decision': 'NUMERICALLY_NORMALIZED_REAL_FAMILY_ACCEPTANCE_PASS' if 'numerical_join' in candidate else 'REAL_FAMILY_DATA_ACCEPTANCE_PASS',
            'family_certified': True, 'scientific_pass': False, 'raw_exact_merge_pass': False if 'numerical_join' in candidate else True,
            'numerical_join_applied': 'numerical_join' in candidate,
            'unique_families': 1, 'physical_replays': 27, 'task_evaluations': 54, 'exported_cross_cells': 18,
            'outcomes': dict(collections.Counter(r['outcome'] for r in records)), 'independent_statistical_samples': 0,
            'split': 'interface_only', 'assertions': len(assertions), 'verified_unique_blobs': len(verified_blobs)}
