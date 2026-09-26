"""Register real TRAIN prefixes for later four-action outcome collection.

This program does not execute an alternative or create any outcome label.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def rank(value):
    return hashlib.sha256(('execution-branches-v1:' + value).encode()).hexdigest()


def select_prefix_sources(episodes):
    """One natural and one perturbed distinct route per house, without scores."""
    if len({e['id'] for e in episodes}) != len(episodes):
        raise ValueError('DUPLICATE_SOURCE_ID')
    houses = {}
    routes = {}
    for e in episodes:
        if e['partition'] not in ('FIT', 'DEV'):
            raise ValueError('ONLY_OFFICIAL_TRAIN_FIT_DEV')
        if e['house'] in houses and houses[e['house']] != e['partition']:
            raise ValueError('HOUSE_SPLIT_LEAK')
        if e['route_family'] in routes and routes[e['route_family']] != e['partition']:
            raise ValueError('ROUTE_SPLIT_LEAK')
        houses[e['house']] = routes[e['route_family']] = e['partition']
    selected = []
    for house in sorted(houses):
        used = set()
        for variant in ('natural', 'turn_prefix'):
            candidates = [e for e in episodes if e['house'] == house
                          and e['variant'] == variant and e['route_family'] not in used]
            if not candidates:
                raise ValueError('MISSING_DISTINCT_ROUTE_VARIANT:' + house)
            chosen = min(candidates, key=lambda e: (rank(house + ':' + variant + ':' + e['route_family']), e['id']))
            selected.append(chosen)
            used.add(chosen['route_family'])
    return selected


def prefix_record(entry, trace):
    actions = [x for x in trace if x['event'] == 'action']
    if [a['step'] for a in actions] != list(range(1, len(actions) + 1)):
        raise ValueError('NONCONTIGUOUS_ACTION_LOG')
    if not actions or len(actions) > 500:
        raise ValueError('INVALID_TRAJECTORY_BUDGET')
    if any(a['executed_action'] not in range(4) for a in actions):
        raise ValueError('INVALID_ACTION')
    if any(a['executed_action'] == 0 for a in actions[:-1]):
        raise ValueError('ACTION_AFTER_STOP')
    reset, = [x for x in trace if x['event'] == 'reset']
    if reset['instruction_sha256'] != hashlib.sha256(entry['instruction'].encode()).hexdigest():
        raise ValueError('INSTRUCTION_MISMATCH')
    for i, a in enumerate(actions):
        previous_rgb = reset['rgb_sha256'] if i == 0 else actions[i - 1]['after_rgb_sha256']
        if a['before_rgb_sha256'] != previous_rgb:
            raise ValueError('RGB_CHAIN_MISMATCH')
    target = (16, 32, 64)[int(rank(entry['route_family'] + ':' + entry['variant'])[:8], 16) % 3]
    queries = [x for x in trace if x['event'] == 'generation']
    steps = [q['environment_step'] for q in queries]
    if steps != sorted(set(steps)):
        raise ValueError('AMBIGUOUS_QUERY_BOUNDARY')
    eligible = [q for q in queries if q['environment_step'] <= target
                and q['environment_step'] < len(actions)]
    if not eligible:
        raise ValueError('NO_REGISTERED_QUERY_BOUNDARY')
    query = eligible[-1]
    cutoff = query['environment_step']
    current_rgb = actions[cutoff]['before_rgb_sha256']
    if query['input']['rgb_sha256'] != current_rgb:
        raise ValueError('QUERY_RGB_MISMATCH')
    prefix_actions = [a['executed_action'] for a in actions[:cutoff]]
    if 0 in prefix_actions:
        raise ValueError('STOP_IN_PREFIX')
    return dict(source_id=entry['id'], split=entry['partition'], house=entry['house'],
        route_family=entry['route_family'], variant=entry['variant'],
        episode_id=entry['episode_id'], trajectory_id=entry['trajectory_id'],
        instruction=entry['instruction'], target_cutoff=target, cutoff=cutoff,
        prefix_actions=prefix_actions,
        prefix_rgb_sha256=[reset['rgb_sha256']] + [a['after_rgb_sha256'] for a in actions[:cutoff]],
        reference_processed_input=query['input'],
        reference_query_steps=[s for s in steps if s <= cutoff],
        remaining_decisions=500 - cutoff, status='PLANNED_NOT_EXECUTED',
        training_admission=False)


def build(base, output):
    began = time.time()
    source = base / 'intervention_v2/runs/recovery_001/train'
    manifest_path = source / 'DATA_MANIFEST.json'
    source_protocol = read(source / 'PROTOCOL.json')
    if source_protocol['split'] != 'OFFICIAL_TRAIN':
        raise ValueError('SOURCE_NOT_OFFICIAL_TRAIN')
    episodes = read(manifest_path)['episodes']
    # Selection precedes reading outcomes. Unseen manifests/results are never opened.
    selected = select_prefix_sources(episodes)
    complete = {}
    for session in sorted((source / 'evaluation').iterdir()):
        seal = session / 'STATE_SEAL.json'
        if not seal.is_file():
            continue
        identity = read(seal)
        expected = source_protocol['expected_base_state_sha256']
        if identity['base_before'] != expected or identity['base_after'] != expected:
            raise ValueError('SOURCE_BASE_CHANGED')
        for path in sorted(session.glob('episodes/*/COMPLETE.json')):
            item = read(path)
            if item['id'] in complete:
                raise ValueError('DUPLICATE_COMPLETE_GROUP')
            complete[item['id']] = (path, item, seal)
    if set(complete) != {e['id'] for e in episodes}:
        raise ValueError('SOURCE_DENOMINATOR_INCOMPLETE')
    files = {str(p): sha(p) for p in (Path(__file__), manifest_path, source / 'PROTOCOL.json')}
    prefixes = []
    requests = []
    for e in selected:
        path, item, seal = complete[e['id']]
        trace_path = path.parent / 'NATIVE/TRACE.jsonl'
        if sha(trace_path) != item['trace_hashes']['NATIVE']:
            raise ValueError('SOURCE_TRACE_CHANGED')
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        prefix = prefix_record(e, trace)
        config = Path(e['config'])
        fixture = config.with_suffix('.json.gz')
        for p in (path, seal, trace_path, config, fixture):
            files[str(p)] = sha(p)
        prefix.update(prefix_id=len(prefixes), source_trace=str(trace_path),
            source_trace_sha256=files[str(trace_path)], config=str(config),
            config_sha256=files[str(config)], fixture=str(fixture),
            fixture_sha256=files[str(fixture)])
        prefixes.append(prefix)
        for action in range(4):
            requests.append(dict(request_id=len(requests), prefix_id=prefix['prefix_id'],
                candidate_first_action=action, full_budget=500,
                remaining_decisions=prefix['remaining_decisions'],
                status='NOT_RUN', success=None, cost=None, training_admission=False))
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(id='EXECUTION_BRANCH_DATA_PREPARATION_V1',
        status='MANIFEST_ONLY_RUNTIME_NOT_IMPLEMENTED',
        source=str(source), official_train_only=True,
        source_selection='One hash-ranked natural and one perturbed distinct route per house; no outcome selection.',
        cutoff_selection='Last real generation <= hash-assigned 16/32/64; no candidate method scores.',
        candidate_actions=['STOP', 'FORWARD', 'LEFT', 'RIGHT'],
        continuation=dict(policy='Same frozen native StreamVLN, no residual or goal teacher',
            backbone=source_protocol['backbone'], source_commit=source_protocol['source_commit'],
            expected_base_state_sha256=source_protocol['expected_base_state_sha256'],
            branch_rule='Force only the first action token at the registered query; remaining tokens and queries use the same native policy.',
            prefix_rule='Reset to registered initial episode and physically replay prefix with original query timing; no interior set_state.',
            state_rule='Rebuild the same causal backbone/controller state for each branch; record paired input/state parity.',
            budget=500, stop_rule='STOP consumes a decision and creates no new observation',
            success_rule='Active STOP and registered geodesic distance <3m; ordinary collisions remain recorded, not relabeled.',
            missing_rule='Retain NOT_RUN/ERROR/UNKNOWN and full planned denominator; never infer candidate outcomes.'),
        information_firewall='Instruction, causal sensor history and executed actions only in policy; goal/metrics offline audit only.',
        labels_currently_available=0, new_physical_executions=0,
        relation_scope='Same-prefix action consequences; this alone does not establish history necessity or algorithm novelty.',
        training_rule='Every future arm receives the same admitted branches; direct rollout-cost learning is a required strong control.',
        limits='Selected prefixes are known TRAIN trajectories; all alternative continuations still require actual execution.',
        gpu_launch_authorized_by_this_file=False)
    write(output / 'PROTOCOL.json', protocol)
    write(output / 'PREFIXES.json', prefixes)
    write(output / 'REQUESTS.json', requests)
    result = dict(status='PREPARED_NOT_EXECUTED', source_trajectories=len(episodes),
        source_route_families=len({e['route_family'] for e in episodes}),
        selected_prefixes=len(prefixes), selected_distinct_routes=len({p['route_family'] for p in prefixes}),
        planned_branches=len(requests), actually_executed_branches=0,
        new_training_labels=0, gpu_hours=0, optimizer_updates=0, unseen_read=False,
        by_split=dict(Counter(p['split'] for p in prefixes)),
        by_variant=dict(Counter(p['variant'] for p in prefixes)),
        cutoff_counts=dict(Counter(p['cutoff'] for p in prefixes)),
        houses_by_split={s: sorted({p['house'] for p in prefixes if p['split'] == s}) for s in ('FIT', 'DEV')},
        all_selected_traces_hash_verified=True, wall_seconds=time.time() - began)
    write(output / 'RESULT.json', result)
    write(output / 'SOURCE_LOCK.json', dict(files=files,
        artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.output)
