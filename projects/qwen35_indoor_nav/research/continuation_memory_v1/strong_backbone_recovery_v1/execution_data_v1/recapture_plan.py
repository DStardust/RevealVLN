"""Plan missing chunk-token captures using only the original query's causal inputs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_index import read, sha, write


def requests_from_rows(rows):
    chunks = defaultdict(list)
    for row in rows:
        chunks[(row['trajectory_id'], row['query_index'])].append(row)
    requests = []
    for (trajectory_id, query_index), chunk in chunks.items():
        if not any(row['requires_actor_recapture'] for row in chunk):
            continue
        chunk.sort(key=lambda row: row['action_token_offset'])
        first = chunk[0]
        query_start = first['query_start_step']
        if [row['action_token_offset'] for row in chunk] != list(range(first['chunk_length'])):
            raise ValueError('Incomplete actual generated chunk')
        if first['environment_step'] != query_start or not first['has_existing_actor_feature']:
            raise ValueError('Missing original query actor anchor')
        actions = []
        for offset, row in enumerate(chunk):
            if row['query_start_step'] != query_start or row['environment_step'] != query_start + offset:
                raise ValueError('Physical action/query boundary mismatch')
            if offset and (row['has_existing_actor_feature'] or row['actor_feature_index'] is not None):
                raise ValueError('Later action cannot reuse query actor feature')
            if offset and row['existing_actor_supervision_known']:
                raise ValueError('Missing actor cannot already have a known captured-actor label')
            if row['terminal_stop'] and (row['has_observed_transition'] or row['next_memory_feature_index'] is not None):
                raise ValueError('STOP must not create an observation')
            actions.append(dict(environment_step=row['environment_step'], action_token_offset=offset,
                executed_action=row['executed_action'], recapture_required=row['requires_actor_recapture'],
                existing_actor_feature_index=row['actor_feature_index'],
                original_actor_context_query_start=query_start,
                allowed_memory_feature_index=query_start,
                physical_feature_index_audit_only=row['memory_feature_index'],
                physical_next_feature_index_audit_only=row['next_memory_feature_index'],
                autoregressive_prefix_action_ids=[previous['executed_action'] for previous in chunk[:offset]],
                original_existing_actor_supervision_known=row['existing_actor_supervision_known'],
                original_action_in_registered_supervised_region=row['action_in_registered_supervised_region'],
                label_admission='UNCHANGED_NO_NEW_TRAINING_ADMISSION',
                terminal_stop=row['terminal_stop'], next_observation_status=row['next_observation_status']))
        requests.append(dict(request_id=f'{trajectory_id}:q{query_index}', trajectory_id=trajectory_id,
            pool_row_index=first['pool_row_index'], split=first['split'], kind=first['kind'],
            query_index=query_index, query_start_step=query_start,
            original_actor_context_query_start=query_start,
            memory_replay_exclusive_end=query_start + 1,
            allowed_memory_feature_index=query_start,
            chunk_length=len(chunk), missing_actor_positions=sum(row['requires_actor_recapture'] for row in chunk),
            capture_status='NOT_CAPTURED', actions=actions))
    return requests


def build(index, output):
    began = time.time()
    original = read(index / 'RESULT.json')
    if original['status'] != 'TOKEN_INDEX_COMPLETE':
        raise ValueError('Token index is incomplete')
    for name, digest in original['artifact_hashes'].items():
        if sha(index / name) != digest:
            raise ValueError('Sealed token-index artifact changed')
    rows = [json.loads(line) for line in (index / 'ACTION_TOKENS.jsonl').read_text().splitlines()]
    if len(rows) != original['actions']:
        raise ValueError('Frozen action count mismatch')
    requests = requests_from_rows(rows)
    counts = defaultdict(Counter)
    stop_subset = []
    for request in requests:
        count = counts[request['split'] + ':' + request['kind']]
        count['queries'] += 1
        for action in request['actions']:
            if not action['recapture_required']:
                continue
            count['missing_actor_positions'] += 1
            count['in_original_registered_supervision_region'] += action['original_action_in_registered_supervised_region']
            count['original_existing_actor_supervision_known'] += action['original_existing_actor_supervision_known']
            count['unknown_registered_prefix_positions'] += not action['original_action_in_registered_supervised_region']
            if action['terminal_stop']:
                count['terminal_stop_missing_positions'] += 1
                stop_subset.append(dict(request_id=request['request_id'], trajectory_id=request['trajectory_id'],
                    split=request['split'], kind=request['kind'], query_start_step=request['query_start_step'],
                    environment_step=action['environment_step'], action_token_offset=action['action_token_offset'],
                    allowed_memory_feature_index=request['query_start_step'], next_observation=None))
    missing = sum(value['missing_actor_positions'] for value in counts.values())
    expected = sum(value['requires_actor_recapture'] for value in original['by_split_kind'].values())
    if missing != expected:
        raise ValueError('Not all originally missing actor positions are covered')
    expected_stops = sum(value['terminal_stop_requires_actor_recapture'] for value in original['by_split_kind'].values())
    if len(stop_subset) != expected_stops:
        raise ValueError('Not all missing terminal STOP actors are covered')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'REQUESTS.json', requests)
    result = dict(status='RECAPTURE_PLAN_COMPLETE_NOT_CAPTURED', planned_queries=len(requests),
        planned_missing_actor_positions=missing, planned_terminal_stop_positions=len(stop_subset),
        by_split_kind={key: dict(value) for key, value in sorted(counts.items())},
        priority_terminal_stop_subset=stop_subset,
        original_token_index=str(index), source_hashes={str(path): sha(path) for path in
            (Path(__file__), index / 'RESULT.json', index / 'ACTION_TOKENS.jsonl', index / 'TRAJECTORIES.json')},
        requests_sha256=sha(output / 'REQUESTS.json'),
        causal_rule='All action tokens in a generated chunk use memory replay only through its original query_start; later physical observations are posterior audit evidence only.',
        teacher_forcing_rule='Only preceding action tokens in the same original generated chunk may enter the later token context; no later physical RGB, memory update or outcome.',
        supervision_rule='Original known mask and registered-supervision-region flags preserved separately. Planning or recapture does not admit new labels.',
        feature_semantics='Original mixed visual/instruction/previous-action representation, not pure visual.',
        deployment_boundary='Current deployment modifies only the first action token. Additional actor captures do not fix deployment; all-token residual changes require a new shared version for every comparison arm.',
        generated_actor_features=0, source_assets_modified=False, gpu_hours=0,
        optimizer_updates=0, simulated_steps=0, unseen_read=False, wall_seconds=time.time() - began)
    write(output / 'RESULT.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'planned_queries', 'planned_missing_actor_positions',
                                                 'planned_terminal_stop_positions', 'by_split_kind', 'wall_seconds')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, default=Path(__file__).resolve().parent / 'token_index')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'recapture_plan_001')
    args = parser.parse_args()
    build(args.index, args.output)
