"""Read the frozen FIT/DEV pool once; audit supervision without training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch


def read(path):
    return json.loads(path.read_text())


def histogram(values):
    return dict(sorted(Counter(int(x) for x in values).items()))


def main(base, output):
    began = time.time()
    torch.set_num_threads(2)
    run = base / 'recovery_action_v1/runs/action_001'
    pool_path = run / 'data/POOLS.pt'
    admission = read(run / 'data/ADMISSION.json')
    # mmap retains a single pool load and avoids materializing unused features.
    pack = torch.load(pool_path, map_location='cpu', weights_only=True, mmap=True)
    manifest = {e['id']: e for e in read(run / 'features/DATA_MANIFEST.json')['episodes']}
    lock = read(run / 'features/SOURCE_LOCK.json')['files']
    groups = defaultdict(list)
    row_results = []
    for row in pack['rows']:
        assert row['partition'] in ('FIT', 'DEV')
        entry = manifest[row['id']]
        path = Path(entry['trajectory'])
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == lock[str(path)]
        trace = json.loads(raw)
        actions = torch.tensor(trace['actions'], dtype=torch.long)
        queries = row['query_steps'].long()
        target = row['targets'].long()
        known = row['known'].bool()
        native = row['base_logits'].argmax(-1)
        assert len(actions) == len(row['memory_features']) == len(trace['rgb_sha256'])
        assert queries.tolist() == trace['query_steps']
        assert torch.equal(actions[queries], target)
        expected = torch.ones_like(known) if row['kind'] == 'PRESERVATION' else queries >= row['cutoff']
        assert torch.equal(known, expected)
        assert torch.isfinite(row['base_logits']).all()
        assert trace['admitted'] and trace['success'] and int(actions[-1]) == 0
        assert len(actions) <= 500 and int(queries[-1]) < len(actions)
        cutoff = int(row['cutoff'])
        supervised_steps = (len(actions) if row['kind'] == 'PRESERVATION' else len(actions) - cutoff)
        collision_steps = [i for i, e in enumerate(trace['events']) if e['collision']]
        native_stop_continue = (native == 0) & (target != 0) & known
        native_continue_stop = (native != 0) & (target == 0) & known
        item = dict(id=row['id'], split=row['partition'], kind=row['kind'], house=row['house'],
                    route_family=entry['route_family'], physical_steps=len(actions), queries=len(queries),
                    known_queries=int(known.sum()), unknown_prefix_queries=int((~known).sum()),
                    supervised_action_steps=supervised_steps,
                    supervised_actions_without_actor_features=supervised_steps - int(known.sum()),
                    cutoff=cutoff, query_gaps=(queries[1:] - queries[:-1]).tolist(),
                    target_actions=target[known].tolist(), all_actions=actions.tolist(),
                    native_actions=native[known].tolist(),
                    known_native_disagreements=int(((native != target) & known).sum()),
                    native_stop_teacher_continue=int(native_stop_continue.sum()),
                    native_continue_teacher_stop=int(native_continue_stop.sum()),
                    terminal_stop_is_query=int(queries[-1]) == len(actions) - 1,
                    stop_conflict_query_steps=queries[native_stop_continue | native_continue_stop].tolist(),
                    native_terminal_success=bool(entry['native_outcome']['success']),
                    native_final_steps=entry['native_outcome']['steps'],
                    collected_terminal_success=trace['success'],
                    collision_count=len(collision_steps),
                    supervised_collision_count=sum(i >= cutoff or row['kind'] == 'PRESERVATION' for i in collision_steps),
                    adjacent_observed_transitions=len(actions) - 1,
                    action_at_observed_transition=actions[:-1].tolist(),
                    trajectory_path=str(path), trajectory_sha256=hashlib.sha256(raw).hexdigest())
        assert 0 not in actions[:-1].tolist(), 'STOP cannot supply a next physical observation'
        groups[(row['partition'], row['kind'])].append(item)
        row_results.append(item)

    summaries = {}
    for (split, kind), rows in sorted(groups.items()):
        matrix = [[0] * 4 for _ in range(4)]
        for r in rows:
            for n, t in zip(r['native_actions'], r['target_actions']):
                matrix[n][t] += 1
        fields = ('physical_steps', 'queries', 'known_queries', 'unknown_prefix_queries',
                  'supervised_action_steps', 'supervised_actions_without_actor_features',
                  'known_native_disagreements', 'native_stop_teacher_continue',
                  'native_continue_teacher_stop', 'collision_count', 'supervised_collision_count',
                  'adjacent_observed_transitions')
        summaries[split + ':' + kind] = dict(rows=len(rows), houses=len({r['house'] for r in rows}),
            route_families=len({r['route_family'] for r in rows}),
            **{f: sum(r[f] for r in rows) for f in fields},
            target_action_counts=histogram(a for r in rows for a in r['target_actions']),
            native_action_counts=histogram(a for r in rows for a in r['native_actions']),
            all_executed_action_counts=histogram(a for r in rows for a in r['all_actions']),
            query_gap_counts=histogram(a for r in rows for a in r['query_gaps']),
            cutoff_counts=histogram(r['cutoff'] for r in rows),
            terminal_stops_without_actor_feature=sum(not r['terminal_stop_is_query'] for r in rows),
            native_by_target_confusion=matrix,
            observed_transition_action_counts=histogram(a for r in rows for a in r['action_at_observed_transition']))
    houses = {s: {r['house'] for r in row_results if r['split'] == s} for s in ('FIT', 'DEV')}
    routes = {s: {r['route_family'] for r in row_results if r['split'] == s} for s in ('FIT', 'DEV')}
    assert not houses['FIT'] & houses['DEV'] and not routes['FIT'] & routes['DEV']
    # Keep the output small; aggregates preserve action distributions.
    for row in row_results:
        for key in ('all_actions', 'target_actions', 'native_actions', 'query_gaps', 'action_at_observed_transition'):
            del row[key]
    result = dict(status='CPU_DATA_AUDIT_COMPLETE', method_benefit='NOT_MEASURED',
        pool_path=str(pool_path), pool_sha256_recorded=admission['pools_sha256'],
        pool_sha256_recomputed=False, pool_loads=1, pool_keys=list(pack), row_keys=list(pack['rows'][0]),
        trace_hashes_checked=len(row_results), summaries=summaries, action_order=['STOP', 'FORWARD', 'LEFT', 'RIGHT'],
        confusion_definition='Rows=cached native argmax, columns=actual query-time target; only known labels.',
        houses={s: sorted(v) for s, v in houses.items()}, fit_dev_overlap=False,
        objective='recovery CE + 0.5 ordinary CE + 5 (ordinary KL + target margin)',
        class_weights=pack['class_weights'].tolist(),
        branch_asset_scope=dict(recovery_rows=sum(r['kind'] == 'RECOVERY' for r in row_results),
            compared_policies='Recorded failed native continuation versus actually executed goal teacher from exact replay cutoff.',
            alternative_single_action_outcomes_in_pool=False,
            privilege='Goal position is used only by offline teacher; its success is not language-program fidelity.',
            causal_limit='Two whole remaining policies differ; do not label each native action wrong or each alternative unexecuted action FAIL.'),
        transition_scope=dict(features='(normalized visual + normalized instruction + normalized previous executed-action embedding)/3',
            dense_inputs_are_causal=True, separate_visual_components_in_pool=False,
            all_actual_actions_in_trajectory=True, raw_rgb_arrays_in_pool=False,
            endpoint_observation_after_stop=False, unseen_read=False),
        rows=row_results, gpu_hours=0, optimizer_updates=0, wall_seconds=time.time() - began)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], summaries=summaries, wall_seconds=result['wall_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'DATA_ACTION_AUDIT.json')
    args = p.parse_args()
    main(args.base, args.output)
