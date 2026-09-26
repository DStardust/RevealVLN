"""Full denominators and same-session comparisons for all-token adaptation."""
import csv
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import common as u


def records(run):
    rows = {}
    if not (run / 'PROTOCOL.json').exists():
        return rows
    p = u.read(run / 'PROTOCOL.json')
    names = {'NATIVE', *p['heads']}
    for session in sorted((run / 'evaluation').glob('*')):
        seal_path = session / 'STATE_SEAL.json'
        if not seal_path.exists():
            continue
        seal = u.read(seal_path)
        if not seal['heads_unchanged'] or seal['base_before'] != seal['base_after'] or seal['base_before'] != p['expected_base_state_sha256']:
            raise ValueError('INVALID_EVALUATION_SEAL')
        identity = u.read(session / 'RUNTIME_IDENTITY.json')
        if identity['heads'] != p['heads'] or identity['base_updates'] != 0:
            raise ValueError('EVALUATION_IDENTITY_CHANGED')
        identity_sha = u.sha(session / 'RUNTIME_IDENTITY.json')
        for path in session.glob('episodes/*/COMPLETE.json'):
            row = u.read(path)
            if row['id'] in rows or set(row['outcomes']) != names:
                raise ValueError('DUPLICATE_OR_INCOMPLETE_MODEL_GROUP')
            if row['runtime_identity_sha256'] != identity_sha:
                raise ValueError('GROUP_IDENTITY_CHANGED')
            if set(row['audits']) != set(p['heads']) or set(row['seed_pair_audits']) != {str(s) for s in p['seeds']}:
                raise ValueError('MISSING_PAIR_AUDIT')
            for a in [*row['audits'].values(), *row['seed_pair_audits'].values()]:
                if not a['input_prefix_matched'] or not a['action_prefix_matched'] or a['argmax_flip_count']:
                    raise ValueError('INVALID_PAIR_PREFIX')
            rows[row['id']] = dict(row, path=str(path))
    return rows


def summarize(rows, entries, names, seeds):
    expected = {e['id'] for e in entries}
    if len(expected) != len(entries) or not set(rows) <= expected:
        raise ValueError('REGISTERED_DENOMINATOR_CHANGED')
    n = len(expected); complete = len(rows) == n
    if not n:
        raise ValueError('EMPTY_DENOMINATOR')
    if any(set(r['outcomes']) != set(names) for r in rows.values()):
        raise ValueError('INCOMPLETE_MODEL_GROUP')
    arms = {}
    for arm in names:
        values = [r['outcomes'][arm] for r in rows.values()]
        successes = sum(v['success'] for v in values)
        arms[arm] = dict(complete=len(values), planned=n, successes=successes,
            sr=successes / n if complete else None,
            sr_identification_bounds=[successes / n, (successes + n - len(values)) / n],
            spl=statistics.mean(v['spl'] for v in values) if complete else None,
            mean_steps=statistics.mean(v['steps'] for v in values) if complete else None)
    comparisons = [(a, 'NATIVE') for a in names if a != 'NATIVE']
    comparisons += [(f'DELTA_s{s}', f'CURRENT_s{s}') for s in seeds]
    paired = {}
    for left, right in comparisons:
        wins = [i for i,r in rows.items() if r['outcomes'][left]['success'] > r['outcomes'][right]['success']]
        losses = [i for i,r in rows.items() if r['outcomes'][left]['success'] < r['outcomes'][right]['success']]
        net = len(wins) - len(losses)
        paired[left + '_vs_' + right] = dict(wins=wins, losses=losses,
            retained_successes=sum(bool(r['outcomes'][left]['success'] and r['outcomes'][right]['success']) for r in rows.values()),
            delta_sr=net / n if complete else None,
            difference_identification_bounds=[(net - (n-len(rows))) / n, (net + n-len(rows)) / n])
    by_house = {}
    for house in sorted({e['house'] for e in entries}):
        planned = sum(e['house'] == house for e in entries)
        selected = [r for r in rows.values() if r['house'] == house]
        by_house[house] = {a:dict(planned=planned, complete=len(selected),
            successes=sum(r['outcomes'][a]['success'] for r in selected)) for a in names}
    return dict(evaluation_complete=complete, complete_groups=len(rows), planned_groups=n,
        arms=arms, paired=paired, by_house=by_house,
        delta_minus_current_by_seed=[paired[f'DELTA_s{s}_vs_CURRENT_s{s}']['delta_sr'] for s in seeds],
        statistical_unit='Seeds share routes and houses; executions are not independent generalization samples.',
        old_candidate_retained=True, automatic_replacement=False)


def summary(run):
    p = u.read(run / 'PROTOCOL.json')
    return summarize(records(run), u.read(run / 'DATA_MANIFEST.json')['episodes'], ['NATIVE', *p['heads']], p['seeds'])


def review(run):
    u.verify_sources(run)
    p = u.read(run / 'PROTOCOL.json'); rows = records(run)
    entries = u.read(run / 'DATA_MANIFEST.json')['episodes']
    result = summarize(rows, entries, ['NATIVE', *p['heads']], p['seeds'])
    if not result['evaluation_complete']:
        raise ValueError('INCOMPLETE_UNSEEN_EVALUATION')
    metadata = {e['id']: e for e in entries}
    output = []
    for i, row in sorted(rows.items()):
        group = Path(row['path']).parent
        for arm, outcome in row['outcomes'].items():
            trace = group / arm / 'TRACE.jsonl'
            if u.sha(trace) != row['trace_hashes'][arm]:
                raise ValueError('EVALUATION_TRACE_CHANGED')
            output.append(dict(id=i, episode_id=metadata[i]['episode_id'], house=row['house'], arm=arm,
                success=outcome['success'], spl=outcome['spl'], steps=outcome['steps'],
                trace=str(trace), trace_sha256=row['trace_hashes'][arm]))
    with (run / 'ROLLOUTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    result.update(status='COMPLETE', split='val_unseen', full_1839=False, exposed=True, base_updates=0)
    u.write(run / 'RESULT.json', result)
    lines = ['COMPLETE', '', f"{len(rows)}个完整组、{len(output)}次执行；全部种子和屋分母保留。", '',
        '主比较为同数据同容量的 DELTA−CURRENT，原生策略为同进程参照。全动作覆盖的共同变化不算 DELTA 独有贡献。', '']
    for arm, value in result['arms'].items():
        lines.append(f"{arm}: SR {value['sr']:.4f}; SPL {value['spl']:.4f}; 平均动作 {value['mean_steps']:.2f}")
    lines += ['', 'DELTA−CURRENT 各种子SR差：' + str(result['delta_minus_current_by_seed']),
        '旧CONCAT正向候选继续保留，本轮不自动替换。新结果不等于完整1839或盲测泛化。']
    (run / 'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return result
