"""Recompute frozen extension results; keep old discovery routes separate."""
import csv
import random
import statistics
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent/'recovery_confirmation_v2')]
import common as u
from transfer_pipeline import records
from confirmation_review import summarize


def summary(run):
    return summarize(records(run, 'evaluation', True), u.read(run/'DATA_MANIFEST.json')['episodes'], u.read(run/'PROTOCOL.json'))


def review(run, hours):
    p = u.read(run/'PROTOCOL.json')
    u.verify_sources(run)
    entries = u.read(run/'DATA_MANIFEST.json')['episodes']
    rows = records(run, 'evaluation', True)
    assert set(rows) == {e['id'] for e in entries}, 'INCOMPLETE_EXTENSION'
    for r in rows.values():
        path = Path(r['path'])
        session = path.parents[2]
        seal = u.read(session/'STATE_SEAL.json')
        assert seal['heads_unchanged'] and seal['base_before']==seal['base_after']==p['expected_base_state_sha256']
        assert u.sha(session/'RUNTIME_IDENTITY.json') == r['runtime_identity_sha256']
        assert u.read(session/'RUNTIME_IDENTITY.json')['heads'] == p['heads']
        assert set(r['audits']) == set(p['models']) and set(r['seed_pair_audits']) == {str(s) for s in p['seeds']}
        for a in [*r['audits'].values(), *r['seed_pair_audits'].values()]:
            assert a['input_prefix_matched'] and a['action_prefix_matched'] and a['argmax_flip_count']==0
        for arm, digest in r['trace_hashes'].items():
            assert u.sha(path.parent/arm/'TRACE.jsonl') == digest
    result = summary(run)
    houses = sorted({r['house'] for r in rows.values()})
    differences = {}
    for reference in ('NATIVE', 'LOCAL'):
        house_values = {}
        for h in houses:
            selected = [r for r in rows.values() if r['house']==h]
            house_values[h] = [statistics.mean(r['outcomes'][f'CONCAT_s{s}']['success']-
                r['outcomes']['NATIVE' if reference=='NATIVE' else f'LOCAL_s{s}']['success'] for s in p['seeds']) for r in selected]
        rng = random.Random(1209)
        draws = []
        for _ in range(2000):
            sample = rng.choices(houses, k=len(houses))
            draws.append(sum(sum(house_values[h]) for h in sample)/sum(len(house_values[h]) for h in sample))
        draws.sort()
        differences[reference] = dict(mean_paired_delta=statistics.mean(v for group in house_values.values() for v in group),
            house_equal_delta=statistics.mean(statistics.mean(v) for v in house_values.values()),
            descriptive_house_bootstrap_95=[draws[49], draws[1949]], houses=len(houses),
            limitation='House-cluster resampling retains all seeds and routes within a house. Same known houses, few clusters; descriptive uncertainty, not blind broad generalization.')
    result.update(status='COMPLETE', gpu_hours=hours, base_updates=0, optimizer_updates=0, heads_updated=False,
                  full_1839=False, adopted=False, paired_uncertainty=differences,
                  old_discovery_report=str(Path(p['source_run'])/'RESULT.json'),
                  pooled_old_new_score_used_for_decision=False, exposure_scope=p['exposure_scope'])
    deltas = result['seed_summary']['CONCAT']['deltas_vs_native']
    result['descriptive_positive_replication'] = statistics.mean(deltas)>0 and sum(d>0 for d in deltas)>=2
    names = ['NATIVE', *p['models']]
    metadata = {e['id']: e for e in entries}
    with (run/'ROLLOUTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['id','episode_id','trajectory_id','house','arm','success','spl','steps','trace','trace_sha256'])
        writer.writeheader()
        for i, r in sorted(rows.items()):
            e = metadata[i]
            for arm in names:
                v = r['outcomes'][arm]
                writer.writerow(dict(id=i, episode_id=e['episode_id'], trajectory_id=e['trajectory_id'], house=e['house'], arm=arm,
                    success=v['success'], spl=v['spl'], steps=v['steps'], trace=str(Path(r['path']).parent/arm/'TRACE.jsonl'), trace_sha256=r['trace_hashes'][arm]))
    u.write(run/'RESULT.json', result)
    lines = ['COMPLETE', '', f"新增 {len(entries)} 条物理路线 × 原生及六个固定模型，共 {len(entries)*7} 次真实执行。旧200条单列，不并入主结论。", '']
    for name, v in result['arms'].items():
        lines.append(f"{name}: {v['successes']}/{v['planned']}; SR {v['sr']:.4f}; SPL {v['spl']:.4f}; 平均动作 {v['mean_steps']:.2f}")
    lines += ['', '各 seed 对原生变化：'+str(deltas), '各 seed 对 LOCAL 变化：'+str(result['seed_summary']['history_deltas']),
              '新增路线描述性正向复验：'+str(result['descriptive_positive_replication']),
              '公开 benchmark 与这些房屋已暴露；路线新于本 StreamVLN 分支登记清单，不是新房屋或盲测。',
              '记忆机制诊断见 mechanism/REPORT_ZH.md；不能把离线标签变化当新增闭环 SR。',
              f'本轮 GPU 会话小时 {hours:.4f}；0 次参数更新；未自动采用或开启新训练。']
    (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    review(args.run, u.read(args.run/'STATUS.json')['gpu_hours'])
