"""One predeclared engineering screen; never choose parameters using DEV."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import *
from event_loss import ROLES, metrics


def summarize(records, gate):
    if len(records) != 8 or {(r['arm'], r['fold']) for r in records} != {(a, i) for a in ('ORIGINAL', 'EVENT') for i in range(4)}:
        raise ValueError('INCOMPLETE_EVENT_PROBE')
    by_house = []
    for fold in range(4):
        a = next(r for r in records if r['arm'] == 'ORIGINAL' and r['fold'] == fold)
        b = next(r for r in records if r['arm'] == 'EVENT' and r['fold'] == fold)
        if a['held_house'] != b['held_house'] or a['initial'] != b['initial']:
            raise ValueError('UNMATCHED_PROBE')
        delta = {key: sum(b['metrics'][role][key]-a['metrics'][role][key] for role in ROLES)/2
                 for key in ('brier', 'recall', 'fpr')}
        by_house.append(dict(house=a['held_house'], delta=delta))
    means = {arm: {key: sum(r['metrics'][role][key] for r in records if r['arm'] == arm for role in ROLES)/8
                   for key in ('brier', 'recall', 'fpr')} for arm in ('ORIGINAL', 'EVENT')}
    delta = {key: means['EVENT'][key]-means['ORIGINAL'][key] for key in means['EVENT']}
    positive = sum(r['delta']['brier'] < 0 for r in by_house)
    advance = delta['brier'] < 0 and delta['recall'] >= gate['minimum_recall_delta'] and positive >= gate['minimum_positive_house_count']
    return dict(advance_full_policy=advance, macro=means, delta=delta, houses=by_house,
                positive_brier_houses=positive, gate=gate, completed_probes=8,
                interpretation='Equal house-role metrics on four held-out FIT houses; engineering screen only. Correlated folds, single seed, not independent confirmation or navigation benefit.')


def main(run):
    cfg = config(run); records = []
    for fold in range(4):
        for arm in cfg['arms']:
            folder = run/'event_probe'/f'{arm}_{fold}'; result = read(folder/'RESULT.json')
            if result['steps'] != cfg['event_probe_steps'] or result['base_updates'] != 0 or not result['frozen_parameters_unchanged']:
                raise ValueError('INCOMPLETE_OR_INVALID_PROBE')
            if sha(folder/'FINAL.pt') != result['final_file_sha256'] or sha(folder/'PREDICTIONS.json') != result['prediction_sha256']:
                raise ValueError('PROBE_FILE_CHANGED')
            predictions = read(folder/'PREDICTIONS.json')
            if metrics([r['probability'] for r in predictions], predictions) != result['metrics']:
                raise ValueError('PROBE_METRIC_RECOMPUTATION')
            records.append(result)
    result = summarize(records, cfg['preflight_gate'])
    result.update(records=records, actual_event_probe_updates=sum(r['steps'] for r in records),
                  full_policy_updates=0, autonomous_rollouts=0)
    immutable(run/'EVENT_PREFLIGHT.json', result)
    if not result['advance_full_policy']:
        immutable(run/'RESULT.json', dict(status='EVENT_PREFLIGHT_NOT_SUPPORTED', complete=0, planned=768,
            completed_probes=8, event_probe_updates=result['actual_event_probe_updates'], full_policy_updates=0,
            navigation_gain=None, adopted=False, reason='Fixed FIT-only event screen not met; no parameter search.',
            event_screen=result, independent_generalization=None))
    text = ['# '+('EVENT_PREFLIGHT_ADVANCED' if result['advance_full_policy'] else 'EVENT_PREFLIGHT_NOT_SUPPORTED'), '',
        '8/8 个留一 FIT 屋事件读出对照完成，3200 次真实更新；不是完整策略训练或闭环收益。', '',
        '|项（屋与角色等权）|ORIGINAL|EVENT|差值|', '|---|---:|---:|---:|']
    for key in ('brier', 'recall', 'fpr'):
        text.append(f"|{key}|{result['macro']['ORIGINAL'][key]:.6f}|{result['macro']['EVENT'][key]:.6f}|{result['delta'][key]:+.6f}|")
    text += ['', f"Brier 改善 {result['positive_brier_houses']}/4 屋。固定检出容限 −0.05；全部门槛见 EVENT_PREFLIGHT.json。",
        '只训练原初始化的事件 MLP，编码器、归一化、动作策略与记忆均冻结。未访问 DEV 得分或新 TEST。',
        '该诊断只检验监督分布修复，不宣称新模型架构、论文贡献或普通 VLN 收益。', '',
        '下一阶段：自动训练六个完整策略并执行原 DEV 768 次续接。' if result['advance_full_policy'] else
        '按预注册停止本项修复。完整策略 0/6，续接 0/768；导航收益未知，不继续调同一参数。',
        'GPU 使用与占位恢复记录见 RESOURCES.jsonl 和 PLACEHOLDER_RESTORATION_*.json。']
    (run/'REPORT_ZH.md').write_text('\n'.join(text)+'\n')
    print({k: v for k, v in result.items() if k != 'records'}, flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
