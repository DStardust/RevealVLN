"""Append truthful closure reports; never rewrite an existing executed result."""
import collections
import hashlib
import json
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]
NODES = ['Q35N_G1R_COVERAGE_V1', 'Q35N_G1R_ROTATION_HISTORY_V1',
         'Q35N_G1R_EVENT_CONSTRAINT_CORRECTION_V1', 'Q35N_G1R_NUMERICAL_JOIN_V1',
         'Q35N_G1R_EVENT_AWARE_SUFFIX_V1', 'Q35N_G1R_TASK_INSTANCE_V3']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as f: json.dump(value, f, indent=2, ensure_ascii=False)


def main():
    for name in NODES:
        out = LINE/'reviews'/name
        execution = json.loads((out/'EXECUTION.json').read_text())
        restored = json.loads((out/'LEASE_RESTORED.json').read_text())
        ledgers = {}
        for file in ['DISCOVERY_ATTEMPTS.jsonl', 'BASE_ATTEMPTS.jsonl', 'FAMILY_ATTEMPTS.jsonl']:
            if (out/file).exists():
                rows = [json.loads(x) for x in (out/file).read_text().splitlines()]
                ledgers[file] = {'completed_records': len(rows), 'reasons': dict(collections.Counter(x['reason'] for x in rows))}
        if name == 'Q35N_G1R_COVERAGE_V1':
            progress = json.loads((out/'PROGRESS.json').read_text())
            save(out/'result.json', {'decision': 'EARLY_STOPPED_FOR_VERSIONED_CONSTRUCTION_CORRECTION',
                'family_certified': False, 'scientific_pass': False,
                'completed_candidate_rejections': ledgers['DISCOVERY_ATTEMPTS.jsonl']['completed_records'],
                'full_256_budget_completed': False, 'interrupted_in_flight_candidate_possible': True,
                'primitive_actions_exact': None, 'primitive_actions_lower_bound': progress['counts']['primitive_actions'],
                'early_stop_reason': 'EARLY_STOP_REVIEW.md', 'worker_terminated_by_main_agent': True})
        result = json.loads((out/'result.json').read_text())
        report = {'result': result, 'failure_ledgers': ledgers, 'execution': execution,
                  'gpu_restored': restored, 'full_family_labels_exported': 0,
                  'candidate_preflight_is_not_independent_certification': True}
        save(out/'NODE_REVIEW.json', report)
        with (out/'REPORT_ZH.md').open('x') as f:
            f.write(f'# {name} 节点收口\n\n')
            f.write(f'结论：`{result["decision"]}`。\n\n')
            f.write('此目录没有验收通过的机制训练标签；候选预检不等于独立族验收。失败与候选完整保留。\n\n')
            f.write(f'本轮runner wall：{execution["wall_seconds"]:.2f} 秒；worker返回码 {execution["returncode"]}；自身GPU清理 {execution["cleanup_complete"]}，占位恢复 {restored["restored"]}。未停止真实任务。\n\n')
            f.write('失败账本：\n\n```json\n'+json.dumps(ledgers, indent=2, ensure_ascii=False)+'\n```\n\n')
            if name == 'Q35N_G1R_COVERAGE_V1':
                f.write('coverage按主agent构造复核提前停止，不是256候选耗尽。精确总动作数未知，仅保存运行快照下界；在途候选未完成不算正式拒绝。\n\n')
            if name == 'Q35N_G1R_TASK_INSTANCE_V3':
                f.write('TV/sink任务实例已找到并冻结首个完整预检候选；原餐椅任务没有通过。独立结果另见 ../Q35N_G1R_FAMILY_CERTIFICATION_V1/。数值共同状态重建显式登记，原始精确像素汇合仍未通过。\n\n')
            f.write('未运行Qwen、SFT或机制训练；scientific_pass=false，不声称泛化、导航收益或投稿贡献成立。\n')
        paths = sorted(p for p in out.rglob('*') if p.is_file() and 'cache' not in p.relative_to(out).parts and '__pycache__' not in p.relative_to(out).parts and p.name != 'SHA256SUMS')
        with (out/'SHA256SUMS').open('x') as f:
            for path in paths: f.write(f'{sha(path)}  {path.relative_to(out)}\n')
        print(name, result['decision'], len(paths), flush=True)


if __name__ == '__main__': main()
