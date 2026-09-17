"""CPU-only final audit and no-overwrite delivery after the GPU worker exits."""
import io
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'p0_driver_v1'), str(HERE)]
from prepare import sha, sealed, DATA, LINE
from runtime_journal import Journal


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    cpu = HERE/'p0_cpu_v1'
    cpu.mkdir()
    tests = {}
    for module in ('test_prepare', 'test_family_job', 'test_p0_supervisor'):
        log = io.StringIO()
        outcome = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(module))
        with (cpu/(module+'.log')).open('x') as f:
            f.write(log.getvalue())
        tests[module] = {'tests': outcome.testsRun, 'failures': len(outcome.failures),
                         'errors': len(outcome.errors), 'pass': outcome.wasSuccessful() and outcome.testsRun > 0}
    save(cpu/'result.json', {'tests': tests, 'pass': all(v['pass'] for v in tests.values()),
                           'total_tests': sum(v['tests'] for v in tests.values()),
                           'P0_runtime_executed': False, 'scientific_pass': False})
    assert all(v['pass'] for v in tests.values()), tests
    # Original smoke code must remain byte-identical to its pre-execution lock.
    smoke = HERE/'smoke_v1'
    auth = json.loads((smoke/'EXECUTION_AUTH.json').read_text())
    assert all(sha(HERE/k) == v for k, v in auth['code_sha256'].items())
    assert sha(smoke/'EXECUTION_CONFIG.json') == auth['config_sha256']
    protected = [DATA, LINE/'data_pipeline/mechanism_factory_v2',
                 LINE/'reviews/Q35N_G1R_TASK_INSTANCE_V3', LINE/'parallel_readiness/v2/multifamily_plan']
    for root in protected:
        sealed(root)
    cfg = json.loads((smoke/'EXECUTION_CONFIG.json').read_text())
    with Journal(smoke/'journal', cfg, resume=True):
        pass
    result = json.loads((smoke/'result.json').read_text())
    supervisor = json.loads((smoke/'SUPERVISOR_RESULT.json').read_text())
    restore = json.loads((smoke/'GPU_RESTORE_RESULT.json').read_text())
    assert result['runtime_pass'] and supervisor['returncode'] == 0 and supervisor['error'] is None
    assert supervisor['cleanup_complete'] and restore['cleanup_complete']
    samples = json.loads((smoke/'RESOURCE_SAMPLES.json').read_text())
    usage = {'peak_sampled_RAM_bytes': max(s['ram_bytes'] for s in samples),
             'peak_sampled_GPU_mib': max(s['gpu']['memory_mib'] for s in samples),
             'final_smoke_disk_bytes': sum(p.stat().st_size for p in smoke.rglob('*') if p.is_file()),
             'wall_seconds': supervisor['wall_seconds'], 'monitoring_is_sampled_not_continuous': True}
    assert usage['final_smoke_disk_bytes'] < cfg['disk_cap_bytes']
    assert usage['peak_sampled_RAM_bytes'] < cfg['ram_cap_bytes']
    assert usage['peak_sampled_GPU_mib'] < cfg['gpu_cap_mib']
    audit = {'smoke_locked_code_unchanged': True, 'journal_chain_and_HEAD_pass': True,
             'protected_sha256_manifests': {str(p.relative_to(LINE)): sha(p/'SHA256SUMS') for p in protected},
             'resources': usage, 'gpu_cleanup_complete': True,
             'SFT_results_read': False, 'SFT_processes_stopped': False}
    save(HERE/'FINAL_AUDIT.json', audit)
    final = {'node': 'Q35N_MECHANISM_RUNTIME_INTEGRATION_V1',
             'decision': 'ACCEPT_REAL_BACKEND_V4_INTEGRATION_P0_DRAFT_ONLY',
             'runtime_pass': True, 'cpu_integration_tests': 58,
             'p0_driver_cpu_tests': sum(v['tests'] for v in tests.values()),
             'real_replays': 9, 'confirmed_motion_actions': 2877,
             'matrix_cells': 18, 'pass_cells': 12, 'fail_cells': 6,
             'policy_prefix_records': 1242, 'ce_owners': 1410,
             'legacy_pixel_semantic_pose_exact': True, 'new_physical_families': 0,
             'existing_family_split': 'interface_only', 'P0_candidates_prepared': 5,
             'P0_configurations_prepared': 232, 'P0_runtime_executed': False,
             'P0_runtime_admitted': False, 'training_started': False,
             'scientific_pass': False, 'navigation_gain': None,
             'next_gate': 'REVIEW_AND_ADMIT_FIXED_FIVE_BUNDLE_P0_WITH_THROUGHPUT_AND_RESOURCE_CHECK',
             'gpu_cleanup_complete': True, 'resources': usage}
    save(HERE/'result.json', final)
    report = f'''# 真实后端与 V4 数据接口：主 agent 收口

结论：**接入验收通过，P0 扩量尚未运行。** 当前唯一真实机制族仍为旧 interface_only，未增加独立族；本轮不是模型训练或导航收益实验。

## 实际完成

- 58 项接口 CPU 测试通过；另 {final['p0_driver_cpu_tests']} 项五候选调度/准入/异常收尾 CPU 测试通过。
- GPU2 实际回放 9 条轨迹，确认 2877 次运动动作和 9 次 STOP；18 格求值为 12 正/6 负。
- 所有 canonical RGB、语义和姿态与旧 seed1109 轨迹逐帧一致，差异为零。这里不是原始未经数值规范化的自然汇合通过；仍沿用既有 ≤1e-5 数值共同状态重建协议。
- 完整 V4 导出及独立 loader 回读：1242 条因果前缀决策、5772 条完整动作流决策、1410 个去重 CE owner。Y、M2、query 和 owner 均重新校验。
- 五个已固定 FIT 屋的 R2R-CE 源路线已解析为 56/40/48/48/40 配置，并哈希登记 20 个场景资产；总 232 个配置只是候选，不是 232 个数据族。

## 资源与隔离

监督器耗时 {usage['wall_seconds']:.2f} 秒；采样峰值 RAM {usage['peak_sampled_RAM_bytes']/1024**3:.3f} GiB、整卡显存 {usage['peak_sampled_GPU_mib']:.0f} MiB，最终 smoke 目录 {usage['final_smoke_disk_bytes']/1024**3:.3f} GiB。额度内完成，监控为离散采样，不夸称绝对连续峰值。

没有停止外部实际任务或占位程序；自身进程与 GPU 上下文已清理，不需要新增占位。未读取外部 SFT 结果、未改其数据/协议。旧族、旧 factory_v2、旧候选和多族协议全部封存哈希复核通过。

## 科学边界与下一步

本节点说明数据接口能忠实复现已有合法族，不证明新场景产率、记忆贡献、泛化或竞争力。累计动作计数/事件集合/强 M2 仍是必要对照；不能将重新导出的同一族重复计数，不能直接加入外部 SFT。

下一步按 [P0 交接](NEXT_P0_ZH.md)完成单独准入并运行五候选诊断。三 seed 认证的预算必须结合本次实测吞吐审视：不要把资源截断当成科学否证。当前 P0 draft 保持 runtime_allowed=false，未从已关闭的 smoke 自动延长 GPU 借用。

证据：[实际结果](smoke_v1/result.json)、[逐轨迹差分](smoke_v1/LEGACY_REPLAY_DIFFERENTIAL.json)、[回读](smoke_v1/LOADER_READBACK.json)、[最终审计](FINAL_AUDIT.json)、[机器裁决](result.json)。
'''
    with (HERE/'REPORT_ZH.md').open('x') as f:
        f.write(report)
    # Exclude ephemeral cache only; all executed source, arrays and evidence are sealed.
    files = [p for p in HERE.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'
             and 'cache' not in p.relative_to(HERE).parts and '__pycache__' not in p.relative_to(HERE).parts]
    assert all(p.resolve().is_relative_to(HERE) for p in files)
    with (HERE/'SHA256SUMS').open('x') as f:
        for path in sorted(files):
            f.write(sha(path)+'  '+str(path.relative_to(HERE))+'\n')
    sealed(HERE)
    print(json.dumps({'decision': final['decision'], 'sealed_files': len(files), 'resources': usage}, indent=2))


if __name__ == '__main__':
    main()
