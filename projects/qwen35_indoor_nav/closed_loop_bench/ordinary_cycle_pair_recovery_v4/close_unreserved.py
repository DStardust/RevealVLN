"""Record the resource-blocked outcome once. This is not a GPU launcher."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    if (HERE / 'RESULT.json').exists():
        raise FileExistsError('V4 already closed; no overwrite or automatic retry')
    spec = importlib.util.spec_from_file_location('v4_audit', HERE / 'audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    allocation_env = {k: v for k, v in os.environ.items()
                      if k.startswith(('SLURM_', 'PBS_', 'LSB_'))}
    commands = {k: shutil.which(k) for k in ('scontrol', 'squeue', 'salloc', 'qstat', 'bjobs')}
    if allocation_env or any(commands.values()):
        raise RuntimeError('Scheduler evidence available: inspect it before declaring unavailable')
    source = read(HERE / 'SOURCE_AUDIT.json')
    tests = read(HERE / 'cpu_attempts/attempt_002/CPU_TEST_RESULT.json')
    if not source['passed'] or not tests['passed']:
        raise RuntimeError('Resolve CPU/source audit before closing this prepared package')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    reason = 'No scheduler allocation or actual-user dedicated GPU window receipt was available'
    lease = dict(status='RESOURCE_NOT_RESERVED', checked_at=now,
                 host=os.uname().nodename, selected_gpu_uuid=None, selected_gpu_index=None,
                 scheduler_commands=commands, scheduler_environment=allocation_env,
                 coordination_receipt=None, confirmation_requested_in_session=True,
                 confirmed_window_start=None, confirmed_window_end=None,
                 inspected_legacy_flock='ordinary_cycle_pair_gpu1_v3/gpu1_eval.lock',
                 legacy_flock_is_acceptable_proof=False, idle_device_is_acceptable_proof=False,
                 resource_polling_attempts=0, gpu_selection_attempts=0,
                 gpu_reservation_acquired=False, reason=reason)
    write('RESOURCE_LEASE.json', lease)
    test_summary = dict(tests, accepted_attempt='cpu_attempts/attempt_002',
                        preserved_failed_attempt='cpu_attempts/attempt_001',
                        first_failure='Test Sim returned list positions; official SPL requires ndarray',
                        correction='Only CPU fixture type corrected; frozen metrics unchanged')
    write('CPU_TEST_RESULT.json', test_summary)
    assets = [r for r in source['entries'] if '/runtime/models/' in r['path']]
    unavailable = dict(status='unavailable', reason='GPU resource gate closed; policy was never loaded')
    driver = Path('/proc/driver/nvidia/version')
    identity = dict(status='NOT_STARTED_RESOURCE_NOT_RESERVED',
                    cpu_preflight_python=sys.version, cpu_preflight_executable=sys.executable,
                    on_disk_source_assets=assets, model_loaded=False, model_load_count=0,
                    checkpoint_sha256=source['checkpoint_sha256'],
                    model_source_sha256=source['model_source_sha256'],
                    all_loaded_parameter_fingerprints=unavailable,
                    persistent_buffer_fingerprints=unavailable,
                    pre_post_model_state_comparison=unavailable,
                    effective_processor_tokenizer_fingerprints=unavailable,
                    model_runtime_software_versions=unavailable,
                    actual_attention_fla_dispatch=unavailable,
                    triton_compilation_selection_config=unavailable,
                    selected_gpu_uuid=unavailable,
                    host_driver_read_only=driver.read_text() if driver.is_file() else 'unavailable',
                    disk_hashes_are_not_loaded_state_evidence=True)
    write('RUNTIME_IDENTITY.json', identity)
    gaps = [
        'Verified exclusive GPU UUID/window allocation is missing',
        'V4 evaluator/launcher integration was not implemented past the mandatory resource stop',
        'Full loaded parameters/buffers and effective processor fingerprints were not measured',
        'Seven-tensor per-decision hashing and immediate online prefix stop are not integrated',
        'Full 100-pair metrics/cost/latency/uncertainty reporting has no new trajectory evidence',
    ]
    write('PREFLIGHT.json', dict(status='RESOURCE_NOT_RESERVED', checked_at=now,
          workspace_audit='WORKSPACE_AUDIT.json', source_audit='SOURCE_AUDIT.json',
          source_integrity_passed=True, cpu_contract_tests_passed=True,
          runtime_integration_ready=False, resource_reserved=False, formal_launch_allowed=False,
          formal_starts=0, gpu_contexts_created=0, not_run_requirements=gaps))
    result = audit.blocked_result(reason)
    result.update(closed_at=now, source_integrity_passed=True, cpu_contract_tests_passed=True,
                  runtime_integration_ready=False, full_val_unseen_run=False,
                  deployment_changed=False, required_pairs=100, missing_pairs=100)
    write('RESULT.json', result)
    write('REVIEW.json', dict(status='RESOURCE_NOT_RESERVED', result='RESULT.json',
          task_execution_success=False, resource_gate_obeyed=True,
          evaluation_valid=False, intervention_benefit='unknown',
          scientific_generalization_established=False, engineering_candidate_pass=None,
          paired_gates=None, historical_guard=None, pair_wins=None, pair_losses=None,
          retained_successes=None, by_house=None, repetition_and_action_costs=None,
          stop_failure_classes=None, latency_p50_p95=None, descriptive_uncertainty=None,
          next_evidence_needed=gaps, automatic_retry_allowed=False,
          no_partial_or_historical_metrics_admitted=True))
    with (HERE / 'POLICY_STEPS.jsonl').open('x'):
        pass  # Zero decisions actually occurred; no placeholder policy records.
    with (HERE / 'EVENTS.jsonl').open('x') as stream:
        for event in ('WORKSPACE_AND_SOURCE_CHECKED', 'CPU_CONTRACT_TESTS_COMPLETED',
                      'RESOURCE_NOT_RESERVED_NO_FORMAL_START'):
            stream.write(json.dumps(dict(recorded_at=now, event=event,
                                         formal_starts=0, environment_decisions=0)) + '\n')
    report = '''RESOURCE_NOT_RESERVED

本轮在用户规定的资源准入门前停止：没有调度器分配，也没有实际 GPU 使用者协调确认的
专用窗口凭据。没有选择 GPU、启动模型/仿真、轮询抢卡或更换 GPU。正式启动 0 次、
环境决策 0 次、optimizer update 0 次；没有本轮 GPU 子进程需要清理，未向外部进程发信号。

工作区和来源核验

审查基准提交为 9d22783e646ef86232ba3a3a59d7bb9ce8a8c0a7。开始时只有根目录
.gitignore、AGENTS.md、README.md 的既有未提交改动，Q35N 已跟踪文件没有变化。
这些既有修改保留；WORKSPACE_AUDIT.json 记录检查结果。
V3 源锁 198 项全部通过，指定 best4k 和模型源码 SHA 符合用户要求。
EPISODES_PRIVILEGED.json 的前后 100 条逐项相同，保留五屋分母和 V3 调度。
cycle_policy.py 按字节复制，SHA 保持 75eb778a4a8b93f1ef00de1704330a83fdd23b4a5cac294982cde4e59cafb430。

CPU 验收与边界

15 项纯 CPU 合同测试通过，覆盖 reset、原生 argmax、STOP、平局、实际动作历史、
500 步预算、冻结官方成功定义、缺 episode 拒绝准入、精确前缀/首分歧、无覆盖完整
轨迹与终态、全部七种输入哈希、参数变化拒绝准入、进程清理身份拒绝。
首轮测试的模拟对象错误地返回 list 位置，官方 SPL 要求 numpy 数组；只修正测试夹具，
未改冻结评测实现。首次失败、日志和当时源码保留于 cpu_attempts/attempt_001；
通过记录在 cpu_attempts/attempt_002。没有重跑 FIT 重放、小集拟合或旧 GPU 启动自检。

资源停止发生在运行集成之前。本目录的 audit.py 是 CPU 审计合同，未接入 GPU 评估器；
不存在已完成的 V4 evaluate.py/launch.py 实测，也未声称启动链路就绪。现有 V3 仍缺
全模型参数/持久 buffer 指纹、七种处理后输入逐步哈希和首分歧在线立即停止，不能直接
运行 V3 冒充本任务。CPU 通过不能替代这些运行时验收。

未运行结果

A/B 的 SR、SPL、nDTW、OSR、ΔSR、胜负 episode、按屋结果、重复比例、覆盖次数、
碰撞/动作成本、STOP 分类、时延分位数和描述性区间全部未测，记为 null。
POLICY_STEPS.jsonl 为空，代表没有实际决策；EVENTS.jsonl 仅记录本轮 CPU 收口事件。
RUNTIME_IDENTITY.json 将实际加载参数、持久 buffer、processor、dispatch、Triton 配置
明确标成 unavailable。源文件和底模磁盘 SHA 不等于已加载状态的指纹。

四个判断分别为：实验执行未完成；评测未形成有效比较；干预收益 unknown；
科学泛化无新证据。资源门前停止不支持也不否定循环恢复规则。未采用规则，未修改
部署配置、checkpoint、旧 FINAL_REVIEW 或旧运行；未进入完整 val_unseen 或方向 A/B。

后续缺口

首先需要可核验的 GPU UUID、所在主机和足够覆盖 4100 秒正式预算及清理的专用窗口。
其后仍需完成并检查 V4 的只读身份/输入记录、在线前缀失败和严格清理集成，再决定
一次正式启动是否满足条件。本节点已终止，不自动重试或继续第二项实验。
即使未来得到 INTERNAL_DEV 工程候选，也不能当作 val_unseen 提升、论文创新或部署完成。
INTERNAL_DEV 与既有 val_unseen1839 已暴露；科学泛化需要冻结后的独立新测试或合规隐藏测试。
'''
    with (HERE / 'REPORT_ZH.md').open('x') as stream:
        stream.write(report)
    old_lock = read(HERE.parent / 'ordinary_cycle_pair_gpu1_v3/SOURCE_LOCK.json')
    files = dict(old_lock['files'])
    for path in sorted(HERE.rglob('*')):
        if path.is_file():
            files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    write('SOURCE_LOCK.json', dict(base_commit=source['base_commit'],
          inherited_v3_source_lock_sha256=source['source_lock_sha256'], files=files,
          scope='Resource-blocked CPU package and read-only V3 assets; not a runtime admission'))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
