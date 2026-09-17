# 初步跨房屋重复生产门槛 V1 与只读审核 wrapper

按本次主 agent 指令冻结判据，机器规格见 [SPEC.json](SPEC.json)。`executable=false` 表示本交付**不授权或启动生产批次**；提供的 CLI/函数只读已结束运行并写新的审核产物。

## 唯一允许的结论范围

通过时只称“初步跨房屋重复生产证据”。不是统计稳定、模型泛化、科学收益或论文贡献 PASS。

- 两个不同、分别事前冻结的批次。
- 每批至少 3 个已尝试独立物理 hub，每批至少 2 个质量合格独立 hub。
- 合计至少 6 个不同物理合格 hub，至少 3 个 FIT 房屋。
- 同房屋 hub 距离 **小于 1 m** 取连通分组；相隔恰好 1 m 可以分别计数。改角色、改 seed、改朝向不产生新 hub。连通分组是保守去重，不声称统计样本独立。
- 冻结候选集合中的所有尝试与终态必须保留。缺失、运行中、未封存、未知、失败不能算合格。
- control_type 分层计数，不将 completed-subgoal-revisit、event_free、spatial_detour 改称同一控制机制。

## 已实现的调用

[acceptance.py](acceptance.py) 只读加载已封存 quality.py，不修改它。

```python
from acceptance import build_family_evidence, audit_batches

# 不限于 V2，同样接受 short_revisit_v3/后续独立 run。
report = build_family_evidence(run_root, bundle_id, new_review_output)

# 必须由 wrapper 自己重建 family evidence 和审核，不能把外部 PASS 字典当结果。
batch_report = audit_batches([first_closed_run, second_closed_run], new_batch_output)
```

所有新输出必须是本目录下尚不存在的新子目录。不修改任何源 run 文件。参数 `postprocess_result=path` 可为独立后处理接受文件；原失败 result 仍保留，实际 27 回放/54 求值/18 格必须完整复核。正常新 worker 可直接用 bundle/result.json。

单族 CLI（项目根，替换实际 run/bundle 和新输出路径）：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/witness_first_v1/quality_cpu/batch_acceptance_v1/acceptance.py --run-root RUN_ABSOLUTE_PATH --bundle BUNDLE_ID --output NEW_REVIEW_ABSOLUTE_PATH
```

## 前后两种哈希不能混淆

1. **动作前配置证据**：INPUT_LOCK 中确实有当前 EXECUTION_CONFIG 哈希，锁定源码/资产全部仍匹配；本地 INPUT_LOCK/配置文件时间不晚于 PROCESS.started_unix；journal genesis 与配置逐值相同，完整链及 HEAD 校验通过，后续实际 action/phase 记录在其后。实际 physical candidate 的 freeze 记录则必须在 certification 阶段之前。
2. **生成后完整性封存**：builder 在运行完全收口后采集最终 export、内容、证书、readback、预算、资源、trace 与来源文件哈希；写 EVIDENCE.json 并调用原严格 family auditor。这些新哈希**不是事前注册时间证明**。

动作前证据依赖受信任的本地主机时钟、既有 source/input lock 与源 worker 的同步日志顺序；不声称具有外部可信时间戳或能对抗管理员重写整条哈希链。源 INPUT_LOCK、配置、PROCESS、全部资源快照、supervisor/worker 终态、预算/store 终态、journal/HEAD 都绑定到审核 source_files。

## 如何区分 discovery 与 27 条 certification

逐条读取真实 journal，维护最后一份 budget.active。当 trace_saved 发生且 active 精确等于 `[bundle_id, 'certification']` 时才选取；索引对应实际 trace 文件、SHA256 和 complete 必须匹配。必须恰好 27 个。

因此不会把 discovery 的重复轨迹混入三 seed 认证，不靠“最后 27 个文件”猜阶段。冻结 candidate JSON 也必须与认证前 FreezeLedger 记录逐值相同。最终标签、query、计数和内容复算仍由未改 quality.audit_family 完成。

## 资源、预算与 FIT 绑定

- 验证所有已有 G/C/C+G process 快照及最终 gpu_after 的 UUID、外部单进程≤768 MiB/合计≤2048 MiB、自有保守上界<4096 MiB及计量一致性。
- 实际 worker PID 必须已从 graphics-aware 最终清单消失，源 supervisor cleanup_complete=true、无外部进程被停止、未触碰占位；不发任何信号。
- 保留资源样本和 supervisor 终态来源。RSS 未逐项写入快照，不能宣传离线重算了所有 RSS 数值；原 supervisor 的 RSS 检查代码与正常资源终态被来源绑定。
- 原预算类只读校验 final state，必须无 active phase；store 必须正常关闭且完整审计通过。
- 通过 INPUT_LOCK 中与 cfg.split_sha256 相同的实际 split 文件确认每个房屋属于 FIT，不接受调用方临时自称 FIT。

## 实际验证与当前状态

14 项 CPU 测试通过：1 m 边界、hub/角色/seed 去重、每批分母、未知与未绑定结果、控制分层、journal 配置/哈希/尾截断、认证阶段选择、G renderer 与显存边界、缺收口证据不放行。合成正例只检验规则，不计合格物理数据。

对已收口 short_revisit_v2 的真实只读 builder 已执行：锁定来源、阶段日志、预算、资源和 cleanup 核验通过，但缺少 FROZEN_CANDIDATE/CERTIFICATE/export；结论仍为 `quality_pass=false`。原运行 8 traces/1648 actions/0 collisions/0 族保留，不用审核生成文件制造正例。

V3/后续运行未收口时不得预先生成质量通过结果。完整真实成功族和两个正式批次的 acceptance 仍须实际运行结果到齐后检验。`gate_from_verified_batches` 是 CPU 测试辅助逻辑；生产报告必须来自会自行验证所有来源的 `audit_batches` wrapper。
