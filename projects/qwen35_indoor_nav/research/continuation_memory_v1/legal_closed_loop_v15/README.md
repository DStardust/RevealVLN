# V15：合法历史数据与执行记忆小型对照

本目录实现实际训练和自主续接。模型基座仍为 best4k，记忆、动作读出采用 V14 的 8×64 连续记忆；未训练 Qwen、扩展 LoRA 或改变循环恢复算法。最终事实以 `REPORT_ZH.md` 和对应运行结果为准。

`DATA.json` 来自 `data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/` 的真实原始回放：6 族、72 次完整续接执行、20 条去重物理历史、12 个实际初始状态（来自 3 个粗略起点提案）、1 个 FIT 屋和 1 个 CHECK 屋。它们是经过物理及关系审核的 debug 数据，原 `training_admission=false` 保留。72 次执行不等于 72 条独立历史或多个独立测试屋。

实际入口：

|文件|用途|
|---|---|
|`build_data.py`|因果输入与监督分离；未知不变成负标签|
|`extract_features.py` / `launch_features.py`|真实 best4k 前向、处理后输入哈希、全参数首尾指纹|
|`objective.py`|原 V14 记忆与损失；变长前缀、填充掩码、真正的跨步梯度|
|`train.py` / `launch_training.py`|1 族和 3 族两档、三种子、B1/B2/Ours，共 18 个固定模型|
|`continuation_service.py`|真实 Habitat 历史回放、动作执行和只供评测使用的 SEE2 真值|
|`evaluate_continuations.py` / `launch_continuations.py`|同一常驻 best4k、216 次固定自主续接、输入/动作前缀和运行身份核验|
|`review.py`|完整分母、实际动作与原始轨迹重算、UNKNOWN 和未完成项单列|
|`diagnose_heads.py`|只读检查保存的模型、原生 STOP 保护和 B2 精确状态读出|
|`diagnose_events.py`|逐因果时刻定位 B2 事件误报、识别与保持问题；不更新参数|
|`runtime_r2/`|保留首组后复用场景、续跑剩余 198 次的实际编排修复|
|`audit_continuation_content.py` / `test_review.py`|原始数组与像素计数复核、完整组续跑和状态封存验收|

模型只能读取原指令、最近两张 RGB、最近八个实际执行动作及自己形成的有限记忆。训练查询只进入独立结果读出器。部署动作前向不读取查询、任务检查器状态、语义实例或位置。每个任务重新计算自己的因果记忆。

本 pilot 的训练查询读出使用四个已审核的后缀上下文字段（是否在当前点 STOP、是否 STOP、后缀终点前是否出现 anchor、终点是否满足 terminal）；不是已经实现了通用语言续接编码器。精确状态 B2 可以组合这些字段求出同一结果，见 `QUERY_STATE_SUFFICIENCY.json`。

自主评测先实际执行登记历史，再由策略自行行动；原历史和 STOP 都计入 500 次决策。重复运行同一测试条件时，经原始输入核验后共享本次常驻模型产生的因果前缀特征；自主决策仍执行真实 Qwen 前向。没有完整 KV 或历史文本输入动作读出。

这里的任务是“连续两帧看到指定物体—房间原语，再连续两帧看到终点物体并主动 STOP”。它不是“到访房间”，也不是 R2R 的距离成功率。沿用的工厂检查器对碰撞轨迹返回 UNKNOWN；因此报告 PASS、FAIL、UNKNOWN 和完整分母下的 PASS 下界，不删除碰撞轨迹来提高得分。

本轮执行使用项目内 Qwen 与 Habitat 环境。启动器核验共享 GPU 余量和 UUID，并只清理本轮核实的进程组。主要命令（从仓库根执行）为：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/launch_training.py
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/launch_continuations.py
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/runtime_r2/launch_continuations.py
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/review.py
```

既有 run 只读保留，启动器创建新的编号目录。未完成的检查不能当作完整结果。`CONTINUATION_PROTOCOL_INITIAL.json` 保留查看模型分数前的清单；后续基础设施修订记录源锁变化，未修改清单、策略或成功定义。

上面的 launcher 命令记录本轮执行入口，不是让复核者再次训练或重跑已完成条件。无 GPU 复核从 `review.py` 和 `test_review.py` 开始；完整日志打包在 `EVIDENCE_LOGS.tar.gz`，清单与逐文件 SHA 在 `EVIDENCE_MANIFEST.json`。

`DATA_INFORMATION_AUDIT.json` 明确记录标签偏置和有限续接的局限：始终预测 PASS 已有 77.8% 准确率；所有任务都有通用于各历史的已测成功续接，条件化教师分叉只说明已测路径集合中的效率机会。辅助准确率、教师动作准确率、自主任务成功和科学泛化分别判断。

`RAW_DATA_EVIDENCE.html` 使用原始数组无损转为 PNG 展示可见证据和相同终端窗口；没有生成或修饰传感器像素。原始数组、完整轨迹、特征缓存和权重保存在本地运行目录，不能仅凭代码仓库宣称已复现运行。

## 独立于 Codex 的运行入口（2026-09-20）

后续长任务用 `standalone.py start <唯一任务名> -- <绝对解释器路径> <脚本及参数>` 提交。入口将完整流水线交给 systemd 系统服务，使用调用者相同的 UID/GID，无 sudo、无权限提升。服务拥有独立 cgroup、stdin、日志和退出记录，不依赖终端、Codex API、对话额度或 assistant 轮询。现机器已实测提交进程退出后后台命令完成，见 `STANDALONE_CPU_TEST_RESULT.json`。

本轮 GPU 已完成，实际后台复核入口为 `finish_v15.py`，按顺序执行日志重算、内容核验、消耗汇总、报告和证据打包；不会重启导航或训练。查看命令：

```bash
systemctl status q35n-v15-finalize-20260920.service
tail -f projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/standalone_jobs/v15-finalize-20260920/job.log
cat projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/standalone_jobs/v15-finalize-20260920/STATUS.json
```

提交新任务的形式（替换唯一任务名及实际脚本；不要重跑本轮已完成模型）：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/standalone.py start TASK_NAME -- /ABS/PROJECT/PYTHON -I -B /ABS/PROJECT/PIPELINE.py
```

`JOB.json` 固定命令、工作目录及 UID/GID，`STATUS.json` 原子记录 RUNNING/COMPLETE/FAILED/INTERRUPTED 和退出码，`job.log` 保存输出。同名任务拒绝重复提交；失败不会自动换模型、调参或无限重跑。GPU 资源监控仍由真实 launcher 执行。完整流水线应自行执行已确定阶段，不能把“下一步由 Codex 发送命令”作为运行依赖。

需要手动停止时仅操作该任务明确的 `q35n-<任务名>.service`；systemd 只清理该服务 cgroup。CPU 测试仅证明独立执行，不是导航或方法通过。当前为 transient service，可跨 Codex/终端退出，但不承诺跨机器断电、重启、OOM 或外部管理员停止；这些情况用既有完整组/模型检查点另行恢复，不能把半组接成配对结果。
