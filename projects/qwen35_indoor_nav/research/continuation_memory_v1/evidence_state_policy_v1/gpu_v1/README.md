# 证据状态策略：GPU 开发实验

冻结 Qwen best4k 特征，训练 DIRECT / MONOTONIC / REVISE 三种执行状态策略，各三个种子、1200 次更新。九个轻量模型全部从同 seed 的共同初始化训练，不复用旧 B2Fix。底模在训练期只提供已审计缓存；真实续接期重新加载并逐步前向。当前没有方法收益结论。

本轮沿用 CPU `../runs/cpu_003` 的真实数据、损失、模型和 schedule；不修改旧封存。DIRECT 直接预测四位状态，MONOTONIC 累积事件，REVISE 学习修订历史信念；三者共享事件/状态监督与动作读出。对比只能归因于这些模式的差异，不能归因于本轮共享新增模块。活跃参数/梯度路线不同，在逐步日志中保留。

训练入口 `train.py` 调用已做真实反向与恢复验收的父目录训练器。每200步保存优化器、RNG及绑定；最终固定1200。评测128条件×九模型=1152续接，每个完整条件组同进程比较；每臂历史任务192、无关控制192。完整九模型组才准入聚合，不拼接半组。固定500决策，包括真实历史和STOP，不额外生成STOP观测。

数据仍为已暴露的FIT四屋/32父族、DEV一屋/8父族，不是普通VLN或独立泛化测试。只读 `calibration.py` 在封存自主轨迹上计算固定十箱ECE与Brier；不修改动作或挑选阈值，也没有调用Jev API。

## 当前任务

- run：`runs/gpu_001`
- 训练/评测服务：`q35n-evidence-state-gpu-20260921-02.service`
- 监控：服务器 `http://127.0.0.1:18770/`，仍用原来的端口转发。
- 七卡1–7。GPU0有其他真实工作，未操作。按已核实的占位租约借用GPU2–7，成功/失败退出时恢复。
- 第一次服务提交使用了错误的相对入口路径，在启动流水线前失败；`standalone_jobs/evidence-state-gpu-20260921-01` 原始失败保留。02使用绝对路径启动同一尚未执行的run。

```bash
systemctl status q35n-evidence-state-gpu-20260921-02.service
```

服务自己依次完成 prepare → train → diagnose → evaluate_continuations → review → calibration → publish。关闭终端或Codex不影响执行；硬件/正确性错误会停止并留账，不按分数重试。进度与分数来自真实文件，训练进度不包含复用模型。

恢复仅在服务已经退出且原因排除后执行，使用新服务名、同一run和`--resume`；不修改冻结源码、配置或数据，不覆盖完整组：

```bash
ROOT="$(git rev-parse --show-toplevel)"
G="$ROOT/projects/qwen35_indoor_nav/research/continuation_memory_v1/evidence_state_policy_v1/gpu_v1"
PY=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
V16_STANDALONE_PYTHON="$PY" "$PY" -I -S -B "$G/standalone.py" start evidence-state-gpu-resume-UNIQUE -- \
  "$PY" -I -B "$G/pipeline.py" --config "$G/PROTOCOL.json" --run-id gpu_001 --resume
```

将UNIQUE替换为小写日期/编号。监控服务配置中的作业路径也应更新为实际恢复服务，不把旧服务状态当新状态。

## 验收与证据

`CPU_TEST_RESULT.json` 记录GPU接线CPU测试及修复前失败；父目录CPU003另有8项模型、梯度和恢复测试。`monitor_v1/HTTP_TEST_SNAPSHOT.json` 与 `RENDER_TEST.js` 验证实际HTTP状态及页面渲染。模型训练与真实GPU导航结果分别记录，CPU通过不替代方法收益。

每个模型：`train/TAG/PROGRESS.json`、`attempt_*/STEPS.jsonl`、可恢复checkpoint、`FINAL.pt`、`RESULT.json`。每个GPU：`attempts/`、`evaluate/session_gpu*/`。完整组包含原始输入/处理后输入指纹、native/method/executed动作、状态/事件概率、私有真实轨迹和状态封存。最终 `RESULT.json` 保留全部计划分母及配对未知界。

GitHub仅发布代码、报告、可复核文本日志包及轻量最终权重。底模、授权场景、原始RGB/语义、特征缓存、环境与优化器二进制保留本机，路径和SHA登记。共享发布锁避免与旧任务提交混合；上传失败会明确记录 `COMPLETE_UPLOAD_FAILED`。
