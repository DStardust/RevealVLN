# B2 KL 冲突位置可控修复

本目录是独立的新实验，旧 B2 对照与诊断资产只读。

唯一差异：B2Fix 的特殊任务保持损失在 `action_mask && native_argmax != teacher_target && preservation_mask` 的位置取消分子贡献。**仍除以原 preservation_mask 总和**，其余位置权重不变；普通动作CE、四位状态辅助、动作/接管CE、初始化、1200步schedule、数据和推理全部不变。这是训练期已注册教师不一致，不等于证明原生动作非法。12个FIT状态存在多个合法教师选择，旧标签保持原样。

仅训练 B2Fix×1209/1210/1211，3600次新更新；原B1/B2六个final1200权重按SHA复用。三个新模型并行训练，八卡评测。每个条件九模型同进程重跑；128条件、1152次续接，每臂主任务192/控制192。禁止以旧分数替代本轮基线。模型、原始/处理后输入前缀与STOP/500步审计继承已运行路径。

主要比较 B2Fix−B2，同时看B1、控制任务、成本、种子和接管分层。正差只代表已暴露DEV开发信号，不自动采用，不承诺修复视觉/语义泛化。

启动：以项目独立Python调用 `standalone.py start <新服务名> -- <独立Python> -I -S -B pipeline.py --config PROTOCOL.json --run-id repair_001`（所有脚本参数使用绝对路径）。失败恢复用新服务名与同一run-id，附加`--resume`。每200步保存完整优化器/RNG，评测只承认完整九模型封存组，不重跑完整低分组。

进度：`runs/repair_001/STATUS.json`、`TRAIN_PROGRESS_*.json`、`EVALUATION_PROGRESS_*.json`及原18770监控端口。独立systemd流水线不依赖Codex在线；完成后CPU重算并由publish.py自动推送证据。

GPU2–7只通过现有占位租约借用并恢复，不影响真实外部任务。三个模型约束与24 GPU会话小时上限沿用注册资源，现场竞争单独记录。

GitHub入口：`projects/qwen35_indoor_nav/reviews/Q35N_B2_KL_REPAIR_20260921/README_ZH.md`。发布包含源码、协议、测量报告、日志包和轻量最终权重；底模、原始场景/数组、特征缓存及优化器二进制留在服务器，不能把仅有公开仓库当作完整复现环境。
