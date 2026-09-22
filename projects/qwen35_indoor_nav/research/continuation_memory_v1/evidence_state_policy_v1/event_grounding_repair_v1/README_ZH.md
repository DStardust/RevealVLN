# 事件识别损失修复 V1

状态与进度以 runs/event_001/STATUS.json、EVENT_PREFLIGHT.json 和 RESULT.json 为准。实现和 CPU 测试不代表正向模型结果。

本轮保留 ORIGINAL 的完整动作教师、MONOTONIC 8×64 记忆、因果输入、状态监督、KL、普通动作监督、推理与 STOP 规则。只将事件 BCE 从重复轨迹和全局类别权重，改为输入/角色去重后按房屋/角色等权，单元内保留实际正负比例。使用此前经真实 SEE2 证据核验的 8271 个 FIT 输入/角色标签，无新造数或伪造负例。

先做四次留一 FIT 屋的事件读出对照，ORIGINAL/EVENT 共八次，每次 400 更新、种子 1209。事件模块从原始初始化开始，归一化及其余模块冻结；两臂使用相同全部训练输入，差别仅是损失质量分配。类别统计只来自三个训练屋。ORIGINAL 是原家庭 schedule/class-weighted BCE 的精确期望，不是端到端原策略。单种子和重叠训练集使该诊断不能视为独立确认。

固定筛选：屋和角色等权 Brier 下降、宏平均 recall 降幅不超过 0.05、至少两个屋 Brier 下降。失败则封存/上传并停止，不搜参数。通过则从共同原初始化开始，ORIGINAL/EVENT × 三种子各 1200 步，再完成原暴露 DEV 的 128 个六模型组、768 次自主续接。主任务每臂 192 条，task_T 每臂 192 条。筛选通过不等于闭环收益或自动采用。

EVENT 完整策略训练每步通过独立可重放 RNG 抽取 256 个事件标签；ORIGINAL 的原 family 内事件曝光保留。更新数相同，辅助实际曝光和计算量另报。所有动作及 state/KL loss 都由原函数计算，未来查询、房屋 ID、真值不进入 actor。

8 个纯 CPU 测试验证实际损失等价、标签/屋隔离、去重、只替换事件损失、真实梯度及优化器/RNG 恢复。不将这些测试计作导航或方法效果。完整研究最多 10400 次轻量更新，另有有界 CPU 接口检查；底模更新 0。本轮 GPU 会话累计上限 12 小时，单 worker 3 小时，产物 40 GiB。只借用已核实的用户占位，成功/失败后自动恢复。

独立启动（先在本目录，STD 为项目自带 Python）：

```bash
STD=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
export V16_STANDALONE_PYTHON="$STD"
"$STD" -I -S -B standalone.py start event-repair-20260922-01 -- "$STD" -I -B "$PWD/pipeline.py" --config "$PWD/PROTOCOL.json" --run-id event_001 --resume
```

流水线不依赖聊天 API 或终端；相同 run-id 恢复前须核验源锁和输入指纹，完整模型/条件组不重跑，基础设施故障不按成绩重试。监控沿用本机 127.0.0.1:18770，/api/status 提供事件筛选、正式更新和完整封存组计数。完成后自动上传本版本代码、日志、最终轻量权重；原始场景/图片、特征缓存和优化器文件保留本地。

已关闭的 teacher_alignment、STOP 修复、旧模型和旧分母保持只读。此项是监督与校准的工程验证，不宣称通用 VLN、新架构或论文贡献。正负结果均按预注册回交。
