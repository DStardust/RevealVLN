# 缺少伤害样本的有界修复

本轮从已完成的 `intervention_runs/calibrate_001` 继续，旧源码、日志、权重只读。旧实验 200 条 TRAIN 路线没有伤害例，三个选择器实际更新均为零；旧 `gate_steps=2000` 是配置值，不是完成证据。

本轮新增 20 个 TRAIN 房屋、400 条去重路线，每条真实执行正常起点与八个共同转向动作前缀两种条件。左右方向预先平衡，不看方法分数选路。所有动作、原生 STOP 和恢复共用原有 500 步；不传送，不伪造观察，不把转向本身当作选择器干预。新采集共 800 四臂组/3200 次完整导航；同路线的所有变体留在同一 split，16 屋 FIT、4 屋 DEV。旧 200 组只读复用且房屋分割独立。没有保证该扩充一定产生伤害或 unseen 提升。

每条真实分歧的标签描述“选择整个后续原生/方法策略”的任务成功差，而不是局部动作好坏。训练保持线性选择器、固定 2000 更新与固定 `p(gain)>2*p(harm)`；使用 FIT 类别平衡 CE，不用旧 EU6 动作锚点。缺少真实 gain/harm 的臂明确跳过，实际更新记 0。DEV 只诊断，不选阈值/最好 checkpoint。所有已完成低分条件保留；无按得分重试。

可训练的选择器随后自动接入真实推理，在首次候选动作分歧处做一次决定：整段余程沿用原生或对应冻结方法。每一步都重新执行真实模型/环境。固定官方 unseen 200 条无转向扰动，与本轮原生和不加门控的同一方法同组比较；不以离线挑分支冒充这次测试。未训练 gate 的预期槽位另记 NOT_RUN，不挑剩余好模型报全体成功。

此 200 条已暴露，且排除了旧记忆 FIT 屋 EU6（官方 val_unseen）。旧头仍含 EU6 训练历史，因此不是完整 1839 条的干净论文主模型。新的 `evidence_arch_v1` 架构独立，本轮不声称训练或验证了它。当前目标是修通真实监督和 unseen 闭环，不把线性门控本身称为新论文贡献。

## 独立执行与进度

配置：`runs/recovery_001/PROTOCOL.json`。启动通过同 UID/GID 的上层 `standalone.py`；运行器独立完成采集、训练、真实 unseen 和报告。最多 8 张用户授权卡、64 GPU 会话小时/12 墙钟小时/32 GiB 新产物；额度不足或服务错误保存已封存组并退出，只清理自身 Popen 进程组。每个会话最多 25 个完整组；首波每卡 2 组验证共同前缀，计入正式分母。断点不拼接半组，重新核验源码/数据/权重。资源占位通过用户自己的控制 socket 借用并恢复。

运行器命令（由独立 service 调用）：

```bash
python -I -B pipeline.py --run-id recovery_001
# 基础设施中断后，新 service 名运行同一 run-id：
python -I -B pipeline.py --run-id recovery_001 --resume
```

同一监控站点 `http://127.0.0.1:18770/` 顶部新卡与 `/api/intervention_v2` 显示阶段、封存组、真实训练步数和分母。终态 `COMPLETE / UNSEEN_REVIEW_COMPLETE` 才表示本轮真实 unseen 评测完成；`DATA_LIMITED` 表示没有可训练门控，`FAILED/INTERRUPTED` 表示异常或预算退出。结果正负与任务完成分开。

关键产物：`train/RESULT.json`、`gate_data/COUNTS.json`、`GATE_REVIEW.json`、`unseen/RESULT.json`、根 `RESULT.json/REPORT_ZH.md`，每组 TRACE/FEATURES/COMPLETE、每会话 RUNTIME_IDENTITY/STATE_SEAL 和每次资源账。CPU 测试 `test_recovery.py` 检查转向计数、干预筛选、一次性选择、真实更新及实际 split 隔离。
