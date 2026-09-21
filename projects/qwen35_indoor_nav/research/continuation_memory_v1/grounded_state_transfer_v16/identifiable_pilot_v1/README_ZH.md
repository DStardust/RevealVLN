# 已核验历史样本：三臂独立运行

本目录测试 32 个 FIT 族、8 个已暴露 DEV 族上的历史条件停止。数据来自此前真实执行和原始数组核验；本轮不重新生成或修改标签。保持 best4k Qwen、8×64 记忆和短窗不变。B1/B2/Ours 共享数据、动作教师、初始权重和每 seed 的 1200 步 schedule；仅辅助项分别为无、精确状态、交叉结果。新增 cutoff CE 三臂共用，不能算作 Ours 的独立贡献。

后台服务：`q35n-ident-pilot-20260921-01.service`。确认运行于 system.slice，父进程为 PID 1，stdin 关闭。退出聊天不会取消任务；断电、OOM 或管理员停止仍可能中断。

## 进度

```bash
systemctl status q35n-ident-pilot-20260921-01.service --no-pager
```

本目录下：

- `runs/pilot_001/STATUS.json`：当前阶段或具体停止原因。
- `runs/pilot_001/FEATURE_PROGRESS.json`：5850 个真实 Qwen 前向的进度。
- `runs/pilot_001/TRAIN_PROGRESS.json`：当前模型/step，九模型共10800更新。
- `runs/pilot_001/EVALUATION_PROGRESS.json`：576 个续接槽位的进度；每九模型完整组封存。
- `runs/pilot_001/attempts/*/stdout.log`：各阶段完整输出。
- `runs/pilot_001/RESOURCES.jsonl`：所有阶段的会话墙钟；不是 CUDA 活跃时间。
- `standalone_jobs/ident-pilot-20260921-01/STATUS.json`：服务完成或失败。

流水线会自行推进特征、训练、诊断、真实自主续接、CPU 复核。每模型每200步有模型/优化器/RNG断点，最终使用固定1200步。正常会话分段会自动进入下一段；故障保留原 attempt 后停止。恢复须以新服务名、同一 run-id 和 `--resume` 启动，源码/数据/配置锁必须保持一致，不能覆盖失败证据。

## 结果边界

主历史任务每臂96次，task_T历史无关对照每臂96次，共576次；两个终点分开，只有一个 DEV 屋。停止预算包含全部历史和 STOP，没有原生 STOP 硬覆盖，没有查询/语义/位姿输入 actor。输入/数值审计沿用 V16，缓存与现场差异单列。

最终 `RESULT.json`、`ROLLOUTS.json` 和 `REPORT_ZH.md` 报完整分母及未知界。`TEACHER_DIAGNOSTICS.json` 和 `MEMORY_INTERVENTIONS.json` 是强制真实历史后的动作诊断，不能替代闭环。`ACTION_ONLY_LOOKUP.json` 是仅 FIT 拟合的全动作模式查表；其低分不能排除更强动作序列捷径。

CPU契约5项通过，真实掩码/合成特征的三臂反向冒烟通过。这些不是模型收益。当前站立转向 SEE2 任务的正向信号，也不等于正常 VLN-CE、跨屋泛化或可发表贡献。没有按 DEV 选种子、checkpoint、训练步数或重新尝试低分条件。
