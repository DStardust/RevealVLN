# 内核探测 v1 结果报告

日期：2026-09-10。范围：AUTHORIZATION.json first_gpu_gate（一张 drain 卡、≤1200s、≤28GiB、≤512 forward 决策、≤8 丢弃更新、0 正式训练、0 导航 episode）。实际：98 forward 决策、2 次诊断更新（已丢弃）、GPU3 精确借还。

## 结论

1. **官方 FLA 0.5.2 算子本身数值合格**：单元级（unit_kernel_check.py，生产形状 B=1/H=HV=16/K=V=128，T=320 与 T=805）forward 最大绝对差 ≤7.4e-4，五个输入梯度 relative L2 ≤0.0067、cosine ≥0.99993。见 run_unit_v1/UNIT_KERNEL_CHECK.json。
2. **端到端探测按预注册门槛 FAIL**（run_v1/COMPARISON.json，未放宽任何阈值）：logits 最大 Δ2.51、loss 相对差 7.8%、更新相对 L2 1.95。分歧模式：每个 chunk 的第 0 决策 Δ≈0.05，随后沿 chunk 递增（0.05→0.38→0.90→1.33）。定位为 **v6 自定义记忆递归对每层微小算子差异的逐步放大**（记忆写回→作为下一步输入嵌入→再写回），不是算子错误。
3. **性能证据（batch=1、短记录、非 10× 门槛）**：fallback 1.55 → FLA 6.27 decisions/s，内核单项 4.05×；显存峰值 26.7GB → 14.7GB。
4. **集成机制已证实**：transformers 5.15 `use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule","fla")` 在 fla 可导入时绑定官方函数；注意 `fla/__init__` 不暴露 `.ops`，必须先显式 `import fla.ops.gated_delta_rule` 再首次导入 modeling 模块（候选进程闭包断言 `implementation.__module__ == fla.ops.gated_delta_rule.chunk` 已通过）。

## 对后续的影响（如实登记）

- v6（含记忆递归）换核在注册门槛下不通过，**v6 继续用原 fallback 路径训练，不换核**。本 FAIL 保留，门槛不回改。
- v3 普通基座无记忆模块（结构修订，RECIPE_DECISION.md）：不存在递归放大通道。v3 是否采用 FLA，由 v3 自己的冻结协议做**单前向级**数值验收（门槛在运行前登记，与本 FAIL 无关、不冲突）。
- 10× 目标（≥18.224896976 decisions/s）未达成；当前证据：内核单项 4.05×，叠加真实 batch 后另测。不伪报。

## 失败与事故记录

- 4 次早期探测尝试分别因配置键名/接口 kwarg/CUBLAS 环境变量/旧 policy 512 token 断言失败（run_failed_v1..v4），占位每次都完整恢复。
- unit_lease 首次因复用 run_v1 输出目录导致 restore_dead_holder 的排他写冲突，占位未自动恢复；已按原 argv/cwd 手动 respawn 并核验身份/显存/remain-on-exit，事故与修复见 MANUAL_RECOVERY_UNIT_LEASE_20260910.json。unit lease 已改用独立 run_unit_v1 目录。
- 全部借还：GPU3 占位 6 次恢复均核验 PID/argv/cwd/uid/显存再占用，external_processes_stopped=0。
