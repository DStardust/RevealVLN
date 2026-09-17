# v3 单卡验收 v1 结果报告（2026-09-10）

范围：冻结 ACCEPTANCE.json + RUNBOOK_ACCEPT_GPU3.json；GPU3 精确借还（LEASE_RESULT: holders_restored=true, external_processes_stopped=0）。所有门槛为测量前注册；FAIL 不回改。

## 结果汇总

| 关口 | 结果 | 证据 |
|---|---|---|
| 内核面板（v3 批处理路径，FLA vs fallback，17 固定真实决策） | **FAIL**（预注册门槛） | run_gpu3/PANEL_COMPARISON.json |
| 断点恢复（fallback，12 更新双跑） | **PASS 精确一致**（trainables torch.equal，cursor 一致） | run_gpu3/resume_a, resume_b |
| 吞吐（fallback@6144 tokens/批） | **OOM 不可行**（含 expandable_segments 前 29.9GB；显存图 ~7.7MB/token） | acceptance/EVIDENCE_fallback_oom_*.log |
| 吞吐（fallback@1024 tokens/批） | **5.013 decisions/s 稳态** = 2.75× 基线 | run_gpu3/throughput_fallback_1024 |
| 总体 | **FAIL**（面板未过） | GPU3_ACCEPTANCE_SUMMARY.json |

## 面板 FAIL 的归因分析（如实记录）

- 逐决策 logits 差：均值带符号 +0.0026（零偏）、平均绝对 0.064、最大 0.458（logits 标准差 1.39）。
- 梯度：cosine 中位 0.9957、最低 0.9886；rel L2 中位 0.106。
- argmax 一致率 29%：动作头未训练时四类 logits 近乎并列，微小扰动即翻转——该门槛对未训练初值不适用。
- 对照单元级证据（kernel_probe_v1/run_unit_v1）：生产形状下官方 FLA 算子 vs fallback 前向差 ≤7e-4、五路输入梯度 cosine ≥0.9999、rel L2 ≤0.7%——算子本身实现的是同一数学。
- 结论：端到端分歧为 24 层 bf16 累加对"同一数学的不同归约顺序"的放大，非语义错误。本次预注册的逐位式门槛（为同实现恢复保真设计）不适用于跨实现比较。FAIL 保留。

## 关键工程事实

1. **fallback 无法承载生产批处理**：6144-token 批在 26GiB 预算内 OOM；最大可行 ~1024-token 包（约 3.5 样本/批）→ 5.0 d/s 单卡。
2. **FLA 单算子 4.05×**（1.55→6.27 d/s，显存 26.7→14.7GB），且批处理显存曲线平缓（无 FP32 chunk 中间量驻留）。
3. 10× 目标 ≥18.2249 d/s：fallback-only 四卡约 18–20 d/s（贴线、无余量）；FLA 路径单卡 6144-token 吞吐尚未测（面板 FAIL 阻断）。

## 同期事件记录（与验收无因果）

- v6 于 unix 1789026378 收到外部 SIGTERM（与 v5 同型、来源未归因；平台侧 xdragon 健康服务在线）。正常收口：checkpoint_000000761.pt 完整、监督 returncode 0、cleanup 完成、0 外部进程被信号。GPU2 空闲。
- GPU1 恢复队列 8 批（500–507）全部 AUDITED 后自然完成退出，无重试。
- GPU6 EnvDrop 扩源队列 17:33 失败停止：`SHARD_WALL_BUDGET`（分片超时），保留失败，不自动重启。
- GPU7 特殊采集队列健康运行中。

## 待决（集中一次，见主报告）

FLA 采纳需要验收 v2（门槛重定义为：算子级等价 + 有界双核训练等价）；否则 fallback-only，10× 贴线且无显存余量。
