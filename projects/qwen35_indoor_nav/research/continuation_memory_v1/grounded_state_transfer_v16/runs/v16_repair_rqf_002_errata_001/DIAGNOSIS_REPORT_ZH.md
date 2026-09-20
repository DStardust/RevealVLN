# Q35N V16 rqf v2.1 离线勘误与 CPU 补验报告

## 结论

本轮完成的范围是离线字节封存、语义勘误、分母重算、CPU 行为验收和下一节点报审材料编制。终态为 `OFFLINE_ERRATA_COMPLETE_PENDING_REVIEW`；这不是物理采集、训练、模型失败或方法收益结论。

- 精确复用原字节的条目数：3485
- revision 上下文条目数：8
- 原字节不可恢复的条目数：0
- CPU 测试实际执行/通过/失败/未执行：56/56/0/0
- terminal-only 真实回归：PASS
- 发布与恢复修复：PASS
- 原 24 族按值复用：VERIFIED；引用范围为声明 JSON，原始数组未重验
- 新增物理执行：0
- 新增接纳族：0
- 目标屋缺失族：2（未因本轮补齐）

## 勘误边界

中性门槛现在只由 `anchor_A`/`anchor_B` 触发；terminal-only 保留诊断但不拒绝。每个 proposal 固定建立八个 yaw 槽，缺证据、未到达、证据无效和程序错误都会令聚合成为 `INCOMPLETE_EVIDENCE`，不会成为“八个均无效”。历史 own/other anchor 分开统计，部分 FAMILY 通知不进入历史分母。

FAMILY 的未来提交路径要求四历史、十二后缀、引用哈希、精确 replay 前缀、STOP、预算、碰撞和标签矩阵全部通过；发布采用同目录临时文件、fsync、排他硬链接和目录 fsync。旧 24 族未重采、未重认证。

## 下一节点

待审批 manifest 状态 `READY_FOR_REVIEW`，包含 4 个具体 trial。当前 driver 没有物理、训练、G1、Qwen 或自动推进命令。只有新的明确审核裁决才能执行另一个独立节点。
