# 主 agent：真实机制族补齐与并行分工收口

2026-09-09，用户要求“SFT验收交由其余codex会话进行，真实机制数据族你这边补齐”。

## 当前完成

已接收[真实族V3](reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1/REPORT_ZH.md)与[落盘复核](reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1/READBACK_AUDIT.json)：1个TV/sink→bed任务族，18个实际cross cell（12 pass/6 fail）、27次完整物理重放、54任务求值、1,242条因果prefix记录、474个RGB/443个semantic数组。运行与CPU复核均通过；GPU3已恢复占位，未停止真实任务。SFT使用[独立交接指令](HANDOFF_SFT_ACCEPTANCE_V1.md)，本会话没有模型训练。

其中4组是同任务/同query、不同历史、相反实际标签；这完成数据能表达所需历史机制的接口验收，不证明模型已经利用它。

## 必须保留的边界

- V3 task实例将TV/sink设为任务anchor，餐椅为无关绕行。原餐椅任务的失败未被改写为PASS，见下方历史节点。
- 原始返回u有微米量级浮点差，逐像素哈希不同。本族明确采用一次<=1e-5m/rad共同状态重建；实际最大位置修正1.66893e-6m，边界四类事件集合全部未变。规范化后u/s跨27回放pose spread=0、RGB/semantic哈希完全相同。该过程是声明的离线反事实数据干预，不是未经干预的自然精确汇合。
- 场景是已暴露MP3D 17DRP5sb8fy，split=interface_only。27/54是重放一致性检查，不是独立样本；1个族不能代表正式训练规模、泛化或科学收益。
- 不改变主线V3的学习算法/创新主张；目前仍需普通SFT、同数据M1/M2/M4匹配训练及独立导航实验。科学PASS、贡献充分性、近期竞争力均未成立。

## 已保留的生成过程

1. [覆盖修订](reviews/Q35N_G1R_COVERAGE_V1/REPORT_ZH.md)：扩大u覆盖；因过强TV绕行约束提前停止，不假称256耗尽。
2. [短转向历史](reviews/Q35N_G1R_ROTATION_HISTORY_V1/REPORT_ZH.md)：256base没有完整family。
3. [绕行约束修正](reviews/Q35N_G1R_EVENT_CONSTRAINT_CORRECTION_V1/REPORT_ZH.md)：事件构造可行，原始像素汇合失败。
4. [数值共同状态](reviews/Q35N_G1R_NUMERICAL_JOIN_V1/REPORT_ZH.md)：共同尾部可构造，但greedy C0新增餐椅事件。
5. [事件约束续接](reviews/Q35N_G1R_EVENT_AWARE_SUFFIX_V1/REPORT_ZH.md)：12000展开/36000分支动作未找到C0；不证明数学不可能。
6. [V3任务实例](reviews/Q35N_G1R_TASK_INSTANCE_V3/REPORT_ZH.md)：首个完整候选冻结；之后独立验收通过，没有在冻结失败后改选。

上述都是数据构造/工程调试，不是反复查看模型测试集挑正号；所有失败账本、资源、GPU恢复和哈希留存。

## 下一步

先接收外部SFT验收；数据侧下一节点应冻结多族采样与家屋/模板分组协议，报告构造尝试数、失败类型与yield，再分批扩充，不把18格复制成“大数据”。使用真实query验证完整M4 reader和长历史训练接口应另开节点，不能冒称G2合成两步探针已经覆盖它。既有SFT协议不自动混入本族。
