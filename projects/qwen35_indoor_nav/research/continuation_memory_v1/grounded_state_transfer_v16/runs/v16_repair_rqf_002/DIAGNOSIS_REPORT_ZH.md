# Q35N V16 rqf v2 离线诊断报告

## 结论与边界

本 run 只执行了 `v16_rqf_isolation_diagnosis_v2` 的封存输入、CPU 证据重算和正确性验收。新增物理候选执行数为 **0**；没有启动 Habitat、GPU、Qwen、训练、导航评测、G1 或自动晋级。允许的结论仍是：**修复候选仍未找到两个物理合法族。**这不是训练失败、模型失败或导航收益结论。

输入截止时间：`2026-09-20T10:12:46Z`。v1 失败证据按文件哈希封存；活跃目录未 chmod、移动、改名或加锁。旧 run、旧源码锁和失败结果未覆盖。

## 复用与缺口

- `v16_formal_001` 逐 HOUSE 校验得到 `24` 个完整族（批准口径为 24），按值复用，不重新采集。
- `rqfALeAoiTq`/TEST 仍缺 2 个族；v1 新修复 run 的新增接受数为 0。
- v1 的一个 `FAMILY.json` 被识别为 `PARTIAL_FAMILY`：缺 `content_root`，不可接纳；这与成功路径重复排他发布的静态缺陷相符，但不把它倒推为所有物理拒绝的原因。

## 初始中性视角（问题 A）

审计键为 `(hub, anchor_A, anchor_B, yaw)`，每个已保存探测逐帧记录 role、实例和相邻帧像素。逐 yaw 的计数为：`{"INVALID_YAW": 318, "VALID_NEUTRAL_START": 787}`。proposal 汇总严格区分某些 yaw 不合法、八个 yaw 均不合法和证据缺失；没有把未保存的证据写成“八个均不合法”，也没有外推为整屋所有朝向均不合法。

## 完整历史隔离（问题 B）

逐条读取已保存 `H_*_DISCOVERY.json`，继续调用原 `Compiler.atoms()`：同一 eligible 实例在相邻两帧各至少 256 像素才构成 SEE2，不拼接尝试、不跨实例合并、不以房间替代 role。重算计数：`{"ISOLATED_FOR_SAVED_TRACE": 1047, "OWN_ANCHOR_NOT_SEEN": 123, "PARTIAL_FAMILY_NOT_ACCEPTED": 1, "RECOMPUTED_OTHER_ANCHOR_PRESENT": 591}`。

旧 trace 没有可靠的阶段边界字段，因此首次污染阶段按要求记为 `UNRESOLVED`，没有按路线外观猜测“去程/返程/公共尾段”。证据缺失、未到达和程序错误均与物理拒绝分开。

## 修复与 CPU 验收

`GENERATOR_PATCH_PLAN.json` 只提出局部修补：完整组装后一次发布、异常分流、独立诊断入口；本次不执行物理验证搜索。`CPU_TEST_RESULT.json` 记录一次性发布、半成品恢复拒绝、SEE2 同实例阈值、旧文件不变和禁止模型阶段的测试结果：`PASS`。

## 下一步门槛

只有主 agent 审核本诊断和有限验证方案后，才可另行授权物理验证。候选顺序、候选分母和预算在 `GENERATOR_PATCH_PLAN.json` 中明确；本 run 正常终止状态为 `DIAGNOSIS_ONLY_COMPLETE`，不自动开搜。
