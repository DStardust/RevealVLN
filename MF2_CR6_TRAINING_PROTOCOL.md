# MF2-CR6 训练协议 v1

**冻结日期：** 2026-08-26  
**范围：** RxR-train 上的多分支工程训练准备；不修改 `FROZEN_SPEC.md`，不产生
val_unseen/test 结果。

## 1. 在线输入

每个低层因果前缀只使用截至当前的 63° 前向观测、完整指令、已稳定建立的 ECOG
候选以及剩余预算。冻结 ETP-R1 产生三类 768 维表示：指令均值、当前因果视觉均值、
每条持久分支最近一次可验证的候选 token。未建立分支使用 mask，不注入未来 token。

## 2. 监督

- `target_index`：严格 Reveal 前为 `-1`，禁止强制提前提交；Reveal 后才监督目标；
- `target_in_set / separation / evidence_complete / reveal_hazard`：分别监督集合、完整
  竞争分离、指令证据闭合和首次闭合时刻；
- `option_cost / current_feasibility`：来自每分支、每预算的双控制器可复现 `T_X`；
- `checkpoint_value`：某预算下至少一条分支只有经保存节点返回才安全可行的比例；
- 候选顺序无语义，模型对完整集合打分，在线仅激活动态 Top-2。

## 3. 划分与授权

场景按 `SHA256("mf2-cr6-scene-split/1:" + scene_id)` 固定排序，分配为 36 个
train、8 个 development、其余 Gold，场景不得跨集合。旧 300 条单人 pairwise
审核只作 development 诊断，不是 v2 Gold。

自动标签、成本和冻结特征可以在新鲜人工抽审前生成；正式训练入口必须同时看到：

1. 完整集合语言门通过且在线未来帧数为 0；
2. 每分支 `T_X` 两个独立进程证据哈希一致；
3. 冻结特征 manifest 全量哈希与场景隔离通过；
4. 新鲜 full-set 人工抽审通过并由独立 finalizer 写入授权字段。

缺任一项，训练器 fail closed。现阶段 synthetic smoke 只能证明软件链路可执行，不能
作为论文指标。

## 4. 首轮训练与消融

首轮只训练因果头，ETP-R1 全冻结。主方法固定 Top-2；在相同候选前端、标签、预算和
训练步数下比较 Top-1、Top-2、Top-3、full-set、pairwise-v1，以及去掉 `T_X`、去掉
checkpoint value、去掉 Reveal hazard。开发集只用于选择一次预登记 checkpoint；Gold
和公开验证 benchmark 不参与阈值选择。
