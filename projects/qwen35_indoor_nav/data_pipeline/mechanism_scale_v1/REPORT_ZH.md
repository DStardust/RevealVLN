# 特殊机制数据扩量：CPU 语义构造准备 V1

结论：**可用于新 runtime 批次的候选池已建立，但尚无新增真实合格机制族。**

与普通生产使用同一冻结分房屋划分：51 FIT / 5 INTERNAL_DEV / 5 INTERNAL_CONFIRM。实际仅读取 51 FIT 房屋的 `.house`，未读取内部留出房屋的场景内容或官方 val/test。原 5 FIT_PILOT 不变，旧暴露不被清零。共享划分 SHA256：`c292a652f2f08aaddabb3f0df51c2a78d6c60638710444d1163d8e742499d379`。

## 实际交付数量

| 指标 | 数量 |
|---|---:|
| CPU 审计 FIT 房屋 | 51 |
| 存在声明四角色组合的房屋 | 43 |
| 确定性有界语义候选 | 1,376（每适用屋 32） |
| 元数据不适用房屋 | 8，全部留账 |
| 新真实回放族 | 0 |
| 新机制训练合格族 | 0 |
| CPU 单元测试 | 17 项通过 |

51 屋中 43 屋的比例仅为此次词汇/规则的元数据覆盖率，不是生成率、可见率或科学可行通过率。1,376 个候选之间仍可能共享大量实际轨迹，不能称为 1,376 个独立数据族。

## 为什么不再盲重复旧 P0

旧 P0 4/5 被固定 TV/living room + sink/kitchen + bed/bedroom + chair/dining room 组合拒绝。新版本保持原任务检查器的两帧实例可见语义，改为在各房屋真实存在的 category/room/raw 精确匹配集合上事前枚举组合。

旧五屋均获得新角色组合；这是新任务候选，不是将旧任务失败改为成功。旧 0/5、2393 动作、74 探测轨迹、58 碰撞截断完整引用在 [旧失败登记](OLD_FAILURE_REFERENCE.json)。新预筛不会修复开环几何路径碰撞、回返精度或动作计数捷径。

CPU parser 的 R/C/O 字段依据项目固定 Habitat v0.1.7 源码读取；与五屋已存 runtime inventory 逐对象检查 raw、mpcat40、room 一致。仅在监督侧保存原 house 坐标用于不变距离预筛；不把它当 Habitat 可执行坐标，更不进入策略。

## 质量与主线

[质量分层](QUALITY_TIERS.json)保留 3 历史 × 3 续接 × 2 任务的 18 格、27 次重放、可观察证据、标签翻转、无关绕路不变、语言依赖、公共最近窗口、因果导出及强程序状态监督。动作长度、当前画面和 query-only 捷径未排除的物理族不得升级为 `MECHANISM_TRAIN_READY`。

当前 `SEMANTIC_CANDIDATE` 一律 training_admission=false，未运行项为 null；资源截断和缺证据不自动变成负标签。将来产量按独立族、轨迹、指令、前缀和矩阵格分别统计。

论文主线冻结 V3 未修改：普通数据训练通用导航，特殊数据通过交叉续接约束同一个运行期执行记忆。没有新增模型、文献轮询、GPU、仿真、下载或训练。

## 主 agent 接入入口

- [候选清单](acceptance_v1/candidates.json)：每 job 的 roles/tasks/expected_eligible。
- [分屋覆盖](acceptance_v1/coverage.json)、[资产读取账本](acceptance_v1/source_reads.json)、[失败账本](acceptance_v1/failure_ledger.json)。
- [最小 runtime 修订与残余风险](RUNTIME_HANDOFF_ZH.md)：新 worker 必须采用每候选角色，不能继续全批 cfg.roles；先解决实际路径与回返问题后再量产。
- `planner.py` 是无仿真的候选生产器；输出要求新目录，避免覆盖旧结果。
- `audit.py` 进行全候选语义/分房屋/哈希回读并封存。CPU 数据检查不构成科学或导航收益。
