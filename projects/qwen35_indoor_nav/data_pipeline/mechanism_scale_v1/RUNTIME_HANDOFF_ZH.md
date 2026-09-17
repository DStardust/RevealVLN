# 机制批量生产 CPU 修订交付

当前仅 CPU 候选准备，不启动仿真、GPU 或训练。所有历史 P0、factory_v2、runtime_v1 封存不变。

## 已实现的修订

1. 根据项目固定 Habitat v0.1.7 的 `Mp3dSemanticScene.cpp` 解析 MP3D ASCII 1.1 的 R/C/O 字段。对旧 P0 五屋全部已有 runtime 对象的 raw/category/room 做逐对象差分。
2. 改为在每个 FIT 房屋实际存在的严格 category+room+raw 组合上枚举四个角色，而不是固定 TV/living room、sink/kitchen、bed/bedroom、chair/dining room。类别/raw 白名单在 planner.py 冻结，不用别名悄悄改旧实验。
3. A/B 交换不再重复计候选；两个 anchor 与 terminal 使用不同房间类别，四角色要求共同楼层候选，anchor 中心最小距离 1.5m。这里只是元数据预筛：没有证明可见、独立触发、可达或可精确回返。
4. 每屋按确定性 SHA256 顺序保留最多 32 组合，跨房屋 round-robin；使用普通线同一 SPLIT_FREEZE，不读内部 dev/confirm 房屋资产，不读官方 val/test。
5. 产量必须分开：semantic candidate、physical replay certified、mechanism train-ready。前两个不能冒充已完成特殊训练数据。

## 最小 runtime 接入修改

旧 worker 将 cfg.roles / cfg.tasks 作为全批固定常量，需要在新目录的新 worker 改为 row.roles / row.tasks，并在新 backend 实际 inventory 与 row.expected_eligible 完全一致后启动该 job。旧 worker、封存结果不要 monkey patch。

新 worker 可只读复用 `mechanism_runtime_v1.habitat_backend.HabitatBackend`、`p0_driver_v1.family_job.run_bundle`、factory_v2 Compiler/BudgetLedger/FreezeLedger 和 V4 export/loader。必须用独立 import 命名/明确 sys.path，不改变旧代码。

下一次小批建议：从同一 manifest 顺序取前 5 个不同 FIT 房屋，每屋先 1 个角色组合；登记完整资产 SHA 和至多 2 条 R2R train 不同物理路线，每路线至多 8 个原坐标/24朝向合法几何配置。未完成的组合记 unattempted，而不是失败。实际预算由主 agent 统一普通生产资源后冻结，配置保持 executable=false，不能照此文字自动运行。

## 不能直接大规模复用的剩余问题

旧 P0 74 探测轨迹有 58 次碰撞截断，剩余候选 metadata 修订并不能修复离散几何动作的回返失败。第一 runtime 节点应先在一个已登记角色候选验证：几何提案的真实去程、可观察两帧见证、合法回返、公共尾段与 1e-5 数值汇合界限。发生碰撞立即记录，不能放大汇合容差或用 teleport 补造历史。

如去程离散跟踪持续碰撞，应在新版实际状态反馈 follower 中修订，而不是枚举更多失败开环路径；该变更属于数据构造工程，需单独核验动作尺度和确定性，不是新论文模块。navmesh 路径不是执行记录。即使更换 follower，保留 TraceRunner 的逐动作预记账、真实碰撞/观测和所有失败。

首次新 runtime 的工程优先顺序固定为：① 一个候选的反馈式合法去程，逐步保存实际动作/碰撞并得到两帧可观察见证；② 根据真实执行轨迹构造并重放逆路，验证确实回到共同状态，不从几何计划直接反转冒充可执行逆路；③ 加公共尾部，再真实完成 3×3×2 与三种种子认证；④ 导出/回读并按质量 tier 验收，反捷径不过仍只进入接口/诊断池；⑤ 才按既定 manifest 跨房屋扩大。

存储效率作为并行工程检查：检查 ContentStore 每帧全目录扫描的实际时间占比，若采用增量字节计数，必须对断点恢复、同 hash 去重、新写入失败和最终全量盘点做一致性测试。不能为了吞吐关闭证据写入、fsync/账本或资源上限。1,376 个候选不说明上述生产瓶颈已解决。

旧已认证单族存在动作累计计数捷径：18格、标签翻转和无关绕路都通过还不够。应优先构造真实等动作长度的反标签历史，或在跨族数据中事前冻结长度交叉平衡，然后用内部开发分房屋的 length-only/current-view-only/query-only 对照检查。不能虚构空白动作 padding 或裁掉关键历史。强 M2 程序状态标签应完整保留，供同数据对照。

语义像素只核验实例可观察；房间元数据与学生能理解该房间不是同一个事实。语言使用严格模板，不用未经核验 LLM 润色。近义措辞增加指令记录，不增加独立数据族。

正式批量特殊数据必须逐层提升质量 tier；未达到最高 tier 的产物保留为诊断/接口素材。论文要求的自然指令迁移和模型独立收益仍需后续实验，不能从元数据覆盖率推出。
