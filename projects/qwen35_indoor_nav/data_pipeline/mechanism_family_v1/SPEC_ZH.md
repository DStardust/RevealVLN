# 真实机制族编译和验收 V1

本包不进行新候选搜索，不加载模型，不训练。只接受已经冻结且通过完整预检的真实候选。当前正在运行的 coverage 节点未产生候选时，本包不得导出训练标签。

- 保留 V2 任务、事件、传感器、动作、预算、精确 RGB/semantic 汇合标准。索引沿用已批准 G1F 澄清：动作 a_t 的 decision step=t，首个 continuation target step=cutoff，因此 X03 使用 >=cutoff，不能把未来观测加入 prefix。
- 先保存候选哈希、源代码哈希、资产/配置/运行指纹；冻结后只检查此候选，不回退到其他候选。
- 3 seeds (1109/2209/3309)，每个 seed 9 条完整 history+continuation 真实轨迹，每条用两个任务从头求值，共 27 physical / 54 evaluations；不称 54 独立样本。
- 所有原始 RGB/semantic 按像素内容哈希保存；逐帧原始计数重新计算。独立核对三条历史在 u、s 的 poses/sensors/hashes、公共尾部最近两帧与八动作、等长、无最近 task event、零碰撞、预算、实际续接 query、跨 seed 冷状态重放一致性。
- 验收程序不引用预期标签生成 Y，预期矩阵仅在真实求值完成后比对。损坏/缺证据/非法轨迹不得转成可靠负样本。
- 先通过当前 DATA_SCHEMA_V2 使用的全部 schema vocabulary（自包含 fail-closed 子集验证器，不冒充通用 Draft 2020-12 实现），再逐项 X01–X13 跨记录验收和阴性破坏测试。
- policy 仅含语言、截至 t 的最近 RGB 和已经执行动作；整段 memory 前缀必须逐步重放，不能只读末两帧。监督侧未来 query/semantic/pose/ID/Y 独立保存，不进入 policy。
- 接口族旧场景暴露，split=interface_only、independent statistical samples=0；该节点通过不代表泛化、训练收益或论文贡献成立。
- 首次实际 certification 的资源上限为 1 小时、8 GiB 输出、16 GiB RAM、8 GiB GPU，仅 GPU3，在 coverage worker 退出并恢复占位后独立借用并恢复。

本文件先于实际 certification 冻结。后续具体构造规则如有版本修订，必须引用其独立规格，不能由本包暗改。
