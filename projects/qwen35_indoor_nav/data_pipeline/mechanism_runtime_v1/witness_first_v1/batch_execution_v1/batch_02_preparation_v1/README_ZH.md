# Batch02 事先固定来源、合法词面与CPU构造验收

仅本目录新增。输入固定为已封存 `winding_balance_v1/next_source_v1/snapshot_v2` 的索引 `[1,2,3]`，顺序保持：2n8kARJN3HM hub1、5LpN3gDmAk7 hub0、29hnd4uzFmX hub0。未读取batch00或batch01r1的导航结果/trace/Journal选择候选；只读它们不可变EXECUTION_CONFIG核对既有hub位置。既有失败、中断和未完成结果不替换、不覆盖。

对新三候选与既有两个batch共6候选逐对核验（18比较），另核验新候选彼此3对：相同房屋必须欧氏距离>=1m，不跨屋比较无意义坐标。29hnd新hub距batch01r1已选hub1为1.3528726489722782m；另外两屋在既有两批中未出现。不同hub不自动代表独立房屋或模型泛化。

所有词面修订发生在新不可执行source composite内，使用封存 language_realization_v1：familyroom/lounge写family room/lounge，其他合法词面保持。每候选只修改task instruction，再加显式 `_B02_LR1` 新ID、language_version与before/after provenance。roles、eligible IDs、动作、平衡系数、padding、原anchor/terminal谓词全部断言不变。旧template仍保留在LANGUAGE_REVISIONS.json中，不隐藏修订来源。

CPU构造器使用原BalancedFactory、Compiler及禁止任何backend访问的NoRuntime对象，核对原模板与新模板task_program一致、三history逐F/L/R计数相等、padding精确一致、history和continuation上限不变。不会执行histories()或replay_seeds()，因此不证明spin中性、观察汇合、18cell、27replay或导航收益。

`composite/CONFIG_DRAFT.json` + `SOURCE_LOCK.json` 可供主agent的现有batch prepare快照接口使用，选择此composite本地索引 `[0,1,2]`（原来源索引仍是[1,2,3]）。此交付不创建batch_02运行目录、worker或launcher，不准入GPU。注意现有通用prepare只接受GPU1/2；GPU5精确占位借用需主agent独立运行运输版本/授权，不能直接把gpu5塞进旧函数或修改活跃共享源码。

所有候选executable=false，resource_budget/gpu_device未设。主agent审核后独立冻结资源、确认GPU5占位身份并finally恢复；本CPU节点不操作GPU。查询长度是已存组件重组的明确source观察估计，非新物理replay结果，实际查询仍要重新验收。
