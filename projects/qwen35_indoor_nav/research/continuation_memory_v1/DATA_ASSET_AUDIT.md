# 实际资产与缺项

本轮判定：**DATA_ASSET_GAP**，不是“完全没有数据”，也不是数据已满足研究准入。
可复查机器报告：`../../reviews/Q35N_CODEX_RETURN_20260917/DATA_ASSET_AUDIT.json`、
`ASSET_INVENTORY.json`；记录本机原始文件SHA与CPU重算范围，无新仿真。

## 实际候选与接口

选取现成目录：
`data_pipeline/mechanism_runtime_v1/witness_first_v1/short_revisit_v3/run_v1/bundles/WF_SHORT_REVISIT_V3_004/`。
其 `export_v4/MANIFEST.json` SHA 为
`7df77c63813665d3319256e7c7807b579723eac6cd66d189b1bfb3e1ec7eb271`。

|对象|本轮实际核查|
|---|---|
|来源|已暴露FIT房屋17DRP5sb8fy；source记录physical_certified=true，training_admission=false|
|组合|3历史×2任务×3续接=18格；9条canonical轨迹，不是18条独立轨迹|
|既有证书|27次重放、54次求值；整个旧运行另记42条轨迹、11352动作，不能混作本轮生成量|
|CPU重算|18标签：12 PASS、6 FAIL；492个去重动作监督owner；424个内容数组文件与raw像素哈希校验|
|历史|每条248个运动动作，249个观察；动作计数L116/F16/R116匹配|
|续接|C0=14，C_A=50，C_B=50动作，包含STOP|
|短窗|导出最近2RGB＋8实际动作只有1种共同窗口；并非全前缀相同|
|可调用接口|Compiler.evaluate/m2/encode_query；FamilyLoader.prefix_records/policy_payload/supervision/action_stream|

本轮确实调用 `FamilyLoader.validate_supervision_contract()`，并从内容文件解码真实
2×150528字节RGB，经新 `schema.from_existing_v4` 投影到策略白名单。
没有用文件路径、实例ID、语义像素或监督字段代替RGB输入。

## 不能忽略的差异

1. 程序是`observable_see2_then_stop.v4`：两帧看见床/椅子，随后两帧看见电视并STOP；
   不是进入餐厅／卧室。V5 `ordered_visit_v5_cpu/visit.py:evaluate` 有距离区域与可见性
   约束，但只是既有CPU原型，没有把本族升级为真实到访证书。
2. H_A_I是“已完成子目标重访的位置改变”，不是无事件的无关空间绕行。两类控制不能混称。
3. 本族两任务都历史敏感，缺附件g2那样的历史无关目标任务负控制。不能擅自把旧轨迹重标为
   新程序通过；需先审核新程序与语言、语义见证和动作池。
4. 9条canonical轨迹各有一次`bounded_numerical_join.v1`，记录在step240。原始与规范化RGB
   哈希不同，位置修正量记录约1e-6m。最后2帧在247/248，不能把整个历史说成未经规范化的
   原始逐位一致；未来共享状态契约必须明确是否接受这种数值处理和其证据，保留raw/canonical。
5. 最早关键事件在H_A/H_A_I的17步、H_B的14步，监督在248。若梯度需覆盖首次形成任务
   状态的两帧事件，分别涉及233/236个观察的展开；不能仅凭最近一次见证11/19/177步前就
   宣称短展开足够。全249观察展开的显存和吞吐未知。

## 库存范围与独立性

`witness_first_v1`下找到215份V4导出清单、11个声明房屋、211个候选内容hash，累计3870格；
全部manifest的training_admission为false。这是元数据盘点，不是215个独立物理族认证。
只对上述一个族做了本轮完整内容/标签回读；其余未重做质量证书或重放。候选hash也不能
替代物理hub/路线/历史/续接/语言的关联分组。不能说整个项目只有一个族，也不能把11屋当
新独立测试屋。新实验family split与完整同池动作曝光manifest尚缺。

下一数据关需要：与冻结任务语义对应的完整族；无关绕行与任务负控制；转移状态/预算/随机性
证书；观察证据与共享短窗；跨屋、路线及语言全族去重和暴露账；全臂共享动作轨迹表。
本轮不自动补造数、不改变旧quality结果、不把接口或元数据数量升级为训练准入。
