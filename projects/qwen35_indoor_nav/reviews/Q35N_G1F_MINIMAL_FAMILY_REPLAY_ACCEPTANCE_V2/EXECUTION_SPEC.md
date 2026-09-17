# G1F 执行规格：V2 的运行具体化

主 agent 依据用户“做”批准本节点，授权见 ../../authorizations/G1F_EXECUTION_AUTHORIZATION_V1.json。
继承 P2R1 MINIMAL_FAMILY_SPEC_V2、FAMILY_REPLAY_GATE_DRAFT、DATA_SCHEMA_V2 和主 agent 索引澄清。原草案不覆盖、不改 executable。
固定 runtime 为 G0R_DEPENDENCY_RECOVERY_V1 的成功构建；使用同一 GPU 2，不安装、不修改环境、不终止其他进程。

执行顺序：资产/运行指纹核对 → eligible 集合与有界 witness 预览 → 稳定候选 discovery → 首个合格候选冻结 → 27 次独立物理重放、54 次任务检查 → validator 和裁决。任一前置必要条件失败，不将下游未运行部分写成通过。

以下只是补全原规格未指定的执行细节，在任何新渲染前固定，不改变任务、阈值和动作预算：

- episode_id 按整数排序，start_position 作为 path_index=-1，reference_path 按原下标；坐标首次出现去重，取前32点，再 snap。yaw=0..23，正值为绕世界 y 左转15°。全部保留资产溯源，不能送入策略。
- witness 环以 OBB.center 的水平中心，半径固定0.75/1.25m、角0..7×45°生成，保留原中心高度后 snap 到 navmesh；这是未成轨迹的候选 viewpoint 初始化，不是给真实历史 teleport。对每个对象最多16个候选，按实例/类别/原始位置排序，记录重复/不可达/证据失败。初始帧+L+R 是观察预览；SEE2 必须来自实际相邻动作帧、同实例每帧>=256像素。
- 路径由固定 Habitat greedy follower（goal radius=0.1875m，fix_thrashing=true）生成，移除规划停止符，再用最短15°转向朝目标OBB中心并执行 L/R 两帧观察。单段规划动作超过原上限即拒绝。真实 witness 必须在实际落点重验，不把预览作为轨迹证书。
- 逆原语 F→L^12 F R^12，L→R，R→L；只压缩连续旋转，保留所有平移。记录未压缩/压缩动作及理想SE(2) transform；压缩前后均须实际可重放才可冻结。
- family candidate 外层顺序为 u点、yaw、8种tail。各候选内按已排序witness列表依次寻找第一个通过实际路线事件/预算的D、K、L、B组合；子搜索次数单列，不充当family数量。路径缓存仅去重同参数的确定性尝试，失败仍追溯。
- H_D_L 为 H_D 后再执行从u出发的TV闭环。H_D/H_K/H_D_L 以中性回环补成等长；尝试顺序 LR、RL、L^24、R^24，仅采用不会产生任务事件的循环。每个族固定之前都完整检查三条历史和8步公共尾部、三个续接。
- O_0 是有效首帧，SEE2 的时间域从t=1开始；不因缺少不存在的O_-1把整条合法轨迹标为U。STOP 由runner执行终止，不调用未知Habitat动作，也不产生新观测。续接第一动作的decision index等于cutoff。
- 严格沿用 RGB/semantic 精确哈希匹配与1e-4m/rad物理阈值。不将数值近似汇合的pose强行round/teleport成相同。若因此候选全部失败，只判本构造规则未通过，不判主算法不可能。
- 当前场景是旧暴露interface_only；所有预览/重放/失败记录在本节点内。发现阶段保存逐帧hash、像素计数和动作，必要图像按内容去重；正式族须保存完整可复算证据。
- 时间与资源达到上限即停止；首个frozen candidate之后禁止换候选。无候选则没有Y训练集、没有G2准入。

本节点是数据工程验收，不是模型训练、导航收益、泛化或投稿竞争力实验。
