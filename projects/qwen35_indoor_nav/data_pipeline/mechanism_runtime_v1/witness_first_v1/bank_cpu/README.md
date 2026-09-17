# 跨房屋 Witness Bank CPU API

`bank.propose(records, max_programs=64, require_closed_anchors=True, min_irrelevant_forward=0)`。

这是复用封存 Compiler.atoms 与有限角色词表的纯 CPU 算法，没有加载新资源、读取实时轨迹或运行 GPU。测试中的轨迹是显式人工构造的单元测试 fixture，不是仿真数据或科学结果。

每个 records 元素：

```python
{
    'id': 'globally_unique_trace_id',
    'trace_ref': 'caller_owned_source_reference',  # 可选，默认 id
    'house_id': 'house',
    'scene_fingerprint': 'fixed_asset_and_scene_configuration_identity',
    'split': 'FIT',
    'trace': actual_complete_trace,  # actions/observations/complete/collisions
    'closed_loop': True,            # caller 指示已实际闭合的完整 loop
    'roles': [
        {'signature': ['chair', 'living room', 'chair'], 'eligible_ids': [12, 13]},
        # 全房屋完整有限词表角色 catalog，而不是只挑当前看见的实例
    ],
}
```

trace.observations 必须满足真实 compiler 契约，且包含 step/pixels/evidence_complete/pose/rgb_hash/semantic_hash。pose 包含 agent 及 rgb/semantic 的 position、rotation。不同 house、scene_fingerprint、初始 agent/sensor pose 或初始 RGB/semantic hash 不混成同 hub。相同房屋场景的角色 catalog 必须一致。函数不从邻近坐标猜测同一 hub，也不把不同实例的像素拼接成连续见证。

角色 signature 为 `[mpcat40, room, normalized_exact_raw]`，使用已有 planner 的有限词表。不要求 A/B/T 三房间不同，但四角色 `(category,room)` 组合互异；排除仅 raw 不同而 query 语义相同。eligible_ids 必须由 caller 按真实 metadata 完整映射；本函数不访问场景或猜测实体标签。

算法步骤：

1. 严格验证完整轨迹，按上述完整 hub 标识分组；使用原 Compiler.atoms 的同实例、连续两帧、每帧≥256像素规则。
2. 找实际 A-only 与 B-only 互斥见证。默认需要完整闭合 loop；caller 的 closed_loop 标记还会通过保存的 agent/sensor 末端闭合阈值复核，不能仅凭布尔值放行。
3. 查找同 hub 已执行的中性 LRLRLRLR 公共尾，不出现 A/B/T；找不出现 A/B 的 terminal-only 因果前缀。I 不出现 B/T，优先选择至少含2个F的闭合 loop。
4. 给出原始动作列表、trace refs/hash、所需/禁止角色及事件步。组件齐全时提供 `histories_unbalanced`、`continuations_unreplayed` 和动作计数；C0=terminal前缀+STOP，C_A/C_B=各闭合anchor loop+C0。这些只是待回放的拼接方案。
5. 返回确定性排序的有界数量提案。无移动 I 时如实标记；`min_irrelevant_forward=2` 可强制排除纯转向 I。未提供完整 I 回路时保留缺口，不编造逆路径。`require_closed_anchors=False` 可作更早期前缀发现，但不会输出齐全闭合组件的动作拼接。

必须在主 agent 后续独立运行中完成：同动作计数控制/中性padding、H_A_I真实拼接、共同状态与短窗口、合法continuation/STOP、18格交叉矩阵、重复seed及捷径审核。数值闭合不等于拼接像素一致或后续物理合法；CPU 输出始终 `physical_replay_certified=false`、`training_admission=false`。

固定排序优先组件齐全、I含至少2F、I动作不同A/B、未平衡动作长度在上限内，再按总动作长度与稳定ID排序。数据发现的优先级不是效能结果筛选；后续仍需记录所有提案、失败和资源截断。`max_programs` 限制输出数量，不代替 renderer 的资源预算。
