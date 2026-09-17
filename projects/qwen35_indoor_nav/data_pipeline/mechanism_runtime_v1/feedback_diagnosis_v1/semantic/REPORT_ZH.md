# 特殊数据 CPU 失败审计

只读取旧日志/trace/config，不运行 navmesh、仿真或 GPU；旧结果保持原样。
日志链及 HEAD 通过：85897 条。141 条 trace 均通过 journal SHA256 核验。

| 房屋 | 配置 | 空/非空 target 调用 | 保存 trace/动作 | 未封存尾动作 | 终态 |
|---|---:|---:|---:|---:|---|
| 17DRP5sb8fy | 16 | 0/4 | 57/7396 | 23 | RESOURCE_CENSORED |
| 1LXtFkjw3qL | 16 | 16/0 | 16/0 | 0 | REJECTED |
| 1pXnuDYAj8r | 16 | 4/10 | 68/6748 | 38 | RESOURCE_CENSORED |

## 可证结论与界限

- 1LXtFkjw3qL：16 配置全部在 anchor_A 的 target 生成处返回空；全部 NO_VALID_LOOP(anchor_A)，其他 role 未尝试。只有 16 个零动作初始 trace。证实的是当前生成器无候选，不是场景没有可达目标。
- 该屋选择 4 个起点，同一 y=0.084411；chair 对象中心 y 约 1.05，metadata level=1。对象中心高度不等于地面高度，level 编号也不等于 y 坐标；没有证据确证楼层错配。代码要求起点 snap 偏移不超过 1e-5，实际 reset 偏差已逐 trace 统计；没有保存起点筛选时逐点 snap 结果。
- 3 个选定起点的全部原始（未 snap）anchor_A 环采样点都超过 8 m；另一个有小于 8 m 的环点。距离上限是具体可疑限制，但未记录 snap/find_path/距离超限计数，因此不能把全部空候选归因到单一条件，更不能宣称调整距离已经有效。
- 17DRP5sb8fy 与 1pXnuDYAj8r 有非空目标和完整动作 trace，不能归因于统一“走不动”；显式拒绝集中在 anchor_B 的 OUTBOUND_EVENT_OR_LEGALITY / NO_VALID_LOOP。原 reason 合并了事件与合法性，需要下一版本记录细分谓词。
- journal 完成动作共 14205，保存 trace 动作共 14144；相差 61。差额逐屋与最后一个 trace 后的动作尾一致，是资源截断后的未封存轨迹，不可作为合格训练动作。141/141 complete 是“已保存 trace”的条件统计，不代表全部实际尝试都完整。
- 下一版本至少增加：target 筛选原因计数、起点/目标 snap 偏移与 floor/geodesic 证据、role-specific 拒绝谓词、未完成 trace 持久化；本审计没有实现或授权新生产。
