# QUERY_LENGTH：只读复算与最小候选修订

实际来源为 `revisit_v1/run_v1` 已结束的 trace 000006、000007，使用未改 Compiler、cutoff=248 复算：

| 续接 | 动作数 | query 项数 | 分解 |
|---|---:|---:|---|
| C0 | 14 | 28 | 13 movement + 14 observe + 1 STOP |
| C_A | 130 | 210 | 94 movement + 115 observe + 1 STOP |

C_A 的事件包括 77 步 irrelevant、37 步 anchor_A、1 步 terminal。第一次 anchor_A see2 在续接第 17 步。配置 a 回环 116 动作，terminal 13 动作；动作长度合法不等于 query 长度合法。

原因：原 query_from_trace 每个可见事件时间步都追加 observe；这些项也会打断相邻 movement 合并。不能只按动作上限 160 预测 query 项数。

不改封存 checker、不放宽 160 的最小建议：**只缩短 C_A/C_B 续接，保留已通过的三历史和精确计数结构**。从共同 hub 沿已录短前缀抵达第一次真实 A see2（当前第 17 步），再真实反馈导航直接去 terminal；不强制走完 116 动作 A 回环回到 hub 后再走 terminal。B 分支同理。

这是新几何候选，不是已验证轨迹。必须真实重放、保留 query 跨历史一致与 18 格标签、用原 compiler 计算实际 query<=160，再 27 重放/54 求值。不能裁 query、删除重复 observe、忽略 irrelevant 或凭几何路径推断成功。
