# Witness bank 到跨房屋计数匹配批次

CPU准备器和可供主agent调用的 `BalancedFactory`，不主动启动GPU或仿真。

## 通用组合与实际验收

对于源bank已实际执行、完整闭合的 a/b/i：

- H_A = a + i + a × (k−1)
- H_A_I = a × k + i
- H_B = b × j

枚举 k=2…16、j=1…16，每proposal完整保留240个计数尝试的通过/拒绝原因。先要求F次数及L−R相同，再仅追加LR对补到最大L，所有F/L/R计数精确相同，三序列必须不同，公共尾部之前最多504动作。

H_A同样包含i；这里只是已完成子目标重访或无关闭环**放置顺序**的控制，不声称H_A没有I，也不保证i包含平移。a、b、i、terminal原动作不做最短旋转、角度取模或其他压缩。实际重复和LR补齐仍可能引入新事件，故必须重放。

`BalancedFactory` 接口：

```python
factory = BalancedFactory(
    backend, compiler, budget, emit, "task_A", "task_B",
    context={"house_id": row["house_id"], "asset_config": row["assets"]},
    components=row["components"], balance=row["balance"],
)
candidate = factory.construct(row["configuration"])
# 主worker仍须按冻结协议执行三种子重放、冻结账本和V4导出/回读。
```

histories首先实际验证a含A不B、b含B不A、i含I不B/T及原1e−5 agent/sensor闭合；随后实际检查三整条历史及相同计数。C0=t+STOP，C_A=a+t+STOP，C_B=b+t+STOP。继承原 `construct`、`validate_matrix`、`replay_seeds`，18格标签、公共tail、query160及三种子全部不改。

## 安全快照

仅读取 `scout_v1/run_v1/houses/<house>/result.json` 已标记完整关闭的房屋，再读其完整BANK_RECORDS；未关闭house一律deferred，不读正在append的bank。逐文件stat前后稳定检查、哈希及源trace自证由bank验证，完工后再次检查全部来源哈希。

每已完成hub重新调用原 `bank.propose(max_programs=1000)`，不是只用原前32。完整保留这1000上限内的proposal catalog、每个计数尝试、拒绝原因和截断数；截断之外不声称已穷举。每hub按最小历史动作数、最大续接动作数、proposal_id排序，首选一个。两hub共享同一house，不能当成两独立房屋或证明场景泛化。

`source_observation_query_estimates` 来自已存组件观测的CPU拼接，仅用于提前观察长度风险，绝不输出为新训练轨迹或替代真实完整查询核验。物理端点微差、拼接边界和事件路径变化均可能导致实际结果不同。

快照配置与scout的资产及其完整语义词汇registry一致；按bank当次冻结role任务精确编译expected_eligible，不强行复用scout最初仅用于资产许可的四角色。它们均在FIT，所有原文/源码/trace哈希入SOURCE_LOCK。

## 增量使用

从项目根运行现成项目stdlib Python：

```text
python -I -S -B .../batch_plan_v1/prepare.py --output .../batch_plan_v1/snapshot_v2
```

必须指定不存在的新snapshot目录，旧snapshot不覆盖。新house完成后可以新快照追加准备；配置始终executable=false，GPU和资源预算null，交主agent登记新runtime。候选数不等于物理族，更不等于合格训练数据。旧失败和主线冻结均不改。
