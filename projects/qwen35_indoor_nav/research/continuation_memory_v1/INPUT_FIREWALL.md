# 输入防火墙

|区域|字段与用途|进入策略|
|---|---|---|
|policy_input|原指令、224×224原始RGB最近1/2帧、最近≤8实际运动动作|是|
|policy_memory|仅由该任务和已到达的因果观察递推的8槽连续状态|是，未来模型尚未实现|
|train_query|合法续接动作与核验事件的有序语义；绑定词汇表|仅独立训练读出器R|
|supervision|PASS/FAIL/UNKNOWN、动作target、精确组合式状态、有效mask|仅loss|
|audit_only|episode/lane/house/family/route/语言族ID、pose、实例ID、程序、源SHA、预算和状态证书|否|

运行入口仅接收 `PolicyObservation` 与显式memory；传入query、y、pose或ID即报错。
文件定位和hash只用于CPU加载校验；`from_existing_v4` 调用原loader读取RGB，再删除
task_type、memory_reset等运输字段，只返回instruction/rgb/executed。reset、step、lane
由外层拥有者管理，不能成为token、embedding索引或可学习查表键。

query必须先经旧 `Compiler.semantic_query` 精确投影，再交 `continuation_query` 校验；
不允许整条SUPERVISION记录传入。旧query的action_trace_ref/RGB refs是审核信息，不编码。
旧typed query的category/room整数依赖各compiler词表；必须绑定vocabulary并转成词义一致
的统一token或描述后再编码，不能把局部整数当全局语义。未来查询中不放Y、已完成结论、
检查器状态或历史是否满足任务的答案。

先执行 `m = encode_prefix(I,h)` 与动作前向，再执行 `R(m,I,q)`。
替换q/Y只允许改变读出/损失与反向梯度，不得改变已计算的m/action；训练梯度经过m合法，
未来信息作为m/action的前向输入不合法。同一I/h供多q重用同一m计算图；换I必须重算整个
任务条件前缀。缓存键只能作外层正确性约束，不能进入模型。

episode或lane reset重新初始化记忆；每次新观察只更新一次。未执行的native动作不得写回。
正式运行不得隐藏完整KV、全历史token、额外摘要、程序解析器或教师旁路。原短窗直连动作
可以保留，但必须所有研究臂一致且计入成本。

CPU当前已测类型/字段拒绝、实际loader投影、query词表绑定、因果精确状态、全族split
与UNKNOWN。尚未测试真实Qwen缓存、张量因果不变性或autograd；不能把CPU类型契约当成
已证明模型无泄漏。未来验收必须做query替换、任务替换、lane交错、reset和参数/激活审计。
