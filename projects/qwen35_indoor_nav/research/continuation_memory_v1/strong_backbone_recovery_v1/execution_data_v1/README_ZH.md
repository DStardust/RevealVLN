# 模型能力改进的数据入口

2026-09-26。本轮只执行CPU数据整理与模型实现检查，室外按用户要求仅备案。正在运行的369路线评测代码、参数和条件未修改。

## 已完成的真实数据工作

|产物|实际内容|尚未完成|
|---|---|---|
|token_index/|334条已执行轨迹，28,859实际动作，28,525动作后的真实观测关系|没有新增物理轨迹|
|recapture_plan_001/|7,211原query的补采集请求，覆盖21,396缺失actor位置，其中133为终止STOP|尚未生成这些actor特征|
|branch_requests_001/|从完整800条官方TRAIN记录中固定40个真实前缀，登记160个候选动作分支|160个替代分支均NOT_RUN；新增结果标签为0|

索引核验了原POOL及334条物理轨迹的SHA。可用actor特征目前只有7,463个，且只有原known mask中的4,842个query具有已准入动作监督。21,396个缺位中13,533个位于原已监督轨迹区间，其余是未准入的恢复前缀；不能把缺位统一升级成训练标签。

133个缺失STOP来自成功保留轨迹：FIT102、DEV31。整个四动作chunk在生成时只能读取query_start的观察和记忆；之后执行得到的观察不能提前回填给该chunk的actor。补特征仍不等于部署修复：现有残差只作用于首动作，全部token残差需新公共执行版本。

新的分支清单覆盖16个FIT屋32个前缀、4个DEV屋8个前缀；每屋一个自然历史、一个真实转向扰动历史，使用不同路线。选择只用固定哈希，不看成功率。每个前缀登记STOP/前进/左/右，固定同一个原生续接策略。实际采集必须从起点重放，无中途传送，四分支共享预算与策略；当前没有实现/启动分支运行器，不把请求数量当数据量。

这些关系可支持学习“动作后实际发生了什么”和“同一位置怎样选更好”。它们本身既不是新算法，也不能证明长历史必需；未来与同架构、同数据的直接动作代价学习比较。

## 可复算命令

从本目录运行，输出必须是新目录：

```bash
PY=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
"$PY" -I -S -B test_token_index.py
"$PY" -I -S -B test_recapture_plan.py
"$PY" -I -S -B test_branches.py
"$PY" -I -S -B token_index.py --output token_index_new
"$PY" -I -S -B recapture_plan.py --index token_index_new --output recapture_plan_new
"$PY" -I -S -B build_branches.py --output branch_requests_new
```

19项CPU边界测试通过，见CPU_TEST_RESULT.json。无仿真、无新VLM前向、无GPU、无新导航SR。新架构及可恢复训练代码见[模型改进目录](../parallel_improvements_v1/README_ZH.md)。
