# 通用真实机制构造核心 V2

状态：`GENERIC_FACTORY_CPU_CORE_PASS`，80项CPU测试通过。实现而非仅协议草案；但**不是已可生产的完整Habitat管线**。真实backend、资源watchdog与V4训练文件导出尚未实现/验收，generate/certify入口明确拒绝运行。本轮零GPU、零新物理族，不干预外部SFT。

## 已完成

| 文件 | 实际功能 |
|---|---|
| [compiler.py](compiler.py) | 实例化角色/任务/实例资格；像素重算、unknown、严格事件时序、强M2、query、policy白名单、动作监督key |
| [planning.py](planning.py) | 固定首批5/全部20候选校验；前8不同起点、tail/yaw/u最多64配置；房屋/配置缓存键、预算、冻结与几何聚类 |
| [factory.py](factory.py) | backend注入的真实动作回放调度、环路/逆路/padding/公共尾部/续接构造、18格检查、3seed一致性检查 |
| [journal.py](journal.py) | 独占文件锁、fsync追加日志、SHA256链与HEAD、配置锁定和拒绝损坏恢复 |
| [cli.py](cli.py) | 核验封存候选来源并生成首批P0准备配置；拒绝GPU/生产命令 |

主构造器不再导入旧engine、字符串替换源码或交换全局任务字母。任务角色为anchor_A/anchor_B/terminal/irrelevant槽位，对象/房间及资格实例由Compiler配置；P0仍固定TV/sink/bed/chair语义组合，不自动扩任务或房屋。两任务显式绑定角色。

## 运行与回归

从项目根执行，使用项目内标准库Python：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/data_pipeline/mechanism_factory_v2/run_acceptance.py --attempt acceptance_v2
```

`acceptance_v1`已封存，禁止覆盖；复跑使用新目录并按后续授权管理，不能重封旧SHA。首批配置见[PREPARED_P0.json](acceptance_v1/prepared_p0/PREPARED_P0.json)。其中GPU、可执行起点和语义资格保持null，executable=false；不是未验证字段填默认值的伪生产配置。

80测试分为compiler14、planning31、factory16、journal15、CLI3、集成1。旧27真实日志×2任务的54个标签、逐时刻M2与事件均差分一致。RecordedBackend仅回读这些旧日志，验证27次文件流及8,631次后端动作返回的预算覆盖；这不等于新执行27个物理episode。全部18格仍12正/6负，首族仍interface_only。完整日志见[验收结果](acceptance_v1/result.json)。

## Backend接口与下一步

`FamilyFactory(backend, compiler, budget, emit, task_a, task_b, context=...)`要求：

- context固定house_id及完整asset_config，候选hash和上下文必须一致；部署策略不接收它。
- backend.reset(position,yaw_bin,seed)、observe()、step(F/L/R)→collision bool、reconstruct(pose)、routes(position,yaw,role)。位置/四元数/传感器姿态使用有限数字JSON列表；observe提供真实像素计数及原始像素hash。
- routes只能提供固定顺序的几何动作候选，不得内部执行未计账动作；GreedyFollower若隐式移动agent，不可直接塞入此接口。几何提案不是可达性证书，全部通过runner真实回放验证。
- 所有真实动作，包括失败试走和raw/compressed逆路都经过reserve-before-step。保守预占与确认返回动作分开，不因错误退款、不因重试归零。长阻塞的硬超时、RAM/GPU/磁盘上限需外部watchdog，本模块不能强制中断backend。
- emit须持久化完整原始/规范化边界及轨迹；生产另存按内容寻址RGB/semantic数组并核验像素hash。Journal只处理元数据，不承担数组导出。若存储不支持flock/fsync/原子rename，需要先验收，不假设NAS语义天然成立。
- 数值重建最多一次，目标必须是执行前合法初始姿态；agent和两sensor修正均≤1e-5，重建前后(t-1,t)及(t,t+1)事件都检查。后一比较是保存像素的局部替换诊断，不是未干预全后缀模拟。禁止把它称自然精确汇合。

下一节点先接Habitat适配器和V4导出/loader兼容，复用现成环境，不重新安装或改SFT。运行准入后首批固定5候选、每屋一个，所有搜索实际动作纳入原P0总预算，首个完整预检冻结后不替换，批次结束无论成功几个均回交。程序输出还需核对物理数组、schema、CE owner与输入隔离，不能仅通过validate_matrix就直接标训练集可用。

## 科学边界不变

本节点没有解决单族动作计数捷径，也没有证明交叉续接监督优于强M2或无序视觉记忆。多族生成必须带相应诊断，不能放大当前18格就称贡献成立。V4 query同一时刻按语义pair排序，类别词表显式传递；不兼容旧loader的硬编码V2接口，不等于神经query encoder已实现。SFT、真实长历史梯度、机制独立收益依然分开验收。

代码审查发现并修复的三个漏洞保存在[修订记录](REVISION_LOG_ZH.md)。没有新算法主张、没有泛搜文献、没有把失败改成PASS。
