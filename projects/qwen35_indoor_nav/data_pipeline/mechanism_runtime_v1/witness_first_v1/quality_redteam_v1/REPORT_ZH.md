# 真实正例与损坏负例：CPU 质量红队检查

结论：真实 V3 正例重新调用封存 `quality.audit_family` 通过；原运行再次调用
生产 builder 的 `acceptance.inspect_run` 通过。7 个有界损坏变体全部按预期拒绝。
本轮新增真实物理族为 0，`scientific_pass=false`，不得计入训练数据或量产数。

| 损坏 | 实际拒绝原因 |
| --- | --- |
| 同时翻转 y 与 outcome，重新封测试导出哈希 | LABEL_RECOMPUTATION |
| H_A 多一个 L，重算候选哈希并同步 manifest/candidate | ACTION_COUNT_SHORTCUT |
| 删除一个实际 certification trace 的证据索引 | REPLAY_PATH_COUNT |
| 将实际 future query 加入 policy 记录并重封测试哈希 | POLICY_FIELDS |
| 从证据锁移除候选 | UNSEALED_FILE |
| 候选声明哈希错误 | SEALED_HASH_MISMATCH |
| 生产 builder 输入锁内源码哈希错误 | LOCKED_SOURCE_CHANGED |

前三个涉及导出内容的变体重算了测试副本的文件哈希，故拒绝不是仅仅依赖旧哈希。
这不是给伪造数据签发真实性：副本均位于 `TEST_FIXTURES`，单独标记测试用途，
没有伪造新的仿真结果，也没有复制、改写原 RGB/semantic content。

原正例报告仍限定单屋、controlled task instructions、candidate FIT，不能扩大为
自然指令泛化、模型收益或稳定跨屋量产。603 个原文件前后哈希全部一致。

重要边界：`quality.audit_family` 的 seal 是调用方提供的输入，并不自身证明预执行
时间或源码归属；生产时必须沿封存 builder 的 `inspect_run`/journal/source binding
入口，不能拿手填 evidence 或手填 PASS 替代。第七项直接调用此源码绑定入口。
这次未发现所测七类放行漏洞，不代表穷尽所有验证器错误、语义误标或攻击方式。

一次测试夹具准备中断被保留在 `HARNESS_INTERRUPTION.json`：原 V3 源锁尚未包含
后来的 shared.py，第七个用例选择源码时 StopIteration，未调用验证器。
`finish_v1.py` 改为选择确实存在的 core_bridge.py 锁后完成原第七项；前六份结果
均未重写。`redteam.py` 保留初始执行源码，节点已经关闭，不可原地重跑。

主要机器产物：`result.json`、`SOURCE_LOCK.json`、`SOURCE_UNCHANGED.json`、
`REAL_POSITIVE_REAUDIT.json`、`REAL_SOURCE_REAUDIT.json`、各测试副本的 `REPORT.json`。
