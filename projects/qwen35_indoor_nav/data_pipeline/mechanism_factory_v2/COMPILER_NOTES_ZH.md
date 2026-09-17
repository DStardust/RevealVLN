# 通用纯编译器交付

范围：CPU-only；仅新 compiler.py、test_compiler.py 和本文。未改封存源码/数据，未运行仿真、GPU、模型或外部 SFT。

## API 与变化

`Compiler(roles, tasks, eligible, task_revision=..., sensor=None)` 持有独立复制的只读映射与 tuple；无可变全局 KINDS/TASKS。角色槽名任意，任务显式指定 anchor/terminal/instruction，不硬编码单屋/起点/终点类别。默认传感器保持冻结版本；其他传感器需新协议，当前拒绝。

实例接口：atoms、complete、evaluate、m2、query_from_trace、encode_query、policy_at、policy_semantics、task_program、action_keys、event_certificate。module 提供 canonical/digest/ref/pixel/complete/slice_continuation/semantic_query/policy_semantics。

- 固定 256 像素、同一实例连续两帧，anchor 必须严格先于最后 STOP 的 terminal 见证。
- 不完整、缺证据、碰撞、非法动作/STOP/步序、非法像素计数返回 unknown；不自动转成负例。M2 只用到当前的证据与已执行动作，未来损坏不改变先前监督。
- V4 query 保留顺序、事件阈值及 movement repeat；同一时刻多个事件按语义排序，去除任意角色字母顺序影响。只是序列规范化升级，不改变任务标签。
- encode_query 返回 token_ids、显式 vocabulary、encoding_schema。动态类别词表随返回值传递，不能把不同实例的裸整数当共享词表。该函数不是已训练的 Qwen query encoder。
- policy 白名单独立构造，不复制像素计数、实例、pose、house、标签、query；record ID 仅作存储索引，policy_semantics 不输出 ID。只读取最近两帧、已执行八动作、文本指令与冻结传感器规格。
- action_keys 仅为 pass 轨迹生成监督 payload，失败/unknown 拒绝。V4 policy schema 改版；不承诺新旧文件/哈希字节一致。

## 验证与边界

测试命令（项目根）：

```
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/data_pipeline/mechanism_factory_v2/test_compiler.py
```

14 个 unittest 首次执行全部通过，约 2.5 秒。真实回归在只读的 27 条封存 trace 上逐条比较 complete/逐步 atoms，按两个任务比较共 54 个 evaluate 与全长 M2，并核对 18 个 canonical cell 的封存预期；同时比较截止 206 的 policy semantics。新增 query 回归比较旧 V2 与新 V4 的保序动作和同一时刻事件集合。测试末复核 27 个源 trace 字节哈希未变。

其余测试为明确的合成工程夹具，包括阈值、实例混淆、同步首次见证、unknown、非法 STOP/时间、配置隔离、角色重命名、输入白名单、因果性、query 非法字段和动作去重。

差分测试只在其独立载入的旧模块对象中设置历史 V3 实例配置，不修改磁盘旧 compiler。既有真实日志重算不等于新房屋物理运行。没有新增族、导航收益、梯度结果或训练准入；生产 Habitat 适配及 5 候选实际生成由后续独立节点验证。
