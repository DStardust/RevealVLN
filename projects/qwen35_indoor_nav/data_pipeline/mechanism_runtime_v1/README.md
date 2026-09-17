# 机制数据生产接口 V1

独立写入目录，旧 factory_v2、旧机制族和外部 SFT 只读。本节点不是模型训练。

已执行的权威证据：`cpu_v2/result.json`、`smoke_v1/result.json`、`smoke_v1/SUPERVISOR_RESULT.json`。旧单族实际 9 条回放通过，不能增加独立族计数。

代码分工：

- `habitat_backend.py`：固定 Habitat 传感器、语义实例、实际动作与纯几何路径提案。
- `core_bridge.py`：校验并只读加载封存的 Compiler V4、构造器和预算/冻结机制。
- `guard.py`、`runtime_journal.py`：内容寻址图像存储、持久证据、资源额度。
- `exporter.py`、`loader.py`：九条轨迹交叉成十八格，独立重算监督，部署策略输入与离线元数据隔离。
- `worker.py`、`run.py`：已关闭的旧族真实接口验收。禁止重跑覆盖 smoke_v1。
- `p0_driver_v1/`：后续五候选调度器的 CPU 实现；`p0_batch_v1/CONFIG_DRAFT.json` 未准入真实运行。

数据来源仍为已授权 MP3D 场景与官方 R2R-CE v1-3 train 路线。机制标签由真实语义渲染、固定事件判定和交叉续接程序自动生成，不使用人工 U/A/D 标签，不向策略输入语义、坐标或未来帧。

查看 [主报告](REPORT_ZH.md) 与 [下一节点交接](NEXT_P0_ZH.md)。CPU 测试、物理数据有效性、模型收益分别记账。
