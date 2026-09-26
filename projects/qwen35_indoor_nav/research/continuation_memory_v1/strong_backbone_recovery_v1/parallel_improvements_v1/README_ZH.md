# 等待评测期间的改进研究

本目录新增于2026-09-26；不修改或占用正在运行的369路线评测。主线以 [整体VLN方案V2](../paper_narrative_20260925/WHOLE_VLN_PROGRAM_V2_ZH.md) 为准，目标覆盖普通指令导航与室内外迁移，数据/模型/训练三贡献均仍需验证。

- `RELATED_WORK_NOTES.md`：一手先例与具体差异边界，不把进度、未来预测、记忆、DPO或两阶段本身当新贡献。
- `data_audit.py` / `DATA_ACTION_AUDIT.json` / `DATA_FINDINGS_ZH.md`：334条真实轨迹的动作、STOP、known mask和实际转移审计。定位共享覆盖缺项及候选执行结果的资产缺口。
- `control_models.py`：单位范数读出的LOCAL、EMA、RECURRENT强简单对照。LOCAL不读取原完整历史范数，避免把诊断中的幅度借用变成部署旁路。三者存储参数量相同，有效递归参数使用不同并明确报告。
- `test_controls.py`：三个CPU行为测试，验证正幅度缩放不改变读出、因果历史梯度与reset、初始化原生动作保持。
- `cpu_controls.py` / `CPU_CONTROLS_RESULT.json`：同初始化、真实FIT两条完整轨迹，三个模型各3次轻量CPU更新。无GPU/无新VLM前向/无DEV或unseen选模，不保存候选训练权重。只是实现检查。

## 本轮新增的模型能力实现

- `execution_memory.py`：动作条件观测变化记忆原型。保留8×64递归状态，在写入时显式增加上一实际动作、当前与上一因果特征的差；动作头读取单位范数状态。START独立，STOP后无新观察，全序列不断梯度。特征包含视觉、指令及上一动作编码，差分不是纯视觉变化，显式动作项也可能重复已有信息。
- `test_execution_memory.py` / `CPU_EXECUTION_MEMORY_RESULT.json`：8项测试，以及真实FIT117步轨迹的前向、两次反向/参数更新。零头初始化保持原生logits，打开动作头后验证未来特征不会影响此前动作；末端监督能沿96步展开传到早期输入与差分写入。增加1,837,568参数，总计4,695,688；后续必须有同容量普通writer对照，不能把容量增加归功于机制。
- `train_controls.py`：LOCAL/EMA/RECURRENT三个单位范数记忆对照的真实可恢复训练入口。默认同42/43/44种子、原INITIAL与共享3000步schedule、原AdamW与恢复/保留损失；每200步保存模型、optimizer、RNG及源码/数据绑定。不会读取unseen损失或自动挑checkpoint。
- `test_train_controls.py` / `CPU_TRAINER_TEST_RESULT.json`：3项测试，真实FIT两条轨迹上连续3步与1+2恢复后的模型、优化器、RNG相同；FINAL落盘而RESULT缺失也可恢复。270条FIT及所有种子完整schedule已零更新预检。

数据侧已将全部实际动作及缺失STOP位置导出为[可复算索引和补采集清单](../execution_data_v1/README_ZH.md)。室外暂只保留入口备案，不下载、不实现适配。

## 当前边界与下一次训练

本轮没有新GPU训练或SR结果。真实缓存上的梯度/更新是实现证据，不能称为导航能力已提升。等范数对照用于查清已有收益是否依赖递归内容；动作变化原型是下一候选，两者没有偷偷混成一次实验。

训练入口从本目录调用：

```bash
PY=/mnt/data_nas/deeprobotics/daiyang/vla/.envs/b33_streamvln_v1/bin/python
"$PY" -I -B train_controls.py --run runs/unit_controls_001 --device cuda:0
# 中断后同一配置恢复，已完成模型不会重训：
"$PY" -I -B train_controls.py --run runs/unit_controls_001 --device cuda:0 --resume
```

这些命令尚未提交GPU。现有独立服务入口可用控制Python执行os.execv进入上述环境解释器，保持同UID/GID和独立systemd生命周期；不能直接把环境解释器传给旧standalone的顶层白名单。实际提交时使用获准设备并接入进度页。

新单位范数权重必须由`UnitReadoutMemory`加载，动作变化原型需对应实际动作接口；现有评测的旧模型构造器不能代替。闭环适配、补采集与全部token动作残差仍未完成。STOP共同修补应让所有比较臂共享，且需要新的执行版本；不能把共同修复收益独占归因于提出的架构。
