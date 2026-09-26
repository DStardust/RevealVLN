# 等待评测期间的改进研究

本目录新增于2026-09-26；不修改或占用正在运行的369路线评测。主线以 [整体VLN方案V2](../paper_narrative_20260925/WHOLE_VLN_PROGRAM_V2_ZH.md) 为准，目标覆盖普通指令导航与室内外迁移，数据/模型/训练三贡献均仍需验证。

- `RELATED_WORK_NOTES.md`：一手先例与具体差异边界，不把进度、未来预测、记忆、DPO或两阶段本身当新贡献。
- `data_audit.py` / `DATA_ACTION_AUDIT.json` / `DATA_FINDINGS_ZH.md`：334条真实轨迹的动作、STOP、known mask和实际转移审计。定位共享覆盖缺项及候选执行结果的资产缺口。
- `control_models.py`：单位范数读出的LOCAL、EMA、RECURRENT强简单对照。LOCAL不读取原完整历史范数，避免把诊断中的幅度借用变成部署旁路。三者存储参数量相同，有效递归参数使用不同并明确报告。
- `test_controls.py`：三个CPU行为测试，验证正幅度缩放不改变读出、因果历史梯度与reset、初始化原生动作保持。
- `cpu_controls.py` / `CPU_CONTROLS_RESULT.json`：同初始化、真实FIT两条完整轨迹，三个模型各3次轻量CPU更新。无GPU/无新VLM前向/无DEV或unseen选模，不保存候选训练权重。只是实现检查。

当前不启动新GPU训练。等范数简单对照不是论文提出方法，STOP覆盖修补也不能独占归因；新的执行证据更新模型及室外接口尚待实现和实测。
