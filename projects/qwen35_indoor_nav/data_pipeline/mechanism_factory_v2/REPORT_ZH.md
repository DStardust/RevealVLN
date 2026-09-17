# Q35N 通用构造核心 CPU 节点裁决

结论：`GENERIC_FACTORY_CPU_CORE_PASS`。用户“实现下一步”已落实为代码和测试，节点收口。

- 新增实例化任务编译、backend无关构造调度、候选顺序/预算/冻结/几何去重、持久化账本与CPU准备CLI。
- 整体验收80/80测试通过，0失败/0错误，实测约10.51秒；[逐模块结果及日志](acceptance_v1/result.json)。
- 旧27条真实日志×2任务的54次标签差分保持一致，完整M2/事件及18格也一致；另用文件回放backend验证27流和8,631次确认返回计账。这不是新物理回放或新族。
- 代码审查的边界事件、像素键冲突和超时接收漏洞已修复，回归测试通过；[修订记录](REVISION_LOG_ZH.md)保留发现。
- 原封存族、旧候选、前轮三支线和主审核、5个旧compiler锁均前后复核一致；[前](acceptance_v1/PROTECTED_BEFORE.json)/[后](acceptance_v1/PROTECTED_AFTER.json)。
- 已实际生成首批5候选的[CPU准备配置](acceptance_v1/prepared_p0/PREPARED_P0.json)，不是可运行场景配置。56个reserved屋与外部SFT不变。

主agent接收范围仅CPU构造核心。**完整生产管线尚缺Habitat backend、真实资源watchdog、V4数组/训练文件导出及loader适配**；无runtime_pass、无新物理族、无科学收益。generate/certify拒绝执行，不留下可误启动的默认卡号或默认场景。

下一步明确为上述生产接入，之后按已登记P0首批5候选做有界构造诊断。旧环境和权重只读复用，不重新安装、不改正在进行的普通SFT。真实模型query编码与207步梯度仍由独立节点检验；不能因为CPU通过就启动机制训练。

完整接口和复现命令见[README](README.md)。论文主线未改变；学术预审不升级为贡献充分性、CVPR竞争力或导航收益PASS。
