# 近邻差异补记，未做新的方向搜索

本轮只核查用户列出的部分一手入口，其余沿用明确注明范围的旧审查。无端到端复现，
不证明全球无同构先例。当前研究新颖性与经验主张仍为UNTESTED。

|工作|本轮读取范围|对本契约的约束|
|---|---|---|
|[Progress-Think](https://arxiv.org/abs/2511.17097)|摘要、版本记录：v2于2026-04-14修订；本轮未重读RoboOrchardLab实现|已有语义进度推理与策略训练，不能把进度/状态记忆本身当创新；须比较高质量精确状态|
|[PSR](https://arxiv.org/abs/1207.4167)|摘要及UAI2004出处；2012是arXiv上传时间|历史×未来检验与预测性状态已有理论基础；不声明新PSR或有限Y的全局等价|
|[Reward Machines](https://arxiv.org/abs/2010.03950)|摘要、版本与JAIR2022出处|有限状态任务结构、非Markov奖励与自动监督并非空白；Y不比充分程序状态信息更多|
|Dual-Anchoring、Instruction-as-State、HAM-VLN、Noisy Symbolic Abstractions|本轮读取用户附件与旧P1报告中的核查说明；未新增完整论文/代码审计|保留进度、任务结构、失败/记忆近邻；不把附件的读取经历说成本轮复现|

未来模型增量只能是具体可执行交叉关系监督相对于匹配B1/B2/B3的证据。最强简单替代
不是粗进度数；本地已有Compiler.m2，使B2可自动生成且优先级高。

方向B仅更新边界：[SAP-Nav](https://arxiv.org/abs/2608.12707)于2026-08-13提交，摘要明确
包括观察充分性判断、必要时主动重定位、类别/属性目标验证；入口写接收后公开代码。
本轮核查到的是摘要/版本与发布说明，没有核验完整官方可运行实现。因此“换视角再验证”
不能独立作为新贡献，也不能因代码未核验降低思想先例的重要性。本轮不推进B。

保留 [LangMap/PlaNaVid](https://arxiv.org/abs/2602.02220v2)、
[FindThis](https://proceedings.mlr.press/v229/majumdar23a.html)、
[Small Object Navigation](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27487)、
[LangNav/MLFM](https://3dlg-hcvc.github.io/langmonmap/) 在继承先例表中；具体旧源码疑点见
`../../FUTURE_OPTION_1_SMALL_OBJECT_NAV_ZH.md`，本轮未重做其完整数据/权重/协议审计。
论文、代码入口、实际函数、数据权重与协议一致性分别登记，README不等于复现。
