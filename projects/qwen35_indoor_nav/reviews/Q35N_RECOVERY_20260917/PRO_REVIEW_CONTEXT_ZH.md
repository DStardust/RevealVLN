# 给GPT网页版Pro的研究方向审查任务

请先读完这个文件和下列指定源码/结果，再给方向意见。若无法访问GitHub分支或某个文件，请明确说明已读范围，不要假装已检查整个项目。目标是帮助一个进展不顺的Qwen3.5-2B室内导航项目收敛到可用基座，再选择一个小而明确、可部署的贡献。

## 用户希望得到什么

1. 优先定位有证据的修复点，争取在完整R2R-CE v1-3 val_unseen 1839条上达到SR≥40%。这是工程目标，不是已实现结果或已有方法的性能保证。
2. 基座稳定后开展任务条件执行记忆研究；不要仅堆新模块、泛泛提出更多数据或更长训练。
3. 用户新增未来方向备选1：“语言指定的小物体导航”。请评估其适合作为未来方向的程度，保留当前主线；不要把小物体类别/实例目标导航与R2R路径指令导航混为一谈。

## 必须保留的事实

模型：Qwen3.5-2B冻结底模/视觉塔，四类动作move_forward/turn_left/turn_right/STOP；原指令、最多2帧224×224 RGB、最多8个已执行动作，batch1 greedy。LoRA原覆盖6个full-attention层的q_proj/v_proj、rank8；导航输入中没有目标坐标、场景ID、真值碰撞、未来帧、路线真值。训练监督与在线输入隔离。

现有普通池：37114指令、25415物理路线、51个FIT房屋、2650347动作。R2R 8700指令/532416动作，RxR 8848/862328，EnvDrop 19566/1255603。EnvDrop约占47.4%动作；自然STOP约1.4%。分布描述尚不构成退化归因。

结果不能跨checkpoint或split拼接：

| 记录 | SR | SPL | nDTW | 身份 |
| --- | ---: | ---: | ---: | --- |
| 旧41800步完整val_unseen1839条 | 22.02% | 18.68% | 38.68% | 405成功，历史已暴露结果 |
| 保留best4k开发100条 | 21% | 18.02% | 36.59% | 本轮独立推理入口使用它 |
| 原全池一epoch开发100条 | 14% | 12.99% | 35.37% | 更长训练没有带来目标收益 |
| 最近2帧/8帧配对开发 | 10% / 12% | 见原交接 | 见原交接 | 没有可保留的新基座 |

完整原生FIT重放5487个唯一输入/7225次出现，logits与动作差为0，已完成；不要建议重跑同一关卡。旧V12有80个特征不一致的失败仍保留。

本轮新诊断：原结构+FP32训练桥接，按房屋分离的均衡256训练/128检查决策，完成400更新12800次读取；100步起训练准确率和四类召回全部100%，全部28个可训练张量更新且有限。检查准确率60.16%→67.19%，交叉熵0.9147→2.2252。因此跳过条件LoRA扩覆盖；小集权重不是导航候选。这能排除什么、不能排除什么，请严格限定范围。

本轮闭环故障证据：原best4k开发100条中23条失败有完全相同的指令+2RGB+8已执行动作输入重复，重复logits完全相同；8556/19122决策来自这些重复输入。另有26条进入目标范围却没有成功STOP。两种失败集可能重叠，不能简单相加。

工程试验：只加入完整输入重复时的最少尝试运动选择，保留模型原生STOP，全部恢复动作计入500步。
V1因首次干预前动作差异在33/100条处主动停止，不采纳部分结果。同GPU两个进程也存在最大0.11094的
接口logit差异；差异来源尚未归因到具体算子，不能说已确定GPU或模型初始化有bug。
V2改为单一模型进程内原策略/恢复策略各100条，但因外部任务进入GPU4在3/200处资源中断；V3同协议迁移到GPU1，最终结果见 `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/run_001/`。
必须核对首次干预前完整输入、动作和logits；若文件未完结，不自行补结果，也不把两组混合为200条单一SR。

最终V3在19/200条时也因外部任务进入GPU1资源中断，全部自有进程已清理。故完整的
配对RESULT/REVIEW尚不存在，恢复增益为null、没有被采用；终态集中于本目录
`FINAL_REVIEW.json`。请依据已有诊断给下一步方向，不将资源中断当作算法否证。

数值差异的一条待验证线索：本地FLA 0.5.2默认autotune，forward存在多个warps/stages
候选，各进程使用独立编译缓存。尚未比较实际kernel配置、冻结参数指纹和逐层激活；
不要把这一线索写成已确认原因，也不要建议跳过原数值门槛来得到更好分数。

## 需要实际读取的文件

所有相对路径以 `projects/qwen35_indoor_nav/` 为起点：

- `CURRENT_STATUS.json`：当前终态和待办。旧 `STATUS.json` 含历史运行信息，不能据此判断进程仍活跃。
- `reviews/Q35N_RECOVERY_20260917/REPORT_ZH.md`、`DATA_MIX.json`、`REPEATED_INPUTS.json`。
- `sft_acceptance/ordinary_sync_recovery_v1/model.py`、`data.py`：实际处理器、动作查询、LoRA、后缀编码、forward与数据加载。
- `sft_acceptance/ordinary_learnability_v1/PLAN_ZH.md`、`run.py`、`current/RESULT.json`。
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/SPEC_ZH.md`、`cycle_policy.py`、`evaluate.py`、`review.py`和 `run_001/LAUNCH_RESULT.json`；完整效果报告未生成，V1/V2终止说明另保留。
- `deployment/ordinary_v1/predict.py`、`MODEL_CARD.json`、`DEPLOYMENT_ACCEPTANCE.json`：可调用基线、边界和真实数值验证。
- `HANDOFF_CODEX_LUNA_MAX_20260914_ZH.md`：历史失败、当前数据资产和不可混淆的指标身份。
- `MAINLINE_FREEZE_V3.md`：原研究假设是任务条件历史经真实交叉续接监督，训练实际部署记忆；训练期query读出在部署删除，记忆本体保留。需强任务状态监督和相同数据/架构action-only控制。
- `FUTURE_OPTION_1_SMALL_OBJECT_NAV_ZH.md`：小物体备选的一手来源、协议差异和可验证起点。

## 请给出的具体结论

先列不超过3个最可能阻碍SR40的原因，并为每个区分“已证实”“有根据的假设”“缺证据”。引用这里的真实文件和具体行为，不把“还有bug”当作解释。

然后只选择一个最值得执行的下一实验：明确唯一变更、固定初始化与数据、运行预算、可观察到的支持/否定结果、停止条件，以及它为何比继续扩模型/扩历史/长训更能减少不确定性。如信息不足，指出一个最小额外测量，勿提出全面重构。

最后分别裁定两条未来路线：原任务条件执行记忆、小物体语言导航。小物体已有 [Small Object Navigation](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27487)、[FindThis](https://proceedings.mlr.press/v229/majumdar23a.html)、[LangMap/PlaNaVid](https://arxiv.org/abs/2602.02220v2)、[LangNav/MLFM](https://3dlg-hcvc.github.io/langmonmap/)。请检索最新一手论文/源码，给具体差异和最小控制，勿以未检索到完全同名方法判新颖。LangMap等已含记忆/上下文；“先找桌子再找杯子”“加记忆”“用大模型”不足以成贡献。

如果认为当前Qwen导航路线不值得继续，也请指出导致这一判断的证据，以及一个成本明确、能反驳你的最后实验；不能只给鼓励或宏观方向。
