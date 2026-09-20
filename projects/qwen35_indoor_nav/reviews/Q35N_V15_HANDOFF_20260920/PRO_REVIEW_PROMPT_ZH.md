你正在一个全新的会话中审查我的视觉语言导航项目。请只依赖下面的背景和你实际打开的证据，不假设能看到我与 Codex 的历史聊天，也不假设能访问服务器本地路径。请用中文回答。

我的目标是把当前 Qwen3.5-2B 导航系统修到可信，并判断“任务条件执行记忆”是否值得继续研究。我需要可证伪的下一步，不能预设必须得到正结果或一定能发文章。

一、准确定位仓库和证据

仓库：https://github.com/DStardust/RevealVLN
工作分支：codex/q35n-v15-legal-data-20260919
本次审查固定提交：754d574bc98fd49fe30b2be76e30cbc53f0f25c3
项目根：projects/qwen35_indoor_nav/

上述提交已经包含完整结果和补充资产。请按下面的完整 URL 直接打开；不要只搜索项目名称或根据目录名猜文件。链接固定到已上传的提交，新版 prompt 的提交变化不影响这些证据。

1. 总入口与文件导航（GitHub 网页）：
https://github.com/DStardust/RevealVLN/blob/754d574bc98fd49fe30b2be76e30cbc53f0f25c3/projects/qwen35_indoor_nav/reviews/Q35N_V15_HANDOFF_20260920/README_ZH.md

2. 完整中文实验报告（原文）：
https://raw.githubusercontent.com/DStardust/RevealVLN/754d574bc98fd49fe30b2be76e30cbc53f0f25c3/projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/REPORT_ZH.md

3. 全部 216 次结果，每行一次完整续接：
https://raw.githubusercontent.com/DStardust/RevealVLN/754d574bc98fd49fe30b2be76e30cbc53f0f25c3/projects/qwen35_indoor_nav/reviews/Q35N_V15_HANDOFF_20260920/ROLLOUTS.csv

4. 配对、数值、失败类型和动作成本汇总：
https://raw.githubusercontent.com/DStardust/RevealVLN/754d574bc98fd49fe30b2be76e30cbc53f0f25c3/projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/CONTINUATION_REVIEW.json

5. 全部模型的训练诊断：
https://raw.githubusercontent.com/DStardust/RevealVLN/754d574bc98fd49fe30b2be76e30cbc53f0f25c3/projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/TRAINING_REVIEW.json

6. 实际数据规模、标签偏置与可辨识性限制：
https://raw.githubusercontent.com/DStardust/RevealVLN/754d574bc98fd49fe30b2be76e30cbc53f0f25c3/projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/DATA_INFORMATION_AUDIT.json

7. 实际训练损失、强 B2 对照和查询接口代码：
https://raw.githubusercontent.com/DStardust/RevealVLN/754d574bc98fd49fe30b2be76e30cbc53f0f25c3/projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/objective.py

8. 关键事件识别、误报和保持的只读诊断：
https://raw.githubusercontent.com/DStardust/RevealVLN/754d574bc98fd49fe30b2be76e30cbc53f0f25c3/projects/qwen35_indoor_nav/research/continuation_memory_v1/legal_closed_loop_v15/EVENT_STATE_DIAGNOSIS.json

先读 1、2、3，再针对判断读 4–8。总入口还链接了当前状态、原始数据证书、查询/精确状态充分性、视觉捷径审计、源码锁、真实梯度测试、完整日志压缩包和资产清单。需要代码细节时继续打开对应链接。本消息是本次审阅指令，旧交接文本仅作背景，不恢复已经被替代的执行门槛。

网页无法解析时尝试对应 Raw 文件；Raw 无法访问时尝试对应 GitHub blob 页面或可用的 GitHub 连接器。如果仍有 403/404、权限或工具限制，请明确列出，不能把“访问不到”写成“仓库没有文件”。下述背景足以开始提出暂定诊断，但未经读取不能宣称已完成仓库审计；指出要我补贴的最小文件即可。下载了压缩包、权重或缓存不等于已解包/加载/复现。

二、项目、候选方法和已发生的结果

当前基座是保留的 best4k Qwen3.5-2B。历史 41800 步 checkpoint 在完整 R2R-CE val_unseen1839 上 SR22.02%；best4k 的内部开发100条 SR21%。两者是不同 checkpoint、不同分母，不能混用。本轮 V15 是执行记忆探索，不是完整 R2R SR 评估。SR40、论文贡献和真机部署都未完成。

候选主张：从“真实历史 × 合法续接 × 任务”的交叉结果学习部署时使用的任务条件记忆，是否比动作模仿、精确程序状态监督带来额外闭环收益？若精确状态已足够，则交叉标签并不提供更多信息；优势只能通过匹配实验证明。循环计数恢复器不是这种学习记忆，不能拿控制器成绩证明该研究。

V15 冻结基座、dtype、attention/FLA 路径，采用同一 8×64 轻量递归记忆；输入是原指令、最近至多两张 224×224 RGB 和最近至多八个实际执行动作。旧历史只能通过记忆进入动作策略。未来续接查询只进入训练读出器，不进入记忆或动作前向；本 pilot 查询是四个审核后的后缀字段，并非已完成通用语言续接编码器。

数据是 6 个真实 debug 族、72 次完整交叉执行、20 条去重物理历史，只有 1 FIT 屋和 1 CHECK 屋，各 3 族。原 training_admission=false 保留。真实回放和证书通过，不等于正式研究数据准入或跨屋泛化证据。

对照为：B1=仅动作监督；B2=动作监督加精确组合式程序状态；Ours=动作监督加交叉续接结果。S1 使用 1 个 FIT 族，L3 使用同屋全部 3 个 FIT 族；3 个种子，共 18 个模型、10,800 次轻量更新，基座更新为 0。同一数据规模下的匹配臂共享架构、初始化、合法轨迹池和更新预算。

训练族的教师分叉都能学会；交换错误历史的记忆会破坏动作，匹配状态的 sham 记忆保留成绩。真实反向覆盖相隔 135 步的关键事件并更新记忆/动作参数。但 B1、B2、Ours 都有这些信号；所有规模、种子、监督臂的新屋教师分叉均为 0/12。不要将训练可拟合写成 Ours 的独有贡献。

216 次实际自主续接已全部完成：1 PASS、128 FAIL、87 UNKNOWN，唯一 PASS 来自 B1。这里是实际执行固定历史后自主行动；完整历史、动作和 STOP 共用 500 决策。任务要求同一语义实例连续两帧各至少 256 像素的 SEE2 事件顺序和合法主动 STOP，不是“到访房间”。原工厂检查器将碰撞轨迹标为 UNKNOWN；不能事后转为 FAIL、删除它们，或把部分已知标签当完整成功率分母。

144 项组内输入和原生动作前缀审计通过，原生 logits 最大差 0；另一个层面上，训练缓存与现场的跨会话比较有 5 次原生 argmax 翻转。这两个问题必须分别判断。全部 GPU 会话累计约 3.2034 小时，已结束。

独立 systemd 流水线已实际验证可在启动终端退出后继续执行。后续长任务不能依赖 Codex 在线。完整代码、18 个轻量 checkpoint、3 份初始化、因果特征缓存和日志已上传；大型基座、场景及原始 RGB/语义数组未上传，资产清单已说明。

三、请解决的诊断问题

请比较以下解释，区分“有证据支持”“被反证”“仍混杂/未知”，不要直接默认答案是更多数据或更大模型：

- 数据覆盖/事件识别：一个训练屋，类别、房间、尺度和屋同时变化，B2 存在缺失事件误报。应该补怎样的跨屋正负例才能隔离问题？
- 可辨识性/捷径：初始 RGB 与历史身份相关，所有任务又有通用已测 PASS 后缀。当前信号是否只支持实例记忆和有限集合效率？
- 闭环恢复：教师动作可学为何自主续接几乎不成功？是否缺少共享的合法偏离/恢复轨迹，还是任务识别、终止或动作读出问题？
- 评测：为数据工厂设置的碰撞 UNKNOWN 是否适合自主效果估计？任何修订只能提出前瞻新版本，保留旧结果，不能靠放宽成功条件制造收益。
- 数值：跨会话漂移是否必须先修，能否解释主要问题？固定缓存上的新屋教师分叉已经为 0，不可全部归因于 FLA 猜测。
- 简单替代：精确状态与查询能重建所有交叉标签，为什么还需要 Ours？什么结果支持保留其模型主张，什么结果应优先采用 B2？

四、请按以下格式交付

1. 实际读取的 URL/文件、无法访问项，以及证据化的诊断排序。
2. 明确建议下一步优先补何种数据、修何条路径，或为何已有证据足够改架构；指出当前不能得出的结论。
3. 只选一个最有判别力的下一实验：假设、固定因素、唯一变化、强对照、完整分母、房屋/历史/语言族隔离、主要终点、成本、支持与否定条件。说明什么结果才支持改架构，不重复同一小集拟合。
4. 给一份可直接交回 Codex 的执行指令：对应现有文件/接口、必须补的实现、独立后台运行、日志与断点、正确性停止条件、最终产物。不要只给 idea 或禁止执行的契约。

允许共享且资源足够的 GPU、分段试跑和基础设施修复，不恢复旧的专用窗口、4100 秒、一次启动或必须 SR40 才能研究的限制。预算依据实测吞吐给范围和未知项，不捏造运行时长；不按是否正分反复改同一 CHECK 集。

本次请先完成诊断与下一步任务设计，不自行启动训练。不要承诺一定能发文章或得到正结果，也不要把有限失败夸大为整个执行记忆方向不可能。小物体导航方向 B 暂不启动。
