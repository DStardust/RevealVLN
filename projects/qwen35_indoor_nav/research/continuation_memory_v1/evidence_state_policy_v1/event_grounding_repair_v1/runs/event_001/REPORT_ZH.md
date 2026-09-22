VALID_COMPLETE_DEV_EVENT_GROUNDING

完整续接 768/768；完整六模型组 128/128。本轮6模型各1200次更新，共7200次；底模更新0。

|终点|方法|PASS/N|未知|成本|碰撞轨迹|耗尽|未满足STOP|
|---|---|---:|---:|---:|---:|---:|---:|
|main|ORIGINAL|66/192|0|0.7115|22|26|96|
|main|EVENT|56/192|0|0.7739|39|52|83|
|control|ORIGINAL|98/192|0|0.5413|27|24|65|
|control|EVENT|95/192|0|0.5826|41|44|50|

EVENT−ORIGINAL 主终点差值：-5.21%；未知分配识别界 [-5.21%, -5.21%]。这不是置信区间。
方向为正的房屋 0/1、种子 0/3；原DEV局部修复信号：False。

两臂同一MONOTONIC架构、初始化、完整动作监督、状态/KL/普通动作loss及1200步。EVENT仅替换事件BCE的监督分布：唯一输入、屋/角色等权、实际正负比例；不是新数据或新推理规则。
父族内历史、终点变体、同屋和多个种子有相关性，不能将768次执行当成768个独立泛化样本。
缺失历史主任务配对差：2.08%，识别界[2.08%,2.08%]；事件留一屋诊断与完整训练事件复核另报，不将读出指标当闭环收益。
本轮不自动采用、不追加训练；普通VLN-CE收益与算法新颖性仍需另外证据。
Exposed DEV1 house; same MONOTONIC architecture and original trajectory pool, only unique-input house/role-uniform event BCE differs. Stationary SEE2, no independent test or natural VLN-CE claim.
