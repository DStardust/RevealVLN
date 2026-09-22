VALID_COMPLETE_DEV_TEACHER_ALIGNMENT

完整续接 768/768；完整六模型组 128/128。本轮6模型各1200次更新，共7200次；底模更新0。

|终点|方法|PASS/N|未知|成本|碰撞轨迹|耗尽|未满足STOP|
|---|---|---:|---:|---:|---:|---:|---:|
|main|ORIGINAL|66/192|0|0.7115|22|26|96|
|main|ALIGNED|47/192|0|0.7746|47|36|98|
|control|ORIGINAL|98/192|0|0.5413|27|24|65|
|control|ALIGNED|93/192|0|0.5513|42|31|57|

ALIGNED−ORIGINAL 主终点差值：-9.90%；未知分配识别界 [-9.90%, -9.90%]。这不是置信区间。
方向为正的房屋 0/1、种子 0/3；原DEV局部修复信号：False。

两臂为同一MONOTONIC架构、相同初始化/普通动作池/辅助监督/1200步。ALIGNED只改变经真实重放认证的首次合法完成动作及随后动作mask；未把共享原轨迹的辅助尾部伪称为STOP后观测。
父族内历史、终点变体、同屋和多个种子有相关性，不能将768次执行当成768个独立泛化样本。
缺失历史主任务配对差：-7.29%，识别界[-7.29%,-7.29%]；原有12处跨变体运动教师冲突另报，不宣称已修复。
本轮不自动采用、不追加训练；普通VLN-CE收益与算法新颖性仍需另外证据。
Exposed DEV1 house; same MONOTONIC architecture and original trajectory pool, only certified teacher action targets/masks differ. Stationary SEE2, no independent test or natural VLN-CE claim.
