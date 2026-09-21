VALID_COMPLETE_DEV_REPAIR

完整续接 768/768；完整六模型组 128/128。本轮更新0；复用初版的3600次STOP读出更新，原模型参数更新0。

|终点|方法|PASS/N|未知/未采集/未运行|成本|
|---|---|---:|---:|---:|
|main|MONOTONIC|66/192|0|0.7115|
|main|STOPFIX|62/192|0|0.7319|
|control|MONOTONIC|98/192|0|0.5413|
|control|STOPFIX|98/192|0|0.5357|

STOPFIX−MONOTONIC 主终点差值：-2.08%；未知分配识别界 [-2.08%, -2.08%]。这不是置信区间。
方向为正的房屋 0/1、种子 1/3；原DEV局部修复信号：False。

复用初版固定权重，无新增训练。动作教师偏差与任务状态错停已经分开；本轮以真实闭环检查收益和代价，未把原DEV改称盲测。
父族内历史、终点变体、同屋和多个种子有相关性，不能将768次执行当成768个独立泛化样本。
本轮不自动采用、不继续训练；普通VLN-CE收益与算法新颖性仍需另外证据。
Exposed original DEV house, frozen MONOTONIC vs a trained STOP-only residual. No independent holdout or natural VLN-CE claim.
