VALID_COMPLETE_CONTROLLED_HOLDOUT

完整续接 1488/1536；完整六模型组 248/256。新增训练更新 7200（两组×三种子×1200）；基座更新0。

|终点|方法|PASS/N|未知/未采集/未运行|成本|
|---|---|---:|---:|---:|
|main|OLD|131/384|12|0.7112|
|main|EXPANDED|129/384|12|0.7066|
|control|OLD|215/384|12|0.5055|
|control|EXPANDED|193/384|12|0.5440|

EXPANDED−OLD 主终点差值：-0.52%；未知分配识别界 [-3.65%, 2.60%]。这不是置信区间。
方向为正的房屋 1/4、种子 1/3；有限数据扩充信号：False。

既有开发屋结果用于选择本次候选，未重命名为盲测。本轮训练配置和评测顺序在新分数产生前冻结；四屋已暴露，不能称新盲测。
父族内历史、终点变体、同屋和多个种子有相关性，不能将1536次执行当成1536个独立泛化样本。
本轮不自动采用、不继续训练；普通VLN-CE收益与算法新颖性仍需另外证据。
Four previously exposed heldout development houses, shared instruction templates and stationary SEE2 task. Base-training exposure unresolved; no natural VLN-CE, deployment or paper acceptance claim.
