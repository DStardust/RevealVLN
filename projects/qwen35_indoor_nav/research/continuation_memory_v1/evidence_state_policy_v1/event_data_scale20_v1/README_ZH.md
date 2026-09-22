# 20倍真实数据扩充（生成任务，未训练）

基准为当前32个FIT父族、40个全池父族。要求新增至少800个经过原始数组和冻结SEE2证书审核的父族；固定17屋各最多64个，容量1088。终点缺失变体不增加父族数。每个变体真实执行4历史×3续接，产生24个交叉标签，均单独计数。

13个未使用库存屋加入FIT，另复用4个旧FIT屋的新起点；旧DEV、TEST、已暴露holdout和ordinary CHECK屋排除。新起点离旧父族及本轮已接受父族至少1米，但同屋样本依然相关，不能当作独立泛化样本。状态和结果来自原SEE2检查器；角色像素阈值、RGB224、动作尺度与500总决策预算不变。UNKNOWN不当负例；STOP不增加观察。原资产不改。

本轮扩大事件正负、见证尺度、历史间隔、终点可停/需继续的实物覆盖。沿用可核验的原地转向生成器，**F动作仍为0；不能声称获得普通VLN平移恢复或自然导航数据**。所有方法以后共享该池，当前不加载Qwen、不训练。物理提案耗尽时报告不足，不按模型成绩挑样本或放宽证书。

后台由同UID/GID systemd服务推进8GPU房屋队列，半族保留失败attempt并从整族重放；完成族保留。只重试登记基础设施故障，源码/数组/证书/输入错误停止。目标/候选数有限；48小时调度上限、384 GPU会话小时、300GiB产物上限是保护上限而非完成预测。最后核验并释放本任务占位租约，恢复2–7卡原占位。配置/源码哈希变更拒绝旧run续接。

用法（项目独立Python）：

~~~bash
PY=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
export V16_STANDALONE_PYTHON="$PY"
"$PY" -I -S -B standalone.py start scale20-data-JOB -- "$PY" -I -B "$PWD/pipeline.py" --config "$PWD/PROTOCOL.json" --run-id scale_001
# 恢复用新服务名、同run-id，加 --resume。不得直接重启覆盖已有完整族。
~~~

实时网页沿用127.0.0.1:18770。COUNTS.json统计已审核父族、变体、真实执行、标签和每屋状态。ATTEMPTS.jsonl保留发现、拒绝、失败及通过；每族AUDIT.json带数组SHA、原始像素回读、事件与尺度统计；FAMILY.json引用完整轨迹。后台结尾产生DATASET.json和报告并推送必要元数据到GitHub；受许可约束的场景、RGB、语义数组及完整私有轨迹保留本地。

