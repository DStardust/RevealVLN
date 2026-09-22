# 普通导航 STOP 行训练与闭环检验

VALID_COMPLETE，完整 100/100 对。

本轮原生 A 成功 21.0，候选 B 成功 25.0；新增成功 10，丢失成功 6。完整 ΔSR：0.04。缺项不从计划分母删除。

仅 STOP 行在 16 个 FIT 屋的真实状态上重新训练；运动输出行、Qwen/LoRA、观察与评测不变。12/4 屋离线诊断不等于闭环收益；缓存跨会话数值差异见各 CACHE_LIVE_PARITY.json，旧 V12 FAIL 未改写。

当前不自动部署；完整 100 对才判断开发信号。没有独立测试泛化、完整 val_unseen SR 或新算法贡献。若失败，不改成功定义、参数或挑选重跑。

逐对轨迹在 sessions/；训练记录 TRAIN_RESULT.json；完整机器可读表 REVIEW.json。
