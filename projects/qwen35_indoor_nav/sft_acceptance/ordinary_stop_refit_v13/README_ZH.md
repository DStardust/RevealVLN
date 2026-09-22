普通导航第一项修复：只训练STOP输出行，保留运动能力；真实闭环比较原生与候选100对。不是已获得SR收益。新协议允许记录跨会话浮点差异，旧V12严格数值FAIL只读保留。

运行：独立standalone服务执行 pipeline.py --run-id stop_001；训练结果在runs/stop_001/TRAIN_RESULT.json，逐对结果在sessions，最终REVIEW.json。不改变或重新启动任何旧训练。
