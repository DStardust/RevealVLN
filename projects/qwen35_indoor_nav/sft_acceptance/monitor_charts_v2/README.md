# 近期训练质量视图

只读导入封存monitor_charts_v1，在相同18766端口增加近期约1000动作的准确率/永远前进基线、分动作recall/目标/预测与覆盖不足警示。数据来自训练日志累计矩阵之差，不是固定面板或DEV评估，不把累计loss下降当导航收益。

已获准GPU3/4/5让卡的队列显示DRAINING_CURRENT_BATCH_AND_AUDIT；只有原当前批次完成、GPU清理、占位恢复和强审自然结束后才能按新授权借卡。页面不发送进程信号。

旧监控所有读取上限、loopback、SSH及只读HTTP限制不变。服务可用命令为项目stdlib Python -I -S -B server.py；仅主agent在验证新版本后切换，不重复启动。CODE_SEAL同时绑定本地HTML/脚本和quality_diagnostics依赖。
