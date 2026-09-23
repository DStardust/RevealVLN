# 普通导航动作监督对照

本目录是真实执行入口，不是方法获益声明。前置 V15 完成 640 条真实训练轨迹、STOP-only 训练以及 DEV100/公开 unseen1839 后，自动执行同池运动监督对照。独立 systemd 服务不依赖 Codex 在线；结果低分不会自动换数据或参数。

两臂共享 best4k 底模、因果窗口、传感器、真实数据池和公共指标。A 为 V15 STOP-only；B 额外利用已执行教师轨迹中距离≥3m的 F/L/R 标签，并拟合全部四个输出行。这个比较包含运动行可训练与额外 CE 的共同作用，不分别声称两者的独立因果效应。没有扩展 LoRA、底模、历史或记忆结构。输出始终四类最终 logits 的 argmax，主动 STOP 且距离<3m才成功，500决策包括STOP。

训练采用与V15相同的按屋/轨迹归一化STOP权重，运动权重在合格教师轨迹上单独归一化；运动标签冲突掩码排除。32/8训练屋诊断与40屋最终拟合，不使用DEV/TEST调参。CPU冻结特征拟合每次最多200迭代/250损失计算，记录真实梯度、更新量与保存后FP32输出。CPU测试通过不代表闭环收益。

随后固定DEV100和完整unseen1839，各自一对A+B在同一模型进程完成。完整pair原子封存，部分结果分母清楚标记。输入和原生决策前缀不一致即停止；有限浮点位差单列。模型指纹与trace SHA可复核。运行失败保留attempt和完成pair，需要核实基础设施问题后显式resume，不自动重试低分pair。

最后另行收集最多64条训练屋恢复轨迹：按V15训练策略的过早STOP或连续三次前进碰撞，逐屋确定性选择；清单在本版本评测前冻结。真实回放源历史，到故障决策点才让私有路线教师恢复，没有中途传送；实际动作写回历史，恢复和前缀共用500预算。该批数据不混入本次训练，不报告为策略SR，不自动再训练。

运行上限：GPU阶段24小时墙钟/48 GPU会话小时，训练前等待V15最多25小时；设备1–7顺序绑定UUID，每次进程加载底模一次。共享设备监测整卡余量与自身RSS；仅通过已核验占位服务租约释放自己的占位进程。V15未结束时V16不使用GPU。产物上限400GiB；模型显存上限8GiB、进程组RSS64GiB。没有承诺完成时间或SR提升。

监控复用原地址18770，同时展示V15、V16、独立恢复采集和已有100k记忆训练历史。评测均是已有暴露公开/开发数据，不是盲测证据。

```bash
BASE=/mnt/data_nas/deeprobotics/daiyang/vla
PY="$BASE/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3"
V16="$(git rev-parse --show-toplevel)/projects/qwen35_indoor_nav/sft_acceptance/ordinary_action_repair_v16"
V16_STANDALONE_PYTHON="$PY" "$PY" -I -S -B "$V16/standalone.py" start ordinary-motion16-20260923-01 -- "$PY" -I -S -B "$V16/pipeline.py" --run-id motion_001
cat "$V16/runs/motion_001/STATUS.json"
```

恢复使用新服务名、相同run-id和`--resume`；先确认错误已解决且锁定源码/数据身份可核验。不存在对未完成中间权重的自动准入。最终结果位于`runs/motion_001/{RESULT.json,REPORT_ZH.md}`，未产生前不能声称已经完成评测。
