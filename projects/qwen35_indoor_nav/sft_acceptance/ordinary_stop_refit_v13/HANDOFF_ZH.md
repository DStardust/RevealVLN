# 当前任务：先修普通导航

普通导航的新工作在本目录；原 EXPANDED 100k 记忆长训保持运行。当前没有普通 SR 提升结论。

已完成：在旧 V12 的 16 个 FIT 屋、64 条真实轨迹、5487 个去重状态上完成 STOP 行训练（2049 参数，CPU float64 LBFGS）；不是只写方案。12 屋拟合 / 4 屋诊断，停车精确率 18.07%→27.05%，召回率 2.29%→10.02%，BCE .988→.551。这个离线结果不等于闭环 SR。最终固定配方在全部 16 屋拟合后测试；未调 lambda 或看导航结果换参数。Qwen/LoRA 与三个运动输出行不变。

当前独立服务：q35n-ordinary-stop13-20260922-03.service；run：runs/stop_002。GPU1/2/3并行，各卡约一个常驻模型，8GiB模型上限；每条A+B同进程。100对/200次导航，原始指令、2帧224RGB、8个实际动作、500决策、主动STOP且距离<3m不变；不会用旧SR21%替代本轮A。训练候选不是循环控制器，因此候选自身的4类argmax定义执行动作，不强制原模型的STOP。

网页沿用原18770端口，新增顶部“普通导航”卡片；/api/ordinary与原/api/status均可访问。新监控服务q35n-ordinary-stop13-monitor-20260922-01.service，只替换监控，没有停止记忆训练。

首个提交因 SOURCE_LOCK 未生成而在0.17秒启动前退出；第二次已加载三次真实模型，各测3个固定FIT输入，随后因当前checkout缺少Habitat-Lab相对路径在首次reset前退出，0个导航决策。旧run_001、日志、gold诊断、源码快照保留。修复只改为读取原资产根目录下逐字节相同的旧executor和官方测距实现，已锁SHA；候选权重原样复制到stop_002，没有重新训练/挑结果。

旧V12 FEATURE_PARITY FAIL原样保留。新协议核验缓存head重构（最大差5.72e-6）、每会话3个固定FIT现场差异，以及两臂相同输入前缀上的底模动作/数值一致性。所有原始/处理后输入、实际动作、终态和指标可复核。无差异的pair必须相同终态；有差异前输入和底模动作必须匹配；不复制logits制造一致。

查看：
```bash
systemctl status q35n-ordinary-stop13-20260922-03.service
cat projects/qwen35_indoor_nav/sft_acceptance/ordinary_stop_refit_v13/runs/stop_002/STATUS.json
```
完整结果由pipeline自动写REVIEW.json和REPORT_ZH.md；不需Codex在线。训练与测试结束后不自动部署、不以未知/缺失替代失败、不改旧结果。若有源码/数值正确性失败，保留首分歧停止，不因低分重试。

本轮先用最小已找到的未完成修复得到真实闭环反馈，不重复已失败的全池延训、R2R-only低学习率延训或KL配方。不将只训练STOP行描述为全模型重训或新算法贡献。
