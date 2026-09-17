# Batch01 只读进展与失败诊断

本目录不操作GPU/进程、不修改active代码/config/guard，也不代替Newton的正式质量审核。

## 首次观察

`snapshot_v1`时，batch01只有EXECUTION_CONFIG、INPUT_LOCK和SUPERVISOR.lock；没有GPU_BEFORE、PROCESS、worker.log、PROGRESS、bundle或trace，`ps`也未观察到对应run/worker进程。因此当前三候选物理进展均为0，不能报告已进入真实生成、spin失败或history失败。

三候选冻结计数计划回读一致：

- 17DRP hub0：F104/L64/R40，208动作，需L24中性验证。
- 1pXnu hub1：F30/L39/R39，108动作，无整圈padding。
- 29hnd4 hub1：F48/L55/R31，134动作，需L24中性验证。

原supervisor先获取SUPERVISOR.lock，再在写GPU_BEFORE前强制GPU utilization=0；此断言位于运行try/finally之外。启动瞬间其他renderer非零利用率可能导致直接退出、没有SUPERVISOR_RESULT。本次没有launch stdout，故只能提出此明确可核查的启动门槛原因，不能冒充已证实GPU_NOT_IDLE异常。已通知主agent核对launch返回；没有自行重启或改护栏。

另有独立语言质量风险：首候选terminal房间码tv直接进入“see the couch in the tv”，而非TV room。物理/程序标签可能仍正确，正式自然语言合格性应由独立quality处理；此处不修改active或自动授予训练质量通过。

## 后续只读采样

用项目stdlib Python运行：

```text
python -I -S -B .../batch01_diagnostic_v1/inspect.py --output .../batch01_diagnostic_v1/snapshot_v2
```

新目录不覆盖旧快照。脚本只读取稳定JSON，正在写入/读取期间变化的文件保留为未判。它核对实际F/L/R/S、源trace自哈希、原Compiler两帧事件、整圈/组件/历史动作匹配、端点agent及sensor偏差；对完整匹配的矩阵轨迹复算原checker标签和query长度。脚本输出是局部读证诊断，不是完整三种子、独立质量或科学收益认证。
