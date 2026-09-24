# Q35N 当前进度与完整日志快照 · 2026-09-24

**这是北京时间 2026-09-24 11:30:16 的一次固定快照，后台测试仍在继续。不是最终 1839 对结果。** 本次只做 CPU 日志封存与 GitHub 发布，不停止服务、不启动新训练，也不要求 Codex 持续在线查看。

|比较|已完成 / 计划完整对|A：V13 成功数 / 已完成|B 成功数 / 已完成|当前配对 ΔSR|B 赢 / 输|
|---|---:|---:|---:|---:|---:|
|V20，INTERNAL_DEV|100 / 100|25 / 100（25.00%）|26 / 100（26.00%）|+1.00 个百分点|3 / 2|
|V20，公开 val_unseen|360 / 1839|68 / 360（18.89%）|69 / 360（19.17%）|+0.28 个百分点|11 / 10|
|最终 100k 记忆模型，公开 val_unseen|1538 / 1839|265 / 1538（17.23%）|318 / 1538（20.68%）|+3.45 个百分点|214 / 161|

V20 尚缺 1479 对，100k 尚缺 301 对。部分分数仅描述快照中已经完整封存的配对；未完成项全部列入计划分母，没有用旧 V13 分数代替本轮 A。100k 的部分结果有正向信号，尚不代表完整集结论或可以自动采用。V20 暂时接近持平。两者完成后均由各自原流水线汇总，不按分数挑选重试。

完整 DEV100 的 V20 SPL 为 0.21546→0.23314，nDTW 为 0.43925→0.45697，OSR 为 0.36→0.33；通过原注册 DEV SR/SPL 门槛后进入公开集。本快照 100k 的 SPL 为 0.15561→0.18353，nDTW 为 0.43106→0.45881，平均动作数为 93.99→60.69。这些仍是部分分母描述。按屋、暴露分层、胜负 episode、待补项及全计划分母识别界见下列机器可读文件；识别界不是置信区间。

## 从这里复核

- [SNAPSHOT.json](SNAPSHOT.json)：快照时间、三个分母、逐屋与暴露分层结果、配对审计和限制。
- [ROLLOUTS.csv](ROLLOUTS.csv)：3778 个计划配对槽位，包含全部未完成项；1998 对已完成。每对列出 SR/SPL/nDTW/OSR、动作、碰撞、审计字段及源 SHA。
- [PAIR_INDEX.json](PAIR_INDEX.json)：全部 1998 个原子 PAIR 封存文件的路径与 SHA256。
- [TRACE_ARCHIVES.json](TRACE_ARCHIVES.json)、[TRACE_INDEX.json](TRACE_INDEX.json)、[traces/](traces/)：21 个独立 tar.gz 包，共 147,902,715 字节；原始 PAIR、双方逐步动作/输入哈希、私有轨迹、初始状态和终态日志共 29,970 个文件。导出时逐项核验原 PAIR 中登记的日志 SHA。
- [CAPTURE_INDEX.json](CAPTURE_INDEX.json)、[captured/](captured/)：运行协议、源码锁、加载身份、CPU 结果、训练与资源日志、失败与重启记录、服务状态的时间快照；每个文件有独立采集时间，不把可能滞后的 LIVE_RESULT 当作封存分母。
- [INCOMPLETE_PAIR_ATTEMPTS.json](INCOMPLETE_PAIR_ATTEMPTS.json)：截点时未封存完整对的目录记录。半对不参与统计，原始在途文件仍留在服务器。
- [PUBLICATION_TEST_RESULT.json](PUBLICATION_TEST_RESULT.json)：CPU 发布完整性验收。它不是新模型测试或新的收益证明。
- [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json)：本交付包的内容索引与 SHA。

## 本轮实际代码与修复

代码起点提交为 `b87637a0ab27e5b77e1bd849fa0eab2e7699771b`；本次发布将此未推送提交及本快照一起推送。V20 与 100k 原启动实现见前一提交 `e51b26cdbfbc438fe0f66b9a386fa51c94bc545f`。发布不是重新训练或修改策略。

- [V20 实现](../../sft_acceptance/ordinary_stop_contrast_v20/)：同指令终点/困难负例训练，2049 个 STOP 参数；V13 运动行和 best4k 底模冻结。单次 CPU 拟合后，GPU0/1 运行固定 DEV→公开集。
- [最终 100k 迁移实现](../../closed_loop_bench/ordinary_long100k_transfer_v1/)：GPU2–7；A=V13，B=best4k 加原 EXPANDED/MONOTONIC/seed1209 的最终 100000 步记忆。没有重新训练，没有与 V13 STOP 行混装，也没有从 20 个旧 checkpoint 里按本轮分数择优。
- [100k 资源恢复](../../closed_loop_bench/ordinary_long100k_transfer_v1/ops_recovery_r1/)：旧磁盘全树扫描超过 120 秒导致 `DISK_SCAN_DEADLINE` 停止。已换成独立 CPU 扫描、显式 session/attempt 计时，避免阻塞健康检查；保留 1290 个原封存对及其哈希，重启补齐剩余对。策略、模型、输入、动作、成功条件和分母均未改变。
- [V20 有界资源恢复守护](../../sft_acceptance/ordinary_stop_contrast_v20/ops_recovery_r1/)：不打断活跃测试，只在同一种磁盘扫描超时且原服务已退出时自动提交一次恢复；不读取分数决定是否重试，不重试其他正确性错误。
- [1290 对固定诊断](../../closed_loop_bench/ordinary_long100k_transfer_v1/analysis_snapshot_1290/)：旧快照的 140 个退步 episode 中，139 个首次分叉是运动动作变化。首次分叉位置是定位证据，不证明该单步独立造成结局。此诊断没有触发新训练或使用测试结果构造路由策略。
- [前一轮 V19 终态](../../sft_acceptance/ordinary_stop_calibrated_v19/runs/calibrated_001/RESULT.json)：DEV100 为 25%→22%，未进入新公开集，未采用；旧失败记录保留。

本快照 1998 对的记录均为输入前缀一致、底模动作前缀一致、底模 logits 差 0、状态不变。发布脚本核验了对应封存字节，不重新调用模型验证这些字段。最终实验仍以原流水线的完整复核为准。

100k 的公开集与其 head 训练有 6 屋 / 873 条重合；另有记忆开发暴露 2 屋 / 564 条；未列出此类重合的 3 屋 / 402 条也不能称为盲测。详见 [EXPOSURE_AUDIT.json](../../closed_loop_bench/ordinary_long100k_transfer_v1/EXPOSURE_AUDIT.json)。当前没有新的 SR40、论文泛化或真机部署结论。

## 后台与实时进度

继续使用原来的 **18770 端口监控网站**，无需 Codex 在线。GitHub 此目录是固定快照，不自动随网页刷新。

```bash
systemctl status q35n-ordinary-contrast20-20260924-02.service
systemctl status q35n-ordinary-long100k-unseen-20260924-03.service
systemctl status q35n-ordinary-contrast20-resource-guard-20260924.service
curl -s http://127.0.0.1:18770/api/status
```

如果 V20 因登记的磁盘扫描错误恢复，新服务名是 `q35n-ordinary-contrast20-20260924-resource-r1.service`。本目录的 captured 状态只反映采集时刻，实时状态读服务器 run/STATUS.json。各服务自行保存进度、汇总和退出，并按既有租借机制恢复用户占位；本次发布不操作 GPU 进程或占位。

本地实时文件（相对项目根）：

```text
sft_acceptance/ordinary_stop_contrast_v20/runs/contrast_001/STATUS.json
closed_loop_bench/ordinary_long100k_transfer_v1/runs/transfer_001/STATUS.json
```

## 下载后的 CPU 验证与资产边界

在本目录执行 `python3 verify_snapshot.py` 可独立验证归档、日志 SHA、CSV/原 PAIR 计数、主动 STOP 与预算字段。无需模型、GPU 或 Habitat。若要解包，请解压到新的空目录，**不要直接覆盖正在运行的工作区**。

所有已封存对的上述 JSON/逐步日志已打包。原始 RGB、场景/语义资产、约 312 MB V20 特征缓存及底模大权重不重复上传；其本地来源和身份见源码锁、数据绑定和运行身份。代码、轻量候选权重及最终 100k HEAD 已在此分支的源路径提交。缺少这些本地大资产时可以复核日志，不能仅凭 GitHub 克隆声称重新运行了导航。

本次回交后停止 Codex 持续查看，让已有独立服务完成剩余清单。后续决策先读完整结果与按暴露分层表现，不把本快照的正差升级为稳健泛化，也不因部分分数低提前更换模型或清单。
