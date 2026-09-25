# Q35N：完整 VLN 工作方案与机制验证留档

本入口保存截至 2026-09-25 的整体研究定位、真实实验与复核证据。当前恢复机制是未来“数据—模型—训练”完整 VLN 工作的一个阶段，不作为整篇论文的边界。

## 从这里阅读

1. [整体 VLN 工作方案](../../research/continuation_memory_v1/strong_backbone_recovery_v1/paper_narrative_20260925/WHOLE_VLN_PROGRAM_V1_ZH.md)：数据、模型和两阶段训练如何围绕真实执行相互支撑；均区分当前实现与未来候选。
2. [阶段性论文叙事](../../research/continuation_memory_v1/strong_backbone_recovery_v1/paper_narrative_20260925/PAPER_NARRATIVE_ZH.md)：恢复/保护子机制的已完成结果及相关工作边界。
3. [证据快照](../../research/continuation_memory_v1/strong_backbone_recovery_v1/paper_narrative_20260925/EVIDENCE_SNAPSHOT.json)：叙事引用的原结果与来源。
4. [三种子对照实现](../../research/continuation_memory_v1/strong_backbone_recovery_v1/recovery_confirmation_v2/README.md)：六模型、共享数据和预算，完整 unseen200 正在执行。
5. [新 CPU 机制诊断](../../research/continuation_memory_v1/strong_backbone_recovery_v1/mechanism_probe_v3/README.md)及[完整结果](snapshots/mechanism_probe/RESULT.json)。

## 已完成与未完成

- 已完成 discovery unseen200：原生112/200，CONCAT116/200，EVIDENCE111/200。CONCAT 救回6条、损害2条，属于单种子的已暴露开发正向信号。
- 已完成 confirmation DEV64：CONCAT 三种子35/35/33成功，LOCAL33/35/38成功；LOCAL均值更高，不主张记忆优势。DEV 是32失败恢复＋32原生成功条件，不能冒充普通 unseen。
- confirmation unseen200 仍在运行。此次不可变日志包只收录已经完成参数封存的40组；其余已记录或待运行条件不当成完整结果。不能用部分组选择种子或报告最终 SR。
- 已完成 CPU 机制诊断384/384：在固定已记录的DEV教师恢复/原生成功轨迹上，比较新增分支原读出与零状态读出。没有新导航SR、GPU使用或参数更新。

CPU诊断中，CONCAT三种子的完整记忆相对零记忆，在488个恢复动作标签上的净正确数变化为+11/+9/−4。它证明了读出依赖并显示作用不稳定；零状态可能离开训练分布，输入梯度不是长期记忆有效性的因果证明。LOCAL的状态只含当前观测，其零状态对照不能解释成删除历史。

## 复核文件

- [ROLLOUTS.csv](ROLLOUTS.csv)：1520次实际完成的执行，来自 discovery DEV64×3、unseen200×3，confirmation DEV64×7及本次封存 unseen40×7；不同实验分母分列。
- [ARCHIVE_STATE.json](ARCHIVE_STATE.json)：不可变会话选择、时间、文件数和日志包SHA。
- [SEALED_LOGS.tar.gz](SEALED_LOGS.tar.gz)：相对仓库路径保存的完整逐步TRACE、逐组COMPLETE、运行身份及STATE_SEAL。解压后可以按CSV中的trace_path定位；所有纳入轨迹SHA已核对。
- `snapshots/action/`：完整发现轮协议、审计与结果。
- `snapshots/confirmation/`：当前对照协议、完整DEV结果和进行中状态快照。快照不会自动变成最终结果。
- `snapshots/mechanism_probe/`：新的只读机制诊断协议、结果及全部逐轨迹输出。
- [MODEL_ASSETS.json](MODEL_ASSETS.json)：八个最终纠错头的路径与SHA。张量权重、缓存特征、原始RGB和场景未放入Git，本入口不声称单靠公开仓库即可运行全部环境。
- `BASE_ASSET_PROVENANCE.json`：底模/公开代码资产来源（若来源文件存在）。

## 独立进程

导航对照服务：`q35n-strong-confirmation-20260925-01.service`。
CPU诊断服务：`q35n-strong-mechanism-cpu-20260925-01.service`，已完成。
用户服务器现有监控：`http://127.0.0.1:18770/`，需原SSH转发；它不是公网部署地址。

此次只提交范围内代码、方案与证据到独立留档分支。未修改旧失败结论、未中止当前评测、未改动当前实验的源码锁和权重。最新完整结果应在运行结束后追加新的留档提交。
