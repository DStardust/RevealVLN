# GPU2 普通训练后特殊生产接续审查

只读核实时间：2026-09-10 11:19–11:21 CST。范围：旧GPU2特殊lane、batch244–247及其锁定依赖、进程/tmux和有限调度器列表。未调用GPU查询、未启动/信号任何进程、未取得GPU锁、未修改任何旧文件。本报告不构成训练或恢复运行准入。

结论：batch244在worker产生前因外部显存负载拒绝启动；245–247确实尚未尝试，具备在训练首段结束后由一个新AUTO lane调度的条件。当前没有观测到仍活跃、会自动启动GPU2的本线队列。训练与后续生产必须共享现有GPU2队列锁。

## 失败及未尝试证据

旧lane：`data_pipeline/auto_production_v1/special_gpu2_scale_v2/lane_gpu_2`。

- `RESULT.json`：240–243为`AUDITED_SEE_PER_ITEM_QUALITY`，244为`FAILED_NO_AUTOMATIC_RETRY`，245/246/247为`PENDING`，自动重试0，耗时5224.6506秒。
- 最后`EVENTS.jsonl`：244开始、child PID3631328，随后`PRODUCTION_NONZERO_NO_RETRY:1`关闭；没有245–247 reservation/start事件。
- `batch_244/run_v1/LAUNCH_RESULT.json`：`EXTERNAL_RESOURCE_LOAD`，`process_record_present=false`，`supervisor_result_present=false`，`first_real_idle_observed=false`。这次是准入失败，不是实际轨迹认证失败，也不能改为成功。
- `READINESS_SAMPLES.jsonl`原始记录：GPU UUID `GPU-be1b30d0-517b-b079-871b-de195d35a1a2`，设备1907MiB；外部PID2781015和2900192各246MiB，新增PID3631067为1380MiB。1380MiB超过原外部单进程768MiB阈值。不能根据它之后消失就重试244或放宽原guard。
- 复核时3559124（旧queue）、3631328（244 launcher）、3631067均已不在`ps`中；2781015/2900192仍为外部PointGoal评估进程，必须保留。

实际调用封存`special_scale_transport_v1/transport.py`的只读`check_inputs(batch, worker=True)`，并复核旧AUTO job哈希、逐批精确审批，结果如下。导入采用项目stdlib Python `-I -S -B`，没有运行`run_main`或`worker_main`。

| batch | run_v1已有内容 | 全锁核验 | 原job哈希/审批 | 对应历史预约 |
| --- | --- | --- | --- | --- |
| 245 | 仅EXECUTION_CONFIG.json、INPUT_LOCK.json | 577项通过 | 通过 | batch109，未尝试复核通过 |
| 246 | 同上 | 577项通过 | 通过 | batch110，未尝试复核通过 |
| 247 | 同上 | 560项通过 | 通过 | batch111，未尝试复核通过 |

九个candidate_id分别为：

```text
245: WF_MULTI_a3460384f085c1977305b2fd
     WF_MULTI_9db3f64418a4dfd74e2bbd91
     WF_MULTI_ac97f26c847762bd16a25ce4
246: WF_MULTI_dcd173260901c58ec8e18ffc
     WF_MULTI_a0302a0a1577dbc6eeed1720
     WF_MULTI_5c38f79778ef9adb07002c6f
247: WF_MULTI_f031322d1102e6b14ca31fa4
     WF_MULTI_3af60f6ec3d450afa99b3e89
     WF_MULTI_3eb7f3e8e50ab4ecd036de1f
```

这些仍为候选，不能计为9个合格族。原109–111预约不能再启动，因为已迁移给245–247。

## 与训练互斥的现有自动入口

1. 六条活跃AUTO queue固定在GPU1/3/4/5/6/7，计划没有迁移GPU2的逻辑；queue执行只筛选其启动参数指定GPU。它们继续运行，不因本报告调整。
2. GPU2只有两份已有AUTO计划：`special_gpu2_scale_v2`和更早`special_queue_v1`，两者lane均已关闭且保存失败结果。旧queue使用`lane_gpu_2.mkdir(exist_ok=False)`，不会自行恢复、循环重启或跳过失败。当前tmux无GPU2生产session。
3. 旧SFT `progress_server.py`（PID2802410）是本地18765只读仪表盘及一次CPU收口器，没有GPU2启动路径；不是新训练接口。
4. 可见root crontab及`/etc/crontab`、`/etc/cron.d`未检出本线/GPU2重启命令，`atq`为空；user timer列表只有grub-boot-success。未遍历他人私有调度配置，不能据此保证外部应用未来永远不再使用GPU2。
5. 现有GPU2队列锁为`data_pipeline/auto_production_v1/gpu_locks/gpu_2.lock`，stat为device34/inode15987212，读取`/proc/locks`未发现该inode有持锁者。此为时间点读证，未尝试取得锁；不能作为后续运行的锁替代。

新训练supervisor应取得**同一个**GPU2队列flock，并把fd传给worker，覆盖模型加载、probe、训练、checkpoint、异常清理全过程。不得另造一个互不认识的training.lock作为唯一互斥。此锁只协调遵守它的本线任务；外部进程仍要靠实时身份与显存护栏监控，不能被停止。

## 建议的最小新版本接续

推荐新namespace：`data_pipeline/auto_production_v1/special_gpu2_after_training_v3`。本轮没有创建PLAN或启动脚本。

新`PLAN.json`只包含从旧计划提取的245/246/247三个**完整原job字典**，保持command、completion、audit_command、candidate_ids、input_hashes全部原值。新的node、有限wall_seconds及训练结束前置证据由主agent另行冻结；新`MAIN_AGENT_APPROVAL.json`绑定新plan hash及既有queue.py hash，另留新恢复审查/审批记录。

这样只产生新的AUTO lane输出，并在245–247既有但从未运行的run_v1中写入首次生产产物。旧PLAN、旧失败lane、旧源码、INPUT_LOCK及逐批审批无需变化；不复制或修改封存`special_scale_transport_v1`，也不调用它的prepare重做已有batch。若要求连batch产物目录也全新，则需另建新的batch编号、迁移预约和新输入锁/审批；本报告没有实现该较大变更。

启动前必须同时成立：

- 普通训练首段已有终态、checkpoint/失败记录及只清自己进程的cleanup凭据；训练worker不再持有CUDA context，同GPU队列锁已由训练正常释放。
- 实时再核245–247 run_v1仍只有两份准备文件，旧109–111仍未尝试，逐批锁、原job哈希和审批仍通过。
- 外部GPU2 context保持原协议门槛；若再出现大外部进程，保留拒绝结果，不发送信号、不自动重试。
- 新恢复plan只含这三份未尝试预约，244和更早103等失败不入新队列；第三份或任何中途失败后仍停止本lane。
- 新queue的5400秒/批预约、4200运输上界、1200审核及原工厂60000动作/3600秒/7GiB预算不变。
- 原GPU2 lane绝对截止为**2026-09-10 21:27:41.296 CST**。本报告核实时还剩约36384秒。不能把新版本目录当作自动获得新的12小时预算；主agent冻结的新lane结束上界应落在该原截止内，或另明确登记新的有界授权。三批共有16200秒最坏预约，额外调度余量也需计算；若时间不足，只排期容得下的完整批数。

完成上述CPU冻结和主agent实际运行前核验后，唯一建议的执行形态如下。**此路径当前尚无PLAN，命令本轮没有执行，也不应直接复制立即启动。** 由主agent将其放入一次受控tmux启动，不能用shell无限重试。

```bash
cd /mnt/data_nas/deeprobotics/daiyang/vla
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -B \
  projects/qwen35_indoor_nav/data_pipeline/auto_production_v1/queue.py \
  --plan /mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/data_pipeline/auto_production_v1/special_gpu2_after_training_v3/PLAN.json \
  --gpu 2
```

`queue.py`本身会非阻塞申请现有GPU2锁、逐job核hash并保留生产/审核终态，但不认识训练cleanup收据。若要在训练结束后自动接续，应在新namespace另做一个有界一次性handoff门：先绑定训练终态与cleanup哈希、再次检查准确worker身份退出，再启动上述队列；不能仅等待GPU利用率下降或以旧PID消失为唯一条件。本报告没有实现或启动该handoff。

## 关键证据哈希

路径均相对LINE。

| 证据 | SHA256 |
| --- | --- |
| data_pipeline/auto_production_v1/special_gpu2_scale_v2/PLAN.json | 1b6e43f0cb759ac9e4a8c5c0636bef225023bbb1f88a6f22793c0a3dd20c977d |
| data_pipeline/auto_production_v1/special_gpu2_scale_v2/lane_gpu_2/RESULT.json | 6478d1bfa61111d25aec53e1e190e74c2d76175849c875063ee9cd523faad203 |
| data_pipeline/mechanism_runtime_v1/witness_first_v1/batch_execution_v1/batch_244/run_v1/LAUNCH_RESULT.json | 885f64d41bbd6538937f85eb7bf187141905e19e26026686bced7c3fbb6aaca0 |
| data_pipeline/mechanism_runtime_v1/witness_first_v1/batch_execution_v1/batch_244/run_v1/READINESS_SAMPLES.jsonl | e6db8081966b82beaf86da4790aaf40a1fb497f0ff2f1d6bfc2db78e56454fc4 |

本报告仅验证接续可行条件，不主张已恢复队列、已产生新族或已完成普通训练。
