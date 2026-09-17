# 43 屋固定新 hub 采集运输 V1

仅沿已冻结 `new_hub_scale_v1/snapshot_v1/SCOUT_QUEUE.json` 的 43 FIT 屋、172 官方位置采集；无新算法、无训练、无导航收益声明。每屋固定四点、不能补位；同屋同物理 hub 的语义变体相关，候选不得计完整族。

`common.py/adapters.py` 从已封存 scout exact-source 派生，仅新输出路径、GPU7 UUID 与已单独审核的 active conservative telemetry。原像素/连续帧/动作闭合/对象身份规则、原全部组件轨迹不变；原 idle/final/restore guard 不经 telemetry。原硬上限仍 40000 动作、2700 工厂秒、2400 discovery 秒、3000 supervisor 秒、6 GiB 内容、7 GiB 总盘、8 GiB RAM。冻结来源中旧的 4500/8GiB/2hub 描述在新 PREPARED_CONFIG 明确改为实际硬上限 3000/7GiB/4hub，旧文件不改。

`prepare.py --authorization <主 agent 写入的授权>` 仅 CPU：要求 holder 节点已封存 SHA、完整依赖、初始精确身份、逐 job 来源锁。生成独立 jobs/scout_NNN，不能覆盖。不会写 MAIN_AGENT_APPROVAL，不会启动 GPU。每 job 准入值由 `runtime.approval_value(job)` 返回。首次身份必须在前置 certification lane 结束后由 main 重新核实；其余 job 仅继承此队列直接上一 job 的成功 RESTORATION，任一中断不跳过、不重试。

`runtime.py --job <job>` 主 agent 唯一运行：使用与 certification 完全相同的 `special_scale_holder_v1/locks/gpu_7.lock`，持锁覆盖借卡至恢复；完整身份二次核验与 finally 清理/恢复复用已封存 holder API。无 pidfd，保留原精确 PID 核验残余竞态。借卡失败、恢复失败或未知进程均停止。GPU7 不可同时执行认证队列和 scout。

恢复成功后释放 GPU 锁，主队列通过独立 `audit_command` 自动调用 `audit.py --job` 建库；该命令再次验证闭合，超时保持失败/partial，不重试。max_seconds5100=transport3300+CPU1800，匹配已封存调度器的审核预算上限；这是执行上限，不声称最差供给一定在1800秒完成。lane 总墙钟43200，期满只保留未运行队尾，不假定43屋能全部跑完。

`bank.py --job <job> --check-only` 只读验证自然闭合、完整来源 SHA、事前配置/许可、STORE close audit、journal/trace 原银行检查、GPU清理及恢复，并逐四位置给终态。允许实际1–4 hub；未可达点必须明确拒绝，任何 resource-censored house 不建库、不得替换位置。0hub不生成银行。`bank.py` 不带 check-only 调用原完整银行算法（每hub最多24规范化不同 A/B+T 程序，同160 query与计数门槛），仅有限词面规范化，输出 bank/language_ready_v1/{CONFIG_DRAFT,NEXT12,SOURCE_LOCK,result}.json。所有候选仍需将来的27真实 replay/18cell/54求值和强质量审计，不能直接训练。

授权必须含：approved=true；node=Q35N_NEW_HUB_SCALE_RUNTIME_V1；gpu=7及UUID；queue_path/queue_sha256；按原顺序的唯一 job_ids（1..43）；budget 原字典；supervision_wall_seconds=3000；lane_wall_seconds=43200；shared_gpu_lock 固定路径；training_allowed=false；no_unregistered_retry=true；initial_holder_identity_path/sha256。授权是 main 的新任务，不可从旧 cert 授权直接推断。

验收范围：CPU源差分、真实旧闭合源的4hub验证、显式1hub测试副本、删终态/删geometry/无解释掉点/资源截断/换点拒绝、配置结构不变、零GPU导入与运行调用顺序。测试副本不计数据。未在本节点执行仿真、生成新内容或获得新数据族。
