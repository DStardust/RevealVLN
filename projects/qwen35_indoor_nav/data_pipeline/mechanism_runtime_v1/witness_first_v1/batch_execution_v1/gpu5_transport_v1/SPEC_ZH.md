# GPU5 transport V1：固定占位借还与原 winding batch

状态：CPU_READY_FOR_MAIN_AGENT_REVIEW。17 项 CPU 测试通过，本节点未执行 GPU
查询、进程信号、tmux 操作、仿真或训练。唯一运行者为主 agent。

方法、候选及门槛不变。新 prepare 只做四个计数严格为 1 的源码替换：固定 GPU5、
对应 UUID、配置加入运输授权/身份路径、输入锁加入运输代码及授权。保留老 prepare
源码及其 SHA256，不修改其 globals 来绕过 GPU 限制。新运行复用原 shared worker，
固定 winding_v1；监督 3900 秒、factory 3600 秒、60000 动作、原 check_gpu/RSS/
磁盘门槛完全不变。首次真实 idle 最多等待 60 秒，复用封存 readiness_v1。

prepare 命令参数：`--snapshot`、`--name batch_02`、`--indices 0 1 2`、
`--authorization`、`--holder-identity`。快照、选择、GPU、预算和身份文件路径必须与
`authorizations/WITNESS_CROSS_HOUSE_BATCH02_GPU5_V1.json` 完全对应；会打印需要
主 agent 写入 batch 目录的 `MAIN_AGENT_GPU5_APPROVAL.json` 内容，审批绑定最终
INPUT_LOCK 和 holder identity 的 SHA256。没有该单独审批不能借卡。

固定 GPU5 UUID `GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b`，仅借主 agent 冻结的
PID3133144 / starttime131277976 / UID0 / 项目根 cwd / 原命令与 argv / tmux
`vla_idle_occupancy_20260904:2.0` pane `%150`。借前完整身份核验，持久写入
LEASE_BEFORE（原配置、实测 GPU、原 remain-on-exit、恢复命令），信号前再次连续
核验身份。仅一次 exact PID SIGTERM，20 秒未退出不升级 SIGKILL。

本机 stdlib 无 pidfd；双重 /proc 校验与 exact PID 信号之间仍存在微小竞态，不能
宣称原子身份绑定。未知进程或身份变化拒绝执行。GPU 借前占位本身不适用导航
worker 4GiB 上限；借前单独核实占位 >20000MiB、所有其他 context ≤768MiB、合计
<1024MiB。借出后所有实测快照执行原 check_gpu，不伪造、扣减或修改其值。

outer finally 覆盖成功、异常及 SIGTERM/SIGINT，清理期间忽略第二次交互信号。
原 supervisor 首先清理自己的 worker；新层只额外持有该 supervisor 本实例创建
的唯一 Popen 对象，用于原 finally 异常时的同组 TERM/20秒/KILL/10秒有界兜底。
不会根据任意外部 PID 清理进程。确认原 worker 不在、pane 仍是自身的旧 dead pane、
实测 GPU <1024MiB 后，以不带 `-k` 的 respawn-pane 恢复相同命令/cwd。未知活 pane
即拒绝，绝不强杀。恢复后最多 30 秒实测正确新进程占用 >20000MiB，且恢复原
remain-on-exit；结果独立写 RESTORATION 和 LEASE_RESULT。

若 GPU 被外部任务新占用、pane 被改变、tmux/文件系统失效、或 SIGKILL/主机断电，
不能保证自动恢复。失败会明确标记 manual recovery，不以强杀外部任务来制造成功。
原 SUPERVISOR_RESULT 描述其自身 worker 清理，不将 outer 借卡伪写为原 supervisor
行为；完整借还证据要另外审查 LEASE_RESULT/RESTORATION。

测试覆盖正/负正常恢复、SIGTERM、preworker错误、错误身份、信号失败、退出超时、
未知 pane、外部高占用、原 worker 清理、清理失败、真实占位观察门槛、固定替换、
原 guard/source 未变、实际身份 schema 与实际授权范围。没有物理或科学效果主张。

旧 ordinary recovery_v4 借还逻辑仅作为只读工程参考（最终输入锁记录其源），不复用
其旧 PID、旧结果或旧准入。成功恢复后的新 holder PID 会变，本固定版本不得直接
复用到下一批；下一批需新的主 agent 身份冻结与版本化授权。
