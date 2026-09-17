# GPU5 transport V2：退出窗口与 pane 恢复修正

CPU_READY_FOR_MAIN_AGENT_REVIEW；24 项 CPU 测试通过。未操作 GPU/进程/tmux。
不改 V1、失败 batch_02、任何旧封存源码。V1 实际失败和主 agent 人工恢复另存
`BATCH02_MAIN_RECOVERY_RECEIPT_V1.json`，本版本输入锁包含该记录与 V1 哈希清单。

原故障：SIGTERM 后 `/proc/PID/stat` 尚在，但 `cwd` 已消失；V1 等待和 finally
仍完整读取身份而异常，随后将 remain-on-exit 恢复 off 删除了 dead pane。
V2 的退出等待仅解析 stat 的 starttime/state；不存在或 Z/X 为终态，PID复用拒绝。
finally 如完整身份遇 ENOENT，按退出中继续有界等待，不据此判为未知进程。

恢复前至多20秒等待原 pane_dead，与旧pane ID/PID及进程终态一起核验。失败时
remain-on-exit 保持 on，只有确实恢复原占位且 pane 存活才恢复原选项。respawn
不带 -k，未知活pane绝不杀。holder停止仍一次exactPID SIGTERM，不升级kill。

退出终态与 GPU context 消退可能不同步：执行前和恢复前各最多20秒保存原始
GPU_DRAIN真实快照，等待旧holder context消失且总显存<1024MiB，随后执行原
check_gpu。只在这个已核实退出holder的过渡期单独检查其余context，所有其余
PID仍每个≤768MiB且合计<1024MiB；不修改快照，不伪造空闲。随后首次真实idle
复用readiness_v1的60秒等待。原worker 3900/3600秒预算与全部物理质量门槛不变。

固定新PID3210151 / starttime131709588 / UID0 / pane%235 / 原target2.0及原命令。
授权和身份来自 `WITNESS_CROSS_HOUSE_BATCH02R1_GPU5_V1.json` 与
`WITNESS_BATCH02R1_GPU5_HOLDER_IDENTITY_V1.json`。同snapshot、indices0/1/2、
同winding算法；新batch名batch_02r1。prepare接口与V1相同，独立MAIN_AGENT_GPU5_APPROVAL
绑定新INPUT_LOCK与身份哈希后才能运行。仅主agent执行。

outer finally先让原supervisor清理自己的worker，并用本实例唯一Popen句柄作
有界兜底，再恢复占位。忽略清理期间第二次SIGTERM/SIGINT。没有权限停止外部任务。
SIGKILL、掉电、外部抢占、tmux失效仍可能要求人工恢复；pidfd不可用，身份核验与
exactPID信号之间的极小竞态不能声称完全消除。恢复失败留独立证据且保持pane on。

24项CPU覆盖V1已有17项及新故障：stat尚存/cwd已消失、退出窗口中断、pane死亡
延迟、恢复失败保留on、pane死亡超时保留on、PID复用拒绝、退出GPU context延迟。
未产生仿真或科学验证结果，`scientific_pass=false`。bulk源扩量准备暂停等待本次接收。
