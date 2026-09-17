# 普通导航训练已启动：当前运行入口

实读验收时间：2026-09-10 12:01:18 CST。本文是启动快照，不替代后续实时PROGRESS/RESULT。用户最新已要求开始普通训练、必要重启及远程监控，覆盖交接文档中历史“无训练授权”的本任务部分。旧交接、主线、源码和输入锁均保留。

当前节点：`sft_acceptance/ordinary_execution_v6`。训练tmux `q35n_ordinary_train_v6`，worker PID3710394（只作快照，操作前必须复核ticks/argv/cwd/UUID）；监控tmux `q35n_ordinary_monitor_v6`。

验收时已累计14次正式优化更新、420条动作决策，在线平均CE约2.031，恢复后吞吐约1.76决策/秒。第10次更新的真实checkpoint已写盘并校验哈希。模型仍在训练，未完成一个epoch，未执行独立房屋闭环导航，不宣称基本导航能力已成功。

## 远程接口

在自己的电脑保持SSH隧道，使用实际账户/SSH主机别名，必要时沿用跳板配置：

```bash
ssh -N -L 18766:127.0.0.1:18766 <用户名>@hpda-jushendaohang-005
```

浏览器打开 `http://127.0.0.1:18766/`；JSON为 `/api/status`；健康检查为 `/healthz`。只绑定loopback，经SSH认证访问，不公开无认证HTTP控制端口。主agent实测GET200、POST405、未授权文件路径404；页面每10秒刷新loss/更新数/动作统计/STOP召回/GPU/检查点/更新时间。原18765服务未动。

## 输入、训练和实际边界

- 仅已冻结审核的51 FIT屋、5849物理路线、17548官方人工指令、1394744指令条件动作；不加入EnvDrop、特殊机制样本、DEV或CONFIRM。
- Qwen3.5-2B原始预训练基座，rank8 LoRA、8执行记忆槽、冻结视觉塔；普通未加权动作CE、AdamW1e-4、4步TBPTT、8chunk累计。完整有序走指令路线并保留STOP，固定seed1109，不改变标签或未来信息隔离。
- 本次接续最多86100秒，累计更新上限仍10000、训练决策软上限约320000；时间/数量先到先停止并保存。不自动无限重试或开启后续无界段。每100更新保存checkpoint，另已在启动检查点10保存。
- 现阶段单卡约1.6–1.8决策/秒，一整轮139万动作约9–10天。因此首段24小时只覆盖部分数据，不是全池3epoch已经启动完成；后续应在获得有序让卡许可、验证多卡接口后扩卡/加速。

## 已完成的真实验收与恢复

最长实际805-token输入四步反向通过；旧/新前向差为0；真实STOP动作头梯度非零有限；模型及优化器重载、后续更新、中途非零记忆/cursor恢复后的下一chunk均精确一致。序列、累计token/decision、wall超限拒绝已实际检查。probe参数更新全部丢弃，不计作正式训练收益。

V1的UUID前缀失败、V2/V3的更新重复性失败、V4的外部资源截断全部保留。修复只在新版本：规范化物理UUID后仍精确比对；每次restore复制载入state，避免CPU AdamW step引用污染；启用PyTorch/CuBLAS确定性执行，未降低exact门槛。57项CPU/接口相关测试通过，根历史注册表check通过；真实GPU证据另存PROBE.json，不能用CPU通过替代。

V5真实训练6更新/158决策后收到来源未记录的SIGTERM，正常保存checkpoint_000000006.pt；其supervisor随后因NVML查询5秒超时未能确认cleanup，旧失败结果不改。后续只读复核确认旧PID及GPU context消失、GPU2空闲，V6先再次通过真实probe，再从该checkpoint精确恢复参数、优化器、RNG、数据位置及记忆；已核实继续第3条指令而非重做前158决策。新初始checkpoint为6，随后新checkpoint为10。

## GPU与生产现状

训练仅GPU2，和原生产共用同一个gpu_2.lock并传给worker。新有界分区为自身26GiB、外部合计5GiB/单进程4.5GiB，两分区总和必须低于实际显存至少512MiB；每10秒持久化原始GPU记录再检查，越界仅清理自己的worker。GPU2没有借用占位，不恢复不存在的holder。

本轮未停止任何健康生产队列或真实外部任务。验收时五条生产队列仍在：GPU3/4/5/7特殊认证、GPU6 EnvDrop。GPU2旧244失败在本轮开始前已发生；GPU1 batch225随后因外部GPU1负载（实录外部2701MiB）自行停止，226–228未尝试。GPU1/2失败与预约保留，不原地重启旧lane、不争卡；后续接续必须新有限排期，不能以新目录重置旧授权截止。

所有已重启的监控均仅为本轮自己创建的版本化服务，精确PID/starttime/argv/cwd核验记录在各节点MONITOR_RESTART.json。GPU0未使用，其他GPU的holder身份链及冻结生产源码/锁不变。

## 权威运行文件

- [启动验收与实读快照](sft_acceptance/ordinary_execution_v6/START_ACCEPTANCE.json)
- [真实GPU验收](sft_acceptance/ordinary_execution_v6/PROBE.json)
- [真实断点恢复](sft_acceptance/ordinary_execution_v6/run_0001/RESUME_RECEIPT.json)
- [实时进度](sft_acceptance/ordinary_execution_v6/run_0001/PROGRESS.json)
- [监督进程状态](sft_acceptance/ordinary_execution_v6/STATUS.json)
- [第10步检查点收据](sft_acceptance/ordinary_execution_v6/run_0001/checkpoint_000000010.pt.json)
- [冻结执行协议](sft_acceptance/ordinary_execution_v6/PROTOCOL.json)

不要重复执行已启动的supervise.py，或删除RESULT/checkpoint来强行重试。离开当前聊天/SSH不停止tmux训练。后续先读实时状态，区分“正在训练”“中断已保存”和“模型导航能力已验证”。
