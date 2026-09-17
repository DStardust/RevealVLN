# 普通导航基座：修复后续训入口

2026-09-11 13:58 更新。本文件替代“旧 v3 仍有有效训练进度”的理解，不覆盖旧日志、失败记录、源码或输入锁。

## 当前结论

三卡训练已恢复并通过运行验收。最后一次实时读取已到 **31,020 步、第二个 epoch**；第 30,600 和 30,800 步新检查点均完成 SHA、协议/数据绑定、权重和优化器有限值回读。验收文件生成时为 30,960 步，之后继续推进；31,000 步亦已产生落盘凭据。

这是普通基座的运行修复，不是模型创新或闭环导航收益。当前仍未融合特殊数据，未测正式导航成功率。不要根据只有十来个 STOP 正例的单个训练窗口判断效果升降。

新工作目录：[ordinary_sync_recovery_v1](sft_acceptance/ordinary_sync_recovery_v1/)。唯一活跃训练会话 `q35n_sync_recovery_v1`；不要再启动旧 watchdog 或第二份训练。

## 修复了什么

1. 旧进度停在 30,499 步，各 rank 按本地墙钟决定是否进入日志 all-reduce，存在通信顺序分叉。源码和 CPU 测试证实此风险；旧现场未抓到 Python 栈，不把推断写为堆栈确证。
2. 新版按共同更新步记录日志，停止/预算通过无条件共同通信汇总，NCCL 设 180 秒有限超时；不再依赖每个 rank 自己决定是否跳出。
3. CPU 恢复验收发现本机 TCPStore 在主机名、回环地址和 libuv 变体上均超时；普通回环 TCP 正常。新版本使用每个 attempt 独立 FileStore，真实三进程 Gloo 偏时钟/单 rank 停止/墙钟预算测试通过，随后真实三卡 NCCL 训练也已推进。
4. 模型初始化直接绑定各自 GPU，避免 rank 1/2 在第一张卡各留一个约 4.8 GiB 额外副本。逐文件比较确认只改设备位置，模型定义、数据、loss、batch 计划、优化器与 LR 方案不变。
5. 校正恢复时把 rank 0 的本地计数复制给其他 rank 的账目问题，按原确定性 batch 计划重建各 rank 计数；吞吐改用本窗口新计算量，避免把历史量除以本次运行时间。

通信行为与有限超时的依据见 [PyTorch 分布式文档](https://docs.pytorch.org/docs/2.9/distributed.html)；本机实际 torch 为 2.8.0+cu128，采用的接口已经本地 CPU/GPU 验证，不升级环境。

## 恢复与资源账

- 恢复点：旧 `checkpoint_000030400.pt`，SHA256 `49e59020ef5fa4b9b870ec524816d1a9b4217380dc3ae6ed15ef13344562d73e`。
- 旧已报告但未保存的 99 步回退并重新计算；更后面的未记录工作无法精确恢复，保持未知。
- 旧最高计数 2,236,987、旧已记录窗口量 2,232,128 分别保留，不冒称精确总消费或独立样本。新预算起始计费 2,401,387，其中 **164,400 是旧未观测尾部的保守预算预留，不是已测计算量**。
- 保留旧 60,000 更新 / 4,500,000 计费决策上限，三 epoch 计划和历史墙钟不重置；最迟墙钟终点为 2026-09-12 02:02:54 CST 左右，以 PROTOCOL 中 unix 为准。达到任一边界停止。
- 新监督器最多 3 个 attempt，启动超时 600 秒、有进度后停滞阈值 120 秒；失败先清自身进程，再从校验通过的最近 checkpoint 有界恢复。重放和失败尾部预算不清零。
- 本次旧训练和旧重启器均按实时 PID/starttime/父子关系精确停止，原租约已确认 GPU3/4/5 占位恢复成功。之后新租约重新核验并借用同三张卡；新训练结束/失败/受控中断后恢复原占位。GPU2 新出现的外部作业及 GPU6/7 占位未操作。
- 预热后多个窗口约 100–140 决策/秒，偶有数据/形状相关波动；这不是本轮新做的十倍加速对照试验。

## 远程监控

2026-09-11 14:15，按用户要求已将**原 18766 网页**接回当前续训：`http://127.0.0.1:18766/`、`/api/status`、`/healthz`。仅精确替换旧监控进程，训练进程身份保持不变；旧曲线保留，恢复处分段，预算保守预留不计入跨段吞吐。原被冻结的网页/训练源码不改，独立适配层为 `sft_acceptance/monitor_charts_recovery_v1`，切换验收见其 `SWITCH_RESULT.json`。

页面只读，POST实测405。展示日志与监督器心跳、更新数、检查点、原预算、CE、近期至少4000决策的动作统计及GPU/生产队列。恢复后的18767页面仍可作为备用，但用户首选入口是原18766。

在本地电脑建立 SSH 隧道（服务器地址、账号和 SSH 端口沿用平时登录配置）：

```bash
ssh -N -L 18766:127.0.0.1:18766 用户名@服务器地址
```

然后浏览器打开 `http://127.0.0.1:18766/`。已有该隧道时只需刷新原网页；服务仅绑定回环，不向公网裸露训练信息。

## 接手时读这些文件

- [真实续训验收](sft_acceptance/ordinary_sync_recovery_v1/LIVE_ACCEPTANCE.json)：CPU 测试、两次 checkpoint 回读、GPU 快照、未验证范围。
- [当前状态](sft_acceptance/ordinary_sync_recovery_v1/formal/STATUS.json) 与 [新进度](sft_acceptance/ordinary_sync_recovery_v1/formal/attempt_001/PROGRESS.json)。如果监督器进入新 attempt，按 STATUS 的 run_dir 读取，不固定旧 attempt。
- [活动协议](sft_acceptance/ordinary_sync_recovery_v1/PROTOCOL_FILESTORE.json)、[执行审核](sft_acceptance/ordinary_sync_recovery_v1/MAIN_AGENT_APPROVAL.json)、[租约](sft_acceptance/ordinary_sync_recovery_v1/RUNBOOK_FILESTORE.json)。
- [旧停滞现场](sft_acceptance/ordinary_sync_recovery_v1/INCIDENT_BEFORE.json) 与旧租约 `formal/lease_3gpu_r1/LEASE_RESULT.json`。旧中断结果不是成功训练。
- [模型创新及近邻审查](reports/MODEL_ARCHITECTURE_NOVELTY_REVIEW_20260911_ZH.md)：新增架构尚未准入，普通基座与方法实验分开。

下一步是保持这份有界普通训练，在独立版本准备固定评估和真实导航 bench；新的记忆结构先通过针对性近邻/算子审查，再开展同池、同预算机制实验。本次未启动新造数或任何架构实验。
