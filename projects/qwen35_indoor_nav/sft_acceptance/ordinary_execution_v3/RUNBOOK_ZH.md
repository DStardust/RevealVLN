# 普通导航训练与远程监控

本节点是用户2026-09-10要求的普通导航首段训练。不要重跑已存在的supervise.py，不要删除RESULT或checkpoint来重试。实际状态以STATUS.json、run_0001/PROGRESS.json和RESULT.json为准。

训练tmux：`q35n_ordinary_train_v3`；监控tmux：`q35n_ordinary_monitor_v3`。离开当前聊天或SSH不终止后台任务。

## 数据与预算

- 仅冻结的普通训练数据：51 FIT屋、5849物理路线、17548官方人工指令、1394744指令条件动作。DEV/CONFIRM、EnvDrop和特殊机制数据均不加入。
- 普通未加权动作CE；通用Qwen3.5-2B初始化，rank8 LoRA、8记忆槽、视觉冻结、4步TBPTT、8chunk累计、AdamW1e-4。完整遍历各指令路线，保留STOP，不随机截断或改变旧输入锁。
- 本首段最多24小时、10000优化更新、约320000训练决策，先到先停；因此不等于承诺首段完成一个epoch，更不等于已完成3个epoch。
- 每100更新及正常收口保存trainable/optimizer/RNG/cursor/memory/statistics。失败不自动无限重试，后续须在新运行目录明确恢复绑定与新增有限预算。

## 远程查看

在自己的电脑运行，使用你实际的SSH账户/主机别名；必要时使用已有跳板配置：

```bash
ssh -N -L 18766:127.0.0.1:18766 <用户名>@hpda-jushendaohang-005
```

浏览器打开 `http://127.0.0.1:18766/`。JSON为 `/api/status`，服务健康为 `/healthz`。只读接口，不提供停止/重启训练的HTTP功能；不监听公网地址，不把认证凭据放进URL。每10秒刷新，显示loss/updates/decisions/动作混淆矩阵与STOP召回、GPU、checkpoint及更新时间。loss是在线teacher-forcing指标，不是闭环导航成功率。

## 已保留的失败与修复

- ordinary_execution_v1：模型加载前UUID表示不同被拒绝，0正式更新、真实cleanup完成。PyTorch缺`GPU-`前缀；V2/V3规范化后仍核对精确物理UUID，不取消检查。
- ordinary_execution_v2：真实长输入反向、STOP梯度与checkpoint前向通过检查，但重复恢复更新不一致，0正式训练更新；诊断更新仅probe，未准入首段。失败和54MiB诊断checkpoint原样保留，GPU已清理。
- 已由项目Torch CPU实证：AdamW非capturable/fused的step张量可能与载入state共享，第一次更新污染供第二次恢复使用的原state。未保护时1→2且更新不一致；每次restore前deepcopy后原state保持1且更新精确一致。V3新增私有restore包装和正反两个CPU回归测试，未改旧runner或降低exact门槛；最终仍须真实GPU重复检查通过才开训。

## 生产与恢复边界

GPU2训练持有与原生产相同的gpu_2.lock并传给CUDA worker；六条健康生产队列继续不动。原GPU2生产batch244自身失败，245–247未尝试，保留待另排有限接续；不与训练争卡，不因新目录重置旧生产授权截止。

外部任务/contexts不发送信号。守护每10秒写原始GPU XML，再检查外部单进程768MiB/合计2048MiB及自身28GiB显存、心跳/时限/运行产物预算；超限只清理本次worker。GPU2没有借用占位，不能凭空恢复占位。健康其他GPU占位链、冻结源码/输入锁和旧失败均保留。

本轮仅重启了自己新建的训练监控以指向新版本；精确旧PID/starttime/argv/cwd与信号记录见各版本MONITOR_RESTART.json。原18765监听服务未动。

CPU验收：旧34项、资源/cursor7项、监控11项、optimizer所有权正反例2项通过。根research.sh check也通过历史注册表检查，但该检查不替代本节点真实GPU和导航验收。当前未做独立场景闭环SR/SPL，不能称基础导航能力已训练成功。
