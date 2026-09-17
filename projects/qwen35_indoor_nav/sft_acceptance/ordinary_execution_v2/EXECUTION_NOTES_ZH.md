# 普通导航首段训练运行说明

用户2026-09-10最新明确要求开始训练、必要重启及远程监控。本节点仅普通动作CE基座训练，不启动特殊reader或改写V3主线。旧ordinary_baseline_v2仍为PREPARATION_ONLY，旧协议、数据快照、源码、输入锁及失败均不改。

输入是原封存51 FIT屋、5849物理路线、17548官方人工指令、1394744指令条件决策。每条指令完整有序遍历，固定seed1109按epoch重排，保留末尾STOP；不混入EnvDrop、特殊数据、DEV、CONFIRM。训练参数沿用准备协议：Qwen3.5-2B原始预训练基座、rank8 LoRA、8记忆槽、冻结视觉、普通未加权CE、4步TBPTT、8个chunk累计、AdamW1e-4、clip1、单步上限1024且不截断。

首段最多24小时/10000次优化更新/约320000决策，实际以先达到者为止。这是访问全数据索引的有界首段，不是已完成3个epoch，也不保证首段遍历一个epoch。每100次更新保存模型增量、优化器、随机数状态、route位置、非零记忆和累计统计；退出前正常flush并保存。损坏或失败不自动重试、不覆盖旧checkpoint，后续恢复须在新运行目录验证绑定和清理状态。代码目前自动执行本首段，未设置无限自动续训。

## V1失败与V2修复

V1在模型加载前报告LIVE_GPU_UUID_MISMATCH，0正式训练更新，supervisor已实读确认worker退出和自身CUDA context消失。PyTorch真实返回`be1b30d0-517b-b079-871b-de195d35a1a2`，nvidia-smi/外部准入使用`GPU-be1b30d0-517b-b079-871b-de195d35a1a2`。这是同一物理UUID的字符串前缀差异，不能取消GPU身份核验。

V2只在新入口把PyTorch UUID规范为GPU-前缀后与冻结GPU2 UUID逐字比较；随后将PyTorch原生表示交给旧backend的原生表示比较。CUDA_VISIBLE_DEVICES继续必须精确等于原带前缀UUID。旧runner和V1源码/封存不改。实际映射保存在run_0001/GPU_UUID_BINDING.json。

## 真实技术验收

真实模型加载后依次检查：新旧前向输出一致；最长真实805-token附近输入四步反向；真实终末STOP动作头梯度非零有限；模型/优化器序列化恢复及下一次更新完全一致；非零中途历史记忆与cursor保存恢复后下一chunk记忆、更新、优化器完全一致；序列、累计token、累计decisions、wall越界明确拒绝；真实输入扰动和吞吐诊断。probe的参数更新全部恢复到初始状态后才开始普通训练，累计实际compute账不因恢复清零。

仅上述实际检查通过才写PROBE.json和run_0001/RUN_ADMISSION.json并自动进入首段。接口和teacher-forcing loss不是闭环导航收益；独立DEV的SR/SPL及STOP闭环评估尚未执行。

## 资源与生产

GPU2没有占位需要借还，外部两个context不停止。supervisor取得既有auto_production_v1/gpu_locks/gpu_2.lock并传给worker，其他六条健康生产队列继续不动。原GPU2生产244失败现场及245–247未尝试预约保留，不与训练同时启动。GPU2是否接续生产需训练终态、真实cleanup和新的有限排期，不能重置旧12小时预算或直接重启旧lane。

每10秒保存GPU原始XML及状态，再检查资源；只允许本训练worker使用最多28GiB，外部每进程≤768MiB/合计≤2048MiB，未知或超限时只退出本worker。总时限、心跳、运行产物大小有外层守护。该锁仅协调本线遵守锁的进程，不能防止外部应用自行上卡。

## 远程监控

已为本运行提供独立只读网页及JSON接口，固定127.0.0.1:18766。原18765服务未动；本轮自行创建的V1监控经精确PID/启动ticks/argv/cwd核验后退出，改用V2监控，记录见MONITOR_RESTART.json。访问方法见monitor/README.md。

监控tmux：q35n_ordinary_monitor_v2；训练tmux：q35n_ordinary_train_v2。关闭本会话不终止它们。检查页面上的真实status、cursor、更新时间和RESULT；陈旧历史终态不等于正在训练。

主agent复跑原34项适配/准备/runner测试、新7项cursor/资源测试、11项监控测试全部通过。独立CPU代码复核补齐raw-before-guard、真实GPU cleanup和中途memory恢复检查。旧失败保持失败，新运行真实结果另读PROBE/PROGRESS/RESULT。
