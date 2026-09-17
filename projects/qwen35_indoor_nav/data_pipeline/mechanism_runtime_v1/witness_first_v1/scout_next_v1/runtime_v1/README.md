# 两分片只读运输运行适配

新增代码仅在本 runtime_v1，旧 scout_next_v1/INPUT_LOCK、配置及所有 scout 生产源码不改。exact()要求每个替换恰好一次，否则停止。实际发现/反馈/闭合/Compiler/bank算法无变化，只明确运输 common 的 HERE/RUNTIME/bank/budget依赖路径、worker的HERE/GPU参数和supervisor的HERE/LINE/UUID/nvidia目标/worker命令/已批准预算。store类在独立common模块内绑定各片根，不共用可写全局模块。

片0：GPU1 / GPU-734a5268-31fe-6452-105b-36cd08c3d9c8；片1：GPU2 / GPU-be1b30d0-517b-b079-871b-de195d35a1a2。输出分别为 scout_next_v1/shard_0/run_v1 与 shard_1/run_v1，严格沿之前冻结 intended_output_root，不覆盖运行目录。ENV固定本线 .envs/q35n_habitat_v017_g0r。只收真实组件与候选，不模型/训练，不认证新族。

main 审核后分别建立 `MAIN_AGENT_APPROVAL_SHARD_0.json` 或 `MAIN_AGENT_APPROVAL_SHARD_1.json`，内容必须精确等于 `transport.approval_value(shard)`，包括 approved=true、shard、gpu、prepared_input_lock_sha256 和 runtime_input_lock_sha256。不可在一个审批文件授权两片。入口先验证两个完整锁，再验证对应审批，之后才创建新输出、加载监督或查询GPU。有效cfg的runtime_allowed/executable/runtime_adapter_ready同时true，training_allowed始终false，源PREPARED文件保持false。

主 agent 调用（本轮尚未运行）：

`项目标准Python -I -S -B .../runtime_v1/run.py --shard 0`

或 `--shard 1`。也可导入 `run.execute(0/1)`；一进程只执行一片，后台并行需两个独立进程。不要在现有GPU任务未自然释放时启动；监督仍要求初始util0、外部每PID<=768MiB/总<=2048MiB，包含图形进程；自身GPU保守上界<4096MiB，仅清理自身worker，不动外部/占位。每片4500秒监督/4200秒工厂/60000动作/8GiB总磁盘/6GiB内容/8GiBRAM，未改变durable预算与Journal。

CPU测试覆盖精确替换计数、两个GPU目标、标准ENV、不同输出根与空store、审批双flag转换、跨片审批拒绝、缺审批无输出无GPU、图形进程预算。测试不是渲染验收；实际运行需main核验当前GPU资源并单独授权。
