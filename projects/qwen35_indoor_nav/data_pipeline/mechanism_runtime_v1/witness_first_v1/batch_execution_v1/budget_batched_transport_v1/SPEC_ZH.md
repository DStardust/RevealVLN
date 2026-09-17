# Batch 07：独立 fresh-only 时钟提交合并工程节点

主 agent 已授权 CPU 准备，GPU 启动由主 agent 审核后另行执行。只为新 `batch_07/run_v1`；不属于、不修改 03r1/04r1/05r1/06r1 旧 cohort。

从 `multi_program_bank_v2/language_ready_v1/NEXT12.json` 顺序取第一次出现的不同 house/hub：位置 [0,2,4]，CONFIG_DRAFT indices [0,34,60]，3 屋 3 程序。不是字面首三，不事后换失败候选、不增候选、不把同 hub 变体算独立。

只改变 worker 全局 `durable_budget` 绑定为固定 SHA 的 ClockBatchingBudgetLedger；原 shared.worker_source 逐字生成的主流程、WindingBalancedFactory、PartialTraceRunner、原 Journal、export、FamilyLoader 和 27 次完整认证保留，18/54 科学标准不变。每 reserve 原 Journal.append 同步完成后才可能 step；没有异步 writer。

fresh-only：run_v1 只能一次准备，原 readiness 拒绝已启动 run；worker 拒绝任何 bundles/content/journal/progress/result/budget 残留；原 Journal mkdir(exist_ok=False)、未传 resume；新绑定不接受 state 参数、仅允许原 Journal genesis 一条记录。不能恢复旧 budget，更不能重新授予中断 phase 预算。任何中断保留原失败文件，恢复只能另行科学内容审计，不能继续此 run。

GPU2 UUID 固定；原 readiness 60 秒和原 supervisor 3900 秒不变；budget 总 3600 秒/60000 动作，discovery 1000 秒/15000，certification 1500 秒/20000。原各外部进程 ≤768 MiB、合计 ≤2048 MiB、自有保守上界 <4096 MiB、RSS <8 GiB、disk <7 GiB、原 cleanup 权限全保留，不操作任何 holder。所有导入依赖/config/选样输入在动作前 INPUT_LOCK 封存；新程序上限仍 1024 输入文件。

CPU 测试不能替代实际物理完整族验收。实际运行后独立报告产量、失败、总动作、阶段墙时、预算日志数及资源退出；与历史轨迹不构成配对实验，不声称物理吞吐倍数、模型收益或论文创新。训练准入仍 false。

准备命令：项目 Python `-I -S -B <此目录>/prepare.py`（仅一次，生成新 batch_07）。启动命令：项目 Python `-I -S -B <BE>/batch_07/run.py`，由主 agent 审核后执行；本节点准备 agent 不启动 GPU。
