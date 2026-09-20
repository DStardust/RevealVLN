# RUNNING

V16 训练模型 0/9，自主续接 0/720；主任务计划每臂 N=192，尚无完整效应比较。干预获益 UNKNOWN，未采用。

实际已认证族 24/26：FIT=16，DEV=2，TEST=6。去重后统计，跨 run 复用不重复计数。屋清单：`{"FIT": ["5ZKStnWn8Zo", "QUCTc6BB5sX", "jtcxE69GiFV", "pa4otMbVnkk"], "DEV": ["2azQ1b91cZZ"], "TEST": ["PX4nDJXEHrG", "fzynW3qQPVF", "yqstnuAEVhm"]}`。

冻结 best4k 已加载；FIT 黄金输入 16 条，真实前向 48 次，原生 logits 最大差 0.0，argmax 翻转 0；基座首尾未变。这是数值/运行证据，不是方法收益。

已结束 GPU 会话累计 1.1262 小时，包含失败。运行中的服务：`['v16-formal-20260920-01']`；活跃会话消耗未加入已结束账本，见对应 RESOURCES.jsonl。

共同修复包括统一 method argmax、前瞻碰撞安全终点、稠密因果监督、可恢复 optimizer/RNG 与完整组封存；不能将这些公共变化归于 Ours。现有 V15 18 模型、216 条续接、87 个 UNKNOWN 及旧 DATA 准入标记均未修改。

已实际读取固定入口、V15 报告/源码锁/资产清单、数据/teacher/objective/训练/编码器/runtime_r2/旧 STOP 选择器与原 SEE2 编译器。可用本地资产包括 best4k、底模、项目 Python、Qwen/Habitat 环境、MP3D 场景和原始数组。初次路径猜测 feedback_v12/feedback.py 与 runtime_r2/prefix_replay.py 不存在；实际实现分别位于 feedback_generation_v1/feedback.py 与 runtime_r2/continuation_service.py，已改为读取真实入口。

早期采集失败和 CuBLAS 环境缺项保留在 runs/。同一族中的多个任务、续接和相同物理历史不计作独立样本。SEE2 始终是同一实例连续两帧各至少256像素，不等于到访房间。没有完整 R2R SR、SR40、新架构或真机部署证据。

运行命令见 README.md；独立 service 的 PPID=1、UID/GID、cgroup、退出码见 standalone_jobs。pipeline 按同一注册协议自行推进，未达完整数据规模时明确报不足，不把部分数据当正式720条实验。
