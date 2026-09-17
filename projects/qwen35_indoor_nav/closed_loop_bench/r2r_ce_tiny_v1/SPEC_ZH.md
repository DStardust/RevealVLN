# 普通基座：R2R-CE八路线闭环诊断

2026-09-11，用户明确请求一个很小的真实基准测试。本节点只评估固定普通基座，不进行训练、结构改动或特殊融合；不停止现有训练、队列或占位。

## 事前固定

- 检查点：ordinary_sync_recovery_v1 / attempt_001 / checkpoint_000034800.pt，按实存回执与内部binding校验；不追逐最新检查点或按新成绩选权重。
- 数据：本项目合法MP3D资产及本地R2R_VLNCE_v1-3_preprocessed_xlmr的val_unseen副本。只用原始instruction_text、起点/朝向、目标和评测几何；不使用其XLM-R token。全train的10819条原文和几何已与登记的官方minimal train逐项完全一致；val_unseen为1839条/11屋，历史其他研究暴露保持未知，不声称预训练从未见过。
- 选择：对11屋按sha256("Q35N_TINY_R2R_20260911:"+house)排序，前4屋，每屋按同种盐的trajectory_id排序取2条不同物理路线，各取哈希排序首条指令。选择不依赖路线长度或模型表现。全8条清单在GPU运行前冻结，不替补失败样本。
- 与当前普通基座51个FIT屋，以及原内部DEV/CONFIRM屋交集均须为空。此次8条及其房屋登记为公开开发评测暴露，禁止流入训练。
- Habitat-Sim为已独立验收0.1.7环境；224×224 RGB，90°FOV，相机高1.25m，agent高1.5m/半径0.1m，前进0.25m，转向15°。按官方R2R配置allow_sliding=True；与训练示范False的差异明示，不临时更改模型。
- 每episode最多500次策略动作，STOP计一步；实际贪心四动作，无距离自动STOP、oracle回退、最短路规划或传送纠错。初始pose严格一致。策略仅获指令、最近两帧RGB及最近8个已执行动作，碰撞后保留实际执行的动作名；不追加训练中未用的碰撞embedding。

## 指标与范围

直接运行已核对官方Habitat-Lab v0.1.7 (d6ed1c0a0e786f16f261de2beafe347f4186d0d8) 的DistanceToGoal、Success、SPL类源码。只剥离完整包导入依赖，类主体不改；采用同等逐步update顺序，CPU覆盖边界和对照公式。仿真控制为本项目独立轻量适配器，不冒称运行完整官方trainer或完整榜单。

报告SR（STOP且测地距离严格<3m）、SPL、终点NE、OSR（包括初始观察的曾进入3m范围）、路程、碰撞、STOP/步数耗尽与失败分类、每episode轨迹和推理耗时。nDTW/SDTW本轮不接入，保持null，不拿旧geodesic-DTW顶替官方指标。

仅8条/4屋的描述性诊断，主报告成功条数和逐例结果；无性能PASS阈值、不声称统计稳定、提升或SOTA。运行验收与导航表现分开。服务错误/资源截断保持独立，不能伪装模型失败或偷偷移出预选分母。

## 资源和文件

新增仅写closed_loop_bench/r2r_ce_tiny_v1及本次authorizations条目；源码、episode清单、来源哈希、checkpoint绑定、CPU验收和资源许可在启动前冻结。GPU1先核空闲UUID GPU-734a5268-31fe-6452-105b-36cd08c3d9c8，无占位借还；用途不明进程出现只停止本节点自身。

单模型+单renderer，总显存≤28GiB，总运行≤2400秒（包括初始化/编译），最多4000个环境动作；内存≤40GiB、输出≤2GiB。RPC超时180秒。一个正式attempt，不自动扩量/重试/换checkpoint；工程失败另版本留档再决定。

验收：动作映射、STOP/500步边界、四元数、近期窗口/episode重置、输入白名单与特权字段拒绝、官方指标边界、checkpoint加载/finite、真实GPU映射、8条实际episode可追溯以及结束后自身GPU context清理。CPU模拟器/虚构轨迹测试仅验接口，不计导航成绩。

一手依据：[VLN-CE任务配置](https://raw.githubusercontent.com/jacobkrantz/VLN-CE/master/habitat_extensions/config/vlnce_task.yaml)、[官方数据说明](https://jacobkrantz.github.io/vlnce/data)、[Habitat-Lab v0.1.7指标](https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py)。官方minimal下载本机TLS失败，未下载成功；使用项目既有登记体系内的预处理资产，来源差异完整记录。
