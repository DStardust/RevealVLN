# 真实中性整圈校验的 winding 计数匹配 V1

本目录独立于封存 `batch_plan_v1`。新增构造器，不修改原checkers、任一源组件动作、18格矩阵、query160、1e−5汇合界限或最终逐类F/L/R计数相等要求。

## 构造

H_A=a+i+a×(k−1)，H_A_I=a×k+i，H_B=b×j，枚举 k=2…16、j=1…16。i仍须真实闭合、含I且不B/T，可含A，并至少2F；不声称H_A没有i。

先使F次数相同，且各净转数差必须为24的整数倍。从targetL=maxL、targetR=maxR出发，使共同目标净转余数与原行一致：分别计算仅上调L或仅上调R的0…23步方案，取总填充最小者，平局固定优先L。这样各行gapL−gapR确为24倍，随后用缺少方向的完整L24/R24补圈，再补LR对。最后≤504动作、三序列不同、各F/L/R精确一致。所有计数尝试保留原因。

注意：原行net模24相同，不在数学上保证独立maxL/maxR的gap仍满足24倍。主agent已在首次准备、未运行/未封存阶段批准上述目标模24对齐；这修复构造器提案数学缺口，不放宽checker。初次未对齐的CPU快照进程已停止，未作为候选结果交付；没有物理运行被中止或改判。

## 不假设整圈中性

`method.py` 导出 `BalancedFactory`（也是 `WindingBalancedFactory`）。constructor仍接原Factory参数以及 `components` / `balance`，配置 `factory_variant="winding_v1"`。

histories的第一实际关卡是对本候选每个非零padding方向分别执行一次24动作整圈：原checker要求全轨迹完整、无碰撞、不触发A/B两帧事件，以及agent/RGB/semantic完整pose闭合。见T但没有STOP不算任务完成，按原程序允许。失败立即抛Reject；不继续给该方向授予“中性”属性。

通过后仍实际验证a/b/i组件，逐条执行完整已填充历史。多圈复用或LR补齐后的整条路径、事件及闭合必须再次通过。最后由未改动的原construct/validate_matrix/replay_seeds完成公共tail、交叉18格、查询160与三种子校验。单次spin通过不替代完整历史或矩阵。

如公共hub的整圈看到A/B，本版直接拒绝。未来可讨论在各anchor loop内部有真实记录的位置加圈，但本版未实现，也不改变停止规则。

## 数据来源及选择

沿用scout已完整关闭house的只读BANK，完整来源SHA锁；不读取正在append的house。每hub重新枚举最多1000程序，保留截断及全部计数拒绝账本。每hub选最短历史，再最短最大续接，再proposal_id；最终按每屋最优hub轮转排序，优先跨屋首批，不把同屋两hub称为两个独立房屋。

source_observation_query_estimates只是已存组件观测的CPU拼接诊断，不是新回放或query认证。component_provenance完整保留V2格式，源动作不压缩。

CPU测试含目标模24对齐反例、求解、次数/动作上限、无源修改、原checker继承，以及fake-runner的spin执行/失败传播控制流；这些不是物理仿真。所有配置executable=false，实际spin/物理族结果未运行。
