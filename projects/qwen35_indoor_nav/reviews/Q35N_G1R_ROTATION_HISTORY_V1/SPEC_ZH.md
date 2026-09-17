# G1R 原地可观察历史构造 V1

主 agent 依据 PARALLEL_SFT_AND_MECHANISM_V1 授权执行独立数据构造修订。coverage-only 因过强 L 启发式提前结束，所有实际失败保留，不把未完成256当作完整失败。

保持任务程序/模板、Eligible 类别、两帧同实例256px、传感器、动作原语、等长历史<=512、续接<=160、8 public actions、精确 RGB/semantic hash 和1e-4 pose、安全及信息边界不变。

修订的只有发现规则：

1. 用原32个 u 的单点真实原地转向回环构造历史。优先 yaw 顺序为[0,12,6,18,3,15,9,21]，每个 yaw 依原顺序覆盖32u，最多256个 base configurations。不是256完整family组合；每base最多8tails，完整尝试数单列，最多2048。
2. 每kind回环依次尝试 n=1..12，L^n R^n、R^n L^n。全部是真实标准15°动作，原始逆序和压缩逆序仍各物理replay，平移数为0。这个短历史机制接口不主张已经覆盖一般长距离导航历史。
3. D loop 禁止K；K loop禁止D；L loop禁止K/B，但允许已完成的D再次出现。整个H_D_L必须仍满足D发生后存在L；两个任务的既定完整18格矩阵不改。
4. 原中性padding和精确u汇合预检保留；能形成base才逐个原8种tail检查。base失败不重复跑相同参数。所有子搜索/失败/数值不一致明确记账。
5. 续接仍为原真实greedy导航到椅子/水槽/床，不改变原continuation构造。第一个完全通过预检的候选立即冻结，不因后续certification失败改选。

独立预算：1小时wall、8GiB输出、16GiB RAM、8GiB GPU、零下载、不加载模型或训练。GPU3独立借用恢复。成功后由独立数据验收包做27 replay/54求值和两级validator；发现成功不是最终PASS。
