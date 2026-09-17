# 真实机制数据 CPU loader 规格 V2

本规格先于实现/测试冻结。仅在本目录写入；封存认证族、compiler、普通 SFT 和环境只读。stdlib-only，不运行 GPU、Qwen、仿真、下载或训练。

输入固定为 Q35N_G1R_FAMILY_CERTIFICATION_V1，1 个 interface_only、old_line_exposed 的 TV/sink V3 数值规范化族。读取 6 条 task×history prefix，各自从零重放完整 207 决策；3 个未来查询必须在 prefix 处理完成后单独使用。不是 18 个独立历史，也不是 27 个独立训练样本。

模型侧白名单固定为 instruction、RGB 原始 uint8 bytes、已执行动作、memory_reset。step、ID、hash、路径、pose、semantic、task程序状态、未来query和Y只在loader控制/监督侧，绝不返回到policy payload。RGB形状固定224×224×3；验证NPY头、dtype、shape、长度和原始像素SHA256。读取路径必须位于封存族根内，拒绝path traversal。

prefix记录乱序输入按cutoff排序恢复0..206完整序列，缺口、重复、窗口越界、reset错误、指令变化、已执行动作不一致均拒绝。独立迭代器无持久模型状态，任何训练caller仍必须实际执行reset并隔离模型cache，CPU验证不能证明神经状态隔离。

query单独严格验证固定字段/动作/类别/房间/阈值/顺序/唯一末尾STOP；语义投影仅schema、坐标系、sequence，排除integrity refs和ID。输出保序typed整数元组：movement(kind, action, repeat)，observe(kind, fixed category, fixed room,256,2)，STOP(kind)。这是数据编码，不是神经queryencoder，禁止直接连接G2 toy Embedding(8)+mean。

supervision的Y与query分别提供；fail是BCE负例但动作mask0，unknown/rejected BCE mask0。动作目标step从cutoff开始即a_t，明确继承DATASET_USE中对旧schema X03的>=cutoff纠正。ACTION_DEDUP owners唯一，验证key/payload/lineage和所有owner；只一次CE，保留其余真实不同续接的动作。

数据采样显式按family→task→有效cell分组均匀；当前仅一个族，不能声明多族采样经验验证。动作CE采用单独owner遍历，不把有放回BCE采样次数当CE权重。未来决策的policy从相应真实trace逐步构建因果窗口，绝不将整段future trace传入当前policy。

验收：真实全prefix顺序/重置、乱序规范化、ID更名、不同query不改变policy、深拷贝/跨样本污染、query注入/未知类别、坏像素hash、未来时间、坏mask、坏owner、action时间边界与完整真实CE样本检查。前后核验封存SHA256SUMS及compiler锁。失败不删改原证据；输出result/测试记录/真实形状统计/README/哈希。scientific_pass=false，mechanism_training_authorized=false。
