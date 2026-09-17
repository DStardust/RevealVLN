# G2 最小Qwen数据/记忆/梯度接口：通过

2026-09-09，主agent实测裁决：`G2_MINIMAL_INTERFACE_PASS`。
这次已实际加载固定Qwen3.5-2B并完成两步前向、反向、一次诊断更新与重置复测，不是纸面shape检查。通过的是有界工程接口，不是算法科学收益或完整训练实现验收。

## 运行来源和配置

- 模型：Qwen/Qwen3.5-2B，固定revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。4,548,221,488字节权重的SHA256与官方LFS一致：`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`。
- 独立环境：Python3.10.20、torch2.8.0+cu128、torchvision0.23.0+cu128、Transformers5.15.0、PEFT0.18.0。55个安装包的版本锁及依赖检查已保存；执行模块位于Q35N内部，原Habitat环境没有修改。
- 安装的modeling_qwen3_5.py与官方v5.15.0源码逐字节哈希相同。模型processor实测为Qwen3VLProcessor，原生对象路径与hidden2048匹配。
- 视觉冻结、BF16主干、rank8 LoRA、8×2048显式记忆、4动作输出。图中可训练参数共5,197,061。
- 数据取普通pilot最终索引的首条配对，在决策t=1/2使用三个真实RGB、对应原指令及已执行动作。没有按预测正确性选样本，没有读留出集。
- 辅助reader的query和target明确是synthetic interface probe。未使用或产生真实交叉续接Y；这些测试标签不进入训练数据。

## 实测检查

| 项目 | 结果 |
|---|---|
| 官方视觉特征与占位 | 两张图总计128个packed视觉token，与占位数精确相同 |
| 完整序列MRoPE | 两步长度188/189，position分别[3,1,188]/[3,1,189]；append后计算 |
| 记忆 / 动作输出 | [1,8,2048] / [1,4]，均合法 |
| 跨步隐式缓存 | past_key_values=None/use_cache=false，无返回KV/hybrid缓存；rope_delta不持久化 |
| 续接查询边界 | 在prefix算完后调用两个不同query，不改变已算好的memory/action张量 |
| 重置 | 同一参数、同一输入、零记忆的两次重算，memory最大差值0 |
| 视觉塔冻结 | requires_grad=false且grad=None |
| 参数实际更新 | writer/write_query/选定LoRA B/reader在1次诊断SGD后均改变 |

仅对合成续接BCE做反向时，梯度L2实测：

- 早期冻结视觉feature leaf：0.495063；
- 早期memory：0.060394；
- writer：14.712831；write query：66.980293；
- 指定LoRA B：0.515124；reader：12.105509。

均finite并超过事前阈值1e-12。它证明后一步损失有可微路径经过显式记忆返回早期特征，不证明视觉特征语义可识别、记忆足够或实际任务可泛化。
LoRA探针预选的是layer3 self_attn q_proj的B矩阵；没有把零初始化时A梯度可为0伪装成失败或声称所有LoRA参数均被检验。
随后加入真实动作CE梯度，执行一次SGD(lr0.01)诊断更新。没有执行完整训练或保存更新后的policy checkpoint。

## 成本与资源清理

- 峰值torch allocated：7,053,307,904字节，约6.57GiB；reserved约6.65GiB。这是本次batch1、两步短输入、SGD探针的峰值，不是长序列训练预算或机器人部署显存承诺。
- 探针main内约10.25秒，包含模型加载后的检查；worker全生命周期约30.08秒。日志阶段计时不是同步p50/p95延迟基准，不能用于宣称部署速度。
- 新增独立环境约6.42GiB、缓存约6.43GiB、模型约4.26GiB，总计约17.11GiB（不含少量报告）。依赖获取与安装约16.1分钟，占主要耗时；网络传输总流量没有逐包计量，不能把磁盘大小当精确下载字节数。
- 按原授权只借用经核实GPU3占位，worker已退出，原tmux占位已恢复（PID2732592）；真实任务停止数0。

## 剩余缺口与决定

**已解除的阻塞**：小Qwen能否加载、接收真实图像、按官方MRoPE接入显式记忆、保留跨步梯度、连接LoRA和动作头，以及基本资源是否可承受。

尚未验收：完整schema-v2真实query encoder、真实Y的分组BCE、多任务/多样本/变长batch隔离、长时域训练稳定性、任务切换测试、导航闭环SR/SPL和泛化。当前query编码器只是有界接口测试夹具，不能冒充完整M4算法实现。
G1F/G1R的完整机制族仍为0，原失败保留；本G2没有替代或绕过真实标签准入。

下一步可分开推进：基于现成普通数据制定小规模基础SFT验收，先测模型能否学会动作与闭环导航；机制线继续G1R首族认证。两者均需各自冻结运行协议，基础SFT结果不能替代创新方法的独立收益。
当前停止在本节点，不自动训练、扩量、公开数据或推进导航评估。

主要机器证据：[result.json](result.json)、[GRADIENT_PROBE.json](GRADIENT_PROBE.json)、[DIAGNOSTIC_UPDATE.json](DIAGNOSTIC_UPDATE.json)、[ENVIRONMENT_ACCEPTANCE.json](ENVIRONMENT_ACCEPTANCE.json)、[模型文件锁](MODEL_FILE_LOCK.json)。
官方接口来源与一项404纠正见 [SOURCE_INSPECTION_NOTE.md](SOURCE_INSPECTION_NOTE.md)。
