# G2 最小模型接口验收 V1

用户“可以尝试”承接普通数据后Qwen接口建议；主agent批准本有界节点。范围：独立模型环境与固定权重获取、真实RGB输入、动作读出、8槽记忆、两步因果图和梯度验收，不开展完整训练或导航评估。
复用Qwen/Qwen3.5-2B固定revision `15852e8c16360a2fea060d615a32b45270f8a8fc`、Transformers5.15.0，保持P2R1接口。D2048/K8，最近RGB最多2张、最近已执行动作最多8个；冻结视觉塔，rank8 LoRA。
从已验收TRAINING_INDEX稳定顺序首个配对选两个相邻决策时刻；不通过预测正确与否换样本。真实动作CE只作图连通验收，不衡量训练/泛化。
机制数据仍0族：续接reader只做明确标记synthetic interface target的BCE梯度探针，绝不写入真实机制标签或scientific PASS。
检查：官方对象路径、完整token/mask/type/官方MRoPE长度、视觉占位数、memory/action形状、无KV/混合线性attention缓存持久化、任务重置、q对因果memory/action非干扰。
梯度：早期冻结视觉feature显式leaf；后一步loss到该leaf、writer/slot、至少一项LoRA B参数、reader参数均finite且L2>1e-12。LoRA A在B零初始化的首步可为0，不伪装全部LoRA都有非零梯度。视觉参数grad必须None。
允许最多1次诊断optimizer step检验参数实际更新，记录为diagnostic update，不保存训练策略权重，不称基础SFT已完成。若BF16量化使微小更新不可见，应报告而非偷偷调阈值。
资源：单GPU先空闲卡；必要时仅释放核实占位并恢复。模型/依赖新增下载≤15GiB、增量磁盘≤40GiB、单worker RAM≤48GiB、GPU≤28GiB、GPU阶段≤30分钟；依赖获取阶段≤1小时。所有内容只写Q35N，旧Habitat环境/封存数据不修改。
固定软件候选torch2.8.0/torchvision0.23.0，PEFT版本在安装前核对官方metadata并写依赖锁。只接受官方PyPI/HuggingFace/PyTorch来源；失败可按用户许可使用proxyon，不切不明镜像/不降模型版本。
若发现固定官方API与P2R1静态猜测矛盾，保留实际证据并做明确的实现纠正；不得偷偷改成另一个Qwen。无法闭合时停止对应子项，已完成工程结果不冒充全部G2通过。
