# 普通导航基座 v3：配方决策（一页）

日期：2026-09-10。依据：三份官方实现关键训练代码已实读核对（证据行号见下），不扩展文献调研。本版是**结构修订**：去掉未验证的自定义 8-slot 记忆/writer，做"Qwen3.5-2B + 因果 RGB/指令/历史 → 四动作"的独立普通基座。新 run，不续 v6 优化器状态，不伪称同构。

## 选谁的哪些部分

| 成分 | 来源 | 精确出处 | 许可证 |
|---|---|---|---|
| Teacher forcing / 行为克隆（离线专家轨迹，逐步 CE） | VLN-CE | dagger_trainer.py p=1.0 退化路径；base_il_trainer.py L162-165 | MIT |
| 长度分桶真实 batch（按长度排序+块内 shuffle，padding 归一化） | VLN-CE | dagger_trainer.py L136-186；`((w*ce).sum(0)/w.sum(0)).mean()` | MIT |
| Inflection weighting：step0 及动作变化步权重 coef，其余 1.0 | VLN-CE | dagger_trainer.py L199-211；coef=3.2(R2R)/1.9(RxR) | MIT |
| LLM SFT 信封：lr 1e-4、cosine、warmup 3%、bf16、真实 batch+padding | NaVILA | scripts/train/sft_8frames.sh L25-42 | Apache-2.0 |
| 损失只落在动作答案位置（上下文 -100 mask） | NaVILA | llava/utils/tokenizer.py L134-172 | Apache-2.0 |
| 历史帧在前、当前观测在后的提示组织 | NaVILA | llava/data/dataset.py L397-399 | Apache-2.0 |
| 阶段划分（预训练→SFT→RFT）仅作路线参考，不取代码 | ETP-R1 | 本地只读副本 third_party/ETP-R1 | 见其仓库 |

不采用：DAgger（首版不开，属后续独立授权）；GRPO/RFT；NaVILA 全参 8 帧配置与文本动作解析；ETP-R1 全景/depth/拓扑接口。

## 动作/传感器差异（必须声明）

- VLN-CE 原模型含 depth、RNN 状态；本基座只借**训练配方**，不借架构。无 depth、无全景、无拓扑航点。
- NaVILA 用 8 帧历史 + 文本动作生成；本基座用 ≤2 张近期 224RGB + ≤8 已执行动作 token，动作是 4 类离散分类头（F/L/R/STOP），不做文本解析。
- 本基座无记忆模块：移除记忆后，决策样本在因果输入（指令+近期帧+已执行动作窗）下条件独立，因此**决策级批处理合法**；路线仅用于分桶效率与 FIT/DEV 隔离，不携带跨决策状态。路线边界不再有需要 reset 的状态。

## 初始化

- Qwen3.5-2B_15852e8 本地权重（哈希沿用 v6 协议登记），冻结基模与视觉塔；rank-8 LoRA(q_proj/v_proj, alpha 16)；动作头 2048→4（FP32）随机初始化。无记忆/writer/old_slot 参数。seed 1109。

## 训练目标与损失

- 逐步 4 类 CE，仅动作查询位计入（等价 NaVILA -100 mask 约定）；末端 STOP 始终监督（数据接口已保证 terminal STOP）。
- Inflection weighting：默认 use_iw=true、coef=3.2（R2R 值；数据为 R2R+英语 RxR 混合，coef 差异登记为可调项，首版不扫描）。按序列归一化后 batch 取均值。
- 明确：这不是 action_balance_v1 的简单类别加权（该 FAIL 保留）；不承诺修复 STOP 偏置，以冻结面板趋势裁决。

## Batch / LR / 调度

- 真实 batch：决策级样本按 token 长度分桶、桶内 shuffle；首版有效 batch 24 决策/step（冻结协议时定值），grad accum 仅用于显存收口，loss reduction 与更新频率变更在协议登记。
- AdamW lr=1e-4、weight_decay=0.01、clip=1.0（沿用 v6 以便对照），cosine + warmup 3%（NaVILA 信封）；bf16。
- Epoch 配置 3（沿用），实际段由冻结预算收口；不承诺单段跑完。

## 数据来源

- 只读复用冻结 `ordinary_baseline_v2/snapshot_v1`：51 FIT 屋、5849 物理路线、17548 指令、1394744 指令条件决策/epoch。不加 EnvDrop/特殊族/DEV/CONFIRM；新数据须先审计+新 snapshot。

## 评估协议

- 冻结固定 FIT 诊断面板（覆盖 F/L/R/STOP、多屋多路线），每 checkpoint 同面板观察趋势；DEV/CONFIRM 不参与训练，DEV 评测方案独立冻结。
- 记录：CE、每类 recall、macro recall、目标/预测分布、一直前进基线差值、端到端 decisions/s、样本覆盖/epoch、累计预算。闭环 SR/SPL/终点距离/停止行为属后续单独有限授权，不由本配方自动推定。

## 复用声明

VLN-CE（MIT, Krantz 等）与 NaVILA（Apache-2.0, Cheng 等）的上述配方成分为成熟方法复用，不申报为本项目新贡献；正式报告须引用原始出处。不排除任何测试泄漏、额外传感器/真值输入或不公平协议。
