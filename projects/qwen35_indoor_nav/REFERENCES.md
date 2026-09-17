# 一手来源与证据范围

核验日期：2026-09-09。以下不是穷尽检索，也不是所有仓库已完整复现。

更新：具体算子的最新核验范围及新增近邻（CFNBC、AdaTurn、EgoPush、LSP-AIG、TANDEM、CCC-VLN、GLiDE）见 [G0 审查 V1](reviews/Q35N_G0_CONTRIBUTION_REVIEW_V1.md)，不可变代码版本和 SHA256 见 [源登记](reviews/SOURCE_AUDIT_V1.json)。

## 模型、数据与同组前序

- [Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)：视觉模型与可下载权重；不提供本项目导航结果。
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)、[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)：前序选型核验，未本地运行。
- [VisualThink-VLA](https://arxiv.org/abs/2605.30011)：用户确认同组；前序已读方法正文和 [soft_evidence_adapter.py](https://github.com/DCDmllm/VisualThink-VLA/blob/main/visualthink_vla/models/soft_evidence_adapter.py)。
- [EgoCoT-Bench](https://arxiv.org/abs/2605.19559)、[数据卡](https://huggingface.co/datasets/DStardust/EgoCoT-Bench)：前序已读；评估数据不自动转为导航训练数据。
- [AI2-THOR 环境状态](https://ai2thor.allenai.org/ithor/documentation/environment-state/)、[随机化接口](https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/)、[ProcTHOR 项目](https://procthor.allenai.org/)：本轮静态可用性核查，未在本项目安装/运行。

## 核心方法近邻

- [PRODEN，ICML 2020](https://proceedings.mlr.press/v119/lv20a.html)：前序读取 [main.py](https://github.com/lvjiaqi77/PRODEN/blob/main/main.py) 和 [utils_loss.py](https://github.com/lvjiaqi77/PRODEN/blob/main/utils/utils_loss.py)。
- [GPO，ICLR 2026](https://arxiv.org/abs/2505.15418)：v2 相关正文；本轮完成 GPO.py 全文件阅读，具体实现与固定提交见 G0 审查。
- [Leveraging Fully Observable Policies for Learning under Partial Observability](https://proceedings.mlr.press/v205/nguyen23a.html)：一手论文摘要与导航反例，说明问题已有基础。
- [DESPOT](https://arxiv.org/abs/1609.03250)、[官方实现](https://github.com/AdaCompNUS/despot)：本轮已读完整 src/solver/despot.cpp，非完整仓库复现；叶节点界限风险见 G0 审查。
- [CAST](https://arxiv.org/abs/2508.13446)：前序方法阅读；本轮已完整核对 counterfactual.py 与 action_generation.py。
- [ERASER](https://aclanthology.org/2020.acl-main.408/)：证据保留/删除诊断已有，不单独算新算法。

## 当代导航近邻/对照候选

- [AwareVLN](https://arxiv.org/abs/2605.22816)：已有自省/自动训练数据；前序读取 [loss.py](https://github.com/GWxuan/AwareVLN/blob/main/llava/model/loss.py)，不代表完整训练审计。
- [Progress-Think](https://arxiv.org/abs/2511.17097)：前序方法记录，语义进度不是新问题。
- [OpenBelief-Nav](https://arxiv.org/abs/2608.13923)：本轮一手摘要核验；证据保持记忆近邻，未审计代码。
- [SpaceVLN](https://arxiv.org/abs/2606.08992)：本轮一手摘要核验；统一路线/目标导航不是独占贡献，未复现分数。

不因为论文年份较新就假定可公平比较；正式对照需另锁数据、传感器、动作、预算、训练访问和不可变代码版本。

## 工作方法

scientific-critical-thinking 技能用于区分数据、机制假设、推断与实验支持，并要求匹配对照和因果信息边界。

Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner. 2026.
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。
本轮核对 arXiv 最新记录为 2026-09-02 修订的 v2；引用使用无版本 DOI。不是机器评审准确性的实验证明。
