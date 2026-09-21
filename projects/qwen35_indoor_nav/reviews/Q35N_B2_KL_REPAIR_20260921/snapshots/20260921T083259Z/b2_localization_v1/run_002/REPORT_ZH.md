# B2 冻结特征定位

COMPLETE_LOCALIZATION

零新导航、零策略更新、零底模前向。探针只在 FIT 训练，固定 final400；DEV 仅诊断。

|种子|当前事件探针|事件|FIT 平衡准确率|DEV 平衡准确率|
|---|---|---|---:|---:|
|1209|linear|current_anchor_SEE2|98.2%|67.9%|
|1209|linear|current_terminal_SEE2|97.6%|67.8%|
|1209|mlp128|current_anchor_SEE2|100.0%|65.1%|
|1209|mlp128|current_terminal_SEE2|99.9%|69.6%|
|1210|linear|current_anchor_SEE2|97.4%|66.2%|
|1210|linear|current_terminal_SEE2|97.1%|70.6%|
|1210|mlp128|current_anchor_SEE2|100.0%|62.7%|
|1210|mlp128|current_terminal_SEE2|100.0%|70.6%|
|1211|linear|current_anchor_SEE2|98.1%|62.1%|
|1211|linear|current_terminal_SEE2|97.3%|68.8%|
|1211|mlp128|current_anchor_SEE2|100.0%|64.4%|
|1211|mlp128|current_terminal_SEE2|100.0%|71.2%|

|种子|冻结记忆来源|FIT 历史状态探针|DEV 历史状态探针|
|---|---|---:|---:|
|1209|B1|72.9%|60.0%|
|1209|B2|88.5%|56.0%|
|1210|B1|68.7%|54.9%|
|1210|B2|89.2%|55.7%|
|1211|B1|69.5%|56.3%|
|1211|B2|90.2%|56.6%|

|种子|B2 原状态头：DEV 历史状态平衡准确率|当前终点状态平衡准确率|
|---|---:|---:|
|1209|55.0%|59.8%|
|1210|52.0%|59.2%|
|1211|54.5%|58.8%|

## 梯度与动作

- action_vs_auxiliary：负余弦 12/24；中位数 -0.0547。仅为固定 FIT 样本上的局部梯度关系。
- action_vs_preservation：负余弦 24/24；中位数 -0.7028。仅为固定 FIT 样本上的局部梯度关系。
- seed 1209：B2 四位状态都预测正确的 DEV 教师决策 175 条，与已登记教师动作集合一致 100 条。
- seed 1210：B2 四位状态都预测正确的 DEV 教师决策 232 条，与已登记教师动作集合一致 134 条。
- seed 1211：B2 四位状态都预测正确的 DEV 教师决策 113 条，与已登记教师动作集合一致 62 条。

## 判断边界

- One exposed DEV house; no independent test or navigation improvement.
- Probes are diagnostic-only and trained on FIT; failure is not proof of absent information.
- Teacher-path state correctness is not autonomous accuracy; teacher action mismatch may have other legal alternatives.
- Gradient cosine is a local conflict signal, not evidence that removing a loss improves navigation.
- Full-state auxiliary and actor share recurrent parameters, but the actor does not directly consume state_head outputs.
