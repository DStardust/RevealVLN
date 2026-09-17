# 首个真实机制数据族 V3：主 agent 数据验收

结论：**NUMERICALLY_NORMALIZED_REAL_FAMILY_ACCEPTANCE_PASS**。首个TV/sink任务实例的真实机制数据已补齐。不是原餐椅任务通过，也不是模型正收益。

## 实际交付

- 1个旧暴露MP3D房屋中的真实族；2任务×3历史×3续接=18个cell，12 pass、6 fail。
- 3个固定seed×9条完整物理轨迹=27次回放、54次独立任务求值；不计作54个独立统计样本。
- 1242条因果prefix记录，474个不同RGB像素数组、443个semantic数组，原始与重建前后证据均保留。
- 20866项运行断言、13项合成组件单测、导出后schema/NPY原始像素/27轨迹hash/保护文件复核通过。
- 4组同任务同query但不同历史标签相反的实际关系；Y由原始像素事件重算，不由纸面矩阵赋值。

## 关键限定与修订

原餐椅实例受到额外事件混入影响，在有界搜索后仍失败。V3改用TV/sink作任务anchor、餐椅作无关绕行；任务ID/模板/角色/schema明确升版，旧失败另目录保留。

真实历史回到u后仅实施一次登记的数值共同状态重建。最大位置修正 1.66893005e-06 m、旋转修正 4.17232513e-07 rad，均低于事前1e-5上限；27次边界事件集合全部不变。原始逐像素自然汇合=false，规范化后的u/s跨27回放位置/旋转spread为0，RGB/semantic哈希精确一致。不能在论文中省略此干预。

这是合法授权MP3D场景中实际运动和真实渲染形成的离线数值规范化对照数据，不是人工编辑图像、纸面标签或导航性能测试。此单族仅interface_only，不能扩称训练集规模、跨场景泛化、SOTA或贡献已成立。

## 数据入口

[FAMILY_MANIFEST](FAMILY_MANIFEST.json)、[监督侧18格](SUPERVISION_ONLY.jsonl)、[策略当前输入](POLICY_INPUT.jsonl)、[完整因果prefix索引](POLICY_PREFIX_INDEX.json)、[复核](READBACK_AUDIT.json)。按DATASET_USE_ZH使用，不得把监督侧future query/pose/semantic/ID喂给策略。

GPU3已恢复原占位，无真实任务被停止。本会话未加载Qwen或进行SFT/机制训练；SFT由另一个会话按独立handoff负责。
