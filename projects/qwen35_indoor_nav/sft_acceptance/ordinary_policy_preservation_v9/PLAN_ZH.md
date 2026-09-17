# V9：固定基座输出约束（普通导航工程试验）
本轮用户要求“继续直到第一个正向结果”。本版本仅检验一个因素：在V8R1同配方上加入固定最佳4k的输出保持损失，不将基础蒸馏包装为UAD创新。

## 三次顺序判断
1. 失败归因：V6新增11成功/丢13、V8R1新增8/丢14；精度修复确实累积更新，但完整导航SR15%、SPL13.4293%、nDTW31.4371%，全部不及最佳4k。FIT-only新诊断：4983去重输入中，1152来自原模型最终成功路线，760也被几何教师要求换动作。不同动作不等于错误标签，但当前监督并不保护已经成功的行为；“只是更新被吃掉”不能解释和解决全部退化。
2. 最小方案：保持同一模型、初始参数/矩/RNG、数据、抽样顺序、步数及学习率，只增加固定参考策略的四动作分布KL约束。不加结构、记忆、STOP控制器，也不再次全池延长训练。
3. 反对检查：属于已知输出蒸馏/防遗忘工程。权重约束或参数平均并不直接保持输出；V7等权平均已因FIT丢5条超过2而关闭，不复活。KL也可能阻碍真正纠错，不能承诺不会掉点。只测下述一个系数，不扫系数或中间checkpoint。

## 唯一新增目标
L = sum_i w_i [CE(y_i,p_i) + KL(p_4k_i || p_i)] / sum_i w_i。
lambda=1，temperature=1，四动作类别求和、样本按原inflection权重归一化。原CE项不乘1/2。rank局部和乘world/W_global，经梯度平均得到全局目标。teacher detached、固定最佳4k，不是EMA、不是V8失败最终权重，不把teacher动作当真值。
参考与student具有同一可见输入：指令、最多两帧既往RGB、最多8个已执行动作。teacher无目标/位姿/未来输入、无新标签。训练teacher使用同批次形状；保持的是此训练接口的源模型分布，并非保证所有batch1闭环决策不变。

## 实现与真实关卡
复用已审计V8R1的FP32 master初始化桥（原最佳4k数值不变、optimizer step4000），参数数量/前向架构不变。teacher只存约28组小训练参数，复用冻结基模：每批student前向之前，无梯度替换到固定参考参数，算teacher后finally原位恢复，再student正常前向/反向。绝不在student计算图存活时改参数；检查身份、恢复值、梯度、RNG及参考恒定。无第二份大基模、无付费API。
CPU测试：KL公式及梯度方向、全局权重与分rank一致；旧参数临时替换及异常恢复、参数对象/优化器状态保持、teacher不求导。
真实每rank首批：参考/学生初始分布检查沿用max_abs<=0.15、relative_L2<=0.03、argmax全同，并加mean_abs_KL<=1e-3；不因失败放宽。每200步检查参考值与恢复值及RNG不变。每步记录teacher前向计数、KL/总loss；首200与最终1000验收均检查实际KL执行、非零漂移、有限梯度、原始CE单列。CPU toy不是导航收益。

## 数据、预算、最终评测
同V6固定三rank计划1000更新/99047决策（普通89172、纠错9875；4983新增可用输入不等于本段全用）。不重新生成、不增加独立样本、不改输入锁。相同peak LR5e-6、原cosine时钟4000→5000、6144tokens/rank。
新增teacher每决策额外一次无梯度前向；最终99047次teacher + 99047次student，共198094次policy-forward决策，绝不按相同步数声称同算力。训练墙钟上限2400s、lease2700s、GPU3/4/5各26GiB；运行前核实精确占位，结束finally恢复。短TMP用项目内.t9（AF_UNIX长度<108），不写满的系统盘，不安装依赖。
固定最终1000步，首200仅正确性检查、不按分数选权重。最终CPU/资源验收后，GPU1仅空闲时跑同100条INTERNAL_DEV（5屋、batch1、500动作、显式STOP<3m），逐动作审核。主正向门槛仍相对最佳4k21%：ΔSR>0、ΔSPL>=0、ΔnDTW>=-0.01。相对V8R1的恢复只是消融，不是主要成功条件。无自动重试、下一轮、alpha/LR扫描、DEV训练或特殊数据训练。
资源/运输失败保留新版本；科学门槛失败保持最佳4k。

## 已核实外部依据（普通工程，不是新方法论证）
- Li & Hoiem，Learning without Forgetting，ECCV2016；使用新任务输入保持旧网络响应的思想：https://arxiv.org/abs/1606.09282 。已读摘要及作者官方仓库README，不声称完整复现多任务论文：https://github.com/lizhitwo/LearningWithoutForgetting 。
- PyTorch官方KL文档说明输入log概率与按类别求和/按样本平均：https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.KLDivLoss.html 。本地实现以当前安装torch的实际公式CPU测试为准，无依赖升级。
