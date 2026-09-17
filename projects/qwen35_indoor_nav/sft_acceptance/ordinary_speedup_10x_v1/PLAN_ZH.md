# 普通导航训练10倍提速：只读诊断与下一步准入

用户要求：2026-09-10，“然后训练的太慢了，显存占用也不高，起码得快10倍”。本节点独立于正在运行的ordinary_execution_v6；本次诊断未修改环境/训练源码/数据锁，未停止或新增GPU任务。

## 实际基线

当前训练只使用GPU2，一张RTX5090；2026-09-10本轮实读26156/32607 MiB（约25.5/31.8 GiB），瞬时利用率24%。其余GPU1/3/4/5/6/7执行生产；GPU0不在授权范围。整体训练只用一张卡，不应把全机低显存误当单训练卡空闲。

已读PROGRESS：130次更新/4063训练决策，累计实际训练吞吐1.82249决策/秒。10倍门槛为18.2249决策/秒；1394744决策一轮在此门槛下仍约21.3小时。不得把缓存命中速度、局部算子速度、累积梯度数量或GPU利用率当作达标证据。

环境只读metadata：torch2.8.0、transformers5.15.0、triton3.4.0；fla-core、flash-linear-attention、causal-conv1d、kernels均不存在。已读实际hub_kernels.py：原包导入失败后选torch_function。实际Qwen3.5 torch_chunk_gated_delta_rule包含每chunk63次Python展开与逐chunk递推，FP32中间量进入反向图。故慢速回退路径可由实际环境/代码确认；它占总时间的比例还未通过GPU profiler测量，不能凭代码声称贡献了某个确定倍数。

当前runner每动作单样本forward，四个顺序动作构成TBPTT反向，再累计8个chunk；gradient accumulation不是并行batch。动作之间的记忆依赖不能随意拆开或用未来帧打包换速度。

历史证据：efficiency_run_v1双卡短集约3.515决策/秒，缓存热速比1.0128/1.0007；不以不同数据的历史双卡比值冒充匹配加速。triangular_gpu_v1虽然局部速度1.944倍，logit与梯度均未通过数值门槛，继续禁止作为替换，不重写旧FAIL。

## 候选顺序与不变量

1. 独立版本/独立依赖目录接入官方FLA训练内核；不向现有运行环境pip安装或升级torch/transformers，不启用未审核远程代码。固定来源版本/哈希，CPU导入与真实GPU测试分别记账。
2. 匹配真实短/中/最长指令以及含STOP的四步因果片段，比较原实现与官方内核的logit、记忆、loss、输入梯度、全部可训练梯度和优化器更新。先登记数值门槛，旧数值FAIL不变PASS；未通过不能上线。
3. 对独立路线做真正batch，按长度分组但保留每条路线顺序、独立记忆/reset/STOP。不能把同一路线的时间步当独立样本并行；改变调度需明确新版本和恢复映射，不能声称旧串行optimizer轨迹精确不变。
4. 单卡实测后决定是否多卡；如扩卡，使用实际分布式归约验收、全局决策计数与每卡显存上限，保存并检查optimizer/RNG/每stream cursor/记忆。GPU3/4/5当前都有健康生产任务，须用户明确同意其当前批次及审核完成后让卡，不能借低利用率直接停止它们。
5. 性能验收覆盖数据读取/真实RGB解码与核验/预处理/forward/backward/optimizer/同步，冷编译与稳态分别计时。稳态匹配窗口至少10分钟或100次更新中的更长者；完整有效决策每秒至少10倍已登记基线，报告每卡与全局吞吐，不将丢弃样本、缩短路线、去STOP或改模型当等价加速。

训练卡临时切换测试必须先保存完整断点，核实进程身份和GPU锁；新实现通过前不将当前生产训练改为未验收内核。当前没有测得任何新加速倍数，也没有启动多卡训练。

## 资源选择待用户确认

建议GPU3/4/5各自完成当前批次、清理、原强审后停止新批接续，随后与GPU2组成四卡训练；GPU1恢复、GPU6普通扩源、GPU7特殊生产保持。若仍要求全部生产队列不停止，则先仅在GPU2通过断点有界切换做单卡内核/批处理验收，不承诺单卡能达到10倍。

## 一手来源

- [Transformers Qwen3.5说明](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)：可选快速内核缺失时回退更慢且更耗内存的PyTorch实现。
- [FLA官方仓库](https://github.com/fla-org/flash-linear-attention)：Gated DeltaNet训练内核、依赖和安装说明。安装支持不等于本项目数值/性能通过。
- 本地运行代码：ordinary_baseline_v2/runner.py、data.py；实际环境transformers/models/qwen3_5/modeling_qwen3_5.py及integrations/hub_kernels.py；原失败报告triangular_gpu_v1/REPORT_ZH.md。
