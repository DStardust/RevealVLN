TRAINING_RUNNING_EFFECT_UNKNOWN

V9三种子完整闭环不采用。V10已启动真实GPU轻量训练，未得到完整方法比较或导航收益结论。

修复对象是动作读出：将当前2048维因果特征与原512维记忆输入同一128宽MLP，末层零初始化。N0将记忆输入置零；B1/B2/Ours读取真实记忆。四臂使用相同初始化、原V9采样序列、轨迹池、600更新和三个种子；保留原损失、编码器、输入、STOP保护和成功定义。N0的记忆写入/递归参数不应更新。

CPU已完成5项真实缓存前反向/梯度测试，闭环编排9项测试。训练与闭环源文件在启动前分别复核22/33个冻结指纹。实际7200更新完成数以训练日志为准，不把计划数写成实测。此轮复用已审计因果缓存，计划新Qwen特征前向0、底模更新0。

训练完成后读回全部12份固定最终权重，再按预先固定三种子、每种子100组、原生+B2+Ours+B1+N0五臂执行。完整计划1500次episode，当前尚未启动该导航。只有真实配对SR和对照能支持收益；N0改善不代表记忆或交叉监督贡献。

协议与入口：research/continuation_memory_v1/contextual_readout_v10/{PROTOCOL.json,launch.py,run.py,train.py,model.py}；closed_loop_bench/ordinary_memory_transfer_v10/{PROTOCOL.json,launch.py,evaluate.py}。原始SEE2资产准入限制、数据已暴露状态及旧失败记录不变。
