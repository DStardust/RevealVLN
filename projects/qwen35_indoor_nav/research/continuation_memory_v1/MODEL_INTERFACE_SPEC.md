# 未来模型接口规格（未实现）

当前可调用基座是 `sft_acceptance/ordinary_sync_recovery_v1/model.py:PolicyV3.forward_batch`，
输入7种张量，输出[B,4]；其中没有memory。其processor/DecisionDataset/make_collate可作为
短窗编码接口。实际config的hidden_size=2048、24层，其中6个full_attention层含现有
q_proj/v_proj rank8 LoRA；18个linear_attention层不新增LoRA。本轮不改这些代码。

旧 `sft_acceptance/v1/policy.py:Policy.step(instruction,images,executed,memory)` 有可读8槽
原型，返回memory/logits，但含旧seed、硬编码设备、writer等旧配置；不能直接加载它冒充
当前best4k，也不能从旧代码存在推断跨步训练通过。

## 明确的参考接口

以下是未来可实施的两阶段参考规格；不在本轮实现或用效能搜索结构。研究预注册前须用
另行授权的无效能接口/资源检查确定能否承受成本，然后全部研究臂冻结同一实现。

```python
reset(batch_size, instruction, *, owner) -> MemoryEnvelope
update_memory(observation: PolicyObservation, previous: MemoryEnvelope) -> MemoryEnvelope
action_logits(observation: PolicyObservation, current: MemoryEnvelope) -> Tensor[B,4]
encode_prefix(prefix: PrefixTrace) -> MemoryEnvelope
auxiliary_logits(current_memory, instruction, train_query) -> Tensor[N_queries]
```

MemoryEnvelope的唯一可学习输入是[B,8,D]连续slots。owner、task绑定和最新step仅作
外层校验；不进入Qwen或任何读出器。D从实际配置读出；当前8×2048 BF16理论槽存储
32768字节/episode，不含激活、缓存和元数据，不是实测峰值显存。

`update_memory`复用已核对的原图像/文本编码与Qwen层，在嵌入序列中依次放原指令/短窗、
旧记忆槽、原已执行动作后缀、8个学习write-query；取write-query最后隐藏状态，经共享
writer(D→D，无bias)生成m_t。不传query。初始slots为零，slot位置向量可学习。

`action_logits`用同一Qwen权重、同一原短窗和显式m_t，放memory槽与原action-query，
不再次写memory；原4类action_head读最后action-query隐藏状态。这消除“只返回新memory
但当前动作没有读取它”的接口歧义。每决策2次主干前向是明确成本；不得按1次报告。
未来若要采用单次联合前向，须在预注册前另审实际张量依赖和成本，不能训练后为分数切换。

共享可训练范围：现有LoRA、exec_embed、action_query、action_head，加old_slot[8,D]、
write_query[8,D]、writer[D,D]。冻结基座、视觉塔及其余attention参数。B1/B2/B3/Ours
运行结构、参数初始值、memory初值与处理器完全相同；发布一份初始state_dict及全参数
shape/dtype/SHA清单。资格基座SHA当前为null：不得默认永远用尚未通过G1的best4k。

R是训练专用的小型有序query编码器＋二分类读出，不生成未来图像。参考query编码先以
统一词义序列取冻结token embedding，再用单层GRU(128)和小MLP；序列顺序与repeat须保留。
与B2/B3尽量匹配辅助读出容量，记录实际参数数和前向成本；最终宽度/词表/loss系数在
预注册前固定，当前预算null阻止训练，不能暗中做超参搜索。

动作前向和memory图不得接收query、程序或target。R部署时删除，m与写入/动作计算保留。
未来train.py按同轨迹池索引采样，动作CE对共同去重动作owner取mean；辅助BCE先族内有效
格mean，再对有有效格的族mean，UNKNOWN无损失。B2按已知组合状态字段mean、B3按已知
未来事件mean，再族mean。lambda和action/aux配比必须共同预注册，报告各臂实际有效标签
数、前向数、token数、更新时间和任何辅助成本差异。

本轮无神经模型实现、反向、优化或动作实测。新规格不能作为部署配置或新checkpoint。
