# Real mechanism family CPU loader V2

已通过 28 项 CPU 数据接口测试；不是模型、神经 query encoder、训练或导航验收。共享数据和 SFT 未修改。本目录代码 stdlib-only。

## 复现

从项目根执行：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/parallel_readiness/v2/loader/run_acceptance.py
```

`loader.py` 可用 `importlib.util.spec_from_file_location` 按绝对路径导入。不要改 shared compiler；不要把本族混入正在运行的 ordinary SFT。

## 策略接口：必须跑完整历史

```python
dataset = FamilyLoader()  # 默认核验封存文件、compiler锁、cross-cell和CE lineage
for task_id, history_id in dataset.prefix_groups():
    memory = zero_memory()           # 外部模型caller负责，不由CPU loader伪装验证
    for item in dataset.iter_prefix(task_id, history_id):
        p = item['policy']           # 只把这一个字段送给策略
        if p['memory_reset']:
            memory = zero_memory()
            clear_all_model_caches()
        memory = policy_update(memory, **p)
    # 到此才可使用该task/history对应3个query，不能把query输入上面的policy_update
```

`policy` 精确只有四个字段：

- `instruction: str`，已登记V3模板原文。
- `rgb: tuple[bytes,...]`，一或两个224×224×3 C-order uint8原始像素，不含路径/hash/metadata。可由模型适配器转成image/tensor；本节点不调用PIL/torch。
- `executed_actions: tuple[str,...]`，最多8个真实已执行动作，不含当前target；只能MOVE_FORWARD/TURN_LEFT/TURN_RIGHT。
- `memory_reset: bool`，仅流内第0步为true。

`control.decision_step` 只服务调度/审计，不是模型输入。不要tokenize整个item，更不能tokenize `FamilyLoader` 的内部metadata。Python对象层面的字段隔离不是安全沙箱；模型端真实cache/batch隔离仍需单独测试。

当前6条流是2任务×3历史，每条207步（0..206）。换任务必须重新从zero memory计算；不得只读共同边界的最后两帧就声称模型用了历史。输入JSONL乱序可规范化，重复/缺失/不一致会拒绝；模型调用时间不能打乱。

## 查询与标签：两个独立接口

```python
query = dataset.query_input(sample_id)  # 训练reader输入；在prefix memory完成之后
label = dataset.label(sample_id)        # target + bce_mask；不是reader输入
```

query为保序typed tuple列表：`(3, action_code, repeat)` / `(4, category_code, room_code, 256, 2)` / `(5,)`。固定schema和coordinate版本由严格验证保证，integrity refs、trace/sample/house ID均被剔除。类别chair/sink/bed/TV映射固定20/21/22/23，不随任务角色重映射。

真实query有129/146/149个序列项，展开flatten整数数目495/560/577。repeat和256是数值字段，不应盲目充当分类embedding ID。**G2的Embedding(8)+mean是synthetic probe，不能直接接此接口。** 下一模型接口需为typed字段定义类别/数值处理，并保留序列顺序；此处未实现或验证神经queryencoder。

pass与fail均是有效BCE标签，分别1/0；unknown或rejected输出target=None、bce_mask=0。mask=0不得参与BCE均值分母；当前真实族没有unknown/rejected，相关行为为内存扰动单测。

## 两种训练调度，不能混用权重

`sample_grouped([dataset], count, seed)` 按family→task→valid cell分别均匀采样，返回的ID只作loader join。当前只验一个族的运行，不是多族经验验收。不能将3个确定性replay seed当独立样本或拆split。

动作CE用 `iter_action_stream(sample_id)`：每个cell独立从0重放，0..205仅warming context/CE mask0，206开始监督真实a_t；直到STOP。每项包含 `policy`、`control`、独立 `action_supervision`。读取下一个决策时才扩展下一真实观测；不返回整段未来trajectory给policy。为了保持因果memory，应从流开头迭代，不直接跳到后半段target。

每个cell每动作遍历周期至多遍历一次，以ACTION_DEDUP唯一owner为CE mask。当前1410个owner、2064个含masked目标；6个fail的动作CE全部0。先验证owner key、task/history/step/context/真实trace lineage以及target action，不只去重ID。

有放回BCE采样的cell频率不能乘到动作CE：BCE grouped sampling与独立owner action epoch分开调度。正式训练的损失权重、采样预算、warmup梯度/TBPTT与M1/M2/M4共享数据尚需训练协议冻结；本loader不替调用者决定。

## 已验与未验

见 [真实形状统计](REAL_DATA_SHAPES.json)、[28项日志](TEST_LOG.txt)、[结果](result.json)和[报告](REPORT_ZH.md)。核验包含所有474RGB的NPY格式和raw pixel SHA、全部6prefix与18动作流、ID/未来query/未来trace扰动不改变当前policy、坏hash/时间/owner/mask拒绝。

单族仍是old_line_exposed/interface_only，存在已披露的<=1e-5数值共同状态重建。CPU顺序通过不等于207步Qwen梯度、TBPTT有效、跨样本attention隔离、记忆可学习、泛化或导航收益。这里不授权扩量/机制训练，也不改变外部SFT协议。
