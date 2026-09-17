# 真实机制数据 CPU loader 验收

结论：`CPU_LOADER_ACCEPTANCE_PASS`，回交主agent审核。28/28测试通过，0失败/0错误，实际5.49秒。没有GPU操作、模型调用、仿真、训练、下载或环境修改，scientific_pass=false。

## 实际完成

- 6条task×history完整prefix流，各207步，总1242决策，0步reset、完整因果窗口和task重置均已验。
- 474个不同RGB的NPY头/dtype/shape/长度/原始像素hash全部复核，shape224×224×3 uint8；策略仅拿raw pixels，无路径/hash/semantic/pose/程序state/ID。
- 18格实际监督12pass/6fail；query和Y分离，所有fail动作CE mask0。
- 1410唯一CE owner、2064个含masked动作target的key/context/lineage/真实动作全部核对；18条动作流含warmup共5772决策。
- 真实query保序typed整数接口已实现，3个不同query长度129/146/149，固定类别词表，不含答案或integrity refs。
- ID改名、future query独立替换、future trace内存扰动、返回payload修改均不改变原始当前policy；乱序prefix恢复，重复/缺失/坏hash/越界时间/错误owner/错误mask被拒绝。
- 封存源数据SHA256清单及5个compiler源码锁前后均通过。所有写入局限本loader目录。

首轮全部测试通过，未进行结果驱动协议修改或隐藏重试。SPEC在实现和运行前落盘，CODE_LOCK记载运行文件hash。

## 接口裁决

数据加载器可以交给后续机制模型接口节点复用，不应接入正在运行的普通SFT。

发现的接口重要边界不是数据FAIL：真实query远大于G2 toy query，不能直接用G2 Embedding(8)+mean；后续须验证typed类别/数值与顺序编码。CPU loader没有神经encoder，也没有Qwen长历史训练和梯度结果。

本节点将policy、control、query、label和action_supervision按不同API/字段分开，是可审计的数据接口，不是能阻止恶意caller读取内部metadata的系统安全边界。模型adapter必须只接policy白名单，必须实际reset memory/cache；此节点不能替代batchattention/长历史神经验收。

当前1个家屋1个interface_only族，不构成机制训练集就绪或跨场景统计证据。数值规范化历史限制继承封存数据，原餐椅失败未改写。多族扩量与M1/M2/M4公平训练必须另行冻结并准入。
