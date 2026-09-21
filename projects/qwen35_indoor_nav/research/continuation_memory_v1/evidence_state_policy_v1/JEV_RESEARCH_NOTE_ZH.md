# Jev 与执行状态研究：一手核查及可测问题

核查日期2026-09-21。按“上周发布Jev”暂定为TypeSafe的Jev；若用户指另一个对象，保留本记录并更正对象，不混称已经确认。

## 实际核查

TypeSafe于9月15日发布Jev，强调结构化决策而非文本生成。其发布页同时说明Doom演示接收结构化游戏状态，不是原始图像。这不能直接当RGB导航能力证明。[官方发布](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

官方接口返回Choice、Score、Noul等类型化判断。Choice/Score的confidence来自输出分布；格式合法不等于事实判断正确，分布集中也不自动代表在导航数据上已校准。[接口说明](https://docs.typesafe.ai/introduction)，[confidence说明](https://docs.typesafe.ai/confidence)

官方称RLCD面向校准决策；本次读到的说明不足以重建完整训练算法，不能把自己的BCE或置信门控命名为“复现RLCD”。校准描述的是分组频率关系，不是单次判断保证。[官方AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)

已读官方API调用入口，但没有调用服务、安装SDK、获取权重或上传实验数据。本记录不声称Jev代码/权重已公开，也不声称已测试其速度。[官方quickstart](https://docs.typesafe.ai/introduction/quickstart)

## 对当前工程的判断

我们本来就是四类logits直接决策，并不逐token生成动作文本。因此仅换成“类型化决策头”不足以产生明确新贡献，也不能据Jev营销延迟推算本地视觉推理会加速。

可测的具体问题是：在相关、重复且可能误识别的视觉证据下，模型何时真的知道历史条件已满足；它的ready预测在主动STOP时是否可靠。当前事件/状态BCE有类别权重，sigmoid输出不应先验解释为自然分布下已校准的概率。

本轮只加入`gpu_v1/calibration.py`：对同一批封存的自主轨迹计算事件与状态Brier、固定十箱可靠性、ECE，分动作和主动STOP；不选择阈值，不改动作，不调用Jev，不新增训练臂。评测真值只在轨迹封存后进入CPU统计。task_T未指定anchor的标签保持屏蔽。

## 后续候选（未执行）

若确有过度自信且错误STOP集中的证据，再单独比较：原始预测、简单校准强基线、把错误状态修正与不可逆STOP风险一起训练的方案。校准数据须来自FIT内完整家族隔离的校准折，不能反复用已暴露DEV调门槛。另报动作/观察预算与任务成功，不能用拒绝动作或延迟STOP换取漂亮的局部准确率。

直接调用Jev作外部决策器会引入网络、服务版本和额外语言表征变量；当前没有这样做。即便以后调用，也只能传学生实际可观测/可预测的状态，不能把Habitat实例、目标坐标或精确任务真值当输入。若借鉴其思想，明确引用；“不生成文字”“输出概率”“分类头”均不单列为本工作的发明。
