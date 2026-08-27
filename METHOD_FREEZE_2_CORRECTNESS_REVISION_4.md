# RevealNav Method-Freeze-2 正确性修订 4

**修订编号：** `MF2-CR4`  
**日期：** 2026-08-24  
**状态：** Phase-0C 指令子句定位修订；最终人工复核关卡通过前禁止特征生成和训练。  
**替代范围：** 仅暂停 `MF2-CR3` 直接使用完整 instruction 与四张局部图填写
`instruction_clause` 的流程，并替代其“当前审核包已可直接开始三轨审核”的就绪结论。
`MF2-CR1`、`MF2-CR2` 以及 `MF2-CR3` 的来源真实性、审核者类型披露、固定事件
集合、阈值、数据边界、模块、验收阈值和非结论要求保持不变。  
**规范边界：** `FROZEN_SPEC.md` 与 `PHASE0_PROTOCOL.md` 继续保持逐字节不变。

## 1. 修订原因

RxR train 的 `train_guide.json.gz` 提供完整 instruction 和参考路径，
`train_guide_gt.json.gz` 提供位置、动作和 forward step，但当前授权材料没有
“instruction 字词/子句到 waypoint 或动作时刻”的官方对齐。旧审核板只展示完整长指令
与局部的 `P,D1,D2,D3` 画面，审核者仅凭四帧选择最短子句时，可能把路线其他阶段的
描述错误归到当前 Reveal 候选。这一缺口会污染
`branch_dependent_instruction` 和 `target_branch_matches_instruction`，不能由表单结构正确
或视觉图更清楚来消除。

因此旧 `phase0c_hybrid_review` 空白表单保持为历史证据，但自本修订起暂停填写；其中任何
新填写结果都不能满足 Phase-0C，也不能授权训练。

## 2. 固定的 MLLM 子句定位输入

只对既有 35 项 RxR-train 私有候选进行处理。输入构造固定为：

1. 完整原始 instruction；
2. 按标点确定性划分的、带字符区间和 SHA-256 的逐字子串；模型不得改写子串；
3. 同一 episode 沿参考低层轨迹按时间排列的 RGB 图像序列；
4. 全局均匀抽样 20 帧，并在各事件 `P,D1,D2,D3` 周围半径 6 密集抽样，四个因果帧强制
   纳入；
5. 每帧显示稳定的 `P####` 帧号和执行动作，原图及其 SHA-256 在项目内保存；
6. 输入为有序独立图像，而不是服务端不透明抽帧的视频。

固定输入 manifest：

`artifacts/phase0/phase0c_clause_grounding_mllm/MLLM_CLAUSE_GROUNDING_INPUTS.json`

SHA-256：
`d576b49122b7b3a90c71d6ff6648926afd6b063f433e8feb9a0f85488bb1f4ca`

输入验收：

`artifacts/phase0/phase0c_clause_grounding_mllm/MLLM_CLAUSE_GROUNDING_INPUTS_ACCEPTANCE.json`

当前验收状态必须为 `PASS`。输入只来自 RxR train，不得读取或推断
`val_unseen`、`test` 或 `test_challenge`。

## 3. MLLM 的固定权限和输出

第一提议轨道固定为 DashScope OpenAI-compatible API，base URL
`https://dashscope.aliyuncs.com/compatible-mode/v1`，请求模型标识
`qwen3.8-max`，温度为 0。真实运行必须保存服务端返回的模型标识、usage、请求证据指纹、
逐事件原始结构化结果和重试记录；服务拒绝该模型或视觉输入时必须失败关闭，禁止静默换
模型。

MLLM 每项只能返回：

- `UNIQUE_MATCH`：选择 1–3 个相邻的确定性子串；
- `MULTIPLE_PLAUSIBLE`：保留全部合理的相邻子串组；
- `NO_MATCH`；
- `INSUFFICIENT_VISUAL_EVIDENCE`；
- 支持结论的已有 frame ID、置信度和简短理由。

MLLM 结果是 **proposal**，不是官方 RxR 对齐、人工标注、RevealEvent 标签或训练真值。
它不得改变事件集合、候选拓扑、因果窗口、资源代价、expiry 定义或六项最终审核字段；也
不得凭自身输出授权训练。API 密钥不得写入源码、命令、日志或交付物。Matterport 派生
媒体仅按用户授权发送到上述固定处理端点，不得公开发布或转发到其他服务。

## 4. MLLM 后的人工审核

只有结构、来源和哈希均通过验证的 proposal 才能进入新版审核板。新版板必须同时展示：

- 完整 instruction，MLLM 提议子句仅高亮，不得隐藏上下文；
- 所有被 MLLM 引用的帧及其稳定 frame ID；
- 当前事件的局部 `P,D1,D2,D3` 因果画面和局部俯视图；
- `UNIQUE_MATCH`、歧义、无匹配或证据不足状态；
- 明确的“MLLM 仅作建议”提示。

H 轨道的人类审核者必须逐项确认或推翻子句定位，并独立填写六个冻结判断。以下任一情况
必须拒绝：非唯一定位、人类无法从完整上下文确认、视觉窗口不足、分支语义不一致，或
MLLM 与人类对关键子句存在未解决分歧。禁止自动接受高置信度 proposal。

M1/M2 若继续用于 CR3 定义的独立 VLM 审计，必须与用于子句预定位的调用在提示词、输出
文件和来源字段上分开，且不得把同一次 proposal 复制为独立审核票。若相同模型家族同时
承担预定位和 M1，论文与内部报告必须披露其相关性，不能将其当成独立人类证据。

## 5. 修订后的验收与训练边界

Phase-0C 只有同时满足下列条件才可提出新的版本化 canonical method 并请求训练授权：

1. 35/35 项存在经过 schema、哈希和来源验证的 MLLM proposal；
2. 35/35 项存在封存的人类 H 审核，且人类可查看完整上下文并确认或拒绝子句；
3. CR3 规定的 M1/M2 独立审计和真实 reviewer type 披露仍完成；
4. 三轨六项布尔判断完全一致接纳至少 15 个唯一事件，覆盖至少 10 个 MP3D 场景；
5. 接纳集合中子句歧义、缺失字段、哈希漂移、未解决人机分歧均为 0；
6. 新版汇总器明确验证 proposal 不被当作人工票或官方标签。

在上述关卡通过前，语言依赖 Reveal Event 的已验证数量仍为 0，整体保持 `NO_GO`，不得
生成训练特征、启动 imitation/RL/GRPO 训练、报告 benchmark 改进或提出 SOTA/CVPR
录用概率主张。

## 6. 当前非结论

本修订及 MLLM 输入构造只证明审核输入可追溯，并修复长指令与局部视觉窗口之间的定位
风险；不证明 35 项中任何一项是真实 RevealEvent，不证明该方法优于基线，也不构成
CVPR 创新性或录用保证。
