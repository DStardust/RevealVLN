# 三包实现接口规格

状态：`STATIC_SPEC_ONLY`。这里给张量流、断言和伪代码，不包含算法源代码；Qwen wrapper、LoRA、模拟器和梯度均未运行。

## Package 1：family compiler / validator

### 输入与输出

输入：固定 simulator/source commit、scene 四件套完整哈希、传感器与动作配置、seed、任务模板、候选 region/object IDs、候选 `u/s`、history/continuation action traces、阈值版本。输出分离存储：

1. `policy_input`：仅 instruction、因果 RGB 内容引用、实际执行 action/collision、reset 与 sensor profile；
2. `supervision_only`：scene/family/split/坐标、程序、semantic event、q/Y、动作监督与 mask；
3. `family_manifest`：house 级分组、交叉矩阵、来源、权重和失败统计。

cache identity 必须是下列规范化 JSON 的 SHA256：asset hashes、simulator+lab commit、compiler commit、action/sensor config、seed、task/threshold version、所有 trace action 和 reset strategy。不能只用文件名或 scene id。

### 编译伪流程

```text
validate category names and region/object membership
for each history candidate:
  reset from canonical initial state
  step every history action; save RGB/semantic hashes, collisions and events
  assert physical arrival at u
  step identical 8-step public tail; assert arrival at s
for every task × history × continuation cell:
  reset; replay full history + public tail + continuation end to end
  evaluate ordered program over complete event log
  emit pass/fail/unknown; never infer missing semantics
cross-check three seeded repeats and shared-state certificate
only then emit non-synthetic records
```

`Agent.set_state()` 不是完整 world snapshot，默认禁用于证书 replay。只有 reset+action trace 与 state restore 在 frame/RNG/static fingerprint/sensors/observations 上都等价后，可作为缓存优化；原始 end-to-end replay 仍保留。

失败码最少包括：`ASSET_HASH_MISMATCH`、`SEMANTIC_SCENE_MISSING`、`CATEGORY_MAPPING_UNVERIFIED`、`REGION_AMBIGUOUS`、`WITNESS_MISSING`、`NO_NAV_PATH`、`ACTION_COLLISION`、`HISTORY_TOO_LONG`、`CONTINUATION_BUDGET_EXCEEDED`、`MERGE_POSITION_MISMATCH`、`MERGE_YAW_MISMATCH`、`SENSOR_POSE_MISMATCH`、`OBSERVATION_MISMATCH`、`STATIC_STATE_MISMATCH`、`NONDETERMINISTIC_REPLAY`、`EXPECTED_LABEL_MISMATCH`、`POLICY_FIELD_LEAK`、`LICENSE_BLOCKED`。生成失败不得静默转成负样本。

### 自动静态/数据检验

- JSON schema、内容哈希、相对资源引用、sample_id 唯一与一对一 join；sample_id 不进入 tokenizer。
- policy record 顶层 key 精确等于 schema 白名单；递归禁止 scene/house/family/split/pose/position/coordinate/program/event/query/q/outcome/Y/path/filename。
- `causal_cutoff` 后的 observation/action 不可达；q 和 outcome 文件由独立 dataloader 在 memory 已计算后连接。
- action 必须属于四动作表，`executed=true`，步号单调；碰撞 cell 拒绝。
- history 实际 action 数完全相等；公共尾段 actions 与逐步 observation hashes 相等；残余预算相同。
- event step、semantic/RGB hash、instance-region-category 关系一致；未知与失败分开。
- 每个 house 只在 train/development/confirmation 之一；`old_exposure` 独立于 split，旧暴露 house 不进入 clean confirmation。
- family 先均匀采样，再在 family 内均匀 task，再均匀有效 cell；BCE 在 family-task 内平均再跨 family 平均。action CE 依 unique target sequence 去重，不乘以 cross-cell 数。
- 捷径审计：打乱 sample id/存储顺序/文件名不改变张量；遮掉关键过去应只通过 memory 影响预测；未来 q 不可进入 prefix；同一 q 的字符串模板不能唯一编码 Y。

## Package 2：Qwen3.5-2B VLN policy

### 冻结版本与已核接口

- model `Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`；Transformers `v5.15.0`。
- official class `Qwen3_5ForConditionalGeneration`；text hidden `d=2048`、24 layers、BF16；vision encoder 输入由 processor 的 `pixel_values` 和 grid 表示。
- conditional model/text model 接受 `inputs_embeds`、显式 position ids；输出 hidden state。官方默认可 cache，但本项目必须关闭。
- MRoPE 与视觉 grid 耦合。先用未替换的 `input_ids + mm_token_type_ids + grid` 计算位置，再替换预留 memory placeholder embedding；这一 wrapper 组合为 `UNVERIFIED`，须 G2 smoke。

### 有界运行时状态与张量流

固定首验参数：`K=8` 个连续 memory slots，每槽 2048 维 BF16；最近 RGB `W_rgb=2` 帧、最近 executed actions `W_a=8`，公共尾段 8。选择依据是“最小有界、能将关键事件移出近期窗口”的机制需求，不声称最优。裸 memory payload 为 `8×2048×2=32,768 bytes/episode`，不含激活和优化器。

每个 navigation step：

```text
processor(instruction, recent causal RGB) -> input_ids, pixel_values,
    image_grid_thw, mm_token_type_ids, attention_mask
official embedding + get_image_features -> multimodal token embeddings
official rope-index routine(original token ids/types/grid) -> position_ids
replace K reserved old-memory placeholder embeddings by m_(t-1)
append K learned write-query embeddings + 1 learned action-query embedding
language_model(inputs_embeds, position_ids, attention_mask,
               use_cache=false, past_key_values=None)
write hidden[K] -> shared bounded writer -> m_t [B,8,2048]
action hidden[1] -> linear action head -> logits [B,4]
```

每个 episode/task 开始清零 `m`，且 `model.rope_deltas=None`；绝不跨 episode 持久化 hybrid cache、hidden states 或图像 tokens。只有显式 `m_t` 是历史状态。策略输入仅为 language、causal RGB、actually executed actions；semantic/pose/Y/q 不可进入。

writer 对所有时刻共享参数，输出经过有界归一化/门控；memory 不 detach 的短序列路径为：

```text
early RGB -> official vision/text layers -> write hidden -> m_t
          -> later memory placeholders -> later hidden -> action loss
          -> training-only query reader -> continuation BCE
```

训练专用 continuation reader 是独立小网络：

```text
r_(i,k,j) = Reader(stop_gradient? NO for m_i,k,
                   m_(i,k), causal instruction encoding recomputed for I_k,
                   isolated query encoder(q_j))
logit_y = head(r)
```

q 只在 prefix memory 计算完成后进入 Reader，不回调 Qwen prefix、不修改 action cache、不写 memory。每个 crossed instruction `I_k` 从同一原始 RGB/action history 重新前向计算 `m_(i,k)`；不得把旧任务的 memory 复用给新任务。部署完全删除 Reader 和 q 管线，但保留 memory writer/slots 的实际开销。

### 可训练参数与精度

- BF16 依据官方 config；视觉塔在首个模型门冻结。
- language modules 候选 LoRA `r=8, alpha=16`：全注意力 `q_proj,k_proj,v_proj,o_proj`，linear attention `in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`，MLP `gate_proj,up_proj,down_proj`；只有 G2 枚举实际模块并通过 PEFT forward/backward 后才能冻结列表。
- memory placeholders/write/action queries、writer、action head、training-only Reader/head 可训练。静态估算 LoRA 与 heads 合计约 8–10M 参数，G2 必须输出实际 trainable count 和参数名差异。
- 输入 token 粗估：instruction cap 256；两张 224 图在 patch/merge 假设下约 49 tokens/图；memory/query/action 少于 32，总计 <400。视觉 token 数必须由 processor smoke 实测，不能用估算作 contract。

### G2 最便宜的接口验收（不属于 G1）

1. 固定一个 synthetic 16-step prefix，`use_cache=false`，逐步检查只保留 `m`；更换未来 q 不改变已计算 memory/action logits。
2. 同一 RGB/action 换 `g_D/g_K` 重新前向，确认 instruction 条件化 memory 不被复用。
3. 修改关键早期 RGB，而保持最近窗口、q、后缀相同，确认 `m` 变化；不得要求未训练 logits 有正确语义。
4. BCE backward：早期 event RGB embedding/vision-to-writer 路径和至少一个 writer/LoRA 参数梯度 finite 且 nonzero；optimizer step 后目标参数确实变化。无 detached memory。
5. MRoPE/placeholder/grid shape、BF16、gradient-checkpointing、LoRA module matching 与 reset/cache assertions 全部通过。

长序列另设 gate：可采用跨固定 chunk 的 state carry + truncated BPTT，但必须报告 truncation horizon、detach 边界与哪些早期事件拿不到梯度。在短序列通过前不声称长历史可训练。

## Package 3：training / closed-loop evaluation

### 训练读取与损失

loader 先按 `family_id` 取 family，均匀取 task，再均匀取有效 cross cell。policy 与 supervision 分文件，以 sample_id join；join 后先计算 prefix memory，再将 q 送独立 Reader。

```text
L = L_action_unique + lambda_cont * mean_family(mean_task(mean_valid_cell(BCE)))
```

- history prefix、负/unknown continuation 的 `action_loss_mask=0`；满足任务的 normal/recovery continuation 和认证普通 R2R 动作为 1。
- action target sequence 只计一次；cross matrix 只增加 continuation BCE 证据，不能重复放大 CE。
- `lambda_cont`、普通/机制 mixture 和 optimizer/steps 尚未冻结；必须在同一初始化比较 M1/M2/M4 前预注册。
- M1 action-only；M2 用冻结主线定义的 program-state head；M4 用本文 continuation reader。三者必须同 base revision/init、数据池、采样、token/step、action masks、seed 和资源上限，唯一机制差异通过字段 diff 检查。

### 闭环与 episode 记录

每 episode 硬 reset simulator、memory、Qwen rope/cache 状态和 dataloader task state。只给当前 instruction、最近 causal RGB 与已执行动作；动作经四动作表执行。STOP 或固定 max steps 终止。逐 episode 记录匿名 episode id、house split、seed、完整 actions/collisions、观测内容哈希、task success/SPL 等契约指标、memory reset/carry 断言、模型/config/data hashes、wall time/peak memory 和所有 failure codes。

模型接口验收、数据 G1 与首次导航收益试验必须是三个独立节点：

1. G1 数据工程 replay：不加载 Qwen，不报告 navigation gain；
2. G2 模型/梯度接口 smoke：synthetic/minimal tensor，不报告导航收益；
3. G3 经主 agent 冻结样本量和效应门槛后，才做 M1/M2/M4 闭环试验。

## M1/M2/M4 可比较配置字段

每个 run manifest 必须含：model id/revision、Transformers/PEFT commit、base init hash、processor config、vision freeze、精度、LoRA module/rank/alpha 与实际 trainables、memory K/d、RGB/action windows、action尺度、data/family manifest hash、house splits/old exposure、ordinary:mechanism mixture、family/task/cell sampling、unique-action dedup、optimizer/lr/schedule、token/step/episode budget、seed、action-mask hash、auxiliary type/weight、reset/cache policy、max episode steps、evaluation episode list、hardware、peak resources、failure handling。字段缺失或除预注册机制差异外不相同，则不可归因比较。

确认样本量和科学效应阈值当前为 `UNVERIFIED/NOT_FROZEN`：没有本线方差或闭环结果，不用任意数字伪装统计依据。
