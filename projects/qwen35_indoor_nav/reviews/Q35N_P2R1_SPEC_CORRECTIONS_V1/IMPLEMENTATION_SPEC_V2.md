# 实现接口规格 V2

状态：`STATIC_SPEC_ONLY`。继承原三包结构，只补 R3–R6；不创建算法代码。以下官方 Qwen 对象有固定源码依据，但自定义序列 wrapper、PEFT、forward/backward 仍为 `UNVERIFIED`。

## Package 1：compiler 与两级 validator

### 数据流

输入为 runtime fingerprint、物理配置/hash、冻结 candidate、完整 executed trace、aligned RGB/semantic、V2 task/query。输出仍分为 `policy_input`、`supervision_only`、`family_manifest`，且任何 rejected discovery/final cell 单列 failure ledger。

validator 分两级：

1. Draft 2020-12 schema：类型、枚举、白名单、常量阈值、T/F/U 局部约束；
2. semantic validator：逐项实现 `DATA_SCHEMA_V2.json` 的 `X01–X13`，包括 array 长度、因果步号、三值程序、18-cell Cartesian completeness、query/trace 等价、ID 改名不变、同 q 异标签、M2 causal state、action dedup lineage 和 9 physical→18 task accounting。

只有两级全过才能由 `synthetic_spec_only=true` 转为 false。JSON 能 parse、schema 能 compile 或示例能 validate 都不代表实际证书通过。

原子事件和有序程序使用 `MINIMAL_FAMILY_SPEC_V2.md` 的强三值逻辑。`F` 必须有完整证据链；`U` 不进入 BCE，`REJECTED` 不产生 Y。validator 输出每个 assertion 的 input hashes、status、failure code，不能只给一个总布尔值。

## Package 2：Qwen3.5-2B policy、memory 与两个训练头

### 2.1 固定对象层级

固定 `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` 与 Transformers v5.15.0：

```text
Qwen3_5ForConditionalGeneration
└── model: Qwen3_5Model
    ├── visual: official vision encoder
    ├── get_image_features(...)
    ├── get_rope_index(...) / compute_3d_position_ids(...)
    └── language_model: Qwen3_5TextModel
```

G2 必须用对象 introspection 对固定源码逐名核验；不匹配即 `QWEN_OBJECT_PATH_MISMATCH`，不能退而引用其他 Qwen 版本。hidden `D=2048`、memory `K=8`、RGB window 2、executed-action window≤8、BF16；视觉塔首验冻结。

### 2.2 完整 token 序列与动作编码

每个决策时刻先由 official processor 生成 instruction + 两张 causal RGB 的 `input_ids_base/attention_mask_base/pixel_values/image_grid_thw/mm_token_type_ids_base`。随后在 **计算 position IDs 之前** 构造完整序列：

```text
S_full = [processor text/image sequence]
       + [OLD_MEMORY_PLACEHOLDER × K]
       + [EXECUTED_ACTION_STATUS_TOKEN × A, 0<=A<=8]
       + [WRITE_QUERY_PLACEHOLDER × K]
       + [ACTION_QUERY_PLACEHOLDER × 1]
L = L_base + K + A + K + 1
```

动作不是漏掉的旁路字符串。每个已执行 action 与 collision bit 映射为一个受控 added special token，例如 `EXEC_MOVE_FORWARD_OK` 或 `EXEC_MOVE_FORWARD_COLLISION`；四动作×二状态共 8 个，按真实时间顺序编码。计划动作、未执行动作、pose 和 future action 禁止进入。

tokenizer 变更、special-token IDs、embedding resize 后 shape/hash 都写入 G2 manifest。OLD/WRITE/ACTION placeholder 可以复用类别 token ID，但 embedding 替换按显式 position index 完成；slot-specific learned embedding 防止 8 个槽不可区分。

### 2.3 position/type/attention 与 shape 断言

1. append 后得到 `input_ids_full [B,L]`；`attention_mask_full [B,L]` 对所有非 padding token 为 1。
2. `mm_token_type_ids_full [B,L]` 保留 processor 为 text/image block 生成的值；memory/action/write/action-query 复制**同一 processor 输出中普通文本位置的 token type**，不得硬猜数值。没有可识别 text type 即失败。
3. 在 embedding 替换前，把完整 `input_ids_full`、完整 mask/type、原 `image_grid_thw` 交给固定 `Qwen3_5Model.get_rope_index` 路径；其内部 `compute_3d_position_ids` 处理视觉网格。期望并断言 `position_ids_full [3,B,L]`，但以固定源码实际返回为最终接口；rank/length 不符即失败，不手写另一 Qwen 的 MRoPE。
4. 不允许“先对 L_base 算 position，再 append 17+A tokens”。所有 old memory、actions 和 queries 必须已在 `L` 内且各有 position、attention、mm type。
5. official input embedding 得到 `inputs_embeds [B,L,2048]`；official `get_image_features(pixel_values,image_grid_thw)` 的 packed feature 数必须精确等于 image placeholder 数，再 scatter。视觉 token 数不再静态假设为 49/frame。
6. 将 OLD positions 替为 `m_(t-1)[B,8,2048]+old_slot_embedding[8,2048]`，WRITE positions 替为 learned `write_query[8,2048]`，ACTION_QUERY 替为 learned `[1,2048]`。executed-action tokens保留 resized official embeddings。
7. 调 `Qwen3_5TextModel(inputs_embeds=..., position_ids=position_ids_full, attention_mask=attention_mask_full, past_key_values=None, use_cache=false)`；要求 `last_hidden_state [B,L,2048]`。读取 WRITE indices 得 `[B,8,2048]`，经 shared writer 得 `m_t`；ACTION_QUERY index 得 `[B,2048]`，action head 得 `[B,4]`。

每个 episode/task 开始 `m=0`，模型侧 rope delta 清空；每步 `past_key_values=None/use_cache=false`。返回的 hybrid cache/rope delta 不持久化。除显式 `m_t` 外，不能保存旧 hidden/image tokens/KV/linear-attention recurrent 或 conv state。

### 2.4 structured continuation reader（M4）

q 采用 `continuation_query.v2`。semantic projection 的 canonical serializer 只编码 schema version、agent-relative frame、枚举 movement/repeat、observable category/room、顺序和 STOP；storage/sample/continuation IDs 以及 action/RGB integrity hashes 均不进入 tensor。独立 query encoder 输出 `q_emb [B,Lq,Dq]`，reader 输入为已算好的 `m_(i,k) [B,8,2048]`、因果 instruction pooling 和 q_emb，输出 binary logit `[B]`。

执行顺序是：

```text
m, action_logits = policy(prefix, instruction)   # q 不存在
freeze that forward's causal boundary
q_emb = query_encoder(canonical_query)
y_logit = continuation_reader(m, causal_instruction_pool, q_emb)
```

q 不调用 policy prefix、不写 m、不改 action logits/cache。换 `I_k` 必须从原 RGB/action history 重算 m。部署删除 query encoder/reader。

### 2.5 强程序状态监督（M2），与物理匹配分开

物理 shared-state certificate 只回答 histories 是否在 `u/s` 同物理状态；M2 监督回答当前 task program 进度。两者使用不同字段、head 和证据状态，绝不再用 `strong_shared_state_match` 指代 M2。

在每个 action decision `t`，只用 `events<=t` 自动产生三类 causal target：

- `WAIT_ANCHOR`：完整证据下 anchor 尚未 SEE2；
- `WAIT_TERMINAL_WITNESS`：anchor 已完成，但当前 terminal bed SEE2 未就绪；
- `READY_TO_STOP`：anchor 已完成且 `SEE2(bed,bedroom)@t=T`，应在 t 选择 STOP；
- 证据不完整为 `UNKNOWN`，mask=0，不成为第四个可学习真值类。

M2 head 与 M4 使用相同 policy/memory backbone，输入同一 causal `m_t [B,8,2048]` 与当前 action-query hidden `[B,2048]`，不读 program truth、q、semantic 或 future。训练输出 `state_logits [B,T,3]`；target `[B,T] int64`，mask `[B,T] bool`，masked cross entropy 先在 episode 内有效时刻平均，再按 family/task 平均。M2 head 是 training-only label reader，部署移除；它不是 oracle policy input。

M1 无 auxiliary；M2 使用上述 `L_state`；M4 使用 grouped continuation BCE。三者保持同 policy architecture、memory capacity、action CE/data/seed/token/optimizer budget；分别登记 auxiliary head 参数与算力。M2 state target 和 M4 Y 都由同一完整证书派生，但一个只看时刻 t 的前缀，一个检查未来 continuation。

### 2.6 frozen vision gradient probe

视觉塔冻结时其 parameters `requires_grad=false`，所以 `.grad is None` 是预期，不能据此否定历史梯度路径。G2 分开两个检查：

1. **feature sensitivity probe**：把早期 event 的 frozen `image_features` detach/clone 成显式 leaf `v_probe.requires_grad=true`，保持其后 wrapper 不 detach；对 continuation BCE backward 后要求 `v_probe.grad` finite 且 norm>预注册数值下限。这只证明损失对 frozen feature 有微分敏感性，不证明视觉参数可训练。
2. **trainable parameter probe**：不解冻 vision，在真实训练图上要求 early-step writer/slot path，以及至少一个预注册 LoRA parameter 和 continuation reader parameter 的 gradient finite/nonzero；optimizer step 后这些目标参数 hash/value 改变。vision parameter gradients 仍应为 None。

probe 的数值下限、loss scaling 与 batch 在 G2 执行前冻结。本轮不运行 backward，状态 `DEFERRED_RUNTIME`。

## Package 3：sampling、action dedup 与 losses

### 3.1 只移除 cross-copy 的 action key

每个动作监督位置的离线去重 key：

```text
sha256(canonical_json({
  task_hash,
  history_hash,
  decision_step,
  causal_policy_context_hash,
  target_continuation_lineage_hash,
  target_action,
  action_mask_version
}))
```

`causal_policy_context_hash` 覆盖 instruction、截止时刻、因果 RGB/action hashes、memory reset/config；`target_continuation_lineage_hash` 覆盖 demonstration trace 与该 decision 在 trace 中的位置。task、history 或 time 不同，即使 action string 相同也保留。只有由于同一 task/history/demonstration decision 被多个 crossed q/Y cell 复制且 key 完全相同，才选择规范 owner 计一次 CE；其他 copy mask=0 并登记 `CROSS_COPY_DEDUP`。该 key 是监督侧权重控制，不进入 policy。

### 3.2 losses 与重置

```text
M1: L_action_unique
M2: L_action_unique + lambda_state * grouped_masked_CE(state_logits,z_t)
M4: L_action_unique + lambda_cont * grouped_valid_BCE(y_logit,Y)
```

fail/unknown/rejected cell 不给正 action CE；unknown/rejected 不进 BCE。family→task→valid cell 分层均匀；action owner 单独从 unique action table 采样，避免 cross matrix 改变其频率。每 episode/task 重置 simulator、m、rope/cache 和 task evaluator。

闭环 episode 与模型接口验收仍属于后续独立节点；本轮不实现、不运行，也不把任何静态 shape 规格写成 Qwen runtime 已通过。
