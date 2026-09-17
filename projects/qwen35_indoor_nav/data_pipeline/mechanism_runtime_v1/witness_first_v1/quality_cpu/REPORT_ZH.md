# 特殊族质量裁决 CPU 模块（批次门槛待主审冻结）

交付 [quality.py](quality.py)、[test_quality.py](test_quality.py)。14 项 CPU 测试通过（1.296 秒），包括原 FamilyLoader/Compiler 对旧 interface export 的真实回读；**旧数据只验证接口，不计新合格族**。没有 GPU、模型或新物理回放。

## family API

`audit_family(export_root, evidence)` 读取实际 V4 export，返回质量等级、缺失/拒绝原因、control_type、M2 完整性以及复算数量。必须提供：

- `sealed_files`：所有被读取证据的**绝对路径→已由调用方冻结的 SHA256**。包括 export 的 SHA256SUMS、其所有列举文件、全部 CONTENT_INDEX 引用、下面各证据文件和 27 个 replay 文件。模块不自制哈希当作既有封存证据。
- `candidate`、`certificate`、`readback`、`result`、`store_close`、`control_evidence`、`budget_final`、`resource_final`：各自明确文件路径，不能凭名称或 PASS 文本放行。
- `replay_paths`：恰好 27 个实际完整回放日志。按 seed 与精确动作序列匹配 3×3×3，不靠文件名或列表顺序。
- `house_split='FIT'`、`source_scope='actual_physical_fit'`，并与封存 control_evidence 内字段一致。export 必须为 `candidate_fit_pool`，`interface_only`/合成来源不能通过训练候选质量门槛。

`result` 可以是**独立 CPU 后处理的物理回放复核结果**：`physical_certified=true` 或 `physical_replay_verified=true`、`error=null`。原 runtime 即使因导出 API 失败，也不等于已录物理轨迹不存在或全无效；原失败应单独保留，不把原 result 改成成功。后处理必须真的重建证书并满足其他证据，不能只改布尔字段。

资源终态必须有 `own_renderer_absent=true` 的实际 graphics-aware 最终检查；旧 compute-apps cleanup_complete 不能代替。预算状态通过原类的完整恢复状态约束校验，不调用持久化或执行动作。

## 独立复核内容

- 原 FamilyLoader 校验文件哈希并逐格重算 18 个标签、query、M2 和 CE owner；readback 必须与实际重算结果相等。
- 三条历史 F/L/R 逐类计数完全一致，历史序列不同；完整 3 history × 2 task × 3 query，无 unknown。
- 同 task/query 有 H_A/H_B 标签反转；H_A/H_A_I 的六个标签全部相同；同 history/query 有 task 条件差异。
- 同 query 跨历史逐值相同；当前两帧 RGB/semantic/pose 及同任务的当前 policy 短窗/八动作完全相同。
- 所有 prefix/full-policy 记录与原 Compiler.policy_at 逐值复算；调用原 FamilyLoader.policy_payload 检查真实白名单。离线 query 和 M2 不进入策略输入，不允许未来 step。
- 27 个回放都有正确 trace_hash、合法完整性和三个规定 seed；其动作/观测与九条 canonical trace 相同。实际重算 54 个任务评价并与 certificate rows 对齐。
- 内容逐文件验证 hash/NPY；semantic uint32 像素计数再与实际 observation.pixels 核对，避免只核标签 JSON。
- M2 只有完整逐步程序状态确实存在且原 loader 重算相等才标 `COMPLETE_RECOMPUTED_STRONG_PROGRAM_STATE`；这不是已经训练过 M2 模型的结论。

## 控制类型不混称

保留原始 `control_type`，只接受：

1. `completed_subgoal_revisit_placement_not_event_free_detour`：必须有真实前进，两个 A 历史均出现至少两段彼此分开的 A 事件；Y 行不变。不能称“完全无事件绕行”。
2. `event_free`：需要封存 `intervention_ranges={H_A:[start,end],H_A_I:[start,end]}`；区段不得触发任何登记 role 的事件，不泛指整个环境所有未登记事件。
3. `spatial_detour`：同样需要范围证据，无 A/B/terminal 任务事件且区段有 F；可包含已登记 irrelevant，不能混称 event_free 或第一类。

当前质量模块只声明 `controlled_task_instructions_only_no_natural_language_transfer_claim`。既有模板语句不会因通过物理检查被描述成自然语言多样性数据。

## batch API 与草案门槛

`evaluate_batch_draft(batches, families)` 为纯规则草案检查：两个事前冻结批次，各至少 3 个尝试、各至少 2 个已验证合格报告；整体至少 6 个不同物理 hub、至少 3 个 FIT 房屋。所有尝试必须保留，冻结必须早于开始，未知结果不计成功。

同屋距离不超过 **0.5 m 的 hub 取连通分组**，忽略改角色、改朝向、重复 seed，不能充数为 6。0.5 m 是本交付显式**草案参数**，须主 agent 审核冻结后再用于正式节点。

该函数的 `draft_threshold_met` 仅表示给定输入满足草案。它**始终返回 engineering_pass=false、main_agent_freeze_required=true**；目前没有生产级 wrapper 独立验证批次输入封存、每个尝试和 family 报告绑定，因此不可把外部提供的字典当作已审核实测批次 PASS。

即使后续工程门槛通过，也不是统计泛化、算法收益、投稿充分性或模型性能 PASS。控制类型单独计数，不能将不同类型合并成同一机制效果证据。

## 验证范围和剩余工作

14 项测试覆盖矩阵轴、计数捷径、unknown、控制不变性、hub 去重、冻结时间、尝试缺失、控制分层、无封存不放行、旧 interface 原 loader 及 future-query 字段拒绝。**完整 audit_family 成功分支尚无新真实合格族可供验收**，不能提前报告其实际 family acceptance 已 PASS。

读取受项目路径、单文件 256 MiB、已显式计量读入 12 GiB、export 最多 64 文件及历史/续接固定长度限制。`read_bytes_counted` 只计显式封存读取，原 Loader 内部重复读取未包含在这个计数中；这是有限离线审核，不是吞吐基准。

另外已向主 agent 报告：旧 worker 的 `split='FIT'` 与原 exporter 契约不兼容；原支持 `candidate_fit_pool`。该问题应另版修复，不改活跃原代码。当前 revisit 另有 QUERY_LENGTH 失败，见独立 [QUERY_LENGTH_DIAGNOSIS.md](QUERY_LENGTH_DIAGNOSIS.md)。
