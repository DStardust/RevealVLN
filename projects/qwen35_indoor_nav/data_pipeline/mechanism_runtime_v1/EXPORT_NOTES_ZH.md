# V4 真实机制导出与 CPU loader

本支线只实现 `exporter.py`、`loader.py`、`test_export.py`。没有仿真、GPU、模型、下载或 SFT 操作。输入是真实物理日志；重导出本身不创造新物理族，也不授予训练或科学 PASS。

## 接入 API

```python
manifest = export_family(
    new_output_dir, compiler, candidate,
    {(history_id, continuation_id): complete_trace},
    content_dir,
    family_id="explicit_unique_export_id",
    split="interface_only",  # 或 candidate_fit_pool，仍不是训练准入
    provenance={"execution": "caller_recorded_runtime_or_sealed_log_readback"},
    repeated_traces=None,     # 可选另外两个同样结构的 seed grid
)
data = FamilyLoader(new_output_dir, compiler)
audit = data.validate_supervision_contract()
```

`candidate` 必须含 `histories`、`continuations` 与非空 `context.house_id`；每格 trace 必须包含完整 RGB/semantic hash、真实语义像素计数、pose、动作及完备证据。canonicals 为 3 历史 × 3 续接，compiler 含 2 个任务。cutoff 直接由相同长度的 histories 推导，动作必须精确等于注册历史加续接。重复 seed 不计作独立样本。

输出仅允许在当前 runtime 目录的新路径；来源内容必须在本独立路线内且只读。NPY 原像素 hash 与文件 hash 分别验证，semantic uint32 像素直方图必须逐观察与日志计数精确一致；数值重建的 raw/canonical 边界也核验。共享源不会被改写或全量复制。跨历史同 query、跨续接同历史前缀，以及共同最后两帧 RGB/semantic/pose 必须一致。

## 数据分层

- `prefixes/`：每 task × history 完整因果决策序列。每一步仅最近 2 帧、最近 8 个真实历史动作；从第 0 步依次运行才能构成长期记忆。
- `full_policy/`：每 task × history × continuation 的完整动作流前输入，支持成功续接 CE，也包含失败流供明确的诊断，不能据此自动给失败流 CE。
- `SUPERVISION_ONLY.jsonl`：有序真实 query、Y、BCE mask、强 M2 全流逐时状态、成功续接 action targets/steps/owner mask。M2 全流是监督，训练时仍应按对应 decision step 取值，不允许把未来状态送进 prefix memory。
- `ACTION_OWNERS.json`：每个成功续接决策唯一 owner；跨 query 重复只算一次。不能把 owner ID、文件名、hash 等当学生特征。
- `traces/`、`MANIFEST.json`、`CONTENT_INDEX.json`：完整证据、house/family 分组、来源与只读像素引用；仅数据基础设施可访问。

`policy_payload(record)` 严格白名单，只返回 task_type、instruction、原始 RGB bytes、已执行动作字符串和 memory_reset。sample ID、hash、任务修订、时间编号、query、语义标签、pose、house ID 不返回给模型。RGB 为 C-order uint8 `[224,224,3]` 原始 bytes，转换为 tensor 由后续模型接口负责，不能把 bytes 的编码文本拼进 prompt。

`supervision(cell_id)` 是独立 teacher/reader 接口，返回保序 typed query 编码而不是未实现的 neural encoder。配置必须与 manifest 的 roles/tasks/eligible/task_revision 完全一致，避免错词表静默读取。unknown 映射到 `y=None, bce_mask=0`，绝不映射为负标签；未完备 trace 导出在写入前拒绝并留给调用者失败账本处理。

## 本支线实测

命令：项目标准库 Python `-I -S .../mechanism_runtime_v1/test_export.py`。

最终 15 项测试通过，约 11.77 秒。使用原封存 27 条日志（9 canonical + 两个重复 seed grid）在新目录临时重导出和实际像素核验，得到 18 cells（12 pass、6 fail）、6 个完整 prefix 共 1,242 决策、1,410 个成功续接 CE owner；逐格 Y/query/M2/CE owner 重新计算一致。完整动作流、像素篡改、路径/覆盖保护、未知标签、额外特权字段、query 重排、compiler 错配、缺失 house 分组均有测试。原封存 SHA256 清单的 987 项逐项通过；临时测试输出已清理。

这仍是旧单族的 **interface_only CPU reexport**，不是新 Habitat replay、训练收益或新增数据族。真实后端 provenance、物理 certification、候选发现与资源许可由主 agent 独立登记。两个 Python 模块不加载模型，不证明真实 query 神经编码或 207 步梯度路径。
