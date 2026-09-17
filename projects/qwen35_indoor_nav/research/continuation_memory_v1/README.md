# 交叉续接执行记忆：规格与CPU契约

**RESEARCH_HYPOTHESIS_UNTESTED**。本目录仅包含规格、真实旧资产的接口检查和纯CPU
防火墙/精确状态代码；没有新的学习模型、训练器、导航评测器或GPU结果。

实际可调用：`schema.py` 的 `PolicyObservation`、`PrefixTrace`、`LabelCertificate`、
`FamilySplit`、`from_existing_v4`、`continuation_query`、`family_aux_weights`；
`check_tasks.py:exact_state_targets`。它们复用冻结的 `FamilyLoader` 和 `Compiler`。
具体已调用证据在 `CPU_TEST_RESULT.json` 与回交目录 `DATA_ASSET_AUDIT.json`。

阅读顺序：`DATA_ASSET_AUDIT.md` → `DATA_SCHEMA.json` → `INPUT_FIREWALL.md` →
`MODEL_INTERFACE_SPEC.md` → `GRADIENT_PATH_SPEC.md` → `BASELINE_MATRIX.json` →
`ACCEPTANCE_TEST_SPEC.md`。主张裁决在 `../../reviews/Q35N_CODEX_RETURN_20260917/`。

本轮已执行命令（vla根目录）：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B \
  projects/qwen35_indoor_nav/research/continuation_memory_v1/test_contracts.py
```

测试输出用exclusive创建并已封存，勿重跑覆盖；查看现有结果即可。
新 `compile_families.py`、`model.py`、`train.py`、`evaluate.py` 尚未实现，不能从文档或
旧8槽原型推断运行模型已存在。未来有资格的基座SHA、资源预算和接口检查应先冻结；
`FUTURE_PROTOCOL_DRAFT.json` 明确不可启动。研究预算空缺不阻碍本次CPU材料回交。
