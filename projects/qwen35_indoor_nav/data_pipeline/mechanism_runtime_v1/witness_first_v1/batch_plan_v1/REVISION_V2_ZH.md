# 主 agent 接入修订 V2

V1 代码、SHA256SUMS 与 snapshot_v1 全部保留，不覆盖或重封。主 agent 只准备用新版本。

新增 `balanced_v2.py` / `prepare_v2.py`：

1. control_type 统一为已经验收的 `completed_subgoal_revisit_placement_not_event_free_detour`，仍明确 H_A 包含 i，控制的是已完成A后的闭环放置，不声称 event-free。
2. 按主agent更新要求，i 必须含至少2个F动作；不再接受纯旋转i。这是更严格的候选筛选，不放宽物理/矩阵门槛。
3. 每row新增 `component_provenance`：源proposal、hub、house、选取规则、各原组件trace_ref/digest/文件SHA及scout配置SHA，可由runtime exporter原样登记。
4. main worker导入 `balanced_v2.py` 的 `BalancedFactory`；接口保持 components+balance，原checker/query160/三种子仍继承不变。
5. `snapshot_v2` 重新对当前已关闭house建快照。正在append的house仍不读，单hub1000上限的截断数保留。旧snapshot不是新生产输入。

V1 的13项CPU测试保持通过；新增V2的5项CPU测试验证控制类型、移动i、计数匹配、枚举上限和继承原checker。实际物理/新数据族结果仍不由CPU判定。
