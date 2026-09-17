# 恢复验收交付：等待主 agent 数据接收审核

3 个原运行中已完成的物理族均通过独立的 **family-required-content** 恢复验收；不是生成了 3 个新族，也不是原失败批次改判通过。

| 原来源 | 实际恢复报告 | required blobs | 完整 store 冷审 |
|---|---|---:|---|
| batch00 首族 | [报告](batch00_family0_required_v2/REPORT.json) | 834 | 通过只读内容盘点，非原 store close |
| batch01r1 首族 | [报告](batch01r1_family0_required_v2/REPORT.json) | 855 | 拒绝：保留一个无关 partial |
| batch01r1 第二族 | [报告](batch01r1_family1_required_v2/REPORT.json) | 420 | 拒绝：保留同一个无关 partial |

每族独立重算并通过 18 格、27 个实际认证重放、54 次评估、M2 程序状态、严格动作计数、历史反转/任务差异/A 控制及因果隔离。3 族对应 3 个不同 hub、2 个 FIT 屋；与旧 V3 合并时 batch00 首族同 hub，不能重复计算独立空间支持。

batch01r1 的 200832 字节 partial 完整名称、SHA256、目标像素 SHA256、字节数均保留；它及目标 hash 在两族完整导出、27 认证轨迹、其他发现轨迹及 metadata 中均无引用。其整 store 仍失败，只有两个已完成族的依赖闭包通过新等级：`RECOVERED_FAMILY_REQUIRED_CONTENT_VERIFIED_RUNTIME_CENSORED`。

原 batch00 的 HEAD 已提交前缀通过，943 字节未提交尾部完全排除于预算与标签；batch01r1 HEAD 与日志整文件一致。仅使用每族自身已提交的认证状态与完成预算，不伪造全局终态。

原资源错误保留：`EXTERNAL_RESOURCE_LOAD` / `MEMORY_ACCOUNTING`。全部已保存采样通过原限制、自己的 renderer 已退出；触发失败的样本未保存，不能声称连续资源合规或精确逐族跨时钟对齐。

26 项 CPU 测试通过；原运行、HEAD、store、validator 未修改，无 GPU 操作。早期 V1 通过/拒绝和完整冷审失败证据均保留，正式接收请使用上表 V2 报告及其嵌套 V1 记录。

`training_admission=false`，待主 agent 将该独立恢复来源等级纳入 FIT 训练接收政策。原 `quality_pass`、原批次通过和模型收益均不因此变为 true。目标仍是训练量级普通/机制数据生产，本次只恢复已具完整证据的数据，不把此小样本规模当作量产目标。

调用入口：`family_scope_v2.audit_family_scope(run_root, candidate_id, new_output_dir)`；输出必须为本目录的新子目录。接收条件与资源边界见 [SPEC_ZH.md](SPEC_ZH.md)。
