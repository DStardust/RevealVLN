# 下一 scout 来源适配：CPU-only

本目录仅读取 `witness_first_v1/scout_next_v1/shard_0/run_v1`。适配器与父目录封存prepare的差异严格限于来源路径、目录定位、原method导入/哈希位置及输出限定；5项CPU测试逐文本核对这些限定变更。

仍调用父目录封存的 `method.py:BalancedFactory`，不改变模24对齐、排序、计数尝试账本、role/任务定义、source action、原checker或query160。

```text
项目stdlib Python -I -S -B .../next_source_v1/prepare.py --output .../next_source_v1/snapshot_v2
```

输出必须是本目录下不存在的直接子目录 `snapshot*`。仅 `SCOUT_COMPONENT_BANK_COMPLETE` 的house会读取完整BANK；活跃或资源截断house保留deferred，不读正在append的半行。以后house关闭后可增量生成新的snapshot，不覆盖旧目录。

每个snapshot保留全部来源锁、候选catalog、计数拒绝账本及不可执行配置；runtime/GPU预算仍须主agent另行准入。本目录不操作GPU或仿真。

附带batch00只读诊断：截至本轮初查，首候选仍在运行、10条完整trace/1458已记录动作/0碰撞，没有终端result，尚无可归因的spin/history失败。它不需要整圈padding；不可把仍在运行写成成功或失败。后续若终端证据出现，使用新诊断文件记录，不改旧结果。
