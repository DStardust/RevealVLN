# Q35N_G1R_COVERAGE_V1 节点收口

结论：`EARLY_STOPPED_FOR_VERSIONED_CONSTRUCTION_CORRECTION`。

此目录没有验收通过的机制训练标签；候选预检不等于独立族验收。失败与候选完整保留。

本轮runner wall：762.40 秒；worker返回码 -15；自身GPU清理 True，占位恢复 True。未停止真实任务。

失败账本：

```json
{
  "DISCOVERY_ATTEMPTS.jsonl": {
    "completed_records": 58,
    "reasons": {
      "NO_VALID_K_LOOP": 54,
      "NO_VALID_L_LOOP": 2,
      "NO_VALID_D_LOOP": 2
    }
  }
}
```

coverage按主agent构造复核提前停止，不是256候选耗尽。精确总动作数未知，仅保存运行快照下界；在途候选未完成不算正式拒绝。

未运行Qwen、SFT或机制训练；scientific_pass=false，不声称泛化、导航收益或投稿贡献成立。
