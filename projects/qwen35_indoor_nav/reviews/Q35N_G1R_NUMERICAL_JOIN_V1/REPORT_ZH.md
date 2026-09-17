# Q35N_G1R_NUMERICAL_JOIN_V1 节点收口

结论：`NUMERICAL_JOIN_DISCOVERY_EXHAUSTED`。

此目录没有验收通过的机制训练标签；候选预检不等于独立族验收。失败与候选完整保留。

本轮runner wall：97.75 秒；worker返回码 0；自身GPU清理 True，占位恢复 True。未停止真实任务。

失败账本：

```json
{
  "FAMILY_ATTEMPTS.jsonl": {
    "completed_records": 16,
    "reasons": {
      "PUBLIC_TAIL_LEGALITY_OR_EVENTS": 8,
      "NO_VALID_CONTINUATION": 6,
      "PUBLIC_TAIL_EVENT_REJECT": 2
    }
  }
}
```

未运行Qwen、SFT或机制训练；scientific_pass=false，不声称泛化、导航收益或投稿贡献成立。
