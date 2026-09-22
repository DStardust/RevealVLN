# 独立运行与进度

本轮运行目录：`runs/aligned_001/`。原 STOPPLUS 已结束；此处是新的匹配教师监督实验。

当前服务：`q35n-teacher-alignment-20260922-01.service`。
进度服务：`q35n-teacher-alignment-monitor-20260922-01.service`，沿用服务器 `http://127.0.0.1:18770/`。

```bash
systemctl status q35n-teacher-alignment-20260922-01.service
```

在本目录查看日志：

```bash
cat runs/aligned_001/STATUS.json
tail -n 80 standalone_jobs/teacher-alignment-20260922-01/job.log
```

细分阶段日志在 `runs/aligned_001/attempts/*/stdout.log`；每个训练模型的实时步数在 `runs/aligned_001/train/*/PROGRESS.json`。
网页仅对完整六模型组计算当前评测分数。未完成、未运行与 FAIL 分开。

修复记录在 `runs/aligned_001/amendments/`，包括加载器所需 best4k 训练协议哈希的补录，以及训练复核结果合并字段修复；此前协议/源码副本原样保留。两项都未改变教师计划、模型、优化、动作或评测语义。

只有原服务退出后才恢复；使用新服务名、同一 run-id 和 `--resume`。恢复检查配置和源码身份，重用已认证物理轨迹、已封存训练检查点与完整评测组；不覆盖失败尝试，不按得分重跑。

```bash
ROOT="$(git rev-parse --show-toplevel)"
TASK="$ROOT/projects/qwen35_indoor_nav/research/continuation_memory_v1/evidence_state_policy_v1/teacher_alignment_v1"
PY=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
JOB="teacher-alignment-resume-$(date -u +%Y%m%d-%H%M%S)"
V16_STANDALONE_PYTHON="$PY" "$PY" -I -S -B "$TASK/standalone.py" start "$JOB" -- \
  "$PY" -I -B "$TASK/pipeline.py" --config "$TASK/PROTOCOL.json" --run-id aligned_001 --resume
```

完成后流水线自动恢复本用户 GPU 2–7 占位，复核并发布代码、六个新 head 和文本证据。原始许可场景、数组与大缓存保留在服务器。不需要 Codex 保持在线，也不承诺跨断电或管理员停止。
