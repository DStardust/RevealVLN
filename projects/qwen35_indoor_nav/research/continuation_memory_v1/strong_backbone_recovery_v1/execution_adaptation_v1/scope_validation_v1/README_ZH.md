# 首动作纠偏的同权重验证

最新调度修订：用户要求“不要等旧评测，直接开始”。旧评测已按要求中断，保留 289 个正式封存组及另外 33 个尚无整批状态封存的记录组；没有把旧实验标为完整或方法失败。新服务 `q35n-strong-scope-validation-now-20260926-01.service` 通过 `dispatch_now_v1.py` 直接运行原 80×13 协议，权重、清单和评测代码不变。原等待服务已经停止；调度授权与源码绑定见 `runs/scope_001/START_NOW_AUTHORIZATION.json`、`START_NOW_SOURCE_LOCK.json`。恢复使用该新入口加 `--resume`，不要再调用下方历史等待入口。监控地址不变。

本轮是工程定位实验：检验 ALL（整段动作纠偏）改为 FIRST（仅首动作纠偏）能否减少对强基线的破坏。当前没有这项调整的闭环结果，不能称新方法已经有效。

- 六个已完成 3000 步的 CURRENT/DELTA 权重只读复用，各自同时提供 ALL/FIRST，两者 SHA 一致。0 新训练，0 底模更新。
- 80 条固定、已经暴露的 val_unseen 路线，按房屋轮流取样，屋内按 `SHA256(scope_v1:id)` 排序。所有 13 个版本在同一模型进程完成一条路线后封存，三个种子不挑最好。
- 原 369 条评测已按后续指令中断并释放本项目租约；本轮直接开始，最多使用 8 卡完成 1040 次执行。原等待条件由单独授权的调度入口替代；本轮评测仍最多 24 GPU 会话小时、12 小时运行墙钟。不抢卡、不删旧失败、不按分数重试。
- FIRST 每个生成块只有 offset=0 允许残差，后续词元完全保留底模分数。实际动作仍写回历史，每个实际观测仍更新记忆；STOP、EOS、空输出行为和 500 决策定义沿用原路径。初始动作也可能被改错，因此本轮必须看净 SR，而非只看干预减少。
- 主要配对：每个种子/写入方式的 FIRST−ALL；同时报告同范围的 DELTA−CURRENT 和各版本−NATIVE。完整分母、胜负 ID、各屋、SPL、动作成本一并保留。80 条仅作开发定位，不能充当完整 1839 条或盲测泛化证据。

运行目录：`runs/scope_001/`，协议及清单在启动前固定。旧 CONCAT 正向候选继续保留，不自动替换。

```bash
ROOT="$(git rev-parse --show-toplevel)"
BASE="$ROOT/projects/qwen35_indoor_nav/research/continuation_memory_v1/strong_backbone_recovery_v1"
cat "$BASE/execution_adaptation_v1/scope_validation_v1/runs/scope_001/STATUS.json"
systemctl status q35n-strong-scope-validation-now-20260926-01.service
tail -n 80 "$BASE/standalone_jobs/strong-scope-validation-now-20260926-01/job.log"
```

监控网站地址不变：`http://127.0.0.1:18770/`，新增 `/api/scope_validation`；原评测与补采页面保留。队列和流水线都独立于 Codex 在线状态。

只有当前服务已经退出、原因排除后，才用新服务名调用本目录 `dispatch_now_v1.py --run .../runs/scope_001 --resume`；恢复保留完整封存组及全部失败尝试。完整低分组不得重试。

新增五项 CPU 测试覆盖 ALL 与原 processor 精确等价、FIRST 仅屏蔽后续残差、新观测重启纠偏、STOP/EOS/记忆写入不变，以及不完整分母不得给出完整 SR。第一次测试的假输入未先经过 assistant header 边界，三项报错已留档；修正测试输入以匹配真实生成调用后五项通过，未修改真实边界规则。另实测官方模块导入及同权重加载一致。
