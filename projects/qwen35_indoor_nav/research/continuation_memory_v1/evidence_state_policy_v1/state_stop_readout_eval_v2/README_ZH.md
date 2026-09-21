# STOP 读出 V2：更正诊断后测量真实闭环取舍

初版 state_stop_readout_v1 已完成三个种子共3600次读出更新。随后发现初版局部 gate 将“STOP偏离教师动作”称为“错停”，该指标不能直接等同任务失败：原FIT中1816个、DEV中444个有效教师继续动作发生在精确任务状态已经满足的位置（每模型，未乘种子数）。这些继续动作不一定非法，但提前合法STOP不能因与教师动作不同就判错。

初版全部日志、结果和首次局部拒绝记录保留。只读 SEMANTIC_STOP_AUDIT.json 给出修正：DEV实际状态满足时选择STOP 247→279；未满足时选择STOP 46→53。三种子/3168个有效教师位置，不是独立episode；不能据此直接宣称成功率收益。先前的154→189是教师STOP动作不一致，不是189次任务语义错停。

本版只测量同一批六个冻结权重，更新数0，不做第二轮训练、阈值搜索或重新选择seed。执行原本注册的原DEV16变体×4历史×2任务×6模型=768次完整真实续接，每臂主任务192、task_T192。没有读取或重跑已暴露四个新屋作为本轮分母。环境、动作、STOP观测、原安全评测器、全部500预算不变。输入/前缀/基座身份和完整组仍是正确性门槛。

本次协议更正将离线模仿偏差作为诊断，不作为禁止测量闭环的理由。用户V5授权明确区分实验完成、指标取舍、收益和采用；有风险的静态诊断不能替代闭环因果比较。此变更在第一次本版闭环前登记，不把初版gate改为通过。即使主指标上升，也必须展示task_T、碰撞、过早STOP和成本；原DEV只提供开发证据，不自动采用。

后台独立服务自行完成剩余完整组、审计、GPU占位恢复与GitHub上传。监控仍在 http://127.0.0.1:18770/ 。运行目录 runs/eval_001；查看 STATUS.json、MONITOR.jsonl 与 evaluate/session_*/GROUP_*.json。服务不依赖Codex或终端。

```bash
export V16_STANDALONE_PYTHON=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
"$V16_STANDALONE_PYTHON" -I -S -B "$PWD/standalone.py" start state-stop-eval-20260921-01 -- "$V16_STANDALONE_PYTHON" -I -B "$PWD/pipeline.py" --config "$PWD/PROTOCOL.json" --run-id eval_001
```

不要重启已经运行的服务。基础设施恢复使用新服务名指向原run-id并加 --resume，保留全部失败attempt。所有完整组只评一次，不因低分重试。
