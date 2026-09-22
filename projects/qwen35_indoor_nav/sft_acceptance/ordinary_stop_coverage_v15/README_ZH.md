# 普通导航真实覆盖修复 V15

本轮是工程验证，尚未取得新的 SR 结果。V13 的旧 INTERNAL_DEV100 SR 为25%；V14为25%，未采用。新目录保留V13为配对对照。

自动运行链：40个训练房屋、320条固定指令，各执行V13自主策略和真实参考路线教师，共640条采集轨迹；使用因果模型特征与实际距离生成STOP监督；沿用V13的BCE与L2训练2049个STOP行参数，其他权重冻结；随后固定候选，依次完成开发100对及完整val_unseen1839对，逐对封存。没有根据开发分数挑选是否运行unseen。

教师仅用于训练采集，路径规划真值不传给模型。实际动作写入历史。教师不能继续时保留已执行片段与失败计数，不伪造STOP。每条轨迹最多500决策，合法成功仍要求主动STOP且测地距离小于3米。训练与两套评测房屋分离；全unseen为公开、已暴露验证，不是盲测。

新缓存来自真实Qwen前向。训练使用独立训练房屋的真实观测，未重放旧5487输入。三条固定旧FIT输入仅在评测模型加载时作数值诊断。训练诊断不是SR。模型一次加载供同一对A/B使用；输入、底模动作前缀不一致则停止并保留证据。

## 独立运行与查看

项目实际Python路径已写入PROTOCOL.json。以仓库根为工作目录：

```bash
PY=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
V=projects/qwen35_indoor_nav/sft_acceptance/ordinary_stop_coverage_v15
V16_STANDALONE_PYTHON="$PY" "$PY" -I -S -B "$V/standalone.py" start ordinary-coverage15-20260922-01 -- "$PY" -I -S -B "$V/pipeline.py" --run-id coverage_001
systemctl status q35n-ordinary-coverage15-20260922-01.service
cat "$V/runs/coverage_001/STATUS.json"
```

已有run恢复需新服务名、相同run-id与`--resume`。完整封存轨迹/配对不覆盖；无法解释的正确性失败禁止跳过。基础设施失败的尝试和资源计量保留。恢复校验源码、配置和权重身份。

原监控地址端口18770显示采集、训练、DEV、unseen及既有100000步记忆任务。任务独立于终端和聊天进程，故障会写STATUS及FAILURE。共享GPU1–7，GPU0原任务保持运行；仅清理本任务拥有的进程，不因他人进程出现就作废。

源码和小型证据可发布GitHub；原始RGB、场景、特征与权重的完整身份/本地路径保留于运行目录，不声称GitHub包含授权数据集。CPU验收见CPU_TEST_RESULT.json。
