# Q35N 实测训练与生产图表监控

这是独立新版本，不修改冻结 `ordinary_execution_v6/monitor`。固定 `127.0.0.1:18766`；主agent精确停止旧监控并核验端口后才可启动本服务。此交付未启动监听或发任何进程信号。

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/sft_acceptance/monitor_charts_v1/server.py
```

远程访问（在自己的电脑执行并保持SSH连接）：

```bash
ssh -N -L 18766:127.0.0.1:18766 <用户名>@hpda-jushendaohang-005
```

打开 `http://127.0.0.1:18766/`；本地端口占用时只改SSH左侧端口。服务没有HTTP认证，依赖loopback＋SSH认证，禁止公开暴露。三个固定GET/HEAD路由 `/`、`/api/status`、`/healthz`；没有文件浏览、POST控制、CDN或GPU控制。

图表取 `ordinary_execution_v6/run*/PROGRESS.json` 与 `PROGRESS.jsonl`、执行STATUS/PROBE、冻结普通snapshot counts。只使用日志自身的unix/cursor；CE绘制记录的last_chunk值与最近20个有效日志点均值，日志有抽样间隔，不伪称每个chunk全量。cursor回退/时间倒退会断线并重置均值。吞吐是记录的累计吞吐，不伪造瞬时速度。四动作的目标数、预测数与recall从累计4x4混淆矩阵推导，无目标类别的recall=null。

只显示在线训练统计，不是DEV、闭环或导航收益。ETA用记录吞吐线性估算，到当前段决策预算和当前epoch分别显示；wall/token/update限制可能更早触发。阶段预算不足一个epoch会照实显示。checkpoint只显示记录名称，不反序列化权重。没有历史日志时不将当前值伪造成曲线。

全部读取限项目范围；JSON上限2MiB，PROGRESS.jsonl只读最后8MiB、最多1000完整行/点，不完整尾行忽略并标记。损坏/缺失/陈旧（120秒）明确显示，缺数据为null/破折号，禁止填假0。nvidia-smi只读2秒超时。生产队列表浅读AUTO一层PLAN、RESULT、256KiB EVENTS尾部及/proc队列命令，包含新 `manual_failure_recovery_gpu1_20260910_v1`；RUNNING不代替强审合格量。此表不自动恢复任何任务。

CPU测试不启动HTTP监听、GPU或模型；临时夹具仅新目录内：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/sft_acceptance/monitor_charts_v1/test_server.py
```

`server.py --check` 打印一次真实有界读回，不启动服务。用户浏览器每10秒刷新；原始状态在页底折叠区保留供核查。源码不依赖Pillow、torch或第三方前端。
