# 普通导航训练只读监控

固定监听 `127.0.0.1:18766`。18765 已有不相关服务，不能停止或复用。
实现交付不启动服务；由主 agent 核验端口后启动。

项目根内启动命令：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/sft_acceptance/ordinary_execution_v6/monitor/server.py
```

远端访问，在自己的电脑终端保持 SSH 转发：

```bash
ssh -N -L 18766:127.0.0.1:18766 <用户名>@hpda-jushendaohang-005
```

浏览器打开 `http://127.0.0.1:18766/`。若本地端口占用，可仅改 SSH 命令左侧端口，例如 `-L 18767:127.0.0.1:18766`，浏览器用18767。需要跳板时沿用自己的 SSH Host/ProxyJump 配置；不要把密码、私钥或访问令牌填入 URL。

只有 `/`、`/api/status`、`/healthz` 三个固定 GET/HEAD 路由；没有文件浏览、查询路径选择、写入接口或训练控制。HTTP 服务本身不提供认证，只绑定 loopback，使用已有 SSH 认证访问；不要通过反向代理公开。

读取范围为父目录 `PROBE.json`、`STATUS.json` 和一层 `run*/PROGRESS.json`、`RESULT.json`。显示状态、更新时间、训练 loss/steps/decisions、嵌套 cursor/metrics/budget/throughput、STOP 统计、最新 checkpoint 名称。缺字段保持缺失，缺文件/损坏/过大文件分别明确显示，不填假零。凭证、环境字段与不相关顶层数据不向页面输出；页面用 textContent 防止状态文本注入 HTML。

页面每10秒刷新，文件超过120秒未更新标记陈旧；已经正常结束的历史结果也可能陈旧，这不是失败判定。GPU采用最多2秒的只读 nvidia-smi 查询，失败显示 unavailable；利用率不代替训练进度。checkpoint只读取进度字段或浅层文件名，不加载权重。JSON读取上限2MiB、最多64个run、最多256个checkpoint目录项，避免遍历生产content。

纯CPU测试（不启动监听、不导入模型/GPU框架；临时夹具仅在monitor内）：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/sft_acceptance/ordinary_execution_v6/monitor/test_server.py
```

仅打印当前状态而不启动服务：在启动命令末尾添加 `--check`。接口存在不代表训练启动或样本学会，依据真实 PROGRESS/RESULT 判读。
