# V2：仅修复并发缓存目录扫描

V1保留在 `closed_loop_bench/ordinary_stop_calibration_fit_v1/run_001/`：运行58.71秒，模型已加载，Triton临时目录原子重命名期间 `Path.rglob` 报 FileNotFoundError，launcher安全停止自己的进程组。全部lane无正式episode/动作记录；GPU1清空、其他占位不变。属于服务错误，不是科学失败。

V2新目录 `ordinary_stop_calibration_fit_v2` 使用完全相同的64条FIT输入、相同checkpoint、协议参数、选择规则和FIT准入门槛。仅替换输出用量扫描：允许并记录扫描期间ENOENT；权限/I/O错误、缺失根目录仍失败，不遍历符号链接。使用量仍是实时观察，非原子快照，现有5秒监督频率和4GiB上限不变。

这是主agent定位后的单次版本化重试，不是自动重试循环。旧锁/源码/日志不改。新运行仍32,000动作/3,600秒，另列前次58.71秒启动成本；不把失败准备成本忽略或当新数据。counterfactual和fit科学实现只读复用，fit_v2.py仅绑定新采集目录。
