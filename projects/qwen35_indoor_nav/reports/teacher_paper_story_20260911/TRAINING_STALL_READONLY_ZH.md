# 普通训练停滞：本轮只读发现

2026-09-11 13:06–13:08 CST。没有修改训练源码、协议、输入锁或进程，没有重启或发信号。

## 已确认

- GPU3/4/5 对应 train.py 的三个进程 PID 1009994、1009995、1009996 仍存在，运行 PROTOCOL_3GPU_SEG2.json、formal/run_0001。
- PROGRESS.json 与 PROGRESS.jsonl 最后一项均为 30499 更新，日志事件时间 12:40:56 CST；累计指令条件动作计算 2236987，最近已记录 checkpoint 为 30400。
- formal/supervise_v3.log 也止于同一更新，没有新增报错或更新；因此 TRAINING 字符串、GPU 利用率和进程存活不能证明持续推进。
- 本轮 snapshot_training.py 保存了该最后记录的原始内容摘要和读取时间，详见 TRAINING_READONLY_SNAPSHOT.json。累计量包含恢复与重复计算，不是独立样本覆盖率。

## 明确代码风险；当前卡点仍属推断

train.py 的 due 由各 rank 本地单调时钟判断；当前协议只有 progress_seconds=10，未设置 log_every_updates。due 分支包含 packed(float64,4) 和 confusion(float64,4x4) 两次 all_reduce，分支外还有停止标志的 all_reduce。当一个 rank 的本地间隔为9.99秒、另一个为10.01秒时，两者可能进入不同的 collective 序列。这是可直接由源码构造的风险，不依赖模型或数据质量。

目前表现与多卡 collective 不一致相符，但未进行进程栈采样，不能把这一推断写成已定位的实际阻塞堆栈。仍需排除数据加载、内核或存储等待。epoch 已为1，当前不是“首轮未递增 epoch”那一旧问题。

## 后续修复入口

只读源码绑定：train.py SHA256 为 8c2228df11f8b4b025c4b07ef253dd8bcc2c59306857666d7f2d267883a04a59；PROTOCOL_3GPU_SEG2.json SHA256 为 c3839f783a07bb18f32d041890ca6738aff7671d61120a9a463f5904bd731b98。上述文件均未改动。

另开训练修订版本：所有 rank 使用相同更新步触发日志；停止信号先通过统一 collective 协调，再分支或退出；验证正常迭代、日志边界、checkpoint、单 rank 停止请求及 epoch 结束时 collective 序列一致。不能只修改状态文件或原输入锁。

先核验最新完整 checkpoint，再安排受控恢复并保留当前停滞记录和累计资源账。当前会话仅完成多样性实现与汇报，不据此扩大为停止现有训练的操作。
