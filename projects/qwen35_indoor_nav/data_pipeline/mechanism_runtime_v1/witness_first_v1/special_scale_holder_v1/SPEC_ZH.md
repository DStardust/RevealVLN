# 特殊数据四卡占位链式借还 V1

独立 sibling，绝不修改 GPU1/2 已封存运输。GPU3/4/5/7 UUID、初始 PID/start/cmdline/cwd/pane/cache-env 均来自主 agent 冻结 collection。仅运行事前固定队列，数据仍 FIT、三候选/批、完整27认证及原18格/54评估。无失败/partial重试，无训练，无GPU0/6操作。

每批借还后 PID 可变，但只能取该卡事前固定立即前批的已批准 INPUT_LOCK + 正常 SUPERVISOR/LEASE/RESTORATION/LAUNCH；不得跳过失败、不凭进程名扫描寻找占位。新批在借卡前封存 CHAIN_IDENTITY_BINDING，继承原 V2 lease 逐身份双检、只向精确 holder SIGTERM、只清理自有 worker、finally 恢复及保护死 pane 的规则。原 lease diagnostic 字符串可能保留 GPU5 前缀，不代表实际设备。PIDfd 未新增，继承的 /proc 双检查仍有其明示残余竞态。

初始和批间精确占位身份不满足则停卡，不停止外部进程。缓存环境用白名单恢复，未登记白名单键明确 unset；不得注入其他 shell/环境键。tmux 仅用原 -w 选项，不使用旧版不支持的 -p。统一互斥文件为本目录 locks/gpu_N.lock；同卡 scout 必须共用并等待明确前驱。

每批仍 supervisor3900秒/factory3600秒/60000动作/7GiB，原外部显存限制及 idle/restore guard 不变。仅继承已审活跃 MEMORY_ACCOUNTING 的 gross=max(device,sum) 显式修订，原XML保存、派生判别独立、新质量等级如实标注。每卡队列12小时；单批总预留5400秒（transport4200+audit1200），不足预留不启下一批。4200为原已批准有限阶段运输预留，I/O/系统停滞不应被误称硬实时保证；必须保留finally恢复权。

主 agent 在 CPU 封存与每批完整输入核验后另写 MAIN_AGENT_SCALE_APPROVAL.json 才能启动。自动队列失败停该 lane，不自动重新尝试。跨批标号无结果不等于未暴露，所有失败和partial保留。

CPU 单测覆盖原 lease 成功/错误/信号/退出窗口/未知 pane/PID复用/显存排水/恢复不足等故障；链式身份不得冒充统计泛化或模型收益。实际生产关闭后需通过新版 strong audit 和额外 holder terminal/source 证据闭合。
