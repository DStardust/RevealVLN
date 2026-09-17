# 遥测审核 V2：仅修正报告输出目录

V1 的20项协议反例测试通过，但最后保存阶段调用旧 acceptance.save，其允许目录仅 quality_cpu/batch_acceptance；V1 的新 BE 输出目录因此会被拒绝。实际06r2仍运行时已发现，未伪造完整审核成功，也未改任何生产输入。

V2 校验 V1 源 SHA 后，仅将最后 old.save 替换为本模块 `_write_report`：限定当前V2子树、拒绝符号链接、固定 REPORT.json 名称、独占创建不覆盖。反向替换全源码相同。所有逐XML/原guard/重采窗口/源绑定/完整终态判据，以及 ACK 证据限制，均保持V1不变。V1封存原样保留。

7个V2测试通过，其中最后一个在V2函数上复跑原20项协议反例；没有实际GPU或全run审核。实际入口改用本目录 `audit.py --run <BE>/batch_06r2/run_v1 --output <此目录内新输出目录>`。
