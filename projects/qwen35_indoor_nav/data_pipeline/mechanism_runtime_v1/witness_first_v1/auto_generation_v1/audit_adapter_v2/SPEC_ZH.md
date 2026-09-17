# 审核输出路径适配 V2

主审事前发现 V1 audit.py 的输出在原 acceptance.py 授权目录之外，会 OUTPUT_SCOPE 失败。旧文件、旧队列、12 个批次锁全部保留。此版本仅将唯一 `out=HERE/'audits'/batch.name` 改为 `out=GATE.parent/'auto_generation_v1'/batch.name`，反替换后原源码逐字一致；所有强审计、SHA、闭合及质量阈值不变。

主调度 PLAN 应使用本目录 audit.py --batch 绝对批路径，并锁入新入口与本目录 SHA256SUMS。原 JOBS.json 中旧审核命令只作历史，不用于启动。CPU 测试调用真实旧 save 并回读 TEST_FIXTURE，不是物理数据或强审计 PASS。scientific_pass=false。
