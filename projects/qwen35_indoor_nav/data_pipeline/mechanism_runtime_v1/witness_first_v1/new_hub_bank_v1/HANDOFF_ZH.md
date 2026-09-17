# 四新hub的CPU候选银行后处理

固定输入 `../new_hub_scout_v1/run_v1`；不接收任意路径，不运行GPU。当前只准备代码和测试，没有伪造采集结果或配置。

实际自然关闭后可先运行：

```bash
/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B /mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/witness_first_v1/new_hub_bank_v1/prepare.py --check-only
```

通过后去掉 `--check-only` 构建。输出仅在本目录的新 `snapshot_v1/`、`language_ready_v1/`，包含全候选、计数/提案/去重拒绝账、规范词面、NEXT12及来源锁。已有输出拒绝覆盖。

前置：SCOUT_CLOSED、单屋完整银行、supervisor自然rc0/error null/cleanup、LAUNCH正常返回、内容store闭合；INPUT_LOCK等于parent SOURCE_LOCK，全部SHA、官方FIT分组、approval、事前mtime与prepared→runtime配置逐字段一致。四个实际hub须与事前选点集合相同、完整几何重查且距原hub/彼此至少1米；不足四个或censor不能冒充本次四hub关闭。

随后精确复用封存多程序bank算法：完整原trace/committed journal、语义身份、闭合、所有计数组合、history/160-query门槛、每hub24语义程序去重及有限词面。新hub尚无整圈中性证据时仍为UNTESTED，不能因旧屋曾通过就推断新位置通过。全部候选仍待后续27完整真实认证，不计合格族或模型收益；同屋同hub变体相关。
