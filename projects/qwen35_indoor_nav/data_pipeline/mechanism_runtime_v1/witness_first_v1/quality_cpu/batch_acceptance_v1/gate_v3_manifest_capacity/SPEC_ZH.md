# 来源清单工程容量适配 V3

只读复用封存 acceptance.py 和 gate_v2/gate.py；所有原文件保持不变。

唯一审核源码替换：`len(lock)<=1024` → `len(lock)<=2048`，完整源码先核实固定 SHA256，替换必须精确命中一次，反替换后全源码相同。32 GiB 来源总量上限、8 GiB 单文件上限、逐项 SHA256、事前冻结/因果阶段、27 真实认证轨迹、原18交叉格/54求值/M2、每批与跨屋门槛均不变。

gate_v2 按固定 SHA256 导入独立实例，`old` 指向私有适配 acceptance；`HERE` 仅指向本新目录限制新审计输出。没有修改共享实例、封存文件、物理结果或策略输入。旧cohort仍完整曝光，新cohort须事前冻结并被每个成员INPUT_LOCK锁定。

调用 `audit_all(batch_root, output, recovery_index=None, cohort_manifest=None)`，或 `gate.py --batch-root ... --output <本目录内新目录> --cohort-manifest ...`。只有后续完整关闭的真实运行才能取得常规审核资格。审计会附加 `MANIFEST_CAPACITY_ADAPTER.json`，不改变原REPORT裁决。

CPU差分测试12项：精确反替换、原源码哈希、篡改源拒绝、原范围相同、新1025–2048范围完整SHA循环、2049拒绝、32 GiB边界、哈希篡改拒绝、输出隔离、原cohort字节码及失败门槛保持。使用明确CPU mock到journal边界，不是物理证据或科学PASS。本节点没有GPU操作、没有运行数据审核。
