# Scout 下一批六屋 CPU 准备

仅扩大固定 FIT 房屋覆盖，不根据正在运行 scout 的成功结果重新排序房屋、放宽事件/闭合阈值或修改算法。沿已冻结机制 manifest 首次出现顺序，排除 17DRP5sb8fy、1LXtFkjw3qL、1pXnuDYAj8r，取后续六屋：29hnd4uzFmX、2n8kARJN3HM、5LpN3gDmAk7、5q7pvUzZiYa、759xd9YjKW5、7y3sRwLe3Va。按此顺序连续三屋一片，GPU1/GPU2 仅意向配置，不代表当前资源可用或运行授权。

来源保持原 feedback prepare 规则：官方已登记 R2R-CE v1-3 train.gz 固定 SHA，读取各屋全部示范 start_position/reference_path 去重排序；各屋 glb/house/navmesh/semantic.ply 四项完整资产分别哈希。原四角色 metadata 仅用于 backend permission 和实例身份核验，不充当新 A/B/T/I 选择，更不作为可见性证据。完整有限词表、真实连续两帧事件、公共尾中性、闭合筛选保持 scout_v1 原算法。

每片完全独立输出根 `shard_0/run_v1` 和 `shard_1/run_v1`，没有共享可写 bank、content、journal 或预算。各片保持每屋2hub、每hub12组、每组2targets；4200秒/60000动作总工厂、每屋1200秒/20000动作、4500秒监督；content6GiB/总8GiB、RAM8GiB。两片合计上限120000动作、16GiB磁盘，不能共享或借用另一片未消耗预算。候选每hub最多32，完整物理族为0，训练量为0。

`prepare.py` 只生成两份 PREPARED_CONFIG、SOURCE_INVENTORY、INPUT_LOCK、PREPARATION_RESULT，均 executable/runtime_allowed=false。同输入重运行仅检查字节完全一致，不覆盖冻结文件；有变更须新版本。`test_prepare.py` 是六项纯CPU选择/来源范围测试。算法代码及依赖按旧 scout INPUT_LOCK 指向的实际源码锁定，不读取/修改活跃输出树。

## 下一运行适配必须由主 agent 审核

本交付没有 GPU launcher 或新 worker，runtime_adapter_ready=false，不可直接执行旧 scout worker。旧 worker 的 HabitatBackend GPU 参数硬编码1；GPU2版本需唯一且计数断言的显式源码运输变换，将该参数替换为对应冻结 gpu_device，不能暗改旧源。旧 common.HERE、store.FEEDBACK_ROOT、worker.HERE、监督 OUT/ENV/UUID 也必须全部在新独立模块绑定到各 shard，避免旧层级输出根错误。新配置的资产列表与角色身份仍须 runtime 前复验，approval 必须绑定新增适配代码后的新锁。

GPU XML须包含图形进程。每片沿旧准入：初始util0、外部单PID<=768MiB且总<=2048MiB，自身保守显存上界<4096MiB；只清自身worker，不碰GPU5占位、外部实际任务或另一片。GPU1当前 scout、GPU2当前 assembly 未结束前不能因为本配置存在而抢占。候选汇总只读各片封存 bank，保留 house/shard/sourcehash；平衡动作计数与完整18-cell/27-replay认证仍是独立节点。
