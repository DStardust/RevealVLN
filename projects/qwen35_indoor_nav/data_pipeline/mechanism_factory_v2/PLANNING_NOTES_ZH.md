# Planning 模块 CPU 交付

实现 `planning.py`，31 项纯 CPU / fake-clock 单元测试通过（首次运行，无失败修改）。本模块不导入 Habitat、模型或 GPU，不生成真实族，也不读取 SFT。

接口：

- `select_candidates(records, stage="P0")` 验证冻结的 20 行路线秩/房屋顺序、FIT_PILOT 身份和各屋 route hash 严格升序；返回首批 5 行。`ALL` 只供清单准备，不授权后 15 个运行。源表哈希仍由主端核验。
- `enumerate_configurations(start_position, reference_path)` 取原始坐标精确不同的前 8 个位置，按 tail → yaw → u 枚举最多 64 格，不补齐、不替换。坐标未进行会改变候选身份的隐式取整。
- `cache_key(house_id, asset_config, roles, extra=None)` 对房屋、完整资产/传感器配置及角色绑定隔离缓存；角色别名在缓存内不可乱用。
- `BudgetLedger(limits=None, state=None, clock=monotonic, persist=None)`；`start_bundle(id, phase)`、`check_time()`、`reserve_action(count=1)`、`finish_phase()`、`snapshot()`。执行调用前计入保守动作 reservation；后端失败或实际次数未知不退回。总墙钟含同机重启/停顿时间；单调时钟回退拒绝恢复。资源耗尽记 `resource_censored`，不是物理不可行标签。
- `FreezeLedger.start/freeze/certify/reject_discovery/resume` 保留首个完整预检候选，认证失败不可替换；仅未完成节点允许带 crash 原因的 linked attempt，预算由同一持久化账本恢复。
- `exact_duplicate_key(record)` 依据物理轨迹和资产/共同状态，忽略外层 family ID、seed、措辞和 history/continuation 别名。调用方必须提供不含这些别名的规范物理轨迹 payload；不能将未经投影的杂项元数据整个放进去。
- `geometry_clusters(records)` 按 Habitat 的 Y-up 坐标计算同屋保守连通分量：水平 <1 m 且垂直 <0.5 m，或同实例集合且 0.25 m 三维占据格 Jaccard ≥0.8。传递闭包会合并链式邻近族；不同房屋不合并；空占据不算重叠证据。

## 生产接入边界

`persist=None` 仅用于 CPU 夹具。生产适配器必须提供原子提交并 fsync 的持久化回调和进程外互斥锁，且恢复同一快照；本纯模块不自行写文件。回调失败会 poison 对象，禁止继续动作。单调时钟重启回退需要人工主审恢复，不能归零继续。

所有实际动作（包括失败 discovery/search 分支和证书复放）必须经过主端统一的 reserve-before-call 包装；预算模块无法拦截绕过包装的后端调用。长时间不可中断调用结束后仍应 check_time；它不能强制中断后端自身的阻塞。必须另行接入实际动作完成账本；保守 reservation 不得直接冒称实测动作数。

模块目前只检验 CPU 控制流。Habitat 生产适配、真实执行预算覆盖、磁盘/RAM/GPU 上限和新的物理族产率没有运行验证；不得从 31 项测试推导 runtime/scientific PASS。候选序列检查不替代冻结源清单哈希，几何去重依赖调用方提供真实轨迹和真实实例身份。
