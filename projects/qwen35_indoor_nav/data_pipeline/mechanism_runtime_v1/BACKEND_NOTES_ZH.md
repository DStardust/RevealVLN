# Habitat 后端接入说明

本文件及 `habitat_backend.py`、`test_backend.py` 仅实现并以 CPU mock 验证接口；子 agent 未导入真实 Habitat / NumPy、未加载场景、未使用 GPU。物理运行结论由主 agent 单独记录，不能从本测试推断。

API 为 `HabitatBackend(scene_glb, gpu_device, roles, content_store, permit)`，另提供同义 `Backend`。permit 需含 `runtime_allowed=true`、与已解析真实场景绝对路径完全相同的 `scene_glb`、匹配整数 `gpu_device`。它是应用层准入检查，不能替代主 agent 的代码/场景资产哈希与 GPU lease 审核。未经准入即拒绝，先于所有可选依赖导入。

roles 按任意 slot 显式定义，例如 `{"anchor_A":{"mpcat40":"tv_monitor","room":"living room","raw_match":{"mode":"exact","value":"tv"}}}`。餐椅可设置 raw_match 的 mode=token、value=chair。exact/token 匹配只标准化大小写、# 和空白；类别、房间严格一致。不存在按 A/D/L 字母或 slot 名猜类别的逻辑。实例映射从 MP3D 原 semantic object ID 末段取得；self.objects/self.eligible 含全实例/合格实例，self.compiler_roles 返回 V4 Compiler 需要的类别房间二元组。空角色集合由后续 factory 拒绝，不制造合格实例。

`reset(position,yaw,seed)` 的 yaw 为 0–23 离散 bin；Y-up，scalar-first quaternion。复用已验收 engine 的 initialize_agent→reset 顺序；配置完全相同：224×224 RGB 与 semantic、HFOV=90°、传感器高1.25m、agent 高1.5m/半径0.1m、0.25m前进/15°转向、禁止滑动/物理。`observe()` 缓存实际 reset/step/重建返回的图像，内容存储调用 `put_array(arr,kind)`，支持主端 dict 的 pixel_sha256 并自主验算 raw bytes hash。观测只返回像素hash、计数、全pose与完整性；不将存储路径透传策略。uint8 RGB 和 uint32 semantic 布局严格验证。

`step(F/L/R)` 恰好执行一次真实 primitive，要求 collided 字段；STOP由TraceRunner负责，不调用后端。`reconstruct(pose)` 沿旧引擎 set_state(reset_sensors=True,infer_sensor_states=True) 显式重建，不暗中更新传感器状态；所有 ≤1e-5 前后范围、精确sensor pose及边界事件不变性仍由 TraceRunner 校验。后端自身不宣称这些条件成立。

`routes(position,yaw,role)` 固定按实例ID、半径(.75,1.25)、8角度顺序，从物体中心周边构造viewpoint，调用纯 navmesh snap_point/find_path，然后将路径点离散为理想动作并末端朝向物体+LR。近点半径0.1875m，最多504动作，同动作序列去重。**不调用 reset/step/GreedyFollower、渲染或修改agent**。这是新的明确几何提案算法，不是旧 follower 的精确复现；尚未实测产率。垂向移动、离散corner切入、碰撞、可见性和逆路线回返全由真实TraceRunner判定，几何返回不证明任何证书。若低产须归因具体工程提案，不据此否定记忆科学机制。route_diagnostics 保留所有几何拒绝/候选原因，主端需落盘。

CPU 测试涵盖准入先于可选import、显式语义映射、15°符号约定、离散路径上限、禁止隐式动作、模拟纯几何路线、动作/STOP与碰撞信号、内容hash及unknown、重复close。真实 reset/传感器重建像素一致性、实际物理seed重放不属于CPU测试证据。
