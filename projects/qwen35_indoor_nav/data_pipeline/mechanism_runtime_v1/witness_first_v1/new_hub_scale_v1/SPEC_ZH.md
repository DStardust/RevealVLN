# 跨屋新 hub 来源扩展 V1（CPU 准备）

保持43个已准入FIT屋及官方R2R-CE train来源，优先此前没有closed scout的房屋，再按原manifest顺序。每屋从官方start/reference_path完整3D位置选择4个新位置，距所有旧observed/declared hub及本批其他新位置≥1m。先稳定字典序，再确定性最远点空间覆盖；不是随机yaw、raw别名或padding凑hub。

新屋没有原navmesh实测，因此候选明确为PROSPECTIVE、reachable/witness未验证。不假装通过旧需要prior geometry的CPU selector。将来复用原四hub scout时，必须逐位置真实navmesh/可达角色检查，保持原事件像素、连续帧、闭合组件及完整trace规则；不得将被拒位置替换为未冻结位置。当前仅source queue，不提供可启动的GPU7运输，gpu_uuid=null、runtime_allowed/executable=false。

每屋沿已实测4hub构造的40000动作、2700秒worker、2400秒discovery、3000秒supervisor、6GiB content/7GiB disk预算。拟GPU7采集须主agent另核验运输、UUID与占位借还；不影响现有认证队列。≥30屋指本次事前覆盖源，不表示已经物理采集30屋，更不是已有万族数据。

完整官方来源行号/episode/start_rotation保留；reset为原yaw0及1.25m sensor高，标REQUESTED而非observed。每屋资产4件完整SHA，所有旧hub排除依据仅closed scout配置和冻结认证配置，无活动日志/HEAD输入。维持整屋FIT分组，不称全局未见；metadata角色不是实际见证角色。

SCOUT_QUEUE.json、每屋PREPARED_CONFIG/POSE_PROVENANCE/SOURCE_LOCK提供给后续新scout adapter。原27重放/18cells/54eval正式认证流程不变；本节点0GPU、0新物理hub、0新family，scientific_pass=false。
