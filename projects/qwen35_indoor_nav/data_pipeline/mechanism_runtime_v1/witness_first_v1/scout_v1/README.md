# 跨屋真实 witness-bank scout V1

当前只有 CPU 代码与配置准备，executable=false，未启动 GPU。固定沿用 V1 三个已开发 FIT 房屋 17DRP5sb8fy / 1LXtFkjw3qL / 1pXnuDYAj8r 的资产、source_positions，既有 A/B/I/T 角色仅用于 backend permission 和初始 metadata 身份核验，不决定新程序。

每屋按实际 backend.objects 的完整有限词表 raw/category/room 分组。用只读 ProbeBackend 代理调用原 candidate_targets，保留原≤8m目标规则。对源坐标要求与 snap_position 偏移≤1e-5；按可达组覆盖数、最近距离、总距离和坐标稳定排序，选择至多2个相距≥1m的 hub，yaw0。每hub最多12组、每组2个目标；几何检查及所选配置在该屋首动作前保存，不将可达提案当可见证书。

每hub先实际 LRLRLRLR 公共尾，再对每目标用原反馈 follower 做实际 outbound，随后实际执行 outbound+紧凑逆回路。紧凑逆算子直接逐片压缩，未构建/执行旧完整展开 raw 路径。每次真实动作仍由 SingleCommitBudgetLedger 同步 durable reserve；原Journal的fsync、HEAD和链检查保持不变。实际回放使用已封存PartialTraceRunner，反馈包装只清本次捕获并记录异常，不改正常接口。

完整无碰撞motion trace写traces和BANK_RECORDS；其余写partial_traces，不进入bank。闭合标志须agent/rgb/semantic末端与起点均≤1e-5；事件使用完整角色映射的原Compiler阈值。同hub调用bank_cpu.propose，最多32个程序，仅候选，不认证新族。资源截断可从已完成组件输出CENSORED_HUB_PROGRAMS，未完成轨迹仍明确剔除。

预算：GPU1、监督4500秒、工厂4200秒/60000动作、每屋discovery1200秒/20000动作、content6GiB/总8GiB、RAM8GiB、自身GPU保守上界<4096MiB。XML包含图形进程；外部单PID≤768MiB且总≤2048MiB，初始util必须0。超过限制仅清自身worker，不操作GPU5占位、GPU2assembly或外部进程。继承监督器明确修正本层级ENV为qwen35_indoor_nav/.envs/q35n_habitat_v017_g0r。

TrackedContentStore通过新独立模块将FEEDBACK_ROOT显式绑定scout_v1，未修改旧store，已有CPU空store开闭测试覆盖此路径。整个run_v1不允许覆盖或自动恢复；失败保留，新运行须新版本。未知backend/IO异常上报并停止，不自动把未知当negative。

prepare.py只创建PREPARED_CONFIG.json与INPUT_LOCK.json，配置runtime_allowed=false。主agent审核后另建MAIN_AGENT_APPROVAL.json，内容为 approved=true、gpu=1、input_lock_sha256等于实际锁哈希；run.py验证后才新建run_v1并生成实际EXECUTION_CONFIG。缺审批时不会访问GPU。实际数据族认证、平衡动作计数、三seed、跨历史短窗口、18格及捷径审核全部留给独立assembly；本scout从不声称科学或训练PASS。

审批前语义修订：经主agent要求，runtime_config仅在严格验证approval与锁后将有效配置的runtime_allowed及executable同时置true；PREPARED_CONFIG继续保持两者false，training_allowed始终false。两个新增CPU断言覆盖已批准转换和未批准拒绝。此修订发生于任何run_v1创建之前，没有覆盖运行数据。
