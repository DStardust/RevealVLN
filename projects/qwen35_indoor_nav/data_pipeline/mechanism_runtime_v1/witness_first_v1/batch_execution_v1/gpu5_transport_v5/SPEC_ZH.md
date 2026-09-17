# GPU5 Transport V5：只修改运行中 worker 遥测采样协议

独立导入固定 SHA 的 V4→V3→V2 运输代码，不修改任何旧源、旧配置或旧批次。单个新工程重试 `batch_06r2`，仍用同 `language_ready_v2` 的 indices [9,10,11]；不替换四批 cohort 中已失败的 06r1，也不加入该 cohort。

旧 supervisor 源经原 GPU5 adapter 后，只有循环的一句 `snapshot=gpu(); upper=check_gpu(snapshot,proc.pid)` 替换成 `_telemetry_sample(proc.pid,started+3900)`。私有 build_supervisor 绑定已审核 wrapper，并在 main 的 finally 关闭日志。worker/factory/PartialTraceRunner/27 次认证/导出/数据质量完全不改；没有使用 batch07 的 budget batching。

每次 worker 遥测查询保留原始 XML，原 parse_gpu、check_gpu 不改；通过 DurableLog 在 guard 前 fsync 保存到 `GPU_RAW_SAMPLES.jsonl`。只允许精确 `AssertionError('MEMORY_ACCOUNTING')` 且通过 wrapper 更严格 gross-memory 条件时，在原 wall 截止内额外最多两次查询；从首次错误起总重采窗口 ≤1 秒（包括日志 I/O）。最终必须原 guard 真正 PASS；原失败样本不改写。其他错误、I/O 错误、超时直接失败。

这是**采样协议改变**，不是声称所有策略完全一样：原数值资源阈值、语义/动作/认证算法未改变。配置显式标注 `sampling_amendment`、`thresholds_unchanged=true`、`supervision_sampling_policy_changed=true`。所有 lease、holder 识别、信号、GPU drain、restore 查询和 guard 保持原 V2 函数，不经过 wrapper；初始 readiness、最终 cleanup GPU 查询也未包裹。

保持原 60 秒 readiness、3900 秒 supervisor、3600 秒/60000 动作总 factory，单候选 discovery 1000 秒/15000、certification 1500 秒/20000；输入解析上限沿用 V4 的 2048，不额外放宽。所有新增依赖、代码与主 agent 独立 AUTH/ID 在 run_v1 INPUT_LOCK 前置锁定。只有主 agent 主审及 MAIN_AGENT_GPU5_APPROVAL 全匹配后运行。

06r1 的 -15、MEMORY_ACCOUNTING、11.268614 秒账单、一条已提交零动作真实观察和缺失触发样本仍保留；此重试既非未暴露样本，也不修补此前证据。运行后需要完整 family/source/预算/资源/新原始遥测审计，不凭成功导出自动接纳。

重试来源不是字符串声明：准备/启动校验 AUTH.previous_supervisor_wall_seconds 等于旧实际值、旧 error/returncode/cleanup、LEASE/RESTORATION、同一源的三候选完整结构及旧恢复 holder 的精确 PID/starttime。重新验旧 HEAD/完整 journal hash chain、绑定唯一零动作单观察 trace。旧 config/input lock/process/终态/resource/readiness/journal/trace/semantic inventory/两个实际 content blobs 共16个证据文件加入新 INPUT_LOCK；原失败文件不改。

CPU 准備：`prepare.py --snapshot <WF>/multi_program_bank_v1/language_ready_v2 --name batch_06r2 --indices 9 10 11 --authorization <LINE>/authorizations/WITNESS_MULTI_PROGRAM_BATCH06R2_GPU5_V1.json --holder-identity <LINE>/authorizations/WITNESS_BATCH06R2_GPU5_HOLDER_IDENTITY_V1.json`。

准备 agent 不运行 GPU。新 run.py 由主 agent 独立审核后启动；不宣称物理吞吐倍数、模型收益或科学泛化。
