# 实际读取、来源差异与无法访问项

审查基准：`9d22783e646ef86232ba3a3a59d7bb9ce8a8c0a7`。本轮实际起点：`09bdc47e5684cd06411b8d9ba8bb1aa5bbcf30db`，分支`codex/q35n-cycle-pair-v4-20260917`。
旧V4目录已存在且终止，不能按“新建”要求覆盖重置。此次工程补记位于其`codex_return_20260917/`子目录，旧结果和源码保持只读。

## 已实际读取的指定源文件与事实记录

- `CURRENT_STATUS.json`
- `MAINLINE_FREEZE_V3.md`
- `reviews/Q35N_P1_PAPER_CORE_ADJUDICATION_V2/REPORT_ZH.md`
- `reviews/Q35N_RECOVERY_20260917/FINAL_REVIEW.json`
- `reviews/Q35N_RECOVERY_20260917/REPEATED_INPUTS.json`
- `reviews/Q35N_RECOVERY_20260917/NUMERIC_TRANSPORT_DIAGNOSIS.json`
- `closed_loop_bench/ordinary_cycle_recovery_v1/review.py`
- `closed_loop_bench/r2r_ce_tiny_v1/common.py`
- `closed_loop_bench/r2r_ce_tiny_v1/metrics.py`
- `sft_acceptance/ordinary_sync_recovery_v1/model.py`
- `sft_acceptance/ordinary_sync_recovery_v1/data.py`
- `sft_acceptance/ordinary_baseline_v2/data.py`
- `deployment/ordinary_v1/MODEL_CARD.json`
- `deployment/ordinary_v1/DEPLOYMENT_ACCEPTANCE.json`
- `deployment/ordinary_v1/predict.py`
- `sft_acceptance/v1/policy.py`
- `data_pipeline/mechanism_runtime_v1/exporter.py`
- `data_pipeline/mechanism_runtime_v1/witness_first_v1/ordered_visit_v5_cpu/visit.py`
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/cycle_policy.py`
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/evaluate.py`
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/common.py`
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/executor.py`
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/aggregate.py`
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/review.py`
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/launch.py`
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/PROTOCOL.json`
- `closed_loop_bench/ordinary_cycle_pair_gpu1_v3/SOURCE_LOCK.json`
- `closed_loop_bench/r2r_ce_full_v2/run_001/RESULT.json`
- `closed_loop_bench/ordinary_expanded_dev_after_single_v1/run_001/RESULT.json`
- `closed_loop_bench/ordinary_full_epoch_dev_v4/run_001/RESULT.json`
- `closed_loop_bench/ordinary_history_pair_eval_r1/COMPARISON_RESULT.json`
- `sft_acceptance/ordinary_learnability_v1/current/RESULT.json`
- `reviews/Q35N_FULL_FIT_NATIVE_REPLAY_V1/RESULT.json`
- `runtime/models/Qwen3.5-2B_15852e8/config.json`

还读取根/子目录AGENTS与研究执行规则；Compiler、FamilyLoader、exporter、V5到访CPU原型、旧8槽policy.step和本机模型config。
研究资产机器清单见`READ_MANIFEST.json`（逐文件字节数/SHA）、`FACT_SOURCE_READS.json`；这些读取含原始content数组校验，
不是只复述交接包。完整列表含215份导出元数据和一个族的内容/标签回读，不等于库内全部物理证书重新认证。

## 差异

固定提交至本轮起点：已提交变化只有旧V4阻断包及CURRENT_STATUS引用；根.gitignore/AGENTS/README为既有未提交修改，保留。
本轮再次对V3源锁198项核验，无不匹配；实际模型和best4k满足指定SHA。工作区清单见`WORKSPACE_AUDIT.json`。
本轮新增回交目录、研究规格/CPU合同与V4补记；不修改冻结模型、训练、公共成功定义、恢复算法、旧MAINLINE或旧FINAL_REVIEW。

## 无法访问与未执行（分开）

指定既有本地源文件没有访问失败。V4 evaluate.py/launch.py运行集成尚不存在；研究model/train/evaluate尚未实现，这是实现缺项，不是网络错误。
没有有效GPU分配或实际使用者专用窗口凭据。模型未加载，参数/buffer运行指纹、dispatch、Triton配置和跨步梯度无法以本轮实测报告。
未访问用户Pro会话，未进行新仿真或端到端论文复现。GitHub包含Compiler/FamilyLoader源码和本轮审计记录，
不含本次候选族的完整export_v4、RGB/场景/底模；现有CPU测试依赖这些本机资产，不能宣称远端克隆即可重跑。
候选族MANIFEST SHA及逐文件读回凭据在审计JSON，远端复核需区分文件证据与可重跑资产。
历史扩窗`ordinary_history_pair_eval_r1/COMPARISON_RESULT.json`也是本机旧记录，未收入固定Git提交；
账本将source_commit记为null并保留实际SHA。交付校验捕获了最初错误的提交归属，
修正和失败证据见`VALIDATION_ATTEMPT_001.json`；旧历史结果没有修改。

外部仅重新读取SAP-Nav、Progress-Think、PSR、Reward Machines的官方摘要/版本入口；其余按附件和旧P1记录归档，详见`RELATED_WORK_DELTA_ZH.md`。
本轮完成14项新CPU合同测试，未重跑旧15项V4、5487输入FIT或256/128拟合；不将旧PASS计为新实验。
