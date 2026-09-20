# Q35N V16 rqf 修复：异机主 pro agent 完整交接

## 交接目的

这是一次“当前事实 + 决策请求”交接，不是新的实验提案。请在异机上先审阅本文件和同目录的 JSON 证据，再决定是否继续当前候选搜索或批准新的物理数据生成修复版本。不要重做历史实验，不要改变论文主线，不要把当前采集状态写成训练失败或导航收益。

## 代码位置

- GitHub 仓库：`DStardust/RevealVLN`
- 分支：`codex/q35n-grounded-state-v16-20260920`
- 本次交接对应提交：`d1349b99fc659f4461737bd9698e8471ea075571`
- 代码目录：`projects/qwen35_indoor_nav/research/continuation_memory_v1/grounded_state_transfer_v16/`
- 本交接目录：`handoff/pro_agent_20260920/`

本分支已经包含以下版本化修复提交：

1. `3a542afe`：新增 `collect_repair_v1.py` 和 `repair_resume_v1.py`；
2. `e2a9fa78`：修复服务优先选择空闲 GPU；
3. `1a122b9c`：允许仅物理采集阶段在有余量的占用卡上运行，模型阶段仍要求空闲卡/显存门槛；
4. `d1349b99`：修复可恢复 preflight 标记写入。

## 必读顺序

请按以下顺序读取，不要从旧草案推断当前状态：

1. 本文件；
2. `STATUS_SNAPSHOT_20260920T083232Z.json`；
3. `LATEST_UPLOAD_STATUS_20260920T084017Z.json`（若存在，以它覆盖前一快照的 live 进度字段）；
4. `FAILURE_EVIDENCE_20260920T083232Z.json`；
5. 上级目录的 `HANDOFF_ZH.md`、`PROTOCOL.json`、`README.md`、`REPORT_ZH.md`、`RESULT.json`、`GITHUB_HANDOFF.json`；
6. `collect.py`、`collect_repair_v1.py`、`repair_resume_v1.py`、`pipeline.py`、`promote_pilot.py`。

若能访问原执行服务器，再读取 live run 中同名文件；本交接 JSON 是固定时间点快照，不能替代 live 状态。

为便于异机审核，本次 GitHub 提交还上传了旧 run 的 shortfall/失败候选证据，以及修复 run 的 `REPAIR_SPEC`、`SOURCE_LOCK`、proposal 列表、当前失败 JSON 和服务 submission/status 快照。它们位于与服务器相同的 `runs/...` 相对路径，是上传时的只读快照；服务之后新增的尝试不会自动出现在这批文件中。

## 历史事实（只读保留）

- 原 run：`runs/v16_formal_001`。
- FIT `16/16` 族、DEV `2/2` 族已经完成；TEST 原有 `6/8` 族。
- 缺失屋为 `rqfALeAoiTq`，缺少 TEST 两族。
- 原候选 `117` 个，全部未通过；原失败结果和源码锁不覆盖、不删除。
- 因此原 V16 在采集阶段停止；九模型训练、720 次真实自主续接、方法收益判断均尚未开始。

## 当前修复事实（快照时间）

快照时间：`2026-09-20T08:32:32Z`（北京时间 16:32:32）。

- 新 run：`runs/v16_repair_rqf_001`。
- 修复版本：`v16_rqf_full_neutral_yaw_search_v1`。
- 已按值复用旧 run 的 `24` 个已认证族；修复只搜索缺失屋的 `2` 个族。
- 独立服务：`q35n-v16-repair-rqf-20260920-04.service`，状态仍为 `RUNNING`。
- 采集阶段：`collect_repair_test`。
- 快照时已完成候选尝试 `103`，接受 `0/2`；journal 为 `104 START`、`103 REJECTED`，第 `104` 个候选已开始。
- `FAMILY.json` 接受数仍为 `0`；修复屋没有最终 `HOUSE` 文件；只有 preflight 完成标记。
- 失败均为物理合法性拒绝，顶层原因为 `REPAIR_NEUTRAL_YAW_EXHAUSTED`。各中性朝向累计原因：
  - `HISTORY_EVENT_STATE_NOT_ISOLATED`: `481`
  - `NO_ANCHOR_NEUTRAL_START_VIEW`: `188`
  - `COLLISION`: `129`
  - `NO_VISIBLE_SEE2_WITNESS`: `26`
- 代表性失败样本保留在 live run 的 `proposals/*/FAILURE.json`，快照聚合见 `FAILURE_EVIDENCE_20260920T083232Z.json`。

## 不变的科学/物理门槛

`collect_repair_v1.py` 只改变候选发现路径：遍历冻结中性 yaw，并在旧候选耗尽后使用确定性 pathfinder 追加 hub。以下门槛没有放宽：

- SEE2 阈值及连续两帧要求；
- 任意碰撞即拒绝；
- 实际主动 STOP 与终止见证；
- 总决策上限 `500`，历史上限 `240`；
- 历史事件状态隔离；
- 不读取模型分数，不按模型成绩选屋、选条件或换测试屋；
- 训练输入与部署策略信息隔离。

已完成的 `24` 族不能重采集。旧 run、旧源码锁、旧失败和当前修复失败必须只读保留。

## 资源和运行边界

- 所有 GPU 在快照时都有外部进程：GPU0/1 为 NavDP，GPU2–7 为 SparseDrive。
- 本次物理采集在 GPU1 的剩余显存上运行；没有停止、暂停或释放任何外部进程。
- 当前服务只做物理采集；尚未加载 Qwen、未训练、未生成正式数据族、未进行导航评测。
- 不要从异机远程停止该服务，也不要杀任何外部 PID。若服务自然结束，读取其最终 receipt；若需要中断，必须先由主 agent 明确裁决并保留资源记录。

## 现在需要主 pro agent 做的裁决

这是当前唯一需要的指导，不是要求立即改代码：

1. 是否允许 `v1` 继续到一个事先写明的候选数/时间上限；或
2. 是否批准另开 `v2` 物理生成版本，针对 `HISTORY_EVENT_STATE_NOT_ISOLATED` 与 `NO_ANCHOR_NEUTRAL_START_VIEW` 先做离线诊断，再用新 run 验证。

在收到裁决前，不要修改当前 live run，不要覆盖 `v1`，不要放宽门槛，不要用模型分数筛选候选，不要进入 G1/训练/720 次续接。若批准 `v2`，必须使用新版本名、新 run、独立 source lock，并明确复用 `24` 族；不得覆盖 `v16_formal_001` 或 `v16_repair_rqf_001`。

## 可复核命令（服务器可访问时）

```bash
cd /mnt/data_nas/deeprobotics/daiyang/vla
git checkout codex/q35n-grounded-state-v16-20260920
git rev-parse HEAD

run=projects/qwen35_indoor_nav/research/continuation_memory_v1/grounded_state_transfer_v16/runs/v16_repair_rqf_001
cat "$run/COLLECTION_PROGRESS.json"
python - <<'PY'
import json, glob, collections
run='projects/qwen35_indoor_nav/research/continuation_memory_v1/grounded_state_transfer_v16/runs/v16_repair_rqf_001'
rows=[json.loads(x) for x in open(run+'/COLLECTION_ATTEMPTS.jsonl') if x.strip()]
print(collections.Counter(x.get('status') for x in rows))
print('FAMILY',len(glob.glob(run+'/proposals/*/FAMILY.json')))
print('FAILURE',len(glob.glob(run+'/proposals/*/FAILURE.json')))
PY
systemctl status q35n-v16-repair-rqf-20260920-04.service --no-pager
```

## 结论边界

当前结果只能说明“修复候选仍未找到两个物理合法族”。它不是训练失败、不是方法失败、不是导航收益，也不是任何正向论文结论。主 agent 的回覆应只包含：继续上限、暂停等待、或批准新版本诊断/修复三者之一，并保留本分支已有的历史证据。
