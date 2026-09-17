# 普通导航基座 V2：CPU 训练准备交付

日期：2026-09-10，Asia/Shanghai。

已经把正式普通生产池接入独立的新训练准备目录。数据快照、因果读取接口、
完整轮次参考训练入口和固定开发来源已落地；没有启动GPU训练，也没有获得新导航结果。
这次目标是先建立可闭环导航的普通基座，特殊机制数据和交叉续接损失暂不加入。

## 实际数据

八个闭合池的凭据与原索引哈希、逐指令/路线元数据及末端STOP核对后，结果为：

| 项目 | 实际值 |
| --- | ---: |
| 训练房屋 | 51 |
| 物理参考路线 | 5,849 |
| 原始指令记录 | 17,548 |
| 不同路线决策 | 464,877 |
| 指令条件动作监督 | 1,394,744 |
| 检出的相同指令/真实rollout重复行 | 0 |

R2R：2,898路线、8,700指令、532,416动作；RxR：2,951路线、8,848指令、862,328动作。
路线长度含STOP为4–416步，中位67，95分位171。
三遍计划暴露4,184,232条动作；它们仍是同一数据池的重复训练，不是新增独立数据。

动作分布：前进870,438，左转259,949，右转246,809，STOP 17,548。
STOP占1.258%，前进占62.4%。保留全部真实终止监督和四类统计，不以多数类accuracy判成功。

只使用既有FIT的官方人工R2R/RxR指令。活动中的EnvDrop和特殊族不在这次训练快照内。
闭合质量凭据是既有生产审核；本轮新增核验的是索引、元数据、因果投影与读取接口，
没有重新执行全部仿真认证，也没有重新对139万条记录逐张解码。

## 已实现与实测

- `prepare.py`：闭合池合并、质量凭据/计数/隔离核验、全量policy/supervision哈希、
  去重与FIT隔离，写入独立`snapshot_v1`；保存快照SEAL，不改变旧生产文件。
- `data.py`：正确解析ROOT相对`sourceRoot`；逐指令零记忆、顺序读取、最近两帧和
  八个过去动作；policy只含instruction/images/executed_actions；监督和控制字段分离。
  PNG延迟读取并校验原RGB像素哈希，源JSON按快照哈希校验。
- `runner.py`：三遍全量调度、普通CE、4步TBPTT、8段按真实决策数归一累计；
  短尾/STOP不丢弃，epoch 1/2/3 checkpoint、epoch边界恢复、完整epoch在线CE和
  混淆矩阵、STOP指标及有界频率进度日志。调度只在每个epoch生成一次全量顺序。
  这是尚未GPU验收的单设备参考实现，不宣称已解决多卡吞吐。
- 主agent复跑34项CPU测试，现有Qwen环境34/34通过，包含真实PNG损坏反例。
  stdlib环境33通过、1项因无PIL跳过；不是GPU模型接口测试。
- 本地AutoProcessor CPU检查全部17,548条指令，24个样例核对完整图像token展开。
  72条指令的保守输入上界超过旧512限制；最大805，1024以内全部容纳。
  新候选配置为1024并拒绝截断；较长输入的GPU显存/梯度尚未验证。
- 按99个来源×房屋组合实际读取首尾样本，297次RGB引用解码/像素哈希核验通过；
  这不是297条独立路线或全图像库重审。全程不加载模型权重、不做模型forward。
- 根框架`bash scripts/research.sh check`通过；其历史实验注册检查不替代本目录接口验收。

精确结果分别见`snapshot_v1/RESULT.json`、`TOKEN_AUDIT.json`、`PREFLIGHT.json`。
`CODE_SEAL.json`封存本版本源码及SPEC，`PROTOCOL.json`绑定代码、来源快照与旧模型资产哈希。
原模型权重哈希来自既有SOURCE_LOCK；本轮只复核processor等模型元数据，训练前必须重验权重。

## 固定评估准备

沿原5个INTERNAL_DEV房屋，从官方train来源元数据确定88条不同路线，各取一个原指令ID。
原拟每个房屋×来源取10条，但PuKPg4mmafe/RxR只有2条、XcA2TqTSSAj/R2R只有6条，
故保留8+4条缺额，不换房屋或从训练池补数。首次CPU准备的配额失败保留在
`PREPARATION_ATTEMPTS.md`，更改发生在任何模型评估前。

开发来源仅为固定episode选择，未生成开发轨迹、未运行闭环；确认房屋不参与训练/调参。
后续按原场景分组，在固定episode评估SR/SPL、nDTW/SDTW、到达但不停、过早停止、
碰撞与预算失败；在线训练混淆矩阵不能替代独立闭环结果。

## 正式开训尚需完成

PROTOCOL保持`PREPARATION_ONLY`、`gpu_run_allowed=false`，本目录无自动借卡/后台开训器。
以旧双卡3.458决策/秒线性外推，本池一遍约4.67天、三遍约14天，尚无新实测速度。
下一执行节点应先在自然释放并核验的GPU上完成新接口长输入、真实前后向、
重载、输入响应/STOP与梯度诊断，验收批处理/多卡吞吐，再明确完整轮次的运行预算。
参考runner只在step边界检查墙钟，正式运行还需独立进程级资源监控与占位恢复transport。
这些未完成项已显式登记，不把CPU通过写成正式长训可立即启动。

不再将十路线95%作为准备大池训练的前置条件；但数据少也不是过去STOP为零的唯一已证实原因。
保持普通基座目标，先查可复现的接口错误，再以足量训练和独立闭环判断导航能力。

## 同期生产状态

本轮没有停止或重复启动任何生产队列。11:07左右只读复核时，六条仍在运行；
GPU2队列在batch243正常关闭后，batch244启动前触发`EXTERNAL_RESOURCE_LOAD`，
依据原规则自行停止，无worker启动、无自动重试、无GPU进程由queue发信号。
原readiness快照含新外部进程1380MiB；不能把这张卡视为空闲训练资源。
失败证据在`data_pipeline/auto_production_v1/special_gpu2_scale_v2/lane_gpu_2/RESULT.json`
及`data_pipeline/mechanism_runtime_v1/witness_first_v1/batch_execution_v1/batch_244/run_v1/LAUNCH_RESULT.json`。
旧锁、失败与未启动预约均保留，本轮没有修订或重启该lane。

## 只读复查命令

从项目根执行：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B \
  projects/qwen35_indoor_nav/sft_acceptance/ordinary_baseline_v2/runner.py \
  --preflight \
  --protocol projects/qwen35_indoor_nav/sft_acceptance/ordinary_baseline_v2/PROTOCOL.json \
  --snapshot projects/qwen35_indoor_nav/sft_acceptance/ordinary_baseline_v2/snapshot_v1
```

新数据必须另建快照版本；不要覆盖当前snapshot_v1或原封存来源。
