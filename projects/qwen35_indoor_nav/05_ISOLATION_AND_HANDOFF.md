# 隔离、目录与下一 session 交接

## 物理边界

本路线唯一新写入位置：`/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav`。
没有移动 V1/V2 审查或旧 A/B/C/UAD 文件；新入口集中在本目录，历史来源用明确路径引用。
这属于目录和流程隔离，不是已经建立的容器安全边界或防泄漏沙箱。

未来确需实现时，目录约定如下；本轮不创建空环境、不下载资源、不提供绕过准入的启动脚本：

- `src/`、`tests/`：本线代码与测试。
- `configs/`：独立实验协议，所有正式字段冻结后才可执行。
- `data/manifests/`、`data/policy_input/`、`data/supervision_only/`：来源、策略输入与特权标签分离。
- `runtime/envs/`、`runtime/cache/`、`runtime/models/`：项目内独立运行资源，建立前核查体积和许可。
- `artifacts/Q35N_*/`：每个版本独立结果、失败、日志和暴露登记。

共享已有场景若以后必要：先核验物理路径仍在根项目内，登记只读来源与哈希，不跨线写缓存。
不在本目录新建嵌套 git 仓库，也不修改根工作树用户未提交的其他文件。

本次 `git check-ignore` 确认根 `.gitignore` 的 `/*` 规则忽略了新 `projects/` 目录。
文件已经实际保存，但当前未纳入 Git 跟踪；这不等于已提交或已有版本备份。
为遵守本次目录内写入边界，没有修改根忽略规则或强制 git add。需要版本管理时再明确纳入范围。

## 只读历史来源

- `research/designs/QWEN35_EVIDENCE_NAV_SCOPE_REVIEW_20260909_V1.md`
- `research/designs/QWEN35_NAV_PREIMPLEMENTATION_VERDICT_20260909_V2.md`
- `artifacts/experiments/B15_CANDIDATE_ITM_NAV_V1/SEMANTIC_DIAGNOSTIC_STATUS_V1.md`

这些是来源路径，不是允许恢复原实验的执行指令。

## 下一 session 可直接使用的任务

以下引用为初始版本历史交接。**当前任务改为**：先阅读 [冻结 V3](MAINLINE_FREEZE_V3.md)、[P1 裁决](reviews/Q35N_P1_PAPER_CORE_ADJUDICATION_V2/REPORT_ZH.md) 和 STATUS；仅执行 `Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1` 的只读资产/官方接口核验、数据与实现规格和资源规划。旧 G0B 暂停，不按旧交接再开泛搜，不安装、不训练、不操作 GPU。主线研究预审不等于数据准入。

> 只推进 `/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav`，先完整阅读根和本目录 AGENTS，以及 README/STATUS/01–04。
> 用户要求先审查创新性和科学可行性，之后才能实现；当前尚未准入。
> 沿 02 的唯一候选完成一项有界任务：给出具体训练示范构造算子，并逐操作核对 GPO、DESPOT 和 CAST 的最近公式与可用官方代码。
> 不以 POMDP 思想相同直接否决，也不把标准规划器造数据再 SFT 自动算原创。
> 同时核查该算子需要的数据字段是否能由声明的模拟器可靠提供，尤其真实场景变体、完整历史可区分性和实例身份。
> 不必搭完整网络、不安装环境、不启动 GPU/训练、不生成正式实验结果、不改 A/B/C/UAD 文件。
> 输出本目录内的新版本审查：精确主张、来源版本、算法映射、最简单反例/替代、监督来源、资源风险、GO/REFORMULATE/NO-GO。
> 若缺口无法闭合，直接指出哪个操作失败；不能留下“创新性已通过，后续再查核心近邻”的状态。

不要主动把未发表的本地方案、代码或轨迹上传第三方研究服务。检索公共概念与论文可以；私有内容外发需另行明确范围。

## 初始版本交付验收（历史记录）

- 所有新增文件在本目录内，旧文件不迁移或覆盖。
- 状态机没有科学 PASS、算法准入或虚构结果。
- 文档中的待建目录不冒充已安装工程。
- JSON 可解析，文档内部相对链接存在；这些仅为文档检查，不是科研实验。
