# 执行 Codex：G0R 独立环境与渲染验收

主 agent 已接收 P2R1 静态规格，主线不变。**用户现已明确批准本 G0R，可按本交接开始执行。** 权限记录见 [G0R 用户授权](authorizations/G0R_EXECUTION_AUTHORIZATION_V1.json)。原 P2R1 草案和接收报告中的“待确认/不准动占位”是历史状态，仅在本授权精确范围内被覆盖。

## 启动前

项目根仅 `/mnt/data_nas/deeprobotics/daiyang/vla`；全部新环境、依赖、构建、缓存、脚本及输出仅在 `projects/qwen35_indoor_nav` 内。
完整读取根/本线 AGENTS、README、STATUS、主线 V3、06 工作流、P2R1 的 RUNTIME_SETUP_GATE_DRAFT.json、RESOURCE_PLAN_V2.json 和 SOURCE_EVIDENCE.md，以及 [主 agent 接收与澄清](reviews/Q35N_P2R1_MAIN_AGENT_ACCEPTANCE_V1/REPORT_ZH.md)。
用户确认已登记，在节点日志引用该记录，不需重复索取同一许可：

1. 用户确认 MP3D 合法取得且有授权文件，作为本节点本地研究使用依据。主 agent 尚未读取文件，不冒称条款已审；若可在项目内定位可登记其路径/hash，不访问其他文件夹、不外发文件。此确认不授权公开发布或商业使用。
2. 用户批准本节点安装/下载/渲染及以下资源上限。
3. 可先选择空闲卡；若没有空闲卡，可暂时释放后面卡上**已确认的用户占位程序**，使用完毕必须恢复。实际训练、推理、其他用户或用途不明的进程不在许可内。

本交接和授权记录构成针对 G0R 的局部工程授权；算法、family replay 和训练的禁止仍有效。不要修改全局 AGENTS 或把 implementation_allowed 全局设为 true。先写本节点 execution config 与权限覆盖记录，原草案 executable=false 不静默修改。

## 唯一任务

创建与旧线隔离的 Habitat-Sim v0.1.7/challenge-2021 环境；固定源码 commit `856d4b08c1a2632626bf0d205bf46471a99502b7`，若使用 Habitat-Lab 则固定 `d6ed1c0a0e786f16f261de2beafe347f4186d0d8`。
依赖先形成节点内锁定计划，记录官方来源、版本、许可证与大小依据，再按计划安装；不引入未经登记的全局包/用户缓存。
只读访问已登记、物理路径位于项目内的场景和固定源码；不在旧代码树构建，不运行旧可写环境，不跟随外部软链接。

只验以下事项：

- 独立环境 import 与源码/依赖指纹。
- 旧暴露接口场景 `17DRP5sb8fy` 加载。
- 224×224 RGB 与 semantic 对齐，HFOV90、sensor 高度1.25m；agent/actions 采用 P2R1 物理规格。
- reset 和一个合法动作返回观测及碰撞反馈，确认 renderer、设备与实际资源。
- 不生成 D/K/B 历史族，不计算 18-cell Y，不运行 Qwen，不训练，不声称导航成功率。

## 资源上限

总新增磁盘 40 GiB（环境12、构建8、cache16、日志4），累计网络下载12 GiB，host RAM32 GiB，墙钟8小时，一张空闲或依下节合法释放的卡且渲染显存8 GiB。
记录 GPU UUID 与当时 ordinal，检查可用性；8 GiB 是本进程渲染显存上限，不是要求物理卡只有8 GiB。其他实际任务占用设备时停止，不抢占。
不下载模型权重、训练数据或额外场景。超预算、需换源码版本或驱动、需越界安装时保留日志并回交，不自作扩大。
所需安装源清单冻结后可以实现本节点最小验收脚本；用 apply_patch 编辑。不是全套 VLN 算法实现授权。

## 交付与退出

节点输出：`reviews/Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1/`。环境/build/cache 按 P2R1 路径，但需先验证实际路径和不存在冲突。
保留用户授权记录、执行配置/hash、source/dependency lock、来源下载账本、GPU/renderer/driver/物理配置指纹、场景资产哈希、smoke 原始观测/日志、资源测量、失败和 SHA256SUMS。
result.json 区分 runtime_pass、renderer_pass、scientific_pass=false、new_training_runs=0、navigation_gain=null；记录实际 simulator smoke/action 次数，不因“不是效能实验”把发生过的环境操作写成0。
工程结论为通过或有具体阻碍；不能借失败更换主线或背离固定版本。
复核旧 P2/P2R1 产物不变；不要改全局 STATUS/主线或旧线文件。退出自己启动的服务并释放自身资源；只恢复本节点暂时释放的原占位程序，不在原本空闲的卡上新增占位。
完成即回交主 agent，禁止自动执行 G1F/G2 或训练。真实 schema/label/索引澄清留待对应 G1F/G2，不扩进本次 G0R。

## GPU 借用与恢复（用户新增授权）

1. 先只读检查 GPU、进程和 tmux；有空闲可用卡就用空闲卡，无需停止占位。
2. 没有空闲卡时，从较高编号卡开始核对候选占位。记录 PID/启动时间/属主/完整命令、项目内脚本和作用、cwd、原 GPU UUID/ordinal、tmux pane/session、remain-on-exit/重启设置。不得仅以 GPU 利用率100%或名称像占位来认定。
3. 停止前确认准确且可恢复的启动方式，保存仅必要的非敏感环境与参数；恢复若依赖项目外脚本/资产或无法核实身份，停止并回交，不能越界读写。运行前再次核对 PID 身份防止 PID 复用。
4. 只针对确切占位进程作最小释放；不批量 pkill、不杀整个 tmux/session 或进程组，不释放多余 GPU。优先程序自带退出/释放机制，其次温和终止；不能用 SIGSTOP 假定显存已释放。未经证据不要升级强杀。
5. GPU 借用只在真正需要 renderer 前发生，避免整个安装期间白白占用。记录释放后实际设备空闲证据及本节点开始/结束时间。
6. 在成功、失败、超时和中断路径都执行清理：关闭本节点 GL/CUDA context，核对无自身残留，再按原命令/设备恢复占位并恢复 tmux 设置。恢复前检查未有同一占位实例，防止重复启动。
7. 保存 GPU_LEASE_BEFORE.json、GPU_LEASE_ACTIONS.jsonl、GPU_RESTORE_RESULT.json（或同等结构记录）。若被无关任务占据不能安全恢复，不杀该任务；报告 cleanup_incomplete 和准确恢复指令。不得声称“已恢复”却没有 PID/设备实测证据。
