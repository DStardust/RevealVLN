# 当前推进状态（更新至2026-09-14 06:17 CST）

最新目标修订：先仅普通导航数据，在完整R2R-CE v1-3 val_unseen 1839条达到SR>=40%；再独立验证轻量贡献，随后此前任务记忆创新向约60%推进。当前40%未达成，后两阶段未准入。详见`reviews/Q35N_SR40_BASELINE_RESET_V1/ROADMAP_ZH.md`。旧“内部100条新正号”目标被用户替代但未达成，所有FAIL保留。

内部开发最好仍为纯扩产4k：SR21%、SPL18.0169839%、nDTW36.5879701%；历史完整1839条是旧41800权重SR22.0228385%、SPL18.6825707%。二者不是同一checkpoint/集合，不能串成学习曲线。新阶段固定batch1，并版本化记录与旧full_v2允许batch8回退的差别。

## 最新结论与下一步（2026-09-14 06:17 CST）

- **普通历史配对R1已全部结束，两组均不采用**。新8帧完整内部100条SR12%、SPL9.9273884%、nDTW32.0730252%、OSR50%；对旧best4k21%为5赢14输，三指标开发门槛失败。相对新2帧10%仅8赢6输、SR+2点，5屋bootstrap区间跨零，不能当稳定提升。自动`COMPARISON_RESULT.json`已闭合、完整1839候选null；没有启动退化权重的完整基准。普通SR40与后续创新目标仍未达成。
- 新8帧21780动作、10381碰撞、70条STOP；12成功、38曾到目标未成功停、29未到目标即停、21未到目标耗尽预算。独立`reviews/Q35N_HISTORY_PAIR_R1_POSTAUDIT/treatment_prefix8_RESULT.json`已PASS_AUDIT_NOT_SCIENTIFIC_PASS：6048源锁、全21780因果帧输入、旧/新200条官方SR/SPL重放、聚合nDTW/sDTW、同协议/权重和清理。0新增前向/仿真/训练；负向结果不因审核PASS改写。
- 8帧评测墙钟2272.37秒，exit0、无遗留进程或外部信号；GPU1实时NVML12MiB且无context，worker3888091与watcher3770736均已自然退出。GPU3/4/5占位已恢复，身份见下方训练收口及实时/proc，不按历史PID操作。原18766已实际展示两组完整10%/12%及无候选结论。当前没有在途本轮训练/评测，也不重启已闭合队列。
- 完整失败解释/配对/成本见`reviews/Q35N_HISTORY_PAIR_R1_POSTAUDIT/PAIR_FAILURE_REVIEW_ZH.md`。R1两组训练启动墙钟与单卡评测按卡数计约14.33 GPU小时；旧V1三卡200.99秒和其他诊断/CPU/借还成本另存，非项目总成本。回执training_processes_unchanged=false是前后6项均null但旧判据要求非null；实时无训练进程且无信号，不能据该字段误报干扰。
- 下一优先区分为原执行入口对完整已存FIT输入的复现，然后最小FIT可学性/监督对齐，再决定是否测试标准全层适配。旧V12的5487输入80动作不一致仍FAIL；原前缀32跨池输入可逐位复现旧输出，模型源码/权重和inference_mode一致，但尚未定位提取器偏差，不能归因为本轮退化。当前仅完成源码只读核对，新的GPU检查尚未冻结/启动；不扫描学习率、不延长已闭合训练、不恢复STOP拟合或启用特殊数据。

## 本轮过程快照（以下在途状态均为历史）

- **8帧训练已完整结束，最终验收通过，GPU1评测实际接续**。05:35已完成18/100条，worker3888091/start168709235（权威记录`closed_loop_bench/ordinary_history8_dev_r1/run_001/PROCESS.json`），launcher3888075/start168708011，watcher3770736仍存活且处于`EVALUATING_FIXED100`，无失败。以上PID仅快照，操作前必须实时核验；不得重复启动。完整8帧SR/SPL/nDTW仍未知，不统计部分SR。
- 8帧固定4000更新/384000普通决策正常完成，最终权重SHA `7c594519a8d152561656376bf0eb7705c8a93910f49e6fe0ec0d6864c5de135e`，评测`PROTOCOL.json`已绑定同一路径/哈希/4000步。`treatment_prefix8/ACCEPTANCE.json`为`PASS_FINAL_TRAINING_ONLY`；三rank全部exit0、无清理信号、预取线程join、28组FP32参数和AdamW矩有限/4000次更新。另作0 GPU CPU只读重算，实际checkpoint参数指纹`266beb8894a28f35714e16e92c99bc2d25755bb4cdd5d08d71885b60b5188014`与三个最终rank记录精确相同，434180可训练参数。启动墙钟9845.41秒，实际训练循环9646.45秒；峰值reserved各rank约16.94GiB。不能据此宣称导航通过。
- 同一lease已正常关闭：`lease_v1/LEASE_RESULT.json`的execute_returned/holders_restored均true、error null、external_processes_stopped0。GPU3/4/5先清空，再恢复占位为3887550/start168705115、3887646/start168705567、3887737/start168705929；本轮已逐一比对实时/proc的start/argv/cwd及NVML对应GPU PID，均真实存活。GPU1上3888091及其8个渲染worker实际存在，不是只有文件显示EVALUATING。旧GPU6/7及18768/18769未操作。

- 04:49最新核验：8帧3020/4000步，同一组三rank3804401/3804402/3804403与自动watcher持续存活，仍等待固定4000验收，无失败、0已完成8帧评测episode。本轮为对已确认活跃句柄的持续有界等待，无新增训练、试验或重启。第3000步检查点已落盘，实际SHA `528300e7810ddd8d81f02b462605e7e7d55379382d66fd559e38567164b5ca4f`与receipt一致，cursor3000/96000本rank决策、全局288000；本次只核存储哈希/回执，不冒称新参数数值或导航验收。CPU RSS52.03GiB，GPU3/4/5分别18512/18504/18504MiB，最近窗口42.67决策/秒，剩余原墙钟预算10366秒。保留固定4000选择，不新增中途评测。

- 04:13最新核验：8帧2060/4000步，原rank3804401/3804402/3804403和自动watcher仍实际存活，无失败，8帧评测尚未开始。本轮对同一活跃运行持续有界等待，没有重启、延训或改配方。第2000步CPU验收通过：`reviews/Q35N_HISTORY_PAIR_R1_CHECKPOINTS/treatment_prefix8_STAGE2000.json`，192000决策、28组FP32参数和AdamW矩有限、原三rank初始化一致；权重SHA `d09e0f8660087b344cd40fe319425dd72e30300e79622ebd7bfd861e41881026`。新增`stage2000.py`仅只读复用冻结first200检查器、替换精确步数/计数/输出名，未改旧脚本或其控制组回执。最终仍只选4000，数值验收不等于SR改善。

- 03:53最新实时核验：8帧1540/4000步，原rank3804401/3804402/3804403及watcher实际存活，无失败回执；2帧完整100条仍SR10%，8帧尚无导航结果。本轮已完成新的CPU只读日志/源码核对，并对同一活跃进程作两次55秒等待后复查；不是重启或延训。
- 新`reviews/Q35N_HISTORY_PAIR_R1_READONLY_DIAGNOSTICS_V1/REPORT_ZH.md`与已实跑`RESULT.json`：固定截至1200步的两组各61行日志前缀SHA、每窗口动作数与目标分布核对通过。1001–1200步同19200决策，2帧/8帧accuracy70.88%/68.42%，STOP recall23.72%/12.41%；这是变化中模型的训练批统计，不是验证/SR。8帧在适应旧2帧初始化，但仍不能宣称收益。源码与实际检查点核对当前可训练434180参数；如未来诊断支持，18层linear attention的qkv/out rank8适配静态增加1769472参数。这仅为已知工程候选的算术/原生调用信息，未验证加速dispatch/真实梯度/总显存，未创建适配器、未准入新训练，不据参数少断言瓶颈。

- **2帧对照已完整结束，但导航结果为负**：训练4000步/384000决策全部完成，最终CPU验收通过；内部100条SR10%、SPL7.9170329%、nDTW30.5185664%、OSR40%。对旧best4k的21%/18.0169839%/36.5879701%，配对3赢14输。该权重不采用，不启动其完整1839条评测。导航效能FAIL不能因工程验收PASS而改写。
- 控制组权重：`sft_acceptance/ordinary_history8_paired_train_r1/control_recent2/run_001/checkpoint_000004000.pt`，SHA `1f380ab88d68a9aa1b946773a8856fee25681847c827d0085fee73240fdce376`。三rank最终fingerprint一致、4000次AdamW更新、预取线程join、根进程均exit0、无清理信号，三卡确已清空后才启动8帧组。训练启动墙钟5921.49秒，实际更新循环5723.33秒；旧V1四步失败成本另列。
- 控制组评测：`closed_loop_bench/ordinary_history2_dev_r1/run_001`，完整100条/23191动作/10913碰撞/67条STOP，评测墙钟2031.20秒。30条曾进入目标范围但未成功STOP、36条未到目标即停、24条预算耗尽且未到目标，成功10条。GPU1评测worker3804985已经正常退出/清理，勿按旧PID发信号。
- 独立第三次CPU核验`reviews/Q35N_HISTORY_PAIR_R1_POSTAUDIT/control_recent2_RESULT.json`已通过：旧/新共200条官方成功率/SPL重放、全部23191实际输入帧选择与原始RGB哈希、6048项源锁、相同物理协议、checkpoint绑定、评测参数不变与GPU清理。0新增仿真/训练；这是重复核验，不是新200条导航。失败解释见该目录`CONTROL_FAILURE_REVIEW_ZH.md`。
- **8帧训练仍在正常进行**：03:28为880/4000步，约32.74决策/秒、CPU RSS51.43GiB、每卡约17.4–18.1GiB显存。rank3804401/3804402/3804403，必须以其`PROCESSES.json`与实时/proc复核身份。第200步已被自动watcher验收通过（19200决策、28组FP32参数/矩有限），receipt为`reviews/Q35N_HISTORY_PAIR_R1_CHECKPOINTS/treatment_prefix8_FIRST200.json`。两组仍同初始化/样本顺序/优化器，8帧不是续训失败的2帧权重；不因已看到控制组负分而中途改8帧配方。
- 03:41实时增量：8帧已1220/4000步，累计34.95决策/秒，以上三个rank和自动watcher仍实际存活、无失败回执。第1000步CPU只读验收通过：`reviews/Q35N_HISTORY_PAIR_R1_CHECKPOINTS/treatment_prefix8_STAGE1000.json`，96000决策、28组FP32参数及AdamW矩有限；权重SHA `47c209abe3d35cc5c2b50417142e7ae9ab985b7d47e2f1f17df447e6314475fc`。查询2002/2048、动作嵌入1664/6144坐标发生BF16可见变化。该检查0 GPU/0新更新、不选中途权重；导航结果仍未知。
- 输入提示只读核实：原`ordinary_sync_recovery_v1/model.py`的`DecisionDataset.__getitem__`按实际`item['images']`逐张构造image项，再附真实指令，不包含固定“两张图”的提示文字。不存在本次怀疑的2图文字与8图数量冲突；这不证明模型已学会利用历史，也未改冻结提示/源码/输入锁。
- 自动watcher PID3770736经实时身份确认仍存活，已完成control绑定/评测并转为`WAIT_FIXED4000_TRAINING_ACCEPTANCE`等待8帧最终权重。之后仅一次8帧完整100条及事前配对比较；不重启任何已完成节点。GPU3/4/5占位须等本次双组lease最终关闭后统一恢复，当前不要手动恢复或打断8帧。
- 只读数值来源核对：新旧评测7个共同Triton缓存键的计时JSON均不同，按安装版实际min选择规则推得4个所选配置不同，见`CACHE_READONLY_COMPARISON.json`。这不证明其导致SR下降；未运行新的同checkpoint因果复现，不修改当前冻结训练/评测配置，不据此洗掉旧FAIL。原V12大范围特征一致性FAIL也未解除。
- 18766当前仍为`monitor_charts_history8_pair_r2`（PID3776448/start167337752/pane%294），网页实际显示控制组完整10%结果、8帧训练和自动接续等待。保留旧21%基座及历史完整1839条22.02%分栏；当前完整SR40未达成，尚无合格新候选或两组完整配对结论。

## 01:44增量记录（以下进度为历史快照）

- R1控制组实际1180/4000步，累计吞吐50.06决策/秒、CPU RSS50.86GiB，三个rank3757402/3757403/3757404经实时身份核验仍存活；无失败回执。8帧组仍等待控制组完成，不重复启动。
- 第1000步检查点已独立CPU检查通过：`reviews/Q35N_HISTORY_PAIR_R1_CHECKPOINTS/control_recent2_STAGE1000.json`，96000训练决策，28组参数与AdamW矩FP32有限；SHA `825ca5255521c43942a34a643856f9478da69aed2872b2833f54611ce499ec62`。仅数值检查，不选中途权重，不是SR改善。原first200封存检查器只读复用，R1训练源码和评测输入锁未改。
- 原18766已更新至`sft_acceptance/monitor_charts_history8_pair_r2`，部署PID3776448/start167337752/pane%294；只替换先前R1监控PID3756415，三训练rank、GPU6/7占位、评测watcher和旧18768/18769身份保持。HTTP/API/JS与无浏览器DOM契约测试通过，未做浏览器视觉检查。旧服务器/回执保留，不重跑deploy。
- 网页已显示两组训练吞吐/CPU资源/200步验收、自动接续阶段、实际评测模型存活PID、完整100条结果与配对筛选。测试明确99条即使被错误标为COMPLETE也不展示SR/完整审核；没有完整结果不绘制预期收益。自动流程阶段更新时间不冒称持续心跳。
- 自动评测watcher PID3770736仍经实时/proc身份确认存活，`WAIT_FIXED4000_TRAINING_ACCEPTANCE`，两组尚未绑定最终权重、0新导航episode。继续既定训练/等待，当前不需要新模块、改配方或重启；完整SR40未达成。

## 普通历史配对训练 R1：2026-09-14 01:26 CST快照与实施细节

- 当前正在实际训练，不是准备草案：`sft_acceptance/ordinary_history8_paired_train_r1`。2帧对照已完成500/4000更新、48000/384000普通决策；最近CE0.63646，累计吞吐39.62决策/秒。仅训练统计，不是SR正向结果。GPU3/4/5实际rank 3757402/3757403/3757404；精确身份见本分支`PROCESSES.json`，信号前必须实时复核。
- 已通过第200步CPU验收：`reviews/Q35N_HISTORY_PAIR_R1_CHECKPOINTS/control_recent2_FIRST200.json`，19200决策，全部28组参数/AdamW矩FP32有限，初始三rank一致，真实数据子进程0。action_query有1880/2048、exec_embed有544/6144坐标产生BF16可见变化；不以此宣称导航提升。固定最终4000步选择不变。
- 训练协议：同一best4k FP32桥、两分支同384000唯一普通决策与顺序/51 FIT屋，fresh AdamW重置矩、LR5e-5、warmup120、4000步cosine、micro4×8累积×3卡全局96；控制最近2图，对照前缀7历史+当前8图。无特殊/旧64纠错数据。每组5小时、每卡26GiB模型上限/28GiB总显存、总RSS64GiB、输出2GiB，不自动重试或延训。
- 运行方式：tmux `q35n_history8_pair_train_r1`，pane%362、lease PID3756971（启动记录，仅作定位）。控制组完整结束并CPU验收后自动训练8帧组，最终finally恢复GPU3/4/5原占位；GPU0外部任务、6/7占位及旧18768/18769监控不动。不要重复启动完成的生产队列。
- 原V1在4步/384训练决策后因`TOTAL_CPU_RSS_BUDGET`停止，200.99秒；旧FAIL/4步权重/回执保留，三卡曾正确恢复为3747280/3747372/3747456，随后才被R1精确借用。不是loss发散判据失败。旧最后成功采样为52,473,430,016字节，触发采样值漏落盘，不能编造超限幅度。
- R1只修数据运输：模型载入后fork两个DataLoader子进程改为每rank一个有界CPU预取线程（队列2批+在制1批），pinned memory绑定本rank；RSS仍按64GiB，不改成PSS或放宽预算。两组各100真实输入全部张量逐位一致，6项生命周期通过；旧源码/输入锁完整核验。启动后编译池仍每rank32 worker，RSS约50.44GiB，至500步未再次超限；此时各卡约13.1–13.8GiB显存。第1步与旧V1的分类矩阵完全一致，但CE/梯度范数并非逐位相同（另见首步比较回执），不宣称整个GPU训练轨迹精确复现。
- micro8对micro4的GPU等价检验R1保留FAIL：2帧梯度relative0.04743/cos0.998997且argmax不同；8帧relative0.06176/cos0.998211。未放宽门槛。另立固定micro4累积检验`reviews/Q35N_HISTORY8_FIXED4_ACCUMULATION_V1`通过：两组重复输出max_abs0，独立求和/实际累积梯度relative约4.4e-8/4.5e-8；192前向决策/32次反向/0更新、170.76秒、GPU1清理正常。原V1目录扫描ENOENT运输FAIL也保留。以上都不是SR收益。
- 固定100条评测已预冻结：`closed_loop_bench/ordinary_history_pair_eval_r1`及`ordinary_history2_dev_r1`/`ordinary_history8_dev_r1`。两组真实线上窗口共510输入、36组离线/在线张量比较、216次错误字段拒绝和500步历史索引检查通过；物理/官方指标源码不变。每组GPU1空闲卡、7200秒/最多50000动作，batch1；旧接口32前向另记。完整1839条需另定执行预算，本次100条不代表SR40达成。
- 自动衔接已实际挂起等待固定4000验收：tmux `q35n_history_pair_eval_r1`，workflow PID3770736/start167260476，实际身份与状态见该评测共享目录`WORKFLOW_PROCESS.json`/`WORKFLOW_STATUS.json`。01:26为`WAIT_FIXED4000_TRAINING_ACCEPTANCE`、尚无新导航episode。控制组可在GPU1评测，同时8帧训练继续；不等待/停止另一组。任何失败不自动重试，空闲卡不满足只等待不杀外部进程。不要重复启动watcher或已封存的freeze/test入口。
- 评测筛选在见结果前登记：8帧相对2帧SR严格升/SPL不降/nDTW下降≤0.01；另分别对旧best4k同标准比较，报告逐屋、配对赢输与5屋bootstrap。完整1839候选须胜过旧best；8帧还需胜过2帧。若都不通过，不自动对退化候选花完整基准预算，不把标准历史输入写成原创贡献。
- 原18766网页已更新至`sft_acceptance/monitor_charts_history8_pair_r1`，PID3756415/start167114618/pane%294，HTTP/API/JS与只读方法405检查通过（无浏览器视觉测试）。可看真实两组步数/损失曲线/存活rank/历史SR与40%目标；旧内存失败单列。当前API新增主状态是`paired_history`；旧`sr40_training`等字段仍是历史准备字段，不以其null判新训练未启动。网页还未接入新评测workflow的实时行。

## 2026-09-13 SR40基座重整 V1：历史准备记录（下列未启动/旧PID均非当前实时状态）

- 23:42 CST：已落实标准全前缀8帧普通输入，路径`sft_acceptance/ordinary_prefix_history8_v1`。通过alphaXiv读取NaVILA实际采样函数，并独立固定commit `76b98f233dd0fff05dfcd69435eec6740febff9d`，官方代码和Apache-2.0许可只读保存在references（不安装/运行其项目、不下载权重数据）。7历史分位图+当前图，不足左黑图；本地黑图224而非原448，明确为采样借鉴、非完整NaVILA复现/原创。
- CPU测试通过10000时间边界/无未来选择、三类普通真实轨迹、18组真实离线/在线/推理上下文逐张量一致。Python无pidfd扩展的CPU V1失败保留，`owned_process_r1.py`使用x86_64 Linux pidfd syscall兼容；7类真实子进程退出/僵尸/竞态/身份拒绝/TERM后KILL测试通过。不要重跑已写独占回执的测试入口。
- `reviews/Q35N_PREFIX_HISTORY8_INTERFACE_V1`实际完成GPU1单卡168.26秒、144决策前向/8次反向/0更新/0仿真；原实际评测模型前缀32输入max_abs0，8帧前向反向有限、全部28组trainable最终未变。exit0、无清理信号、GPU1为空。`POSTAUDIT.json`已核5946冻结文件与输出/资源回执；只是接口通过，不是SR提升，也不解除V12全量5487一致性FAIL。
- 固定微批4的短测：2图28.58样本/秒、峰值7.19GiB；8图15.92样本/秒、11.43GiB。仅3个暖态微批的模型前向反向，不含IO/通信/AdamW，不能当正式吞吐保证。累计仅上述一个新GPU诊断，0新策略checkpoint。
- 已准备`sft_acceptance/ordinary_history8_paired_train_v1`普通配对数据：两分支同384000去重决策、4000更新、全局batch96/rank32，覆盖51 FIT屋/37024指令；RxR125051、R2R76939、EnvDrop182010，无特殊或64条纠错数据。顺序/原target/weight绑定不改；`CPU_LOSS_AND_ORDER_RESULT.json`的跨rank微批4/8/16/32梯度加权代数通过（误差<=1.4e-16），不是实际GPU累积数值验收。runtime_allowed=false，microbatch/accumulation/LR/矩恢复/墙钟预算仍待冻结；训练未启动，下一步实际微批8容量/累积检查、最终优化器协议与训练入口，不从草案直接占卡。
- 已登记新的阶段A目标；优先复用现有2650347普通动作，核对完整标准导航配方和共同执行入口。保留Qwen3.5-2B，不自动换主干/下载付费资产/扩大环境；先标准基座，再原创，不把标准8帧/门控/更多LoRA当论文创新。
- 原STOP行V12支线目前0实际拟合/0新增checkpoint，不自动接续；原5487特征不一致FAIL不改。缓存差异诊断已否定其足以解释偏差。实际评测前缀32输入预热前后max_abs0、hook0、PNG/Window实际编码逐张量相同，只支持原入口的局部可复现，不是全量修复或导航收益。
- 原前缀诊断launcher在子进程退出观察时发生OWNED_IDENTITY_CHANGED并漏最终回执；原FAIL保留，不补造退出码或重跑。新SR40只读审核已确认原PID不存在/GPU1为空；退出/僵尸/身份复用测试及新GPU诊断正常清理现已完成，见上方H8节点，不需要重跑旧诊断。
- `Q35N_SR40_BASELINE_RESET_V1/RESULT.json`是最初只读路线审核快照（该节点0新GPU启动/0更新/0仿真），不是后续H8节点的累计资源账。NaVILA现已另固定commit/许可/实际采样代码，StreamVLN仅已核训练脚本，均未完整复现。普通40%阶段不混入特殊数据。
- 原18766网页已更新为`sft_acceptance/monitor_charts_sr40_v1`，三阶段目标、完整1839条历史成绩和内部100条分开展示，新训练/新完整评测均为null。CPU数据绑定/JS语法、18770预览、18766最终API与只读方法检查通过；无浏览器视觉测试。部署PID3696171/start166411234，仅替换原监控进程，3/4/5/6/7占位及18768/18769监控身份未变，旧源码保留。部署回执为该目录`DEPLOY_RESULT.json`，不要重跑部署脚本。

## 已收口
- V6实际偏离纠错：新合法建议5365，排除382条旧输入重合（含54条标签冲突）后4983种新增输入；固定1000更新/99047决策。原V6在950步墙钟截止，R1原样补50步，无重复前缀。最终100条结果SR19%、SPL17.5273835%、nDTW39.3378687%，11赢/13输，负向，不采用。
- V6最终权重：`sft_acceptance/ordinary_onpolicy_adapt_v6r1/formal/attempt_001/checkpoint_000001000.pt`，SHA d5fd559cb78038c806ca0c13bf97241f7bd4bd3d7326b7bd02779916fce8dc0a。评测与审计闭合目录`closed_loop_bench/ordinary_onpolicy_adapt_dev_v6r3`；主审`reviews/Q35N_ORDINARY_ONPOLICY_ADAPT_V6_EVAL_R3`。R1旧索引绑定、R2漏CPU回执的零动作失败保留；R3全部20853动作审核通过。
- 单一等权参数合并V7R1：仅0.5、无训练。FIT64条SR28.125%、SPL25.1810195%、nDTW50.45929495%，7赢/5输。虽平均提高，但丢5条>事前上限2，FIT门槛失败，不跑DEV、不试其他比例。主审`reviews/Q35N_ORDINARY_EQUAL_MERGE_V7R1`。V7的CPU全FP32假设失败保留，R1只修正FP32算术后回原dtype。此方法属于已知工程，不是UAD创新。

## 新只读数值证据
`reviews/Q35N_BF16_ACTION_PARAMETER_READONLY_V1/RESULT.json`：
原模型24个LoRA因子+动作头2组为FP32，动作查询和执行动作嵌入为BF16，且各自AdamW矩也是BF16。V6 1000更新后，动作查询2048坐标变137个；执行动作嵌入6144坐标只变3个，最后50步嵌入完全不变。
冻结5000步已存矩做静态理想更新/表示舍入（不是实际梯度重放）：BF16中动作查询2024/2048、执行嵌入6144/6144个非零微小更新仍舍入回原值；FP32分别2/2048、237/6144。0 GPU、0模型/优化器修改。支持数值吸收风险，导航因果影响仍未测。

## 旧V12 STOP行前置探针（主线已被用户SR40目标替代，以下为过程记录）

- V11最终闭合负向：完整100条SR13%、SPL10.63728899%、nDTW27.91484633%、OSR52%，5赢13输，27393动作。`reviews/Q35N_ORDINARY_ROUTE_TEACHER_V11_POSTAUDIT/FINAL_RESULT.json`已PASS，543个冻结文件、官方指标CPU重放、清理/监控核对完成。高OSR不能冒充正向导航，最佳仍4k21%。V11训练/评测进程均结束，GPU3/4/5占位恢复为3607568/3607686/3607782（仅历史快照，发信号前须实时核验）。
- 新`sft_acceptance/ordinary_stop_row_v12/PLAN_ZH.md`在特征提取前冻结：原最佳4k，全部主干/LoRA/query/exec/三行运动头不变，只拟合STOP行2049参数；普通线性监督工程，不是UAD贡献。最近2图/8动作和原四类argmax不变，没有全局偏置/系数扫描。
- 原FIT64的7225次决策重建5487种完整因果输入，818种当前距离<3m，0二值冲突。5434PNG全解码hash通过。种子1209固定12屋拟合/4屋内层留出；标签/距离/房屋严格分离。固定带L2锚定的CPU logistic残差，先留出precision/recall不降、F1与BCE改善，才全部FIT一次拟合及完整FIT64闭环；再过门槛才一次同100条DEV。任何FIT/探针正号不是目标达成。
- 仅GPU1空闲卡冻结特征提取、最多6000前向/1200秒/25GiB；CPU固定拟合，无GPU训练、不借其他占位。CPU真实输入/像素/分组/权重、合成线性拟合及折叠/运动行精确不变测试PASS。build初版仅上游状态名断言失败，0GPU/0输出，保留并以build_r1精确状态修正。
- V12首提取因启动环境漏CUBLAS_WORKSPACE_CONFIG在首前向退出，0完成前向；原失败保留。TRANSPORT_R1补与原评测相同的`:4096:8`，另日志/回执，已完成5487前向（399.78s，不含加载），参数不变、GPU1清理通过。FEATURES SHA c367a1c83f7cc214f3849c4f9d891719e9ae0340a73495e1b135be1d69436eb8。
- 但FEATURE_PARITY为FAIL：对原FIT日志max_abs0.762128，80/5487动作不同，禁止继续拟合，没有新STOP权重。只读看到旧FIT与本次6个共同autotune键有4个选用配置不同。新`reviews/Q35N_STOP_FEATURE_CACHE_DIAGNOSTIC_V1`只复制旧FIT的7个autotune配置到自己的新缓存、固定32输入验证，300秒空闲GPU1、0训练/仿真，原输入和提取代码不变；尚不提前归因或放宽一致性门槛。
- 原18766服务现为`monitor_charts_stop_row_v12r1`，部署PID3656439/start166003167，API/只读方法检查PASS（无浏览器视觉测试）。显示V12提取/探针与历史完整导航结果，保留最初运输失败；当前停止在一致性失败待诊断阶段。旧3/4/5/6/7占位及18768/18769身份未变。

## 已关闭V11的过程记录（以下启动/在途为历史快照）

- V10已完整关闭不采用：SR19%、SPL16.8992752%、nDTW34.9375654%，9赢/11输；20,468动作/实际输入帧全审计，333文件锁复核、官方指标重放、GPU1清理与监控同步通过。不是新正向结果。
- 当前`data_pipeline/ordinary_route_teacher_v11`：同原最佳4k的64FIT轨迹/7225状态；新标签教师只按有序reference_path、0.35m geodesic reach推进，再去最终goal。不用V10时间窗、不改模型、不训练，旧labels/pool/生产/输入锁保持不变。
- 前置只读诊断`reviews/Q35N_TEACHER_ROUTE_ALIGNMENT_READONLY_V1`：4983输入中813种STOP，695种未满足原普通示范有序终点必要条件；这只是监督目标不同，不能判原label错误。官方VLN-CE ShortestPathSensor确实追最终goal，新版本不是论文创新/官方复现。
- 18项CPU检查通过、协议/源码冻结；GPU1实时XML空闲后启动tmux `q35n_route_teacher_data_v11`。20分钟、32000显式原语、8GiB GPU，一次同64条双重回放和每条建议真实一步验证；新路点进度也在两次回放分别测量。数据门槛未通过不训练；通过后再冻结新旧ID/标签/普通源冲突对账和训练计划，不声称数据顺序仍完全相同。
- V11数据已正常完成并独立审核PASS：5371合法去重建议/16FIT屋，116未知组隔离；5434PNG全解码核验，7225状态/14450回放帧及14450路点进度行通过。显式原语21523、route geodesic查询22029、教师查询7193、墙钟717.53s、峰值GPU1571MiB，GPU1已清理。不是新增独立路线或模型收益，旧目标教师数据完整保留。
- V11训练`sft_acceptance/ordinary_route_teacher_v11`，主审`reviews/Q35N_ORDINARY_ROUTE_TEACHER_V11`，评测`closed_loop_bench/ordinary_route_teacher_dev_v11`。预先固定交集策略：4951种输入（实际抽4904种），1642种标签变化，752种inflection权重按原3.2/1规则变化；原89172普通读取完全保留，纠错9807次，删除68次新教师不可认证读取，总98979次，1000更新。新index仅恢复部分重映射，三rank目标决策[33115,32982,32882]。
- 新初始化`initial_from_best4000_fp32_metadata_rebind.pt` SHA ba17db7ea899d76015d819776fc7bab8ba6d205a088bc5ad1c5eaad770813ffb，只重新绑定新索引；对原aa4e3... FP32桥的全部参数/矩/RNG逐项精确相同。原last2 RGB/last8动作，无V9 KL/V10时间间隔。CPU103实际样本、3000rank批次/普通index前缀字节、部署原Window测试通过。
- 已启动训练tmux `q35n_route_teacher_v11`，lease PID3590898/start165428590，review为`q35n_route_teacher_v11_review`；三rank MASTER_PRECISION_READY/RESUME_READY通过。固定1800s训练、2100slease，短TMP `.t11`双worker验收PASS，NFS清理警告如实保留。不要重复启动或按旧PID发信号。首200/最终1000验收与同100条最终评测自动接续，不自动下一轮。
- V11训练现已正常EPOCHS_COMPLETED，1000更新/98979决策，优化器5000，三卡占位恢复。首200为19763决策/[6623,6598,6542]验收PASS；最终权重SHA `07cd0381f2957513f87132df5f0ed2c0e8b6f1d630a8b5fb2bf311119768f4df`，CPU最终PASS。GPU1同100条评测已启动，launcher PID3608360（仅记录，不经实时核验不得发信号），目前结果未全，不判收益。
- 原18766已部署`monitor_charts_route_teacher_training_v11`，PID3592395/start165437447，HTTP/API/JS测试PASS（非浏览器视觉验收），正确指向新TRAIN/REVIEW/CASE，保留V10的19%和最佳4k21%。辅助审计`reviews/Q35N_ORDINARY_ROUTE_TEACHER_V11_POSTAUDIT/audit.py`已通过200条旧完整路线CPU官方指标回归；模型评测开始后运行pre，完整闭合后运行final。

## 已关闭V10的过程记录

- V9已完整关闭且不采用：SR17%、SPL15.4055309%、nDTW37.4481924%，3赢/7输，21,946动作审计通过。第三次CPU官方指标重放及326个源锁核对、GPU清理、监控同步全部通过。当前最佳仍21%；恢复到17%不是新正向结果。
- V10训练目录`sft_acceptance/ordinary_action_aligned_history_v10`，主审`reviews/Q35N_ORDINARY_ACTION_ALIGNED_HISTORY_V10`，评测`closed_loop_bench/ordinary_action_aligned_history_dev_v10`。仅把两帧输入从[t−1,t]改为[max(0,t−8),t]，八步已执行动作窗口不变；仍最多两张编码图，缓存最多九张过去RGB，不是持久记忆或UAD创新。
- 同V8R1初始最佳4k FP32 master桥、矩/RNG、V6固定1000更新/99047次决策/顺序/标签/权重/LR，无V9 KL。普通89172次、纠错9875次，原4983纠错来源ID仅重建视图，不增加独立数据。
- CPU完整固定计划检查零新视图目标冲突，5294张恢复像素核验通过；501个选择边界、121步真实在线传输、68个实际训练样本、旧100条分析精确回归通过。28个实际聚合语句测试能拒绝错误帧哈希/索引/stride。首版在线切片与测试属性错误均保留旧文件，仅未启动前版本化修复。
- 训练墙钟1800s、lease2100s，GPU3/4/5精确占位借还；项目短TMP `.t10`双worker64行传输通过，NFS退出警告保留，零遗留socket。固定最终权重后仅GPU1空闲卡同100条评测，不选中间checkpoint，不扫时间跨度。实时是否启动以本节点lease/workflow回执为准，不从本段草案推断运行。
- 原网页18766现已部署`monitor_charts_action_aligned_history_v10r1`，DEPLOY_RESULT为PASS，PID3550921/start165006651（仅记录，信号前须实时核实）。V10监控首版仅评测进度路径误指已闭合V9，R1已修复并补目录/时间戳检查；实际训练/评测从未用错路径，也未重跑。旧源码/部署回执保留，精确评测launcher身份未变。物理协议、官方指标和原输入锁不改；实际送入模型的RGB哈希逐动作独立核对。原门槛不变。
- V10已启动：训练lease PID3529252/start164867531；tmux `q35n_action_aligned_history_v10` 和 `q35n_action_aligned_history_v10_review`。三rank初始精度/恢复检查通过，18:50首20更新/1987决策已记录。不要重复启动；按实时PROGRESS和workflow继续监控。
- 19:04 CST更新：V10训练正常EPOCHS_COMPLETED，完整1000/99047、原优化器5000，GPU3/4/5占位已恢复；最终CPU验收PASS，权重SHA `4b75be8e4a74edbb188482997f152a85324df014c8a49946581c3bf7c04a3c36`。固定100条GPU1评测已启动，launcher PID3545724（仅历史记录，信号前必须实时核实），模型加载完成，无新导航结论。辅助审计`reviews/Q35N_ORDINARY_ACTION_ALIGNED_HISTORY_V10_POSTAUDIT/audit.py`已通过200条旧完整路线CPU官方指标回归；评测完整后执行final核对指标/实际输入帧/冻结源/清理/监控。

## 已关闭V9的过程记录（以下为历史快照）

- V8R1完整100条结果失败：SR15%、SPL13.4293234%、nDTW31.4371217%、OSR37%；8赢/14输，20,158动作审计及第三次CPU官方指标重放通过。数值修复有效但不能宣称导航改善，不采用。原21%最佳4k保留。
- 当前训练`sft_acceptance/ordinary_policy_preservation_v9`，主审`reviews/Q35N_ORDINARY_POLICY_PRESERVATION_V9`，最终评测预冻结`closed_loop_bench/ordinary_policy_preservation_dev_v9`。只加KL(固定最佳4k || student)，lambda1/T1；其余与V8R1相同（同初始化FP32 master桥、同矩/RNG、同数据/顺序/1000更新/LR）。不接着训练V8失败最终权重。
- 依据：V6及V8R1均失去较多旧成功；FIT去重诊断中1152种输入来自原本成功的路线，其中760种得到与原行为不同的几何教师动作。不等于它们标签错误，但表明当前监督没有保护旧成功行为。V1诊断误算重复原始出现6721而非4983独立输入的断言失败保留，R1去重并核实重复目标/原动作/结果一致后完成。
- 已知输出蒸馏/防遗忘工程，参考Learning without Forgetting（https://arxiv.org/abs/1606.09282），不是UAD架构创新。只测试这一系数，不扫参。原门槛仍相对最佳4k：ΔSR>0、ΔSPL≥0、ΔnDTW≥−0.01；改善V8的15%本身不算成功。
- 15项CPU测试通过：KL及梯度、三rank全局加权归一化、固定参考无梯度/无RNG消耗、学生参数和对象身份/优化器状态保持、异常finally恢复。真实三rank首批REFERENCE_READY已全部通过，teacher和student初始logits严格相同（max_abs=0、relative_L2=0、KL=0、argmax全同）。首200/最终1000仍须实际验收，不按中间分数选权重。
- 本段99047优化决策，额外99047教师无梯度前向，总198094次policy-forward决策；不声称同算力。训练墙钟2400s、lease2700s；短TMP为本项目`.t9`，实际socket和64行双worker传输通过，NFS退出清理警告已记录。无新独立数据、特殊训练、生产重启或付费API。
- tmux `q35n_policy_preservation_v9`及`q35n_policy_preservation_v9_review`已启动，lease PID3414201/start164234616（仅启动记录，必须实时再核身份），不得重复启动。自动流程仅同一固定1000步最终评测，不自动下一轮。
- 17:23 CST更新：V9已EPOCHS_COMPLETED，1000更新、99047学生决策、99047额外teacher前向，总198094，三张卡占位已恢复。最终权重SHA `d92f32afbb68522db95c03c9cd6230be6446f8e4b21daaf2b0784a72acd86ac4`，CPU验收通过（原优化器5000，最新加权KL0.0161321）。同100条GPU1闭环评测已启动，launcher PID3441080，仅记录不得不经核验发信号。首批模型已加载；未知导航分数不填零。流程循环状态沿用`EVALUATING_FIXED_CORRECTION_1000`历史标签，实际训练、checkpoint、输出与URL均为V9，不能据标签误认在重跑V6。
- 第三次离线核对脚本`reviews/Q35N_ORDINARY_POLICY_PRESERVATION_V9_POSTAUDIT/audit.py`，200条旧完整路线CPU官方指标重放单测通过；待V9 workflow COMPLETE再用项目Habitat解释器、CUDA_VISIBLE_DEVICES=''运行`final`，核对全结果、冻结源、完整配对门槛、资源清理及原端口接口。不要修改活跃冻结脚本。

## 已关闭的V8R1过程记录（下列22/100为历史快照，最终失败见上）
单因素FP32 master数值修复V8R1，从最佳4k开始复用V6完整同1000步混合计划、相同学习率5e-6/原余弦时钟4000→5000。只将action_query、exec_embed.weight及其AdamW矩转成FP32保存和累积，前向仍显式cast到原BF16视觉语言嵌入，初始有效输入逐rank严格一致。不使用V7合并权重，不变数据/索引/顺序/损失/模型结构/初始数值，不把重跑控制配方算新增独立数据。原代码及FAIL不改。

- 原V8在三个rank通过数值初始检查后，首批DataLoader因AF_UNIX路径过长失败，无完成批次/已观察更新；已按精确lease身份关闭并恢复占位，失败保留。
- V8R1只将临时目录缩短到本项目`.t8r1`并更换输出/编译缓存命名空间，科学源码归一化后与V8完全一致。实际UNIX socket和双worker的64行张量传输测试通过；退出NFS临时目录清理警告如实留档，复查无存活worker/socket，仅两个空测试目录。
- 训练目录`sft_acceptance/ordinary_onpolicy_fp32_master_v8r1`，主审`reviews/Q35N_ORDINARY_FP32_MASTER_V8_TRANSPORT_R1`；训练lease PID3016509/start163836976（仅历史启动身份，不用于未经实时核验的信号）。tmux `q35n_fp32_master_v8r1`及`q35n_fp32_master_v8r1_review`，不得重复启动。
- 首200步已验收：19,780次决策、优化器4200步，28组参数和矩均FP32/有限。查询实际BF16可见变化1829/2048坐标，执行动作嵌入274/6144坐标；不是只统计不可见的FP32低位。此为数值修复有效证据，不是导航收益。
- 固定最终1000步后自动验收同99,047决策、资源恢复，再仅一次GPU1固定100条INTERNAL_DEV评测及逐动作审计：`closed_loop_bench/ordinary_fp32_master_dev_v8r1`。门槛仍相对最佳4k：ΔSR>0、ΔSPL≥0、ΔnDTW≥−0.01，不挑中间权重、不自动延训。
- 现已正常结束1000步，最终权重SHA `611447849348bc4ad7be3d93eaeba9ed7011bf63082b64d7aaea08f6d05f8f94`。28组FP32权重/矩均有限，最终查询1970/2048、执行嵌入706/6144坐标产生有效BF16变化。训练占位恢复完成。最终评测进行中22/100条（仅当时进度，不是已完成结果），不报告部分SR。
- 另一次只读复核`reviews/Q35N_ORDINARY_FP32_MASTER_V8R1_POSTAUDIT_V1/PRE_RESULT.json`：301个冻结文件、约7.45GB逐文件SHA一致；24项物理/评测配置以及三个输入文件与最佳4k相同，0额外训练和仿真。
- 辅助聚合V1单测失败：误拿日志FP64诊断路径长度代入官方SPL（原SPL来自float32位置范数累积），相差约8.1e-9，旧脚本和FAILURE.json保留。R1改为原Habitat解释器、原官方指标类，对保存位置和距离做CPU重放，不改误差门槛、不查新几何、不运行仿真；此前最佳4k及V6共200完整路线复核通过。评测完成后执行`CUDA_VISIBLE_DEVICES='' .envs/q35n_habitat_v017_g0r/bin/python3 -I -B reviews/Q35N_ORDINARY_FP32_MASTER_V8R1_POSTAUDIT_R1/audit.py final`（命令路径相对本路线目录，cwd仍项目根且补全路径），核对逐路线聚合、完整轨迹审计、配对门槛和GPU清理。原运行评测不受此辅助测试影响。

## 资源和监控
原地址127.0.0.1:18766，当前服务`sft_acceptance/monitor_charts_history8_pair_r2`（以其DEPLOY_RESULT.json及实时/proc核实部署），真实主状态见API的`paired_history`（含workflow/evaluation）与本文件顶部。旧V1内存FAIL保留；R1两组训练/完整100条评测均结束，2帧SR10%、8帧SR12%，都不采用，完整基准候选null。HTTP/API/JS及浏览器替身DOM契约检查通过，无浏览器视觉验收。
GPU1本次8帧评测已正常结束，06:17只读核实为空闲；旧V11、V12及micro4节点亦已结束，V12不自动重跑/拟合。GPU3/4/5已由R1 lease恢复占位，身份见05:35过程记录及实时/proc，不得按历史PID直接发送信号。GPU6 PID3996270、GPU7 PID112240及旧18769/18768监控不动。GPU0外部任务可能自然更换，仅只读识别，不接触其环境、文件或进程。
系统盘/和/tmp满，但项目NAS空间充足；不要用需要/tmp临时文件的大heredoc，诊断用python -c或项目内脚本。不要清理系统盘、用户磁盘占位或其他任务文件。
