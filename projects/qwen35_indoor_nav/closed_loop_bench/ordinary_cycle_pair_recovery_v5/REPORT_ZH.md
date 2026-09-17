ENGINEERING_COMPLETE

实际加载模型：True；完整有效pair：100/100；待补：0。
工程累计GPU小时：1.363966；pilot累计GPU小时：0.075927。

GPU小时按本任务launcher运行窗口计入预算，包含加载、编译与失败启动，不是GPU内核活跃时长。
5→20→100为同一个冻结实验的累计进度，对应MILESTONE_005/020/100.json；前段没有因分数被重选。
本轮使用V5行为配对协议，替代旧专用窗口、4100秒及一次启动限制；旧V3/V4 UNKNOWN不追溯改写。
实验完成、输入一致、动作一致、数值逐位一致、干预收益和是否采用分别记录。

|指标|原生A|恢复B|B−A|
|---|---:|---:|---:|
|sr|0.2|0.2|0.0|
|spl|0.17455585823083763|0.17455585823083763|0.0|
|ndtw|0.3664767497098997|0.33911954889253276|-0.027357200817366922|
|osr|0.48|0.49|0.010000000000000009|

完整100对未观察到SR收益，本轮不采用该恢复规则。重复决策减少不能解释为任务成功率提高。

配对胜episode：[]；负episode：[]；保留成功数：20.0。
输入前缀一致：True；动作前缀一致：True；
logits逐位一致：True；最大logit差：0.0；干预前argmax翻转：0。
运行身份及加载后/分段末全参数和持久buffer指纹在sessions；无覆盖pair检查完整轨迹与终态。
行为比较有效：True；完整100对效应可评估：True；
开发正向信号：False；采用：False。
SR效果、路径质量、重复动作和资源成本分别报告，不自动上线。

|成本|A|B|
|---|---:|---:|
|total_decisions|19305|17743|
|repeated_decisions|8589|700|
|overrides|0|693|
|collisions|7827|4963|

|屋|ΔSR|ΔSPL|ΔnDTW|
|---|---:|---:|---:|
|82sE5b5pLXE|0.0|0.0|-0.03665914974912704|
|B6ByNegPMKs|0.0|0.0|-0.02467484365885958|
|PuKPg4mmafe|0.0|0.0|-0.0025138276616392813|
|XcA2TqTSSAj|0.0|0.0|-0.15328425885438127|
|aayBHfsNo7d|0.0|0.0|-0.009396078713583833|

描述性配对区间：{'episode_bootstrap95': [0.0, 0.0], 'house_bootstrap95': [0.0, 0.0], 'caveat': 'Descriptive development intervals; correlated episodes and only five houses, no blind/generalization inference'}。
episode/屋内相关且只有5屋；INTERNAL_DEV已暴露，不能据此主张盲测泛化。

100对成功差值全为0，使经验重采样区间退化为[0,0]；这不意味着总体效应已被精确确定为0。

A STOP分型：{'far_stop_never_entered': 35, 'success': 20, 'left_range_then_stopped_far': 20, 'entered_range_budget_no_stop': 8, 'budget_never_entered': 17}。
A 时延秒（p50/p95）：{'inference_seconds': {'p50': 0.06450344598852098, 'p95': 0.06651113945990801}, 'preprocess_seconds': {'p50': 0.007040586089715362, 'p95': 0.008261020854115486}, 'controller_seconds': {'p50': 3.7550926208496094e-06, 'p95': 7.95791856944561e-06}}。
A 按资源竞争分开的推理时延：{'competition': {'n': 0, 'p50': None, 'p95': None}, 'no_foreign_observed': {'n': 19305, 'p50': 0.06450344598852098, 'p95': 0.06651113945990801}}。

B STOP分型：{'far_stop_never_entered': 41, 'success': 20, 'left_range_then_stopped_far': 21, 'entered_range_budget_no_stop': 8, 'budget_never_entered': 10}。
B 时延秒（p50/p95）：{'inference_seconds': {'p50': 0.06434203498065472, 'p95': 0.06641458778176457}, 'preprocess_seconds': {'p50': 0.006989816902205348, 'p95': 0.00829243848565966}, 'controller_seconds': {'p50': 0.00026497314684093, 'p95': 0.00030548356007784603}}。
B 按资源竞争分开的推理时延：{'competition': {'n': 0, 'p50': None, 'p95': None}, 'no_foreign_observed': {'n': 17743, 'p50': 0.06434203498065472, 'p95': 0.06641458778176457}}。

资源是共享使用；竞争按5秒监视快照分类，预处理耗时含审计哈希，不能宣传为独占或真机时延。
数值/服务问题：0，逐次原因及100对完整清单见RESULT.json。
半对只保留为尝试，不拼接单臂。已完成且有参数核验的pair可续用；没有按分数重试。

工程session数：1；基础设施失败尝试：0；本任务峰值显存MiB：9708；进程组峰值RSS GiB：20.850。

研究pilot：
保留失败尝试run_001：AssertionError('DEBUG_ASSET_CHANGED')；定位：Compiler categories/rooms tuples compared directly against JSON lists。
实际结果：DEBUG_FORWARD_BACKWARD_UPDATES_COMPLETE。真实冻结Qwen特征、249观察递归展开、关键事件写入梯度、动作读出和参数更新均已测。
关键H_B/task_B第14步写入到第248步监督的梯度范数：3.347207166370936e-05。
动作loss对运行memory梯度范数：0.051068585366010666；memory替换使动作logits改变的最大值：0.1196126937866211。
诊断更新1次；B2/Ours各100次，使用同一初始状态、因果特征、492动作owner及预算。

B2：初始{'action_ce': 1.9639860391616821, 'exact_state_bce': 0.6615203619003296, 'crossed_result_bce': 0.6892040967941284, 'action_accuracy': 0.5121951103210449, 'query_accuracy': 0.7222222089767456, 'action_owners': 492, 'state_derived_query_accuracy': 0.6666666865348816, 'effective_query_accuracy': 0.6666666865348816, 'query_cells': 18, 'state_known_steps': 1494}；结束{'action_ce': 1.783623218536377, 'exact_state_bce': 0.29622432589530945, 'crossed_result_bce': 0.7090928554534912, 'action_accuracy': 0.599593460559845, 'query_accuracy': 0.3333333432674408, 'action_owners': 492, 'state_derived_query_accuracy': 1.0, 'effective_query_accuracy': 1.0, 'query_cells': 18, 'state_known_steps': 1494}；训练秒45.15228571789339。
Ours：初始{'action_ce': 1.9639860391616821, 'exact_state_bce': 0.6615203619003296, 'crossed_result_bce': 0.6892040967941284, 'action_accuracy': 0.5121951103210449, 'query_accuracy': 0.7222222089767456, 'action_owners': 492, 'state_derived_query_accuracy': 0.6666666865348816, 'effective_query_accuracy': 0.7222222089767456, 'query_cells': 18, 'state_known_steps': 1494}；结束{'action_ce': 1.7735639810562134, 'exact_state_bce': 0.7252002954483032, 'crossed_result_bce': 0.3287454843521118, 'action_accuracy': 0.6056910157203674, 'query_accuracy': 0.8333333134651184, 'action_owners': 492, 'state_derived_query_accuracy': 0.6666666865348816, 'effective_query_accuracy': 0.8333333134651184, 'query_cells': 18, 'state_known_steps': 1494}；训练秒46.48453693394549。
结束时有效query拟合准确率：B2 100.00%，Ours 83.33%；本次没有Ours优于精确状态监督的证据。

上述数值是单个已暴露SEE2族的拟合/实现检查，没有独立族或记忆闭环评测。
B2的有效query指标由预测精确状态加合法续接事件经原任务逻辑计算，Ours用其训练读出器；不拿B2未训练的交叉读出头充当弱对照。
精确状态监督更密集，辅助反向成本也不同，实际调用与耗时已披露；这个小pilot不支持Ours优越性，也不裁定整个方向失败。
底层Qwen/视觉编码器未训练；原始证书的training_admission=false保持不变。
SEE2不冒充到访房间，既有规范化与负控制缺口保留。未来查询只进入训练读出器；科学主张仍UNTESTED。

工程CPU检查：{'passed': True, 'tests': 10, 'failures': 0, 'errors': 0, 'scope': 'V5 orchestration CPU contracts, not model/runtime acceptance'}；pilot CPU检查：{'passed': True, 'tests': 3, 'real_qwen_features': False, 'gpu_used': False, 'trainer_loss_contract': {'passed': True, 'synthetic_features_only': True, 'real_existing_action_owners': 492, 'action_memory_gradient_present': True, 'loss_finite': True}}。
pilot接口修复CPU回归：{'passed': True, 'tests': 1, 'seconds': 5.020049315877259, 'gpu_used': False, 'real_qwen_features': False, 'regression': 'Compiler tuple vocabulary roundtrips to identical persisted JSON; no label/feature change'}。
pilot/SAVED_EVIDENCE_AUDIT.json已从实际保存的参数文件复核两臂共同初始化、更新后的参数指纹、训练步数及FEATURES哈希。

已保存全量证据独立复核：{'status': 'SAVED_EVIDENCE_VERIFIED', 'complete_pairs': 100, 'complete_episodes': 200, 'decisions': 37048, 'prefix_decision_pairs': 10722, 'no_override_full_trace_pairs': 77, 'checkpoint_unchanged': True, 'frozen_sources_unchanged': True, 'native_and_recovery_recomputed': True, 'executed_history_and_rgb_checked': True, 'terminal_rgb_checked': True, 'log_hashes_checked': True, 'state_seals_required': True, 'cpu_only': True, 'new_model_forwards': 0, 'new_environment_decisions': 0}。
实际编译出的Triton配置见sessions/*/TRITON_CACHE_METADATA.json；它不是逐决策kernel选择轨迹，未把不可获得的选择信息补成已验证。

实际运行命令（vla根目录）：
```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/closed_loop_bench/ordinary_cycle_pair_recovery_v5/launch.py --gpu 1 --target 100
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/research/continuation_memory_v1/pilot/launch.py
```

入口、协议、CPU检查、逐pair结果及原生/实际动作日志均在本目录；pilot实现及真实更新记录在research/continuation_memory_v1/pilot。
审阅起点7f65fd04b54cea7f3844ec128a493f75501bbeb2；源文件身份见SOURCE_LOCK与每session/source，best4k和冻结模型未改。
ROOT现有.gitignore/AGENTS/README修改保留。CURRENT_STATUS更新V5事实，不覆盖旧报告、旧checkpoint或部署配置。
GitHub交付代码、配置、逐步JSON日志、参数指纹与汇总。原始RGB、编译缓存、FEATURES.pt及pilot的INITIAL/B2/Ours参数文件保留本地；参数文件路径和SHA见pilot/SAVED_EVIDENCE_AUDIT.json。复跑仍需当地原有模型、场景和数据资产。
下一判断应依据本轮完整效应及成本，并另定记忆独立族/闭环测量范围。没有SR40、真机部署或论文创新完成声明。
本轮交付后停止，不自动长训、扩LoRA、扩历史或安装方向B环境。
