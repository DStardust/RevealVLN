# 本轮明确人工恢复：当前两个失败生产lane

用户2026-09-10要求“有用的信息做成图表，然后把失败的任务重新挂上”。本节点只恢复当前GPU1/2失败生产任务，不重跑历史SFT/科学失败；不停止GPU2普通训练或其他健康生产，不动GPU0及外部任务。

顺序映射：225→500、244→501、226→502、245→503、227→504、246→505、228→506、247→507。全部在已实读空闲GPU1串行，一个flock和单生产者；本次新有限授权12小时、8批24候选尝试，每批沿用3900秒监督/3600秒factory/60000动作/7GiB/1200秒强审；最多64GiB总工程预算，尾批无足额5400秒不启动。

225、244只限已证明EXTERNAL_RESOURCE_LOAD的失败，分别验证cleanup或无worker/无supervisor。不自动从失败receipt推断权限，本次明确用户命令与MAIN_AUTHORIZATION绑定它们的一次新尝试；其余六批必须仍只有两份准备文件，并由冻结runtime再次核从未启动。旧失败/partial不覆盖、不剪贴到新run、不改PASS；新尝试仍可能质量不通过。

读取冻结prepare/transport/audit，只把prepare模块的authorizations/jobs输出重定向本新目录。生成的runtime wrapper仍直接导入旧封存transport.py；新audit wrapper仍调用原严格审核函数。新QUEUE明确manual_retry_of及failure_receipt，SOURCE_LOCK绑定旧配置、来源锁、失败结果及本节点脚本/授权。运行都是全新batch目录和ledger；没有新增.py到封存special_scale_transport_v1。此为显式一次人工恢复，不是自动失败重试，也不伪称旧候选从未尝试。

18格/27重放/54求值、原FIT划分、原物理/语义/M2门槛、原显存保守计量和原idle/final guard均不改。已正式强审过的batch不重试；重复候选重试不能计为新的独立候选或旧cohort替换。新接受结果按原source/candidate/physical identity去重计入，旧未强审partial仍单独保留。

CPU检查、逐batch INPUT_LOCK/实际worker解释器只读准入通过，图表服务上线并验证后，主agent才用原AUTO queue.py挂唯一新lane。启动不等于审核合格；实际producer、cleanup、strongaudit及自动后继另核。

V2准备修订：V1在CPU precheck读取244的error字段而非真实exception.message，保护性拒绝，未生成QUEUE或启动GPU。V1代码封存保留；V2精确核status/exception.type/message与零外部signal，新增实际receipt回归测试。任务/预算/阈值不变。
