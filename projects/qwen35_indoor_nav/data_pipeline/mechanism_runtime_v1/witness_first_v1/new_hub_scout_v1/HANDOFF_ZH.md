# 四个新位置的有界采集

仅新位置覆盖：同一 FIT 房屋8WUmhLawc2A，既有官方R2R-CE train来源82个完整3D坐标。根据已记录几何可达组数量排序，排除距离既有或已声明hub不足1米的位置，再取四个相互间距至少1米的位置。全部未选/拒绝记录在HUB_SELECTION_LEDGER，官方episode/path索引和原始起点旋转在POSE_SOURCE_PROVENANCE。

原scout只将候选hub上限2改4，运行时原navmesh/目标可达查询重新执行；仍每hub最多12语义组、每组2目标、140去程/504紧凑轨迹门槛，原像素事件、闭合、完整trace不变。所有三维位置和实际agent/sensor pose由原runner保存，CPU canonical reset pose不充当观测。

PREPARED_CONFIG保持runtime/executable/training false。主agent审阅后按runtime.approval_value()冻结MAIN_AGENT_APPROVAL.json，再由主agent独立启动runtime.py。这里没有创建批准文件、运行目录或操作GPU。GPU2只许原闲置资源护栏，不借用、不停止外部任务。

总40000动作/2700秒，单屋discovery2400秒，supervisor3000秒；内容6GiB/磁盘7GiB/RAM8GiB。旧样本线性估算9776动作、约14.4分钟、1.07GiB内容；不是预测保证，最坏动作数超过预算时保留censor。此批只检验扩展供给效率，不是泛化评估，不改变FIT分组、主线、24程序bank上限或27完整认证要求。
