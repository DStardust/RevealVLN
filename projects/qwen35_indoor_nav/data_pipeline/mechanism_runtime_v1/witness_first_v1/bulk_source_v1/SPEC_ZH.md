# Bulk scout 首片

按原43个FIT合法house的manifest首次出现顺序，排除已完成scout的6屋；不按候选
运行结果排名。保留37屋inventory，优先完整锁首3屋（5q7pvUzZiYa、759xd9YjKW5、
7y3sRwLe3Va，以实际manifest为准），独立shard_00，不修改旧未执行shard_1。
两片覆盖相同前三屋，旧shard_1不得同时运行造成重复。

每片3屋、每屋最多2个实际合法hub，使用原scout几何/反馈/闭环/bank方法和预算：
4200秒/60000动作，单屋1200秒/20000动作，监督4500秒，磁盘8GiB、content6GiB、
RAM8GiB。GPU1独立原资源护栏，不借占位、不操作外部任务。新源码适配均计数恰好1，
scopedstore、cache、journal、输出独立。配置初始不可执行；主agent复核后创建
MAIN_AGENT_APPROVAL_SHARD_00.json，精确内容为 runtime.approval_value(0)，然后
`run.py --shard 0`。首次GPU护栏失败会保留LAUNCH_RESULT，不自动重试旧目录。

只收真实组件与候选，物理族/训练族初始均0。新bank_recipe沿原完整闭合house过滤、
同计数方案与排序、1000 proposal界限，不改checker。draft source不能绕过主agent
运行准入。首片之后15屋/5片与剩余19屋只有库存，需新版本asset锁及各片运行准入。

最终目标依主agent正式规模V2，10k族不是本片产量承诺。两hub/屋×43屋最多86独立
hub；当前每hub只选一个族的recipe最多86族，无法据此声称达到万族。角色、seed、
措辞变体不增加独立hub。全部house相关分支按scene_group共同归FIT，不读val/test。
