# EVENT_PREFLIGHT_ADVANCED

8/8 个留一 FIT 屋事件读出对照完成，3200 次真实更新；不是完整策略训练或闭环收益。

|项（屋与角色等权）|ORIGINAL|EVENT|差值|
|---|---:|---:|---:|
|brier|0.225544|0.202910|-0.022634|
|recall|0.600719|0.599704|-0.001016|
|fpr|0.385692|0.382976|-0.002716|

Brier 改善 4/4 屋。固定检出容限 −0.05；全部门槛见 EVENT_PREFLIGHT.json。
只训练原初始化的事件 MLP，编码器、归一化、动作策略与记忆均冻结。未访问 DEV 得分或新 TEST。
该诊断只检验监督分布修复，不宣称新模型架构、论文贡献或普通 VLN 收益。

下一阶段：自动训练六个完整策略并执行原 DEV 768 次续接。
GPU 使用与占位恢复记录见 RESOURCES.jsonl 和 PLACEHOLDER_RESTORATION_*.json。
