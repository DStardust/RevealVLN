# 首个冻结数值汇合族：独立重放验收

继承data_pipeline/mechanism_family_v1/SPEC_ZH与NUMERICAL_JOIN_V1显式修订。
本验收草案此前未执行。EVENT_AWARE_SUFFIX_V1未产生候选，现按显式TASK_INSTANCE_V3修订，首次实际执行只接受TASK_INSTANCE_V3第一个且唯一frozen candidate，hash=1909aea983e255d10f2ff9641dbd6b6fffbbff6f21d13c81d7408d32faa2972e。开始后不搜索、不修改候选、不调阈值，失败保留并停止，不回退到其他candidate。

任务为g_T_v3/g_K_v3，历史H_T/H_K/H_T_I，续接C0/C_T/C_K；原餐椅任务不计为通过。记录结构继承V2，以SCHEMA_USED明确列出V3 task ID/revision/anchor字段。编译器的task角色来自冻结任务配置；查询分类编码仍采用全局固定词表，不随角色交换。

执行27完整物理回放（3 seeds×3 histories×3 continuations）、54独立任务求值，原始/规范化boundary证据、raw数组哈希和像素重算、共享u/s、查询/状态/因果/掩码/去重/18cell完整性验证。真实对照标签源为重新求值，不是纸面矩阵。

区别记录raw_exact_merge_pass=false与normalized merge结果。即使验收通过，scientific_pass=false、independent_statistical_samples=0、interface_only；不能宣称模型收益或泛化。

复用只读Habitat环境，仅GPU3，1h/8GiB磁盘/16GiB RAM/8GiB GPU，结束恢复占位。没有模型/SFT/机制训练。13项合成组件单测先通过，但不能替代本次真实验收。所有compiler源文件在执行前锁hash。
