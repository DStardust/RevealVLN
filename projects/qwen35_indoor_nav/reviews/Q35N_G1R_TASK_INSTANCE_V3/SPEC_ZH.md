# 同一机制的新任务实例：TV / sink / bed V3

## 明确的构造实例修订，不覆盖旧失败

原“餐厅任一椅子 / 水槽 / 床”实例的负续接C0在12000搜索展开、36000真实分支动作后仍未找到。不是数学不可行证明，但不应继续把特定对象组合当作整个机制。餐厅8把eligible椅子覆盖常用通道，导致原本应为中性续接的路线反复触发anchor。

主 agent批准在同一已选主线和同一interface_only场景中更换**任务实例**，不是更换学习算法：

- g_T_v3：先连续两帧看见客厅内的一台电视，再连续两帧看见卧室内的一张床，然后停止。
- g_K_v3：先连续两帧看见厨房内的水槽，再连续两帧看见卧室内的一张床，然后停止。
- H_T：TV且没有sink；H_K：sink且没有TV；H_T_I：TV后附加餐厅chair绕行、没有sink。chair只作为irrelevant control，不再作为必需anchor。
- C0：没有TV/sink而形成bed后立即STOP；C_T：TV后bed，不形成sink；C_K：sink后bed，不形成TV。

Eligible的mpcat40/raw/region过滤逐项沿用原4类，不挑单一椅子以缩小定义；改变的是哪个公开类别承担任务角色。两帧同实例256px、真实动作/零碰撞、预算512/160、8公共尾部、三值语义、程序顺序、数值共同状态证书和信息隔离不变。

任务/历史/续接ID与task_revision全部升为新实例，不能把V3标签写回g_D_v2或宣称原任务已通过。内部构造器的D/L代号分别映射TV/chair，输出用明确TV角色与V3 ID，映射记录在candidate/task config中。数据schema结构可继承V2，但新增专用schema核准新任务ID/anchor字段。

候选固定原u_index22、yaw0、public_tail LRLRLRLR，最多一个base/一个family；greedy路径+原真实inverse+中性padding先构造，不复用旧任务分类后的route cache。第一候选完成9真实trace与18格预检后冻结；随后独立27/54验收，不换候选。

选择依据是已暴露场景的几何与生成失败诊断；不是模型性能筛选，不作未见样本/泛化主张。此次若可构造，仅证明这一新任务实例的接口可行，原餐椅任务继续是失败/未解。后续生产应以公开任务角色/类别组合枚举报告筛选率，不只复制此一例。

预算1h、8GiB输出/16GiB RAM/8GiB GPU，仅GPU3借用恢复，不下载、不训练、不修改SFT会话。该任务实例修订不新增论文创新主张。
