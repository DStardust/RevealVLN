# 有限词表任务词面修订（仅CPU提案）

API：`verbalizer.realize_tasks(roles, tasks)` 返回新的任务字典，只改 instruction，anchor/terminal/task键保持不变；不改原对象、role category/room/raw、eligible ID、动作、Y或M2检查器。`propose_revision(candidate)` 给出 proposed_tasks、逐任务新旧文本、结构/文本SHA与必须重新物理重放标志。主agent只能在新版本配置冻结前应用，不能改已运行配置或复用旧语言哈希/旧policy refs冒充新版重放。

来源依据（本项目登记的官方MP3D标签实现，不自由推断英文别名）：

- `data_pipeline/mechanism_scale_v1/planner.py` 的 ROOMS、INDOOR_ROOMS、RAW 固定有限词表与现 task_spec。
- `data_pipeline/mechanism_runtime_v1/feedback_generation_v1/prepare.py` 已明确登记 `tv -> TV room`、`familyroom/lounge -> family room/lounge`。本修订复用这两个词面扩展。
- `data_pipeline/mechanism_factory_v2/compiler.py` 的 task_program/evaluate/m2 使用角色结构而非 instruction字符串；CPU测试核验修订前后的任务程序与pass/fail/unknown机制含义不变。完整物理重放仍须新做。

类别保留原精确raw子类（dining chair不泛化成chair、couch不替换成sofa）；仅电视raw `tv` 做缩写大小写规范为 `TV`。已自然可读的有限房型原样保留。`toilet` 房型不猜成bathroom，用明确 `room (room type: toilet)`，避免把房间误说成厕所物件；未登记房型默认reject，调用者显式选择unknown_room='explicit_type'才保留为同一类型说明模板，不推断语义。

当前batch_00/batch_01的EXECUTION_CONFIG只读审计，所有修订在PROPOSED_REVISIONS.json，executable=false。旧独立单族及其既有四任务不在本审计写入范围，未读取/修改其文案或成果。旧batch语言不完美并不抹除已完成物理证据，但对新版自然语言训练数据必须明确语言版本并重新导出/认证。

语义边界：此模板仍是 observable see-two-consecutive-frames-then-stop 受控机制任务，不将措辞改善当普通自然指令泛化或论文创新本身。未来更自然的措辞、paraphrase、任务分布变化需独立登记，不能混入本确定性修订。
