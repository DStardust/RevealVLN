# 最终100000步记忆模型的普通导航迁移

固定EXPANDED/MONOTONIC/seed1209的STEP_100000/HEAD.pt，SHA256为7f000bbaa79fd2e437d9f021d4ffde6c6872404b02fecc474c73a4e2c318d161。它是轻量记忆策略100000更新，底模仍是冻结best4k，并非普通导航基座训练100000步。

A=V13，B=原best4k logits加原100k记忆策略。adapter逐观测调用原EvidencePolicy.step，episode重置，实际执行动作回写原窗口；没有更换任务指令、状态真值、未来输入、原生STOP保护或额外STOP观测。训练0步。最终argmax和500决策/3m成功定义继承普通评测。

完整公开val_unseen1839，同对同模型进程；GPU2–7按冻结顺序分片。不是20个checkpoint竞赛，只评估事前指定最终100000步。每对封存、故障留痕、独立systemd运行与断点恢复。最多48GPU会话小时、160GiB产物；仅清理自身核验进程组，占位借用通过既有控制服务lease恢复。

EXPOSURE_AUDIT登记：6个房屋/873条与记忆训练重合，2个房屋/564条只有记忆开发暴露，其余3个房屋/402条在所审计记忆/普通头训练及记忆开发清单中不重合。全部1839已公开暴露，剩余子集也不称盲测。主表完整分母并列三种暴露层，结果不能冒充严格训练未见泛化。

运行：项目Python -I -S -B standalone.py start JOB -- 项目Python -I -S -B pipeline.py --run-id transfer_001。恢复相同run-id加--resume、新服务名。监控沿用18770，由新V20监控统一显示。
