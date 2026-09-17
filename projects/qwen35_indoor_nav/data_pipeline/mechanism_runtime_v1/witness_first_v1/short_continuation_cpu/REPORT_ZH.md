# 保留真实双帧见证的短续接 CPU 提案

目的：保留 `revisit_v1` 已通过真实回放的 240 动作历史，不修改 checker/query 限额，将过长的 A 续接进行有界几何压缩后交给主 agent 新版物理回放。

原真实 `000007.json` 中 C_A 查询由原 Compiler 从后缀复算为 **210 个 sequence 项**，超过原有 160 上限。不是 210 个文本 token；不改编码器或上限。

来源为真实 `000001.json` 的前 116 个动作，即原 a 组件。该 trace 为 a+i，共124动作，完整无碰撞。对每个 A 两帧见证保护原观察 t-1、t 的连接动作：分别调用原 `factory.compress` 压缩其前缀和后缀中的转向，不跨保护边合并。仅按动作数、见证像素量、原帧索引排序后保留最多3个不同动作序列；这是 FIT 数据构造选择，不是未见评估。

| 候选 | A loop动作 | C_A含13动作terminal和STOP | 原保护帧 → 新目标帧 |
|---|---:|---:|---|
| WF_SHORT_A_00 | 34 | 48 | 16/17 → 8/9 |
| WF_SHORT_A_01 | 36 | 50 | 112/113 → 32/33 |
| WF_SHORT_A_02 | 44 | 58 | 32/33 → 24/25 |

首选原保护帧实例5各至少302像素。JSON保留两帧原真实 agent/sensor pose、RGB/semantic哈希、实例和像素门槛，可在实际新重放时对照。CPU 只证明理想运动学起点/保护边/终点等价，不证明浮点 pose、观测、其他事件或可达性相等；不能把旧观测复制给新动作当新数据。没有预言新 query长度，实际值保留 null。

接入方式：新 Factory 用独立 `continuation_a` 仅覆盖 C_A，C0、C_B及原 `components.a/b/i` 完全保持；特别不要将短 a 写回 histories 使用的 `components.a`。实际重新执行 C_A，核验保护帧可观察证据、零碰撞、原汇合界限、原 checker 和 query≤160，再完整18格/三种子重放及原质量/因果门槛。不减阈、不换标签、不以CPU检查代替物理认证。

9项CPU测试通过：保护边几何、原输入不变、真实候选长度/去重、有界与确定性、未知证据/碰撞拒绝、无目标证据不造候选。源trace/config与原factory/compiler/revisit/assembly源码SHA256已写入 `CANDIDATES.json`。无GPU/仿真/训练；只修改本目录。
