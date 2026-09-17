# 已完整认证族的只读恢复验收

本目录不运行 GPU，不修改原运行、HEAD、存储或旧 validator。不是数据量目标或新导航效能实验。

## 等级与不可混淆的结论

- `recovery.py`：严格整目录内容复核，产生 `RECOVERED_FAMILY_CONTENT_VERIFIED_RUNTIME_CENSORED`。任何 partial 均使此等级拒绝。
- `family_scope_v2.py`：主 agent 明确批准的族依赖闭包范围；原 V1 拒绝完整保留。仅当残留项是可完整计量、命名明确的 partial，且其完整名称和目标像素哈希均未出现在已完成族全部导出、真实认证/发现轨迹及元数据中，才可排除该未提交残留对本族的影响。产生 `RECOVERED_FAMILY_REQUIRED_CONTENT_VERIFIED_RUNTIME_CENSORED`。
- 两者都保留 `quality_pass=false`、`original_batch_pass=false`、`training_admission=false`。恢复等级须由主 agent 单独决定能否进入 FIT 训练，不改写失败批次或宣称训练/泛化收益。

## 证据链

1. 核验原 INPUT_LOCK 的全部来源、实际 FIT split、动作前配置时间顺序、运行身份和清理终态。
2. 仅 HEAD 指定已提交字节前缀用于 hash chain/config/freeze/phase；其余尾字节全部保留、记录 SHA256，不用于标签和预算。
3. 要求族完整认证状态、已提交的 discovery 与 certification 完成预算；动作前已持久化的预算记录、不退款、动作确认数、每阶段及总动作/时间限制一致。
4. 复用原 Loader/Compiler，显式复制而非修改旧 validator 的语义条款：18 格、27 种子真实重放、54 评估、M2、精确 F/L/R、历史反转、任务差异、A 控制不变、当前窗口一致、policy 不访问未来 query、语义 NPY 像素重算。
5. 全 store 只读盘点每个目录项、NPY 文件及像素哈希，计入 partial 和未知文件字节；不伪造原 STORE_CLOSE、原内存计数或 fsync ACK。
6. V2 将训练依赖定义为完整导出包含的全历史/续接/规范化/M2/policy/query图；27 认证轨迹已独立证明动作与观测内容和规范导出完全一致。额外把所有本族发现轨迹也纳入 partial/目标哈希排除搜索，但不把未使用发现帧冒充训练样本。
7. 读后重新校验全部已读来源哈希；恢复产物另封存，不写回来源目录。

## 资源证据的真实边界

旧 supervisor 没保存触发失败的原始快照；恢复只能证明全部已持久化资源样本满足原阈值、可信本地文件时序中族已完成且自己的 renderer 最终退出。不能证明连续资源合规或精确跨时钟的逐族采样对齐。明确保留原 `EXTERNAL_RESOURCE_LOAD` / `MEMORY_ACCOUNTING` 等错误。

## 主 agent 接收最小条件

仅接收 `recovery_content_pass=true` 的上述独立等级，并保存原运行失败原因、HEAD/来源锁、族完成预算、18/27/54语义报告、required-content 封存、全 store 残留清单及资源限制声明。任何引用到 partial、缺少认证/预算闭合、源 hash 变化、非法 FIT 来源均拒绝。此接收只批准数据使用，不修改原批次通过率。

自然语言当前仍是受控模板，未验证自然语言迁移。CPU 合成测试只验证接口和拒绝规则，不能计入真实族数。
