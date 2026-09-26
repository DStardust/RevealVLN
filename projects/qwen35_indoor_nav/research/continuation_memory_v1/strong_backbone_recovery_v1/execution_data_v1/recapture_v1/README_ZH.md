# 原轨迹全部动作token的真实特征补采集

用户2026-09-26授权开始补采。这里只采集，既有评测和模型权重保持不变。

独立服务：`q35n-strong-token-recapture-20260926-02.service`。进度页：[http://127.0.0.1:18770/](http://127.0.0.1:18770/)，API为`/api/observation_recollection`。初次提交时为WAIT_DEPENDENCY、0 GPU小时、0已采轨迹；模型未加载，不能把已排队写成已前向。

## 自动执行顺序

1. 等待已有`q35n-strong-history-validation-20260926-01.service`结束。它会在8张卡上自动补位任务，卡短暂空闲不等于已释放。只等待这个已知任务，最长24小时。
2. 核验实际资源，再用首张可用卡完成3条完整真实回放：FIT缺STOP轨迹174、DEV缺STOP轨迹302、FIT恢复轨迹0。三条计入334条总分母。
3. 首批接口通过后，使用已获准且资源足够的0–7卡分批采集全部334条；缺STOP轨迹优先。每批独立模型加载，16条一批，完整轨迹写证书，模型状态封存后可恢复计数。
4. 自动复核完整分母及缓存/轨迹SHA，输出RESULT、CAPTURE_INDEX和报告。采集最多16 GPU会话小时、12小时执行墙钟、20GiB产物；等待不计GPU时间。无按分数重试，不启动训练或导航效能评测。

全部目标：28,859个实际动作的actor特征，包含21,396个原缺位位置、133个原缺位STOP，7,463次原query。FIT/DEV划分保持。133个STOP中FIT102、DEV31；DEV不进入FIT训练。

## 正确性边界

- 使用原StreamVLN、BF16/flash_attention_2及原仿真、输入、四动作chunk与500步预算。
- 从原起点执行完整真实动作历史，逐步核对RGB；逐query核对处理后inputs/images/depths/poses/intrinsics/time_ids。
- 每个action token在teacher forcing之前捕获真实hidden feature及四类native logits。EOS和未执行token不产生训练记录。
- 同chunk全部token的记忆时刻固定为query_start。后续实际观察只能用于后验转移审计，不能提前进入该chunk。
- STOP计入动作预算，不额外生成观察。unknown恢复前缀仍unknown，补特征不自动把它们准入动作监督。
- 新缓存全部使用本次实际前向；旧首token缓存只做数值对照，不拼接不同运行栈。位差及argmax翻转单列，输入不同直接失败。
- 完整trajectory及状态封存才计完成。失败attempt保留；硬中断未完成的资源时长记未知并保守收费，不默认为零。只清理本轮Popen创建且PID/starttime匹配的进程组。

本轮不是160个替代动作分支的新采集，那份清单保持NOT_RUN。这轮首先补已有334条轨迹的特征覆盖，不增加独立物理路线数。

最终约束：**加入新数据后的模型必须在预先固定的val_unseen清单上做最终同分母比较；DEV仅诊断，不能代替最终结果。** 已暴露的unseen仍如实标注，不能改称盲测。

## 进度与恢复

```bash
systemctl status q35n-strong-token-recapture-20260926-02.service
cat runs/capture_001/STATUS.json
```

流水线由独立systemd服务执行，关闭聊天/终端不影响它。若因资源或正确性错误停止，先保留失败记录并定位；恢复使用新服务名，指向同一run并加`--resume`，源码/协议/数据绑定须保持一致。源码或数据语义修复使用新版本，不覆盖旧证据。

控制Python可通过现有`strong_backbone_recovery_v1/standalone.py`提交`pipeline.py --run PATH`；该pipeline自动调用项目GPU环境的worker。进度服务独立，不会控制训练/评测进程。

CPU验收：8项采集合同测试、4项恢复/进程身份测试通过；5个监控路由HTTP200。真实GPU先导尚待资源释放，成功时会产生`runs/capture_001/SMOKE_RESULT.json`；没有该文件就不能宣称模型已验证。

启动前审查补了一处恢复顺序：先拒绝仍存活的旧worker，再移动未封存目录。01服务仅在零GPU等待阶段关闭；02以同run恢复，旧任务记录保留。
