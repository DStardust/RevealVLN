# DIRECT / MONOTONIC / REVISE 匹配对照结果

VALID_COMPLETE_DEVELOPMENT_PILOT
完整配对组 128/128；续接 1152/1152。

|终点|方法|PASS/计划|未知|预算惩罚成本|
|---|---|---:|---:|---:|
|main|DIRECT|55/192|0|0.7619|
|main|MONOTONIC|66/192|0|0.7115|
|main|REVISE|57/192|0|0.7520|
|control|DIRECT|78/192|0|0.6391|
|control|MONOTONIC|98/192|0|0.5413|
|control|REVISE|98/192|0|0.5392|

|接管条件|历史|方法|主任务PASS/N|成本|
|---|---|---|---:|---:|
|terminal_present|all|DIRECT|33/96|0.6999|
|terminal_present|all|MONOTONIC|38/96|0.6532|
|terminal_present|all|REVISE|31/96|0.7010|
|terminal_present|seen|DIRECT|26/48|0.5098|
|terminal_present|seen|MONOTONIC|36/48|0.3297|
|terminal_present|seen|REVISE|31/48|0.4020|
|terminal_present|missing|DIRECT|7/48|0.8899|
|terminal_present|missing|MONOTONIC|2/48|0.9768|
|terminal_present|missing|REVISE|0/48|1.0000|
|terminal_absent|all|DIRECT|22/96|0.8240|
|terminal_absent|all|MONOTONIC|28/96|0.7698|
|terminal_absent|all|REVISE|26/96|0.8030|
|terminal_absent|seen|DIRECT|20/48|0.6752|
|terminal_absent|seen|MONOTONIC|22/48|0.6296|
|terminal_absent|seen|REVISE|18/48|0.7135|
|terminal_absent|missing|DIRECT|2/48|0.9729|
|terminal_absent|missing|MONOTONIC|6/48|0.9100|
|terminal_absent|missing|REVISE|8/48|0.8925|

Controlled stationary SEE2 tasks in an exposed DEV house. No ordinary VLN-CE or paper generalization claim.

DIAG_*.json 是缓存教师路径诊断，CALIBRATION.json 是自主轨迹上的只读校准统计，均不替代闭环；未排除完整动作历史捷径。全部模型固定 final1200，没有按 DEV 选 checkpoint。
