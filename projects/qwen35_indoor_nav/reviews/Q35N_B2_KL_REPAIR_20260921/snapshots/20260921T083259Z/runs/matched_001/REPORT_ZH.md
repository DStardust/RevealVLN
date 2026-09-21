# B1 / B2 / Terminal-only 匹配对照结果

VALID_COMPLETE_DEVELOPMENT_PILOT
完整配对组 128/128；续接 1152/1152。

|终点|方法|PASS/计划|未知|预算惩罚成本|
|---|---|---:|---:|---:|
|main|B1|59/192|0|0.7406|
|main|B2|61/192|0|0.7386|
|main|Terminal|41/192|0|0.8164|
|control|B1|114/192|0|0.4881|
|control|B2|74/192|0|0.6494|
|control|Terminal|80/192|0|0.6288|

|接管条件|历史|方法|主任务PASS/N|成本|
|---|---|---|---:|---:|
|terminal_present|all|B1|34/96|0.6905|
|terminal_present|all|B2|29/96|0.7422|
|terminal_present|all|Terminal|28/96|0.7450|
|terminal_present|seen|B1|34/48|0.3810|
|terminal_present|seen|B2|27/48|0.5139|
|terminal_present|seen|Terminal|26/48|0.5200|
|terminal_present|missing|B1|0/48|1.0000|
|terminal_present|missing|B2|2/48|0.9706|
|terminal_present|missing|Terminal|2/48|0.9701|
|terminal_absent|all|B1|25/96|0.7906|
|terminal_absent|all|B2|32/96|0.7349|
|terminal_absent|all|Terminal|13/96|0.8877|
|terminal_absent|seen|B1|19/48|0.6732|
|terminal_absent|seen|B2|22/48|0.6290|
|terminal_absent|seen|Terminal|11/48|0.8051|
|terminal_absent|missing|B1|6/48|0.9081|
|terminal_absent|missing|B2|10/48|0.8408|
|terminal_absent|missing|Terminal|2/48|0.9703|

Controlled stationary SEE2 tasks in an exposed DEV house. No ordinary VLN-CE or paper generalization claim.

STATE_DIAGNOSIS 与 MEMORY_INTERVENTIONS 是受控动作诊断，不替代闭环；未排除完整动作历史捷径。全部模型固定 final1200，没有按 DEV 选 checkpoint。
