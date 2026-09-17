# G1R TV绕行约束修正：两候选诊断

原短转向方案256个base无完整family，保持失败。主agent现在只修正原greedy真实往返构造中的L loop forbidden=[D,K,B]为[K,B]。D已经在H_D中发生，TV绕行重复看见D不会改变原任务结果；所有最终矩阵/事件/像素证书不变。

确定性诊断候选：coverage账本中两次已确认分别存在D/K合法loop但因L限制拒绝的同一u_index=22，yaw=0与1，依此顺序；每个base保留原8种tail顺序。最多2个base/16个完整family候选，不称独立确认或泛化。

只读复用coverage同GPU3/同资产/同实现的ROUTE_PLANS作为规划缓存；完整loop、padding、公共尾部与最终9条history×continuation仍本轮实际replay。缓存命中和来源哈希单列，不把缓存中的动作算成本轮执行。通过完整预检才冻结候选。

若失败发生在数值汇合，则保存实际pose漂移和逐RGB/semantic哈希，不放松1e-4及精确像素规则，不把“接近”标为通过。任何共同状态重建必须另列与MAINLINE_V3一致的显式版本，不能本轮偷偷round/teleport。

资源预算10分钟wall（runner的1h只是外部兜底；worker检查10分钟）、8GiB输出/16GiB RAM/8GiB GPU，GPU3借用恢复；无模型/训练/下载。第一个完整候选立即冻结，之后交27/54独立验收。
