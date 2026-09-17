# 训练效率与小集验收实测

结论：`FAIL_BOUNDED_TEN_TRAIN_ROUTE_LEARNABILITY`。真实bucket归约：True；小集可学性：False。未执行导航闭环，基础导航基座尚未完成。

## 实際执行与效率

固定10训练路线565决策；每rank更新[200, 200]，共享训练更新200，训练决策6277。

缓存选择：`{"enabled": false, "rank_votes": [false, false], "selection": "engineering throughput only; no dev"}`。

效率实测：`{"median_update_seconds": 8.943092107772827, "steady_effective_decisions_per_second": 3.514821673850933, "training_wall_seconds": 1785.8076703548431, "warm_cache_speedups": [1.0128177324286998, 1.0007240350319826], "four_GPU_speedup": null, "full_epoch_wall_seconds": null}`。缓存短测为同卡同片段forward+backward，冷启动和构建单列于CACHE_PROFILE；未测四卡，不宣称线性加速。与历史全池单卡耗时并非匹配数据的严格加速比。

## 学习结果

before：`{"stage": "before", "decisions": 565, "CE": 3.071228990987339, "accuracy": 0.13628318584070798, "macro_recall": 0.22977243994943108, "recall": [0.061946902654867256, 0.07936507936507936, 0.4777777777777778, 0.3], "STOP_precision": 0.016666666666666666, "STOP_recall": 0.3, "confusion": [[21, 15, 207, 96], [8, 10, 65, 43], [8, 1, 43, 38], [0, 2, 5, 3]], "learnability_pass": false}`

after：`{"stage": "after", "decisions": 565, "CE": 0.9083674631023829, "accuracy": 0.6247787610619469, "macro_recall": 0.2853736479842674, "recall": [0.9970501474926253, 0.05555555555555555, 0.08888888888888889, 0.0], "STOP_precision": 0.0, "STOP_recall": 0.0, "confusion": [[338, 1, 0, 0], [119, 7, 0, 0], [77, 5, 8, 0], [10, 0, 0, 0]], "learnability_pass": false}`

所有阈值、最多200更新和终点评价事前冻结，无dev选优。训练集动作拟合即使通过也不是导航成功/泛化；未通过只否证当前有界配置，不宣称整个方向不可行。

## 工程归约与边界

同次反向的原始bucket与实际SUM/world_size结果逐bucket比较，记录BUCKET_CORRECTNESS。此为分布式数学实现验收，与旧D1独立单卡/双卡AdamW更新向量比较不同；旧0.15关卡FAIL不改写。缓存不缓存可训练语言隐藏状态或记忆，无未来标签进入策略。

## 资源和完整性

执行：`{"error": null, "returncodes": [0, 0], "wall_seconds": 1996.1708154678345, "GPU_count": 2, "conservative_GPU_seconds": 3992.3416318893433, "real_tasks_stopped": 0, "all_leased_holders_restored": true}`。

受保护来源595项哈希通过。GPU5/6占位恢复：True；停止真实任务0。无下载/仿真/机制训练。checkpoint只在完整200更新后生成并实测重载；未生成时不冒充存在。
