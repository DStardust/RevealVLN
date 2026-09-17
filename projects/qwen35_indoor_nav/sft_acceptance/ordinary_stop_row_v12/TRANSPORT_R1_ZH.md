# V12 特征提取运输修订 R1

原launch_extract首个前向在RoPE矩阵乘处因未传CUBLAS_WORKSPACE_CONFIG而明确异常退出，0完成前向、0优化器更新、0FEATURES文件；原LAUNCH_RESULT为FAILED，37.49秒、GPU1清理通过，旧日志/源码/回执保留。

唯一运行环境修正：传入与原已审计最佳FIT评测启动器相同的CUBLAS_WORKSPACE_CONFIG=:4096:8。原extract.py、模型、输入、目标、种子、固定算法和门槛逐字节不变。新launch_extract_r1.py仅替换该env及launcher自己的*_R1日志/回执名；原模型提取未产生的EXTRACTION/FEATURES输出继续由唯一新进程写，禁止已存在FEATURES时启动。一次1200秒运输重试，原失败成本另计，累计前向尝试最多5488，完成前向最多5487；不重复任何成功前缀。GPU1仍空闲卡准入，不操作其他进程或占位。
