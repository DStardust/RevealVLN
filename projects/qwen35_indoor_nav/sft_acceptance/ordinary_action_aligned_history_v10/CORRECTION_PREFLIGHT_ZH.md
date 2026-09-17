# V10 未启动前实现修正 R1
history.py 已被输入视图审计引用，保留原字节。该脚本的在线缓存追加误用了负索引而非切片；尚未加载模型或执行模拟动作，数据构建只使用正确的 indices/choose 函数。
新增 history_r1.py 只修正这一字符（补冒号），训练和在线共同导入 R1；两版 indices/choose 精确相同。原 VIEW_AUDIT 保持不变，另由 CPU_TEST_RESULT 核实新版本与已构建数据一致。本次修正不是新训练配方、不是正向导航结果。
