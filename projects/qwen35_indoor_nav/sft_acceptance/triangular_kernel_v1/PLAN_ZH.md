# 无新增依赖的回退算子优化候选

只做CPU代数/梯度检验，未接入训练、未运行GPU基准、不改当前或已封存环境源码。

本地Transformers 5.15.0 modeling_qwen3_5.py的torch_chunk_gated_delta_rule先得到严格下三角A，然后逐行更新：L_i=A_i+A_i L_<i，最终返回I+L。因L=A+AL，有(I-A)(I+L)=I。因此可用solve_triangular(I-A,I,upper=False,unitriangular=True)表达同一操作。A严格下三角是必要前提；其对角恒零，不能把这个替代用于一般矩阵。

这是已知线性代数重写，不是论文原创算法。CPU测试分别比较FP64/FP32、长度1/8/64、批量矩阵的前向和对原始leaf梯度；测试成功不等于RTX5090上更快或全Qwen梯度已验收。

下一步若准入，用独立wrapper版本替换该局部循环，先真实模型前向/反向与同卡重复误差对照，再测完整forward+backward吞吐。必须计入cuBLAS/TRSM调用与反向代价，若更慢则弃用；不能根据减少Python循环次数宣称实测加速。FP32内部计算和Qwen外部BF16协议保持不变，不改变损失、训练数据或记忆定义。
