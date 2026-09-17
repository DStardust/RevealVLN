# GPU7三角求解工程短测

当前用户要求效率和推进普通导航基座，允许后部占位GPU。主agent已完成该已知代数重写的6项CPU前向/梯度测试，现准入独立工程短测；不动正在运行的GPU5/6或其环境/源码/训练协议。

仅GPU7，最多300秒租借（worker270秒后强制收口，留恢复时间），28GiB显存、48GiB主机RAM、10GiB输出上限。零下载、零参数更新、零导航episode。核验占位PID/命令/pane后释放，finally恢复。初始旧checkpoint只读，固定d1小集第1路线前4决策，不使用dev或评价收益。

算法不变：只在此worker内用本地复制、注明Apache2来源的函数替换torch_chunk_gated_delta_rule里的严格下三角递推为solve_triangular，其余公式与dtype保持一致；当前训练进程不受影响。普通回退warmup1+timed3；重写warmup1+timed3，共8个4步forward/backward，不执行optimizer.step。

事前通过条件：每次全部有限；真实logit maxabs<=0.02+0.01*maxabs_reference；梯度cosine>=0.98且relative L2<=max(0.15,3倍同卡原实现重复误差)，同卡原实现重复误差须<=0.10。若此关失败保留数值结果，不转为训练后端。性能只有重写中位wall至少快10%才作为推荐；冷启动成本单列，不比较不同卡的两个模型。未比较完整训练run，不宣称线性加速或基座学习通过。

准备及运行源码/只读依赖先锁定。该操作是工程优化，不作为论文原创贡献；旧D1失败保持不变。
