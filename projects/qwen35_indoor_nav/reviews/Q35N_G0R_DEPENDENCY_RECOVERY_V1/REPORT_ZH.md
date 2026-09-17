# G0R 主 agent 网络修复与运行验收

2026-09-09。结论：**RUNTIME_AND_RENDERER_ENGINEERING_PASS**。

原 G0R 下载失败不是科学失败。本轮已接管修复并实际完成隔离安装、固定 Habitat-Sim 编译、导入和一场景渲染，不再停留于安装计划。论文主线、模拟器 commit 和原 14 项依赖版本不变。

## 实测结果

| 验收项 | 实际结果 |
|---|---|
| 官方依赖获取 | 22 个 wheel，172,305,548 字节；22/22 与官方 PyPI SHA256 一致 |
| 环境 | 独立 Python 3.10.20；Habitat-Sim 0.1.7 与 Magnum 安装成功，pip check 通过 |
| 固定源码 | 856d4b08c1a2632626bf0d205bf46471a99502b7；headless，CUDA kernels/Bullet 关闭 |
| 实际场景 | 17DRP5sb8fy；GLB、semantic PLY、house 和 navmesh 使用前后哈希一致 |
| 观测 | 原始颜色缓冲为 224×224×4 uint8；保存 RGB 为前三通道，semantic 为 224×224 uint32；相机姿态一致 |
| 非退化语义 | 两帧分别有 9、17 个不同整数值；加载 187 个 object 条目、10 个 region 条目 |
| 动作 | 1 次显式 reset、1 次左转 15°；位置不变、旋转正确，collided=false |
| 渲染器 | RTX 5090，OpenGL 4.6.0，NVIDIA 580.126.09；物理 GPU 2 |
| 清理 | 无自身 GPU 残留；没有停止、恢复或新增任何占位程序 |

成功 smoke 约 7.16 秒，子进程最大 RSS 756,012 KiB；采样到的整卡显存峰值 614 MiB，其中包含原有附带 CUDA 上下文，不能称本方法显存或精确连续峰值。磁盘和各步骤资源详见 [资源记录](RESOURCE_MEASUREMENTS.json)。未穷尽计量 TLS/索引控制流量和并发编译进程 RSS 总峰值，不伪造精确资源数值。

## 修复与失败均保留

1. 直连官方 PyPI 的 DNS 超时；同 shell 显式 proxyon 后，curl 与独立 pip 均可访问。实际 wheel 下载约 37 秒。只证明本次工作连接有效，不臆断前一执行者所有网络失败的深层原因。
2. 复制 Python 带有 EXTERNALLY-MANAGED 标记，首次安装被拒绝；显式 --prefix 指向本线副本后安装通过，没有改系统或原始 Python。
3. 首次 CMake 暴露复制 Python 编译配置仍指向被禁止的旧前缀。立即停止、隔离整份失败 build；不将该尝试判为隔离合格，也不能断言其探测没有接触旧路径。仅修复本线 Python 的 sysconfig 重定位，再从干净构建目录配置并核验 include/library/compile commands，重新编译；旧隔离产物不用于验收。
4. 原生编译 718/718 成功后，旧打包流程触发第二遍编译；主动停止重复任务，显式指定已核验 build_temp 并 skip-build 打包。失败退出不改写为成功。
5. 首次 smoke 脚本误将旧绑定 MapStringString 转为 dict，在 Simulator 构造前失败。实际接口探针后改为原地赋值；保持相同场景/物理参数的 [smoke_v2](smoke_v2/smoke.result.json) 通过。此前没有运行场景、reset 或动作。

这些是版本化环境/验收脚本修复，不是算法改动或寻找有利 episode。所有细节见 [续作记录](RECOVERY_AUTHORIZATION.md)。这是迁移构建，不宣称完整原版依赖环境复现。

## 主 agent 裁决和下一步

接收本次成功重建的运行环境，解除“依赖无法获取”阻塞。原 P2、P2R1 和原 G0R 交付保持不变；复核见 [保护检查](PROTECTED_ARTIFACT_CHECK.json)。完整机器结果见 [result.json](result.json)，全依赖锁见 [requirements.lock.txt](requirements.lock.txt)。

下一科学节点直接是 **G1F 一个真实历史—续接族的数据验收**：核验物理汇合、可观察事件与可复算标签，不重开创新性泛搜或 P2 文档修订。G1F 仍按独立节点批准，不将本次 G0R 的 GPU/安装授权变成训练或造数许可。

本轮 Qwen 加载=0、训练=0、family replay=0、导航效能评估=0。scientific_pass=false、navigation_gain=null。非零 semantic 输出尚不证明实例事件标签可靠，更不证明续接记忆的独立收益或论文贡献成立。

MP3D 本地合法使用依据仍是所有者确认；未审核授权文件条款，不据此授权对外发布或商业使用。报告未对外上传数据或私有代码。
