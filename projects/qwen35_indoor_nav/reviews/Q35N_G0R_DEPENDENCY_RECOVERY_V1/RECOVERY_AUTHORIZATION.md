# G0R 网络恢复：主 agent 工程续作

2026-09-09。依据所有者 G0R 环境、下载预算和 renderer 授权继续工程修复，不扩大科学实验权限。

原 G0R 的 BLOCKED_DEPENDENCY_ACQUISITION 和全部产物保持不变；本次开始前其 SHA256SUMS 全部通过。
复用的仅是 Q35N 本节点新建的独立 Python 和精确源码归档，不使用旧线环境或缓存。

已实测：直连官方 PyPI DNS 5 秒超时；在同一执行 shell 先运行 proxyon 后，官方 attrs 索引 HTTP 200，独立 Python 的 pip index versions attrs 成功。官方 numba 0.67.0 元数据声明 Python >=3.10、numpy >=1.22,<2.6，与既定 numpy 1.23.5 不矛盾。尚不能据此断言全部依赖可解。

本次先保留原 requirements.direct.txt 的全部版本，通过官方 HTTPS PyPI、显式 proxyon、禁用用户配置和缓存完成下载。传递依赖由 resolver 解析，在安装前记录各 wheel 的哈希、版本、许可证和 METADATA；安装使用无索引本地 wheelhouse。不要关闭 TLS 验证。
允许来源为 pypi.org 和 files.pythonhosted.org。网络累计上限仍为原 12 GiB，下载阶段保守限制为 2 GiB；不因换输出目录重置原总预算。

后续构建仅固定 Habitat-Sim commit 856d4b08c1a2632626bf0d205bf46471a99502b7 的已登记归档和 submodules，headless、无 CUDA kernel、无 Bullet、4 个编译任务。构建不更新 submodules、不在线安装未登记依赖。源码版本/驱动更换仍禁止。
Qwen、数据族生成、训练和导航效能实验均未授权。GPU 仅在真正开始最小 renderer smoke 前检查和借用；优先空闲卡，恢复规则不变。

这是修复已获准安装中的传输问题，不是重开 P2 或创新性审查。

执行追记：22 个 wheel 已完整下载，原 14 项版本未变。清单生成脚本曾遇到 email Header JSON 序列化错误，改用标准 email.policy.default 后恢复；该错误未影响下载。首次 install 被物理复制 Python 保留的 uv EXTERNALLY-MANAGED 标记拒绝，未安装包；保留 install.log。第二次显式使用 --prefix 指向已核实的本线独立 ENV，不删除标记、不修改系统/原始 Python。此为安装目标明确化，不更换环境或包版本。

构建追记：首次 CMake 检查发现复制 Python 的 sysconfig 编译时路径仍含被禁止的旧 /mnt/daiyang 前缀。立即终止本节点 Ninja，保留 build.log 与完整 quarantine_build_old_prefix，不用于验收。不能声称这一失败配置满足隔离契约，也不推定其探测未接触旧路径。修复仅作用于 Q35N 独立 Python 的 sysconfig：将原编译前缀重定位到实际 sys.prefix；原文件副本与 SHA256 acb7a458a2db7094ab7c8f3eb7bb90c9bed583bb1241857907787d1978caf9ea 保留。Habitat/submodule 源码不改。重试必须先单独 CMake configure 并核验全部 Python include/library 位于本线，再从空构建目录编译；不复用被隔离对象。这是环境迁移修复，不是更换模拟器版本。

打包追记：重定位后的原生编译 718/718 已成功链接。旧 setup.py 的 build_ext + bdist_wheel 组合随后在另一个临时目录重复发起完整编译；为避免浪费，终止本节点第二次 Ninja，保留该非零退出日志。package_existing 明确指定已核验 build_temp，再用 bdist_wheel --skip-build 打包；不重用任何旧前缀隔离产物，不修改模拟器源码。原生编译有效性最终仍由实际安装、导入和 renderer smoke 判定，而非将被主动终止的整个命令改写为成功。

渲染追记：首次 smoke 在构造 Simulator 前被本节点脚本对 MapStringString 的错误 dict 转换阻止，保留顶层 smoke.log/结果；该次没有场景加载、reset 或动作。经实际 SensorSpec 接口探针确认后改为绑定映射的原地 hfov 赋值，物理参数不变；独立 smoke_v2 成功执行 1 次 Simulator 构造、1 次显式 reset、1 次 turn_left。没有寻找有利任务或筛选结果；这是同一工程验收的脚本纠错。GPU 2 只有已核实 cuda:0 任务的少量附带上下文，无计算活动，未终止/抢占进程；退出后自身 GPU 进程消失。
