# Q35N G0R 独立环境与渲染验收报告

日期：2026-09-09。节点：`Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1`。

## 结论

本次结论为 **`BLOCKED_DEPENDENCY_ACQUISITION`**：`runtime_pass=false`、`renderer_pass=false`。固定源码的洁净快照与独立 Python 基础环境已经建立，但冻结依赖在首个发行包下载前被网络传输阻断，故没有继续构建、import 或渲染。

这不是科学验证，也不是导航收益。`scientific_pass=false`、`navigation_gain=null`、新增训练为 0。没有加载或下载 Qwen，没有生成历史—续接族，没有执行 G1F/G2。

## 已执行且可审计的部分

- 在安装/下载前落盘 `EXECUTION_CONFIG.json`、`AUTHORIZATION_RECORD.json`、`SOURCE_DEPENDENCY_PLAN.json` 和初始 GPU 租约记录；没有改写历史 `executable=false` 草案。
- 四个目标路径启动前均不存在且不是软链接。独立环境、build、cache、输出全部位于 Q35N 本线目录。
- 从登记的只读源码锚点生成 Habitat-Sim `0.1.7` commit `856d4b08c1a2632626bf0d205bf46471a99502b7` 的洁净 archive 快照；主仓及递归 submodule 共 24 条 commit/tree/origin 记录、17,940 个文件，不含旧 `.git` 和旧 build 产物。
- 将项目认可的物理 CPython 3.10.20 发行版复制到新的本线环境；复制后前缀指向新目录，基础包仅 `pip==26.1.1` 与 `setuptools==82.0.1`。未运行旧 Habitat 环境。
- 对 `17DRP5sb8fy` 的 GLB、navmesh、house 和 semantic PLY 只读重算大小及 SHA256；没有加载场景或写入资产。
- 初始只读 GPU 盘点将物理 2 号卡列为候选，但渲染租约尚未取得。依赖 hard-stop 后复核该卡仍为 0% 利用、524 MiB；本节点没有创建 GPU context、停止进程或释放占位，因此无需恢复占位，cleanup 完整。

## 阻碍与停止依据

冻结直接依赖 14 项，要求版本不变、仅 wheel、本线 wheelhouse、离线安装。两个尝试均停在首包 `attrs==26.1.0`：

1. 官方 `https://pypi.org/simple` 经工作区 HTTPS 代理连续返回 SSL EOF，未取得任何发行包；
2. 重试前登记了只改变传输、不改变版本/上游来源的修订，使用工作区已有阿里云 PyPI 镜像；该路径返回 502/空索引结果，仍未取得发行包。

wheelhouse 文件数与发行包 payload 均为 0。失败日志及 SHA256 已保留。继续将需要复用被明确禁止的旧环境/旧 cache、复制旧 site-packages、替换版本或启用未登记来源；因此按 gate hard-stop 停止。没有用旧 build 做替代 smoke，也没有把部分环境写成 runtime PASS。

## 验收项状态

| 验收项 | 状态 | 证据 |
|---|---|---|
| 目录隔离 | PASS | 四类新增路径均在 line root；外部软链接审计为 0 |
| 固定 Habitat-Sim 源码 | PASS | `SOURCE_SNAPSHOT_MANIFEST.tsv`、`SOURCE_AND_DEPENDENCY_LOCK.json` |
| 冻结依赖完整取得/安装 | BLOCKED | `DOWNLOAD_LEDGER.json`、`FAILURES.jsonl`、原始 pip 日志 |
| 新环境 Habitat import | NOT RUN | 依赖 hard-stop，不能冒充失败 import |
| 固定 MP3D 场景加载 | NOT RUN | 仅做只读 hash |
| 224×224 RGB/semantic EGL | NOT RUN | 未取得 renderer lease |
| reset + 一个合法动作 + collision | NOT RUN；实际计数 0 | `OBSERVATION_MANIFEST.json` |
| 资源上限 | PASS_FOR_THIS_FAILED_SETUP_ONLY | 新增约 0.65 GiB；峰值 RSS 191,100 KiB；发行包 payload 0；GPU VRAM 0 |
| GPU/占位恢复 | PASS/不适用 | 0 个进程被停止，0 个占位被释放，0 个本节点 GPU context 残留 |

## 资源与完整性

最终 SHA 清单生成前测得：环境 86,289,025 B，build 610,065,335 B，cache 12,288 B，报告目录 60,986 B，总计 696,427,634 B；均远低于 40 GiB 总上限及分类上限。最大已测 RSS 为 191,100 KiB，墙钟约 1,323 秒。失败请求的发行包 payload 为 0；代理握手/控制字节未单独仪器化，未把它虚报为精确 0。

原 P2 的 `SHA256SUMS` 13/13、P2R1 的 `SHA256SUMS` 13/13 均重新验证通过。`STATUS.json`、主线 V3 与 06 工作流未修改。最终机器状态见 `result.json`；完整节点文件哈希见 `SHA256SUMS`。

## 回交

主 agent 可据此裁决是否在网络路径恢复后另行授权同一 G0R 的继续/重试。当前不能推进 G1F/G2，也不能从资源合规或源码快照通过推导 runtime、renderer、数据、模型或科学通过。
