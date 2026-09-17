# P2R1 直接证据与访问边界

日期：2026-09-09。范围仅为 R1–R7 的静态修订；未 import/运行 Habitat 或 Qwen，未安装、下载资产/依赖、训练或操作 GPU。

## 原交付与审核

- 原 P2 的 [SHA256SUMS](../Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1/SHA256SUMS) 共 13 个条目，本轮重新执行 `sha256sum -c`，全部通过。
- 主 agent [审核报告](../Q35N_P2_MAIN_AGENT_REVIEW_V1/REPORT_ZH.md)与[机器裁决](../Q35N_P2_MAIN_AGENT_REVIEW_V1/result.json)固定七个修正项；本轮不扩大范围。

## R1：renderer 与设备

本地 Habitat-Sim 固定 commit `856d4b08c1a2632626bf0d205bf46471a99502b7`：

- `third_party/habitat-sim/habitat_sim/simulator.py:86-99`：只要 agent 配有传感器即 `create_renderer=true`，semantic sensor 还启用 semantic mesh；SHA256 `ba5ddf5e9507a9eba10f9d64b08076c02942f5cb6c833ea47a6c65d4ec0c90e1`。
- `third_party/habitat-sim/src/esp/sim/Simulator.cpp:178-190`：renderer 配置会创建 `WindowlessContext` 与 renderer；SHA256 `556a801a7d0af3b0ad88daba40d55393664c1b9cf64f8e6a32cabddd4ced4d19`。
- `third_party/habitat-sim/src/esp/gfx/WindowlessContext.cpp:43-53`：EGL build 调用 `config.setCudaDevice(device)`；无 EGL 时依赖 DISPLAY/GLX，源码没有给出本项目已核验的 CPU 软件渲染替代。SHA256 `793cd73978df0560f25890efad338d0d98047b4040d489e38c37be8e2b1324d3`。

因此 headless 不能写成 CPU；未来必须由 runtime gate 在获准的空闲 GPU/驱动上实测。当前无设备授权。

## R2/R3/R7：MP3D 语义与物理配置

- `third_party/habitat-sim/src/esp/scene/Mp3dSemanticScene.cpp:19-51` 将 MP3D region code `b/d/k/l` 映射为 bedroom/dining room/kitchen/living room；`:65-83` 暴露 object `mpcat40/raw` name 和 region category name；`:193-203` 只加载 region position 与 AABB，不能把 AABB 命中自动解释为“进入房间”。文件 SHA256 `45844140b7785accb88fea7fe121a3a2d9b0dcbbcf9402f20119800f28301345`。
- `17DRP5sb8fy.house` SHA256 `bdb833931a9196d2c72df8ead2842d9a161e8dc7d42da1536b10b461a9b14d8a`。静态关系显示：region 0=`bedroom` 含 raw=`bed`, mpcat40=`bed`；region 5=`dining room` 含多个 raw=`sofa#chair`, mpcat40=`chair`；region 9=`kitchen` 含 raw/mpcat40=`sink`；region 7=`living room` 含 raw=`tv`, mpcat40=`tv_monitor`。这些只证明元数据候选存在，不证明从任何相机位姿可见。
- 本地 R2R-CE train gzip SHA256 `f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34`；只读 JSON 统计为 10,819 episodes，其中 `17DRP5sb8fy` 有 75 个 episode。它为有限 `u` 候选池提供来源，不证明路径可回放。
- `third_party/habitat-sim/habitat_sim/agent/agent.py:73-86` 默认 cylinder agent height 1.5 m、radius 0.1 m；SHA256 `5ed0cbc54d37b2dfade09cc6979f0120020286383b16ecc2fff487e096b1be65`。
- `third_party/habitat-lab/habitat/config/default.py:212-228,250-268` 给出 0.25 m forward、sensor position `[0,1.25,0]`/orientation zero、agent 1.5 m/0.1 m 与 GPU device field；SHA256 `fcbd0d9026a87eada087e0b073784f720993f8cdc0137c0ba63d1c78cf8349be`。
- `third_party/ETP-R1/run_r2r/r2r_vlnce.yaml:8-21` 给出 0.25 m、15°、RGB 224×224/HFOV 90；其旧配置 `ALLOW_SLIDING=true`。V2 为严格可逆 replay 明确改为 `false`，这是新 gate 配置而非旧配置事实。文件 SHA256 `3c6b93987064d6592be2a748956ed8be613b583386179c2df4883517ec4b9119`。

## R6：Qwen 官方接口

- 固定模型 config：<https://huggingface.co/Qwen/Qwen3.5-2B/raw/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json>。
- 固定 Transformers 源码：<https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py>。
- 主 agent 已在审核中直接核对 `Qwen3_5ForConditionalGeneration`、`Qwen3_5Model.get_rope_index`/`compute_3d_position_ids`、`Qwen3_5TextModel` 的 `inputs_embeds`、position/cache 接口。原 P2 也已登记 hidden 2048、24 layers、BF16、vision/grid 与 hybrid cache。
- 本执行轮尝试重新读取固定官方 URL，但网页工具未返回正文且 raw GitHub 的 curl 发生 SSL 连接失败；未保存不完整响应，也不把访问失败改写成接口验证。V2 仅把静态张量契约写具体，wrapper 仍为 `UNVERIFIED`，由未来 G2 对固定源码对象做运行核验。

## 许可证据与不能作出的结论

- 官方 VLN-CE：<https://github.com/jacobkrantz/VLN-CE/blob/master/README.md?plain=1>；R2R-CE v1-3 与代码/衍生数据许可入口沿用原 P2 登记。
- MP3D 条款入口：<https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf>。主 agent 记录正文重取失败，本轮同样不签发法律解释。
- 四层权限分开：已有资产获取来源、当前受控本地研究使用、衍生数据对外发布、未来商业部署。文件存在不证明任何一层授权；后两层保持 blocked，前两层需项目所有者确认合法获取和适用范围。
- 本轮未接受网站条款、未对外上传或发布任何数据。
