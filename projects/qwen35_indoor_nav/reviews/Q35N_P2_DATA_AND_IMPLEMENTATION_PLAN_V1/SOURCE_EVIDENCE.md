# 源码与来源证据

本文件只记录 2026-09-09 的静态核对。没有 import、模拟、推理、训练、下载或 GPU 操作。网页版本可能变化；可复现接口以列出的提交/标签为准。

## 首选数据与模拟器

- 官方 VLN-CE README：<https://github.com/jacobkrantz/VLN-CE/blob/master/README.md?plain=1>。其 R2R-CE v1-3 使用 Habitat-Sim 0.1.7 与 Matterport3D，代码 MIT，衍生任务数据受 MP3D 条款与 CC BY-NC-SA 3.0 约束。
- 官方数据说明：<https://jacobkrantz.github.io/vlnce/data>。R2R-CE 列出的 episode 数为 train 10,819/61 scenes、val_seen 778/53、val_unseen 1,839/11、test 3,408/18；本地 `phase0` manifest 的“3,603 train trajectories”是轨迹口径，不与 episode 口径混用。
- Matterport3D 学术数据条款：<https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf>。正式发布图像、轨迹或衍生数据前仍需项目方做许可复核。
- 本地 Habitat-Sim：`third_party/habitat-sim`，origin `https://github.com/facebookresearch/habitat-sim.git`，commit `856d4b08c1a2632626bf0d205bf46471a99502b7`，tag `v0.1.7`/`challenge-2021`，工作树静态检查为 clean。
- 本地 Habitat-Lab：commit `d6ed1c0a...`，tag `v0.1.7`/`challenge-2021`，工作树静态检查为 clean。G1 前必须记录完整 commit；本节点未运行它。
- `third_party/ETP-R1` commit `a94b5c8...` 且工作树有本地修改。它只作为 MP3D 资产和旧配置的只读证据，不作为可复用运行时。

## Habitat-Sim v0.1.7 API 映射

源码位置均相对于 `third_party/habitat-sim`：

- `habitat_sim/simulator.py:58`：`Simulator` 创建、场景与 semantic mesh 配置；`:157` `reset()`；`:300` `initialize_agent()`；`:328` `get_sensor_observations()`；`:393` `step()`；`:425` `make_greedy_follower()`。
- `habitat_sim/agent/agent.py:62`：`AgentState`；`:73` action space；`:158` `act()`；`:186` `get_state()`；`:203` `set_state()`。
- `src/esp/bindings/ShortestPathBindings.cpp:34`：`ShortestPath` 的 start/end、points、geodesic distance；`:83` `PathFinder.find_path()`；`:95` snap/island/navigability bindings。
- `src/esp/bindings/SceneBindings.cpp:169`：region 的 id/AABB/category/objects；`:190` object 的 id/region/AABB/OBB/category；`:210` `SemanticScene` 的 MP3D house/category/region/object bindings。
- `src/esp/bindings/SimBindings.cpp:75`：Simulator 与 `semantic_scene` binding。
- `habitat_sim/nav/greedy_geodesic_follower.py:169`：`find_path()` 返回以 `None` 结束的离散动作，并明确无噪声假设。它可产候选路径，正式标签仍须完整 replay。

关键本地源码 SHA256：

```text
ba5ddf5e9507a9eba10f9d64b08076c02942f5cb6c833ea47a6c65d4ec0c90e1  habitat_sim/simulator.py
5ed0cbc54d37b2dfade09cc6979f0120020286383b16ecc2fff487e096b1be65  habitat_sim/agent/agent.py
0d941189cc6b5887d2fdd8bc93a20147a280d39c46205962952081a4d48611ef  src/esp/bindings/ShortestPathBindings.cpp
ae161f42647c4a25d205d7cbc6c1794c708be5be39aa8cc8defd130e28ebd7b5  src/esp/bindings/SceneBindings.cpp
349937868055b08fc50f44b15a0b151c5bdee443d8fa838c5e3de17be36687d5  src/esp/bindings/SimBindings.cpp
3cc2a0614d3b9b2444286df10f43cba2b37aca0b29f513477c464e50d1b4755d  habitat_sim/nav/greedy_geodesic_follower.py
3c6b93987064d6592be2a748956ed8be613b583386179c2df4883517ec4b9119  third_party/ETP-R1/run_r2r/r2r_vlnce.yaml
```

旧 R2R 配置给出的工程尺度是 `FORWARD_STEP_SIZE=0.25 m`、`TURN_ANGLE=15°`、RGB 224×224/HFOV 90°；本规格沿用为 G1 草案参数，不宣称已经在新环境生效。

## 具体只读资产

首个接口族指定 MP3D house `17DRP5sb8fy`：

```text
ab0f715afe9f1a0d77ca6c91e6ff41f589b7d649618c5c1cd3776e13ccd075b1  17DRP5sb8fy.glb
0f36abb98ee3545d7e1f5103882269206feaaea05dcb54014333290b1d2b00c8  17DRP5sb8fy.navmesh
bdb833931a9196d2c72df8ead2842d9a161e8dc7d42da1536b10b461a9b14d8a  17DRP5sb8fy.house
b7b522bd2d0aa59ee3aec15e509f0b22ea63115b8b66301b50a3a0be0a7a26  17DRP5sb8fy_semantic.ply
```

大小依次为 21,109,340、21,900、3,763,421、74,114,687 bytes。`.house` 静态文本中候选 region `0_0/0_5/0_7/0_9` 的类别码分别为 `b/d/l/k`；只有运行 `SemanticScene.category.name()` 并核对实例归属后才能把它们确认为 bedroom/dining/living/kitchen。该 house 在至少 24 个旧项目文件中出现，故是 `OLD_EXPOSED_INTERFACE_ONLY`，不能充当独立确认集。

本地 R2R-CE：`data/phase0/raw/r2r_vlnce_v1-3/`；train gzip 1,772,932 bytes，SHA256 `f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34`，val_seen 172,881 bytes，SHA256 `7fc94841ebbd2eac0d398e020a2f638426948beaf4a561f6ee310dd67cddce55`。本节点不解包、不生成动作标签。

## Qwen3.5-2B 官方证据

- 官方模型页：<https://huggingface.co/Qwen/Qwen3.5-2B>，本轮观察 revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，Apache-2.0，约 4.57 GB，架构 `Qwen3_5ForConditionalGeneration`。
- 官方 Transformers v5.15.0 源码：<https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py>。
- config 静态事实：text hidden size 2048、24 layers、BF16、context 262,144；vision hidden 1024/depth 24、patch 16、spatial merge 2、temporal patch 2；`mrope_interleaved=true`，sections `[11,11,10]`。
- 官方实现的 conditional model 接受 `input_ids`/`inputs_embeds`、`pixel_values`、grid 与 multimodal token type；text model 接受 `inputs_embeds`、显式 position ids 和 cache 控制。Qwen3.5 cache 是线性注意力 recurrent/conv state 与全注意力 KV 的混合体，因此导航每步必须 `use_cache=false`、`past_key_values=None`，并重置模型侧 rope delta；不能把 cache 当未声明历史。
- 视觉 placeholder 的 MRoPE 位置必须先按原始 token/grid 计算，再替换连续记忆 placeholder embedding。该组合是本项目 wrapper 设计，不是官方开箱 API，状态为 `UNVERIFIED`。
- 官方模块中可见 attention `q_proj/k_proj/v_proj/o_proj`、linear-attention `in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj` 与 MLP `gate_proj/up_proj/down_proj`。LoRA/PEFT 兼容性仍须独立 smoke，不因模块名存在而判定已验证。

## 静态资源证据

只读 `du` 观察：MP3D 约 21 GB；现有 Habitat-Sim 源码/构建目录 7.5 GB；旧隔离环境 8.1 GB；旧 cache 8.4 GB；旧模型目录 19 GB；`data/phase0` 61 MB。它们只为范围估算提供锚点，不授权复用旧环境或旧模型，也不是本线实测峰值。
