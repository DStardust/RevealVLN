# 保留基线的独立推理入口

此入口将已训练的 best4k 导航基线变为可直接读取 JSONL 的 batch1 推理程序。
它复用原模型和处理器，输出四类动作及 logits。当前定位是仿真/离线对接用基线；
开发100条 SR21%，此checkpoint的完整val_unseen和真机验收尚未完成。
小集诊断权重不用于部署，循环恢复是否采用由独立闭环报告决定。

已完成16个原有接口输入的真实GPU验证：与历史评测动作全部一致，logits最大绝对
差0.12736、相对L2差0.01425，均在事前0.15/0.03容限内；不是逐位一致。
GPU4实测首条38.51秒（包含编译），其后15条平均71.95毫秒，整次含加载72.17秒。
这一小批吞吐不代表真机时延或所有指令长度的性能。

每行只接受三个字段。图像路径相对输入 JSONL 文件；图像需为与训练相机一致的
224×224 RGB。两张图按时间从旧到新排列，executed 是实际执行过的最后至多8个
运动动作，不是计划动作；新任务清空历史。日志中的碰撞、位姿或目标位置不能输入。

```json
{"instruction":"Walk to the doorway and stop.","images":["previous.png","current.png"],"executed":["move_forward","turn_left"]}
```

从 vla 根目录运行，显式选择已分配的 GPU：

```bash
projects/qwen35_indoor_nav/.envs/q35n_qwen_g2_v1/bin/python3 -I -B \
  projects/qwen35_indoor_nav/deployment/ordinary_v1/predict.py \
  --gpu 4 --input /absolute/path/requests.jsonl --output /absolute/path/predictions.jsonl
```

输入批次只加载一次模型，输出文件必须不存在。输出耗时包含图片处理和forward，
不包含模型初始化；首条可能包含编译开销。原始训练接口保留了 target/weight 占位，
两项在送入模型前被删除，不来自目标标注。

GitHub快照不包含多GB的Qwen底模、CUDA/Python环境或场景资产；本机现有环境可用。
模型卡绑定本次保留checkpoint和源文件，下载后需按项目环境记录准备相同底模及依赖。
这不是实时ROS节点；机械臂、云台、底盘控制与时延目标应在硬件确定后单独对接。
