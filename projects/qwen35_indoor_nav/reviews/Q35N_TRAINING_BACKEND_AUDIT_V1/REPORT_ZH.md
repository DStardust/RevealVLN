# 下一训练效率优先项：语言模型算子后端

本轮只读检查，不安装或改动正在训练的环境，不操作额外GPU。

实际工程证据：efficiency_run_v1两rank的冻结视觉/预处理缓存热态加速分别1.01282与1.00072，不足5%门槛，缓存未启用。不能由少调用视觉编码器推导训练明显提速。

项目Qwen环境的importlib.find_spec结果：fla=false、causal_conv1d=false、flash_attn=false、kernels=false。本地Transformers 5.15.0的hub_kernels.py中，use_kernel_func_from_hub_with_fallback在fla导入失败时选择torch_function；modeling_qwen3_5.py的torch_chunk_gated_delta_rule包含64块内的逐行循环。未开启Hub kernels，所以当前线性注意力走PyTorch实现。

[Transformers 5.15.0官方Qwen3.5说明](https://huggingface.co/docs/transformers/v5.15.0/en/model_doc/qwen3_5)确认缺少相应可选包时走较慢的回退路径。该文档针对别的GPU/更大模型给出的推理加速数不能套用到本项目RTX5090-2B训练。

下一步建议在独立版本环境或只读基础环境外的隔离依赖目录检验官方FLA/causal_conv1d训练内核，先固定兼容的源码/包版本、许可证、torch/CUDA/Triton依赖和预算；不能pip升级正在运行的环境。[FLA官方仓库](https://github.com/fla-org/flash-linear-attention)是候选来源，未在本节点固定版本或验收安装，不宣称当前可直接使用。

优先比较同输入前向、跨步记忆/LoRA反向、训练吞吐与内存，再决定下一训练使用哪个后端。CPU源码迹象与官方说明支持这一优化优先级，但没有算子profile和实测，不能声称已找出耗时占比或保证几倍加速。
