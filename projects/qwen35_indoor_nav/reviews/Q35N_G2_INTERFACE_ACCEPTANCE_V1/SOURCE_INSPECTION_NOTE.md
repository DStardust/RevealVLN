# 固定接口来源核验

- 官方模型固定revision的config确认Qwen3_5ForConditionalGeneration、hidden2048与视觉out2048。
- 实际preprocessor_config声明Qwen3VLProcessor/Qwen2VLImageProcessorFast。额外源码获取尝试中，modeling_qwen3_5.py成功；同目录processing_qwen3_5.py返回404。本失败保留，不虚称下载完成两个源码文件；后续以安装包和模型声明的官方processor为准。
- Transformers5.15.0的get_rope_index返回[3,B,L]；TextModel接受3平面MRoPE。compute_3d_position_ids调用get_rope_index，并不是静态文档曾描述的相反调用方向。这是调用关系表述纠正，不改变完整序列先append再计算官方MRoPE的约束。
- GatedDeltaNet仅在cache_params非None时写卷积/循环state。测试使用past_key_values=None/use_cache=False，并检查返回cache与model.rope_deltas均None。
- 本地源码只用于静态检查；实际执行安装的transformers==5.15.0。不执行下载的远程源码，不允许trust_remote_code。

来源：[固定模型config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json)、[Transformers固定源码](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
