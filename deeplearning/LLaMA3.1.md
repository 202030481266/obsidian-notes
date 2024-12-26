# Source Links

- modelscope （不需要使用许可证的方法，推荐）
- Meta（官方的渠道，不推荐）
	- 24 hours 过期
	- 每个模型只能下载5次
	- 下载得到的是`pth`文件，需要使用`huggingface` 的转换文件进行`transformers`的适配
- Huggingface模型仓库（需要严格的审核）
# Office Repository

- [llama-models](https://github.com/meta-llama/llama-models) - Central repo for the foundation models including basic utilities, model cards, license and use policies
- [PurpleLlama](https://github.com/meta-llama/PurpleLlama) - Key component of Llama Stack focusing on safety risks and inference time mitigations
- [llama-toolchain](https://github.com/meta-llama/llama-toolchain) - Model development (inference/fine-tuning/safety shields/synthetic data generation) interfaces and canonical implementations
- [llama-agentic-system](https://github.com/meta-llama/llama-agentic-system) - E2E standalone Llama Stack system, along with opinionated underlying interface, that enables creation of agentic applications
- [llama-recipes](https://github.com/meta-llama/llama-recipes) - Community driven scripts and integrations

# Convert Weights to HF

一般来说使用HF提供的转换脚本`convert_llama_weights_to_hf.py` 来完成，具体的可以操作参考llama-recipes这个项目的一个操作，不过如果有conda环境也可以直接找到这个转换的脚本，然后运行就可以了（不需要重复下载一些必要的库）。

[convert_llama_weights_to_hf.py](convert_llama_weights_to_hf.py)

使用方法的命令一般都是：

```bash
python convert_llama_weights_to_hf.py --input_dir <original_dir_path> \
	--output_dir <target_dir_path> \
	--model_size <8B, 70B> \
	--instruct <False, True> \
	--llama_version <2,3,3.1>
```

