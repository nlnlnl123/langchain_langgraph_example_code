from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import torch
import gc

# ======================
# 1. 全局初始化（只加载一次模型！）
# ======================
MODEL_PATH = "Qwen/Qwen3-8B"

# 解决显存碎片问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ["VLLM_USE_MODELSCOPE"] = "True"
# 使用 vLLM 推理引擎生成文本补全


def get_completion(
    prompts,
    model,
    tokenizer=None,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    max_tokens=2048,
    max_model_len=4096,
):
    """创建采样参数。temperature 控制生成文本随机性。=0: 确定性输出，总是选择最可能的词；=1: 标准随机性；>1: 更加随机和创造性。
    top_p : 核心采样概率 (nucleus sampling)，从累积概率达到 top_p 的词汇中选择。
    top_k：顶部k采样，只从概率最高的k个词汇中选择。
    min_p：最小概率阈值，过滤掉概率低于此值的词汇，=0 表示不进行此过滤
    max_tokens：限制模型在推理过程中生成的最大输出长度。
    max_model_len：限制模型在推理过程中可以处理的最大输入和输出长度之和。即模型最大处理长度
    """
    stop_token_ids = [151645, 151643]
    # 151645: 通常对应 <|im_end|>，表示对话结束   151643: 通常对应 <|endoftext|>，表示文本结束   当生成遇到这些token时会自动停止
    # 采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )  #  max_tokens 用于限制模型在推理过程中生成的最大输出长度
    # 初始化 vLLM 推理引擎
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )  # 降低显存利用率阈值（默认0.9），预留更多空间防OOM
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":
    # 初始化 vLLM 推理引擎
    try:
        model = MODEL_PATH  #  指定模型路径
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)  # 这步会自动下载qwen-8b模型

        prompt = "什么是大模型"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # 是否开启思考模式，默认为 True
        )

        outputs = get_completion(
            text, model, tokenizer=None, temperature=0.6, top_p = 0.8, top_k=20, min_p=0
        )
        # 对于非思考模式，官方建议使用以下参数：temperature = 0.6，TopP = 0.95，TopK = 20，MinP = 0。

        # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
        # 打印输出。
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, \nResponse: {generated_text!r}")

    except Exception as e:
        print(e)
        # 清理PyTorch和CUDA缓存
        torch.cuda.empty_cache()
        gc.collect()
