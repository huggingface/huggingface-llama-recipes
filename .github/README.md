# Hugging Face Llama Recipes

![thumbnail for repository](../assets/hf-llama-recepies.png)

🤗🦙Welcome! This repository contains *minimal* recipes to get started quickly
with **Llama 3.x** models, including **Llama 3.1** and **Llama 3.2**.

* To get an overview of Llama 3.1, please visit [Hugging Face announcement blog post (3.1)](https://huggingface.co/blog/llama31).
* To get an overview of Llama 3.2, please visit [Hugging Face announcement blog post (3.2)](https://huggingface.co/blog/llama32).
* For more advanced end-to-end use cases with open ML, please visit the [Open Source AI Cookbook](https://huggingface.co/learn/cookbook/index).

This repository is WIP so that you might see considerable changes in the coming days.

> [!NOTE]
> To use Llama 3.x, you need to accept the license and request permission
to access the models. Please visit [the Hugging Face repos](https://huggingface.co/meta-llama)
and submit your request. You only need to do this once per collection; you'll get access to
all the repos in the collection if your request is approved.

## Getting Started

The easiest way to quickly run a Llama 🦙 on your machine would be with the
🤗 `transformers` repository. Make sure you have the latest release installed.

```shell
$ pip install -U transformers
```

Let us conversate with an instruction tuned model.

```python
import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_31 = "meta-llama/Llama-3.1-8B-Instruct" # <-- llama 3.1
llama_32 = "meta-llama/Llama-3.2-3B-Instruct" # <-- llama 3.2

prompt = [
    {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
    {"role": "user", "content": "What's Deep Learning?"},
]

generator = pipeline(model=llama_32, device=device, torch_dtype=torch.bfloat16)
generation = generator(
    prompt,
    do_sample=False,
    temperature=1.0,
    top_p=1,
    max_new_tokens=50
)

print(f"Generation: {generation[0]['generated_text']}")

# Generation:
# [
#   {'role': 'system', 'content': 'You are a helpful assistant, that responds as a pirate.'},
#   {'role': 'user', 'content': "What's Deep Learning?"},
#   {'role': 'assistant', 'content': "Yer lookin' fer a treasure trove o'
#             knowledge on Deep Learnin', eh? Alright then, listen close and
#             I'll tell ye about it.\n\nDeep Learnin' be a type o' machine
#             learnin' that uses neural networks"}
# ]
```

## Local Inference

Would you like to run inference of the Llama models locally?
So do we! The memory requirements depend on the model size and the
precision of the weights. Here's a table showing the approximate
memory needed for different configurations:

| Model Size | Llama Variant | BF16/FP16 | FP8 | INT4(AWQ/GPTQ/bnb) |
| :--: | :--: | :--: | :--: | :--: |
| 1B | 3.2 | 2.5 GB | 1.25GB | 0.75GB |
| 3B | 3.2 |6.5 GB | 3.2GB | 1.75GB |
| 8B | 3.1 |16 GB | 8GB | 4GB |
| 70B | 3.1 | 140 GB | 70GB | 35GB |
|405B | 3.1 |810 GB | 405GB | 204GB |


> [!NOTE]
> These are estimated values and may vary based on specific
implementation details and optimizations.

Working with the capable Llama 3.1 8B models:

* [Run Llama 3.1 8B in 4-bits with bitsandbytes](../local_inference/4bit_bnb.ipynb)
* [Run Llama 3.1 8B in 8-bits with bitsandbytes](../local_inference/8bit_bnb.ipynb)
* [Run Llama 3.1 8B with AWQ & fused ops](../local_inference/awq.ipynb)

Working on the 🐘 big Llama 3.1 405B model:

* [Run Llama 3.1 405B FP8](../local_inference/fp8-405B.ipynb)
* [Run Llama 3.1 405B quantized to INT4 with AWQ](../local_inference/awq_generation.py)
* [Run Llama 3.1 405B quantized to INT4 with GPTQ](../local_inference/gptq_generation.py)

## Model Fine Tuning:

It is often not enough to run inference on the model. 
Many times, you need to fine-tune the model on some 
custom dataset. Here are some scripts showing 
how to fine-tune the models.

Fine tune models on your custom dataset:
* [Fine tune Llama 3.2 Vision on a custom dataset](../fine_tune/Llama-Vision%20FT.ipynb)
* [Supervised Fine Tuning on Llama 3.2 Vision with TRL](../fine_tune/sft_vlm.py)
* [How to fine-tune Llama 3.1 8B on consumer GPU with PEFT and QLoRA with bitsandbytes](../fine_tune/peft_finetuning.py)
* [Execute a distributed fine tuning job for the Llama 3.1 405B model on a SLURM-managed computing cluster](../fine_tune/qlora_405B.slurm)

## Assisted Decoding Techniques

Do you want to use the smaller Llama 3.2 models to speedup text generation
of bigger models? These notebooks showcase assisted decoding (speculative decoding), which gives you upto 2x speedups for text generation on Llama 3.1 70B (with greedy decoding).

* [Run assisted decoding with 🐘 Llama 3.1 70B and 🤏 Llama 3.2 3B](../assisted_decoding/assisted_decoding_70B_3B.ipynb)
* [Run assisted decoding with Llama 3.1 8B and Llama 3.2 1B](../assisted_decoding/assisted_decoding_8B_1B.ipynb)
* [Assisted Decoding with 405B model](../assisted_decoding/assisted_decoding.py)

## Performance Optimization

Let us optimize performace shall we?

* [Accelerate your inference using torch.compile](../performance_optimization/torch_compile.py)
* [Accelerate your inference using torch.compile and 4-bit quantization with torchao](../performance_optimization/torch_compile_with_torchao.ipynb)
* [Quantize KV Cache to lower memory requirements](../performance_optimization/quantized_cache.py)
* [How to reuse prompts with dynamic caching](../performance_optimization/prompt_reuse.py)
* [How to setup distributed training utilizing DeepSpeed with mixed-precision and Zero-3 optimization](../performance_optimization/deepspeed_zero3.yaml)

## API inference

Are these models too large for you to run at home? Would you like to experiment with Llama 70B? Try out the following examples!

* [Use the Inference API for PRO users](../api_inference/inference-api.ipynb)

## Llama Guard and Prompt Guard

In addition to the generative models, Meta released two new models: Llama Guard 3 and Prompt Guard. Prompt Guard is a small classifier that detects jailbreaks and prompt injections. Llama Guard 3 is a safeguard model that can classify LLM inputs and generations. Learn how to use them as done in the following notebooks:

* [Detecting jailbreaks and prompt injection with Prompt Guard](../llama_guard/prompt_guard.ipynb)
* [Integrating Llama Guard in LLM Workflows for detecting prompt safety](../llama_guard/llama_guard_3_1B.ipynb)

## Synthetic Data Generation
With the ever hungry models, the need for synthetic data generation is
on the rise. Here we show you how to build your very own synthetic dataset.

* [Generate synthetic data with `distilabel`](../synthetic_data_gen/synthetic-data-with-llama.ipynb)


## Llama RAG 
Seeking an entry-level RAG pipeline? This notebook guides you through building a very simple streamlined RAG experiment using Llama and Hugging Face.

* [Simple RAG Pipeline](../llama_rag/llama_rag_pipeline.ipynb)


## Text Generation Inference (TGI) & API Inference with Llama Models
Text Generation Inference (TGI) framework enables efficient and  scalable deployment of Llama models. In this notebook we'll learn how to integrate TGI for fast text generation and to consume already deployed Llama models via Inference API:

* [Text Generation Inference (TGI) with Llama Models](../llama_tgi_api_inference/tgi_api_inference_recipe.ipynb) 

## Chatbot Demo with Llama Models 
Would you like to build a chatbot with Llama models? Here's a simple example to get you started.

* [Chatbot with Llama Models](../gradio_demos/chatbot_demo.ipynb)
