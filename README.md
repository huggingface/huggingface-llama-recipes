# Hugging Face Llama Recipes

![thumbnail for repository](./assets/hf-llama-recepies.png)

🤗🦙Welcome! This repository contains *minimal* recipes to get started quickly
with **Llama 3.x** models, including **Llama 3.1** and **Llama 3.2**.

* To get an overview of Llama 3.1, please visit [Hugging Face announcement blog post (3.1)](https://huggingface.co/blog/llama31).
* To get an overview of Llama 3.2, please visit [Hugging Face announcement blog post (3.2)](https://huggingface.co/blog/llama32).
* For more advanced end-to-end use cases with open ML, please visit the [Open Source AI Cookbook](https://huggingface.co/learn/cookbook/index).

This repository is WIP so that you might see considerable changes in the coming days.

> [!NOTE]
> To use Llama 3.x, you need to accept the license and request permission
to access the models. Please, visit [any of the Hugging Face repos](https://huggingface.co/meta-llama)
and submit your request. You only need to do this once, you'll get access to
all the repos if your request is approved.

## Getting Started

The easiest way to quickly run a Llama 🦙 on your machine would be with the
🤗 `transformers` repository. Make sure you have the latest release installed.

```shell
$ pip install -U transformers
```

### Generate text with a base model

Let's generate some text from a given prompt.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_31 = "meta-llama/Llama-3.1-8B" # <-- llama3.1
checkpoint_32 = "meta-llama/Llama-3.2-3B" # <-- llama3.2

llama = AutoModelForCausalLM.from_pretrained(
    checkpoint_32,
    torch_dtype=torch.bfloat16
).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_32)

prompt = "Alice and Bob"
model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    generation_outputs = llama.generate(
        **model_inputs,
        do_sample=False, # Greedy Decoding
        max_new_tokens=50,
    )

generated_text = tokenizer.batch_decode(
    generation_outputs,
    skip_special_tokens=True
)
print(f"Prompt: {prompt}\nGeneration: {generated_text[0]}")
# Prompt: Alice and Bob
# Generation: Alice and Bob are playing a game. Alice has a deck of
# cards, each of which has a number written on it. Bob has a deck of
# cards, each of which has a number written on it. The numbers on the
# cards are distinct. Alice and Bob
```

### Generate text with a instruction tuned model

Generating some text given a prompt is cool, but what is even better
is conversating with an instruction tuned model. Let's see how to set
that up!

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_31 = "meta-llama/Llama-3.1-8B-Instruct" # <-- llama3.1
checkpoint_32 = "meta-llama/Llama-3.2-3B-Instruct" # <-- llama3.2

llama = AutoModelForCausalLM.from_pretrained(
    checkpoint_32,
    torch_dtype=torch.bfloat16
).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_32)

prompt = [
    {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
    {"role": "user", "content": "What's Deep Learning?"},
]
model_inputs = tokenizer.apply_chat_template(
    prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    generation_outputs = llama.generate(
        model_inputs,
        do_sample=False, # Greedy Decoding
        max_new_tokens=50,
    )

generated_text = tokenizer.batch_decode(
    generation_outputs,
    skip_special_tokens=True
)
print(f"Generation: {generated_text[0]}")

# Generation: system
#
# Cutting Knowledge Date: December 2023
# Today Date: 26 Jul 2024
#
# You are a helpful assistant, that responds as a pirate.user
#
# What's Deep Learning?assistant
#
# Yer lookin' fer a treasure trove o' knowledge on Deep Learnin', eh?
# Alright then, listen close and I'll tell ye about it.
#
#Deep Learnin' be a type o' machine learnin' that uses neural networks
```

## Local Inference

Would you like to run inference of the Llama 3.1 models locally?
So do we! The memory requirements depend on the model size and the
precision of the weights. Here's a table showing the approximate
memory needed for different configurations:

### Llama 3.1

| Model Size | FP16 | FP8 | INT4(AWQ/GPTQ/bnb) |
| :--: | :--: | :--: | :--: |
| 8B | 16 GB | 8GB | 4GB |
| 70B | 140 GB | 70GB | 35GB |
|405B | 810 GB | 405GB | 204GB |

### Llama 3.2
| Model Size | BF16/FP16 | FP8 | INT4 |
| :--: | :--: | :--: | :--: |
| 1B | 2.5 GB | 1.25GB | 0.75GB |
| 3B | 6.5 GB | 3.2GB | 1.75GB |

> [!NOTE]
> These are estimated values and may vary based on specific
implementation details and optimizations.

Working with the capable Llama 3.1 8B models:

* [Run Llama 3.1 8B in 4-bits with bitsandbytes](./4bit_bnb.ipynb)
* [Run Llama 3.1 8B in 8-bits with bitsandbytes](./8bit_bnb.ipynb)
* [Run Llama 3.1 8B with AWQ & fused ops](./awq.ipynb)

Working on the 🐘 big Llama 3.1 405B model:

* [Run Llama 3.1 405B FP8](./fp8-405B.ipynb)
* [Run Llama 3.1 405B quantized to INT4 with AWQ](./awq_generation.py)
* [Run Llama 3.1 405B quantized to INT4 with GPTQ](./gptq_generation.py)

## Model Fine Tuning:

It is often not enough to run inference on the model, you would also
need to fine tune the model on some custom dataset. We have got you
covered.

Fine tune models on your custom dataset:
* [Fine tune Llama 3.2 Vision on a custom dataset](./Llama-Vision%20FT.ipynb)
* [How to fine-tune Llama 3.1 8B on consumer GPU with PEFT and QLoRA with bitsandbytes](./peft_finetuning.py)
* [Execute a distributed fine tuning job for the Llama 3.1 405B model on a SLURM-managed computing cluster](./qlora_405B.slurm)

## Assisted Decoding Techniques

Do you want to use the smaller Llama 3.2 models to speedup text generation
of bigger models? Here we talk about assisted decoding (speculative decoding), which gives you ~2x speedups for text generation on Llama 3.1 70B (with greedy decoding).

* [Run assisted decoding with 🐘 Llama 3.1 70B and 🤏 Llama 3.2 3B](./assisted_decoding_70B_3B.ipynb)
* [Run assisted decoding with Llama 3.1 8B and Llama 3.2 1B](./assisted_decoding_8B_1B.ipynb)

## Performance Optimization

Let us optimize performace shall we?

* [Accelerate your inference using torch.compile](./torch_compile.py)
* [Accelerate your inference using torch.compile and 4-bit quantization with torchao](./torch_compile_with_torchao.ipynb)
* [Quantize KV Cache to lower memory requirements](./quantized_cache.py)
* [How to reuse prompts with dynamic caching](./prompt_reuse.py)

## API inference

Are these models too large for you to run at home? Would you like to experiment with Llama 405B? Try out the following examples!

* [Use the Inference API for PRO users](./inference-api.ipynb)

## Llama Guard and Prompt Guard

In addition to the generative models, Meta released two new models: Llama Guard 3 and Prompt Guard. Prompt Guard is a small classifier that detects jailbreaks and prompt injections. Llama Guard 3 is a safeguard model that can classify LLM inputs and generations. Learn how to use them as done in the following notebooks:

* [Detecting jailbreaks and prompt injection with Prompt Guard](./prompt_guard.ipynb)

## Synthetic Data Generation
With the ever hungry models, the need for synthetic data generation is
on the rise. Here we show you how to build your very own synthetic dataset.

* [Generate synthetic data with `distilabel`](./synthetic-data-with-llama.ipynb)
