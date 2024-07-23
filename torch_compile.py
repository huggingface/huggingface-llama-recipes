# This example showcases how to leverage the 3.1 8B Instruct models using 
# torch.compile to accelerate inference.
#
# You need CUDA and torch >= 2.3 in order to run this example.

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false" # silence warnings when compiling

device = "cuda"
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(ckpt)

prompt = "Why dogs are so cute?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Specify the max length (including both the prompt and the response)
# When calling `generate` with `cache_implementation="static" later, this is also used to create a `StaticCache` object
# with sequence length = `max_length`. The longer the more you will re-use it
model.generation_config.max_length = 128

# without `torch.compile`: each call takes ~ 5.0 seconds (on A100 80G + torch 2.3)
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
print(response)

# `torch.compile(model, ...)` is not recommended as you compile callbacks
# and full generate. We recommend compiling only the forward for now. 
# "reduce-overhead" will use cudagraphs. 
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
model.generation_config.cache_implementation = "static"

# with `torch.compile` (on A100 80G + torch 2.3)
# 1st call: ~ 90 seconds
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
# 2nd call: ~ 60 seconds
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
# 3nd call: ~ 1.5 seconds
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
print(response)
