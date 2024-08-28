import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false" # silence warnings when compiling

device = "cuda"
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Set the quantization config
# You can choose between int4_weight_only (4-bit), int8_weight_only (8-bit) and int8_dynamic_activation_int8_weight (8-bit)
# group_size is only for int4_weight_only and needs to be one of [32,64,128,256]
quantization_config = TorchAoConfig(quant_type="int4_weight_only", group_size=128)
# Loading the quantized model takes 6218 MB
model = AutoModelForCausalLM.from_pretrained(ckpt,
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=quantization_config,
                                             device_map=device
                                             )

tokenizer = AutoTokenizer.from_pretrained(ckpt)

prompt = "Why dogs are so cute?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Specify the max length (including both the prompt and the response)
# When calling `generate` with `cache_implementation="static" later, this is also used to create a `StaticCache` object
# with sequence length = `max_length`. The longer the more you will re-use it
model.generation_config.max_length = 128

# without `torch.compile`: each call takes ~ 5.0 seconds (on A100 80G + torch 2.5)
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]


# `torch.compile(model, ...)` is not recommended as you compile callbacks
# and full generate. We recommend compiling only the forward for now. 
# "reduce-overhead" will use cudagraphs. 
# Compile with torchao requires torch 2.5 or torch nighlty if the version is still not on pipy
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
model.generation_config.cache_implementation = "static"

# with `torch.compile` (on A100 80G + torch 2.5)
# 1st call: ~ 60 seconds
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
# 2nd call: ~ 1.5 seconds
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
# 3nd call: ~ 1 seconds
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
print(response)