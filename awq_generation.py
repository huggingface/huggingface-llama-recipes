# INSTALLATION
# pip install -q --upgrade transformers autoawq accelerate
# REQUIREMENTS
# An instance with at least ~210 GiB of total GPU memory when using the 405B model.
# The INT4 versions of the 70B and 8B models require ~35 GiB and ~4 GiB, respectively.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

model_id = "hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4"
quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512, # Note: Update this as per your use-case
    do_fuse=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=quantization_config
)

messages = [
    {"role": "system", "content": "You are a pirate"},
    {"role": "user", "content": "What's Deep Leaning?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

outputs = model.generate(**inputs, do_sample=True, max_new_tokens=256)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
