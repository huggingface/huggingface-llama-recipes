# INSTALLATION
# pip install -q --upgrade transformers accelerate optimum
# pip install -q --no-build-isolation auto-gptq
# REQUIREMENTS
# An instance with at least ~210 GiB of total GPU memory
# An instance with at least ~210 GiB of total GPU memory when using the 405B model.
# The INT4 versions of the 70B and 8B models require ~35 GiB and ~4 GiB, respectively.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4"
messages = [
    {"role": "system", "content": "You are a pirate"},
    {"role": "user", "content": "What's Deep Leaning?"},
]

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

outputs = model.generate(**inputs, do_sample=True, max_new_tokens=256)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
