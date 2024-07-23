# pip install -q --upgrade transformers autoawq accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

tokenizer = AutoTokenizer.from_pretrained(model_name)

quantized_model = AutoModelForCausalLM.from_pretrained(
  model_name,
  torch_dtype=torch.float16,
  device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)

print(tokenizer.decode(output[0], skip_special_tokens=True))
