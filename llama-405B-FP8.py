
### REQUIRES A 8xH100 + 450GB 
# pip install -q --upgrade transformers torch accelerate fbgemm-gpu

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  

model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"

quantized_model = AutoModelForCausalLM.from_pretrained(
	model_name, device_map="auto", torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)

print(tokenizer.decode(output[0], skip_special_tokens=True))