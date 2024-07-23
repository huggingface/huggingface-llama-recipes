# This example showcases re-using a prompt for all your generation.

# For this to work correctly, please install transformers from source with the following command:
# pip install git+https://github.com/huggingface/transformers
import os, torch, copy
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
device = "cuda"
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"

INITIAL_PROMPT = "From now on, you are going to answer all my questions with historical details. Make sure to always add a bit of french here and there, for style."

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(ckpt)

prompt_cache = DynamicCache()
inputs = tokenizer(INITIAL_PROMPT, return_tensors="pt").to("cuda")
prompt_cache = model(**inputs, past_key_values = prompt_cache).past_key_values


prompt = "Why are french people obsessed with french?"
new_inputs = tokenizer(INITIAL_PROMPT + prompt, return_tensors="pt").to("cuda")
past_key_values = copy.deepcopy(prompt_cache)
outputs = model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=20) 
response = tokenizer.batch_decode(outputs)[0]
print(response)
"""

"""

prompt = "What is the best city to swim in?"
new_inputs = tokenizer(INITIAL_PROMPT + prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**new_inputs, past_key_values=copy.deepcopy(prompt_cache),max_new_tokens=20) 
response = tokenizer.batch_decode(outputs)[0]
print(response)
"""

"""
