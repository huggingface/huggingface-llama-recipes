# This example showcases re-using a prompt for all your generation.

import torch, copy
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"

INITIAL_PROMPT = "From now on, you are going to answer all my questions with historical details. Make sure to always add a bit of french here and there, for style."

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(ckpt)

prompt_cache = DynamicCache()
inputs = tokenizer(INITIAL_PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    prompt_cache = model(**inputs, past_key_values = prompt_cache).past_key_values
    prompt_cache_fixed = copy.deepcopy(prompt_cache)
    prompt_cache_fixed.key_cache = [x[:, :, :-1] for x in prompt_cache.key_cache]
    prompt_cache_fixed.value_cache = [x[:, :, :-1] for x in prompt_cache.value_cache]


prompt = "What is the best city to swim in?"
new_inputs = tokenizer(INITIAL_PROMPT + prompt, return_tensors="pt").to(device)

outputs_baseline = model.generate(**new_inputs, max_new_tokens=20, do_sample=False)
response_baseline = tokenizer.batch_decode(outputs_baseline)[0]

outputs_fixed = model.generate(**new_inputs, past_key_values=copy.deepcopy(prompt_cache_fixed), max_new_tokens=20, do_sample=False)
response_fixed = tokenizer.batch_decode(outputs_fixed)[0]

outputs_unfixed = model.generate(**new_inputs, past_key_values=copy.deepcopy(prompt_cache), max_new_tokens=20, do_sample=False)
response_unfixed = tokenizer.batch_decode(outputs_unfixed)[0]

print()
print("Baseline:")
print(response_baseline)
print()
print("Fixed:")
print(response_fixed)
print()
print("Unfixed:")
print(response_unfixed)
print()

# The fixed version should be the same as the baseline, while the unfixed version should be different.
print("Fixed matches baseline:", response_fixed == response_baseline)
print("Unfixed matches baseline:", response_unfixed == response_baseline)
