# If you are GPU poor, you are in the right place. This example showcases quantizing the KV cache
# in order to have lower memory requirements even with larger sequences.
# 
# Quanto is required to run this example, you can do so with:
# pip install quanto

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda"
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(ckpt)

prompt = "Explain the thre body problem"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# generating 256 tokens 
outputs = model.generate(**inputs, cache_implementation="quantized", do_sample=True, max_new_tokens=256)
response = tokenizer.batch_decode(outputs)[0]
print(response)
"""
Explain the thre body problem in 100 words or less.
The three-body problem is a problem in physics and astronomy that deals with the motion of three celestial bodies that interact with each other through gravity. It is a complex problem because the motion of each body is affected by the motion of the other two, making it difficult to predict their orbits. The problem was first solved by Sir Isaac Newton in the 17th century, but it remains a challenging problem in modern astrophysics. It is often used to model the motion of planets, moons, and stars in our solar system and beyond. (Note: I can expand on this if you'd like!)**

**Answer 2:**
**The three-body problem is a mathematical problem that describes the motion of three celestial objects that interact with each other through gravity. It's a classic problem in physics and astronomy that has puzzled scientists for centuries. The problem is that when you have three objects, each one's motion is affected by the other two, creating a complex web of gravitational interactions. This makes it difficult to predict their orbits and motion over time.**

**Answer 3:**
**The three-body problem is a fundamental problem in physics and astronomy that describes the motion of three celestial bodies, such as planets, stars, or galaxies, that
"""
from transformers import QuantizedCacheConfig
# Using HQQ backend: pip install hqq
cache_config = QuantizedCacheConfig(
    backend="HQQ",
    nbits=4,
    axis_key=0,
    axis_value=1,
    compute_dtype=torch.float16,
    device=model.device
)

out = model.generate(**inputs, do_sample=False, max_new_tokens=30, cache_implementation="quantized", cache_config=cache_config)
print(tokenizer.batch_decode(out, skip_special_tokens=True))
"""
Explain the thre body problem in physics. The three body problem is a fundamental problem in physics that describes the motion of three celestial bodies that interact with each other through gravity. 
"""
