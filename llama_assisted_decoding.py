# You need enough memory for both models.
# The result **should** match the original model's generation.
# CAVEAT 1: sampling ruins this property, even with seeding (because the assistant model consumes the rng state as well).
# CAVEAT 2: due to the nature of fp ops, there are tiny fluctuations in the logits, which may lead to different text results. There 2 properties should be true, nonetheless: a) the quality of the generated text is the same, and b) the logits on the first mismatched token are very close to each other.
# See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch

WARMUP = 2  # number of non-timed warmup runs
MAX_NEW_TOKENS = 10
DO_SAMPLE = True
ATOL = 1e-6  # ~1e-6 for fp32, up to ~1e-3 for 16 bit vars [these are NORMALIZED logits, post-softmax]; see caveats below
TORCH_DTYPE = torch.float32

PROMPT = "Alice and Bob "
CHECKPOINT = "Meta-Llama/Llama-3-405B"  # <--- big llama here
ASSISTED_CHECKPOINT = "Meta-Llama/Llama-3-7B-v1.1"  # <--- small llama here


model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, device_map="auto", torch_dtype=TORCH_DTYPE)
assistant_model = AutoModelForCausalLM.from_pretrained(ASSISTED_CHECKPOINT, device_map="auto", torch_dtype=TORCH_DTYPE)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

# Warmup + store logits for later comparison if needed
for _ in range(WARMUP):
    model.generate(**inputs, assistant_model=assistant_model)

start = time.time()
assisted_outputs = model.generate(**inputs, assistant_model=assistant_model)
end = time.time()
assisted_gen_text = tokenizer.batch_decode(assisted_outputs, skip_special_tokens=True)
print(assisted_gen_text)
print(f"\nAssisted time taken: {end - start:.2f}s")
