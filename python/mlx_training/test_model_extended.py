#!/usr/bin/env python3
"""Extended test for the fine-tuned bio model with multiple prompts."""

from mlx_lm import load, generate
import time

# Model configuration
model_path = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
adapter_path = "./adapters/bio_mistral_lora"

print("Loading model...")
start = time.time()
model, tokenizer = load(model_path, adapter_path=adapter_path)
print(f"Model loaded in {time.time() - start:.2f}s\n")

# Test prompts relevant to bio autocomplete
test_prompts = [
    "We are a fun couple who",
    "Looking for other couples to",
    "We enjoy",
    "New to the lifestyle and",
    "Professional couple seeking",
    "We love to travel and",
    "Open-minded and looking for",
    "Experienced couple who",
]

print("Testing bio completions:")
print("=" * 60)

for prompt in test_prompts:
    full_prompt = f"[INST] Complete the following bio: {prompt} [/INST]"
    
    # Generate with timing
    start = time.time()
    response = generate(model, tokenizer, prompt=full_prompt, max_tokens=50)
    gen_time = time.time() - start
    
    # Extract generated part
    if "[/INST]" in response:
        generated = response.split("[/INST]")[-1].strip()
    else:
        generated = response
    
    print(f"\nPrompt: {prompt}")
    print(f"Completion: {generated}")
    print(f"Time: {gen_time:.2f}s")
    print("-" * 60)

print("\n✅ Testing complete!")