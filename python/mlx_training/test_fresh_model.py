#!/usr/bin/env python3
"""Test the freshly trained model with better parameters."""

from mlx_lm import load, generate
import time

# Model configuration
model_path = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
adapter_path = "./adapters/bio_mistral_fresh"  # Fresh training adapters

print("Loading freshly trained model...")
start = time.time()
model, tokenizer = load(model_path, adapter_path=adapter_path)
print(f"Model loaded in {time.time() - start:.2f}s\n")

# Test prompts
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

print("Testing bio completions with fresh model (500 iterations):")
print("=" * 70)

for prompt in test_prompts:
    full_prompt = f"[INST] Complete the following bio: {prompt} [/INST]"
    
    # Generate
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
    print("-" * 70)

print("\n✅ Testing complete!")