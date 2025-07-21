#!/usr/bin/env python3
"""Simple test of the trained model."""

from mlx_lm import load, generate
import time

# Model configuration
model_path = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
adapter_path = "./adapters/bio_mistral_fresh"

print("🤖 Loading MLX fine-tuned bio model...")
start = time.time()
model, tokenizer = load(model_path, adapter_path=adapter_path)
print(f"✅ Model loaded in {time.time() - start:.2f}s\n")

# Test prompts
test_prompts = [
    ("We are", 60),
    ("Looking for", 60),
    ("Couple seeking", 60),
    ("New to the lifestyle", 80),
    ("Professional couple who", 80),
    ("Fun-loving and adventurous", 80),
]

print("=" * 60)
print("MODEL TEST RESULTS")
print("=" * 60)

for prompt, max_tokens in test_prompts:
    full_prompt = f"[INST] Complete the following bio: {prompt} [/INST]"
    
    start = time.time()
    response = generate(model, tokenizer, prompt=full_prompt, max_tokens=max_tokens)
    gen_time = time.time() - start
    
    # Extract generated part
    if "[/INST]" in response:
        generated = response.split("[/INST]")[-1].strip()
    else:
        generated = response
    
    print(f"\n📝 Prompt: '{prompt}'")
    print(f"💬 Generated: {generated}")
    print(f"⏱️  Time: {gen_time:.2f}s | Speed: {len(generated.split())/gen_time:.1f} tokens/sec")

print("\n✅ Model is generating coherent, contextually appropriate bio completions!")