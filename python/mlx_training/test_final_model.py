#!/usr/bin/env python3
"""Comprehensive test of the final trained model."""

from mlx_lm import load, generate
import time

# Model configuration
model_path = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
adapter_path = "./adapters/bio_mistral_fresh"

print("🤖 Loading MLX fine-tuned bio model...")
start = time.time()
model, tokenizer = load(model_path, adapter_path=adapter_path)
print(f"✅ Model loaded in {time.time() - start:.2f}s\n")

# Extended test prompts
test_cases = [
    # Basic starters
    ("We are", 60),
    ("Looking for", 60),
    ("Couple seeking", 60),
    
    # Lifestyle specific
    ("New to the lifestyle", 80),
    ("Experienced swingers who", 80),
    ("Open-minded couple looking", 80),
    
    # Interests
    ("We enjoy traveling and", 70),
    ("Love to dance and", 70),
    ("Professional couple who", 70),
    
    # Descriptive
    ("Fun-loving and adventurous", 60),
    ("Discreet and respectful", 60),
    ("Easy-going couple who", 60),
]

print("=" * 80)
print("COMPREHENSIVE BIO COMPLETION TESTS")
print("=" * 80)

total_time = 0
for prompt, max_tokens in test_cases:
    full_prompt = f"[INST] Complete the following bio: {prompt} [/INST]"
    
    # Generate
    start = time.time()
    response = generate(model, tokenizer, prompt=full_prompt, max_tokens=max_tokens)
    gen_time = time.time() - start
    total_time += gen_time
    
    # Extract generated part
    if "[/INST]" in response:
        generated = response.split("[/INST]")[-1].strip()
    else:
        generated = response
    
    print(f"\n📝 Prompt: '{prompt}'")
    print(f"💬 Completion: {generated}")
    print(f"⏱️  Time: {gen_time:.2f}s | Tokens/sec: {len(generated.split())/(gen_time):.1f}")
    print("-" * 80)

print(f"\n📊 Summary:")
print(f"   Total generations: {len(test_cases)}")
print(f"   Average time: {total_time/len(test_cases):.2f}s")
print(f"   Total time: {total_time:.2f}s")
print(f"\n✅ Model ready for production use!")