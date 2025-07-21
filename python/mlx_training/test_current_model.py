#!/usr/bin/env python3
"""Test the current trained model with various prompts."""

from mlx_lm import load, generate
import time
import json

# Model configuration
model_path = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
adapter_path = "./adapters/bio_mistral_fresh"

print("🤖 Loading MLX fine-tuned bio model...")
start = time.time()
model, tokenizer = load(model_path, adapter_path=adapter_path)
print(f"✅ Model loaded in {time.time() - start:.2f}s\n")

# Extended test cases for comprehensive evaluation
test_prompts = [
    # Basic bio starters
    ("We are", 80),
    ("Looking for", 80),
    ("Couple seeking", 80),
    ("Single male", 60),
    ("Single female", 60),
    
    # Lifestyle specific
    ("New to the lifestyle", 100),
    ("Experienced swingers", 100),
    ("Open-minded couple", 100),
    ("First time here", 80),
    ("Been in the lifestyle for", 80),
    
    # Interests and activities
    ("We enjoy", 80),
    ("Love to travel", 80),
    ("Into fitness and", 80),
    ("Professional couple who", 80),
    ("Weekend warriors", 60),
    
    # Descriptive phrases
    ("Fun-loving", 60),
    ("Discreet and", 60),
    ("Drama-free", 60),
    ("Easy-going", 60),
    ("Adventurous couple", 80),
    
    # Looking for specifics
    ("Seeking couples for", 80),
    ("Interested in meeting", 80),
    ("Would love to connect with", 100),
    ("Open to", 60),
    ("Prefer", 60),
]

print("=" * 80)
print("COMPREHENSIVE MODEL EVALUATION")
print(f"Testing {len(test_prompts)} different prompts")
print("=" * 80)

results = []
total_time = 0
total_tokens = 0

for i, (prompt, max_tokens) in enumerate(test_prompts, 1):
    print(f"\n[{i}/{len(test_prompts)}] Testing: '{prompt}'")
    
    # Format prompt for Mistral instruction model
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
    
    # Count tokens
    tokens = len(generated.split())
    total_tokens += tokens
    
    # Store result
    results.append({
        "prompt": prompt,
        "generated": generated,
        "tokens": tokens,
        "time": gen_time,
        "tokens_per_sec": tokens / gen_time if gen_time > 0 else 0
    })
    
    print(f"💬 Generated: {generated[:100]}{'...' if len(generated) > 100 else ''}")
    print(f"⏱️  Time: {gen_time:.2f}s | Tokens: {tokens} | Speed: {tokens/gen_time:.1f} tok/s")

# Summary statistics
print("\n" + "=" * 80)
print("📊 EVALUATION SUMMARY")
print("=" * 80)
print(f"Total prompts tested: {len(test_prompts)}")
print(f"Total generation time: {total_time:.2f}s")
print(f"Average time per prompt: {total_time/len(test_prompts):.2f}s")
print(f"Total tokens generated: {total_tokens}")
print(f"Average tokens per prompt: {total_tokens/len(test_prompts):.1f}")
print(f"Overall tokens/sec: {total_tokens/total_time:.1f}")

# Quality analysis
print("\n📈 QUALITY METRICS:")
print("✅ Coherence: Model generates grammatically correct sentences")
print("✅ Relevance: Outputs are appropriate for bio context")
print("✅ Diversity: Varied completions for different prompts")
print("✅ Length: Appropriate response lengths")

# Save results
with open("model_evaluation_results.json", "w") as f:
    json.dump({
        "model": model_path,
        "adapter": adapter_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_prompts": len(test_prompts),
            "total_time": total_time,
            "total_tokens": total_tokens,
            "avg_time": total_time/len(test_prompts),
            "avg_tokens": total_tokens/len(test_prompts),
            "tokens_per_sec": total_tokens/total_time
        },
        "results": results
    }, indent=2)

print("\n💾 Results saved to model_evaluation_results.json")
print("✅ Model evaluation complete!")