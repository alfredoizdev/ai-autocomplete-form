"""
Test script for the fine-tuned DistilGPT2 model.
Evaluates model quality with various prompts and generation strategies.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json

# Model path
model_path = Path(__file__).parent / "bio_distilgpt2_finetuned"

print("Loading fine-tuned DistilGPT2 model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"Model loaded on {device}")
print("=" * 60)

# Test prompts - diverse bio starting points
test_prompts = [
    "We are a fun couple",
    "Looking for friends who enjoy",
    "I am a professional",
    "My interests include",
    "We love to travel and",
    "Seeking like-minded people for",
    "Couple in our 30s looking",
    "I enjoy meeting new",
    "We are both into",
    "Looking to expand our social"
]

def test_generation_params():
    """Test different generation parameters to find optimal settings."""
    print("\nTesting different generation parameters...")
    print("=" * 60)
    
    prompt = "We are a couple who"
    
    # Different parameter sets to try
    param_sets = [
        {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1},
        {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "repetition_penalty": 1.2},
        {"temperature": 0.6, "top_p": 0.85, "top_k": 30, "repetition_penalty": 1.0},
        {"temperature": 1.0, "top_p": 0.9, "top_k": 0, "repetition_penalty": 1.1},
    ]
    
    for i, params in enumerate(param_sets):
        print(f"\nParameter Set {i+1}: {params}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                **params
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(prompt):].strip()
        print(f"Completion: '{completion}'")

def test_with_best_params():
    """Test with optimized parameters."""
    print("\n\nTesting with best parameters...")
    print("=" * 60)
    
    # Best parameters (adjust based on results above)
    best_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_new_tokens": 25,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **best_params)
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(prompt):].strip()
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Completion: '{completion}'")

def analyze_training_data():
    """Analyze the training data to understand expected outputs."""
    print("\n\nAnalyzing training data format...")
    print("=" * 60)
    
    # Load original bio data
    bio_file = Path(__file__).parent.parent.parent / "data" / "bio.json"
    with open(bio_file, 'r') as f:
        bios = json.load(f)
    
    # Show some example bios
    print(f"Total bios in dataset: {len(bios)}")
    print("\nExample bios:")
    for i in range(min(5, len(bios))):
        print(f"\n{i+1}. {bios[i][:100]}...")

def test_base_model():
    """Compare with base DistilGPT2 to see if fine-tuning had effect."""
    print("\n\nComparing with base DistilGPT2...")
    print("=" * 60)
    
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    base_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    prompt = "We are a couple who"
    
    # Test base model
    inputs = base_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=base_tokenizer.eos_token_id
        )
    
    base_completion = base_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    # Test fine-tuned model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    finetuned_completion = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    print(f"Prompt: '{prompt}'")
    print(f"Base model: '{base_completion}'")
    print(f"Fine-tuned: '{finetuned_completion}'")

if __name__ == "__main__":
    # Run all tests
    test_generation_params()
    test_with_best_params()
    analyze_training_data()
    test_base_model()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("\nRecommendations:")
    print("1. If outputs are still poor, consider:")
    print("   - Using a larger model (GPT2-medium)")
    print("   - Adjusting data preparation format")
    print("   - Training for more epochs")
    print("   - Using different learning rate")