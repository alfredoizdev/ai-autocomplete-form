#!/usr/bin/env python3
"""Test the fine-tuned bio autocomplete model."""

from mlx_lm import load, generate
import sys

def test_model(prompt="We are a couple who"):
    """Test the fine-tuned model with a prompt."""
    
    model_path = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
    adapter_path = "./adapters/bio_mistral_lora"
    
    print(f"Loading model with adapter...")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    
    print(f"\nPrompt: {prompt}")
    print("Generating completion...")
    
    response = generate(
        model, 
        tokenizer, 
        prompt=f"[INST] Complete the following bio: {prompt} [/INST]",
        max_tokens=100,
        temp=0.7
    )
    
    print(f"\nResponse: {response}")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "We are a couple who"
    test_model(prompt)
