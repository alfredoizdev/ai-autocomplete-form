#!/usr/bin/env python3
"""
Test script for the trained Llama model.
"""

import requests
import json
import time

def test_completion(prompt, max_tokens=50):
    """Test a single completion."""
    url = "http://localhost:8003/api/autocomplete/mlx"
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data['completion'], data['elapsed_ms']
        else:
            return f"Error: {response.status_code}", 0
    except requests.exceptions.ConnectionError:
        return "Error: Server not running", 0

def main():
    """Run test completions."""
    print("=" * 60)
    print("Testing Trained Llama 3.2 Model")
    print("=" * 60)
    print()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8003/")
        server_info = response.json()
        print(f"✅ Server is running")
        print(f"   Model loaded: {server_info.get('model_loaded', False)}")
        print(f"   Adapter loaded: {server_info.get('adapter_loaded', False)}")
        print()
    except:
        print("❌ Server is not running!")
        print("Please start it with: ./start_trained_llama.sh")
        return
    
    # Test prompts
    test_prompts = [
        "i am a young male swinger looking for",
        "we are a couple interested in",
        "looking for fun people who",
        "i enjoy meeting new",
        "we want to find couples that",
        "single male seeking",
        "attractive couple looking for",
        "i am new to the lifestyle and",
        "we love to party and",
        "looking for someone who"
    ]
    
    print("Testing completions...")
    print("-" * 60)
    
    for prompt in test_prompts:
        completion, elapsed = test_completion(prompt)
        print(f"Prompt: '{prompt}'")
        print(f"Completion: '{completion}'")
        print(f"Time: {elapsed:.0f}ms")
        print("-" * 60)
        time.sleep(0.5)  # Small delay between requests
    
    print("\nTest complete!")
    print("\nTo use in the app:")
    print("1. Make sure the MLX server is running on port 8003")
    print("2. The app should automatically use it for completions")

if __name__ == "__main__":
    main()