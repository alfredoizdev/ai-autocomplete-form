#!/usr/bin/env python3
"""
Test MLX model for grammatical issues with 'looking for' completions.
"""

import httpx
import asyncio

async def test_completions():
    """Test various 'looking for' prompts to identify grammar issues."""
    
    test_prompts = [
        "we are an older couple looking for",
        "We are looking for",
        "Looking for",
        "We're a couple looking for",
        "Couple looking for", 
        "We are seeking",
        "Looking to meet",
        "We want to find",
        "We are a couple looking for"
    ]
    
    print("Testing MLX Model Grammar Issues")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for prompt in test_prompts:
            try:
                response = await client.post(
                    "http://localhost:8003/api/autocomplete/mlx",
                    json={
                        "prompt": prompt,
                        "max_tokens": 30,
                        "temperature": 0.7
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    completion = data['completion']
                    full_text = prompt + " " + completion
                    
                    # Check for grammar issues
                    has_issue = False
                    if "looking for to" in full_text.lower():
                        has_issue = True
                    
                    print(f"\nPrompt: '{prompt}'")
                    print(f"Completion: '{completion}'")
                    print(f"Full: '{full_text}'")
                    if has_issue:
                        print("⚠️  GRAMMAR ISSUE DETECTED: 'looking for to'")
                    
            except Exception as e:
                print(f"Error with prompt '{prompt}': {e}")
    
    print("\n" + "=" * 60)
    print("Testing with more context...")
    
    # Test with more context
    context_prompts = [
        "We are an older couple looking for other",
        "We are an older couple looking for fun",
        "We are an older couple looking for someone",
        "We are an older couple looking for couples"
    ]
    
    for prompt in context_prompts:
        try:
            response = await client.post(
                "http://localhost:8003/api/autocomplete/mlx",
                json={
                    "prompt": prompt,
                    "max_tokens": 30,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                completion = data['completion']
                full_text = prompt + " " + completion
                
                print(f"\nPrompt: '{prompt}'")
                print(f"Full: '{full_text}'")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_completions())