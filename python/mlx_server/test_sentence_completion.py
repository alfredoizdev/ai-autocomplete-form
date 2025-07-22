#!/usr/bin/env python3
"""
Test MLX model for sentence completion issues.
"""

import httpx
import asyncio

async def test_sentence_completions():
    """Test various prompts that have been causing incomplete sentences."""
    
    test_cases = [
        # The specific case from the screenshot
        "Seeking a gentle well hung",
        
        # Other potentially incomplete patterns
        "Looking for a well endowed",
        "Seeking discreet",
        "We want someone who is very",
        "Couple looking for fun loving",
        "Single female seeking hung",
        "Looking for drama free",
        "Seeking clean and",
        "We are an older couple looking for"
    ]
    
    print("Testing MLX Model Sentence Completion")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for prompt in test_cases:
            try:
                response = await client.post(
                    "http://localhost:8003/api/autocomplete/mlx",
                    json={
                        "prompt": prompt,
                        "max_tokens": 50,
                        "temperature": 0.7,
                        "stop": [".", "!", "?", "\n"]
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    completion = data['completion']
                    full_text = prompt + " " + completion
                    
                    # Check if it forms a complete sentence
                    ends_with_punctuation = any(full_text.rstrip().endswith(p) for p in [".", "!", "?"])
                    
                    print(f"\nPrompt: '{prompt}'")
                    print(f"Completion: '{completion}'")
                    print(f"Full: '{full_text}'")
                    print(f"Complete sentence: {'✓' if ends_with_punctuation else '✗ INCOMPLETE'}")
                    
                    # Check for specific issues
                    if "well hung" in prompt and "man" not in completion and "person" not in completion:
                        print("⚠️  Missing noun after 'well hung'")
                    
            except Exception as e:
                print(f"Error with prompt '{prompt}': {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_sentence_completions())