#!/usr/bin/env python3
"""Test script to verify grammar fixes in MLX autocomplete server"""

import requests
import json
from typing import List, Dict

# Server configuration
MLX_SERVER_URL = "http://localhost:8003/api/autocomplete/mlx"

# Test cases for the grammar fix
test_cases = [
    {
        "prompt": "i am a young male swinger looking for",
        "expected_words": ["someone"],  # Should use "someone" for people
        "description": "Person context - should use 'someone'"
    },
    {
        "prompt": "we are a couple looking for",
        "expected_words": ["someone", "couples", "people"],
        "description": "Couple context - should use person-related words"
    },
    {
        "prompt": "single female looking for", 
        "expected_words": ["someone"],
        "description": "Female context - should use 'someone'"
    },
    {
        "prompt": "I am looking for",
        "expected_words": ["something", "someone"],  # Could be either depending on completion
        "description": "Ambiguous context - depends on what follows"
    },
    {
        "prompt": "we are looking for",
        "expected_words": ["something", "someone", "couples", "people"],
        "description": "General context - could be people or things"
    }
]

def test_autocomplete(prompt: str, max_tokens: int = 30) -> str:
    """Test the autocomplete endpoint"""
    try:
        response = requests.post(
            MLX_SERVER_URL,
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("completion", "")
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to MLX server. Is it running on port 8003?"
    except Exception as e:
        return f"Error: {str(e)}"

def check_grammar_correctness(prompt: str, completion: str) -> Dict[str, bool]:
    """Check if the grammar is correct for the given context"""
    checks = {
        "no_double_space": "  " not in completion,
        "no_something_for_people": True,  # Will check below
        "proper_word_choice": True  # Will check below
    }
    
    # Check if we incorrectly used "something" for people
    prompt_lower = prompt.lower()
    completion_lower = completion.lower()
    
    person_indicators = ['male', 'female', 'man', 'woman', 'couple', 'person', 
                        'swinger', 'partner', 'friend']
    person_verbs = ['likes', 'loves', 'enjoys', 'wants', 'shares']
    
    is_about_people = any(ind in prompt_lower for ind in person_indicators)
    if any(verb in completion_lower for verb in person_verbs):
        is_about_people = True
    
    if is_about_people and completion_lower.startswith("something that"):
        checks["no_something_for_people"] = False
        checks["proper_word_choice"] = False
    
    return checks

def main():
    """Run all test cases"""
    print("Testing MLX Grammar Fix")
    print("=" * 60)
    print()
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['description']}")
        print(f"Prompt: '{test['prompt']}'")
        
        # Get completion
        completion = test_autocomplete(test['prompt'])
        print(f"Completion: '{completion}'")
        
        # Check grammar
        if "Error:" in completion:
            print(f"❌ {completion}")
            all_passed = False
        else:
            checks = check_grammar_correctness(test['prompt'], completion)
            
            # Check for expected words
            has_expected = any(word in completion.lower() 
                             for word in test.get('expected_words', []))
            
            if all(checks.values()) and (not test.get('expected_words') or has_expected):
                print("✅ Grammar looks correct")
            else:
                print("❌ Grammar issues detected:")
                for check, passed in checks.items():
                    if not passed:
                        print(f"  - {check}")
                if test.get('expected_words') and not has_expected:
                    print(f"  - Expected one of: {test['expected_words']}")
                all_passed = False
        
        print("-" * 60)
        print()
    
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Please check the grammar rules.")
    
    # Manual test
    print("\nManual Test - Enter your own prompts (type 'quit' to exit):")
    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        
        completion = test_autocomplete(prompt)
        print(f"Completion: {completion}")
        
        checks = check_grammar_correctness(prompt, completion)
        if all(checks.values()):
            print("✅ Grammar looks correct")
        else:
            print("❌ Grammar issues:")
            for check, passed in checks.items():
                if not passed:
                    print(f"  - {check}")
        print()

if __name__ == "__main__":
    main()